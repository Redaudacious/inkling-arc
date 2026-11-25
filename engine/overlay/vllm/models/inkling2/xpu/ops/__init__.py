# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Changed by inkling-arc (Nov 2026) from F3zz1k/vllm inkling2-b70 @ edd722c for the Intel Arc Pro B70; see README.
"""Inkling2 op dispatch (XPU port).

`VLLM_INKLING2_OPS=triton` (default on XPU, and on CPU under
TRITON_INTERPRET=1) runs the Triton kernels; `ref` the torch references of
`ref.py` (CPU default; not graph-capturable).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

from . import ref
from .ref import log_scaling_tau, message_blend, rms_norm, route  # noqa: F401

logger = init_logger(__name__)


def _backend() -> str:
    b = os.environ.get("VLLM_INKLING2_OPS")
    if b:
        return b
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "triton"
    return "triton" if hasattr(torch, "xpu") and torch.xpu.is_available() else "ref"


BACKEND = _backend()


def ref_interp() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1"


def attn_res(stack: torch.Tensor, n: int, last: torch.Tensor, addend: torch.Tensor | None,
             queries: torch.Tensor, norm_weight: torch.Tensor | None = None,
             norm_eps: float = 1e-6, quant: bool = False):
    """Block AttnRes over `[*stack[:n], last (+ addend)]` with normalized values,
    one mix per query row: fp32 `(Q, T, D)`, or bf16 RMS-normalized with the
    matching row of `norm_weight` `(Q, D)` (the next sublayer's norm, fused). With
    `quant`: (out, int8 rows or None), the rows for an int8 GEMM, fused at decode sizes."""
    if BACKEND == "triton" and last.is_xpu or (BACKEND == "triton" and ref_interp()):
        from .triton_attn_res import attn_res as tri

        return tri(stack[:n], last.contiguous(), None if addend is None else addend.contiguous(),
                   queries.contiguous(), None if norm_weight is None else norm_weight.contiguous(),
                   norm_eps, True, quant=quant and norm_weight is not None)
    partial = last.float() if addend is None else last.float() + addend.float()
    mixes = ref.attn_res(stack[:n], partial, queries)
    if norm_weight is None:
        return mixes
    out = torch.stack([ref.rms_norm(mixes[q], norm_weight[q], norm_eps)
                       for q in range(mixes.shape[0])])
    return (out, None) if quant else out


def qkr(attn, yq: torch.Tensor, tau: torch.Tensor | None, fused: bool):
    """q (T, H, D) and the relative logits (T, H, E) from the projection's q / r
    columns (`yq` rows: [q | r | ...]). `fused`: one Triton kernel (distance-order
    logits, for the SYCL attention); else the torch ops."""
    T = yq.shape[0]
    H, D, R = attn.num_heads, attn.head_dim, attn.d_rel
    if fused:
        from .triton_glue import qkr_prep

        return qkr_prep(yq, attn.q_scale, attn.r_scale, tau, attn.rel_proj, H, D, R, H * D,
                        attn.qk_clip, attn.rel_clip)
    q = qr_norm(yq[:, :H * D].view(T, H, D), attn.q_scale, attn.qk_clip, tau)
    r = qr_norm(yq[:, H * D:H * D + H * R].view(T, H, R), attn.r_scale, attn.rel_clip, tau)
    return q, rel_logits(r, attn.rel_proj, attn)


def blend(layer, s: slice, dests: torch.Tensor, messages: torch.Tensor, norm_w: torch.Tensor,
          eps: float) -> torch.Tensor:
    """Loop controllers of streams `s` then the next sublayer's RMSNorm: fp32 `(S, T, D)`
    dests / messages -> bf16 `(S, T, D)`. h = down(cat(rms(dest), rms(message))) (bf16
    GEMM), [gain; forget](h) (fp32 GEMM), (1 - tanh(forget)) dest + tanh(gain) message."""
    if BACKEND == "triton" and dests.is_xpu:
        # one pass, no fp32 scratch row (blend_out 1.3x at prompts; blend_prep one pass up to 64 tokens; tests/kernels/test_blend_1pass.py)
        from .triton_blend import blend_out, blend_prep

        cat = blend_prep(dests, messages)
        h = torch.bmm(cat, layer.ctrl_down[s].transpose(1, 2)).float()
        gf = torch.bmm(h, layer.ctrl_gf[s].transpose(1, 2))
        return blend_out(gf, dests, messages, norm_w, eps)
    rms = ref.unweighted_rms
    cat = torch.cat([rms(dests), rms(messages)], -1).to(layer.ctrl_down.dtype)
    h = torch.bmm(cat, layer.ctrl_down[s].transpose(1, 2)).float()
    gf = torch.bmm(h, layer.ctrl_gf[s].transpose(1, 2))
    D = dests.shape[-1]
    x = (1 - torch.tanh(gf[..., D:])) * dests + torch.tanh(gf[..., :D]) * messages
    return ref.rms_norm(x, norm_w[:, None, :], eps)


def add3(last: torch.Tensor, addend: torch.Tensor | None, m: torch.Tensor) -> torch.Tensor:
    """(last (+ addend) + m) in fp32, rounded to bf16 once (the layer output)."""
    y = last.float() if addend is None else last.float() + addend.float()
    return (y + m.float()).to(torch.bfloat16)


def qr_norm(x: torch.Tensor, scale: torch.Tensor, clip: float,
            tau: torch.Tensor | None) -> torch.Tensor:
    """bf16 `(T, H, d)`: learned scale + L2 clip (rounded), then the log
    scaling `tau` (rounded again), as the reference's two modules."""
    y = ref.l2_clip(x.float() * scale.float(), clip).to(torch.bfloat16)
    if tau is not None:
        y = (y.float() * tau[:, None, None]).to(torch.bfloat16)
    return y


def conv_kv(owner, md, xk: torch.Tensor, wk: torch.Tensor, xv: torch.Tensor, wv: torch.Tensor,
            dynk: torch.Tensor | None = None, dynv: torch.Tensor | None = None, clip: float = 0.0,
            eps: float = 1e-6, kv_owner=None, kv_md=None) -> None:
    """The visit's K short conv (L2 clip, stored as K) and V short conv (RMS norm, stored as V): one Triton
    launch (triton_sconv.short_conv_kv, byte-identical to two conv() calls); the reference path runs the two."""
    have_md = md is not None and owner.state.numel() > 0
    if BACKEND == "triton" and have_md:
        from .triton_sconv import short_conv_kv

        T = xk.shape[0]
        off_k, hph = owner.layout.stream("k")
        off_v, _ = owner.layout.stream("v")
        kv = kv_owner.kv_cache if (kv_owner is not None and kv_md is not None) else None
        short_conv_kv(xk, xv, owner.state, off_k, off_v, hph, owner.layout.width, md.seq_idx[:T],
                      md.query_start[:T], md.query_start_loc, md.block_table, md.seq_lens, wk, wv,
                      dynk=dynk, dynv=dynv, clip=clip, eps=eps, kv=kv,
                      kv_slots=None if kv is None else kv_md.slot_mapping[:T], align=md.align,
                      state_block_size=md.state_block_size,
                      num_accepted=md.num_accepted_tokens if owner.spec else None, spec=owner.spec,
                      ckpt_off=getattr(md, 'ckpt_off', None), ckpt_si=getattr(md, 'ckpt_si', None))
        return
    conv(owner, md, "k", xk, wk, dyn=dynk, norm=1, clip=clip, kv_owner=kv_owner, kv_md=kv_md)
    conv(owner, md, "v", xv, wv, dyn=dynv, norm=2, eps=eps, kv_owner=kv_owner, kv_md=kv_md)


def conv(owner, md, stream: str, x: torch.Tensor, weight: torch.Tensor,
         dyn: torch.Tensor | None = None, norm: int = 0, clip: float = 0.0, eps: float = 1e-6,
         kv_owner=None, kv_md=None, last: torch.Tensor | None = None,
         addend: torch.Tensor | None = None) -> torch.Tensor:
    """Short conv of one stream over the visit's conv state (static taps, or
    dynamic with `dyn`), then optionally a per-head norm (1 = L2 clip, 2 =
    unweighted RMS) and a store into the visit's KV cache (K for stream 'k',
    V for 'v'). Returns bf16 `(T, C)`; with `last` (and `addend`), the layer output
    last (+ addend) + conv in fp32, rounded once. Updates the conv state."""
    T = x.shape[0]
    off, hph = owner.layout.stream(stream)
    have_md = md is not None and owner.state.numel() > 0
    if BACKEND == "triton" and have_md:
        from .triton_sconv import short_conv

        kv = kv_owner.kv_cache if (kv_owner is not None and kv_md is not None) else None
        return short_conv(x, owner.state, off, hph, owner.layout.width, md.seq_idx[:T],
                          md.query_start[:T], md.query_start_loc, md.block_table, md.seq_lens,
                          weight, dyn=dyn, norm=norm, clip=clip, eps=eps, kv=kv,
                          kv_slots=None if kv is None else kv_md.slot_mapping[:T],
                          kv_off=0 if stream == "k" else hph,
                          last=None if last is None else last.contiguous(),
                          addend=None if addend is None else addend.contiguous(),
                          align=md.align, state_block_size=md.state_block_size,
                          num_accepted=md.num_accepted_tokens if owner.spec else None,
                          spec=owner.spec, ckpt_off=getattr(md, 'ckpt_off', None),
                          ckpt_si=getattr(md, 'ckpt_si', None))
    k = owner.kernel_size
    if have_md:
        assert owner.spec == 0, "reference conv: no speculative decoding"
        nreq = md.num_reqs
        qsl = md.query_start_loc[:nreq + 1].tolist()
        lens = md.seq_lens[:nreq].tolist()
        cols = [max(n - 1, 0) // md.state_block_size if md.align else 0 for n in lens]
        slots = [int(md.block_table[r, cols[r]]) for r in range(nreq)]
        init = [lens[r] > qsl[r + 1] - qsl[r] for r in range(nreq)]
        hist = ref.conv_history_state(owner.state, owner.layout, stream, x, qsl, slots, init, k)
    else:
        hist = x.new_zeros((T, k, x.shape[1]), dtype=torch.float32)
        hist[:, -1] = x.float()
    if dyn is None:
        y = ref.static_conv(hist, weight).to(torch.bfloat16)
    else:
        y = ref.dynamic_conv(hist, weight, dyn, owner.layout.num_heads).to(torch.bfloat16)
    if norm:
        yh = y.view(T, -1, hph).float()
        yh = ref.l2_clip(yh, clip) if norm == 1 else ref.unweighted_rms(yh, eps)
        y = yh.to(torch.bfloat16).view(T, -1)
    if kv_owner is not None and kv_md is not None and kv_owner.kv_cache.numel() > 0:
        slots = kv_md.slot_mapping[:T]
        valid = slots >= 0
        cache = kv_owner.kv_cache
        page = cache.shape[2]
        o = 0 if stream == "k" else hph
        cache[slots[valid] // page, :, slots[valid] % page, o:o + hph] = (
            y[valid].view(-1, owner.layout.num_heads, hph).to(cache.dtype))
    return y if last is None else add3(last, addend, y)


def _use_sycl_attention() -> bool:
    if os.environ.get("VLLM_INKLING2_ATTN", "sycl") != "sycl" or BACKEND != "triton":
        return False
    from . import sycl_attention

    return sycl_attention.available()


def rel_logits(r: torch.Tensor, proj: torch.Tensor, attn) -> torch.Tensor:
    """(T, H, E) relative logits r @ proj (bf16, as the reference). The Triton
    kernel takes them reversed along E (from a flipped copy of proj, free);
    the SYCL kernel in distance order."""
    if BACKEND == "triton" and not _use_sycl_attention():
        rev = getattr(attn, "_rel_proj_rev", None)
        if rev is None or rev.data_ptr() == 0 or rev.shape != proj.shape:
            rev = attn._rel_proj_rev = proj.flip(-1).contiguous()
        return torch.matmul(r, rev)
    return torch.matmul(r, proj)


def sycl_ready(owner) -> bool:
    """The SYCL attention (and the glue fused around it) serves this KV cache."""
    return _use_sycl_attention() and owner.kv_cache.shape[2] == 64


def attention(owner, md, q, rel_logits, sinks, scale: float, window: int | None,
              rel_extent: int, gate: torch.Tensor | None = None, quant: bool = False):
    """Attention over the visit's KV cache (K/V already stored by `conv`): (T, H, D)
    bf16, or with `gate` (T, H*D) the sigmoid-gated output (T, H*D) bf16. With `quant`
    (gated, decode sizes, SYCL path): (gated, (int8 rows, fp32 scales)) for an int8 GEMM."""
    T, H, D = q.shape
    out = torch.zeros_like(q)
    gated = None if gate is None else torch.empty((T, H * D), dtype=q.dtype, device=q.device)
    qbuf = None
    if quant and gate is not None and T <= 16 and sycl_ready(owner):
        qbuf = (torch.empty((T, H * D), dtype=torch.int8, device=q.device),
                torch.empty((T,), dtype=torch.float32, device=q.device))
    _attention_into(out, q, rel_logits, sinks, gated, gate, None if qbuf is None else qbuf[0],
                    None if qbuf is None else qbuf[1], owner=owner, scale=scale, window=window,
                    rel_extent=rel_extent)
    if gate is None:
        return out
    return (gated, qbuf) if quant else gated


_ROWWISE_MAX_Q = int(os.environ.get("VLLM_INKLING2_ROWWISE_MAX_Q", "16"))


_ROWWISE_CHECK = os.environ.get("VLLM_INKLING2_ROWWISE_CHECK") == "1"
_ROWWISE_SPLITS = os.environ.get("VLLM_INKLING2_ROWWISE_SPLITS")


def _check_rowwise(md, n, out, qlen):
    """Debug (VLLM_INKLING2_ROWWISE_CHECK=1, syncs): host-side bounds of the row-wise rows."""
    bt, lens, _ = (t.cpu() for t in out)
    qsl, sl = md.query_start_loc.cpu(), md.seq_lens.cpu()
    page = 64
    bad = []
    if int(qsl[-1]) != n:
        bad.append(f"qsl[-1]={int(qsl[-1])} n={n}")
    need = (lens + page - 1) // page
    if int(need.max()) > bt.shape[1]:
        bad.append(f"lens need {int(need.max())} blocks > table width {bt.shape[1]}")
    for i in range(n):
        ids = bt[i, :int(need[i])]
        if (ids < 0).any():
            bad.append(f"row {i}: negative block id, lens {int(lens[i])}")
            break
    if bad:
        logger.error("inkling2 rowwise check: %s; qsl %s seq_lens %s qlen %s max_q %s max_k %s",
                     bad, qsl.tolist()[:40], sl.tolist()[:40], qlen.cpu().tolist()[:40],
                     md.max_query_len, md.max_seq_len)


def _rowwise_metadata(md, n: int):
    """Per-token decode metadata for a batch of short queries: token j of request r (query
    length L_r, S_r keys) attends to its first S_r - L_r + 1 + j keys through r's block table.
    Never cached: under breakable XPU graphs the runner reuses metadata objects with their
    buffers updated in place, and a stale table of the same token count faults the device.
    """
    qsl = md.query_start_loc
    B = qsl.shape[0] - 1
    qlen = qsl[1:] - qsl[:-1]
    tok = torch.arange(n, device=qsl.device, dtype=qsl.dtype)
    # each token's request, bounded whatever the padding of query_start_loc
    req = torch.searchsorted(qsl[1:], tok, right=True).clamp_(max=B - 1)
    lens = (md.seq_lens[req] - qlen[req] + 1 + (tok - qsl[req])).clamp_(min=1)
    lens = lens.to(md.seq_lens.dtype)
    out = (md.block_table.index_select(0, req), lens,
           torch.arange(n + 1, device=qsl.device, dtype=qsl.dtype))
    if _ROWWISE_CHECK and not torch.xpu.is_current_stream_capturing():
        _check_rowwise(md, n, out, qlen)
    return out


# KV splits of the Triton decode attention on full-length (global, K8/V4) caches: 16 leaves
# 128 work-groups per visit walking up to 8K keys each at long context (power of two).
_TRITON_SPLITS_GLOBAL = int(os.environ.get("VLLM_INKLING2_TRITON_SPLITS_GLOBAL", "16"))
# esimd-global-20261123: global-layer decode / DSpark verify on the K8/V4
# cache through _kvq_decode_xe2 (ESIMD XMX, ported from the B70 community kernel attn_esimd_xe2.cpp): 11-16x faster than
# the Triton kernel at 4K-128K (bench_kvq.py, outputs within bf16 of it). Prefill and the sliding layers stay as they were.
_ESIMD_GLOBAL = os.environ.get("VLLM_INKLING2_ESIMD_GLOBAL", "0") == "1"
_KVQ = None
# far-20261123: prefill rows read keys more than rel_extent behind the whole chunk through oneDNN's fused SDPA on fp16 blocks
# (far_attention, needs the KVQ_DNNL build); keys from key_start on stay on the ESIMD kernel, which merges both.
_ESIMD_FAR = os.environ.get("VLLM_INKLING2_ESIMD_FAR", "1") == "1"
# split point rounded down to 512 keys (24 Nov 2026; was 4,096): the up to 4,095-key slack stayed on the slower ESIMD band
# kernel; 512 moved it to the fused SDPA: 32K prompt 8.08 -> 7.82 s, 96K 28.4 -> 27.7 s, needles found
_FAR_ALIGN = int(os.environ.get("VLLM_INKLING2_FAR_ALIGN", "512"))
_FAR_BLOCK = int(os.environ.get("VLLM_INKLING2_FAR_BLOCK", "32768"))


def _kvq_ext():
    global _KVQ
    if _KVQ is None:
        import sys
        sys.path.insert(0, os.environ.get("VLLM_KVQ_EXT_DIR", "/opt/nvf4"))
        import _kvq_decode_xe2 as ext
        _KVQ = ext
    return _KVQ
_VERIFY_PACK = os.environ.get("VLLM_INKLING2_VERIFY_PACK", "0") == "1"
_VERIFY_PACK_CHECK = os.environ.get("VLLM_INKLING2_VERIFY_PACK_CHECK", "0") == "1"
_pack_index: dict = {}
_pack_checks = [0]


def _packed_verify_attention(out, q, rel, owner, md, n: int, B: int, L: int, scale: float,
                             window: int) -> torch.Tensor:
    """Local-window verify attention (Claude, 23 Nov 2026), each request's L rows packed as
    pseudo-heads: pseudo-token (b, g) carries Hk*L heads, head kvh*L + j = row j of q head
    kvh*G + g, so the split-K decode kernel reads the request's window once instead of once
    per row (row-wise). Row j sits L-1-j keys before the request's last key: its relative
    bias is shifted by that much and is -inf outside its own causal window (the kernel's
    window is widened by L-1). Writes out[:n]; returns lse (Hq, n) fp32 without sinks."""
    from . import sycl_attention

    T, Hq, D = q.shape
    Hk = owner.kv_cache.shape[1]
    G = Hq // Hk
    E = rel.shape[-1]
    Ep = E + L - 1
    key = (L, E, rel.device)
    if key not in _pack_index:
        j = torch.arange(L, device=rel.device)[:, None]
        d = torch.arange(Ep, device=rel.device)[None, :]
        src = d - (L - 1 - j)
        _pack_index[key] = (src.clamp(0, E - 1), (src < 0) | (src >= E))
    src, invalid = _pack_index[key]
    qp = q[:n].reshape(B, L, Hk, G, D).permute(0, 3, 2, 1, 4).reshape(B * G, Hk * L, D)
    r = rel[:n].reshape(B, L, Hk, G, E).permute(0, 3, 2, 1, 4)  # (B, G, Hk, L, E)
    rp = torch.gather(r, -1, src.expand(B, G, Hk, L, Ep)).masked_fill_(invalid, float("-inf"))
    rp = rp.reshape(B * G, Hk * L, Ep)
    bt = md.block_table[:B].repeat_interleave(G, dim=0)
    lens = md.seq_lens[:B].repeat_interleave(G)
    qsl = torch.arange(B * G + 1, device=q.device, dtype=md.query_start_loc.dtype)
    op = torch.empty_like(qp)
    splits = int(_ROWWISE_SPLITS) if _ROWWISE_SPLITS else 4
    _, lse = sycl_attention.rel_attention(qp, owner.kv_cache, rp, None, bt, lens, qsl, scale,
                                          window + L - 1, 1, md.max_seq_len, num_splits=splits,
                                          out=op, return_lse=True)
    out[:n] = op.view(B, G, Hk, L, D).permute(0, 3, 2, 1, 4).reshape(n, Hq, D)
    return lse.view(Hk, L, B, G).permute(0, 3, 2, 1).reshape(Hq, n)


def _verify_pack_check(out, q, rel, sinks, owner, md, n, B, L, scale, window) -> None:
    """VLLM_INKLING2_VERIFY_PACK_CHECK=1 (syncs): row-wise vs packed on the same tensors."""
    if _pack_checks[0] >= 60:
        return
    real_k = int(md.seq_lens[:B].max().item())  # the host max is pinned for graph replays
    if not int(os.environ.get("VLLM_INKLING2_VERIFY_PACK_CHECK_MIN_K", "0")) <= real_k < 100000:
        return
    from . import sycl_attention

    bt, lens, qsl = _rowwise_metadata(md, n)
    ref_o = torch.empty_like(q[:n])
    splits = int(_ROWWISE_SPLITS) if _ROWWISE_SPLITS else 4
    _, ref_l = sycl_attention.rel_attention(q[:n], owner.kv_cache, rel[:n], sinks[:n], bt, lens,
                                            qsl, scale, window, 1, md.max_seq_len,
                                            num_splits=splits, out=ref_o, return_lse=True)
    po = torch.zeros_like(q)
    pl = _packed_verify_attention(po, q, rel, owner, md, n, B, L, scale, window)
    do = (po[:n].float() - ref_o.float()).abs().max().item()
    ref = ref_o.float().abs().max().item()
    if ref == 0.0:  # warm-up / dummy steps: nothing to compare
        return
    _pack_checks[0] += 1
    dl = (pl - ref_l[:, :n]).abs().max().item()
    logger.warning("verify-pack check %s B=%d L=%d keys=%d: max|dout| %.4g (max|out| %.3g) "
                   "max|dlse| %.4g", owner.prefix, B, L, real_k, do, ref, dl)


def _far_split(ext, q, kv, md, n_req, rel_extent, scale):
    """Prefill (eager, never inside a captured graph): the far part of every long prefill request -> (key_start, extra_o, extra_lse)."""
    page = kv.shape[2]
    qls = md.query_start_loc[:n_req + 1].tolist()
    sls = md.seq_lens[:n_req].tolist()
    key_start = [0] * n_req
    extra_o = extra_l = None
    for r in range(n_req):
        t0, t1 = qls[r], qls[r + 1]
        ql = t1 - t0
        fe = (sls[r] - ql + 1 - rel_extent) // _FAR_ALIGN * _FAR_ALIGN
        if ql <= 8 or fe < _FAR_ALIGN:
            continue
        if extra_o is None:
            extra_o = torch.empty(q.shape[0], q.shape[1], q.shape[2], dtype=torch.float32, device=q.device)
            extra_l = torch.full((q.shape[0], q.shape[1]), float("-inf"), dtype=torch.float32, device=q.device)
        o, l = ext.far_attention(q[t0:t1], kv, md.block_table[r, :fe // page].contiguous(), fe, scale, _FAR_BLOCK)
        extra_o[t0:t1] = o
        extra_l[t0:t1] = l
        key_start[r] = fe
    if extra_o is None:
        return None
    return torch.tensor(key_start, dtype=torch.int32).to(q.device, non_blocking=True), extra_o, extra_l


def _attention_into(out, q, rel_logits, sinks, gated, gate, xq, xs, *, owner, scale, window,
                    rel_extent):
    """Writes `out` in place. The kernels take host-side lengths (max query / key
    length, token count), so under breakable XPU graphs this is an eager break: on
    replay it runs with the current step's metadata, looked up here, not the
    capture's."""
    from .. import cache  # the harness patches cache.layer_metadata

    md = cache.layer_metadata(owner.prefix)
    n_req = md.query_start_loc.shape[0] - 1
    n = md.num_actual_tokens
    # the SYCL kernels are built for 64-token KV pages only (vLLM's XPU block size)
    if _use_sycl_attention() and owner.kv_cache.shape[2] == 64:
        from . import sycl_attention

        bt, lens, qsl, max_q = md.block_table, md.seq_lens, md.query_start_loc, md.max_query_len
        if (_VERIFY_PACK and window is not None and gated is not None and 1 < max_q <= 8
                and n_req > 0 and n == n_req * max_q):
            if _VERIFY_PACK_CHECK and not torch.xpu.is_current_stream_capturing():
                _verify_pack_check(out, q, rel_logits, sinks, owner, md, n, n_req, max_q, scale,
                                   window)
            lse = _packed_verify_attention(out, q, rel_logits, owner, md, n, n_req, max_q,
                                           scale, window)
            from .triton_glue import attn_post, attn_post_q

            if xq is not None:
                attn_post_q(out, lse, sinks, gate, gated, n, xq, xs)
            else:
                attn_post(out, lse, sinks, gate, gated, n)
            return
        if 1 < max_q <= _ROWWISE_MAX_Q:
            # short multi-token queries (speculative verify): one decode row per token, each
            # with its own causal key count, so the split-K decode kernel serves them (the
            # chunk-prefill kernel has no KV split: a few work-groups per layer at depth)
            bt, lens, qsl = _rowwise_metadata(md, n)
            max_q = 1
        decode = max_q <= 1
        # Fixed per layer type (not per context length): a graph replays the
        # split count it was captured with.
        splits = (int(os.environ.get("VLLM_INKLING2_DECODE_SPLITS_GLOBAL", "16"))
                  if window is None else 4) if decode else 1
        if decode and md.max_query_len > 1 and _ROWWISE_SPLITS:
            splits = int(_ROWWISE_SPLITS)
        if gated is not None:
            from .triton_glue import attn_post, attn_post_q

            _, lse = sycl_attention.rel_attention(
                q[:n], owner.kv_cache, rel_logits[:n], sinks[:n], bt, lens, qsl, scale, window,
                max_q, md.max_seq_len, num_splits=splits, out=out[:n], return_lse=True)
            if (_ROWWISE_CHECK and md.max_query_len > 1
                    and not torch.xpu.is_current_stream_capturing()):
                bad = ~torch.isfinite(out[:n].float()).all(-1).all(-1)
                if bool(bad.any()) or not bool(torch.isfinite(lse).all()):
                    rows = bad.nonzero().flatten()[:8].tolist()
                    logger.error("inkling2 rowwise: non-finite attention rows %s (window %s, "
                                 "rowwise %s): lens %s, lse finite %s", rows, window,
                                 max_q == 1, lens[rows].tolist() if max_q == 1 else None,
                                 bool(torch.isfinite(lse).all()))
                    logger.error("inkling2 rowwise md: prefix %s splits %s q finite %s rel finite %s "
                                 "qsl %s seq_lens %s n %s max_q %s max_k %s bt rows %s",
                                 owner.prefix, splits, bool(torch.isfinite(q[:n].float()).all()),
                                 bool(torch.isfinite(rel_logits[:n].float()).all()),
                                 md.query_start_loc.tolist()[:20], md.seq_lens.tolist()[:20], n,
                                 md.max_query_len, md.max_seq_len,
                                 bt[rows[:2]][:, :4].tolist() if rows and max_q == 1 else None)
            if xq is not None:
                attn_post_q(out, lse, sinks, gate, gated, n, xq, xs)
            else:
                attn_post(out, lse, sinks, gate, gated, n)
            return
        sycl_attention.rel_attention(q[:n], owner.kv_cache, rel_logits[:n], sinks[:n], bt, lens,
                                     qsl, scale, window, max_q, md.max_seq_len,
                                     num_splits=splits, out=out[:n])
        return
    if BACKEND == "triton":
        from .triton_attention import rel_attention

        decode = md.max_query_len <= 8
        if (_ESIMD_GLOBAL and window is None and owner.kv_cache.dtype == torch.uint8
                and owner.kv_cache.shape[3] == 256 and q.shape[1] == 3 * owner.kv_cache.shape[1]):
            # decode/verify: one request past ~96K keys fills the card better with 128 splits, else 32. Prefill chunks and mixed
            # batches (esimd-prefill-20261123): 7.3-7.7x faster than Triton splits=1 at 30K-129K past keys; 1-16 splits are within
            # 3%, 4 keeps the decode rows of a mixed batch parallel at ~100 MB of split partials (bench_kvq.py).
            esplits = (128 if (n_req == 1 and md.max_seq_len > 98304) else 32) if decode else 4
            ext = _kvq_ext()
            far = None
            if (not decode and _ESIMD_FAR and getattr(ext, "has_far", False)
                    and md.max_seq_len - md.max_query_len > rel_extent + _FAR_ALIGN):
                far = _far_split(ext, q, owner.kv_cache, md, n_req, rel_extent, scale)
            if far is None:
                ext.kvq_decode_xe2(q, owner.kv_cache, rel_logits, sinks, md.block_table, md.seq_lens,
                                   md.query_start_loc, scale, n_req, max(1, md.max_query_len), esplits, out)
            else:
                ext.kvq_decode_xe2(q, owner.kv_cache, rel_logits, sinks, md.block_table, md.seq_lens,
                                   md.query_start_loc, scale, n_req, max(1, md.max_query_len), esplits, out, *far)
        else:
            splits = (_TRITON_SPLITS_GLOBAL if window is None else 8) if decode else 1
            rel_attention(q, owner.kv_cache, rel_logits, sinks, md.block_table, md.seq_lens,
                          md.query_start_loc, scale, window, rel_extent, n_req, out=out,
                          num_splits=splits, rel_reversed=True)
    else:
        out[:n] = ref.rel_attention(q[:n], rel_logits[:n], sinks[:n], owner.kv_cache,
                                    md.block_table, md.seq_lens, md.query_start_loc, scale,
                                    window, rel_extent)
    if gated is not None:
        gated.copy_((out.float().view(gated.shape) * torch.sigmoid(gate.float())).to(gated.dtype))


# VLLM_INKLING2_ATTN_CAPTURE=1: attention stays inside the decode graph (one replay per
# step instead of 49 graph segments + 48 eager attention calls). Safe for uniform decode
# graphs: the decode kernels read each sequence's key count from seq_lens on the device
# (the host max_seqlen_k only shapes the problem), the split count is fixed per layer
# type, and the metadata buffers are updated in place (the conv kernels already rely on it).
if os.environ.get("VLLM_INKLING2_ATTN_CAPTURE", "0") != "1":
    try:
        from vllm.compilation.breakable_cudagraph import eager_break_during_capture

        _attention_into = eager_break_during_capture(_attention_into)
    except ImportError:  # pragma: no cover
        pass


# prefill-sized batches use the optional fp8 / bf16 expert copy (VLLM_INKLING2_PREFILL_EXPERTS)
PREFILL_EXPERT_MIN_TOKENS = 64


def _fused_moe(moe, prefill: bool = False):
    attr = "_prefill_kernel" if prefill else "_kernel"
    k = getattr(moe, attr, None)
    if k is None:
        from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

        t = (moe.prefill_store if prefill else moe.store).t
        k = XpuFusedMoe(w13=t["w13"], w13_scales=t.get("w13_scale"), w13_bias=None,
                        w2=t["w2"], w2_scales=t.get("w2_scale"), w2_bias=None,
                        n_experts_per_token=moe.top_k, activation="situ",
                        num_experts=moe.num_experts, ep_rank=moe.ep_rank, ep_size=moe.ep_size,
                        activation_situ_beta=moe.gate_cap,
                        activation_situ_linear_beta=moe.up_cap)
        setattr(moe, attr, k)
    return k


def dense_format() -> str:
    """VLLM_INKLING2_DENSE=bf16 (default) | int8: the attention projections."""
    fmt = os.environ.get("VLLM_INKLING2_DENSE", "bf16").lower()
    assert fmt in ("bf16", "int8"), fmt
    return fmt


def head_format() -> str:
    """VLLM_INKLING2_HEAD=bf16 (default) | int8 | fp32: the output layer (fp32 = the
    config's head dtype, the weight cast every step)."""
    fmt = os.environ.get("VLLM_INKLING2_HEAD", "bf16").lower()
    assert fmt in ("bf16", "int8", "fp32"), fmt
    return fmt


def quant_rows_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, K) float -> int8 (N, K) and fp32 (N,) scales: symmetric per row, RTN."""
    q = torch.empty(w.shape, dtype=torch.int8, device=w.device)
    s = torch.empty(w.shape[0], dtype=torch.float32, device=w.device)
    step = max(1, (64 << 20) // (w.shape[1] * 4))
    for i in range(0, w.shape[0], step):
        blk = w[i:i + step].float()
        sc = (blk.abs().amax(1) / 127.0).clamp(min=1e-8)
        q[i:i + step] = torch.round(blk / sc[:, None]).clamp(-127, 127).to(torch.int8)
        s[i:i + step] = sc
    return q, s


_WO_FORMAT = os.environ.get("VLLM_INKLING2_WO", "int8")  # dpas: W8A16 DPAS GEMM for wo
WO_PACK_MAX_T = int(os.environ.get("VLLM_INKLING2_WO_MAX_T", "64"))
_dense_lib_loaded = False


def wo_dpas() -> bool:
    """VLLM_INKLING2_WO=dpas: wo runs on the W8A16 DPAS kernel (_nvf4.dense, bf16
    activations, no int8 activation rows) at decode sizes; int8_scaled_mm above."""
    global _dense_lib_loaded
    if _WO_FORMAT != "dpas":
        return False
    if not _dense_lib_loaded:
        torch.ops.load_library(os.environ.get("VLLM_DENSE_LIB", "/opt/nvf4/lib_dense_op.so"))
        _dense_lib_loaded = True
    return True


def dense_pack(q: torch.Tensor, s: torch.Tensor, segs=()):
    return torch.ops._nvf4.dense_pack(q, s.float().contiguous(), list(segs))


def linear(x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor | None = None,
           xq: tuple[torch.Tensor, torch.Tensor] | None = None, pack=None) -> torch.Tensor:
    """x @ w.T: bf16, or W8A8 (int8 rows of `w` with fp32 `w_scale`; activations int8
    per token, from the producing kernel (`xq`: int8 rows, fp32 scales) or quantised
    here; oneDNN int8 GEMM, bf16 out). `pack` (from dense_pack): W8A16 DPAS GEMM."""
    if pack is not None:
        return torch.ops._nvf4.dense(x.contiguous(), pack[0], pack[1], w.shape[0])
    if w.dtype != torch.int8:
        return F.linear(x, w)
    if xq is None:
        from vllm import _custom_ops as vops

        q, sc, _ = vops.scaled_int8_quant(x.contiguous(), None, None, True)
        xq = (q, sc.reshape(-1).float())
    return torch.ops._xpu_C.int8_scaled_mm(xq[0], xq[1], w.t(), w_scale, x.dtype, None)


def route_fused(x, weight, bias, top_k: int, route_scale: float):
    """Router: the fused Triton kernel (fp32 logits GEMM + sigmoid/top-k kernel), else the reference. (A one-launch
    SYCL router was tried 23-24 Nov and removed: slower than this path at 8 to 4,096 tokens in graph replay.)"""
    if BACKEND == "triton" and x.is_xpu:
        from .triton_glue import route as tri

        return tri(x, weight, bias, top_k, route_scale)
    return ref.route(x, weight, bias, top_k, route_scale)


def experts(moe, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    if moe.store.fmt == "dpas":  # fp16 DPAS on the lossless packs (_nvf4, NVFP4 and MXFP8 layers alike)
        assert moe.ep_size == 1, "dpas experts: one card"
        s = moe.store
        return torch.ops._nvf4.moe(x, weights, ids, s.t["w13_pack"], s.t["w13_fac"], s.t["w2_pack"],
                                   s.t["w2_fac"], float(moe.gate_cap), float(moe.up_cap), s.kind, s.inter)
    if BACKEND == "triton" and x.is_xpu and moe.store.fmt in ("bf16", "fp8"):
        out = torch.empty_like(x)
        _fused_moe(moe).apply(out, x, weights, ids)
        return out
    if (BACKEND == "triton" and x.is_xpu and moe.prefill_store is not None
            and x.shape[0] >= PREFILL_EXPERT_MIN_TOKENS):
        out = torch.empty_like(x)
        _fused_moe(moe, prefill=True).apply(out, x, weights, ids)
        return out
    if BACKEND == "triton" and x.is_xpu and moe.store.fmt == "nvfp4":
        from . import moe_nvfp4

        t = moe.store.t
        if moe.ep_size > 1 and moe.expert_map is None:
            moe.expert_map = moe_nvfp4.expert_map(moe.num_experts, moe.ep_rank, moe.ep_size,
                                                  x.device)
        return moe_nvfp4.moe(x, weights, ids, t["w13_q"], t["w13_bscale"], t["w13_factor"],
                             t["w2_q"], t["w2_bscale"], t["w2_factor"], moe.gate_cap, moe.up_cap,
                             expert_map=moe.expert_map)
    assert moe.ep_size == 1, "reference experts: no EP"
    return ref.moe_experts(x, weights, ids, moe.expert_weights, moe.num_experts,
                           moe.gate_cap, moe.up_cap)
