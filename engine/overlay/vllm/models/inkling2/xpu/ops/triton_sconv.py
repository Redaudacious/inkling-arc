# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Changed by inkling-arc (Nov 2026) from F3zz1k/vllm inkling2-b70 @ edd722c for the Intel Arc Pro B70; see README.
"""Inkling2 short conv over the per-sequence conv state (Triton, XPU).

`out[t] = (sum_lag w[lag] * x[t - lag] + x[t]) / sqrt(k + 1)` (static taps), or
with dynamic per-token per-head taps added to the static ones and
`/ sqrt(2k + 1)` (full-attention K/V). Past inputs of the same step come from
`x`; the `k - 1` before the request's first row of this step from its state
slot (`[k - 1, H * W]`, oldest first; zero when the slot holds no history).

One program per (token, column block). The program of a request's first row
also computes the rows that read the state (the first `k - 1`) and then
writes the request's new state, so the state is read and written by one
program only and the kernel needs no second pass. Programs of the other
state-reading rows do nothing.

Optional fused epilogues: `NORM` 1 = L2 clip per head, 2 = unweighted RMS
norm per head (one head per program then), a store of the result into the
visit's attention KV cache at the token's KV slot (`KV_OFF` = 0 for K, `D` for
V in the K|V content dim), and the residual sum `last (+ addend) + y`.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _conv_row(t, r, st_base, has_init, X, Out, W, Dyn, KV, KVSlots, Last, Addend,
              c, cm, ROW, eps, clip, inv_norm, dyn_heads_w,
              s_x, s_o, s_w0, s_w1, s_dyn, s_kv_b, s_kv_h, s_kv_n, KV_OFF, hph,
              K: tl.constexpr, BLOCK_C: tl.constexpr, DYNAMIC: tl.constexpr,
              NORM: tl.constexpr, STORE_KV: tl.constexpr, KV_PAGE: tl.constexpr,
              RESID: tl.constexpr, HAS_ADD: tl.constexpr, KVQ: tl.constexpr):
    x = tl.load(X + t * s_x + c, mask=cm, other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK_C], tl.float32)
    for lag in tl.static_range(K):
        if lag == 0:
            v = x
        else:
            rr = r - lag
            if rr >= 0:
                v = tl.load(X + (t - lag) * s_x + c, mask=cm, other=0.0).to(tl.float32)
            else:
                v = tl.load(st_base + (K - 1 + rr) * ROW + c, mask=cm & has_init,
                            other=0.0).to(tl.float32)
        if DYNAMIC:
            # static_w (K, C): tap lag = backward distance
            w = tl.load(W + lag * s_w0 + c * s_w1, mask=cm, other=0.0).to(tl.float32)
            w += tl.load(Dyn + t * s_dyn + (c // dyn_heads_w) * K + lag, mask=cm, other=0.0).to(
                tl.float32)
        else:
            # conv1d weight (C, K): tap K-1 is the current token
            w = tl.load(W + c * s_w0 + (K - 1 - lag) * s_w1, mask=cm, other=0.0).to(tl.float32)
        acc += w * v
    y = ((acc + x) * inv_norm).to(tl.bfloat16).to(tl.float32)  # the conv output is bf16
    if NORM == 1:
        n = tl.sqrt(tl.sum(y * y, 0))
        y = y * tl.minimum(clip / (n + 1e-6), 1.0)
    elif NORM == 2:
        y = y * tl.rsqrt(tl.sum(y * y, 0) / BLOCK_C + eps)
    if RESID:  # the layer output: (last (+ addend) + conv out) in fp32, rounded once
        res = tl.load(Last + t * s_o + c, mask=cm, other=0.0).to(tl.float32)
        if HAS_ADD:
            res += tl.load(Addend + t * s_o + c, mask=cm, other=0.0).to(tl.float32)
        tl.store(Out + t * s_o + c, (res + y).to(Out.dtype.element_ty), mask=cm)
    else:
        tl.store(Out + t * s_o + c, y.to(Out.dtype.element_ty), mask=cm)
    if STORE_KV:
        ks = tl.load(KVSlots + t)
        if ks >= 0:
            if KVQ:
                # K8/V4 slot (uint8, 2*hph bytes): [K int8 x hph | V int4 x hph/2 | K scale f32 at
                # 3*hph/2 | V scale f32 at 3*hph/2+4]. One head per program here (the K and V
                # streams always run with a per-head norm, so BLOCK_C == hph).
                head0 = tl.min(c, 0) // hph
                sb = KV + (ks // KV_PAGE).to(tl.int64) * s_kv_b + head0 * s_kv_h + (ks % KV_PAGE) * s_kv_n
                amax = tl.max(tl.where(cm, tl.abs(y), 0.0), 0)
                if KV_OFF == 0:
                    sc = tl.maximum(amax, 1e-12) / 127.0
                    qk = tl.minimum(tl.maximum(tl.floor(y / sc + 0.5), -127.0), 127.0).to(tl.int32)
                    tl.store(sb + c % hph, qk.to(tl.int8).to(tl.uint8, bitcast=True), mask=cm)
                    tl.store((sb + (3 * hph) // 2).to(tl.pointer_type(tl.float32)), sc)
                else:
                    # V int4, symmetric, one fp16 scale per 16 values (8 per head) at 3*hph/2+4
                    y2 = tl.reshape(y, (BLOCK_C // 16, 16))
                    gs = tl.maximum(tl.max(tl.abs(y2), 1), 1e-12) / 7.0
                    gs = gs.to(tl.float16).to(tl.float32)  # quantize with the stored scale
                    q2 = tl.minimum(tl.maximum(tl.floor(y2 / gs[:, None] + 0.5), -7.0), 7.0) + 8.0
                    qv = tl.reshape(q2.to(tl.int32), (BLOCK_C,))
                    lo, hi = tl.split(tl.reshape(qv, (BLOCK_C // 2, 2)))
                    tl.store(sb + hph + tl.arange(0, BLOCK_C // 2), (lo | (hi << 4)).to(tl.uint8))
                    tl.store((sb + (3 * hph) // 2 + 4).to(tl.pointer_type(tl.float16))
                             + tl.arange(0, BLOCK_C // 16), gs.to(tl.float16))
            else:
                head = c // hph
                tl.store(KV + (ks // KV_PAGE).to(tl.int64) * s_kv_b + head * s_kv_h
                         + (ks % KV_PAGE) * s_kv_n + KV_OFF + c % hph, y.to(KV.dtype.element_ty),
                         mask=cm)


@triton.jit
def _sconv_body(
    t, cb,
    X, Out, State, SeqIdx, QStart, QSL, BlockTable, SeqLens, NumAcc, W, Dyn, KV, KVSlots, Last, Addend,
    CkOff, CkSi,
    C, hph, W_row, off, eps, clip, inv_norm, dyn_heads_w,
    s_x, s_o, s_st, ROW, s_bt, state_bs, s_w0, s_w1, s_dyn, s_kv_b, s_kv_h, s_kv_n, KV_OFF,
    K: tl.constexpr, KP: tl.constexpr, BLOCK_C: tl.constexpr, DYNAMIC: tl.constexpr,
    NORM: tl.constexpr, STORE_KV: tl.constexpr, KV_PAGE: tl.constexpr,
    RESID: tl.constexpr, HAS_ADD: tl.constexpr, ALIGN: tl.constexpr, S: tl.constexpr,
    KVQ: tl.constexpr, CKPT: tl.constexpr,
):
    c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = c < C
    # state column of channel c: head-major rows of width W_row, stream at `off`
    sc = (c // hph) * W_row + off + c % hph
    req = tl.load(SeqIdx + t)
    q0 = tl.load(QStart + t)
    r = t - q0
    if r == 0:
        L = tl.load(QSL + req + 1) - q0
        seq_len = tl.load(SeqLens + req)
        has_init = seq_len > L  # tokens computed before this step
        if ALIGN:
            col = tl.maximum(seq_len - 1, 0) // state_bs
        else:
            col = 0
        si = tl.load(BlockTable + req * s_bt + col).to(tl.int64)
        st = State + si * s_st
        # speculative decode (S > 0): the state holds K - 1 + S rows; a decode row reads
        # its K - 1 prior inputs from row num_accepted - 1 on (vLLM's rolling conv state)
        if S > 0:
            is_dec = has_init & (L <= 1 + S)
            w = tl.load(NumAcc + req, mask=is_dec, other=1) - 1
        else:
            is_dec = False
            w = 0
        st_w = st + w * ROW
        st_c = st_w + sc - c  # so that st_c + row * ROW + c addresses channel c
        for u in tl.static_range(K - 1):
            if u < L:
                _conv_row(t + u, u, st_c, has_init, X, Out, W, Dyn, KV, KVSlots, Last, Addend,
                          c, cm, ROW, eps, clip, inv_norm, dyn_heads_w, s_x, s_o, s_w0, s_w1,
                          s_dyn, s_kv_b, s_kv_h, s_kv_n, KV_OFF, hph, K=K, BLOCK_C=BLOCK_C,
                          DYNAMIC=DYNAMIC, NORM=NORM, STORE_KV=STORE_KV, KV_PAGE=KV_PAGE,
                          RESID=RESID, HAS_ADD=HAS_ADD, KVQ=KVQ)
        # new state, all loads before the stores. Prefill rows: rows 0..K-2 = the last K - 1
        # inputs (older ones from the old state). Decode rows (S > 0): rows j < K - 2 from old
        # row w + 1 + j (the K - 2 inputs before this step), then the step's L inputs from
        # row K - 2 on, whatever L is, so the next step's read from row num_accepted - 1 lands
        # on its history after a verify of any depth (adaptive depth). With L = S + 1 this is
        # the old layout (rows j < R - L from the old state), token-identical at depth S.
        R = K - 1 + S
        rows = tl.arange(0, KP)
        if is_dec:
            from_old = rows < K - 2
            old_row = tl.minimum(w + 1 + rows, R - 1)
            src = rows - (K - 2)
            rm = rows < K - 2 + L
        else:
            src = L - (K - 1) + rows
            from_old = src < 0
            old_row = tl.maximum(K - 1 + src, 0)
            rm = rows < K - 1
        from_x = tl.load(X + (q0 + tl.maximum(src, 0))[:, None] * s_x + c[None, :],
                         mask=(rm & ~from_old)[:, None] & cm[None, :], other=0.0)
        from_st = tl.load(st + old_row[:, None] * ROW + sc[None, :],
                          mask=(rm & from_old)[:, None] & cm[None, :] & has_init, other=0.0)
        new = tl.where(from_old[:, None], from_st.to(tl.float32), from_x.to(tl.float32))
        # 23 Nov 2026: the program's loads of old state rows (above, and in the first rows' convs) and its stores
        # below touch the same rows from different threads of the program; without this barrier a store could land
        # before another thread's load (decode states differed between identical runs in rows 0..K-3).
        tl.debug_barrier()
        if CKPT:
            # Prefix caching (align mode, Claude 23 Nov 2026): the scheduler reserves one block for the
            # state at the prompt's checkpoint position (ck tokens into this step); the state there is
            # the K - 1 inputs before it (older ones from the old state). Loads before any store.
            ck = tl.load(CkOff + req)
            if ck > 0:
                ck_si = tl.load(CkSi + req).to(tl.int64)
                src2 = ck - (K - 1) + rows
                from_old2 = src2 < 0
                rm2 = (rows < K - 1) & (ck_si >= 0)
                x2 = tl.load(X + (q0 + tl.maximum(src2, 0))[:, None] * s_x + c[None, :],
                             mask=(rm2 & ~from_old2)[:, None] & cm[None, :], other=0.0)
                s2 = tl.load(st + tl.maximum(K - 1 + src2, 0)[:, None] * ROW + sc[None, :],
                             mask=(rm2 & from_old2)[:, None] & cm[None, :] & has_init, other=0.0)
                ck2 = tl.where(from_old2[:, None], s2.to(tl.float32), x2.to(tl.float32))
                tl.store(State + tl.maximum(ck_si, 0) * s_st + rows[:, None] * ROW + sc[None, :],
                         ck2.to(State.dtype.element_ty), mask=rm2[:, None] & cm[None, :])
        # the checkpoint block also loads old rows of `st` (s2): finish those loads before overwriting the slot
        tl.debug_barrier()
        tl.store(st + rows[:, None] * ROW + sc[None, :], new.to(State.dtype.element_ty),
                 mask=rm[:, None] & cm[None, :])
    elif r >= K - 1:
        _conv_row(t, r, State, False, X, Out, W, Dyn, KV, KVSlots, Last, Addend,
                  c, cm, ROW, eps, clip, inv_norm, dyn_heads_w, s_x, s_o, s_w0, s_w1,
                  s_dyn, s_kv_b, s_kv_h, s_kv_n, KV_OFF, hph, K=K, BLOCK_C=BLOCK_C,
                  DYNAMIC=DYNAMIC, NORM=NORM, STORE_KV=STORE_KV, KV_PAGE=KV_PAGE,
                  RESID=RESID, HAS_ADD=HAS_ADD, KVQ=KVQ)


@triton.jit
def _sconv_kernel(
    X, Out, State, SeqIdx, QStart, QSL, BlockTable, SeqLens, NumAcc, W, Dyn, KV, KVSlots, Last, Addend,
    CkOff, CkSi,
    C, hph, W_row, off, eps, clip, inv_norm, dyn_heads_w,
    s_x, s_o, s_st, ROW, s_bt, state_bs, s_w0, s_w1, s_dyn, s_kv_b, s_kv_h, s_kv_n, KV_OFF,
    K: tl.constexpr, KP: tl.constexpr, BLOCK_C: tl.constexpr, DYNAMIC: tl.constexpr,
    NORM: tl.constexpr, STORE_KV: tl.constexpr, KV_PAGE: tl.constexpr,
    RESID: tl.constexpr, HAS_ADD: tl.constexpr, ALIGN: tl.constexpr, S: tl.constexpr,
    KVQ: tl.constexpr, CKPT: tl.constexpr = False,
):
    _sconv_body(tl.program_id(0), tl.program_id(1),
                X, Out, State, SeqIdx, QStart, QSL, BlockTable, SeqLens, NumAcc, W, Dyn, KV, KVSlots, Last, Addend,
                CkOff, CkSi, C, hph, W_row, off, eps, clip, inv_norm, dyn_heads_w,
                s_x, s_o, s_st, ROW, s_bt, state_bs, s_w0, s_w1, s_dyn, s_kv_b, s_kv_h, s_kv_n, KV_OFF,
                K=K, KP=KP, BLOCK_C=BLOCK_C, DYNAMIC=DYNAMIC, NORM=NORM, STORE_KV=STORE_KV, KV_PAGE=KV_PAGE,
                RESID=RESID, HAS_ADD=HAS_ADD, ALIGN=ALIGN, S=S, KVQ=KVQ, CKPT=CKPT)


@triton.jit
def _sconv_kv_kernel(
    XK, OutK, WK, DynK, XV, OutV, WV, DynV,
    State, SeqIdx, QStart, QSL, BlockTable, SeqLens, NumAcc, KV, KVSlots, CkOff, CkSi,
    C, hph, W_row, off_k, off_v, eps, clip, inv_norm, dyn_heads_w,
    s_xk, s_xv, s_ok, s_ov, s_st, ROW, s_bt, state_bs, s_wk0, s_wk1, s_wv0, s_wv1, s_dynk, s_dynv,
    s_kv_b, s_kv_h, s_kv_n,
    K: tl.constexpr, KP: tl.constexpr, BLOCK_C: tl.constexpr, DYNAMIC: tl.constexpr,
    STORE_KV: tl.constexpr, KV_PAGE: tl.constexpr, ALIGN: tl.constexpr, S: tl.constexpr,
    KVQ: tl.constexpr, CKPT: tl.constexpr,
):
    """K and V streams of one visit in one launch (grid axis 2: 0 = K with the L2 clip, stored at KV_OFF 0;
    1 = V with the RMS norm, stored at KV_OFF hph). Each program runs exactly the single-stream body."""
    t = tl.program_id(0)
    cb = tl.program_id(1)
    if tl.program_id(2) == 0:
        _sconv_body(t, cb, XK, OutK, State, SeqIdx, QStart, QSL, BlockTable, SeqLens, NumAcc, WK, DynK, KV, KVSlots,
                    XK, XK, CkOff, CkSi, C, hph, W_row, off_k, eps, clip, inv_norm, dyn_heads_w,
                    s_xk, s_ok, s_st, ROW, s_bt, state_bs, s_wk0, s_wk1, s_dynk, s_kv_b, s_kv_h, s_kv_n, 0,
                    K=K, KP=KP, BLOCK_C=BLOCK_C, DYNAMIC=DYNAMIC, NORM=1, STORE_KV=STORE_KV, KV_PAGE=KV_PAGE,
                    RESID=False, HAS_ADD=False, ALIGN=ALIGN, S=S, KVQ=KVQ, CKPT=CKPT)
    else:
        _sconv_body(t, cb, XV, OutV, State, SeqIdx, QStart, QSL, BlockTable, SeqLens, NumAcc, WV, DynV, KV, KVSlots,
                    XV, XV, CkOff, CkSi, C, hph, W_row, off_v, eps, clip, inv_norm, dyn_heads_w,
                    s_xv, s_ov, s_st, ROW, s_bt, state_bs, s_wv0, s_wv1, s_dynv, s_kv_b, s_kv_h, s_kv_n, hph,
                    K=K, KP=KP, BLOCK_C=BLOCK_C, DYNAMIC=DYNAMIC, NORM=2, STORE_KV=STORE_KV, KV_PAGE=KV_PAGE,
                    RESID=False, HAS_ADD=False, ALIGN=ALIGN, S=S, KVQ=KVQ, CKPT=CKPT)


def short_conv_kv(xk: torch.Tensor, xv: torch.Tensor, state: torch.Tensor, off_k: int, off_v: int, hph: int,
                  row_width: int, seq_idx: torch.Tensor, query_start: torch.Tensor, query_start_loc: torch.Tensor,
                  block_table: torch.Tensor, seq_lens: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor,
                  dynk: torch.Tensor | None = None, dynv: torch.Tensor | None = None, clip: float = 0.0,
                  eps: float = 1e-6, kv: torch.Tensor | None = None, kv_slots: torch.Tensor | None = None,
                  align: bool = False, state_block_size: int = 1, num_accepted: torch.Tensor | None = None,
                  spec: int = 0, ckpt_off: torch.Tensor | None = None,
                  ckpt_si: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """short_conv(xk, ..., norm=1, clip, kv_off=0) and short_conv(xv, ..., norm=2, eps, kv_off=hph) in one launch
    (23 Nov 2026, decode profile: 192 conv launches per step). Same arguments and results as the two calls."""
    T, C = xk.shape
    assert xv.shape == (T, C) and (dynk is None) == (dynv is None)
    k = wk.shape[1] if dynk is None else wk.shape[0]
    assert (wv.shape[1] if dynv is None else wv.shape[0]) == k, "K and V convs must have the same kernel size"
    outk = torch.empty((T, C), dtype=torch.bfloat16, device=xk.device)
    outv = torch.empty((T, C), dtype=torch.bfloat16, device=xk.device)
    if T == 0:
        return outk, outv
    assert state.shape[1] == k - 1 + spec and state.stride(2) == 1
    assert spec == 0 or num_accepted is not None
    assert C % hph == 0 and hph & (hph - 1) == 0, "per-head norm needs a power-of-two head"
    inv = 1.0 / math.sqrt((2 * k + 1) if dynk is not None else (k + 1))
    store_kv = kv is not None
    kvq = store_kv and kv.dtype == torch.uint8
    if kvq:
        assert kv.shape[3] == 2 * hph, "K8/V4 store: one head per program"
    kvt = kv if store_kv else xk.view(1, 1, T, C)
    _sconv_kv_kernel[(T, C // hph, 2)](
        xk, outk, wk, dynk if dynk is not None else xk, xv, outv, wv, dynv if dynv is not None else xv,
        state, seq_idx, query_start, query_start_loc, block_table, seq_lens,
        num_accepted if num_accepted is not None else seq_lens, kvt, kv_slots if store_kv else seq_idx,
        ckpt_off if ckpt_off is not None else seq_lens, ckpt_si if ckpt_si is not None else seq_lens,
        C, hph, row_width, off_k, off_v, eps, clip, inv, hph if dynk is not None else 1,
        xk.stride(0), xv.stride(0), outk.stride(0), outv.stride(0), state.stride(0), state.stride(1),
        block_table.stride(0), state_block_size, wk.stride(0), wk.stride(1), wv.stride(0), wv.stride(1),
        dynk.stride(0) if dynk is not None else 0, dynv.stride(0) if dynv is not None else 0,
        kvt.stride(0), kvt.stride(1), kvt.stride(2),
        K=k, KP=triton.next_power_of_2(k - 1 + spec), BLOCK_C=hph, DYNAMIC=dynk is not None,
        STORE_KV=store_kv, KV_PAGE=kvt.shape[2] if store_kv else 1, ALIGN=align, S=spec, KVQ=kvq,
        CKPT=ckpt_off is not None,
    )
    return outk, outv


def short_conv(x: torch.Tensor, state: torch.Tensor, stream_off: int, hph: int, row_width: int,
               seq_idx: torch.Tensor, query_start: torch.Tensor, query_start_loc: torch.Tensor,
               block_table: torch.Tensor, seq_lens: torch.Tensor, weight: torch.Tensor,
               dyn: torch.Tensor | None = None, norm: int = 0, clip: float = 0.0,
               eps: float = 1e-6, kv: torch.Tensor | None = None,
               kv_slots: torch.Tensor | None = None, kv_off: int = 0,
               last: torch.Tensor | None = None,
               addend: torch.Tensor | None = None, align: bool = False,
               state_block_size: int = 1, num_accepted: torch.Tensor | None = None,
               spec: int = 0, ckpt_off: torch.Tensor | None = None,
               ckpt_si: torch.Tensor | None = None) -> torch.Tensor:
    """x (T, C) bf16 (C = heads * hph); state [slots, k - 1, heads * row_width]
    (the stream at column `stream_off` of each head's row); weight (C, k) static
    or (k, C) with `dyn` (T, H * k); returns (T, C) bf16, or with `last` (and
    `addend`), contiguous (T, C), the residual sum last (+ addend) + conv.
    A request's slot is block_table[req, 0], or in align mode the block of its
    last token; it holds history when seq_lens[req] exceeds its rows in the
    step. Updates the state slots of the requests in the step."""
    T, C = x.shape
    k = weight.shape[1] if dyn is None else weight.shape[0]
    out = torch.empty((T, C), dtype=torch.bfloat16, device=x.device)
    if T == 0:
        return out
    assert state.shape[1] == k - 1 + spec and state.stride(2) == 1
    assert spec == 0 or num_accepted is not None
    if norm:
        assert C % hph == 0
        block_c = hph
        assert block_c & (block_c - 1) == 0, "per-head norm needs a power-of-two head"
    else:
        block_c = min(1024, triton.next_power_of_2(C))
    inv = 1.0 / math.sqrt((2 * k + 1) if dyn is not None else (k + 1))
    store_kv = kv is not None
    kvq = store_kv and kv.dtype == torch.uint8
    if kvq:
        assert norm and block_c == hph and kv.shape[3] == 2 * hph, "K8/V4 store: one head per program"
    kvt = kv if store_kv else x.view(1, 1, T, C)
    _sconv_kernel[(T, triton.cdiv(C, block_c))](
        x, out, state, seq_idx, query_start, query_start_loc, block_table, seq_lens,
        num_accepted if num_accepted is not None else seq_lens, weight,
        dyn if dyn is not None else x, kvt, kv_slots if store_kv else seq_idx,
        last if last is not None else x, addend if addend is not None else x,
        ckpt_off if ckpt_off is not None else seq_lens, ckpt_si if ckpt_si is not None else seq_lens,
        C, hph, row_width, stream_off, eps, clip, inv, hph if dyn is not None else 1,
        x.stride(0), out.stride(0), state.stride(0), state.stride(1), block_table.stride(0),
        state_block_size,
        weight.stride(0), weight.stride(1), dyn.stride(0) if dyn is not None else 0,
        kvt.stride(0), kvt.stride(1), kvt.stride(2), kv_off,
        K=k, KP=triton.next_power_of_2(k - 1 + spec), BLOCK_C=block_c, DYNAMIC=dyn is not None,
        NORM=norm, STORE_KV=store_kv, KV_PAGE=kvt.shape[2] if store_kv else 1,
        RESID=last is not None, HAS_ADD=addend is not None, ALIGN=align, S=spec, KVQ=kvq,
        CKPT=ckpt_off is not None,
    )
    return out
