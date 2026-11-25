# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Changed by inkling-arc (Nov 2026) from F3zz1k/vllm inkling2-b70 @ edd722c for the Intel Arc Pro B70; see README.
"""Block attention residual (Triton, XPU): softmax mix of residual blocks.

For each token and pseudo-query `q` (fp32, pre-scaled by `D**-0.5`) the
sources `x_n` = stored blocks, then the partial block (+ an optional
addend), in fp32, are scored `(q . x_n) * rsqrt(mean(x_n^2) + eps)`; the
output is `sum_n softmax_n * x_n * rsqrt(mean(x_n^2) + eps)` (normalized
values), optionally followed by the next sublayer's weighted RMSNorm
(bf16 out) in the same kernel. One program per (query, token); two passes
over the sources (scores, then the mix), D in chunks.

(Upstream's NVIDIA kernel does not build for Intel GPUs: module build failure.)
"""

import torch
import triton
import triton.language as tl

# Calls of 17..ONE_PASS_MAX_T tokens with one query take the one-pass kernel (inkling-arc, 24 Nov 2026; B70, graph
# replay, tests/kernels/test_attn_res_1pass.py: 1.5-2.2x faster at 17-64 tokens, equal at 128, slower from 256 up,
# where its per-source row reductions cost more than the saved re-reads). The 4-chat decode verify batch (32 rows)
# is in this range; decode sizes up to 16 keep the whole-row two-pass path, prompts the chunked one.
ONE_PASS_MAX_T = 64


@triton.jit
def _src(Stack, s_st, Last, Addend, t, s_row, d, dm, n: tl.constexpr, N_STACK: tl.constexpr,
         HAS_ADDEND: tl.constexpr):
    if n < N_STACK:
        x = tl.load(Stack + n * s_st + t * s_row + d, mask=dm, other=0.0).to(tl.float32)
    else:
        x = tl.load(Last + t * s_row + d, mask=dm, other=0.0).to(tl.float32)
        if HAS_ADDEND:
            x += tl.load(Addend + t * s_row + d, mask=dm, other=0.0).to(tl.float32)
    return x


@triton.jit
def _rint(x):
    # round half to even, as the vendor's per-token int8 quant
    f = tl.floor(x)
    d = x - f
    up = (d > 0.5) | ((d == 0.5) & ((f - 2.0 * tl.floor(f * 0.5)) != 0.0))
    return tl.where(up, f + 1.0, f)


@triton.jit
def _attn_res_kernel(Stack, Last, Addend, Q, W, Out, OutB, XQ, XS, s_st, s_row, s_oq, s_ot, eps,
                     norm_eps, D: tl.constexpr, N_STACK: tl.constexpr, HAS_ADDEND: tl.constexpr,
                     APPLY_NORM: tl.constexpr, BLOCK: tl.constexpr, NP2: tl.constexpr,
                     QUANT: tl.constexpr):
    qi = tl.program_id(0)
    t = tl.program_id(1)
    N: tl.constexpr = N_STACK + 1
    lanes = tl.arange(0, NP2)
    scores = tl.full([NP2], float("-inf"), tl.float32)
    inv = tl.zeros([NP2], tl.float32)
    for n in tl.static_range(N):
        dot = 0.0
        ss = 0.0
        for c in tl.static_range(0, D, BLOCK):
            d = c + tl.arange(0, BLOCK)
            dm = d < D
            x = _src(Stack, s_st, Last, Addend, t, s_row, d, dm, n, N_STACK, HAS_ADDEND)
            q = tl.load(Q + qi * D + d, mask=dm, other=0.0)
            dot += tl.sum(x * q, 0)
            ss += tl.sum(x * x, 0)
        r = 1.0 / tl.sqrt(ss / D + eps)
        scores = tl.where(lanes == n, dot * r, scores)
        inv = tl.where(lanes == n, r, inv)
    p = tl.exp(scores - tl.max(scores, 0))
    p = p / tl.sum(p, 0)
    wgt = p * inv  # weight of the raw source n
    # pass 2: the mix, chunk by chunk (and its sum of squares for the norm)
    ss_out = 0.0
    for c in tl.static_range(0, D, BLOCK):
        d = c + tl.arange(0, BLOCK)
        dm = d < D
        acc = tl.zeros([BLOCK], tl.float32)
        for n in tl.static_range(N):
            x = _src(Stack, s_st, Last, Addend, t, s_row, d, dm, n, N_STACK, HAS_ADDEND)
            acc += x * tl.sum(tl.where(lanes == n, wgt, 0.0), 0)
        if APPLY_NORM:
            ss_out += tl.sum(acc * acc, 0)
            tl.store(Out + qi * s_oq + t * s_ot + d, acc, mask=dm)  # fp32 scratch row
        else:
            tl.store(Out + qi * s_oq + t * s_ot + d, acc, mask=dm)
    if APPLY_NORM:  # bf16 out (OutB), and with QUANT (one chunk per row) int8 rows + scales
        r = 1.0 / tl.sqrt(ss_out / D + norm_eps)
        for c in tl.static_range(0, D, BLOCK):
            d = c + tl.arange(0, BLOCK)
            dm = d < D
            acc = tl.load(Out + qi * s_oq + t * s_ot + d, mask=dm, other=0.0)
            w = tl.load(W + qi * D + d, mask=dm, other=0.0).to(tl.float32)
            y = (w * (acc * r)).to(tl.bfloat16)
            tl.store(OutB + qi * s_oq + t * s_ot + d, y, mask=dm)
            if QUANT:
                yf = y.to(tl.float32)
                sc = tl.maximum(tl.max(tl.abs(yf), 0) / 127.0, 1e-10)
                qv = _rint(yf / sc)
                tl.store(XQ + (qi * (s_oq // D) + t) * D + d, qv.to(tl.int8), mask=dm)
                tl.store(XS + qi * (s_oq // D) + t, sc)


@triton.jit
def _attn_res_1pass_kernel(Stack, Last, Addend, Q, W, Out, OutB, s_st, s_row, s_oq, s_ot, eps, norm_eps,
                           D: tl.constexpr, N_STACK: tl.constexpr, HAS_ADDEND: tl.constexpr,
                           APPLY_NORM: tl.constexpr, BLOCK: tl.constexpr):
    # The same mix in ONE pass over the sources (inkling-arc, 24 Nov 2026): the whole row of the current source and
    # the running mix stay in registers (online softmax over sources: the mix is rescaled when the running max
    # grows), so each source row is read once and no fp32 scratch row is written and read back.
    qi = tl.program_id(0)
    t = tl.program_id(1)
    d = tl.arange(0, BLOCK)
    dm = d < D
    q = tl.load(Q + qi * D + d, mask=dm, other=0.0)
    acc = tl.zeros([BLOCK], tl.float32)
    m = float("-inf")
    l = 0.0
    for n in tl.static_range(N_STACK + 1):
        x = _src(Stack, s_st, Last, Addend, t, s_row, d, dm, n, N_STACK, HAS_ADDEND)
        r = 1.0 / tl.sqrt(tl.sum(x * x, 0) / D + eps)
        s = tl.sum(x * q, 0) * r
        m_new = tl.maximum(m, s)
        alpha = tl.exp(m - m_new)
        beta = tl.exp(s - m_new)
        acc = acc * alpha + x * (beta * r)
        l = l * alpha + beta
        m = m_new
    acc = acc / l
    if APPLY_NORM:
        rn = 1.0 / tl.sqrt(tl.sum(acc * acc, 0) / D + norm_eps)
        w = tl.load(W + qi * D + d, mask=dm, other=0.0).to(tl.float32)
        tl.store(OutB + qi * s_oq + t * s_ot + d, (w * (acc * rn)).to(tl.bfloat16), mask=dm)
    else:
        tl.store(Out + qi * s_oq + t * s_ot + d, acc, mask=dm)


def attn_res(stack: torch.Tensor, last: torch.Tensor, addend: torch.Tensor | None,
             queries: torch.Tensor, norm_weight: torch.Tensor | None, norm_eps: float,
             normalize_values: bool = True, eps: float = 1e-6, quant: bool = False):
    """stack (N, T, D) (N may be 0), last/addend (T, D), queries fp32 (Q, D);
    returns fp32 (Q, T, D), or bf16 normalized with norm_weight (Q, D). With `quant`
    (and norm_weight) returns (bf16 out, (int8 (Q*T, D), fp32 scales (Q*T,))) when the
    rows are decode-sized, else (bf16 out, None)."""
    assert normalize_values
    T, D = last.shape
    Q = queries.shape[0]
    N = stack.shape[0]
    assert last.stride(1) == 1 and (N == 0 or (stack.stride(1) == last.stride(0)
                                               and stack.stride(2) == 1))
    whole = T <= 16
    if not whole and T <= ONE_PASS_MAX_T and Q == 1:
        outb = (torch.empty((Q, T, D), dtype=torch.bfloat16, device=last.device)
                if norm_weight is not None else None)
        out = (torch.empty((Q, T, D), dtype=torch.float32, device=last.device)
               if norm_weight is None else outb)
        if T:
            _attn_res_1pass_kernel[(Q, T)](
                stack if N else last, last, addend if addend is not None else last, queries,
                norm_weight if norm_weight is not None else queries, out, outb if outb is not None else out,
                stack.stride(0) if N else 0, last.stride(0), out.stride(0), out.stride(1), eps, norm_eps,
                D=D, N_STACK=N, HAS_ADDEND=addend is not None, APPLY_NORM=norm_weight is not None,
                BLOCK=triton.next_power_of_2(D), num_warps=32)
        if norm_weight is None:
            return out
        return (outb, None) if quant else outb
    out = torch.empty((Q, T, D), dtype=torch.float32, device=last.device)
    outb = (torch.empty((Q, T, D), dtype=torch.bfloat16, device=last.device)
            if norm_weight is not None else out)
    # decode sizes have only Q * T programs: give each one the whole row
    # (measured 6-14 us vs 21-42 us per call at T=1; 4096/16 warps miscompiles)
    blk, nw = (triton.next_power_of_2(D), 32) if whole else (1024, 8)
    do_q = quant and whole and norm_weight is not None
    xq = torch.empty((Q * T, D), dtype=torch.int8, device=last.device) if do_q else out
    xs = torch.empty((Q * T,), dtype=torch.float32, device=last.device) if do_q else out
    if T:
        _attn_res_kernel[(Q, T)](
            stack if N else last, last, addend if addend is not None else last, queries,
            norm_weight if norm_weight is not None else queries, out, outb, xq, xs,
            stack.stride(0) if N else 0, last.stride(0), out.stride(0), out.stride(1), eps,
            norm_eps, D=D, N_STACK=N, HAS_ADDEND=addend is not None,
            APPLY_NORM=norm_weight is not None, BLOCK=blk, NP2=triton.next_power_of_2(N + 1),
            QUANT=do_q, num_warps=nw)
    if norm_weight is None:
        return out
    if quant:
        return outb, ((xq, xs) if do_q else None)
    return outb
