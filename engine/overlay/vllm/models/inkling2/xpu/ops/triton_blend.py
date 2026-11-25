# SPDX-License-Identifier: Apache-2.0
"""Loop-controller blend in one pass (inkling-arc, 24 Nov 2026).

blend_prep: the port's `triton_glue.blend_prep` (unweighted RMS of dest and message rows, bf16 [rms(dest), rms(msg)])
read each row twice (sums, then the scaled store); here once.
blend_out: same arithmetic as the port's `triton_glue.blend_out` per element: `(1 - tanh(forget)) dest + tanh(gain) message`,
then the next sublayer's weighted RMSNorm, bf16 out. The port's kernel walks each row twice in 512-wide chunks and
writes the un-normalised row to an fp32 scratch tensor in between; here the whole row (D <= 4096) stays in
registers, so there is no scratch write and read-back. Only the norm's sum of squares is summed in another order
(bf16 outputs within one bf16 step of the port's)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _blend_prep_1pass_kernel(Dst, Msg, Cat, T, s_ds, s_dt, s_ms, s_mt, eps, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    s = i // T
    t = i % T
    d = tl.arange(0, BLOCK)
    dm = d < D
    a = tl.load(Dst + s * s_ds + t * s_dt + d, mask=dm, other=0.0)
    b = tl.load(Msg + s * s_ms + t * s_mt + d, mask=dm, other=0.0)
    rd = tl.rsqrt(tl.sum(a * a, 0) / D + eps)
    rm = tl.rsqrt(tl.sum(b * b, 0) / D + eps)
    tl.store(Cat + i * (2 * D) + d, (a * rd).to(tl.bfloat16), mask=dm)
    tl.store(Cat + i * (2 * D) + D + d, (b * rm).to(tl.bfloat16), mask=dm)


# blend_prep takes one pass only up to this many tokens (tests/kernels/test_blend_1pass.py, B70: 1.06-2x at 8-64
# tokens, 0.75-1.04x at 256-2,496); larger calls keep the port's kernel.
PREP_ONE_PASS_MAX_T = 64


def blend_prep(dests, messages, eps=1e-6):
    """fp32 (S, T, D) dests / messages (any stride along S and T, unit along D) -> bf16 (S, T, 2D)
    [rms(dest), rms(message)]."""
    S, T, D = dests.shape
    if T > PREP_ONE_PASS_MAX_T:
        from .triton_glue import blend_prep as two_pass

        return two_pass(dests, messages, eps)
    cat = torch.empty((S, T, 2 * D), dtype=torch.bfloat16, device=dests.device)
    if S * T:
        _blend_prep_1pass_kernel[(S * T,)](dests, messages, cat, T, dests.stride(0), dests.stride(1),
                                           messages.stride(0), messages.stride(1), eps, D=D,
                                           BLOCK=triton.next_power_of_2(D), num_warps=32)
    return cat


@triton.jit
def _blend_out_1pass_kernel(GF, Dst, Msg, W, Out, T, s_ds, s_dt, s_ms, s_mt, s_w, eps, D: tl.constexpr,
                            BLOCK: tl.constexpr):
    i = tl.program_id(0)
    s = i // T
    t = i % T
    d = tl.arange(0, BLOCK)
    dm = d < D
    g = tl.load(GF + i * (2 * D) + d, mask=dm, other=0.0)
    f = tl.load(GF + i * (2 * D) + D + d, mask=dm, other=0.0)
    a = tl.load(Dst + s * s_ds + t * s_dt + d, mask=dm, other=0.0)
    b = tl.load(Msg + s * s_ms + t * s_mt + d, mask=dm, other=0.0)
    tg = 2.0 / (1.0 + tl.exp(-2.0 * g)) - 1.0
    tf = 2.0 / (1.0 + tl.exp(-2.0 * f)) - 1.0
    x = (1.0 - tf) * a + tg * b
    r = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
    w = tl.load(W + s * s_w + d, mask=dm, other=0.0).to(tl.float32)
    tl.store(Out + i * D + d, (w * (x * r)).to(tl.bfloat16), mask=dm)


def blend_out(gf, dests, messages, weight, eps):
    """gf fp32 (S, T, 2D) [gain; forget] pre-activations; dests / messages fp32 (S, T, D), any stride along S and T;
    weight (S, D) RMSNorm rows (stride 0 to share one) -> bf16 (S, T, D) normalised blends."""
    S, T, D = dests.shape
    out = torch.empty((S, T, D), dtype=torch.bfloat16, device=dests.device)
    if S * T:
        _blend_out_1pass_kernel[(S * T,)](gf, dests, messages, weight, out, T, dests.stride(0), dests.stride(1),
                                          messages.stride(0), messages.stride(1), weight.stride(0), eps, D=D,
                                          BLOCK=triton.next_power_of_2(D), num_warps=32)
    return out
