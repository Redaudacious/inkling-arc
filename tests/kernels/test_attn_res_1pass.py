# attn_res one-pass kernel (forced for every size above 16 tokens) vs the port's two-pass kernel: same inputs, outputs compared in fp32
# (bf16 outputs may differ by one bf16 step where the two summation orders round differently), plus time per call
# (XPU graph replay). Cases cover the routed calls: 0-8 stored blocks, one query (3-query calls keep the two-pass
# kernel), fused norm or fp32 out,
# with and without the MLP addend.
import os, sys, time
import torch
import vllm.models.inkling2.xpu.ops.triton_attn_res as ar

dev = "xpu"
torch.manual_seed(0)
D = 2304


def call(stack, last, addend, q, w, one):
    ar.ONE_PASS_MAX_T = 1 << 30 if one else 0
    return ar.attn_res(stack, last, addend, q, w, 1e-6)


def graph_time(fn, reps=10):
    fn(); torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        for _ in range(reps):
            fn()
    g.replay(); torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        g.replay()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / (5 * reps) * 1e6


bad = 0
tot_old = tot_new = 0.0
for T in [int(t) for t in os.environ.get("TS", "24,32,48,64").split(",")]:
    for N, Q, norm, add in ((0, 1, True, False), (2, 1, True, False), (5, 1, True, True), (8, 1, True, False),
                            (5, 1, False, True)):
        stack = (torch.randn(N, T, D, device=dev) * 2).to(torch.bfloat16)
        last = (torch.randn(T, D, device=dev) * 2).to(torch.bfloat16)
        addend = torch.randn(T, D, device=dev).to(torch.bfloat16) if add else None
        q = (torch.randn(Q, D, device=dev) * D ** -0.5 * 3).float()
        w = (1 + 0.1 * torch.randn(Q, D, device=dev)).to(torch.bfloat16) if norm else None
        a = call(stack, last, addend, q, w, False).float()
        b = call(stack, last, addend, q, w, True).float()
        rel = ((a - b).abs().max() / a.abs().max()).item()
        # bf16 output: one step at the row's largest values is 2^-7 relative; fp32 output: ~1e-6
        lim = 1e-2 if norm else 1e-5
        ok = rel <= lim and torch.isfinite(b).all().item()
        bad += 0 if ok else 1
        to = graph_time(lambda: call(stack, last, addend, q, w, False))
        tn = graph_time(lambda: call(stack, last, addend, q, w, True))
        if T == 32:
            tot_old += to; tot_new += tn
        print(f"{'PASS' if ok else 'FAIL'} T={T:5d} N={N} Q={Q} norm={norm!s:5} addend={add!s:5}: rel diff {rel:.2e} | "
              f"two-pass {to:7.1f} us  one-pass {tn:7.1f} us ({to / tn:4.2f}x)", flush=True)
print(f"T=32 total over cases: {tot_old:.0f} -> {tot_new:.0f} us")
print("RESULT", "ALL PASS" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
