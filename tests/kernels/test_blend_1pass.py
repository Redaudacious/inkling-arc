# One-pass blend_prep / blend_out (ops/triton_blend.py) vs the port's two-pass triton_glue kernels: same inputs (S = 3 attention
# streams with dests shared across streams by a stride-0 view, S = 1 MLP stream), outputs compared, time per call
# (XPU graph replay).
import os, sys, time
import torch
from vllm.models.inkling2.xpu.ops import triton_glue
from vllm.models.inkling2.xpu.ops import triton_blend

triton_blend.PREP_ONE_PASS_MAX_T = 1 << 30   # compare the one-pass prep kernel at every size

dev = "xpu"; D = 2304
torch.manual_seed(0)


def graph_time(fn, reps=10):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        for _ in range(reps):
            fn()
    for _ in range(3):
        g.replay()
    torch.xpu.synchronize(); t0 = time.perf_counter()
    for _ in range(10):
        g.replay()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / (10 * reps) * 1e6


x = torch.randn(4096, 4096, device=dev)   # bring the GPU clocks up before timing
for _ in range(50):
    x = x @ x.T / 64
bad = 0
for T in [int(t) for t in os.environ.get("TS", "8,32,64,256,1024,2496").split(",")]:
    for S, shared in ((3, True), (3, False), (1, False)):
        dests = (torch.randn(1 if shared else S, T, D, device=dev) * 2).float()
        dests = dests.expand(S, -1, -1) if shared else dests
        messages = (torch.randn(S, T, D, device=dev) * 2).float()
        gf = torch.randn(S, T, 2 * D, device=dev).float()
        w = (1 + 0.1 * torch.randn(S, D, device=dev)).to(torch.bfloat16)
        a = triton_glue.blend_out(gf, dests, messages, w, 1e-6).float()
        b = triton_blend.blend_out(gf, dests, messages, w, 1e-6).float()
        rel = ((a - b).abs().max() / a.abs().max()).item()
        ok = rel <= 1e-2 and torch.isfinite(b).all().item()
        bad += 0 if ok else 1
        t2 = graph_time(lambda: triton_glue.blend_out(gf, dests, messages, w, 1e-6))
        t1 = graph_time(lambda: triton_blend.blend_out(gf, dests, messages, w, 1e-6))
        print(f"{'PASS' if ok else 'FAIL'} out  T={T:5d} S={S} shared={shared!s:5}: rel diff {rel:.1e} | two-pass {t2:7.1f} us "
              f"one-pass {t1:7.1f} us ({t2 / t1:4.2f}x)", flush=True)
        a = triton_glue.blend_prep(dests, messages).float()
        b = triton_blend.blend_prep(dests, messages).float()
        rel = ((a - b).abs().max() / a.abs().max()).item()
        ok = rel <= 1e-2 and torch.isfinite(b).all().item()
        bad += 0 if ok else 1
        t2 = graph_time(lambda: triton_glue.blend_prep(dests, messages))
        t1 = graph_time(lambda: triton_blend.blend_prep(dests, messages))
        print(f"{'PASS' if ok else 'FAIL'} prep T={T:5d} S={S} shared={shared!s:5}: rel diff {rel:.1e} | two-pass {t2:7.1f} us "
              f"one-pass {t1:7.1f} us ({t2 / t1:4.2f}x)", flush=True)
print("RESULT", "ALL PASS" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
