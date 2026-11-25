# oneDNN dense GEMMs at decode sizes (24 Nov 2026): the engine's int8 path (scaled_int8_quant + int8_scaled_mm, W8A8
# per-row weights) vs bf16 torch.mm, on Inkling's shapes. Time = XPU-graph replay of 20 calls; GB/s = weight bytes / time.
import time
import torch
from vllm import _custom_ops as vops

dev = "xpu"
torch.manual_seed(0)
PEAK = 608.0


def graph_time(fn, reps=20):
    fn(); torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        for _ in range(reps):
            fn()
    g.replay(); torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        g.replay()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / (10 * reps)


shapes = [("qkvr", 8600, 2304, (8, 32, 96)), ("wo", 2304, 3072, (8, 32)), ("head", 200064, 2304, (8, 32))]
for name, N, K, Ms in shapes:
    wq = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev)
    ws = (torch.rand(N, 1, device=dev) * 0.01).float()
    wb = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    for M in Ms:
        x = torch.randn(M, K, device=dev).to(torch.bfloat16)

        def int8_path():
            q, s, _ = vops.scaled_int8_quant(x, None, None, True)
            return torch.ops._xpu_C.int8_scaled_mm(q, s, wq.t(), ws, torch.bfloat16, None)

        t8 = graph_time(int8_path)
        tb = graph_time(lambda: torch.mm(x, wb.t()))
        g8 = N * K / t8 / 1e9
        gb = 2 * N * K / tb / 1e9
        print(f"{name:5s} M={M:3d}: int8 {t8 * 1e6:7.1f} us {g8:5.0f} GB/s ({g8 / PEAK * 100:3.0f}%) | bf16 {tb * 1e6:7.1f} us {gb:5.0f} GB/s ({gb / PEAK * 100:3.0f}%)")
    del wq, ws, wb
    torch.xpu.empty_cache()
