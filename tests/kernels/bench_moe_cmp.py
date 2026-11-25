# 23 Nov 2026 (Claude): expert op A/B on the same inputs, one process per library (both register _nvf4):
#   LIB=<.so> TAG=<name> -> times per batch size and saves outputs to /work/moe_out_<TAG>.pt; CMP=a,b compares two saved runs.
import os, time
import numpy as np
import torch

dev = "xpu"
if os.environ.get("CMP"):
    a, b = os.environ["CMP"].split(",")
    A = torch.load(f"/work/moe_out_{a}.pt"); B = torch.load(f"/work/moe_out_{b}.pt")
    for T in A:
        d = (A[T].float() - B[T].float()).abs().max().item()
        print(f"T {T}: max abs diff {d:.3e} {'IDENTICAL' if d == 0 else ''}")
    raise SystemExit
LAYER = int(os.environ.get("LAYER", "0"))
E, H, I, TOPK = 128, 2304, 1152, 8
torch.ops.load_library(os.environ["LIB"])


def pack(name):
    path = f"/packs/L{LAYER}.{name}.s8.nvf4"
    with open(path, "rb") as f:
        hdr = f.read(4096).decode(errors="ignore")
    meta = dict(ln.split("=", 1) for ln in hdr.splitlines()[1:] if "=" in ln)
    nb, e = int(meta["pack_bytes"]), int(meta["E"])
    p = torch.from_numpy(np.fromfile(path, dtype=np.uint8, count=nb, offset=4096)).to(dev)
    fac = torch.from_numpy(np.fromfile(path, dtype=np.float32, count=e, offset=4096 + nb)).to(dev)
    return p, fac


w13_pack, w13_fac = pack("w13")
w2_pack, w2_fac = pack("w2")
ours = lambda x, w, ids: torch.ops._nvf4.moe(x, w, ids, w13_pack, w13_fac, w2_pack, w2_fac, 4.0, 25.0, 0, I)


def timed(fn, *a, iters=10):
    for _ in range(3): fn(*a)
    torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn(*a)
    torch.xpu.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


outs = {}
print(f"[{os.environ.get('TAG')}] tokens | ms per layer | TFLOPS", flush=True)
for T in [int(t) for t in os.environ.get("TS", "64,256,1024,2048,4096,6144").split(",")]:
    g = torch.Generator(device="cpu").manual_seed(T)
    x = (torch.randn(T, H, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    logits = torch.randn(T, E, generator=g)
    skew = torch.linspace(1.5, -1.5, E)          # skewed routing like real layers
    wts, ids = torch.topk(torch.softmax(logits + skew, -1), TOPK, dim=-1)
    wts = (wts / wts.sum(-1, keepdim=True)).float().to(dev)
    ids = ids.to(torch.int32).to(dev)
    outs[T] = ours(x, wts, ids).cpu()
    ms = timed(ours, x, wts, ids, iters=200 if T <= 64 else 10)
    print(f"  {T:5d} | {ms:8.3f} | {T * TOPK * 2 * (3 * H * I) / ms / 1e9:6.1f}", flush=True)
torch.save(outs, f"/work/moe_out_{os.environ.get('TAG')}.pt")
