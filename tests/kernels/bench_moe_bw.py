# Expert op bandwidth at decode sizes (24 Nov 2026): bytes actually read = distinct experts x pack bytes per expert;
# time = XPU-graph replay of 20 calls (no host overhead). Routing: "independent" (each token its own skewed top-8)
# and "drafted" (requests of 8 rows; rows of one request share most experts, like verify rows of one chat).
import os, time
import numpy as np
import torch

dev = "xpu"
torch.ops.load_library(os.environ.get("LIB", "/opt/nvf4/lib_nvf4_op.so"))
LAYER = int(os.environ.get("LAYER", "0"))
E, H, I, TOPK = 128, 2304, 1152, 8
PEAK = 608.0


FMT = int(os.environ.get("FMT", "0"))           # 0 = NVFP4 packs (*.s8.nvf4), 2 = MXFP8 packs (*.mx8.nvf4, layers 20-23)
SUFFIX = "mx8" if FMT == 2 else "s8"


def pack(name):
    path = f"/packs/L{LAYER}.{name}.{SUFFIX}.nvf4"
    with open(path, "rb") as f:
        hdr = f.read(4096).decode(errors="ignore")
    meta = dict(ln.split("=", 1) for ln in hdr.splitlines()[1:] if "=" in ln)
    nb, e = int(meta["pack_bytes"]), int(meta["E"])
    p = torch.from_numpy(np.fromfile(path, dtype=np.uint8, count=nb, offset=4096)).to(dev)
    fac = torch.from_numpy(np.fromfile(path, dtype=np.float32, count=e, offset=4096 + nb)).to(dev)
    return p, fac, nb


w13, f13, nb13 = pack("w13")
w2, f2, nb2 = pack("w2")
per_expert = (nb13 + nb2) / E
print(f"bytes per expert {per_expert / 1e6:.3f} MB")
g = torch.Generator().manual_seed(0)
skew = torch.linspace(1.5, -1.5, E)


def routing(T, mode):
    if mode == "independent":
        logits = torch.randn(T, E, generator=g) + skew
        return torch.topk(logits, TOPK, dim=-1).indices
    reqs = max(1, T // 8)
    rows = []
    for r in range(reqs):
        base = torch.randn(E, generator=g) + skew
        for _ in range(min(8, T)):
            logits = base + 0.6 * torch.randn(E, generator=g)   # small per-row change: drafted rows share experts
            rows.append(torch.topk(logits, TOPK).indices)
    return torch.stack(rows)[:T]


MODES = os.environ.get("MODES", "independent,drafted").split(",")
TS = [int(t) for t in os.environ.get("TS", "1,8,16,32,64").split(",")]
for mode in MODES:
    for T in TS:
        ids = routing(T, mode).to(torch.int32)
        distinct = len(torch.unique(ids))
        wts = torch.full((T, TOPK), 1.0 / TOPK, dtype=torch.float32)
        x = (torch.randn(T, H) * 0.5).to(torch.bfloat16).to(dev)
        idsd, wtsd = ids.to(dev), wts.to(dev)
        fn = lambda: torch.ops._nvf4.moe(x, wtsd, idsd, w13, f13, w2, f2, 4.0, 25.0, FMT, I)
        fn(); torch.xpu.synchronize()
        gr = torch.xpu.XPUGraph()
        with torch.xpu.graph(gr):
            for _ in range(20):
                fn()
        gr.replay(); torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            gr.replay()
        torch.xpu.synchronize()
        ms = (time.perf_counter() - t0) / 200 * 1e3
        gbs = distinct * per_expert / (ms * 1e-3) / 1e9
        print(f"{mode:11s} T={T:3d} distinct experts {distinct:3d}  {ms:6.3f} ms per layer  {gbs:6.1f} GB/s ({gbs / PEAK * 100:4.1f}% of 608)")
