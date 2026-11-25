# 23 Nov 2026 (Claude): new ESIMD K8/V4 global-layer attention (_kvq_decode_xe2) vs our Triton _rel_attn_kernel (the live path):
# same K8/V4 cache, queries, rel bias and sinks; outputs must agree before speed is compared.
import os, sys, time
import torch
sys.path[:0] = ["/cp", "/work/build", os.environ.get("VLLM_KVQ_EXT_DIR", "/opt/nvf4")]
try:
    from triton_attention import rel_attention  # dev layout
except ImportError:
    from vllm.models.inkling2.xpu.ops.triton_attention import rel_attention  # inkling-arc image
import _kvq_decode_xe2 as kx  # noqa: E402

dev = "xpu"
Hq, Hk, D, PAGE, E = 24, 8, 128, 128, 1024
g = torch.Generator(device="cpu").manual_seed(1)

def make(seq_lens, q_lens):
    reqs = len(seq_lens)
    nblk = [(s + PAGE - 1) // PAGE for s in seq_lens]
    NB = sum(nblk) + 1
    kv = torch.randint(0, 256, (NB, Hk, PAGE, 2 * D), dtype=torch.uint8, generator=g)
    ks = (torch.rand(NB, Hk, PAGE, generator=g) * 0.02 + 0.005).float()
    vs = (torch.rand(NB, Hk, PAGE, 8, generator=g) * 0.02 + 0.005).half()
    kv[..., 192:196] = ks.view(torch.uint8).view(NB, Hk, PAGE, 4)
    kv[..., 196:212] = vs.view(torch.uint8).view(NB, Hk, PAGE, 16)
    if os.environ.get("LAYOUT", "live") == "live":   # live server: memory [NB, PAGE, KV_H, 256], viewed [NB, KV_H, PAGE, 256]
        kv = kv.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
    kv = kv.to(dev)
    maxp = max(nblk)
    bt = torch.zeros(reqs, maxp, dtype=torch.int32)
    nxt = 1
    for i, n in enumerate(nblk):
        bt[i, :n] = torch.arange(nxt, nxt + n, dtype=torch.int32); nxt += n
    T = sum(q_lens)
    q = (torch.randn(T, Hq, D, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    rel = (torch.randn(T, Hq, E, generator=g) * 0.3).to(torch.bfloat16).to(dev)
    sinks = (torch.randn(T, Hq, generator=g)).to(torch.bfloat16).to(dev)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lens), 0)), dtype=torch.int32).to(dev)
    return kv, q, rel, sinks, bt.to(dev), torch.tensor(seq_lens, dtype=torch.int32).to(dev), cu, reqs, max(q_lens)

def tri(a, splits=32):
    kv, q, rel, sinks, bt, seq, cu, reqs, mq = a
    out = torch.empty_like(q)
    rel_attention(q, kv, rel, sinks, bt, seq, cu, D ** -0.5, None, E, reqs, out=out, num_splits=splits, rel_reversed=True)
    return out

def esimd(a, splits):
    kv, q, rel, sinks, bt, seq, cu, reqs, mq = a
    out = torch.empty_like(q)
    kx.kvq_decode_xe2(q, kv, rel, sinks, bt, seq, cu, D ** -0.5, reqs, mq, splits, out)
    return out

def timed(fn, *a, iters=30):
    for _ in range(2 if iters < 10 else 5): fn(*a)
    torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn(*a)
    torch.xpu.synchronize()
    return (time.perf_counter() - t) / iters * 1e3

print("== layout", os.environ.get("LAYOUT", "live"), "| correctness (max abs diff / rel L2) vs Triton", flush=True)
cases = [([50], [1]), ([700], [8]), ([1000, 37, 5000], [8, 3, 1]), ([4096], [8]), ([3000, 3000, 2999, 64], [8, 8, 8, 8]),
         ([20000], [5]), ([2048], [2048]), ([7000], [2048]), ([5000, 900, 3000], [1, 900, 8]), ([300, 4000], [300, 1])]
ok = True
for sl, ql in cases:
    a = make(sl, ql)
    tsp = 32 if max(ql) <= 8 else 1
    r = tri(a, tsp).float(); e = esimd(a, 16 if max(ql) <= 8 else 2).float()
    md = (r - e).abs().max().item(); rl = ((r - e).norm() / r.norm()).item()
    good = rl < 0.02 and md == md
    ok &= good
    print(f"  seq {sl} q {ql}: max {md:.4f} rel {rl:.4f} {'OK' if good else 'FAIL'}", flush=True)
print("CORRECT" if ok else "MISMATCH", flush=True)
if not ok and os.environ.get("FORCE") != "1":
    sys.exit(1)

if os.environ.get("SPEED", "1") != "1":
    sys.exit(0)
if os.environ.get("SPEED", "1") != "1":
    sys.exit(0)
print("== speed: ctx | reqs | triton ms (32 splits) | esimd ms (best splits) | speedup", flush=True)
for ctx in (4096, 32768, 65536, 131072):
    for reqs in (1, 4):
        if ctx * reqs > 262144: continue
        a = make([ctx] * reqs, [8] * reqs)
        tt = timed(tri, a)
        best = None
        for sp in (16, 32, 64, 128):
            te = timed(esimd, a, sp)
            if best is None or te < best[0]: best = (te, sp)
        print(f"{ctx:6d} | {reqs} | {tt:7.3f} | {best[0]:7.3f} (splits {best[1]}) | {tt/best[0]:5.1f}x", flush=True)

if os.environ.get("PREFILL", "1") == "1":
    print("== prefill chunk 2048 rows, 1 request: past | triton ms (splits 1, live) | esimd ms (best splits) | speedup", flush=True)
    for past in (0, 30720, 63488, 129024):
        a = make([past + 2048], [2048])
        tt = timed(tri, a, 1, iters=5)
        row = {sp: timed(esimd, a, sp, iters=5) for sp in (1, 2, 4, 8, 16)}
        best = min(row, key=row.get)
        print(f"{past:6d} | {tt:8.2f} | " + " ".join(f"s{sp}={ms:.1f}" for sp, ms in row.items()) +
              f" | best s{best} {tt/row[best]:5.1f}x", flush=True)
