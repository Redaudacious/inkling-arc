# 23 Nov 2026 (Claude): hybrid prefill for Inkling global layers on the K8/V4 cache. Keys more than rel_extent (1024) behind
# every row of the chunk go through far_attention (fp16 blocks + oneDNN fused SDPA with stats); the rest through the ESIMD
# kernel starting at key_start; the combine merges both. Must match the Triton reference before speed counts.
import os, sys, time
import torch
sys.path[:0] = ["/cp", "/work/build", os.environ.get("VLLM_KVQ_EXT_DIR", "/opt/nvf4")]
try:
    from triton_attention import rel_attention  # dev layout
except ImportError:
    from vllm.models.inkling2.xpu.ops.triton_attention import rel_attention  # inkling-arc image
import _kvq_decode_xe2 as kx  # noqa: E402

assert kx.has_far, "extension built without KVQ_DNNL"
dev = "xpu"
Hq, Hk, D, PAGE, E = 24, 8, 128, 128, 1024
ALIGN = int(os.environ.get("FAR_ALIGN", "4096"))
BLOCK = int(os.environ.get("FAR_BLOCK", "32768"))
g = torch.Generator(device="cpu").manual_seed(3)


def make(seq_lens, q_lens):
    reqs = len(seq_lens)
    nblk = [(s + PAGE - 1) // PAGE for s in seq_lens]
    NB = sum(nblk) + 1
    kv = torch.randint(0, 256, (NB, Hk, PAGE, 2 * D), dtype=torch.uint8, generator=g)
    ks = (torch.rand(NB, Hk, PAGE, generator=g) * 0.02 + 0.005).float()
    vs = (torch.rand(NB, Hk, PAGE, 8, generator=g) * 0.02 + 0.005).half()
    kv[..., 192:196] = ks.view(torch.uint8).view(NB, Hk, PAGE, 4)
    kv[..., 196:212] = vs.view(torch.uint8).view(NB, Hk, PAGE, 16)
    kv = kv.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3).to(dev)   # live layout
    bt = torch.zeros(reqs, max(nblk), dtype=torch.int32)
    perm = torch.randperm(NB - 1, generator=g) + 1                          # shuffled physical pages
    nxt = 0
    for i, n in enumerate(nblk):
        bt[i, :n] = perm[nxt:nxt + n].to(torch.int32); nxt += n
    T = sum(q_lens)
    q = (torch.randn(T, Hq, D, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    rel = (torch.randn(T, Hq, E, generator=g) * 0.3).to(torch.bfloat16).to(dev)
    sinks = (torch.randn(T, Hq, generator=g)).to(torch.bfloat16).to(dev)
    cu = [0]
    for x in q_lens: cu.append(cu[-1] + x)
    return dict(kv=kv, q=q, rel=rel, sinks=sinks, bt=bt.to(dev), bt_cpu=bt, seq=torch.tensor(seq_lens, dtype=torch.int32).to(dev),
                seq_cpu=seq_lens, ql=q_lens, cu=torch.tensor(cu, dtype=torch.int32).to(dev), cu_cpu=cu, reqs=reqs)


def tri(a):
    out = torch.empty_like(a["q"])
    rel_attention(a["q"], a["kv"], a["rel"], a["sinks"], a["bt"], a["seq"], a["cu"], D ** -0.5, None, E, a["reqs"], out=out,
                  num_splits=1, rel_reversed=True)
    return out


def hybrid(a, splits=4):
    q = a["q"]; T = q.shape[0]
    key_start = torch.zeros(a["reqs"], dtype=torch.int32)
    extra_o = extra_l = None
    for r in range(a["reqs"]):
        ql = a["ql"][r]; past = a["seq_cpu"][r] - ql
        fe = (past + 1 - E) // ALIGN * ALIGN
        if ql > 8 and fe >= ALIGN:
            if extra_o is None:
                extra_o = torch.empty(T, Hq, D, dtype=torch.float32, device=dev)
                extra_l = torch.full((T, Hq), float("-inf"), dtype=torch.float32, device=dev)
            t0, t1 = a["cu_cpu"][r], a["cu_cpu"][r + 1]
            o, l = kx.far_attention(q[t0:t1], a["kv"], a["bt"][r, :fe // PAGE].contiguous(), fe, D ** -0.5, BLOCK)
            extra_o[t0:t1] = o; extra_l[t0:t1] = l; key_start[r] = fe
    out = torch.empty_like(q)
    if extra_o is None:
        kx.kvq_decode_xe2(q, a["kv"], a["rel"], a["sinks"], a["bt"], a["seq"], a["cu"], D ** -0.5, a["reqs"], max(a["ql"]), splits, out)
    else:
        kx.kvq_decode_xe2(q, a["kv"], a["rel"], a["sinks"], a["bt"], a["seq"], a["cu"], D ** -0.5, a["reqs"], max(a["ql"]), splits, out,
                          key_start.to(dev), extra_o, extra_l)
    return out


def esimd_only(a, splits=4):
    out = torch.empty_like(a["q"])
    kx.kvq_decode_xe2(a["q"], a["kv"], a["rel"], a["sinks"], a["bt"], a["seq"], a["cu"], D ** -0.5, a["reqs"], max(a["ql"]), splits, out)
    return out


def timed(fn, a, iters=5):
    for _ in range(2): fn(a)
    torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn(a)
    torch.xpu.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


print(f"== hybrid (align {ALIGN}, block {BLOCK}) vs Triton: max abs / rel L2", flush=True)
ok = True
for sl, ql in [([9000], [2048]), ([40000], [2048]), ([70000, 3000], [1500, 1]), ([20000, 50000], [700, 2048]), ([5000], [5000])]:
    a = make(sl, ql)
    r = tri(a).float(); h = hybrid(a).float()
    md = (r - h).abs().max().item(); rl = ((r - h).norm() / r.norm()).item()
    good = rl < 0.02 and md == md
    ok &= good
    print(f"  seq {sl} q {ql}: max {md:.4f} rel {rl:.4f} {'OK' if good else 'FAIL'}", flush=True)
print("CORRECT" if ok else "MISMATCH", flush=True)
if not ok and os.environ.get("FORCE") != "1":
    sys.exit(1)

print("== prefill chunk 2048 rows, 1 request: past | triton | esimd only | hybrid  (ms per global visit)", flush=True)
for past in [x for x in (30720, 63488, 129024, 260096, 489472) if x <= int(os.environ.get("PASTMAX", "260096"))]:
    a = make([past + 2048], [2048])
    tt = timed(tri, a, 3) if past <= 129024 else float("nan")
    te = timed(esimd_only, a)
    th = timed(hybrid, a)
    print(f"{past:6d} | {tt:8.1f} | {te:7.1f} | {th:7.1f}  ({te/th:4.1f}x over esimd)", flush=True)
