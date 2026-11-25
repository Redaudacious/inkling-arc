# short_conv_kv (K and V short convs in one launch) vs two short_conv calls: outputs, conv state and the K8/V4 KV
# cache must be byte-identical. Cases: speculative decode (4 requests x 8 rows), prompt rows with and without history,
# static (local) and dynamic (global) taps, align mode. Times both at decode sizes.
import sys, time
import torch
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "/t")
from triton_sconv import short_conv, short_conv_kv

dev = "xpu"
torch.manual_seed(0)
H, hph, W_row, K, S = 8, 128, 512, 4, 7
C = H * hph
off_k, off_v = 0, 128
fails = 0


def case(name, lens_q, seq_lens, dynamic, align, num_acc):
    global fails
    nreq = len(lens_q)
    T = sum(lens_q)
    qsl = [0]
    for L in lens_q:
        qsl.append(qsl[-1] + L)
    seq_idx = torch.tensor([r for r in range(nreq) for _ in range(lens_q[r])], dtype=torch.int32, device=dev)
    qstart = torch.tensor([qsl[r] for r in range(nreq) for _ in range(lens_q[r])], dtype=torch.int32, device=dev)
    qsl_t = torch.tensor(qsl, dtype=torch.int32, device=dev)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    sbs = 16 if align else 1
    cols = 64
    bt = torch.randperm(nreq * cols, device="cpu").view(nreq, cols).to(torch.int32).to(dev)
    slots = nreq * cols
    state0 = (torch.randn(slots, K - 1 + S, H * W_row, device=dev) * 0.5).to(torch.bfloat16)
    NB, PAGE = 64, 128
    kv0 = torch.randint(0, 255, (NB, H, PAGE, 2 * hph), dtype=torch.uint8, device=dev)
    kv_slots = torch.randperm(NB * PAGE, device="cpu")[:T].to(torch.int64).to(dev)
    xk = (torch.randn(T, C, device=dev)).to(torch.bfloat16)
    xv = (torch.randn(T, C, device=dev)).to(torch.bfloat16)
    if dynamic:
        wk = torch.randn(K, C, device=dev).to(torch.bfloat16) * 0.3
        wv = torch.randn(K, C, device=dev).to(torch.bfloat16) * 0.3
        dk = torch.randn(T, H * K, device=dev).to(torch.bfloat16) * 0.3
        dv = torch.randn(T, H * K, device=dev).to(torch.bfloat16) * 0.3
    else:
        wk = torch.randn(C, K, device=dev).to(torch.bfloat16) * 0.3
        wv = torch.randn(C, K, device=dev).to(torch.bfloat16) * 0.3
        dk = dv = None
    na = torch.tensor(num_acc, dtype=torch.int32, device=dev)
    common = dict(align=align, state_block_size=sbs, num_accepted=na, spec=S)
    # reference: two single-stream calls
    st1, kv1 = state0.clone(), kv0.clone()
    ok1 = short_conv(xk, st1, off_k, hph, W_row, seq_idx, qstart, qsl_t, bt, sl, wk, dyn=dk, norm=1, clip=2.5,
                     kv=kv1, kv_slots=kv_slots, kv_off=0, **common)
    ov1 = short_conv(xv, st1, off_v, hph, W_row, seq_idx, qstart, qsl_t, bt, sl, wv, dyn=dv, norm=2, eps=1e-6,
                     kv=kv1, kv_slots=kv_slots, kv_off=hph, **common)
    st2, kv2 = state0.clone(), kv0.clone()
    ok2, ov2 = short_conv_kv(xk, xv, st2, off_k, off_v, hph, W_row, seq_idx, qstart, qsl_t, bt, sl, wk, wv,
                             dynk=dk, dynv=dv, clip=2.5, eps=1e-6, kv=kv2, kv_slots=kv_slots, **common)
    torch.xpu.synchronize()
    same = (torch.equal(ok1, ok2) and torch.equal(ov1, ov2) and torch.equal(st1, st2) and torch.equal(kv1, kv2))
    fails += 0 if same else 1
    detail = "" if same else (f" out_k {torch.equal(ok1, ok2)} out_v {torch.equal(ov1, ov2)} state {torch.equal(st1, st2)}"
                              f" kv {torch.equal(kv1, kv2)}")
    print(f"{'PASS' if same else 'FAIL'} {name} (T={T}){detail}")
    return xk, xv, st2, kv2, wk, wv, dk, dv, seq_idx, qstart, qsl_t, bt, sl, kv_slots, common


def ckpt_case(name, ck_vals, repeats=30):
    """Prompt rows with history in align mode, the prefix-cache checkpoint at ck tokens into the chunk (Fable review:
    the checkpoint block reads old rows of the slot it later overwrites). Pair vs two calls, and repeat determinism."""
    global fails
    lens_q, seq_lens = [20, 37], [70, 137]
    nreq, T = 2, 57
    qsl = [0, 20, 57]
    seq_idx = torch.tensor([r for r in range(nreq) for _ in range(lens_q[r])], dtype=torch.int32, device=dev)
    qstart = torch.tensor([qsl[r] for r in range(nreq) for _ in range(lens_q[r])], dtype=torch.int32, device=dev)
    qsl_t = torch.tensor(qsl, dtype=torch.int32, device=dev)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    bt = torch.randperm(nreq * 64).view(nreq, 64).to(torch.int32).to(dev)
    state0 = (torch.randn(nreq * 64, K - 1 + S, H * W_row, device=dev) * 0.5).to(torch.bfloat16)
    kv0 = torch.randint(0, 255, (64, H, 128, 2 * hph), dtype=torch.uint8, device=dev)
    kv_slots = torch.randperm(64 * 128)[:T].to(torch.int64).to(dev)
    xk = torch.randn(T, C, device=dev).to(torch.bfloat16); xv = torch.randn(T, C, device=dev).to(torch.bfloat16)
    wk = torch.randn(C, K, device=dev).to(torch.bfloat16) * 0.3; wv = torch.randn(C, K, device=dev).to(torch.bfloat16) * 0.3
    na = torch.ones(nreq, dtype=torch.int32, device=dev)
    ck_off = torch.tensor(ck_vals, dtype=torch.int32, device=dev)
    ck_si = torch.tensor([int(bt[0, 63]), int(bt[1, 63])], dtype=torch.int32, device=dev)
    cm = dict(align=True, state_block_size=16, num_accepted=na, spec=S, ckpt_off=ck_off, ckpt_si=ck_si)

    def two():
        st, kv = state0.clone(), kv0.clone()
        short_conv(xk, st, off_k, hph, W_row, seq_idx, qstart, qsl_t, bt, sl, wk, norm=1, clip=2.5, kv=kv, kv_slots=kv_slots, kv_off=0, **cm)
        short_conv(xv, st, off_v, hph, W_row, seq_idx, qstart, qsl_t, bt, sl, wv, norm=2, eps=1e-6, kv=kv, kv_slots=kv_slots, kv_off=hph, **cm)
        return st, kv

    def one():
        st, kv = state0.clone(), kv0.clone()
        short_conv_kv(xk, xv, st, off_k, off_v, hph, W_row, seq_idx, qstart, qsl_t, bt, sl, wk, wv, clip=2.5, eps=1e-6, kv=kv, kv_slots=kv_slots, **cm)
        return st, kv

    a = two()
    b = one()
    torch.xpu.synchronize()
    pair_ok = torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    rep_bad = sum(not torch.equal(a[0], two()[0]) for _ in range(repeats)) + sum(not torch.equal(a[0], one()[0]) for _ in range(repeats))
    torch.xpu.synchronize()
    ok = pair_ok and rep_bad == 0
    fails += 0 if ok else 1
    print(f"{'PASS' if ok else 'FAIL'} {name}: pair equal {pair_ok}, repeats differing {rep_bad}/{2 * repeats}")


ckpt_case("checkpoint at ck=1 (inside the first K-2)", [1, 2])
ckpt_case("checkpoint at ck=5 / 30", [5, 30])
case("decode 4x8 static", [8] * 4, [50, 80, 120, 200], False, False, [1, 3, 8, 5])
case("decode 4x8 dynamic", [8] * 4, [50, 80, 120, 200], True, False, [2, 8, 1, 4])
case("decode 1x8 dynamic align", [8], [333], True, True, [6])
case("prompt rows, no history", [20, 37], [20, 37], False, False, [1, 1])
case("prompt rows with history", [20, 37], [70, 137], True, True, [1, 1])
case("mixed: prompt + decode", [37, 8, 8], [37, 99, 140], False, False, [1, 4, 8])
args = case("timing 4x8 dynamic", [8] * 4, [500, 800, 1200, 2000], True, False, [8, 8, 8, 8])
xk, xv, st, kv, wk, wv, dk, dv, seq_idx, qstart, qsl_t, bt, sl, kv_slots, common = args


def two():
    short_conv(xk, st, off_k, hph, W_row, seq_idx, qstart, qsl_t, bt, sl, wk, dyn=dk, norm=1, clip=2.5, kv=kv,
               kv_slots=kv_slots, kv_off=0, **common)
    short_conv(xv, st, off_v, hph, W_row, seq_idx, qstart, qsl_t, bt, sl, wv, dyn=dv, norm=2, eps=1e-6, kv=kv,
               kv_slots=kv_slots, kv_off=hph, **common)


def one():
    short_conv_kv(xk, xv, st, off_k, off_v, hph, W_row, seq_idx, qstart, qsl_t, bt, sl, wk, wv, dynk=dk, dynv=dv,
                  clip=2.5, eps=1e-6, kv=kv, kv_slots=kv_slots, **common)


for name, fn in (("two launches", two), ("one launch", one)):
    for _ in range(20):
        fn()
    torch.xpu.synchronize(); t0 = time.perf_counter()
    for _ in range(500):
        fn()
    torch.xpu.synchronize()
    print(f"time {name}: {(time.perf_counter() - t0) / 500 * 1e6:.1f} us per K+V pair")
print("RESULT", "ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
