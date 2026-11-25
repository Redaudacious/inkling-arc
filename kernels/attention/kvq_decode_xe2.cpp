// SPDX-License-Identifier: Apache-2.0
// Inkling 2 Mini global-layer decode / DSpark-verify attention on Intel Xe2 (B70), ESIMD + XMX, for OUR K8/V4 KV cache.
//
// 23 Nov 2026 (Claude). Ported from SergiioB/intel-arc-pro-b70-inference-cookbook
// patches/inkling2-mini-vllm-xpu/attn_esimd_xe2.cpp (Apache-2.0/MIT, two-pass ESIMD XMX split-KV paged decode attention for the
// Inkling 2 rel_logits/sink contract). Changes for our port:
//   - KV slot (256 B per token per KV head): [K int8 x128 | V int4 x128 packed lo|hi<<4 (64 B) | K scale f32 @192 | V scale f16 x8 @196],
//     V group = 16 dims; dequant K = k * ks, V = (nib - 8) * vs[g]. Page 128 (any multiple of 16).
//   - many requests per launch (theirs: one): grid axis 2 = req * TPR + tile; decode, DSpark verify AND chunked prefill (any q_len,
//     causal per row), so a mixed batch (one chat reading a prompt while others decode) stays on one kernel.
//   - our rel layout: rel (T, Hq, E) with rel_reversed: bias of key j for a query at position p = rel[t, h, E-1-(p-j)], 0 <= p-j < E.
//   - global layers only (no window); the combine folds the sink (o *= sigmoid(lse_nat - sink)), the gate stays in the caller.
//   - any slot stride: the live cache is viewed [NB, KV_H, PAGE, 256] over slot-major memory (one head's slots 8*256 B apart).
//   - far keys (prefill, KVQ_DNNL build): keys more than rel_extent behind every query of a chunk carry no bias and are visible
//     to all of its rows, so far_attention() expands them block by block to fp16 (KvqDequant) and runs oneDNN Graph's fused SDPA
//     with softmax stats (idea and graph from Danmoreng/intel-b70-qwen38-vllm exl3_ops.sycl, MIT); PASS 1 then starts at
//     key_start[req] and the combine merges the far result as one more split.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <c10/xpu/XPUStream.h>
#include <torch/extension.h>
#include <cmath>
#include <cstdint>

namespace {

namespace xesimd = sycl::ext::intel::experimental::esimd;
using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;

constexpr float LOG2E_F = 1.4426950408889634f;
constexpr float LN2_F = 0.6931471805599453f;
constexpr float NEG = -1.0e30f;
constexpr int VOFF = 128, KSOFF = 192, VSOFF = 196;

ESIMD_INLINE float sexp2(float x) { return exp2(simd<float, 1>(x))[0]; }
ESIMD_INLINE float slog2(float x) { return log2(simd<float, 1>(x))[0]; }

// PASS 1: grid (splits, KV_H, num_reqs * TPR), local (1,1,1). Slot r of a request = query row r/3, head hk*3 + r%3 (GQA 3).
struct KvqDecode {
    const uint16_t* q;          // bf16 [T, H, 128]
    const uint8_t* kv;          // uint8 [NB, KV_H, PAGE, 256]
    const uint16_t* rel;        // bf16 [T, H, E], reversed columns
    const int32_t* block_table; // [num_reqs, max_pages]
    const int32_t* seq_lens;    // [num_reqs]
    const int32_t* cu_q;        // [num_reqs + 1]
    const int32_t* key_start;   // [num_reqs] or nullptr: first key this pass reads (multiple of 16; far keys done by oneDNN)
    float* o_part;              // [splits, T, H, 128]
    float* lse_part;            // [splits, T, H] (log2 domain)
    float softmax_scale;
    int64_t stride_q_t, stride_q_h, stride_kv_blk, stride_kv_slot, stride_kv_h, stride_rel_t, stride_rel_h, stride_bt;
    int H, E, PAGE, TPR, num_tokens, num_splits;

    void operator()(sycl::nd_item<3> it) const SYCL_ESIMD_KERNEL {
        constexpr int BK = 16;
        const int part = static_cast<int>(it.get_global_id(0));
        const int hk = static_cast<int>(it.get_global_id(1));
        const int gt = static_cast<int>(it.get_global_id(2));
        const int req = gt / TPR, tile = gt % TPR;

        const int q_start = cu_q[req];
        const int q_len = cu_q[req + 1] - q_start;
        const int R = q_len * 3;
        const int r0 = tile * 8;
        if (r0 >= R) return;
        const int seqlen_k = seq_lens[req];
        const int past = seqlen_k - q_len;
        const int tok0 = r0 / 3;
        const int tok1 = ((r0 + 7) / 3 < q_len) ? (r0 + 7) / 3 : (q_len - 1);
        const int pos_lo = past + tok0, pos_hi = past + tok1;
        const int hi = (pos_hi + 1 < seqlen_k) ? (pos_hi + 1) : seqlen_k;
        const int hi_chunk = (hi + BK - 1) / BK;
        const int lo_chunk = key_start ? key_start[req] / BK : 0;
        const int per = (hi_chunk - lo_chunk + num_splits - 1) / num_splits;
        const int first = lo_chunk + part * per;
        const int last = (first + per < hi_chunk) ? (first + per) : hi_chunk;
        const int32_t* bt = block_table + int64_t(req) * stride_bt;

        simd<fp16, 1024> Q = 0;
        simd<float, 1024> O = 0;
        simd<float, 8> m = NEG, total = 0.f;
#pragma unroll
        for (int row = 0; row < 8; ++row) {
            const int r = r0 + row;
            if (r < R) {
                const int64_t qoff = int64_t(q_start + r / 3) * stride_q_t + int64_t(hk * 3 + r % 3) * stride_q_h;
                auto q16 = block_load<uint16_t, 128>(q + qoff);
                simd<uint32_t, 128> qb = convert<uint32_t>(q16) << 16;
                simd<fp16, 128> qf = convert<fp16>(qb.template bit_cast_view<float>().read());
#pragma unroll
                for (int c = 0; c < 8; ++c)
                    Q.template select<16, 1>((c * 8 + row) * 16) = qf.template select<16, 1>(c * 16);
            }
        }

        const unsigned surf_w = 127u;                            // K row = 128 bytes
        const unsigned pitch = unsigned(stride_kv_slot) - 1u;    // bytes between consecutive slots of one head
        const uint32_t ss = static_cast<uint32_t>(stride_kv_slot);
        simd<uint32_t, 16> slot_off(0, ss);                      // byte offset of key k inside a 16-key chunk
        simd<uint32_t, 32> v_off;                                // 2 dwords (8 bytes) of V per key
#pragma unroll
        for (int k = 0; k < 16; ++k) { v_off[2 * k] = k * ss; v_off[2 * k + 1] = k * ss + 4; }

        for (int chunk = first; chunk < last; ++chunk) {
            const int index = chunk * BK;
            const uint8_t* base = kv + int64_t(bt[index / PAGE]) * stride_kv_blk + int64_t(index % PAGE) * stride_kv_slot +
                                  int64_t(hk) * stride_kv_h;
            simd<int, 16> keyv(index, 1);
            simd_mask<16> live = keyv < seqlen_k;

            // ---- QK^T: score[8 slots][16 keys], int8 K exact in fp16 ----
            simd<float, 128> score = 0;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                simd<uint32_t, 128> k2;
                auto k4 = xesimd::lsc_load_2d<uint32_t, 4, 16, 1, true, false, xesimd::cache_hint::cached,
                                              xesimd::cache_hint::cached>(reinterpret_cast<const uint32_t*>(base), surf_w,
                                                                          15, pitch, c * 4, 0);
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    simd<uint32_t, 16> four = k4.template select<16, 1>(d * 16);
                    simd<uint16_t, 16> lo = convert<uint16_t>(four);
                    simd<uint16_t, 16> hiw = convert<uint16_t>(four >> 16);
                    simd<fp16, 32> val_lo = convert<fp16>(lo.template bit_cast_view<int8_t>().read());
                    simd<fp16, 32> val_hi = convert<fp16>(hiw.template bit_cast_view<int8_t>().read());
                    k2.template select<16, 1>(d * 32) = val_lo.template bit_cast_view<uint32_t>().read();
                    k2.template select<16, 1>(d * 32 + 16) = val_hi.template bit_cast_view<uint32_t>().read();
                }
                score = xmx::dpas<8, 8, float, float, fp16, fp16>(score, k2.template bit_cast_view<fp16>().read(),
                                                                  simd<fp16, 128>(Q.template select<128, 1>(c * 128)));
            }
            simd<float, 16> ks = gather<float, 16>(reinterpret_cast<const float*>(base + KSOFF), slot_off);
            ks.merge(0.f, !live);   // stale slots past the sequence: no NaN/inf scale reaches the math
            ks *= LOG2E_F * softmax_scale;

            // ---- bias + mask + online softmax ----
            simd<float, 8> alpha;
            simd<fp16, 128> P;
#pragma unroll
            for (int row = 0; row < 8; ++row) {
                const int r = r0 + row;
                simd<float, 16> s = score.template select<16, 1>(row * 16) * ks;
                const int tok_r = r / 3;
                const int pos_r = past + tok_r;
                simd<int, 16> delta = pos_r - keyv;
                simd_mask<16> valid = (delta >= 0) & live & (r < R);
                simd_mask<16> near = valid & (delta < E);
                if (r < R && (index + BK - 1) >= (pos_r - E + 1)) {
                    const int token = q_start + tok_r;
                    const int head = hk * 3 + r % 3;
                    simd<int, 16> col = (E - 1) - delta;
                    simd<uint32_t, 16> off16 = convert<uint32_t>(max(col, simd<int, 16>(0))) * 2;
                    auto b16 = gather<uint16_t, 16>(rel + int64_t(token) * stride_rel_t + int64_t(head) * stride_rel_h,
                                                    off16, near);
                    simd<uint32_t, 16> bb = convert<uint32_t>(b16) << 16;
                    s.merge(s + bb.template bit_cast_view<float>().read() * LOG2E_F, near);
                }
                s.merge(NEG, !valid);
                const float peak = hmax<float>(s);
                const float mx = m[row] > peak ? float(m[row]) : peak;
                alpha[row] = sexp2(m[row] - mx);
                simd<float, 16> p = exp2(s - mx);
                p.merge(0.f, s < NEG * 0.5f);
                P.template select<16, 1>(row * 16) = convert<fp16>(p);
                total[row] = total[row] * alpha[row] + reduce<float>(p, std::plus<>());
                m[row] = mx;
            }

            // ---- P.V: O[8 slots][128 dims], V int4 -> fp16 VNNI per 16-dim group ----
            auto rescale = alpha.template replicate_vs_w_hs<8, 1, 16, 0>(0);
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                simd<uint32_t, 32> vw = gather<uint32_t, 32>(reinterpret_cast<const uint32_t*>(base + VOFF + c * 8), v_off);
                simd<uint8_t, 128> vb = vw.template bit_cast_view<uint8_t>().read();   // key k: bytes [k*8, k*8+8)
                simd<int16_t, 256> nib;
                nib.template select<128, 2>(0) = convert<int16_t>(vb & uint8_t(15));
                nib.template select<128, 2>(1) = convert<int16_t>(vb >> 4);
                simd<fp16, 256> vf = convert<fp16>(nib - int16_t(8));                  // [key][16 dims]
                simd<uint16_t, 16> vs16 = gather<uint16_t, 16>(reinterpret_cast<const uint16_t*>(base + VSOFF + 2 * c), slot_off);
                simd<fp16, 16> vs = vs16.template bit_cast_view<fp16>().read();
                vs.merge(fp16(0.f), !live);
                simd<fp16, 256> v2;
#pragma unroll
                for (int kp = 0; kp < 8; ++kp) {
                    v2.template select<16, 2>(kp * 32) = vf.template select<16, 1>((2 * kp) * 16) * vs[2 * kp];
                    v2.template select<16, 2>(kp * 32 + 1) = vf.template select<16, 1>((2 * kp + 1) * 16) * vs[2 * kp + 1];
                }
                simd<float, 128> acc = simd<float, 128>(O.template select<128, 1>(c * 128)) * rescale;
                O.template select<128, 1>(c * 128) = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, v2, P);
            }
        }

        // ---- epilogue: normalized partial + log2-domain lse ----
#pragma unroll
        for (int row = 0; row < 8; ++row) {
            const int r = r0 + row;
            if (r >= R) break;
            const int token = q_start + r / 3;
            const int64_t dest = (int64_t(part) * num_tokens + token) * H + (hk * 3 + r % 3);
            simd<float, 128> o;
            const float den = total[row];
#pragma unroll
            for (int c = 0; c < 8; ++c)
                o.template select<16, 1>(16 * c) = O.template select<16, 1>((c * 8 + row) * 16);
            block_store<float, 128>(o_part + dest * 128, o * (den > 0.f ? 1.f / den : 0.f));
            lse_part[dest] = den > 0.f ? (float(m[row]) + slog2(den)) : -INFINITY;
        }
    }
};

// PASS 2: grid (num_tokens, H*4); 32 dims of one (token, head): merge splits, fold the sink, write bf16.
struct KvqCombine {
    const float* o_part;
    const float* lse_part;
    const float* extra_o;     // [T, H, 128] normalized far-key partial, or nullptr
    const float* extra_lse;   // [T, H] its log2-domain lse (-inf = none)
    const uint16_t* sink;   // bf16 [T, H]
    uint16_t* out;          // bf16 [T, H, 128]
    int64_t stride_sink, stride_out_t, stride_out_h;
    int H, num_tokens, num_splits;

    void operator()(sycl::nd_item<2> it) const SYCL_ESIMD_KERNEL {
        const int tok = static_cast<int>(it.get_global_id(0));
        const int hd = static_cast<int>(it.get_global_id(1));
        const int head = hd / 4, dc = hd % 4;
        float mx = -INFINITY;
        for (int s = 0; s < num_splits; ++s) {
            const float l = lse_part[(int64_t(s) * num_tokens + tok) * H + head];
            mx = l > mx ? l : mx;
        }
        const float le = extra_lse ? extra_lse[int64_t(tok) * H + head] : -INFINITY;
        mx = le > mx ? le : mx;
        simd<float, 32> o = 0;
        float wsum = 0.f;
        if (mx > -INFINITY) {
            for (int s = 0; s < num_splits; ++s) {
                const int64_t idx = (int64_t(s) * num_tokens + tok) * H + head;
                const float l = lse_part[idx];
                if (!(l > -INFINITY)) continue;
                const float w = sexp2(l - mx);
                o += w * block_load<float, 32>(o_part + idx * 128 + dc * 32);
                wsum += w;
            }
            if (le > -INFINITY) {
                const float w = sexp2(le - mx);
                o += w * block_load<float, 32>(extra_o + (int64_t(tok) * H + head) * 128 + dc * 32);
                wsum += w;
            }
            o *= 1.f / wsum;
            const float lse_nat = (mx + slog2(wsum)) * LN2_F;
            const float sk = sycl::bit_cast<float>(static_cast<uint32_t>(sink[int64_t(tok) * stride_sink + head]) << 16);
            o *= 1.f / (1.f + sexp2((sk - lse_nat) * LOG2E_F));
        }
        simd<float, 32> ob = o;
        simd<uint32_t, 32> bits = ob.template bit_cast_view<uint32_t>().read();
        // round-to-nearest-even fp32 -> bf16
        simd<uint32_t, 32> rounded = bits + (((bits >> 16) & 1u) + 0x7FFFu);
        simd<uint16_t, 32> b16 = convert<uint16_t>(rounded >> 16);
        block_store<uint16_t, 32>(out + int64_t(tok) * stride_out_t + int64_t(head) * stride_out_h + dc * 32, b16);
    }
};

// Far keys -> fp16 head-major: grid (n_keys, KV_H); key s of the block lives in page pages[s / PAGE], slot s % PAGE.
struct KvqDequant {
    const uint8_t* kv;
    const int32_t* pages;
    fp16* ko;   // [KV_H, L, 128]
    fp16* vo;   // [KV_H, L, 128]
    int64_t stride_kv_blk, stride_kv_slot, stride_kv_h;
    int PAGE, L;

    void operator()(sycl::nd_item<2> it) const SYCL_ESIMD_KERNEL {
        const int s = static_cast<int>(it.get_global_id(0));
        const int hk = static_cast<int>(it.get_global_id(1));
        const uint8_t* slot = kv + int64_t(pages[s / PAGE]) * stride_kv_blk + int64_t(s % PAGE) * stride_kv_slot +
                              int64_t(hk) * stride_kv_h;
        simd<uint8_t, 256> b = block_load<uint8_t, 256>(slot);
        simd<uint8_t, 4> kb = b.template select<4, 1>(KSOFF);
        const float ks = kb.template bit_cast_view<float>().read()[0];
        simd<uint8_t, 128> kq = b.template select<128, 1>(0);
        simd<float, 128> kf = convert<float>(kq.template bit_cast_view<int8_t>().read());
        simd<fp16, 128> k = convert<fp16>(kf * ks);
        simd<uint8_t, 64> vb = b.template select<64, 1>(VOFF);
        simd<int16_t, 128> nib;
        nib.template select<64, 2>(0) = convert<int16_t>(vb & uint8_t(15));
        nib.template select<64, 2>(1) = convert<int16_t>(vb >> 4);
        simd<float, 128> vf = convert<float>(nib - int16_t(8));
        simd<uint8_t, 16> vsb = b.template select<16, 1>(VSOFF);
        simd<fp16, 8> vs = vsb.template bit_cast_view<fp16>().read();
        simd<float, 8> vsf = convert<float>(vs);
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const float sg = vsf[g];
            vf.template select<16, 1>(g * 16) *= sg;
        }
        const int64_t o = (int64_t(hk) * L + s) * 128;
        block_store<fp16, 128>(ko + o, k);
        block_store<fp16, 128>(vo + o, convert<fp16>(vf));
    }
};

}  // namespace

#ifdef KVQ_DNNL
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_graph_sycl.hpp>
#include <c10/xpu/XPUCachingAllocator.h>
#include <map>
#include <tuple>
#include <vector>
// oneDNN Graph fused SDPA, non-causal, GQA as [1, Hk, G, Q, D] views of head-major q/out, natural-log softmax stats.
// Graph shape and allocator hook follow Danmoreng/intel-b70-qwen38-vllm engine/direct-xpu/runtime/csrc/exl3_ops.sycl (MIT).
namespace fardnnl {
using namespace dnnl::graph;
using lt = logical_tensor;
static void* alloc_fn(size_t n, size_t, const void*, const void*) { return c10::xpu::XPUCachingAllocator::raw_alloc(n); }
static void free_fn(void* p, const void*, const void*, void*) { c10::xpu::XPUCachingAllocator::raw_delete(p); }
struct Entry { compiled_partition cp; std::vector<lt> ins, outs; };
struct Ctx {
    dnnl::engine eng;
    dnnl::stream strm;
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>, Entry> parts;
    std::map<float, at::Tensor> divisors;
};
static Ctx& ctx(sycl::queue& q) {
    static thread_local std::map<void*, Ctx> m;
    auto it = m.find((void*)&q);
    if (it == m.end()) {
        auto alloc = dnnl::graph::sycl_interop::make_allocator(alloc_fn, free_fn);
        auto eng = dnnl::graph::sycl_interop::make_engine_with_allocator(q.get_device(), q.get_context(), alloc);
        it = m.emplace((void*)&q, Ctx{eng, dnnl::sycl_interop::make_stream(eng, q), {}, {}}).first;
    }
    return it->second;
}
static Entry& build(Ctx& c, int64_t Q, int64_t L, int64_t Hq, int64_t Hk, int64_t D) {
    auto key = std::make_tuple(Q, L, Hq, Hk, D);
    auto f = c.parts.find(key);
    if (f != c.parts.end()) return f->second;
    const int64_t G = Hq / Hk;
    using dt = lt::data_type;
    size_t id = 0;
    lt query(id++, dt::f16, {1, Hk, G, Q, D}, {Hq * Q * D, G * Q * D, Q * D, D, 1});
    lt keyt(id++, dt::f16, {1, Hk, 1, L, D}, {Hk * L * D, L * D, L * D, D, 1});
    lt score(id++, dt::f32, {1, Hk, G, Q, L}, lt::layout_type::strided);
    op bmm1(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({query, keyt});
    bmm1.add_outputs({score});
    lt divisor(id++, dt::f16, lt::dims{1}, lt::layout_type::strided);
    lt scaled(id++, dt::f32, {1, Hk, G, Q, L}, lt::layout_type::strided);
    op div(id++, op::kind::Divide, "scale");
    div.add_inputs({score, divisor});
    div.add_outputs({scaled});
    lt probs(id++, dt::f16, {1, Hk, G, Q, L}, lt::layout_type::strided);
    lt stats(id++, dt::f32, {1, Hk, G, Q, 1}, {Hq * Q, G * Q, Q, 1, 1});
    op sm(id++, op::kind::SoftMax, "softmax");
    sm.set_attr<int64_t>(op::attr::axis, -1);
    sm.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    sm.add_inputs({scaled});
    sm.add_outputs({probs, stats});
    lt value(id++, dt::f16, {1, Hk, 1, L, D}, {Hk * L * D, L * D, L * D, D, 1});
    lt output(id++, dt::f16, {1, Hk, G, Q, D}, {Hq * Q * D, G * Q * D, Q * D, D, 1});
    op bmm2(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value});
    bmm2.add_outputs({output});
    graph g(dnnl::engine::kind::gpu);
    g.add_op(bmm1);
    g.add_op(div);
    g.add_op(sm);
    g.add_op(bmm2);
    g.finalize();
    auto ps = g.get_partitions();
    TORCH_CHECK(ps.size() == 1, "far sdpa: oneDNN did not fuse the pattern (", ps.size(), " partitions)");
    std::vector<lt> ins{query, keyt, divisor, value};
    std::vector<lt> outs{output, stats};
    auto cp = ps[0].compile(ins, outs, c.eng);
    return c.parts.emplace(key, Entry{cp, ins, outs}).first->second;
}
// q [Hq, Q, D] f16, k/v [Hk, L, D] f16, out [Hq, Q, D] f16, lse [Hq, Q] f32 (natural log). All contiguous.
static void sdpa(sycl::queue& q, at::Tensor qh, at::Tensor k, at::Tensor v, at::Tensor out, at::Tensor lse, double scale) {
    auto& c = ctx(q);
    const int64_t Hq = qh.size(0), Q = qh.size(1), D = qh.size(2), Hk = k.size(0), L = k.size(1);
    auto& e = build(c, Q, L, Hq, Hk, D);
    const float inv = static_cast<float>(1.0 / scale);
    auto it = c.divisors.find(inv);
    if (it == c.divisors.end()) it = c.divisors.emplace(inv, at::full({1}, inv, qh.options().dtype(at::kHalf))).first;
    std::vector<tensor> ins{tensor(e.ins[0], c.eng, qh.data_ptr()), tensor(e.ins[1], c.eng, k.data_ptr()),
                            tensor(e.ins[2], c.eng, it->second.data_ptr()), tensor(e.ins[3], c.eng, v.data_ptr())};
    std::vector<tensor> outs{tensor(e.outs[0], c.eng, out.data_ptr()), tensor(e.outs[1], c.eng, lse.data_ptr())};
    dnnl::graph::sycl_interop::execute(e.cp, c.strm, ins, outs, {});
}
}  // namespace fardnnl

// Attention of the rows of ONE request's prefill chunk over its first n_keys keys (all visible, no bias): returns the normalized
// output [Q, H, 128] f32 and its log2-domain lse [Q, H], ready to be passed to kvq_decode_xe2 as the extra split.
std::vector<torch::Tensor> far_attention(torch::Tensor q, torch::Tensor kv, torch::Tensor pages, int64_t n_keys, double scale,
                                         int64_t block) {
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && q.dim() == 3 && q.size(2) == 128, "far: q (Q, H, 128) bf16");
    TORCH_CHECK(kv.scalar_type() == at::kByte && kv.dim() == 4 && kv.size(3) == 256 && kv.stride(3) == 1 && kv.stride(2) % 16 == 0,
                "far: kv uint8 [NB, KV_H, PAGE, 256]");
    TORCH_CHECK(pages.scalar_type() == at::kInt && pages.dim() == 1 && pages.is_contiguous(), "far: pages int32 contiguous");
    const int PAGE = static_cast<int>(kv.size(2));
    const int64_t KVH = kv.size(1), H = q.size(1), Q = q.size(0);
    TORCH_CHECK(n_keys % PAGE == 0 && block % PAGE == 0 && n_keys > 0, "far: n_keys and block multiples of the page");
    auto& queue = c10::xpu::getCurrentXPUStream().queue();
    const int64_t Qp = (Q + 63) / 64 * 64;   // fewer distinct oneDNN shapes
    auto qh = at::zeros({H, Qp, 128}, q.options().dtype(at::kHalf));
    qh.narrow(1, 0, Q).copy_(q.transpose(0, 1));
    at::Tensor acc_o, acc_l;
    for (int64_t b0 = 0; b0 < n_keys; b0 += block) {
        const int L = static_cast<int>(std::min<int64_t>(block, n_keys - b0));
        auto kb = at::empty({KVH, L, 128}, qh.options());
        auto vb = at::empty({KVH, L, 128}, qh.options());
        KvqDequant dq{kv.data_ptr<uint8_t>(), pages.data_ptr<int32_t>() + b0 / PAGE,
                      reinterpret_cast<fp16*>(kb.data_ptr()), reinterpret_cast<fp16*>(vb.data_ptr()),
                      kv.stride(0), kv.stride(2), kv.stride(1), PAGE, L};
        queue.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<2>({size_t(L), size_t(KVH)}, {1, 1}), dq); });
        auto ob = at::empty({H, Qp, 128}, qh.options());
        auto lb = at::empty({H, Qp}, qh.options().dtype(at::kFloat));
        fardnnl::sdpa(queue, qh, kb, vb, ob, lb, scale);
        if (!acc_o.defined()) {
            acc_o = ob.to(at::kFloat);
            acc_l = lb;
        } else {
            auto n = at::logaddexp(acc_l, lb);
            acc_o = acc_o * (acc_l - n).exp().unsqueeze(-1) + ob.to(at::kFloat) * (lb - n).exp().unsqueeze(-1);
            acc_l = n;
        }
    }
    auto o = acc_o.narrow(1, 0, Q).transpose(0, 1).contiguous();
    auto l = (acc_l.narrow(1, 0, Q) * LOG2E_F).transpose(0, 1).contiguous();
    return {o, l};
}
#endif

// q (T,H,128) bf16; kv uint8 [NB,KV_H,PAGE,256]; rel (T,H,E) bf16 reversed; sinks (T,H) bf16; block_table int32 [R, P];
// seq_lens int32 [R]; cu_q int32 [R+1]; out (T,H,128) bf16. max_q = longest q_len in the batch (prefill chunks included).
torch::Tensor kvq_decode_xe2(torch::Tensor q, torch::Tensor kv, torch::Tensor rel, torch::Tensor sinks,
                             torch::Tensor block_table, torch::Tensor seq_lens, torch::Tensor cu_q, double scale,
                             int64_t num_reqs, int64_t max_q, int64_t num_splits, torch::Tensor out,
                             c10::optional<torch::Tensor> key_start, c10::optional<torch::Tensor> extra_o,
                             c10::optional<torch::Tensor> extra_lse) {
    const bool far = key_start.has_value() && key_start->numel() > 0;
    if (far) {
        TORCH_CHECK(key_start->scalar_type() == at::kInt && key_start->numel() == num_reqs, "kvq: key_start int32 [num_reqs]");
        TORCH_CHECK(extra_o.has_value() && extra_lse.has_value() && extra_o->is_contiguous() && extra_lse->is_contiguous() &&
                        extra_o->scalar_type() == at::kFloat && extra_lse->scalar_type() == at::kFloat,
                    "kvq: extra_o [T,H,128] / extra_lse [T,H] f32 contiguous with key_start");
    }
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && rel.scalar_type() == at::kBFloat16 && sinks.scalar_type() == at::kBFloat16 &&
                    out.scalar_type() == at::kBFloat16, "kvq decode: q/rel/sinks/out bf16");
    TORCH_CHECK(kv.scalar_type() == at::kByte && kv.dim() == 4 && kv.size(3) == 256 && kv.stride(3) == 1 &&
                    kv.stride(2) % 16 == 0 && kv.stride(1) % 4 == 0,
                "kvq decode: kv uint8 [NB, KV_H, PAGE, 256], inner-contiguous slots, slot stride a multiple of 16 bytes");
    TORCH_CHECK(q.size(2) == 128 && q.stride(2) == 1 && rel.stride(2) == 1 && out.stride(2) == 1 && sinks.stride(1) == 1 &&
                    block_table.stride(1) == 1, "kvq decode: inner-contiguous tensors");
    TORCH_CHECK(max_q >= 1 && max_q <= 65536, "kvq decode: max_q 1..65536");
    const int T = static_cast<int>(q.size(0)), H = static_cast<int>(q.size(1)), KVH = static_cast<int>(kv.size(1));
    TORCH_CHECK(H == KVH * 3, "kvq decode: GQA 3");
    const int PAGE = static_cast<int>(kv.size(2));
    TORCH_CHECK(PAGE % 16 == 0, "kvq decode: page multiple of 16");
    const int TPR = static_cast<int>((max_q * 3 + 7) / 8);
    auto opts = q.options().dtype(at::kFloat);
    auto o_part = torch::empty({num_splits, T, H, 128}, opts);
    auto lse_part = torch::full({num_splits, T, H}, -INFINITY, opts);
    auto& queue = c10::xpu::getCurrentXPUStream().queue();
    KvqDecode k1{reinterpret_cast<const uint16_t*>(q.data_ptr()), kv.data_ptr<uint8_t>(),
                 reinterpret_cast<const uint16_t*>(rel.data_ptr()), block_table.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
                 cu_q.data_ptr<int32_t>(), far ? key_start->data_ptr<int32_t>() : nullptr,
                 o_part.data_ptr<float>(), lse_part.data_ptr<float>(), static_cast<float>(scale),
                 q.stride(0), q.stride(1), kv.stride(0), kv.stride(2), kv.stride(1), rel.stride(0), rel.stride(1), block_table.stride(0),
                 H, static_cast<int>(rel.size(2)), PAGE, TPR, T, static_cast<int>(num_splits)};
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<3>({size_t(num_splits), size_t(KVH), size_t(num_reqs * TPR)}, {1, 1, 1}), k1);
    });
    KvqCombine k2{o_part.data_ptr<float>(), lse_part.data_ptr<float>(), far ? extra_o->data_ptr<float>() : nullptr,
                  far ? extra_lse->data_ptr<float>() : nullptr, reinterpret_cast<const uint16_t*>(sinks.data_ptr()),
                  reinterpret_cast<uint16_t*>(out.data_ptr()), sinks.stride(0), out.stride(0), out.stride(1), H, T,
                  static_cast<int>(num_splits)};
    queue.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<2>({size_t(T), size_t(H * 4)}, {1, 1}), k2); });
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kvq_decode_xe2", &kvq_decode_xe2, "Inkling 2 global-layer attention on the K8/V4 cache (ESIMD XMX, Xe2)",
          py::arg("q"), py::arg("kv"), py::arg("rel"), py::arg("sinks"), py::arg("block_table"), py::arg("seq_lens"), py::arg("cu_q"),
          py::arg("scale"), py::arg("num_reqs"), py::arg("max_q"), py::arg("num_splits"), py::arg("out"),
          py::arg("key_start") = py::none(), py::arg("extra_o") = py::none(), py::arg("extra_lse") = py::none());
#ifdef KVQ_DNNL
    m.def("far_attention", &far_attention, "far-key prefill attention: K8/V4 pages -> fp16 blocks -> oneDNN fused SDPA (+stats)");
    m.attr("has_far") = true;
#else
    m.attr("has_far") = false;
#endif
}
