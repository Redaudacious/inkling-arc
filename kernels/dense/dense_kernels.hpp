// SPDX-License-Identifier: MIT
// dense_kernels.hpp: W8A16 dense GEMM for Intel Xe2 (Arc Pro B70). Fable, 22 Nov 2026.
// Derived from nvf4_kernels.hpp (MIT): ESIMD XMX DPAS on fp16, weights repacked once into the DPAS B order.
//
//   y[T, Np] (bf16) = x (bf16) . Wq^T * scale      Wq int8 per row (output column), scale fp32 per row.
//
// int8 -> fp16 is exact, bf16 -> fp16 is exact in the fp16 range (clamped to +-65504; below 2^-14 flushes to zero),
// DPAS sums in fp32, the per-column scale multiplies the fp32 sum, the result is rounded to bf16 (nearest even).
// No FP8 arithmetic: weight bytes are integers converted to fp16 in registers.
//
// Pack: 1024 bytes per (16 output columns, 64 K): [16-K group][K pair][column][k parity]  (1 byte per weight, as shipped).
// Row groups ("segments"): the weight rows may be packed as up to 4 segments, each padded to 16 columns. Segment g then
// reads activation rows [g*T, (g+1)*T): one GEMM serves the stacked q / k / v inputs of qkvr and computes only the
// columns each input needs. With one segment this is a plain GEMM.
//
// Threads: one per (row tile of 8*MT rows, 16-column block, K slice). K slices of one block form one work-group and are
// summed through shared local memory after a barrier, so a GEMM is ONE launch: no zero pass, no atomics, no fp32 buffer.
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cstdint>
#include <cstdlib>

namespace dense {
namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using esimd::simd;
using half = sycl::half;

constexpr int MAX_SPLIT = 16;   // K slices per column block (SLM = 16 * 8 * MT * 64 bytes)

// MT: 8*MT rows per tile (1, 2, 4). RED: false = one thread per block (direct store), true = K-split, SLM reduction.
// PRE: activations already fp16 (converted by to_fp16), else bf16 converted in registers. BIG: 256 GRFs.
template <int MT, bool RED, bool PRE, bool BIG> class gemm_k;
class pack_k;
class fp16_k;

struct Geo {
  int T;            // rows per segment
  int Np, K;        // padded columns, inner size
  int b1, b2, b3;   // first column block of segments 1..3 (>= Np/16 when unused)
  int split;        // K slices (1..MAX_SPLIT)
};

template <int MT, bool RED, bool PRE, bool BIG>
sycl::event launch(sycl::queue &q, const void *Xv, const uint8_t *P, const float *S, uint16_t *OUT, Geo g) {
  const int NB = g.Np / 16, steps = g.K / 64, K = g.K, Np = g.Np, T = g.T;
  const int split = RED ? g.split : 1;
  const int csteps = (steps + split - 1) / split;
  const int RT = (T + 8 * MT - 1) / (8 * MT);
  const int cells = RT * NB;
  const int b1 = g.b1, b2 = g.b2, b3 = g.b3;
  const int LWG = RED ? split : 16;
  const size_t gsz = RED ? (size_t)cells * split : (size_t)((cells + 15) / 16) * 16;
  const uint16_t *X16 = reinterpret_cast<const uint16_t *>(Xv);
  const half *XH = reinterpret_cast<const half *>(Xv);
  auto body = [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    if constexpr (RED) esimd::slm_init<MAX_SPLIT * 8 * MT * 64>();
    const int t = (int)it.get_global_id(0);
    int cell, l;
    if constexpr (RED) { cell = t / split; l = t - cell * split; } else { cell = t; l = 0; if (cell >= cells) return; }
    const int rt = cell / NB, nb = cell - rt * NB;
    const int seg = (nb >= b1 ? 1 : 0) + (nb >= b2 ? 1 : 0) + (nb >= b3 ? 1 : 0);
    const int orow0 = rt * 8 * MT;
    const int m = (T - orow0) < 8 * MT ? (T - orow0) : 8 * MT;
    const size_t arow = (size_t)seg * T + orow0;
    const int ks0 = l * csteps;
    const int nst = (steps - ks0) < csteps ? (steps - ks0) : csteps;
    const uint8_t *p = P + ((size_t)nb * steps + ks0) * 1024;
    simd<float, 128 * MT> C = 0.f;
    for (int st = 0; st < nst; ++st) {
      const int k = (ks0 + st) * 64;
#pragma unroll
      for (int sp = 0; sp < 2; ++sp) {
        simd<half, MT * 256> ab;
        if constexpr (PRE) {
          ab = esimd::load_2d<half, 16, MT * 8, 2>(XH + arow * K, (unsigned)(K * 2 - 1), (unsigned)(m - 1), (unsigned)(K * 2 - 1),
                                                   k + 32 * sp, 0);
        } else {
          simd<uint16_t, MT * 256> au = esimd::load_2d<uint16_t, 16, MT * 8, 2>(
              X16 + arow * K, (unsigned)(K * 2 - 1), (unsigned)(m - 1), (unsigned)(K * 2 - 1), k + 32 * sp, 0);
#pragma unroll
          for (int c = 0; c < MT * 2; ++c) {
            simd<uint32_t, 128> u = esimd::convert<uint32_t>(au.template select<128, 1>(c * 128).read());
            u = u << 16u;
            simd<float, 128> f = u.template bit_cast_view<float>().read();
            f = esimd::min(esimd::max(f, simd<float, 128>(-65504.f)), simd<float, 128>(65504.f));
            ab.template select<128, 1>(c * 128) = esimd::convert<half>(f);
          }
        }
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
          const int s = 2 * sp + blk;   // 16-K group inside the 64-K step
          simd<int8_t, 256> wb =
              esimd::block_load<int8_t, 256>(reinterpret_cast<const int8_t *>(p + 256 * s), esimd::overaligned_tag<16>{});
          simd<int16_t, 256> wi = esimd::convert<int16_t>(wb);
          simd<half, 256> b = esimd::convert<half>(wi);
#pragma unroll
          for (int tt = 0; tt < MT; ++tt) {
            simd<half, 128> a = ab.template select<128, 1>(blk * (MT * 128) + tt * 128);
            simd<float, 128> c = C.template select<128, 1>(tt * 128);
            C.template select<128, 1>(tt * 128) = xmx::dpas<8, 8, float>(c, b, a);
          }
        }
      }
      p += 1024;
    }
    const simd<float, 16> sc = esimd::block_load<float, 16>(S + (size_t)nb * 16, esimd::overaligned_tag<16>{});
    if constexpr (RED) {
#pragma unroll
      for (int r = 0; r < MT * 8; ++r)
        esimd::slm_block_store<float, 16>((uint32_t)((l * 8 * MT + r) * 64), C.template select<16, 1>(r * 16).read());
      esimd::barrier();
    }
#pragma unroll
    for (int r = 0; r < MT * 8; ++r) {
      bool mine = r < m;
      if constexpr (RED) mine = mine && (r % split) == l;
      if (mine) {
        simd<float, 16> v;
        if constexpr (RED) {
          v = 0.f;
          for (int j = 0; j < split; ++j) v += esimd::slm_block_load<float, 16>((uint32_t)((j * 8 * MT + r) * 64));
        } else {
          v = C.template select<16, 1>(r * 16);
        }
        v = v * sc;
        simd<uint32_t, 16> bb = v.template bit_cast_view<uint32_t>().read();
        bb += ((bb >> 16u) & 1u) + 0x7FFFu;
        simd<uint16_t, 16> o = esimd::convert<uint16_t>(bb >> 16u);
        esimd::block_store<uint16_t, 16>(OUT + (size_t)(orow0 + r) * Np + nb * 16, o, esimd::overaligned_tag<16>{});
      }
    }
  };
  return q.submit([&](sycl::handler &h) {
    if constexpr (BIG) {
      sycl::ext::oneapi::experimental::properties props{sycl::ext::intel::experimental::grf_size<256>};
      h.parallel_for<gemm_k<MT, RED, PRE, BIG>>(sycl::nd_range<1>(gsz, LWG), props, body);
    } else {
      h.parallel_for<gemm_k<MT, RED, PRE, BIG>>(sycl::nd_range<1>(gsz, LWG), body);
    }
  });
}

inline int env_int(const char *n, int d) { const char *e = std::getenv(n); return e ? std::atoi(e) : d; }

// Tuned on the B70 (results/dense_sweep1.jsonl, DENSE.html): about 1100 threads keep the memory system busy; more K slices
// only add thread starts. 8-row tiles up to 24 rows (more tiles, not wider ones), 16-row tiles up to 64, 32-row beyond.
inline int auto_mt(int T) { return T <= 24 ? 1 : (T <= 64 ? 2 : 4); }
inline int auto_pre(int T) { return T >= 32 ? 1 : 0; }   // many rows: convert bf16 -> fp16 once instead of in every thread
inline int auto_split(int T, int mt, int Np, int K) {
  static const int target = env_int("DENSE_THREADS", 1100);
  const int cells = ((T + 8 * mt - 1) / (8 * mt)) * (Np / 16), steps = K / 64;
  int s = target / cells;
  if (s > MAX_SPLIT) s = MAX_SPLIT;
  if (s > steps) s = steps;
  if (s < 1) s = 1;
  const int cs = (steps + s - 1) / s;
  return (steps + cs - 1) / cs;   // drop empty slices
}

// x: bf16 bits [G*T, K] (pre == 0) or fp16 (pre == 1). mt 0 / split 0 = automatic. big: -1 automatic.
inline sycl::event gemm(sycl::queue &q, const void *x, const uint8_t *P, const float *S, uint16_t *OUT, int T, int Np, int K, int b1,
                        int b2, int b3, int split, int mt, int pre, int big) {
  if (mt <= 0) mt = auto_mt(T);
  if (split <= 0) split = auto_split(T, mt, Np, K);
  if (split > MAX_SPLIT) split = MAX_SPLIT;
  if (big < 0) big = 0;
  if (mt == 4) big = 1;
  Geo g{T, Np, K, b1, b2, b3, split};
  const bool red = split > 1;
#define DENSE_ONE(M, R, PR, B) return launch<M, R, PR, B>(q, x, P, S, OUT, g);
#define DENSE_B(M, R, PR) if (big) { DENSE_ONE(M, R, PR, true) } else { DENSE_ONE(M, R, PR, false) }
#define DENSE_P(M, R) if (pre) { DENSE_B(M, R, true) } else { DENSE_B(M, R, false) }
#define DENSE_R(M) if (red) { DENSE_P(M, true) } else { DENSE_P(M, false) }
  if (mt == 1) { DENSE_R(1) } else if (mt == 2) { DENSE_R(2) }
  if (red) { if (pre) { DENSE_ONE(4, true, true, true) } else { DENSE_ONE(4, true, false, true) } }
  if (pre) { DENSE_ONE(4, false, true, true) }
  DENSE_ONE(4, false, false, true)
#undef DENSE_R
#undef DENSE_P
#undef DENSE_B
#undef DENSE_ONE
}

// bf16 bits [n] -> fp16 [n] (only for the PRE variant)
inline sycl::event to_fp16(sycl::queue &q, const uint16_t *X, half *O, size_t n) {
  const size_t wg = 64;
  return q.parallel_for<fp16_k>(sycl::nd_range<1>((n + wg - 1) / wg * wg, wg), [=](sycl::nd_item<1> it) {
    const size_t i = it.get_global_id(0);
    if (i >= n) return;
    float v = sycl::bit_cast<float>((uint32_t)X[i] << 16);
    v = v > 65504.f ? 65504.f : (v < -65504.f ? -65504.f : v);
    O[i] = half(v);
  });
}

// padded layout of up to 4 segments: columns of segment g start at off[g]; returns Np
struct Segs { int n; int len[4]; int src[4]; int off[4]; int Np; };
inline Segs make_segs(const int64_t *segs, int nseg, int N) {
  Segs s{};
  if (nseg <= 0) { s.n = 1; s.len[0] = N; } else { s.n = nseg; for (int i = 0; i < nseg; ++i) s.len[i] = (int)segs[i]; }
  int src = 0, off = 0;
  for (int i = 0; i < s.n; ++i) { s.src[i] = src; s.off[i] = off; src += s.len[i]; off += (s.len[i] + 15) / 16 * 16; }
  s.Np = off;
  return s;
}

// One-time: W int8 [N, K] row-major + scale fp32 [N] -> pack [Np/16 * K/64 * 1024] and scale_p fp32 [Np] (padding = 0).
inline sycl::event pack(sycl::queue &q, const int8_t *W, const float *SC, uint8_t *O, float *SP, int K, Segs sg) {
  const int steps = K / 64, Np = sg.Np;
  const size_t tot = (size_t)Np * steps, wg = 64;
  return q.parallel_for<pack_k>(sycl::nd_range<1>((tot + wg - 1) / wg * wg, wg), [=](sycl::nd_item<1> it) {
    const size_t gid = it.get_global_id(0);
    if (gid >= tot) return;
    const int np = (int)(gid / steps), st = (int)(gid % steps), c = np % 16;
    int n = -1;
    for (int i = 0; i < sg.n; ++i)
      if (np >= sg.off[i] && np < sg.off[i] + sg.len[i]) n = sg.src[i] + (np - sg.off[i]);
    uint8_t *rec = O + ((size_t)(np / 16) * steps + st) * 1024;
    if (st == 0) SP[np] = n >= 0 ? SC[n] : 0.f;
    const int8_t *wb = W + (size_t)(n >= 0 ? n : 0) * K + st * 64;
    for (int kq = 0; kq < 64; ++kq)
      rec[(kq / 16) * 256 + ((kq % 16) / 2) * 32 + c * 2 + (kq & 1)] = n >= 0 ? (uint8_t)wb[kq] : (uint8_t)0;
  });
}
}  // namespace dense
