// SPDX-License-Identifier: MIT
// nvf4_kernels.hpp: NVFP4 and MXFP8 routed-expert grouped GEMM for Intel Xe2 (Arc Pro B70), no FP4/FP8 hardware.
// Fable, 22 Nov 2026. Derived from libq4k_dpas (q4k_dpas.cpp, MIT): ESIMD XMX DPAS on fp16, weights repacked once into the
// DPAS tile order. One source for the C ABI library (nvf4c.cpp, Rust tools) and the torch operator (nvf4_op.cpp, server).
//
// NVFP4 weight = E2M1(nibble) * E4M3(scale per 16 K) * factor(expert). Nothing is requantized.
//   pack record per (expert, 16-column block nb, 64-K step):
//     [0,512)   uint32 words[8][16]: word wi holds K = 8wi..8wi+7 of column c; low half-word slot j = nibble of K 2j, high
//               half-word slot j = nibble of K 2j+1 (one shift+mask gives a VNNI pair row of the DPAS B operand).
//     [512,..)  scales of the four 16-K groups x 16 columns: S8 = byte code c (576-byte record, 4.5 bit/weight),
//               S16 = fp16 (640-byte record, 5 bit/weight).
//   scale code c: fp16 bits = (c << 7) + 0x1800 = (1 + (c&7)/8) * 2^((c>>3) - 9); every positive E4M3 value has one code;
//   a negative scale flips the sign bits of its 16 nibbles, a zero scale clears them. Integer code, no FP8 arithmetic.
//   nibble -> fp16: bits = 0x3800 + (mag << 9) for mag >= 2, 0.5 for mag 1, 0 for mag 0, sign to bit 15. (The shorter
//   "E2M1 is an fp16 subnormal" decode is wrong on this GPU: the EU flushes fp16 subnormal operands. Measured 22 Nov.)
// MXFP8 weight = E4M3(byte) * 2^(s - 127), one s per 32 K, kept at 8 bits:
//   record = 1056 bytes: [0,1024) E4M3 bytes in DPAS B order [16-K group][K pair][column][k parity];
//   [1024,1056) scale code per 32-K group x 16 columns, c = 8 * (s - smin_expert) + 72; 2^(smin - 127) is the expert factor.
//   byte -> fp16: bits = ((byte & 0x7F) << 7) + 0x2000, E4M3 subnormals patched from an int -> fp16 convert, sign to bit 15.
// b = value * scale is exact in fp16. DPAS accumulates in fp32; the expert factor multiplies the fp32 sum.
// A job = (expert, first row, rows). Rows <= 8 * MT per job; a job with rows == 0 is skipped (GPU-built job lists).
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cstdint>
#include <cstdlib>

namespace nvf4 {
namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using esimd::simd;
using half = sycl::half;

// FMT: 0 = NVFP4 with byte scale codes, 1 = NVFP4 with fp16 scales, 2 = MXFP8.  OUT: 0 = fp16 store (split must be 1),
// 1 = fp32 store, 2 = fp32 atomic add (split > 1, destination zeroed by the caller).
template <int FMT, int MT, int OUT, bool BIG> class gemm_k;
class pack_k;
class unpack_k;
class mxpack_k;
class mxunpack_k;

template <int FMT, int MT, int OUT, bool BIG>
sycl::event launch(sycl::queue &q, const half *A, const uint8_t *P, const float *F, const int32_t *J, void *Cout, int NJ, int N, int K,
                   int split) {
  constexpr int REC = FMT == 0 ? 576 : (FMT == 1 ? 640 : 1056);
  const int NB = N / 16, steps = K / 64, csteps = steps / split;
  const int per_job = NB * split;
  const int threads = NJ * per_job;
  const size_t exp_bytes = (size_t)NB * steps * REC;
  constexpr int LWG = 16;
  const size_t gsz = (size_t)((threads + LWG - 1) / LWG) * LWG;
  half *C16 = reinterpret_cast<half *>(Cout);
  float *C32 = reinterpret_cast<float *>(Cout);
  auto body = [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    const int t = (int)it.get_global_id(0);
    if (t >= threads) return;
    const int job = t / per_job, rem = t - job * per_job;
    const int nb = rem % NB, ks = rem / NB;
    const int e = J[job * 4], row0 = J[job * 4 + 1], m = J[job * 4 + 2];
    if (m <= 0) return;
    const uint8_t *p = P + (size_t)e * exp_bytes + ((size_t)nb * steps + (size_t)ks * csteps) * REC;
    const half *Ab = A + (size_t)row0 * K;
    simd<float, 128 * MT> C = 0.f;
    for (int st = 0; st < csteps; ++st) {
      const int k = (ks * csteps + st) * 64;
      simd<half, 64> sh;   // fp16 scale per 16-K group x 16 columns (MXFP8: per 32-K group, groups 0..1)
      simd<uint32_t, 128> bw;
      if constexpr (FMT == 2) {
        simd<uint8_t, 32> sb = esimd::block_load<uint8_t, 32>(p + 1024, esimd::overaligned_tag<16>{});
        simd<uint16_t, 32> su = esimd::convert<uint16_t>(sb);
        su = (su << (uint16_t)7) + (uint16_t)0x1800;
        sh.template select<32, 1>(0) = su.template bit_cast_view<half>().read();
      } else {
        bw = esimd::block_load<uint32_t, 128>(reinterpret_cast<const uint32_t *>(p), esimd::overaligned_tag<16>{});
        if constexpr (FMT == 0) {
          simd<uint8_t, 64> sb = esimd::block_load<uint8_t, 64>(p + 512, esimd::overaligned_tag<16>{});
          simd<uint16_t, 64> su = esimd::convert<uint16_t>(sb);
          su = (su << (uint16_t)7) + (uint16_t)0x1800;
          sh = su.template bit_cast_view<half>().read();
        } else {
          sh = esimd::block_load<half, 64>(reinterpret_cast<const half *>(p + 512), esimd::overaligned_tag<16>{});
        }
      }
#pragma unroll
      for (int sp = 0; sp < 2; ++sp) {
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
          const int s = 2 * sp + blk;   // 16-K group inside the 64-K step
          // activations of this 16-K block: rows 0..8MT-1 (rows >= m come back as zero); 2D loads stop at 32 rows
          constexpr int R0 = MT * 8 < 32 ? MT * 8 : 32, R1 = MT * 8 - R0;
          simd<half, MT * 8 * 16> ab;
          ab.template select<R0 * 16, 1>(0) = esimd::load_2d<half, 16, R0, 1>(Ab, (unsigned)(K * 2 - 1), (unsigned)(m - 1),
                                                                             (unsigned)(K * 2 - 1), k + 16 * s, 0);
          if constexpr (R1 > 0)
            ab.template select<(R1 > 0 ? R1 : 8) * 16, 1>(R0 * 16) =
                esimd::load_2d<half, 16, (R1 > 0 ? R1 : 8), 1>(Ab, (unsigned)(K * 2 - 1), (unsigned)(m - 1), (unsigned)(K * 2 - 1),
                                                              k + 16 * s, R0);
          simd<half, 32> sc32;
          {
            simd<half, 16> s16 = sh.template select<16, 1>(16 * (FMT == 2 ? sp : s));
            sc32.template select<16, 2>(0) = s16;
            sc32.template select<16, 2>(1) = s16;
          }
          simd<uint16_t, 256> bits;
          if constexpr (FMT == 2) {
            simd<uint8_t, 256> wb = esimd::block_load<uint8_t, 256>(p + 256 * s, esimd::overaligned_tag<16>{});
            simd<uint16_t, 256> x = esimd::convert<uint16_t>(wb);
            simd<uint16_t, 256> v7 = x & (uint16_t)0x7F;
            bits = (v7 << (uint16_t)7) + (uint16_t)0x2000;
            simd<uint16_t, 256> mt = x & (uint16_t)7;
            simd<half, 256> sub = esimd::convert<half>(mt) * half(0.001953125f);   // m * 2^-9
            bits.merge(sub.template bit_cast_view<uint16_t>().read(), v7 < (uint16_t)8);
            bits |= (x & (uint16_t)0x80) << (uint16_t)8;
          } else {
            simd<uint32_t, 128> b32;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
              const int kk = 16 * s + 2 * r;
              const int widx = kk / 8, j = (kk % 8) / 2;
              simd<uint32_t, 16> w = bw.template select<16, 1>(widx * 16);
              b32.template select<16, 1>(r * 16) = (w >> (uint32_t)(j * 4)) & 0x000F000Fu;
            }
            simd<uint16_t, 256> n = b32.template bit_cast_view<uint16_t>().read();
            simd<uint16_t, 256> mag = n & (uint16_t)7;
            bits = (mag << (uint16_t)9) + (uint16_t)0x3800;
            bits.merge(simd<uint16_t, 256>((uint16_t)0x3800), mag == (uint16_t)1);
            bits.merge(simd<uint16_t, 256>((uint16_t)0), mag == (uint16_t)0);
            bits |= (n & (uint16_t)8) << (uint16_t)12;
          }
          simd<half, 256> b = bits.template bit_cast_view<half>().read();
#pragma unroll
          for (int r = 0; r < 8; ++r)
            b.template select<32, 1>(r * 32) = b.template select<32, 1>(r * 32) * sc32;
#pragma unroll
          for (int tt = 0; tt < MT; ++tt) {
            simd<half, 128> a = ab.template select<128, 1>(tt * 128);
            simd<float, 128> c = C.template select<128, 1>(tt * 128);
            C.template select<128, 1>(tt * 128) = xmx::dpas<8, 8, float>(c, b, a);
          }
        }
      }
      p += REC;
    }
    const float fac = F[e];
#pragma unroll
    for (int r = 0; r < MT * 8; ++r) {
      if (r < m) {
        simd<float, 16> v = C.template select<16, 1>(r * 16);
        v = v * fac;
        const size_t o = (size_t)(row0 + r) * N + nb * 16;
        if constexpr (OUT == 0) {
          esimd::block_store<half, 16>(C16 + o, simd<half, 16>(v), esimd::overaligned_tag<16>{});
        } else if constexpr (OUT == 1) {
          esimd::block_store<float, 16>(C32 + o, v, esimd::overaligned_tag<16>{});
        } else {
          simd<uint32_t, 16> off(0u, 4u);
          off += (uint32_t)(o * 4);
          esimd::atomic_update<esimd::atomic_op::fadd, float, 16>(C32, off, v);
        }
      }
    }
  };
  return q.submit([&](sycl::handler &h) {
    if constexpr (BIG) {
      sycl::ext::oneapi::experimental::properties props{sycl::ext::intel::experimental::grf_size<256>};
      h.parallel_for<gemm_k<FMT, MT, OUT, BIG>>(sycl::nd_range<1>(gsz, LWG), props, body);
    } else {
      h.parallel_for<gemm_k<FMT, MT, OUT, BIG>>(sycl::nd_range<1>(gsz, LWG), body);
    }
  });
}

// fmt 0/1/2 as above; mt 1 or 4 (rows per job <= 8 * mt); out 0/1/2 as above (out 0 and 1 need split == 1).
inline sycl::event gemm(sycl::queue &q, const half *A, const uint8_t *P, const float *F, const int32_t *J, void *C, int NJ, int N,
                        int K, int split, int fmt, int mt, int out) {
  static const bool big = [] { const char *e = std::getenv("NVF4_BIG"); return e ? std::atoi(e) != 0 : false; }();
#define NVF4_ONE(FM, M, O, B) return launch<FM, M, O, B>(q, A, P, F, J, C, NJ, N, K, split);
#define NVF4_OUT(FM, M, B)                           \
  if (out == 0) { NVF4_ONE(FM, M, 0, B) }            \
  else if (out == 1) { NVF4_ONE(FM, M, 1, B) }       \
  else { NVF4_ONE(FM, M, 2, B) }
#define NVF4_MT(FM)                                  \
  if (mt == 1) { if (big) { NVF4_OUT(FM, 1, true) } else { NVF4_OUT(FM, 1, false) } }                         \
  else if (mt == 4) { NVF4_OUT(FM, 4, true) } else { NVF4_OUT(FM, 6, true) }
  if (fmt == 0) { NVF4_MT(0) } else if (fmt == 1) { NVF4_MT(1) } else { NVF4_MT(2) }
#undef NVF4_MT
#undef NVF4_OUT
#undef NVF4_ONE
}

// One-time lossless converter. W uint8 [E, N, K/2] (two E2M1 codes per byte, low nibble = even K), S uint8 [E, N, K/16]
// (E4M3 bytes) -> O uint8 [E * N/16 * K/64 * REC]. One work-item per (expert, row, 64-K step).
inline sycl::event pack(sycl::queue &q, const uint8_t *W, const uint8_t *S, uint8_t *O, int E, int N, int K, int s8) {
  const int REC = s8 ? 576 : 640, NB = N / 16, steps = K / 64;
  const bool S8 = s8 != 0;
  const size_t tot = (size_t)E * N * steps, wg = 64;
  return q.parallel_for<pack_k>(sycl::nd_range<1>((tot + wg - 1) / wg * wg, wg), [=](sycl::nd_item<1> it) {
    const size_t gid = it.get_global_id(0);
    if (gid >= tot) return;
    const size_t rowid = gid / steps;
    const int st = (int)(gid % steps), e = (int)(rowid / N), n = (int)(rowid % N), c = n % 16;
    const uint8_t *wb = W + rowid * (size_t)(K / 2) + st * 32;
    const uint8_t *sb = S + rowid * (size_t)(K / 16) + st * 4;
    uint8_t *rec = O + ((size_t)e * NB * steps + (size_t)(n / 16) * steps + st) * REC;
    uint32_t flip[4], keep[4];
    for (int g = 0; g < 4; ++g) {
      const uint32_t v = sb[g], v7 = v & 0x7Fu, ex = v7 >> 3, mt = v7 & 7u;
      uint32_t code;
      if (ex) code = v7 + 16;                       // (1 + m/8) * 2^(e-7)
      else if (mt >= 4) code = 16 + 2 * (mt - 4);   // m * 2^-9 = (1 + (m-4)/4) * 2^-7
      else if (mt >= 2) code = 8 + 4 * (mt - 2);    // 2^-8, 1.5 * 2^-8
      else code = 0;                                // 2^-9 (and zero, with the nibbles cleared)
      flip[g] = (v >> 7) ? 8u : 0u;
      keep[g] = v7 ? 0xFu : 0u;
      if (S8) rec[512 + g * 16 + c] = (uint8_t)code;
      else *reinterpret_cast<uint16_t *>(rec + 512 + (g * 16 + c) * 2) = (uint16_t)((code << 7) + 0x1800u);
    }
    for (int wi = 0; wi < 8; ++wi) {
      const int g = wi / 2;
      uint32_t o = 0;
      for (int i = 0; i < 8; ++i) {
        const int kq = 8 * wi + i;
        uint32_t nib = (kq & 1) ? (uint32_t)(wb[kq / 2] >> 4) : (uint32_t)(wb[kq / 2] & 0xF);
        nib = (nib ^ flip[g]) & keep[g];
        o |= nib << ((i & 1) ? 16 + 4 * (i / 2) : 4 * (i / 2));
      }
      *reinterpret_cast<uint32_t *>(rec + (wi * 16 + c) * 4) = o;
    }
  });
}

// Independent scalar decode of one expert of an NVFP4 pack -> fp32 [N, K] = E2M1 value * scale (no expert factor).
inline sycl::event unpack(sycl::queue &q, const uint8_t *P0, int e, int N, int K, int s8, float *O) {
  const int REC = s8 ? 576 : 640, NB = N / 16, steps = K / 64;
  const uint8_t *P = P0 + (size_t)e * NB * steps * REC;
  const bool S8 = s8 != 0;
  const size_t tot = (size_t)N * (K / 8), wg = 64;
  return q.parallel_for<unpack_k>(sycl::nd_range<1>((tot + wg - 1) / wg * wg, wg), [=](sycl::nd_item<1> it) {
    const size_t gid = it.get_global_id(0);
    if (gid >= tot) return;
    const int row = (int)(gid / (K / 8)), kw = (int)(gid % (K / 8));
    const int st = kw / 8, wi = kw % 8, g = wi / 2, c = row % 16;
    const uint8_t *rec = P + ((size_t)(row / 16) * steps + st) * REC;
    const uint32_t w = *reinterpret_cast<const uint32_t *>(rec + (wi * 16 + c) * 4);
    uint32_t hb;
    if (S8) hb = ((uint32_t)rec[512 + g * 16 + c] << 7) + 0x1800u;
    else hb = *reinterpret_cast<const uint16_t *>(rec + 512 + (g * 16 + c) * 2);
    const int ex = (int)(hb >> 10) - 15;
    float sc = 1.f + (float)(hb & 0x3FFu) / 1024.f;
    for (int i = 0; i < ex; ++i) sc *= 2.f;
    for (int i = 0; i < -ex; ++i) sc *= 0.5f;
    for (int i = 0; i < 8; ++i) {
      const uint32_t nib = (w >> ((i & 1) ? 16 + 4 * (i / 2) : 4 * (i / 2))) & 0xFu;
      const uint32_t mg = nib & 7u, x = mg >> 1, mt = mg & 1u;
      float v = x ? (1.f + 0.5f * (float)mt) * (float)(1u << (x - 1)) : 0.5f * (float)mt;
      if (nib & 8u) v = -v;
      O[(size_t)row * K + kw * 8 + i] = v * sc;
    }
  });
}

// MXFP8: W uint8 [E, N, K] (E4M3 bytes), S uint8 [E, N, K/32] (E8M0), SM int32 [E] (per-expert minimum of s) -> pack.
inline sycl::event mx_pack(sycl::queue &q, const uint8_t *W, const uint8_t *S, const int32_t *SM, uint8_t *O, int E, int N, int K) {
  const int REC = 1056, NB = N / 16, steps = K / 64;
  const size_t tot = (size_t)E * N * steps, wg = 64;
  return q.parallel_for<mxpack_k>(sycl::nd_range<1>((tot + wg - 1) / wg * wg, wg), [=](sycl::nd_item<1> it) {
    const size_t gid = it.get_global_id(0);
    if (gid >= tot) return;
    const size_t rowid = gid / steps;
    const int st = (int)(gid % steps), e = (int)(rowid / N), n = (int)(rowid % N), c = n % 16;
    const uint8_t *wb = W + rowid * (size_t)K + st * 64;
    const uint8_t *sb = S + rowid * (size_t)(K / 32) + st * 2;
    uint8_t *rec = O + ((size_t)e * NB * steps + (size_t)(n / 16) * steps + st) * REC;
    for (int g = 0; g < 2; ++g) rec[1024 + g * 16 + c] = (uint8_t)(8 * ((int)sb[g] - SM[e]) + 72);
    for (int kq = 0; kq < 64; ++kq) rec[(kq / 16) * 256 + ((kq % 16) / 2) * 32 + c * 2 + (kq & 1)] = wb[kq];
  });
}

// scalar decode of one expert of an MXFP8 pack -> fp32 [N, K] = E4M3 value * 2^(s - smin).
inline sycl::event mx_unpack(sycl::queue &q, const uint8_t *P0, int e, int N, int K, float *O) {
  const int REC = 1056, NB = N / 16, steps = K / 64;
  const uint8_t *P = P0 + (size_t)e * NB * steps * REC;
  const size_t tot = (size_t)N * K, wg = 64;
  return q.parallel_for<mxunpack_k>(sycl::nd_range<1>((tot + wg - 1) / wg * wg, wg), [=](sycl::nd_item<1> it) {
    const size_t gid = it.get_global_id(0);
    if (gid >= tot) return;
    const int row = (int)(gid / K), kk = (int)(gid % K), st = kk / 64, kq = kk % 64, c = row % 16;
    const uint8_t *rec = P + ((size_t)(row / 16) * steps + st) * REC;
    const uint32_t b = rec[(kq / 16) * 256 + ((kq % 16) / 2) * 32 + c * 2 + (kq & 1)];
    const int pw = ((int)rec[1024 + (kq / 32) * 16 + c] - 72) / 8;
    const uint32_t ex = (b >> 3) & 15u, mt = b & 7u;
    float v = ex ? (1.f + (float)mt / 8.f) : (float)mt / 8.f;
    const int sh = ex ? (int)ex - 7 : -6;
    for (int i = 0; i < sh + pw; ++i) v *= 2.f;
    for (int i = 0; i < -(sh + pw); ++i) v *= 0.5f;
    O[gid] = (b & 0x80u) ? -v : v;
  });
}

// ------------------------------------------------------------------------------------------------------------------
// One whole routed-expert layer, everything on the queue (no host readback, graph-capturable):
//   (weights, ids) from the router -> counting sort + job list (one work-item) -> fp16 activations grouped by expert ->
//   w13 GEMM (fp32 sums) -> soft-capped SiTU -> w2 GEMM (fp32 sums) -> weighted gather, rounded to bf16.
// X bf16 bits [T, H]; WT fp32 [T, k]; ID int32 [T, k]; P13 / P2 packs; F13 / F2 fp32 [E]; OUT bf16 bits [T, H].
// Scratch (device memory, 64-byte aligned): unp int32 [T*k], jobs int32 [moe_jobs_ints], a16 fp16 [T*k, H],
// c1 fp32 [T*k, 2I], act fp16 [T*k, I], c2 fp32 [T*k, H]. fmt 0 = NVFP4 (byte scale codes), 2 = MXFP8.
class moe_jobs_k;
class moe_prefix_k;
class moe_count_k;
class moe_jobs1_k;
class moe_rows_k;
class moe_situ_k;
class moe_gather_k;
template <int WG> class moe_route1_k;
class zero_k;
// zero a 32-bit buffer with an nd_range launch (torch's bundled libsycl lacks the helper that queue::fill and plain range
// launches call)
inline sycl::event zero32(sycl::queue &q, void *p, size_t n) {
  uint32_t *P = reinterpret_cast<uint32_t *>(p);
  const size_t wg = 64;
  return q.parallel_for<zero_k>(sycl::nd_range<1>((n + wg - 1) / wg * wg, wg), [=](sycl::nd_item<1> it) {
    const size_t i = it.get_global_id(0);
    if (i < n) P[i] = 0u;
  });
}
struct MoeScratch { int32_t *unp; int32_t *jobs; half *a16; float *c1; half *act; float *c2; };

// 32-row tiles when experts average more than 4 rows; 48-row tiles above 24 rows per expert (prompt reading)
inline int moe_mt(int rows, int E) {
  static const int cap = [] { const char *e = std::getenv("NVF4_MT_MAX"); return e ? std::atoi(e) : 6; }();
  return rows <= 4 * E ? 1 : (rows <= 24 * E || cap < 6 ? 4 : 6);
}
inline int moe_max_jobs(int rows, int E) { return (E < rows ? E : rows) + rows / (8 * moe_mt(rows, E)); }
// NVF4_ROUTE_SPLIT=1: the four-kernel job list (zero, count, prefix, jobs) for every size (timing aid / fallback)
inline bool route_split() {
  static const bool v = [] { const char *e = std::getenv("NVF4_ROUTE_SPLIT"); return e && std::atoi(e) != 0; }();
  return v;
}
inline long moe_thread_target() {
  static const long v = [] { const char *e = std::getenv("NVF4_THREADS"); return e ? std::atol(e) : 12000L; }();
  return v;
}
// job list + CNT, POS (zeroed per call) + START, JB (written by moe_prefix_k)
inline int moe_jobs_ints(int rows, int E) { return 4 * moe_max_jobs(rows, E) + 1024; }
inline int moe_split(int jobs, int nb, int steps) {
  int best = 1;
  for (int s = 1; s <= 4 && s <= steps; ++s) {   // measured: 4 is at or near the best from 1 to 64 tokens
    if (steps % s) continue;
    best = s;
    if ((long)jobs * nb * s >= moe_thread_target()) break;
  }
  return best;
}

inline sycl::event moe_layer(sycl::queue &q, const uint16_t *X, const float *WT, const int32_t *ID, const uint8_t *P13,
                             const float *F13, const uint8_t *P2, const float *F2, MoeScratch b, uint16_t *OUT, int T, int H,
                             int k, int I, int E, float gc, float uc, int fmt) {
  const int rows = T * k, mt = moe_mt(rows, E), maxm = 8 * mt, maxj = moe_max_jobs(rows, E);
  int32_t *UNP = b.unp, *JOBS = b.jobs;
  half *A16 = b.a16, *ACT = b.act;
  float *C1 = b.c1, *C2 = b.c2;
  static const int stages = [] { const char *e = std::getenv("NVF4_STAGES"); return e ? std::atoi(e) : 9; }();   // timing aid
  // Job list and row order, all parallel (a one-thread sort took 22-164 us on this GPU):
  //   one token: every (token, slot) pair is its own one-row job, no sort;
  //   more: atomic count per expert, then each pair takes the next row of its expert; a pair that opens a row tile
  //   writes the job. cnt = int32 [2 * 256] scratch at the head of the jobs buffer's tail (see moe_jobs_ints).
  if (T == 1) {
    q.parallel_for<moe_jobs1_k>(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) {
      const int i = (int)it.get_global_id(0);
      if (i >= maxj) return;
      JOBS[i * 4] = i < rows ? ID[i] : 0; JOBS[i * 4 + 1] = i; JOBS[i * 4 + 2] = i < rows ? 1 : 0; JOBS[i * 4 + 3] = 0;
      if (i < rows) UNP[i] = i;
    });
  } else if (rows <= 65536 && E <= 256 && !route_split()) {
    // v3 (Claude, 23 Nov night, from the eager decode profile): zero + count + prefix + job list in ONE work-group
    // launch instead of four (the one-work-item prefix alone took 18 us per call, 48 calls per step). Counts and row
    // cursors live in local memory, the expert row and job offsets come from group scans. Same START / JB values as
    // the four-kernel path; rows inside an expert are taken in atomic order in both, so outputs do not change.
    // v4 (24 Nov 2026): prompt-sized batches (> 4,096 rows) take the same kernel with 1,024 work-items instead of the
    // four-kernel path, whose device-wide atomics on E counters cost ~115 us per kernel at 20K rows (prefill profile).
    auto one_wg = [&](auto wgc) {
      constexpr int WG = decltype(wgc)::value;
      const int nj4 = maxj * 4;
      q.submit([&](sycl::handler &h) {
        sycl::local_accessor<int32_t, 1> lc(WG, h), lp(WG, h), ls(WG, h), lj(WG, h);
        h.parallel_for<moe_route1_k<WG>>(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
          using lref = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                        sycl::access::address_space::local_space>;
          const int l = (int)it.get_local_id(0);
          const auto g = it.get_group();
          for (int x = l; x < nj4; x += WG) JOBS[x] = 0;
          lc[l] = 0;
          lp[l] = 0;
          sycl::group_barrier(g);
          for (int i = l; i < rows; i += WG) lref(lc[ID[i]]).fetch_add(1);
          sycl::group_barrier(g);
          const int c = l < E ? lc[l] : 0;
          ls[l] = sycl::exclusive_scan_over_group(g, c, sycl::plus<int>());
          lj[l] = sycl::exclusive_scan_over_group(g, (c + maxm - 1) / maxm, sycl::plus<int>());
          sycl::group_barrier(g);
          for (int i = l; i < rows; i += WG) {
            const int e = ID[i];
            const int w = lref(lp[e]).fetch_add(1);
            UNP[i] = ls[e] + w;
            if (w % maxm == 0) {
              const int left = lc[e] - w, j = lj[e] + w / maxm;
              JOBS[j * 4] = e; JOBS[j * 4 + 1] = ls[e] + w; JOBS[j * 4 + 2] = left < maxm ? left : maxm;
            }
          }
        });
      });
    };
    if (rows <= 4096) one_wg(std::integral_constant<int, 256>{});
    else one_wg(std::integral_constant<int, 1024>{});
  } else {
    int32_t *CNT = JOBS + (size_t)maxj * 4, *POS = CNT + 256, *START = POS + 256, *JB = START + 256;
    zero32(q, JOBS, (size_t)maxj * 4 + 512);
    const size_t gs = (size_t)((rows + 15) / 16) * 16;
    q.parallel_for<moe_count_k>(sycl::nd_range<1>(gs, 16), [=](sycl::nd_item<1> it) {
      const size_t i = it.get_global_id(0);
      if (i >= (size_t)rows) return;
      sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> c(CNT[ID[i]]);
      c.fetch_add(1);
    });
    // v2 (Claude, 23 Nov): expert row offsets and job offsets once, by one work-item (E <= 256 adds), instead of every
    // (token, slot) pair re-summing CNT[0..e) in moe_jobs_k (18 us per call at 8 tokens on the B70).
    q.parallel_for<moe_prefix_k>(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) {
      if (it.get_global_id(0) != 0) return;
      int s = 0, j = 0;
      for (int x = 0; x < E; ++x) { START[x] = s; JB[x] = j; const int c = CNT[x]; s += c; j += (c + maxm - 1) / maxm; }
    });
    q.parallel_for<moe_jobs_k>(sycl::nd_range<1>(gs, 16), [=](sycl::nd_item<1> it) {
      const size_t i = it.get_global_id(0);
      if (i >= (size_t)rows) return;
      const int e = ID[i];
      const int start = START[e], jb = JB[e];
      sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ps(POS[e]);
      const int w = ps.fetch_add(1);
      UNP[i] = start + w;
      if (w % maxm == 0) {
        const int left = CNT[e] - w, j = jb + w / maxm;
        JOBS[j * 4] = e; JOBS[j * 4 + 1] = start + w; JOBS[j * 4 + 2] = left < maxm ? left : maxm;
      }
    });
  }
  if (stages < 1) return sycl::event{};
  static const int f13s = [] { const char *e = std::getenv("NVF4_S13"); return e ? std::atoi(e) : 0; }();
  static const int f2s = [] { const char *e = std::getenv("NVF4_S2"); return e ? std::atoi(e) : 0; }();
  const int s13 = f13s ? f13s : moe_split(maxj, 2 * I / 16, H / 64);
  const int s2 = f2s ? f2s : moe_split(maxj, H / 16, I / 64);
  // v2: when 2I == H (Inkling: I = 1152, H = 2304) the split-K destinations are zeroed by kernels that already cover
  // the same index range (C1 [rows, 2I] by moe_rows_k, C2 [rows, H] by moe_situ_k), saving two launches per call.
  const bool fold = 2 * I == H;
  const bool z1 = fold && s13 > 1, z2 = fold && s2 > 1;
  {
    // 8 values per work-item with 16-byte vector loads/stores (24 Nov 2026; the one-value-per-item version spent its time
    // on 64-bit index division, and 64-value ESIMD rows were slower still). Same bf16 -> fp32 -> fp16 conversion per value.
    constexpr int RV = 8;
    const uint32_t chunks = (uint32_t)H / RV;
    const size_t tot = (size_t)rows * chunks, wg = 64;
    q.parallel_for<moe_rows_k>(sycl::nd_range<1>((tot + wg - 1) / wg * wg, wg), [=](sycl::nd_item<1> it) {
      const uint32_t g = (uint32_t)it.get_global_id(0);
      if (g >= tot) return;
      const int i = (int)(g / chunks), c = (int)(g % chunks) * RV;
      const sycl::vec<uint16_t, 8> xv = *reinterpret_cast<const sycl::vec<uint16_t, 8> *>(X + (size_t)(i / k) * H + c);
      sycl::vec<half, 8> hv;
#pragma unroll
      for (int j = 0; j < 8; ++j) hv[j] = half(sycl::bit_cast<float>((uint32_t)xv[j] << 16));
      *reinterpret_cast<sycl::vec<half, 8> *>(A16 + (size_t)UNP[i] * H + c) = hv;
      if (z1) *reinterpret_cast<sycl::vec<float, 8> *>(C1 + (size_t)i * H + c) = sycl::vec<float, 8>(0.f);
    });
  }
  if (stages < 2) return sycl::event{};
  if (s13 > 1 && !z1) zero32(q, C1, (size_t)rows * 2 * I);
  gemm(q, A16, P13, F13, JOBS, C1, maxj, 2 * I, H, s13, fmt, mt, s13 > 1 ? 2 : 1);
  if (stages < 3) return sycl::event{};
  {
    const uint32_t ic = (uint32_t)I / 8;
    const size_t tot = (size_t)rows * ic, wg = 64;
    q.parallel_for<moe_situ_k>(sycl::nd_range<1>((tot + wg - 1) / wg * wg, wg), [=](sycl::nd_item<1> it) {
      const uint32_t g = (uint32_t)it.get_global_id(0);
      if (g >= tot) return;
      const size_t r = g / ic;
      const int i = (int)(g % ic) * 8;
      const sycl::vec<float, 8> ga = *reinterpret_cast<const sycl::vec<float, 8> *>(C1 + r * 2 * I + i);
      const sycl::vec<float, 8> ua = *reinterpret_cast<const sycl::vec<float, 8> *>(C1 + r * 2 * I + I + i);
      sycl::vec<half, 8> o;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float gv = ga[j], uv = ua[j];
        o[j] = half(gc * sycl::tanh(gv / gc) * (1.f / (1.f + sycl::exp(-gv))) * uc * sycl::tanh(uv / uc));
      }
      *reinterpret_cast<sycl::vec<half, 8> *>(ACT + r * I + i) = o;
      if (z2) {
        *reinterpret_cast<sycl::vec<float, 8> *>(C2 + r * H + i) = sycl::vec<float, 8>(0.f);
        *reinterpret_cast<sycl::vec<float, 8> *>(C2 + r * H + I + i) = sycl::vec<float, 8>(0.f);
      }
    });
  }
  if (stages < 4) return sycl::event{};
  if (s2 > 1 && !z2) zero32(q, C2, (size_t)rows * H);
  gemm(q, ACT, P2, F2, JOBS, C2, maxj, H, I, s2, fmt, mt, s2 > 1 ? 2 : 1);
  if (stages < 5) return sycl::event{};
  const uint32_t hc = (uint32_t)H / 8;
  const size_t tot = (size_t)T * hc, wg = 64;
  return q.parallel_for<moe_gather_k>(sycl::nd_range<1>((tot + wg - 1) / wg * wg, wg), [=](sycl::nd_item<1> it) {
    const uint32_t g = (uint32_t)it.get_global_id(0);
    if (g >= tot) return;
    const int t = (int)(g / hc), c = (int)(g % hc) * 8;
    sycl::vec<float, 8> s(0.f);
    for (int j = 0; j < k; ++j) {
      const float w = WT[t * k + j];
      const sycl::vec<float, 8> v = *reinterpret_cast<const sycl::vec<float, 8> *>(C2 + (size_t)UNP[t * k + j] * H + c);
#pragma unroll
      for (int e = 0; e < 8; ++e) s[e] += w * v[e];
    }
    sycl::vec<uint16_t, 8> o;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      uint32_t bb = sycl::bit_cast<uint32_t>((float)s[e]);
      bb += 0x7FFFu + ((bb >> 16) & 1u);
      o[e] = (uint16_t)(bb >> 16);
    }
    *reinterpret_cast<sycl::vec<uint16_t, 8> *>(OUT + (size_t)t * H + c) = o;
  });
}
}  // namespace nvf4
