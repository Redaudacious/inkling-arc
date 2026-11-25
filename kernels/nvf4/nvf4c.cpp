// SPDX-License-Identifier: MIT
// libnvf4c: plain C ABI over nvf4_kernels.hpp for the Rust host tools (converter, checker, benchmark driver).
// All pointers are device USM (nvf4_alloc), 64-byte aligned. GEMM calls only submit; nvf4_wait() blocks. Fable, 22 Nov 2026.
#include "nvf4_kernels.hpp"
#include <cstdio>
#include <string>

static sycl::queue &q_() {
  static sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
  return q;
}

extern "C" {
int nvf4_init(char *name, int cap) {
  try {
    auto d = q_().get_device();
    std::string n = d.get_info<sycl::info::device::name>();
    std::snprintf(name, cap, "%s", n.c_str());
    return 0;
  } catch (const std::exception &e) { std::snprintf(name, cap, "ERR %s", e.what()); return 1; }
}
void *nvf4_alloc(size_t bytes) { return sycl::aligned_alloc_device(64, (bytes + 63) / 64 * 64, q_()); }
void nvf4_free(void *p) { q_().wait(); sycl::free(p, q_()); }
void nvf4_h2d(void *dst, const void *src, size_t n) { q_().memcpy(dst, src, n).wait(); }
void nvf4_d2h(void *dst, const void *src, size_t n) { q_().wait(); q_().memcpy(dst, src, n).wait(); }
void nvf4_wait() { q_().wait(); }
void *nvf4_queue() { return &q_(); }
void nvf4_pack(const uint8_t *w, const uint8_t *s, uint8_t *out, int E, int N, int K, int s8) { nvf4::pack(q_(), w, s, out, E, N, K, s8).wait(); }
void nvf4_unpack(const uint8_t *pack, int e, int N, int K, int s8, float *out) { nvf4::unpack(q_(), pack, e, N, K, s8, out).wait(); }
void mx8_pack(const uint8_t *w, const uint8_t *s, const int32_t *smin, uint8_t *out, int E, int N, int K) { nvf4::mx_pack(q_(), w, s, smin, out, E, N, K).wait(); }
void mx8_unpack(const uint8_t *pack, int e, int N, int K, float *out) { nvf4::mx_unpack(q_(), pack, e, N, K, out).wait(); }
// a fp16 [rows, K] grouped by expert; factors fp32 [E]; jobs int32 [nj, 4] = (expert, first row, rows <= 8 * mt, 0).
// split == 1: fp16 result in c16 [rows, N]. split > 1: fp32 sums in c32 [rows, N] (zeroed here).
void nvf4_gemm(const void *a, const uint8_t *pack, const float *f, const int32_t *jobs, void *c16, float *c32, int rows, int nj,
               int N, int K, int split, int s8, int mt, int unused) {
  (void)unused;
  if (split > 1) q_().fill(c32, 0.f, (size_t)rows * N);
  nvf4::gemm(q_(), reinterpret_cast<const nvf4::half *>(a), pack, f, jobs, split > 1 ? (void *)c32 : c16, nj, N, K, split,
             s8 ? 0 : 1, mt <= 1 ? 1 : 4, split > 1 ? 2 : 0);
}
void mx8_gemm(const void *a, const uint8_t *pack, const float *f, const int32_t *jobs, void *c16, float *c32, int rows, int nj,
              int N, int K, int split, int mt, int unused) {
  (void)unused;
  if (split > 1) q_().fill(c32, 0.f, (size_t)rows * N);
  nvf4::gemm(q_(), reinterpret_cast<const nvf4::half *>(a), pack, f, jobs, split > 1 ? (void *)c32 : c16, nj, N, K, split, 2,
             mt <= 1 ? 1 : 4, split > 1 ? 2 : 0);
}
// whole expert layer; scratch pointers as in nvf4::MoeScratch (allocated by the caller)
void nvf4_moe(const uint16_t *x, const float *wt, const int32_t *ids, const uint8_t *p13, const float *f13, const uint8_t *p2,
              const float *f2, int32_t *unp, int32_t *jobs, void *a16, float *c1, void *act, float *c2, uint16_t *out, int T, int H,
              int k, int I, int E, float gc, float uc, int fmt) {
  nvf4::MoeScratch b{unp, jobs, reinterpret_cast<nvf4::half *>(a16), c1, reinterpret_cast<nvf4::half *>(act), c2};
  nvf4::moe_layer(q_(), x, wt, ids, p13, f13, p2, f2, b, out, T, H, k, I, E, gc, uc, fmt);
}
int nvf4_moe_max_jobs(int rows, int E) { return nvf4::moe_jobs_ints(rows, E) / 4 + 1; }   // job scratch size in 16-byte units
}
