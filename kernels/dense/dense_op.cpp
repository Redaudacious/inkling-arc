// SPDX-License-Identifier: MIT
// _nvf4 dense operators for the vLLM-XPU Inkling 2 port (lib_dense_op.so, loaded next to lib_nvf4_op.so):
//   _nvf4.dense_pack(w int8 [N,K], scale fp32 [N], segs int[]) -> (packed uint8, scale_p fp32 [Np])
//   _nvf4.dense(x bf16 [M,K], packed, scale_p, n) -> bf16 [M, n]
//   _nvf4.dense_seg(x bf16 [G*T,K], packed, scale_p, segs int[G]) -> G tensors bf16 [T, segs[g]]
// One kernel launch per GEMM on the torch XPU queue: no host sync, no host read of device data (graph-capturable).
// Fable, 22 Nov 2026.
#include "dense_kernels.hpp"
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>
#include <tuple>
#include <vector>

namespace {
sycl::queue &queue_of(const at::Tensor &t) { return at::xpu::getCurrentXPUStream(t.device().index()).queue(); }
bool aligned64(const at::Tensor &t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 64 == 0; }

std::tuple<at::Tensor, at::Tensor> dense_pack(const at::Tensor &w, const at::Tensor &scale, at::IntArrayRef segs) {
  TORCH_CHECK(w.scalar_type() == at::kChar && w.dim() == 2 && scale.scalar_type() == at::kFloat, "_nvf4.dense_pack: w int8 [N,K], scale fp32 [N]");
  const int N = (int)w.size(0), K = (int)w.size(1);
  TORCH_CHECK(K % 64 == 0 && scale.numel() == N && segs.size() <= 4, "_nvf4.dense_pack: K % 64, scale [N], at most 4 segments");
  int64_t tot = 0;
  for (auto s : segs) tot += s;
  TORCH_CHECK(segs.empty() || tot == N, "_nvf4.dense_pack: segments must sum to N");
  const dense::Segs sg = dense::make_segs(segs.data(), (int)segs.size(), N);
  at::Tensor wc = w.contiguous(), sc = scale.contiguous();
  at::Tensor packed = at::empty({(int64_t)sg.Np * K}, w.options().dtype(at::kByte));
  at::Tensor sp = at::empty({(int64_t)sg.Np}, w.options().dtype(at::kFloat));
  TORCH_CHECK(aligned64(packed) && aligned64(sp), "_nvf4.dense_pack: buffer alignment");
  dense::pack(queue_of(w), wc.data_ptr<int8_t>(), sc.data_ptr<float>(), packed.data_ptr<uint8_t>(), sp.data_ptr<float>(), K, sg);
  return {packed, sp};
}

at::Tensor run(const at::Tensor &x0, const at::Tensor &packed, const at::Tensor &sp, int G, int b1, int b2, int b3) {
  TORCH_CHECK(x0.scalar_type() == at::kBFloat16 && x0.dim() == 2 && packed.scalar_type() == at::kByte && sp.scalar_type() == at::kFloat,
              "_nvf4.dense: x bf16 [M,K], packed uint8, scale fp32");
  const int M = (int)x0.size(0), K = (int)x0.size(1), Np = (int)sp.numel();
  TORCH_CHECK(K % 64 == 0 && Np % 16 == 0 && packed.numel() == (int64_t)Np * K && M >= G && M % G == 0, "_nvf4.dense: shapes");
  at::Tensor x = x0.contiguous();
  if (!aligned64(x)) { at::Tensor y = at::empty_like(x); y.copy_(x); x = y; }
  const int T = M / G;
  at::Tensor out = at::empty({T, Np}, x0.options());
  TORCH_CHECK(aligned64(x) && aligned64(out) && aligned64(packed) && aligned64(sp), "_nvf4.dense: buffer alignment");
  static const int f_split = dense::env_int("DENSE_SPLIT", 0), f_big = dense::env_int("DENSE_BIG", -1), f_pre = dense::env_int("DENSE_PRE", -1);
  const int pre = f_pre >= 0 ? f_pre : dense::auto_pre(T);
  sycl::queue &q = queue_of(x);
  if (pre) {   // many rows: one bf16 -> fp16 pass, then the GEMM reads fp16
    at::Tensor x16 = at::empty({M, K}, x.options().dtype(at::kHalf));
    TORCH_CHECK(aligned64(x16), "_nvf4.dense: buffer alignment");
    dense::to_fp16(q, reinterpret_cast<const uint16_t *>(x.data_ptr()), reinterpret_cast<dense::half *>(x16.data_ptr()), (size_t)M * K);
    x = x16;
  }
  dense::gemm(q, x.data_ptr(), packed.data_ptr<uint8_t>(), sp.data_ptr<float>(), reinterpret_cast<uint16_t *>(out.data_ptr()), T, Np,
              K, b1, b2, b3, f_split, 0, pre, f_big);
  return out;
}

at::Tensor dense_fn(const at::Tensor &x, const at::Tensor &packed, const at::Tensor &sp, int64_t n) {
  const int NB = (int)sp.numel() / 16;
  at::Tensor out = run(x, packed, sp, 1, NB, NB, NB);
  return n == out.size(1) ? out : out.narrow(1, 0, n);
}

std::vector<at::Tensor> dense_seg(const at::Tensor &x, const at::Tensor &packed, const at::Tensor &sp, at::IntArrayRef segs) {
  TORCH_CHECK(segs.size() >= 1 && segs.size() <= 4, "_nvf4.dense_seg: 1 to 4 segments");
  const dense::Segs sg = dense::make_segs(segs.data(), (int)segs.size(), 0);
  TORCH_CHECK(sg.Np == sp.numel(), "_nvf4.dense_seg: segments do not match the pack");
  const int NB = sg.Np / 16;
  int b[3] = {NB, NB, NB};
  for (int i = 1; i < sg.n; ++i) b[i - 1] = sg.off[i] / 16;
  at::Tensor out = run(x, packed, sp, sg.n, b[0], b[1], b[2]);
  std::vector<at::Tensor> r;
  for (int i = 0; i < sg.n; ++i) r.push_back(out.narrow(1, sg.off[i], sg.len[i]));
  return r;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_nvf4, m) {
  m.def("dense_pack(Tensor w, Tensor scale, int[] segs) -> (Tensor, Tensor)");
  m.def("dense(Tensor x, Tensor packed, Tensor scale, int n) -> Tensor");
  m.def("dense_seg(Tensor x, Tensor packed, Tensor scale, int[] segs) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(_nvf4, XPU, m) {
  m.impl("dense_pack", &dense_pack);
  m.impl("dense", &dense_fn);
  m.impl("dense_seg", &dense_seg);
}
