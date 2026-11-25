// SPDX-License-Identifier: MIT
// _nvf4 torch operator for the vLLM-XPU Inkling 2 port: one call runs a whole routed-expert layer from the DPAS packs
// (nvf4::moe_layer in nvf4_kernels.hpp). Everything stays on the torch XPU queue (in order, graph-capturable): no host
// readback, the job list is built by a one-work-item kernel. Fable, 22 Nov 2026.
#include "nvf4_kernels.hpp"
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

namespace {
using nvf4::half;

// x bf16 [T, H]; w fp32 [T, k]; ids int32 [T, k] (expert ids 0..E-1); p13 / p2 packs (uint8); f13 / f2 fp32 [E].
// fmt 0 = NVFP4 (byte scale codes), 2 = MXFP8. inter = expert width I (w13 is [2I, H] rows gate then up, w2 is [H, I]).
at::Tensor moe(const at::Tensor &x, const at::Tensor &w, const at::Tensor &ids, const at::Tensor &p13, const at::Tensor &f13,
               const at::Tensor &p2, const at::Tensor &f2, double gate_cap, double up_cap, int64_t fmt, int64_t inter) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kFloat && ids.scalar_type() == at::kInt,
              "_nvf4.moe: x bf16, weights fp32, ids int32");
  TORCH_CHECK(p13.scalar_type() == at::kByte && p2.scalar_type() == at::kByte && f13.scalar_type() == at::kFloat &&
                  f2.scalar_type() == at::kFloat && (fmt == 0 || fmt == 2));
  TORCH_CHECK(x.dim() == 2 && ids.dim() == 2 && w.dim() == 2 && x.is_contiguous() && w.is_contiguous() && ids.is_contiguous());
  const int T = (int)x.size(0), H = (int)x.size(1), k = (int)ids.size(1), I = (int)inter, E = (int)f13.numel();
  TORCH_CHECK(H % 64 == 0 && I % 64 == 0 && E <= 256 && T >= 1);
  const int rows = T * k, maxj = nvf4::moe_max_jobs(rows, E);
  auto &q = at::xpu::getCurrentXPUStream(x.device().index()).queue();
  auto oi = x.options().dtype(at::kInt), oh = x.options().dtype(at::kHalf), of = x.options().dtype(at::kFloat);
  at::Tensor unp_t = at::empty({rows}, oi), jobs_t = at::empty({(int64_t)nvf4::moe_jobs_ints(rows, E)}, oi);
  at::Tensor a16_t = at::empty({rows, H}, oh), c1_t = at::empty({rows, 2 * I}, of), act_t = at::empty({rows, I}, oh);
  at::Tensor c2_t = at::empty({rows, H}, of), out_t = at::empty({T, H}, x.options());
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a16_t.data_ptr()) % 64 == 0 && reinterpret_cast<uintptr_t>(act_t.data_ptr()) % 64 == 0 &&
                  reinterpret_cast<uintptr_t>(p13.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(p2.data_ptr()) % 16 == 0,
              "_nvf4.moe: buffer alignment");
  nvf4::MoeScratch b{unp_t.data_ptr<int32_t>(), jobs_t.data_ptr<int32_t>(), reinterpret_cast<half *>(a16_t.data_ptr()),
                     c1_t.data_ptr<float>(), reinterpret_cast<half *>(act_t.data_ptr()), c2_t.data_ptr<float>()};
  // the row copy reads x 8 values (16 bytes) at a time: a view that starts off a 16-byte boundary is copied first
  const at::Tensor xa = reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 ? x.clone() : x;
  nvf4::moe_layer(q, reinterpret_cast<const uint16_t *>(xa.data_ptr()), w.data_ptr<float>(), ids.data_ptr<int32_t>(),
                  p13.data_ptr<uint8_t>(), f13.data_ptr<float>(), p2.data_ptr<uint8_t>(), f2.data_ptr<float>(), b,
                  reinterpret_cast<uint16_t *>(out_t.data_ptr()), T, H, k, I, E, (float)gate_cap, (float)up_cap, (int)fmt);
  return out_t;
}
}  // namespace

TORCH_LIBRARY(_nvf4, m) {
  m.def("moe(Tensor x, Tensor weights, Tensor ids, Tensor p13, Tensor f13, Tensor p2, Tensor f2, float gate_cap, float up_cap, int fmt, int inter) -> Tensor");
}
TORCH_LIBRARY_IMPL(_nvf4, XPU, m) { m.impl("moe", &moe); }
