#!/usr/bin/env bash
# Build every inkling-arc GPU library for the Arc Pro B70 (BMG-G31, AOT) inside the builder image. Repo mounted at /work.
#   build/lib_nvf4_op.so   routed experts: fp16 DPAS on the lossless NVFP4 / MXFP8 packs (torch op _nvf4.moe)
#   build/lib_dense_op.so  dense projections (torch ops _nvf4.dense*)
#   build/libnvf4c.so      C ABI of the same kernels for the Rust packer/tester (tools/nvf4)
#   build/_kvq_decode_xe2  global attention on the K8/V4 cache: ESIMD decode/verify/prefill + oneDNN far-key prefill
#   build/libdnnl.so.3     oneDNN v3.14.2 (GPU, graph API) for the far-key fused SDPA
set -euo pipefail
export PATH=/opt/intel/oneapi/compiler/2026.1/bin:/opt/venv/bin:$PATH
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2026.1/lib:${LD_LIBRARY_PATH:-}
PY=/opt/venv/bin/python3
TORCH=$($PY -c "import torch,pathlib;print(pathlib.Path(torch.__file__).resolve().parent)")
PYINC=$($PY -c "import sysconfig;print(sysconfig.get_path('include'))")
W=/work; B=$W/build; mkdir -p $B
AOT=(-fsycl -fsycl-targets=spir64_gen -Xs "-device bmg-g31")
TORCHF=(-I$TORCH/include -I$TORCH/include/torch/csrc/api/include -I$PYINC -D_GLIBCXX_USE_CXX11_ABI=1)
TORCHL=(-L$TORCH/lib -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu -Wl,-rpath,$TORCH/lib)

echo "== experts (_nvf4.moe)"
icpx -O3 "${AOT[@]}" -fPIC -shared -std=c++20 -Wno-deprecated-declarations -I$W/kernels/nvf4 "${TORCHF[@]}" \
  $W/kernels/nvf4/nvf4_op.cpp -o $B/lib_nvf4_op.so "${TORCHL[@]}"
echo "== dense (_nvf4.dense)"
icpx -O3 "${AOT[@]}" -fPIC -shared -std=c++20 -Wno-deprecated-declarations -I$W/kernels/dense -I$W/kernels/nvf4 "${TORCHF[@]}" \
  $W/kernels/dense/dense_op.cpp -o $B/lib_dense_op.so "${TORCHL[@]}"
echo "== pack/test library (C ABI)"
icpx -O3 "${AOT[@]}" -fPIC -shared -std=c++17 -Wno-deprecated-declarations -I$W/kernels/nvf4 $W/kernels/nvf4/nvf4c.cpp -o $B/libnvf4c.so
echo "== oneDNN v3.14.2 (GPU SYCL, graph API)"
if [ ! -f $B/libdnnl.so.3 ]; then
  D=/tmp/onednn; git clone -q --depth 1 -b v3.14.2 https://github.com/uxlfoundation/oneDNN $D
  cmake -S $D -B $D/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_INSTALL_PREFIX=$B/onednn -DONEDNN_CPU_RUNTIME=NONE -DONEDNN_GPU_RUNTIME=SYCL -DONEDNN_GPU_VENDOR=INTEL \
    -DONEDNN_BUILD_GRAPH=ON -DONEDNN_BUILD_TESTS=OFF -DONEDNN_BUILD_EXAMPLES=OFF >/dev/null
  cmake --build $D/build --parallel "${JOBS:-8}" >/dev/null && cmake --install $D/build >/dev/null
  cp -L $B/onednn/lib/libdnnl.so.3 $B/libdnnl.so.3
fi
echo "== global attention (_kvq_decode_xe2, ESIMD + oneDNN far keys)"
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg_g31 -options -ze-opt-large-register-file" -O3 -std=c++20 -fPIC -shared \
  -DTORCH_EXTENSION_NAME=_kvq_decode_xe2 -DKVQ_DNNL "${TORCHF[@]}" -I$B/onednn/include \
  $W/kernels/attention/kvq_decode_xe2.cpp -L$B -Wl,-rpath,'$ORIGIN' -l:libdnnl.so.3 -ltorch_python "${TORCHL[@]}" \
  -o $B/_kvq_decode_xe2.cpython-312-x86_64-linux-gnu.so
sha256sum $B/*.so $B/libdnnl.so.3
echo KERNELS_OK
