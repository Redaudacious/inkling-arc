#!/usr/bin/env bash
# Base image for inkling-arc: vLLM-XPU with the Inkling 2 model from F3zz1k/vllm (inkling2-b70 @ edd722c), built on the
# Lumnus/b70-flash-next kit v0.32.0-b70.7 (both Apache-2.0). CPU compile only, no GPU use; ~1-2 h, ~25 GB of images.
# Produces localhost/vllm-xpu:b70-inkling2 (the tag the rest of inkling-arc builds on).
set -euo pipefail
ENGINE=${INKLING_ARC_ENGINE:-podman}
[ "$ENGINE" = podman ] || { echo "the kit's image scripts use podman; install podman for the base build" >&2; exit 2; }
W=${INKLING_ARC_WORK:-$(cd "$(dirname "$0")/.." && pwd)/data/base-build}
KIT_TAG=v0.32.0-b70.7
VLLM_COMMIT=edd722c73517b43680483187afca93fb6ab41958
mkdir -p "$W" "$W/tmp"
export TMPDIR=$W/tmp   # big layers; a 12 GB /tmp tmpfs is not enough
export MAX_JOBS=${MAX_JOBS:-6}
[ -d "$W/kit" ] || git clone -q --branch "$KIT_TAG" --depth 1 https://github.com/Redaudacious/b70-flash-next "$W/kit"
if [ ! -d "$W/vllm" ]; then
  git init -q "$W/vllm" && git -C "$W/vllm" fetch -q --depth 1 https://github.com/Redaudacious/vllm "$VLLM_COMMIT" && git -C "$W/vllm" checkout -q FETCH_HEAD
fi
export KIT=$W/kit
cd "$W/vllm"
step() { echo "=== $(date -Is) $*"; }
step "source archive";  git archive -o "$W/vllm-src.tar" HEAD
step "env image";       podman image inspect vllm-xpu-b70env:0.1.16.2 >/dev/null 2>&1 || ENV_TAG=vllm-xpu-b70env:0.1.16.2 bash b70/image/build.sh env "$W/vllm-src.tar"
step "env-update";      podman image inspect vllm-xpu-b70env:0.1.16.2-w4a8 >/dev/null 2>&1 || bash b70/image/build.sh env-update "$W/vllm-src.tar"
step "inkling2 kernels"; [ -f "$W/inkling2-out/_inkling2_C.abi3.so" ] || OUT=$W/inkling2-out bash b70/inkling2-kernels/build.sh
step "engine image";    podman image inspect vllm-xpu:b70-inkling2-engine >/dev/null 2>&1 || TAG=vllm-xpu:b70-inkling2-engine bash b70/image/build.sh vllm "$W/vllm-src.tar"
step "kernel layer";    podman image inspect vllm-xpu:b70-inkling2 >/dev/null 2>&1 || podman build --format docker -f b70/inkling2-kernels/Dockerfile.layer --build-arg ENGINE=vllm-xpu:b70-inkling2-engine -t vllm-xpu:b70-inkling2 "$W/inkling2-out"
step "done";            podman image inspect localhost/vllm-xpu:b70-inkling2 --format '{{.Id}}'
