#!/usr/bin/env bash
set -euo pipefail
# Usage: bash build_mxfp4_component.sh SOURCE_DIR OUTPUT_DIR
# Same strict rounding flags as the Q3_K component (build_component.sh). The chain
# extension includes the unchanged parity source, so one DSO carries both APIs.
parity_source=$(cd "$1" && pwd)
mkdir -p "$2"
parity_out=$(cd "$2" && pwd)
parity_nvcc=${PARITY_NVCC:-/usr/local/cuda/bin/nvcc}
parity_arch=${PARITY_CUDA_ARCH:-sm_120a}
tier_dir="$parity_source/ggml/src/ggml-cuda/kimi-vram-tier"
"$parity_nvcc" -std=c++17 -O3 --fmad=false --ftz=false -arch="$parity_arch" \
  -shared -cudart shared -Xcompiler=-fPIC,-Wall,-Wextra \
  -I"$tier_dir" -I"$parity_source/ggml/include" -I"$parity_source/ggml/src" \
  "$tier_dir/kimi-mxfp4-chain.cu" "$tier_dir/kimi-mxfp4-batch.cu" \
  -o "$parity_out/libkimi-mxfp4-parity.so"
