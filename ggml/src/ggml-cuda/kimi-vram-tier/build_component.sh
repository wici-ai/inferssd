#!/usr/bin/env bash
set -euo pipefail
# Usage: bash build_component.sh SOURCE_DIR OUTPUT_DIR [ACTUAL_CPU_REFERENCE_LIB_DIR]
parity_source=$(cd "$1" && pwd)
mkdir -p "$2"
parity_out=$(cd "$2" && pwd)
parity_nvcc=${PARITY_NVCC:-/usr/local/cuda/bin/nvcc}
parity_arch=${PARITY_CUDA_ARCH:-sm_120a}
"$parity_nvcc" -std=c++17 -O3 --fmad=false --ftz=false -arch="$parity_arch" \
  -shared -Xcompiler=-fPIC,-Wall,-Wextra \
  -I"$parity_source/ggml/include" -I"$parity_source/ggml/src" \
  "$parity_source/ggml/src/ggml-cuda/kimi-q3k-chain.cu" \
  -o "$parity_out/libkimi-q3k-parity.so"
cp "$parity_source/ggml/include/ggml-kimi-q3k-parity.h" "$parity_out/"
cp "$parity_source/ggml/include/ggml-kimi-q3k-chain.h" "$parity_out/"
if [[ $# -ge 3 ]]; then
  parity_ref=$(cd "$3" && pwd)
  g++ -std=c++17 -O3 -ffp-contract=off \
    -I"$parity_source/ggml/include" -I"$parity_source/ggml/src" -I"$parity_source/ggml/src/ggml-cpu" \
    "$parity_source/tests/test-kimi-q3k-parity.cpp" -L"$parity_ref" -Wl,-rpath,"$parity_ref" \
    -lggml-cpu -lggml-base -ldl -pthread -o "$parity_out/test-kimi-q3k-parity"
fi
