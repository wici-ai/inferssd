#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/opt/homebrew/bin:$PATH"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "This build profile requires an Apple Silicon Mac." >&2
    exit 1
fi

cmake -S "$repo_dir" -B "$repo_dir/build-mac" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF -DGGML_METAL=ON -DGGML_ACCELERATE=ON \
    -DGGML_OPENMP=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build "$repo_dir/build-mac" \
    --target llama-cli llama-server test-backend-ops -j "${SSD_LLAMA_BUILD_JOBS:-8}"
