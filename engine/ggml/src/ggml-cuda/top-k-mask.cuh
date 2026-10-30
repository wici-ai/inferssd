// SPDX-License-Identifier: MIT
// Ported from llama.cpp 2e0d86bebc.
#include "common.cuh"

void ggml_cuda_op_top_k_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
