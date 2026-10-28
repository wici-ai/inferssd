// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common.cuh"

bool ggml_cuda_should_use_mmvq_q8k(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst);

void ggml_cuda_mul_mat_vec_q4k_q8k(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);
