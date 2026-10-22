#pragma once

#include "common.cuh"

bool ggml_cuda_should_use_mmvf_f32_cpu(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst);

void ggml_cuda_mul_mat_vec_f32_cpu(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);

bool ggml_cuda_should_use_kimi_kda_weighted_sum(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst);

void ggml_cuda_kimi_kda_weighted_sum(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);
