#pragma once

#include "common.cuh"

enum ggml_cuda_activation_quant_type {
    GGML_CUDA_ACTIVATION_QUANT_Q8_1,
    GGML_CUDA_ACTIVATION_QUANT_Q8_K,
};

void ggml_cuda_activation_quant_reuse_begin(ggml_backend_cuda_context & ctx);
void ggml_cuda_activation_quant_reuse_end(ggml_backend_cuda_context & ctx);

void * ggml_cuda_activation_quant_reuse_acquire(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src,
        ggml_cuda_activation_quant_type type,
        size_t size,
        bool & hit);
