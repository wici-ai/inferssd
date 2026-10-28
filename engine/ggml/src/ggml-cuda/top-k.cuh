#include "common.cuh"

#include <cstddef>

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// the most that ggml_cuda_op_top_k takes from the memory pool
size_t ggml_cuda_op_top_k_pool_size(const ggml_tensor * dst);

// Self-test for the Kimi K3 pipeline top-k. This is deliberately reached
// through the CUDA registry rather than added to the public ggml API.
bool ggml_cuda_kimi_pipeline_top_k_self_test(char * digest, size_t digest_size);
