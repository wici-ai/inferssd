#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// the most that ggml_cuda_flash_attn_ext takes from the memory pool, from the kernel it would launch
size_t ggml_cuda_flash_attn_ext_pool_size(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);
