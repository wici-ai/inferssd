#include "common.cuh"

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// the most that ggml_cuda_op_argsort takes from the memory pool
size_t ggml_cuda_op_argsort_pool_size(const ggml_tensor * dst);

#ifdef GGML_CUDA_USE_CUB
void argsort_f32_i32_cuda_cub(ggml_cuda_pool & pool,
                              const float *    x,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              ggml_sort_order  order,
                              cudaStream_t     stream);

// the most that argsort_f32_i32_cuda_cub takes from the memory pool
size_t argsort_f32_i32_cuda_cub_pool_size(const int ncols, const int nrows, ggml_sort_order order);
#endif  // GGML_CUDA_USE_CUB
void argsort_f32_i32_cuda_bitonic(const float *   x,
                                  int *           dst,
                                  const int       ncols,
                                  const int       nrows,
                                  ggml_sort_order order,
                                  cudaStream_t    stream);
