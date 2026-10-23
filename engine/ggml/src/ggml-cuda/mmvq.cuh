#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

// the most that ggml_cuda_mul_mat_vec_q takes from the memory pool
size_t ggml_cuda_mul_mat_vec_q_pool_size(const ggml_tensor * src1);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

#ifdef GGML_SSD_CUDA_EXECUTOR
// One matrix of `type` (nrows rows of ncols values, stride_row_x blocks apart) times n <= MMVQ_MAX_BATCH_SIZE columns of
// an operand quantized as ggml_cuda_mul_mat_vec_q quantizes src1 (quantize_row_q8_1_cuda, rows padded to
// MATRIX_ROW_PADDING). dst receives n columns of nrows floats.
void ggml_cuda_moe_stream_mmvq(
        ggml_type type, const void * x, int64_t ncols, int64_t nrows, int64_t stride_row_x,
        const void * y, int64_t n, float * dst, cudaStream_t stream);
#endif // GGML_SSD_CUDA_EXECUTOR
