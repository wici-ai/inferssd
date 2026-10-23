#include "row-mask.cuh"
#include "convert.cuh"

template <typename T>
static __global__ void row_mask_kernel(const int32_t * n_visible, T * dst, int64_t width) {
    const int64_t row = blockIdx.y;
    const int64_t col = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (col >= width) {
        return;
    }

    dst[row*width + col] = col < n_visible[row] ? ggml_cuda_cast<T>(0.0f) : ggml_cuda_cast<T>(-INFINITY);
}

void ggml_cuda_op_row_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * n_visible = dst->src[0];
    GGML_ASSERT(n_visible->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(n_visible) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_nrows(dst) == n_visible->ne[0]);

    const int64_t width = dst->ne[0];
    const int64_t n_rows = ggml_nrows(dst);
    const dim3 block(256);
    const dim3 grid((width + block.x - 1)/block.x, n_rows);
    const int32_t * counts = (const int32_t *) n_visible->data;

    if (dst->type == GGML_TYPE_F32) {
        row_mask_kernel<<<grid, block, 0, ctx.stream()>>>(counts, (float *) dst->data, width);
    } else if (dst->type == GGML_TYPE_F16) {
        row_mask_kernel<<<grid, block, 0, ctx.stream()>>>(counts, (half *) dst->data, width);
    } else {
        GGML_ABORT("row mask requires F16 or F32");
    }
}
