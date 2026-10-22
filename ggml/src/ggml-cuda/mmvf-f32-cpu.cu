#include "mmvf-f32-cpu.cuh"
#include "cpu-parity.cuh"
#include "dense-parity.cuh"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static bool dense_cpu_semantics_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("GGML_CUDA_DENSE_Q8_K");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0 &&
            std::strcmp(value, "false") != 0;
    }();
    return enabled;
}

static __device__ __forceinline__ float add_rn(float a, float b) {
    return __fadd_rn(a, b);
}

static __device__ __forceinline__ float fma_rn(float a, float b, float c) {
    return __fmaf_rn(a, b, c);
}

static constexpr int F32_CPU_WARPS_PER_BLOCK = 8;

// The native x86 F32 dot uses four 16-lane AVX-512 accumulators over each
// 64-value group, followed by the same fixed reduction tree as the F16 dot.
__launch_bounds__(F32_CPU_WARPS_PER_BLOCK * WARP_SIZE, 1)
static __global__ void mul_mat_vec_f32_cpu(
        const char * __restrict__ weights,
        size_t row_stride,
        const float * __restrict__ input,
        int ncols,
        int64_t nrows,
        float * __restrict__ output) {
    const int lane = threadIdx.x;
    const int64_t row = (int64_t) blockIdx.x * F32_CPU_WARPS_PER_BLOCK + threadIdx.y;
    if (row >= nrows || lane >= 16) {
        return;
    }

    const float * x = (const float *) (weights + (size_t) row * row_stride);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    for (int base = 0; base < ncols; base += 64) {
        sum0 = fma_rn(x[base + lane],      input[base + lane],      sum0);
        sum1 = fma_rn(x[base + 16 + lane], input[base + 16 + lane], sum1);
        sum2 = fma_rn(x[base + 32 + lane], input[base + 32 + lane], sum2);
        sum3 = fma_rn(x[base + 48 + lane], input[base + 48 + lane], sum3);
    }

    float value = add_rn(add_rn(sum0, sum2), add_rn(sum1, sum3));
    constexpr unsigned int mask = 0x0000ffffU;
    value = add_rn(__shfl_sync(mask, value, lane + 8, 16), value);
    value = add_rn(__shfl_sync(mask, value, lane + 4, 16), value);
    value = add_rn(__shfl_sync(mask, value, lane + 2, 16), value);
    value = add_rn(value, __shfl_sync(mask, value, lane + 1, 16));

    if (lane == 0) {
        output[row] = value;
    }
}

static __global__ void kimi_kda_weighted_sum_f32(
        const char * __restrict__ rows,
        const size_t row_stride,
        const float * __restrict__ weights,
        const int ncols,
        float * __restrict__ output,
        const int64_t nrows) {
    const int64_t row = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }

    const float * values = (const float *) (rows + (size_t) row*row_stride);
    float sum = 0.0f;
    for (int col = 0; col < ncols; ++col) {
        sum = __fmaf_rn(values[col], weights[col], sum);
    }
    output[row] = sum;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

} // namespace

bool ggml_cuda_should_use_mmvf_f32_cpu(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
#else
    return dense_cpu_semantics_enabled() && src0 != nullptr && src1 != nullptr && dst != nullptr &&
        src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src0->ne[0] % 64 == 0 && src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        dst->ne[1] == 1 && dst->ne[2] == 1 && dst->ne[3] == 1 &&
        src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
        ggml_cuda_dense_parity_f32_selected(src0->name);
#endif
}

bool ggml_cuda_should_use_kimi_kda_weighted_sum(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
#else
    return src0 != nullptr && src1 != nullptr && dst != nullptr &&
        ggml_cuda_kimi_kda_cpu_parity_enabled(dst) &&
        src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src0->ne[0] >= 2 && src0->ne[0] <= 3 && src0->ne[1] == 7168 &&
        src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src1->ne[0] == src0->ne[0] && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        dst->ne[0] == 7168 && dst->ne[1] == 1 && dst->ne[2] == 1 && dst->ne[3] == 1 &&
        src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float);
#endif
}

void ggml_cuda_kimi_kda_weighted_sum(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_ABORT("CPU-compatible KDA weighted sum is only implemented for CUDA");
#else
    GGML_ASSERT(ggml_cuda_should_use_kimi_kda_weighted_sum(src0, src1, dst));
    constexpr int threads = 256;
    const int64_t nrows = src0->ne[1];
    const dim3 block(threads, 1, 1);
    const dim3 grid((unsigned int) ((nrows + threads - 1) / threads), 1, 1);
    kimi_kda_weighted_sum_f32<<<grid, block, 0, ctx.stream()>>>(
        (const char *) src0->data, src0->nb[1], (const float *) src1->data,
        (int) src0->ne[0], (float *) dst->data, nrows);
#endif
}

void ggml_cuda_mul_mat_vec_f32_cpu(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_ABORT("CPU-compatible F32 dense matvec is only implemented for CUDA");
#else
    GGML_ASSERT(ggml_cuda_should_use_mmvf_f32_cpu(src0, src1, dst));

    cudaStream_t stream = ctx.stream();
    const dim3 block(WARP_SIZE, F32_CPU_WARPS_PER_BLOCK, 1);
    const dim3 grid((unsigned int) ((src0->ne[1] + F32_CPU_WARPS_PER_BLOCK - 1) /
        F32_CPU_WARPS_PER_BLOCK), 1, 1);
    mul_mat_vec_f32_cpu<<<grid, block, 0, stream>>>(
        (const char *) src0->data,
        src0->nb[1],
        (const float *) src1->data,
        (int) src0->ne[0],
        src0->ne[1],
        (float *) dst->data);

    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        std::fprintf(stderr, "ggml_cuda: CPU-compatible F32 router matvec active for %s\n", src0->name);
    }
    static std::atomic<bool> resident_logged{false};
    if (ggml_cuda_dense_parity_resident_selected(src0->name) && !resident_logged.exchange(true)) {
        std::fprintf(stderr, "ggml_cuda: CPU-compatible F32 resident parity active for %s\n", src0->name);
    }
#endif
}
