#include "mmvf-f16-cpu.cuh"
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

static __global__ void quantize_f32_to_f16_cpu(
        const float * __restrict__ input,
        half * __restrict__ output,
        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = __float2half_rn(input[i]);
    }
}

static constexpr int F16_CPU_WARPS_PER_BLOCK = 8;

// blk.N.ssm_f_b has 128 columns.  The native x86 CPU backend uses four
// 16-lane AVX-512 accumulators and then a fixed reduction tree.  One half-warp
// mirrors those lanes so the GPU path consumes the same F16 activation and
// preserves the CPU FMA/add order.
__launch_bounds__(F16_CPU_WARPS_PER_BLOCK * WARP_SIZE, 1)
static __global__ void mul_mat_vec_f16_cpu_128(
        const char * __restrict__ weights,
        size_t row_stride,
        const half * __restrict__ input,
        int64_t nrows,
        float * __restrict__ output) {
    const int lane = threadIdx.x;
    const int64_t row = (int64_t) blockIdx.x * F16_CPU_WARPS_PER_BLOCK + threadIdx.y;
    if (row >= nrows || lane >= 16) {
        return;
    }

    const half * x = (const half *) (weights + (size_t) row * row_stride);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

#pragma unroll
    for (int base = 0; base < 128; base += 64) {
        sum0 = fma_rn(__half2float(x[base + lane]),      __half2float(input[base + lane]),      sum0);
        sum1 = fma_rn(__half2float(x[base + 16 + lane]), __half2float(input[base + 16 + lane]), sum1);
        sum2 = fma_rn(__half2float(x[base + 32 + lane]), __half2float(input[base + 32 + lane]), sum2);
        sum3 = fma_rn(__half2float(x[base + 48 + lane]), __half2float(input[base + 48 + lane]), sum3);
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

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

} // namespace

bool ggml_cuda_should_use_mmvf_f16_cpu(
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
        src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src0->ne[0] == 128 && src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        dst->ne[1] == 1 && dst->ne[2] == 1 && dst->ne[3] == 1 &&
        src0->nb[0] == sizeof(half) && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
        ggml_cuda_dense_parity_f16_selected(src0->name);
#endif
}

void ggml_cuda_mul_mat_vec_f16_cpu(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_ABORT("CPU-compatible F16 dense matvec is only implemented for CUDA");
#else
    GGML_ASSERT(ggml_cuda_should_use_mmvf_f16_cpu(src0, src1, dst));

    ggml_cuda_pool_alloc<half> input_f16(ctx.pool(), src0->ne[0]);
    cudaStream_t stream = ctx.stream();
    quantize_f32_to_f16_cpu<<<1, 128, 0, stream>>>(
        (const float *) src1->data, input_f16.get(), (int) src0->ne[0]);

    const dim3 block(WARP_SIZE, F16_CPU_WARPS_PER_BLOCK, 1);
    const dim3 grid((unsigned int) ((src0->ne[1] + F16_CPU_WARPS_PER_BLOCK - 1) /
        F16_CPU_WARPS_PER_BLOCK), 1, 1);
    mul_mat_vec_f16_cpu_128<<<grid, block, 0, stream>>>(
        (const char *) src0->data,
        src0->nb[1],
        input_f16.get(),
        src0->ne[1],
        (float *) dst->data);

    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        std::fprintf(stderr, "ggml_cuda: CPU-compatible F16 dense matvec active for %s\n", src0->name);
    }
    static std::atomic<bool> resident_logged{false};
    if (ggml_cuda_dense_parity_resident_selected(src0->name) && !resident_logged.exchange(true)) {
        std::fprintf(stderr, "ggml_cuda: CPU-compatible F16 resident parity active for %s\n", src0->name);
    }
#endif
}
