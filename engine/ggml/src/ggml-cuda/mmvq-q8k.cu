// SPDX-License-Identifier: Apache-2.0
#include "mmvq-q8k.cuh"
#include "activation-quant-reuse.cuh"
#include "dense-parity.cuh"
#include "quantize.cuh"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static bool dense_q8k_enabled() {
    return false; // Frozen production profile: historical unset default.
}

static int dense_q8k_warps_per_block() {
    return 8; // Frozen production profile: historical unset default.
}

static __device__ __forceinline__ float mul_rn(float a, float b) {
    return __fmul_rn(a, b);
}

static __device__ __forceinline__ float add_rn(float a, float b) {
    return __fadd_rn(a, b);
}

static __device__ __forceinline__ float fma_rn(float a, float b, float c) {
    return __fmaf_rn(a, b, c);
}

static __device__ __forceinline__ void q4k_scale_min(
        const block_q4_K & block,
        int group,
        int & scale,
        int & minimum) {
    if (group < 4) {
        scale = block.scales[group] & 0x3f;
        minimum = block.scales[group + 4] & 0x3f;
        return;
    }

    scale = (block.scales[group + 4] & 0x0f) |
        ((block.scales[group - 4] >> 6) << 4);
    minimum = (block.scales[group + 4] >> 4) |
        ((block.scales[group] >> 6) << 4);
}

static constexpr int Q4K_Q8K_ROWS_PER_WARP = 4;

// Eight lanes reproduce the eight int32/float accumulators in the AVX2 CPU
// Q4_K x Q8_K dot product.  Four independent lane groups fill each warp, so no
// threads are idle and all 32-byte weight loads remain coalesced.
template<int warps_per_block>
__launch_bounds__(warps_per_block * WARP_SIZE, 1)
static __global__ void mul_mat_vec_q4k_q8k(
        const char * __restrict__ weights,
        size_t row_stride,
        const block_q8_K * __restrict__ input,
        int nblocks,
        int64_t nrows,
        float * __restrict__ output) {
    const int lane = threadIdx.x;
    const int lane8 = lane & 7;
    const int row_in_warp = lane >> 3;
    const int64_t row = (int64_t) blockIdx.x * (warps_per_block * Q4K_Q8K_ROWS_PER_WARP) +
        (int64_t) threadIdx.y * Q4K_Q8K_ROWS_PER_WARP + row_in_warp;
    if (row >= nrows) {
        return;
    }

    const block_q4_K * x = (const block_q4_K *) (weights + (size_t) row * row_stride);
    float acc = 0.0f;
    float acc_min = 0.0f;

    for (int ib = 0; ib < nblocks; ++ib) {
        int scaled_dot = 0;
#pragma unroll
        for (int group = 0; group < QK_K/32; ++group) {
            int scale;
            int minimum;
            q4k_scale_min(x[ib], group, scale, minimum);

            const uint32_t packed = *(const uint32_t *)
                (x[ib].qs + 32*(group/2) + 4*lane8);
            const int q4 = (int) ((packed >> (4*(group & 1))) & 0x0f0f0f0fU);
            const int q8 = *(const int *) (input[ib].qs + 32*group + 4*lane8);
            scaled_dot += scale * ggml_cuda_dp4a(q4, q8, 0);
        }

        const float d = mul_rn(input[ib].d, __half2float(__low2half(x[ib].dm)));
        acc = fma_rn(d, (float) scaled_dot, acc);

        if (lane8 < 4) {
            const int group0 = 2*lane8;
            const int group1 = group0 + 1;
            int unused_scale;
            int minimum0;
            int minimum1;
            q4k_scale_min(x[ib], group0, unused_scale, minimum0);
            q4k_scale_min(x[ib], group1, unused_scale, minimum1);
            const int minimum_dot =
                minimum0 * ((int) input[ib].bsums[2*group0 + 0] + (int) input[ib].bsums[2*group0 + 1]) +
                minimum1 * ((int) input[ib].bsums[2*group1 + 0] + (int) input[ib].bsums[2*group1 + 1]);
            const float dmin = mul_rn(-input[ib].d, __half2float(__high2half(x[ib].dm)));
            acc_min = fma_rn(dmin, (float) minimum_dot, acc_min);
        }
    }

    const unsigned int mask = 0xffU << (8*row_in_warp);
    const float sum04 = add_rn(__shfl_sync(mask, acc, 4, 8), __shfl_sync(mask, acc, 0, 8));
    const float sum26 = add_rn(__shfl_sync(mask, acc, 6, 8), __shfl_sync(mask, acc, 2, 8));
    const float sum15 = add_rn(__shfl_sync(mask, acc, 5, 8), __shfl_sync(mask, acc, 1, 8));
    const float sum37 = add_rn(__shfl_sync(mask, acc, 7, 8), __shfl_sync(mask, acc, 3, 8));
    const float sum_main = add_rn(add_rn(sum04, sum26), add_rn(sum15, sum37));

    const float min02 = add_rn(
        __shfl_sync(mask, acc_min, 2, 8), __shfl_sync(mask, acc_min, 0, 8));
    const float min13 = add_rn(
        __shfl_sync(mask, acc_min, 3, 8), __shfl_sync(mask, acc_min, 1, 8));
    const float sum_min = add_rn(min02, min13);

    if (lane8 == 0) {
        output[row] = add_rn(sum_main, sum_min);
    }
}

template<int warps_per_block>
static void launch_mul_mat_vec_q4k_q8k(
        const ggml_tensor * src0,
        const block_q8_K * input,
        int nblocks,
        ggml_tensor * dst,
        cudaStream_t stream) {
    constexpr int rows_per_block = warps_per_block * Q4K_Q8K_ROWS_PER_WARP;
    const dim3 block(WARP_SIZE, warps_per_block, 1);
    const dim3 grid((unsigned int) ((src0->ne[1] + rows_per_block - 1) / rows_per_block), 1, 1);
    mul_mat_vec_q4k_q8k<warps_per_block><<<grid, block, 0, stream>>>(
        (const char *) src0->data,
        src0->nb[1],
        input,
        nblocks,
        src0->ne[1],
        (float *) dst->data);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

} // namespace

bool ggml_cuda_should_use_mmvq_q8k(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
#else
    return dense_q8k_enabled() && src0 != nullptr && src1 != nullptr && dst != nullptr &&
        src0->type == GGML_TYPE_Q4_K && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src0->ne[0] % QK_K == 0 && src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        dst->ne[1] == 1 && dst->ne[2] == 1 && dst->ne[3] == 1 &&
        src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == sizeof(float) &&
        dst->nb[0] == sizeof(float) && ggml_cuda_dense_parity_q4k_selected(src0->name);
#endif
}

void ggml_cuda_mul_mat_vec_q4k_q8k(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_ABORT("Q4_K x Q8_K dense matvec is only implemented for CUDA");
#else
    GGML_ASSERT(ggml_cuda_should_use_mmvq_q8k(src0, src1, dst));

    const int nblocks = (int) (src0->ne[0] / QK_K);
    ggml_cuda_pool_alloc<block_q8_K> input_q8k_local(ctx.pool());
    cudaStream_t stream = ctx.stream();

    bool reuse_hit = false;
    block_q8_K * input_q8k = (block_q8_K *) ggml_cuda_activation_quant_reuse_acquire(
        ctx,
        src1,
        GGML_CUDA_ACTIVATION_QUANT_Q8_K,
        (size_t) nblocks * sizeof(block_q8_K),
        reuse_hit);
    if (input_q8k == nullptr) {
        input_q8k = input_q8k_local.alloc(nblocks);
    }
    if (!reuse_hit) {
        quantize_row_q8_K_cuda((const float *) src1->data, input_q8k, src0->ne[0], stream);
    } else {
        static std::atomic<bool> reuse_logged{false};
        if (!reuse_logged.exchange(true)) {
            std::fprintf(stderr, "ggml_cuda: byte-exact Q8_K activation quantization reuse active for %s\n", src1->name);
        }
    }

    const int warps_per_block = dense_q8k_warps_per_block();
    switch (warps_per_block) {
        case 2:
            launch_mul_mat_vec_q4k_q8k<2>(src0, input_q8k, nblocks, dst, stream);
            break;
        case 4:
            launch_mul_mat_vec_q4k_q8k<4>(src0, input_q8k, nblocks, dst, stream);
            break;
        case 8:
            launch_mul_mat_vec_q4k_q8k<8>(src0, input_q8k, nblocks, dst, stream);
            break;
        case 16:
            launch_mul_mat_vec_q4k_q8k<16>(src0, input_q8k, nblocks, dst, stream);
            break;
        default:
            GGML_ABORT("invalid Q4_K x Q8_K warps-per-block configuration");
    }

    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        std::fprintf(stderr, "ggml_cuda: Q4_K x Q8_K dense matvec active for %s\n", src0->name);
    }
    static std::atomic<bool> resident_logged{false};
    if (ggml_cuda_dense_parity_resident_selected(src0->name) && !resident_logged.exchange(true)) {
        std::fprintf(stderr, "ggml_cuda: Q4_K x Q8_K resident parity active for %s\n", src0->name);
    }
    static std::atomic<bool> launch_logged{false};
    if (warps_per_block != 8 && !launch_logged.exchange(true)) {
        std::fprintf(stderr, "ggml_cuda: Q4_K x Q8_K launch uses %d warps per block\n", warps_per_block);
    }
#endif
}
