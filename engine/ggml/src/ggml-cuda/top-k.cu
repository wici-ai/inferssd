#include "argsort.cuh"
#include "top-k.cuh"

#include <cfloat>
#include <climits>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

template<int ncols>
static __global__ void top_k_warp_f32_i32(const float * src, int * dst, const int k, const int nrows) {
    constexpr int experts_per_thread = (ncols + WARP_SIZE - 1) / WARP_SIZE;

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows) {
        return;
    }

    const int lane = threadIdx.x;
    src += row * ncols;
    dst += row * k;

    float vals[experts_per_thread];
    uint32_t active_mask = 0;

#pragma unroll
    for (int i = 0; i < experts_per_thread; ++i) {
        const int idx = lane + i * WARP_SIZE;
        const bool active = idx < ncols;
        if (active) {
            active_mask |= 1u << i;
        }
        float val = active ? src[idx] : -INFINITY;
        vals[i] = __isnanf(val) ? -FLT_MAX : val;
    }

    for (int out = 0; out < k; ++out) {
        float max_val = -INFINITY;
        int   max_idx = INT_MAX;

#pragma unroll
        for (int i = 0; i < experts_per_thread; ++i) {
            const int idx = lane + i * WARP_SIZE;
            if (((active_mask >> i) & 1u) && (vals[i] > max_val || (vals[i] == max_val && idx < max_idx))) {
                max_val = vals[i];
                max_idx = idx;
            }
        }

#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            const float other_val = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
            const int   other_idx = __shfl_xor_sync(0xFFFFFFFF, max_idx, mask, WARP_SIZE);
            if (other_val > max_val || (other_val == max_val && other_idx < max_idx)) {
                max_val = other_val;
                max_idx = other_idx;
            }
        }

        if (lane == out) {
            dst[out] = max_idx;
        }

        if (max_idx < ncols && (max_idx & (WARP_SIZE - 1)) == lane) {
            active_mask &= ~(1u << (max_idx / WARP_SIZE));
        }

        // Volta and newer GPUs schedule warp lanes independently.  Make sure
        // the winning lane has removed this candidate before any lane starts
        // the next selection round, otherwise the top-k order can vary.
        __syncwarp();
    }
}

static bool kimi_pipeline_top_k_enabled() {
    return false; // Frozen production profile: historical unset default.
}

bool ggml_cuda_kimi_pipeline_top_k_self_test(char * digest, size_t digest_size) {
    constexpr int ncols = 896;
    constexpr int nrows = 4;
    constexpr int k = 16;
    constexpr int repeats = 32;

    std::vector<float> src((size_t) ncols * nrows);
    for (int row = 0; row < nrows; ++row) {
        for (int col = 0; col < ncols; ++col) {
            // Repeated values exercise the expert-id tie break.  NaNs exercise
            // the same canonicalization used by the device implementation.
            float value = (float) (((col * 17 + row * 29) % 41) - 20);
            if ((col + row * 7) % 223 == 0) {
                value = NAN;
            }
            src[(size_t) row * ncols + col] = value;
        }
    }

    std::vector<int> expected((size_t) nrows * k);
    for (int row = 0; row < nrows; ++row) {
        std::vector<int> ids(ncols);
        std::iota(ids.begin(), ids.end(), 0);
        std::sort(ids.begin(), ids.end(), [&](int a, int b) {
            float va = src[(size_t) row * ncols + a];
            float vb = src[(size_t) row * ncols + b];
            va = std::isnan(va) ? -FLT_MAX : va;
            vb = std::isnan(vb) ? -FLT_MAX : vb;
            return va != vb ? va > vb : a < b;
        });
        std::copy_n(ids.begin(), k, expected.begin() + (size_t) row * k);
    }

    float * d_src = nullptr;
    int * d_dst = nullptr;
    cudaStream_t stream = nullptr;
    bool ok = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess &&
        cudaMalloc((void **) &d_src, src.size() * sizeof(float)) == cudaSuccess &&
        cudaMalloc((void **) &d_dst, expected.size() * sizeof(int)) == cudaSuccess;
    std::vector<int> actual(expected.size());
    uint64_t fnv = UINT64_C(1469598103934665603);

    if (ok) {
        ok = cudaMemcpyAsync(d_src, src.data(), src.size() * sizeof(float),
                cudaMemcpyHostToDevice, stream) == cudaSuccess;
    }
    for (int repeat = 0; ok && repeat < repeats; ++repeat) {
        const dim3 grid(1, 1, 1);
        const dim3 block(WARP_SIZE, nrows, 1);
        top_k_warp_f32_i32<ncols><<<grid, block, 0, stream>>>(d_src, d_dst, k, nrows);
        ok = cudaGetLastError() == cudaSuccess &&
            cudaMemcpyAsync(actual.data(), d_dst, actual.size() * sizeof(int),
                cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
            cudaStreamSynchronize(stream) == cudaSuccess &&
            actual == expected;
        if (ok) {
            const unsigned char * bytes = (const unsigned char *) actual.data();
            for (size_t i = 0; i < actual.size() * sizeof(int); ++i) {
                fnv = (fnv ^ bytes[i]) * UINT64_C(1099511628211);
            }
        }
    }

    if (d_dst) cudaFree(d_dst);
    if (d_src) cudaFree(d_src);
    if (stream) cudaStreamDestroy(stream);
    if (digest && digest_size > 0) {
        std::snprintf(digest, digest_size, "%016llx", (unsigned long long) fnv);
    }
    return ok;
}

// Returns whether a warp per row takes the top k; with launch == false it only answers that.
static bool top_k_warp(const float * src, int * dst, const int ncols, const int nrows, const int k, cudaStream_t stream,
                       const bool launch = true) {
    if (k <= 0 || k > WARP_SIZE) {
        return false;
    }

    constexpr int rows_per_block = 4;
    const dim3 grid((nrows + rows_per_block - 1) / rows_per_block, 1, 1);
    const dim3 block(WARP_SIZE, rows_per_block, 1);

    switch (ncols) {
        case 128:
            if (launch) {
                top_k_warp_f32_i32<128><<<grid, block, 0, stream>>>(src, dst, k, nrows);
            }
            return true;
        case 256:
            if (launch) {
                top_k_warp_f32_i32<256><<<grid, block, 0, stream>>>(src, dst, k, nrows);
            }
            return true;
        case 384:
            if (launch) {
                top_k_warp_f32_i32<384><<<grid, block, 0, stream>>>(src, dst, k, nrows);
            }
            return true;
        case 512:
            if (launch) {
                top_k_warp_f32_i32<512><<<grid, block, 0, stream>>>(src, dst, k, nrows);
            }
            return true;
        case 576:
            if (launch) {
                top_k_warp_f32_i32<576><<<grid, block, 0, stream>>>(src, dst, k, nrows);
            }
            return true;
        case 896:
            // Private canonical for the Task 132 K3 spike.  With the guard
            // absent, the legacy 896-column path remains byte-for-byte on its
            // former CUB fallback.
            if (k != 16 || !kimi_pipeline_top_k_enabled()) {
                return false;
            }
            if (!launch) {
                return true;
            }
            top_k_warp_f32_i32<896><<<grid, block, 0, stream>>>(src, dst, k, nrows);
            {
                static std::atomic<bool> reported{false};
                if (!reported.exchange(true)) {
                    std::fprintf(stderr,
                            "[kimi_pipeline_m1] deterministic top-k active: ncols=896 k=16 tie=expert_id_ascending\n");
                }
            }
            return true;
        default:
            return false;
    }
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
    if (top_k_warp(src0_d, dst_d, ncols, nrows, k, stream)) {
        return;
    }
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    if (shared_mem > max_shared_mem || ncols > 1024) {
        argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    } else {
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    }
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}

size_t ggml_cuda_op_top_k_pool_size(const ggml_tensor * dst) {
    const ggml_tensor * src0  = dst->src[0];
    const int64_t       ncols = src0->ne[0];
    const int64_t       nrows = ggml_nrows(src0);
    const int64_t       k     = dst->ne[0];
    if (top_k_warp(nullptr, nullptr, ncols, nrows, k, cudaStream_t{}, /*launch=*/false)) {
        return 0;
    }
#ifdef CUB_TOP_K_AVAILABLE
    GGML_UNUSED(nrows);
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto env          = cuda::std::execution::env{ cuda::stream_ref{ cudaStream_t{} }, requirements };
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, (const float *) nullptr, cuda::discard_iterator(),
                         cuda::make_counting_iterator(0), (int *) nullptr, ncols, k, env));
    return ggml_cuda_pool_alloc_size(temp_storage_bytes);
#elif defined(GGML_CUDA_USE_CUB)
    GGML_UNUSED(k);
    size_t       size           = ggml_cuda_pool_alloc_size(ncols * nrows * sizeof(int));
    const size_t shared_mem     = next_power_of_2(ncols) * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    if (shared_mem > max_shared_mem || ncols > 1024) {
        size += argsort_f32_i32_cuda_cub_pool_size(ncols, nrows, GGML_SORT_ORDER_DESC);
    }
    return size;
#else
    GGML_UNUSED(k);
    return ggml_cuda_pool_alloc_size(ncols * nrows * sizeof(int));
#endif
}
