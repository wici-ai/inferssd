#include "moe-adapter-dispatch.h"
#include "moe-expert-descriptor.h"
#include "moe-device-descriptor.h"

#ifndef GGML_CUDA_MOE_STREAM

#include <cstddef>
#include <cstdint>

extern "C" {
typedef struct { int32_t i1; int32_t i2; } ggml_moe_decode_row_mapping;

bool ggml_cuda_moe_stream_decode_batch(
    int, const char *, const void *, int64_t, int64_t, int64_t, size_t, size_t,
    const float *, int64_t, size_t, size_t, float *, size_t, size_t,
    int64_t *, const ggml_moe_decode_row_mapping *, int64_t) {
    return false;
}
}

#else

#include "common.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <cuda_runtime.h>

extern "C" {

typedef struct {
    int32_t i1;
    int32_t i2;
} ggml_moe_decode_row_mapping;

bool ggml_cuda_moe_common_stream_decode_batch(
    int, const char *, const void *, int64_t, int64_t, int64_t, size_t,
    size_t, const float *, int64_t, size_t, size_t, float *, size_t, size_t,
    int64_t *, const ggml_moe_decode_row_mapping *, int64_t);

bool ggml_cuda_moe_stream_available(void);
const void * ggml_cuda_moe_stream_one_cache_dev_ptr_or_insert(
    const char * src0_name,
    const void * src0_base_data,
    size_t nb02,
    size_t expert_bytes,
    int64_t expert_idx,
    cudaStream_t stream,
    int * cache_state);
bool ggml_cuda_moe_stream_one_cache_dev_ptrs_or_insert(
    const char * src0_name,
    const void * src0_base_data,
    size_t nb02,
    size_t expert_bytes,
    const int * expert_indices,
    const int32_t * route_ranks,
    int count,
    cudaStream_t stream,
    const void ** output_ptrs,
    int * cache_states,
    ggml_moe_expert_descriptor * descriptors);
void ggml_cuda_moe_stream_one_gate_up_prefetch(
    const char * src0_name,
    size_t expert_bytes,
    const int * expert_indices,
    const int32_t * route_ranks,
    int count);
void ggml_cuda_moe_stream_one_cache_release_descriptors(
    const ggml_moe_expert_descriptor * descriptors,
    int count,
    cudaStream_t stream);
void ggml_cuda_moe_stream_one_cache_release_descriptors_synchronized(
    const ggml_moe_expert_descriptor * descriptors,
    int count);

}

namespace {

constexpr int DECODE_BATCH_MAX_ACTIVE = 8;
constexpr int DECODE_BATCH_CACHE_FROZEN_MISS = 5;

struct decode_batch_ctx {
    cudaStream_t stream = nullptr;
    void * d_src0_stage = nullptr;
    size_t d_src0_stage_sz = 0;
    void * d_src1_f32 = nullptr;
    size_t d_src1_f32_sz = 0;
    void * d_src1_q8 = nullptr;
    size_t d_src1_q8_sz = 0;
    void * d_dst = nullptr;
    size_t d_dst_sz = 0;
    ggml_moe::device_descriptor_buffer descriptor_buffer;
    void * h_src1 = nullptr;
    size_t h_src1_sz = 0;
    void * h_dst = nullptr;
    size_t h_dst_sz = 0;
    cudaEvent_t profile_input_start = nullptr;
    cudaEvent_t profile_input_done = nullptr;
    cudaEvent_t profile_kernel_done = nullptr;
    cudaEvent_t profile_d2h_done = nullptr;
};

struct descriptor_lease_guard {
    ggml_moe_expert_descriptor * descriptors = nullptr;
    int count = 0;
    cudaStream_t stream = nullptr;
    bool active = false;

    ~descriptor_lease_guard() {
        if (active) {
            ggml_cuda_moe_stream_one_cache_release_descriptors(descriptors, count, stream);
        }
    }

    void release_after_queued_work() {
        if (!active) return;
        ggml_cuda_moe_stream_one_cache_release_descriptors(descriptors, count, stream);
        active = false;
    }

    void release_after_synchronized_work() {
        if (!active) return;
        ggml_cuda_moe_stream_one_cache_release_descriptors_synchronized(descriptors, count);
        active = false;
    }
};

decode_batch_ctx g_decode_batch;
std::mutex g_decode_batch_mu;
std::atomic<int> g_decode_batch_first{0};
std::atomic<uint64_t> g_decode_batch_calls{0};
std::atomic<uint64_t> g_decode_batch_cache_hits{0};
std::atomic<uint64_t> g_decode_batch_ram_promotes{0};
std::atomic<uint64_t> g_decode_batch_cache_inserts{0};
std::atomic<uint64_t> g_decode_batch_pack_inserts{0};
std::atomic<uint64_t> g_decode_batch_staged{0};
std::atomic<uint64_t> g_decode_batch_descriptor_batches{0};
std::atomic<uint64_t> g_decode_batch_descriptor_fallback_batches{0};
std::atomic<uint64_t> g_decode_batch_descriptor_fallback_routes{0};
std::atomic<bool> g_decode_batch_lifecycle_registered{false};

static void decode_batch_lifecycle_report() {
    std::fprintf(stderr,
        "[moe_stream_decode_batch] descriptor lifecycle: batches=%lu "
        "descriptor_batches=%lu fallback_batches=%lu fallback_routes=%lu "
        "staged_routes=%lu\n",
        (unsigned long)g_decode_batch_calls.load(std::memory_order_relaxed),
        (unsigned long)g_decode_batch_descriptor_batches.load(std::memory_order_relaxed),
        (unsigned long)g_decode_batch_descriptor_fallback_batches.load(std::memory_order_relaxed),
        (unsigned long)g_decode_batch_descriptor_fallback_routes.load(std::memory_order_relaxed),
        (unsigned long)g_decode_batch_staged.load(std::memory_order_relaxed));
}

struct decode_batch_profile_role {
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> cache_ns{0};
    std::atomic<uint64_t> host_prepare_ns{0};
    std::atomic<uint64_t> enqueue_ns{0};
    std::atomic<uint64_t> sync_ns{0};
    std::atomic<uint64_t> scatter_ns{0};
    std::atomic<uint64_t> total_ns{0};
    std::atomic<uint64_t> input_h2d_us{0};
    std::atomic<uint64_t> kernel_us{0};
    std::atomic<uint64_t> d2h_us{0};
    std::atomic<uint64_t> input_h2d_bytes{0};
    std::atomic<uint64_t> descriptor_h2d_bytes{0};
    std::atomic<uint64_t> d2h_bytes{0};
};

decode_batch_profile_role g_decode_batch_profile[3];
std::atomic<bool> g_decode_batch_profile_registered{false};

int decode_batch_profile_role_index(const char * name) {
    if (name && std::strstr(name, "ffn_gate_exps")) return 0;
    if (name && std::strstr(name, "ffn_up_exps")) return 1;
    if (name && std::strstr(name, "ffn_down_exps")) return 2;
    return -1;
}

void decode_batch_profile_report() {
    static const char * names[3] = {"gate", "up", "down"};
    for (int role = 0; role < 3; ++role) {
        const uint64_t calls = g_decode_batch_profile[role].calls.load();
        if (calls == 0) continue;
        auto ms = [](uint64_t ns) { return ns / 1000000.0; };
        std::fprintf(stderr,
            "[moe_stream_decode_batch_profile] role=%s calls=%lu cache_ms=%.3f host_prepare_ms=%.3f "
            "enqueue_ms=%.3f sync_ms=%.3f scatter_ms=%.3f total_ms=%.3f avg_total_us=%.3f "
            "input_h2d_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f input_h2d_bytes=%lu "
            "descriptor_h2d_bytes=%lu d2h_bytes=%lu\n",
            names[role], calls,
            ms(g_decode_batch_profile[role].cache_ns.load()),
            ms(g_decode_batch_profile[role].host_prepare_ns.load()),
            ms(g_decode_batch_profile[role].enqueue_ns.load()),
            ms(g_decode_batch_profile[role].sync_ns.load()),
            ms(g_decode_batch_profile[role].scatter_ns.load()),
            ms(g_decode_batch_profile[role].total_ns.load()),
            g_decode_batch_profile[role].total_ns.load() / (1000.0 * calls),
            g_decode_batch_profile[role].input_h2d_us.load() / 1000.0,
            g_decode_batch_profile[role].kernel_us.load() / 1000.0,
            g_decode_batch_profile[role].d2h_us.load() / 1000.0,
            g_decode_batch_profile[role].input_h2d_bytes.load(),
            g_decode_batch_profile[role].descriptor_h2d_bytes.load(),
            g_decode_batch_profile[role].d2h_bytes.load());
    }
}

bool decode_batch_profile_enabled() {
    const char * env = std::getenv("GGML_MOE_STREAM_DECODE_BATCH_PROFILE");
    if (!env || !env[0] || env[0] == '0') return false;
    if (!g_decode_batch_profile_registered.exchange(true)) std::atexit(decode_batch_profile_report);
    return true;
}

bool decode_batch_profile_events_init(decode_batch_ctx & ctx) {
    if (ctx.profile_input_start) return true;
    cudaEvent_t * events[] = {
        &ctx.profile_input_start,
        &ctx.profile_input_done,
        &ctx.profile_kernel_done,
        &ctx.profile_d2h_done,
    };
    for (cudaEvent_t * event : events) {
        if (cudaEventCreate(event) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
    }
    return true;
}

bool env_enabled(const char * name, bool fallback) {
    const char * env = std::getenv(name);
    if (!env || !env[0]) {
        return fallback;
    }
    return env[0] != '0';
}

bool ensure_dev(void *& ptr, size_t & cur, size_t need) {
    if (cur >= need) {
        return true;
    }
    if (ptr) {
        cudaFree(ptr);
    }
    ptr = nullptr;
    cur = 0;
    if (need == 0) {
        return true;
    }
    if (cudaMalloc(&ptr, need) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    cur = need;
    return true;
}

bool ensure_host_pinned(void *& ptr, size_t & cur, size_t need) {
    if (cur >= need) {
        return true;
    }
    if (ptr) {
        cudaFreeHost(ptr);
    }
    ptr = nullptr;
    cur = 0;
    if (need == 0) {
        return true;
    }
    if (cudaHostAlloc(&ptr, need, cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    cur = need;
    return true;
}

bool init_decode_batch_ctx(decode_batch_ctx & ctx) {
    if (ctx.stream) {
        return ctx.descriptor_buffer.device != nullptr;
    }
    if (cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking) != cudaSuccess) {
        cudaGetLastError();
        ctx.stream = nullptr;
        return false;
    }
    if (!ggml_moe::device_descriptor_buffer_allocate_fixed(
            ctx.descriptor_buffer, DECODE_BATCH_MAX_ACTIVE)) {
        cudaStreamDestroy(ctx.stream);
        ctx.stream = nullptr;
        return false;
    }
    std::fprintf(stderr,
        "[moe_stream_decode_batch] fixed device descriptor: capacity=%d bytes=%zu\n",
        DECODE_BATCH_MAX_ACTIVE, ctx.descriptor_buffer.bytes);
    return true;
}

bool name_filter_matches_any(const char * filter, const char * name) {
    if (!filter || !filter[0] || !name || !name[0]) {
        return false;
    }
    const char * p = filter;
    while (*p) {
        while (*p == ' ' || *p == '	' || *p == ',' || *p == ':') {
            ++p;
        }
        const char * start = p;
        while (*p && *p != ',' && *p != ':') {
            ++p;
        }
        const char * end = p;
        while (end > start && (end[-1] == ' ' || end[-1] == '	')) {
            --end;
        }
        const size_t len = (size_t) (end - start);
        if (len > 0) {
            for (const char * hit = name; *hit; ++hit) {
                if (std::strncmp(hit, start, len) == 0) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool tensor_name_supported(const char * name) {
    if (!name || !name[0]) {
        return false;
    }
    const char * filter = std::getenv("GGML_MOE_STREAM_DECODE_BATCH_NAME_FILTER");
    if (!filter || !filter[0]) {
        filter = std::getenv("GGML_MOE_STREAM_ONE_NAME_FILTER");
    }
    if (filter && filter[0]) {
        return name_filter_matches_any(filter, name);
    }
    return std::strstr(name, "ffn_gate_exps") ||
        std::strstr(name, "ffn_up_exps") ||
        std::strstr(name, "ffn_down_exps");
}

__global__ void moe_decode_mxfp4_ptr_batch_kernel(
        const ggml_moe_expert_descriptor * __restrict__ descriptors,
        const block_q8_1 * __restrict__ y,
        float * __restrict__ dst,
        uint32_t ncols_x,
        uint32_t nrows_x,
        uint32_t stride_row_x,
        uint32_t stride_col_y,
        uint32_t stride_col_dst,
        uint32_t n_active) {
    constexpr ggml_type type = GGML_TYPE_MXFP4;
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = VDR_MXFP4_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int rows_per_block = 2;

    const uint32_t active = threadIdx.y;
    if (active >= n_active) {
        return;
    }

    const uint32_t row0 = rows_per_block * blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * warp_size / qi;
    const void * x = descriptors[active].role_ptr;
    const block_q8_1 * y_row = y + active * stride_col_y;
    const int kbx_offset = row0 * stride_row_x;

    float tmp[rows_per_block] = {0.0f, 0.0f};
    for (int kbx = threadIdx.x / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi / vdr));
#pragma unroll
        for (int i = 0; i < rows_per_block; ++i) {
            tmp[i] += vec_dot_mxfp4_q8_1(x, &y_row[kby], kbx_offset + i * stride_row_x + kbx, kqs);
        }
    }

#pragma unroll
    for (int i = 0; i < rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    if (threadIdx.x < rows_per_block && row0 + threadIdx.x < nrows_x) {
        dst[active * stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

bool launch_mxfp4_decode_batch(
        const ggml_moe_expert_descriptor * d_expert_descriptors,
        const void * d_src1_q8,
        void * d_dst,
        int64_t ne00,
        int64_t ne01,
        size_t nb01,
        int n_active,
        cudaStream_t stream) {
    constexpr ggml_type type = GGML_TYPE_MXFP4;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    if (ne00 <= 0 || ne01 <= 0 || n_active <= 0 || n_active > DECODE_BATCH_MAX_ACTIVE) {
        return false;
    }
    if ((ne00 % ggml_blck_size(type)) != 0 || (nb01 % ggml_type_size(type)) != 0) {
        return false;
    }
    const int64_t src1_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    const uint32_t stride_row_x = (uint32_t) (nb01 / ggml_type_size(type));
    const uint32_t stride_col_y = (uint32_t) (src1_padded / QK8_1);
    const uint32_t stride_col_dst = (uint32_t) ne01;
    const dim3 grid((uint32_t)((ne01 + 1) / 2), 1, 1);
    const dim3 block(warp_size, (uint32_t)n_active, 1);

    moe_decode_mxfp4_ptr_batch_kernel<<<grid, block, 0, stream>>>(
            d_expert_descriptors,
            (const block_q8_1 *) d_src1_q8,
            (float *) d_dst,
            (uint32_t) ne00,
            (uint32_t) ne01,
            stride_row_x,
            stride_col_y,
            stride_col_dst,
            (uint32_t) n_active);
    return cudaGetLastError() == cudaSuccess;
}

__device__ __forceinline__ int ds4_mxfp4_value(int q) {
    constexpr int values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    return values[q & 0x0F];
}

__device__ __forceinline__ int ds4_pack_mxfp4_4(const uint8_t * qs, int offset, bool high) {
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint8_t q = qs[offset + i];
        const int value = ds4_mxfp4_value(high ? q >> 4 : q);
        packed |= (uint32_t) (uint8_t) (int8_t) value << (8 * i);
    }
    return (int) packed;
}

__device__ __forceinline__ int ds4_pack_q80_4(const int8_t * qs, int offset) {
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        packed |= (uint32_t) (uint8_t) qs[offset + i] << (8 * i);
    }
    return (int) packed;
}

__global__ void ds4_mxfp4_q80_cpu_order_ptr_batch_kernel(
        const ggml_moe_expert_descriptor * __restrict__ descriptors,
        int64_t ne00,
        int64_t ne01,
        int64_t nb01,
        const block_q8_0 * __restrict__ q80,
        size_t q80_blocks_per_row,
        float * __restrict__ dst,
        int64_t n_active) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = n_active * ne01;
    if (idx >= total) return;

    const int64_t active = idx / ne01;
    const int64_t col = idx - active * ne01;
    const char * expert = (const char *) descriptors[active].role_ptr;
    const block_mxfp4 * x = (const block_mxfp4 *) (expert + (size_t) col * (size_t) nb01);
    const block_q8_0 * y = q80 + (size_t) active * q80_blocks_per_row;
    const int64_t blocks = ne00 / QK_MXFP4;

    float accum1[8] = {};
    float accum2[8] = {};
    int64_t ib = 0;
    for (; ib + 1 < blocks; ib += 2) {
#pragma unroll
        for (int lane = 0; lane < 8; ++lane) {
            const int base = lane * 4;
            int p1 = 0;
            int p2 = 0;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int qidx = base + t;
                const int xidx = qidx & 15;
                const uint8_t q1 = x[ib + 0].qs[xidx];
                const uint8_t q2 = x[ib + 1].qs[xidx];
                const int v1 = qidx < 16 ? ds4_mxfp4_value(q1) : ds4_mxfp4_value(q1 >> 4);
                const int v2 = qidx < 16 ? ds4_mxfp4_value(q2) : ds4_mxfp4_value(q2 >> 4);
                p1 += y[ib + 0].qs[qidx] * v1;
                p2 += y[ib + 1].qs[qidx] * v2;
            }
            const float scale1 = __fmul_rn(__half2float(y[ib + 0].d), ggml_cuda_e8m0_to_fp32(x[ib + 0].e) * 0.5f);
            const float scale2 = __fmul_rn(__half2float(y[ib + 1].d), ggml_cuda_e8m0_to_fp32(x[ib + 1].e) * 0.5f);
            accum1[lane] = __fmaf_rn(scale1, (float) p1, accum1[lane]);
            accum2[lane] = __fmaf_rn(scale2, (float) p2, accum2[lane]);
        }
    }
    float sum8[8];
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) sum8[lane] = __fadd_rn(accum1[lane], accum2[lane]);
    float r0 = __fadd_rn(sum8[4], sum8[0]);
    float r1 = __fadd_rn(sum8[5], sum8[1]);
    float r2 = __fadd_rn(sum8[6], sum8[2]);
    float r3 = __fadd_rn(sum8[7], sum8[3]);
    r0 = __fadd_rn(r0, r2);
    r1 = __fadd_rn(r1, r3);
    float sum = __fadd_rn(r0, r1);
    for (; ib < blocks; ++ib) {
        int p1 = 0;
        int p2 = 0;
#pragma unroll
        for (int j = 0; j < QK_MXFP4 / 2; ++j) {
            const uint8_t q = x[ib].qs[j];
            p1 += y[ib].qs[j] * ds4_mxfp4_value(q);
            p2 += y[ib].qs[j + QK_MXFP4 / 2] * ds4_mxfp4_value(q >> 4);
        }
        const float scale = __fmul_rn(__half2float(y[ib].d), ggml_cuda_e8m0_to_fp32(x[ib].e) * 0.5f);
        sum = __fadd_rn(sum, __fmul_rn(scale, (float) (p1 + p2)));
    }
    dst[(size_t) active * (size_t) ne01 + (size_t) col] = sum;
}

__global__ void ds4_mxfp4_q80_cpu_order_lane8_ptr_batch_kernel(
        const ggml_moe_expert_descriptor * __restrict__ descriptors,
        int64_t ne00,
        int64_t ne01,
        int64_t nb01,
        const block_q8_0 * __restrict__ q80,
        size_t q80_blocks_per_row,
        float * __restrict__ dst,
        int64_t n_active,
        bool cache_q80) {
    extern __shared__ char shared_q80[];
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 7;
    const int64_t output = tid >> 3;
    const int64_t total = n_active * ne01;
    if (output >= total) return;

    const int64_t active = output / ne01;
    const int64_t col = output - active * ne01;
    const int64_t outputs_per_block = blockDim.x >> 3;
    const int64_t first_output = (int64_t) blockIdx.x * outputs_per_block;
    const int64_t last_candidate = first_output + outputs_per_block - 1;
    const int64_t last_output = last_candidate < total ? last_candidate : total - 1;
    const int64_t block_active = first_output / ne01;
    const bool same_active = cache_q80 && block_active == last_output / ne01;
    const size_t q80_row_size = q80_blocks_per_row * sizeof(block_q8_0);
    if (same_active) {
        const char * source = (const char *) (q80 + (size_t) block_active * q80_blocks_per_row);
        for (size_t offset = threadIdx.x; offset < q80_row_size; offset += blockDim.x) {
            shared_q80[offset] = source[offset];
        }
    }
    __syncthreads();

    const char * expert = (const char *) descriptors[active].role_ptr;
    const block_mxfp4 * x = (const block_mxfp4 *) (expert + (size_t) col * (size_t) nb01);
    const block_q8_0 * y = same_active ?
        (const block_q8_0 *) shared_q80 : q80 + (size_t) active * q80_blocks_per_row;
    const int64_t blocks = ne00 / QK_MXFP4;

    float accum1 = 0.0f;
    float accum2 = 0.0f;
    int64_t ib = 0;
    const int base = lane * 4;
    const int group_base = (threadIdx.x & 31) & ~7;
    const unsigned mask = 0xFFu << group_base;
    for (; ib + 1 < blocks; ib += 2) {
        const int xoffset = base & 15;
        const bool high = base >= 16;
        const int p1 = ggml_cuda_dp4a(
            ds4_pack_mxfp4_4(x[ib + 0].qs, xoffset, high),
            ds4_pack_q80_4(y[ib + 0].qs, base), 0);
        const int p2 = ggml_cuda_dp4a(
            ds4_pack_mxfp4_4(x[ib + 1].qs, xoffset, high),
            ds4_pack_q80_4(y[ib + 1].qs, base), 0);
        float scale1 = lane == 0 ?
            __fmul_rn(__half2float(y[ib + 0].d), ggml_cuda_e8m0_to_fp32(x[ib + 0].e) * 0.5f) : 0.0f;
        float scale2 = lane == 0 ?
            __fmul_rn(__half2float(y[ib + 1].d), ggml_cuda_e8m0_to_fp32(x[ib + 1].e) * 0.5f) : 0.0f;
        scale1 = __shfl_sync(mask, scale1, group_base);
        scale2 = __shfl_sync(mask, scale2, group_base);
        accum1 = __fmaf_rn(scale1, (float) p1, accum1);
        accum2 = __fmaf_rn(scale2, (float) p2, accum2);
    }

    const float lane_sum = __fadd_rn(accum1, accum2);
    const float s0 = __shfl_sync(mask, lane_sum, group_base + 0);
    const float s1 = __shfl_sync(mask, lane_sum, group_base + 1);
    const float s2 = __shfl_sync(mask, lane_sum, group_base + 2);
    const float s3 = __shfl_sync(mask, lane_sum, group_base + 3);
    const float s4 = __shfl_sync(mask, lane_sum, group_base + 4);
    const float s5 = __shfl_sync(mask, lane_sum, group_base + 5);
    const float s6 = __shfl_sync(mask, lane_sum, group_base + 6);
    const float s7 = __shfl_sync(mask, lane_sum, group_base + 7);
    if (lane == 0) {
        float r0 = __fadd_rn(s4, s0);
        float r1 = __fadd_rn(s5, s1);
        float r2 = __fadd_rn(s6, s2);
        float r3 = __fadd_rn(s7, s3);
        r0 = __fadd_rn(r0, r2);
        r1 = __fadd_rn(r1, r3);
        float sum = __fadd_rn(r0, r1);
        for (; ib < blocks; ++ib) {
            int p1 = 0;
            int p2 = 0;
#pragma unroll
            for (int j = 0; j < QK_MXFP4 / 2; ++j) {
                const uint8_t q = x[ib].qs[j];
                p1 += y[ib].qs[j] * ds4_mxfp4_value(q);
                p2 += y[ib].qs[j + QK_MXFP4 / 2] * ds4_mxfp4_value(q >> 4);
            }
            const float scale = __fmul_rn(__half2float(y[ib].d), ggml_cuda_e8m0_to_fp32(x[ib].e) * 0.5f);
            sum = __fadd_rn(sum, __fmul_rn(scale, (float) (p1 + p2)));
        }
        dst[(size_t) active * (size_t) ne01 + (size_t) col] = sum;
    }
}

bool launch_mxfp4_q80_cpu_order_decode_batch(
        const ggml_moe_expert_descriptor * d_expert_descriptors,
        const void * d_src1_q80,
        void * d_dst,
        int64_t ne00,
        int64_t ne01,
        size_t nb01,
        size_t q80_row_size,
        int n_active,
        cudaStream_t stream) {
    if (!d_expert_descriptors || !d_src1_q80 || !d_dst || ne00 <= 0 || ne01 <= 0 ||
            ne00 % QK_MXFP4 != 0 || q80_row_size != (size_t) (ne00 / QK8_0) * sizeof(block_q8_0)) {
        return false;
    }
    const int threads = 128;
    const int64_t total = (int64_t) n_active * ne01;
    if (env_enabled("GGML_MOE_STREAM_DOWN_Q80_CPU_ORDER_LANE8", false)) {
        const bool cache_q80 = env_enabled("GGML_MOE_STREAM_DOWN_Q80_CPU_ORDER_LANE8_SHARED", false);
        const int64_t total_threads = total * 8;
        ds4_mxfp4_q80_cpu_order_lane8_ptr_batch_kernel<<<
            (total_threads + threads - 1) / threads, threads, cache_q80 ? q80_row_size : 0, stream>>>(
            d_expert_descriptors, ne00, ne01, (int64_t) nb01,
            (const block_q8_0 *) d_src1_q80, q80_row_size / sizeof(block_q8_0),
            (float *) d_dst, n_active, cache_q80);
    } else {
        ds4_mxfp4_q80_cpu_order_ptr_batch_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
            d_expert_descriptors, ne00, ne01, (int64_t) nb01,
            (const block_q8_0 *) d_src1_q80, q80_row_size / sizeof(block_q8_0),
            (float *) d_dst, n_active);
    }
    return cudaGetLastError() == cudaSuccess;
}

} // namespace

extern "C" bool ggml_cuda_moe_stream_decode_batch(
    int src0_type_int,
    const char * src0_name,
    const void * src0_data,
    int64_t n_as,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    size_t nb02,
    const float * src1_f32,
    int64_t src1_ne1,
    size_t src1_nb1,
    size_t src1_nb2,
    const void * src1_q8_0,
    size_t src1_q8_0_row_size,
    float * dst,
    size_t dst_nb1,
    size_t dst_nb2,
    int64_t * matrix_row_counts,
    const ggml_moe_decode_row_mapping * rows,
    int64_t rows_stride) {
    const size_t expert_bytes = (size_t) ne01 * nb01;
    ggml_moe::adapter_select_model_shape(n_as, expert_bytes, "decode-batch");
    if (ggml_moe::adapter_uses_common()) {
        return ggml_cuda_moe_common_stream_decode_batch(
            src0_type_int, src0_name, src0_data, n_as, ne01, ne00, nb01,
            nb02, src1_f32, src1_ne1, src1_nb1, src1_nb2, dst, dst_nb1,
            dst_nb2, matrix_row_counts, rows, rows_stride);
    }
    if (!env_enabled("GGML_MOE_STREAM_DECODE_BATCH", true)) {
        return false;
    }
    if (!ggml_cuda_moe_stream_available()) {
        return false;
    }
    if (src0_type_int != GGML_TYPE_MXFP4 || !tensor_name_supported(src0_name)) {
        return false;
    }
    if (!src1_f32 || !dst || !matrix_row_counts || !rows || rows_stride <= 0 || rows_stride > DECODE_BATCH_MAX_ACTIVE) {
        return false;
    }
    if (n_as <= 0 || ne00 <= 0 || ne01 <= 0 || src1_ne1 <= 0 || nb01 == 0 || nb02 == 0) {
        return false;
    }

    int active_experts[DECODE_BATCH_MAX_ACTIVE] = {};
    int32_t dst_ids[DECODE_BATCH_MAX_ACTIVE] = {};
    int32_t token_ids[DECODE_BATCH_MAX_ACTIVE] = {};
    int n_active = 0;
    for (int64_t expert = 0; expert < n_as; ++expert) {
        const int64_t count = matrix_row_counts[expert];
        if (count <= 0) {
            continue;
        }
        if (n_active + count > DECODE_BATCH_MAX_ACTIVE) {
            return false;
        }
        const ggml_moe_decode_row_mapping * expert_rows = rows + expert * rows_stride;
        for (int64_t row = 0; row < count; ++row) {
            if (expert_rows[row].i1 < 0 || expert_rows[row].i2 < 0) {
                return false;
            }
            active_experts[n_active] = (int) expert;
            dst_ids[n_active] = expert_rows[row].i1;
            token_ids[n_active] = expert_rows[row].i2;
            ++n_active;
        }
    }
    if (n_active <= 0 || n_active > DECODE_BATCH_MAX_ACTIVE) {
        return false;
    }

    const int64_t src1_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    const bool down_q80_cpu_order = std::strstr(src0_name, "ffn_down_exps") != nullptr &&
        env_enabled("GGML_MOE_STREAM_DOWN_Q80_COMPAT_BATCH", false) &&
        src1_q8_0 != nullptr && src1_q8_0_row_size > 0;
    const bool down_q80_fast_mmvq = down_q80_cpu_order &&
        env_enabled("GGML_MOE_STREAM_DOWN_Q80_FAST_MMVQ", false);
    size_t src1_f32_bytes = 0;
    size_t src1_q8_bytes = 0;
    size_t dst_bytes = 0;
    const int routed_active = n_active;

    std::lock_guard<std::mutex> lock(g_decode_batch_mu);
    decode_batch_ctx & ctx = g_decode_batch;
    if (!init_decode_batch_ctx(ctx)) {
        return false;
    }
    if (!g_decode_batch_lifecycle_registered.exchange(true)) {
        std::atexit(decode_batch_lifecycle_report);
    }
    const bool profile = decode_batch_profile_enabled();
    const int profile_role = profile ? decode_batch_profile_role_index(src0_name) : -1;
    const bool profile_cuda = profile_role >= 0 && decode_batch_profile_events_init(ctx);
    const auto profile_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    int cache_hits = 0;
    int ram_promotes = 0;
    int cache_inserts = 0;
    int pack_inserts = 0;
    int staged = 0;
    int frozen_misses = 0;
    int n_gpu_active = 0;
    const void * expert_ptrs[DECODE_BATCH_MAX_ACTIVE] = {};
    ggml_moe_expert_descriptor expert_descriptors[DECODE_BATCH_MAX_ACTIVE] = {};
    for (int j = 0; j < DECODE_BATCH_MAX_ACTIVE; ++j) expert_descriptors[j].slot_id = -1;
    int batch_cache_states[DECODE_BATCH_MAX_ACTIVE] = {};
    const bool stage_misses = env_enabled("GGML_MOE_STREAM_DECODE_BATCH_STAGE_MISS", true);
    const bool cache_only = env_enabled("GGML_MOE_STREAM_DECODE_BATCH_CACHE_ONLY", false);
    const bool use_one_cache = env_enabled("GGML_MOE_STREAM_DECODE_BATCH_ONE_CACHE", true);
    const bool allow_unleased_cache_fallback =
        env_enabled("GGML_MOE_STREAM_DECODE_BATCH_UNLEASED_CACHE_FALLBACK", false);
    const auto cache_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ggml_cuda_moe_stream_one_gate_up_prefetch(
        src0_name, expert_bytes, active_experts, dst_ids, n_active);
    const bool batch_cache_done = use_one_cache && ggml_cuda_moe_stream_one_cache_dev_ptrs_or_insert(
        src0_name, src0_data, nb02, expert_bytes, active_experts, dst_ids, n_active,
        ctx.stream, expert_ptrs, batch_cache_states, expert_descriptors);
    if (batch_cache_done) {
        g_decode_batch_descriptor_batches.fetch_add(1, std::memory_order_relaxed);
    } else if (use_one_cache) {
        g_decode_batch_descriptor_fallback_batches.fetch_add(1, std::memory_order_relaxed);
        g_decode_batch_descriptor_fallback_routes.fetch_add((uint64_t)n_active, std::memory_order_relaxed);
        for (int j = 0; j < n_active; ++j) {
            expert_ptrs[j] = nullptr;
            expert_descriptors[j] = {};
            expert_descriptors[j].slot_id = -1;
        }
    }
    const auto cache_done = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    descriptor_lease_guard lease_guard{
        expert_descriptors, n_active, ctx.stream, batch_cache_done
    };
    for (int j = 0; j < n_active; ++j) {
        int cache_state = batch_cache_done ? batch_cache_states[j] : 0;
        const void * expert_ptr = batch_cache_done ? expert_ptrs[j] :
            (allow_unleased_cache_fallback && use_one_cache) ?
            ggml_cuda_moe_stream_one_cache_dev_ptr_or_insert(
                src0_name, src0_data, nb02, expert_bytes, active_experts[j], ctx.stream, &cache_state) : nullptr;
        if (expert_ptr) {
            if (cache_state == 1) {
                ++cache_hits;
            } else if (cache_state == 2) {
                ++ram_promotes;
            } else if (cache_state == 3) {
                ++cache_inserts;
            } else if (cache_state == 4) {
                ++pack_inserts;
            } else {
                ++cache_hits;
            }
        } else if (cache_state == DECODE_BATCH_CACHE_FROZEN_MISS) {
            ++frozen_misses;
            continue;
        } else if (!stage_misses || cache_only || !src0_data) {
            return false;
        } else {
            ++staged;
        }

        active_experts[n_gpu_active] = active_experts[j];
        dst_ids[n_gpu_active] = dst_ids[j];
        token_ids[n_gpu_active] = token_ids[j];
        expert_ptrs[n_gpu_active] = expert_ptr;
        expert_descriptors[n_gpu_active] = expert_descriptors[j];
        expert_descriptors[n_gpu_active].role_ptr = expert_ptr;
        ++n_gpu_active;
    }
    lease_guard.count = n_gpu_active;

    if (n_gpu_active == 0) {
        return false;
    }
    n_active = n_gpu_active;
    src1_f32_bytes = down_q80_cpu_order ? 0 : (size_t) n_active * (size_t) ne00 * sizeof(float);
    src1_q8_bytes = down_q80_fast_mmvq ?
        (size_t) n_active * (size_t) (src1_padded / QK8_1) * sizeof(block_q8_1) :
        down_q80_cpu_order ? (size_t) n_active * src1_q8_0_row_size :
        (size_t) n_active * (size_t) (src1_padded / QK8_1) * sizeof(block_q8_1);
    dst_bytes = (size_t) n_active * (size_t) ne01 * sizeof(float);

    if (!ensure_dev(ctx.d_src0_stage, ctx.d_src0_stage_sz, (size_t) staged * expert_bytes) ||
        !ensure_dev(ctx.d_src1_f32, ctx.d_src1_f32_sz, src1_f32_bytes) ||
        !ensure_dev(ctx.d_src1_q8, ctx.d_src1_q8_sz, src1_q8_bytes) ||
        !ensure_dev(ctx.d_dst, ctx.d_dst_sz, dst_bytes) ||
        !ensure_host_pinned(ctx.h_src1, ctx.h_src1_sz, std::max(src1_f32_bytes, src1_q8_bytes)) ||
        !ensure_host_pinned(ctx.h_dst, ctx.h_dst_sz, dst_bytes)) {
        return false;
    }

    int stage_index = 0;
    for (int j = 0; j < n_active; ++j) {
        if (expert_ptrs[j]) {
            continue;
        }
        const char * expert_host = (const char *) src0_data + (size_t) active_experts[j] * nb02;
        void * expert_dev = (char *) ctx.d_src0_stage + (size_t) stage_index * expert_bytes;
        if (cudaMemcpyAsync(expert_dev, expert_host, expert_bytes, cudaMemcpyHostToDevice, ctx.stream) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        expert_ptrs[j] = expert_dev;
        expert_descriptors[j].role_ptr = expert_dev;
        expert_descriptors[j].slot_id = -1;
        ++stage_index;
    }

    char * h_src1 = (char *) ctx.h_src1;
    const auto host_prepare_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto host_prepare_done = std::chrono::steady_clock::time_point{};
    if (down_q80_cpu_order) {
        const char * q80_base = (const char *) src1_q8_0;
        for (int j = 0; j < n_active; ++j) {
            const int32_t i1 = src1_ne1 > 0 ? dst_ids[j] % (int32_t) src1_ne1 : 0;
            const int32_t i2 = token_ids[j];
            const char * src_row = q80_base + ((size_t) i1 + (size_t) i2 * (size_t) src1_ne1) * src1_q8_0_row_size;
            if (down_q80_fast_mmvq) {
                const block_q8_0 * src_blocks = (const block_q8_0 *) src_row;
                block_q8_1 * dst_blocks = (block_q8_1 *) h_src1 +
                    (size_t) j * (size_t) (src1_padded / QK8_1);
                for (int64_t block = 0; block < ne00 / QK8_0; ++block) {
                    std::memcpy(&dst_blocks[block], &src_blocks[block].d, sizeof(ggml_half));
                    std::memset((char *) &dst_blocks[block] + sizeof(ggml_half), 0, sizeof(ggml_half));
                    std::memcpy(dst_blocks[block].qs, src_blocks[block].qs, QK8_0);
                }
            } else {
                std::memcpy(h_src1 + (size_t) j * src1_q8_0_row_size, src_row, src1_q8_0_row_size);
            }
        }
        host_prepare_done = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (profile_cuda && cudaEventRecord(ctx.profile_input_start, ctx.stream) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        if (cudaMemcpyAsync(ctx.d_src1_q8, ctx.h_src1, src1_q8_bytes, cudaMemcpyHostToDevice, ctx.stream) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
    } else {
        const char * src1_base = (const char *) src1_f32;
        for (int j = 0; j < n_active; ++j) {
            const int32_t i1 = src1_ne1 > 0 ? dst_ids[j] % (int32_t) src1_ne1 : 0;
            const int32_t i2 = token_ids[j];
            const char * src_row = src1_base + (size_t) i1 * src1_nb1 + (size_t) i2 * src1_nb2;
            std::memcpy(h_src1 + (size_t) j * (size_t) ne00 * sizeof(float), src_row, (size_t) ne00 * sizeof(float));
        }
        host_prepare_done = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (profile_cuda && cudaEventRecord(ctx.profile_input_start, ctx.stream) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        if (cudaMemcpyAsync(ctx.d_src1_f32, ctx.h_src1, src1_f32_bytes, cudaMemcpyHostToDevice, ctx.stream) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
    }
    if (!ggml_moe::device_descriptor_buffer_copy_async(
            ctx.descriptor_buffer, expert_descriptors, (size_t)n_active, ctx.stream)) {
        return false;
    }
    if (profile_cuda && cudaEventRecord(ctx.profile_input_done, ctx.stream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    if (down_q80_fast_mmvq) {
        if (!launch_mxfp4_decode_batch(
                ctx.descriptor_buffer.device, ctx.d_src1_q8, ctx.d_dst,
                ne00, ne01, nb01, n_active, ctx.stream)) return false;
    } else if (down_q80_cpu_order) {
        if (!launch_mxfp4_q80_cpu_order_decode_batch(
                ctx.descriptor_buffer.device, ctx.d_src1_q8, ctx.d_dst,
                ne00, ne01, nb01, src1_q8_0_row_size, n_active, ctx.stream)) return false;
    } else {
        quantize_row_q8_1_cuda(
                (const float *) ctx.d_src1_f32,
                nullptr,
                ctx.d_src1_q8,
                GGML_TYPE_MXFP4,
                ne00,
                ne00,
                ne00 * n_active,
                ne00 * n_active,
                src1_padded,
                n_active,
                1,
                1,
                ctx.stream);
        if (cudaGetLastError() != cudaSuccess || !launch_mxfp4_decode_batch(
                ctx.descriptor_buffer.device, ctx.d_src1_q8, ctx.d_dst,
                ne00, ne01, nb01, n_active, ctx.stream)) return false;
    }
    if (profile_cuda && cudaEventRecord(ctx.profile_kernel_done, ctx.stream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (cudaMemcpyAsync(ctx.h_dst, ctx.d_dst, dst_bytes, cudaMemcpyDeviceToHost, ctx.stream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (profile_cuda && cudaEventRecord(ctx.profile_d2h_done, ctx.stream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    const auto sync_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (cudaStreamSynchronize(ctx.stream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    lease_guard.release_after_synchronized_work();
    const auto sync_done = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    const float * h_dst = (const float *) ctx.h_dst;
    for (int j = 0; j < n_active; ++j) {
        float * dst_row = (float *) ((char *) dst + (size_t) dst_ids[j] * dst_nb1 + (size_t) token_ids[j] * dst_nb2);
        std::memcpy(dst_row, h_dst + (size_t) j * (size_t) ne01, (size_t) ne01 * sizeof(float));
        matrix_row_counts[active_experts[j]] = 0;
    }
    const auto profile_done = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    if (profile_role >= 0) {
        auto ns = [](auto begin, auto end) {
            return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        };
        decode_batch_profile_role & counters = g_decode_batch_profile[profile_role];
        counters.calls.fetch_add(1, std::memory_order_relaxed);
        counters.cache_ns.fetch_add(ns(cache_start, cache_done), std::memory_order_relaxed);
        counters.host_prepare_ns.fetch_add(ns(host_prepare_start, host_prepare_done), std::memory_order_relaxed);
        counters.enqueue_ns.fetch_add(ns(cache_done, sync_start), std::memory_order_relaxed);
        counters.sync_ns.fetch_add(ns(sync_start, sync_done), std::memory_order_relaxed);
        counters.scatter_ns.fetch_add(ns(sync_done, profile_done), std::memory_order_relaxed);
        counters.total_ns.fetch_add(ns(profile_start, profile_done), std::memory_order_relaxed);
        counters.input_h2d_bytes.fetch_add(src1_f32_bytes + src1_q8_bytes, std::memory_order_relaxed);
        counters.descriptor_h2d_bytes.fetch_add(
            (uint64_t) n_active * sizeof(ggml_moe_expert_descriptor), std::memory_order_relaxed);
        counters.d2h_bytes.fetch_add(dst_bytes, std::memory_order_relaxed);
        if (profile_cuda) {
            float input_ms = 0.0f;
            float kernel_ms = 0.0f;
            float d2h_ms = 0.0f;
            if (cudaEventElapsedTime(&input_ms, ctx.profile_input_start, ctx.profile_input_done) == cudaSuccess &&
                    cudaEventElapsedTime(&kernel_ms, ctx.profile_input_done, ctx.profile_kernel_done) == cudaSuccess &&
                    cudaEventElapsedTime(&d2h_ms, ctx.profile_kernel_done, ctx.profile_d2h_done) == cudaSuccess) {
                counters.input_h2d_us.fetch_add((uint64_t) (input_ms * 1000.0f), std::memory_order_relaxed);
                counters.kernel_us.fetch_add((uint64_t) (kernel_ms * 1000.0f), std::memory_order_relaxed);
                counters.d2h_us.fetch_add((uint64_t) (d2h_ms * 1000.0f), std::memory_order_relaxed);
            } else {
                cudaGetLastError();
            }
        }
    }

    g_decode_batch_calls.fetch_add(1, std::memory_order_relaxed);
    g_decode_batch_cache_hits.fetch_add((uint64_t) cache_hits, std::memory_order_relaxed);
    g_decode_batch_ram_promotes.fetch_add((uint64_t) ram_promotes, std::memory_order_relaxed);
    g_decode_batch_cache_inserts.fetch_add((uint64_t) cache_inserts, std::memory_order_relaxed);
    g_decode_batch_pack_inserts.fetch_add((uint64_t) pack_inserts, std::memory_order_relaxed);
    g_decode_batch_staged.fetch_add((uint64_t) staged, std::memory_order_relaxed);
    if (g_decode_batch_first.fetch_add(1, std::memory_order_relaxed) == 0 ||
            env_enabled("GGML_MOE_STREAM_DECODE_BATCH_TRACE", false)) {
        std::fprintf(stderr,
                "[moe_stream_decode_batch] active tensor=%s routes=%d gpu_routes=%d cpu_fallback_routes=%d one_hits=%d ram_promotes=%d one_inserts=%d pack_inserts=%d staged=%d ne01=%ld ne00=%ld\n",
                src0_name ? src0_name : "", routed_active, n_active, frozen_misses, cache_hits, ram_promotes, cache_inserts, pack_inserts, staged, (long) ne01, (long) ne00);
    }
    return frozen_misses == 0;
}

#endif
