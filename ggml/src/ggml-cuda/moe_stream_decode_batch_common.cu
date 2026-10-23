#include "moe-common-adapter-symbols.h"
#include "moe-expert-descriptor.h"
#include "moe-common-device-descriptor.h"

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
    ggml_moe_common::device_descriptor_buffer descriptor_buffer;
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
    if (!ggml_moe_common::device_descriptor_buffer_allocate_fixed(
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

static __device__ __forceinline__ float vec_dot_mxfp4_planar_q8_1(
        const uint8_t * __restrict__ quant,
        const uint8_t * __restrict__ scales,
        const block_q8_1 * __restrict__ bq8_1,
        int kbx,
        int iqs) {
    const uint8_t *qs = quant + (size_t) kbx * (QK_MXFP4 / 2);
    const int *q8 = (const int *) bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b1(qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    const float d = ggml_cuda_e8m0_to_fp32(scales[kbx]) * 0.5f *
        __low2float(bq8_1->ds);
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_mxfp4_scale3_q8_1(
        const uint8_t * __restrict__ compact,
        const block_q8_1 * __restrict__ bq8_1,
        int kbx,
        int iqs) {
    constexpr int group_blocks = 8;
    constexpr int group_quant_bytes = group_blocks * (QK_MXFP4 / 2);
    constexpr int group_bytes = group_quant_bytes + 3;
    constexpr uint8_t scale_base = 118;
    const int group = kbx >> 3;
    const int local_block = kbx & (group_blocks - 1);
    const uint8_t *group_data = compact + (size_t) group * group_bytes;
    const uint8_t *qs = group_data + local_block * (QK_MXFP4 / 2);
    const uint8_t *packed_scales = group_data + group_quant_bytes;
    // The MMVQ mapping assigns one consecutive block to each lane, so every
    // eight-lane subgroup consumes one packed scale word.
    const int group_leader = (threadIdx.x & 31) & ~(group_blocks - 1);
    uint32_t scale_word = 0;
    if ((threadIdx.x & 31) == group_leader) {
        scale_word = static_cast<uint32_t>(packed_scales[0]) |
            (static_cast<uint32_t>(packed_scales[1]) << 8) |
            (static_cast<uint32_t>(packed_scales[2]) << 16);
    }
    scale_word = __shfl_sync(0xffffffffU, scale_word, group_leader);
    const uint8_t scale = static_cast<uint8_t>(
        scale_base + ((scale_word >> (3 * local_block)) & 7U));
    const int *q8 = (const int *) bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b1(qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    const float d = ggml_cuda_e8m0_to_fp32(scale) * 0.5f *
        __low2float(bq8_1->ds);
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_mxfp4_planar_scale3_q8_1(
        const uint8_t * __restrict__ quant,
        const uint8_t * __restrict__ packed_scales,
        const block_q8_1 * __restrict__ bq8_1,
        int kbx,
        int iqs) {
    const uint8_t *qs = quant + static_cast<size_t>(kbx) *
        (QK_MXFP4 / 2);
    const int group = kbx >> 3;
    const int local_block = kbx & 7;
    const uint8_t *group_scales = packed_scales +
        static_cast<size_t>(group) * 3;
    const int group_leader = (threadIdx.x & 31) & ~7;
    uint32_t scale_word = 0;
    if ((threadIdx.x & 31) == group_leader) {
        scale_word = static_cast<uint32_t>(group_scales[0]) |
            (static_cast<uint32_t>(group_scales[1]) << 8) |
            (static_cast<uint32_t>(group_scales[2]) << 16);
    }
    scale_word = __shfl_sync(0xffffffffU, scale_word, group_leader);
    const uint8_t scale = static_cast<uint8_t>(
        118U + ((scale_word >> (3 * local_block)) & 7U));
    const int *q8 = (const int *) bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b1(qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    const float d = ggml_cuda_e8m0_to_fp32(scale) * 0.5f *
        __low2float(bq8_1->ds);
    return d * sumi;
}

template<int layout_mode>
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
    const void *x = descriptors[active].role_ptr;
    const bool planar = layout_mode == 1 || (layout_mode == 2 &&
        (descriptors[active].role &
            GGML_MOE_EXPERT_LAYOUT_PLANAR_MXFP4) != 0);
    const bool scale3 = layout_mode == 3 || (layout_mode == 2 &&
        (descriptors[active].role &
            GGML_MOE_EXPERT_LAYOUT_SCALE3_MXFP4) != 0);
    const bool planar_scale3 = layout_mode == 4 || (layout_mode == 2 &&
        (descriptors[active].role &
            GGML_MOE_EXPERT_LAYOUT_PLANAR_SCALE3_MXFP4) != 0);
    const uint8_t *planar_quant = static_cast<const uint8_t *>(x);
    const uint8_t *planar_scales = planar_quant +
        (size_t) nrows_x * stride_row_x * (QK_MXFP4 / 2);
    const block_q8_1 * y_row = y + active * stride_col_y;
    const int kbx_offset = row0 * stride_row_x;

    float tmp[rows_per_block] = {0.0f, 0.0f};
    for (int kbx = threadIdx.x / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi / vdr));
#pragma unroll
        for (int i = 0; i < rows_per_block; ++i) {
            const int block_index =
                kbx_offset + i * stride_row_x + kbx;
            if constexpr (layout_mode == 4) {
                tmp[i] += vec_dot_mxfp4_planar_scale3_q8_1(
                    planar_quant, planar_scales, &y_row[kby],
                    block_index, kqs);
            } else if constexpr (layout_mode == 3) {
                tmp[i] += vec_dot_mxfp4_scale3_q8_1(
                    static_cast<const uint8_t *>(x), &y_row[kby],
                    block_index, kqs);
            } else if constexpr (layout_mode == 1) {
                tmp[i] += vec_dot_mxfp4_planar_q8_1(
                    planar_quant, planar_scales, &y_row[kby],
                    block_index, kqs);
            } else if constexpr (layout_mode == 0) {
                tmp[i] += vec_dot_mxfp4_q8_1(
                    x, &y_row[kby], block_index, kqs);
            } else {
                tmp[i] += planar_scale3 ?
                    vec_dot_mxfp4_planar_scale3_q8_1(
                        planar_quant, planar_scales, &y_row[kby],
                        block_index, kqs) : scale3 ?
                    vec_dot_mxfp4_scale3_q8_1(
                        static_cast<const uint8_t *>(x), &y_row[kby],
                        block_index, kqs) : planar ?
                    vec_dot_mxfp4_planar_q8_1(
                        planar_quant, planar_scales, &y_row[kby],
                        block_index, kqs) :
                    vec_dot_mxfp4_q8_1(
                        x, &y_row[kby], block_index, kqs);
            }
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
        int layout_mode,
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

    if (layout_mode == 4) {
        moe_decode_mxfp4_ptr_batch_kernel<4><<<grid, block, 0, stream>>>(
                d_expert_descriptors, (const block_q8_1 *) d_src1_q8,
                (float *) d_dst, (uint32_t) ne00, (uint32_t) ne01,
                stride_row_x, stride_col_y, stride_col_dst,
                (uint32_t) n_active);
    } else if (layout_mode == 3) {
        moe_decode_mxfp4_ptr_batch_kernel<3><<<grid, block, 0, stream>>>(
                d_expert_descriptors, (const block_q8_1 *) d_src1_q8,
                (float *) d_dst, (uint32_t) ne00, (uint32_t) ne01,
                stride_row_x, stride_col_y, stride_col_dst,
                (uint32_t) n_active);
    } else if (layout_mode == 1) {
        moe_decode_mxfp4_ptr_batch_kernel<1><<<grid, block, 0, stream>>>(
                d_expert_descriptors, (const block_q8_1 *) d_src1_q8,
                (float *) d_dst, (uint32_t) ne00, (uint32_t) ne01,
                stride_row_x, stride_col_y, stride_col_dst,
                (uint32_t) n_active);
    } else if (layout_mode == 0) {
        moe_decode_mxfp4_ptr_batch_kernel<0><<<grid, block, 0, stream>>>(
                d_expert_descriptors, (const block_q8_1 *) d_src1_q8,
                (float *) d_dst, (uint32_t) ne00, (uint32_t) ne01,
                stride_row_x, stride_col_y, stride_col_dst,
                (uint32_t) n_active);
    } else {
        moe_decode_mxfp4_ptr_batch_kernel<2><<<grid, block, 0, stream>>>(
                d_expert_descriptors, (const block_q8_1 *) d_src1_q8,
                (float *) d_dst, (uint32_t) ne00, (uint32_t) ne01,
                stride_row_x, stride_col_y, stride_col_dst,
                (uint32_t) n_active);
    }
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
    // The opt-in runtime streaming compute path is retired.
    return false;
}

#endif
