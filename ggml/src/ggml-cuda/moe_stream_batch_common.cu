#include "moe-common-adapter-symbols.h"
#include "moe-adapter-dispatch.h"
#include "moe-expert-descriptor.h"
#include "moe-common-device-descriptor.h"
#if defined(GGML_SSD_STORE)
#include "catalog.h"
#include "parser.h"
#endif

#ifndef GGML_CUDA_MOE_STREAM_BATCH
#include <cstdint>
#include <cstddef>
extern "C" {
typedef struct { int32_t i1; int32_t i2; } ggml_moe_row_mapping;
typedef struct { float direct_up; float direct_gate; float direct_fused; float repack_up; float repack_gate; float repack_fused; } ggml_moe_iq2_replay_result;
void ggml_cuda_moe_stream_batch_link_anchor(void) {}
void ggml_cuda_moe_ttft_trace_mark(const char *) {}
bool ggml_cuda_moe_iq2_prompt_replay(int, const void *, const void *, int64_t, int64_t, size_t, const float *, int64_t, int, float, ggml_moe_iq2_replay_result *) { return false; }
bool ggml_cuda_moe_stream_batch(int, const char *, const void *, int64_t, int64_t, int64_t, size_t, size_t, const float *, int64_t, size_t, size_t, const void *, size_t, int64_t, float *, size_t, size_t, const int64_t *, const ggml_moe_row_mapping *, int64_t) { return false; }
bool ggml_cuda_moe_stream_stage_active_experts(const char *, const void *, int64_t, size_t, size_t, const int64_t *, void *, size_t, void *) { return false; }
bool ggml_cuda_moe_stream_handoff_upload(const float *, int64_t, int64_t) { return false; }
int ggml_cuda_moe_stream_batch_preload_gate_updown(const char *, int, size_t) { return 0; }
int ggml_cuda_moe_stream_batch_flush_gate_updown_cosubmit(void) { return 0; }
int ggml_cuda_moe_stream_batch_preload_active_from_pack(int, const char *, int64_t, size_t, const int64_t *) { return 0; }
int ggml_cuda_moe_stream_q4_colibri_pilot_prefetch_from_gate(int, const char *, int64_t, size_t, const int64_t *) { return 0; }
int ggml_cuda_moe_stream_route_group_preload_updown_from_gate(const char *, int64_t, const int64_t *) { return 0; }
bool ggml_cuda_moe_stream_preload_tensor(int, const char *, const void *, int64_t, size_t, size_t) { return false; }
bool ggml_cuda_moe_stream_preload_tensor_prompt(int, const char *, const void *, int64_t, size_t, size_t) { return false; }
bool ggml_cuda_moe_stream_register_tensor(int, const char *, const void *, int64_t, size_t, size_t) { return false; }
bool ggml_cuda_moe_stream_cache_contains(const char *, size_t, int) { return false; }
const void * ggml_cuda_moe_stream_cache_dev_ptr(const char *, size_t, int) { return nullptr; }
void ggml_cuda_moe_stream_batch_phase_report(const char *) {}
bool ggml_cuda_moe_iq2_xxs_q8k_selftest(void) { return false; }
const void * ggml_cuda_moe_expert_pack_mmap_ptr(const char *, int, size_t) { return nullptr; }
const void * ggml_cuda_moe_expert_pack_mmap_ptr_debug(const char *, int, size_t, const char **, size_t *, uint64_t *) { return nullptr; }
const void * ggml_cuda_moe_expert_pack_mmap_ptr_source_debug(const char *, int, size_t, const char **, size_t *, uint64_t *, int *, const char **, int *) { return nullptr; }
bool ggml_cuda_moe_expert_pack_v2_lookup_debug(const char *, int, int *, int64_t *, int64_t *, size_t *, size_t *) { return false; }
bool ggml_cuda_moe_expert_pack_v2_read_debug(const char *, int, void *, size_t, size_t *) { return false; }
bool ggml_cuda_moe_expert_pack_read_to_host(const char *, int, size_t, void *, size_t, size_t *, int *) { return false; }
bool ggml_cuda_moe_expert_pack_resolve_direct(const char *, int, size_t, int *, uint64_t *, size_t *) { return false; }
bool ggml_cuda_moe_expert_pack_copy_ranges_to_device(const char *, const int32_t *, const int32_t *, size_t, int64_t, size_t, size_t, bool, void *, size_t, void *, uint64_t *, uint64_t *, uint64_t *, uint64_t *, uint64_t *) { return false; }
bool ggml_cuda_moe_dense_has_tensor(const char *, size_t, size_t *) { return false; }
bool ggml_cuda_moe_dense_stage_prepare(size_t *) { return false; }
bool ggml_cuda_moe_dense_read_ahead(const char *, size_t) { return false; }
bool ggml_cuda_moe_dense_copy_to_device(const char *, void *, size_t, void *, bool *) { return false; }
}
#else
// Decode-only batched streaming MoE path.
//
// This file intentionally includes the MMQ-id kernel family without the MMVQ
// headers used by moe_stream.cu; several CUDA helper headers define unguarded
// device functions and cannot be mixed in one translation unit.

#include "common.cuh"
#include "ggml-quants.h"
#include "moe-common-object-pipeline-runtime.h"
#include "moe-common-io-submit.h"
#include "moe-common-role-cache.h"
#include "moe-common-dynamic-tier.h"
#include "q4-colibri-worker-state-common.h"
#include "../../../vendor/nlohmann/json.hpp"
#include "quantize.cuh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <deque>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

extern "C" bool ggml_cuda_moe_stream_one_cache_dev_ptrs_or_insert(
    const char *src0_name,
    const void *src0_base_data,
    size_t nb02,
    size_t src0_bytes,
    const int *expert_indices,
    const int32_t *route_ranks,
    int count,
    cudaStream_t stream,
    const void **output_ptrs,
    int *cache_states,
    ggml_moe_expert_descriptor *descriptors);
extern "C" void ggml_cuda_moe_stream_one_cache_release_descriptors(
    const ggml_moe_expert_descriptor *descriptors,
    int count,
    cudaStream_t stream);
extern "C" void ggml_cuda_moe_stream_one_cache_release_descriptors_synchronized(
    const ggml_moe_expert_descriptor *descriptors,
    int count);

#if !defined(_WIN32)
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cuda_runtime.h>

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define GGML_MOE_NVTX_HEADER_AVAILABLE 1
#else
#define GGML_MOE_NVTX_HEADER_AVAILABLE 0
#endif

#if !defined(_WIN32)
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#endif

#include "moe-common-object-transport.h"

namespace ggml_moe_common_adapter {

static bool common_dynamic_tier_requested() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_MOE_COMMON_TIER_DYNAMIC");
        return env && env[0] && env[0] != '0';
    }();
    return enabled;
}

static void common_dynamic_tier_report();
static void common_dynamic_tier_phase_report(const char * label, uint64_t sequence);

#define MOE_STREAM_MAX_ACTIVE 4096





struct moe_nvtx_api {
    bool initialized = false;
    bool enabled = false;
#if !GGML_MOE_NVTX_HEADER_AVAILABLE
    int (*range_push_a)(const char *) = nullptr;
    int (*range_pop)() = nullptr;
#endif
};

static moe_nvtx_api g_moe_nvtx;
static std::once_flag g_moe_nvtx_once;

static void moe_nvtx_init_once() {
    const char *env = std::getenv("GGML_MOE_NVTX");
    if (!env || !env[0] || env[0] == '0') {
        g_moe_nvtx.initialized = true;
        return;
    }
#if GGML_MOE_NVTX_HEADER_AVAILABLE
    g_moe_nvtx.enabled = true;
#else
#if !defined(_WIN32)
    void *handle = dlopen("libnvToolsExt.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        handle = dlopen("libnvToolsExt.so", RTLD_LAZY | RTLD_LOCAL);
    }
    if (handle) {
        g_moe_nvtx.range_push_a = reinterpret_cast<int (*)(const char *)>(dlsym(handle, "nvtxRangePushA"));
        g_moe_nvtx.range_pop = reinterpret_cast<int (*)()>(dlsym(handle, "nvtxRangePop"));
        g_moe_nvtx.enabled = g_moe_nvtx.range_push_a && g_moe_nvtx.range_pop;
    }
#endif
#endif
    g_moe_nvtx.initialized = true;
}

static bool moe_nvtx_push(const char *label) {
    std::call_once(g_moe_nvtx_once, moe_nvtx_init_once);
    if (!g_moe_nvtx.enabled || !label || !label[0]) return false;
#if GGML_MOE_NVTX_HEADER_AVAILABLE
    nvtxRangePushA(label);
#else
    g_moe_nvtx.range_push_a(label);
#endif
    return true;
}

static void moe_nvtx_pop() {
    if (g_moe_nvtx.enabled) {
#if GGML_MOE_NVTX_HEADER_AVAILABLE
        nvtxRangePop();
#else
        if (g_moe_nvtx.range_pop) {
        g_moe_nvtx.range_pop();
        }
#endif
    }
}

class moe_nvtx_range {
public:
    explicit moe_nvtx_range(const char *label) : active(moe_nvtx_push(label)) {}
    ~moe_nvtx_range() {
        if (active) moe_nvtx_pop();
    }
    moe_nvtx_range(const moe_nvtx_range &) = delete;
    moe_nvtx_range & operator=(const moe_nvtx_range &) = delete;

private:
    bool active = false;
};

#if !defined(_WIN32)
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#endif

#include "moe-common-object-transport.h"

extern "C" {

typedef struct {
    int32_t i1;
    int32_t i2;
} ggml_moe_row_mapping;

typedef struct {
    float direct_up;
    float direct_gate;
    float direct_fused;
    float repack_up;
    float repack_gate;
    float repack_fused;
} ggml_moe_iq2_replay_result;

bool ggml_cuda_moe_iq2_prompt_replay(
        int src0_type_int,
        const void *up_expert,
        const void *gate_expert,
        int64_t ne01,
        int64_t ne00,
        size_t nb01,
        const float *src1_row_f32,
        int64_t col,
        int unary_op,
        float limit,
        ggml_moe_iq2_replay_result *out);

void ggml_cuda_moe_stream_batch_link_anchor(void) {
    (void)&ggml_cuda_moe_iq2_prompt_replay;
}

void ggml_cuda_moe_ttft_trace_mark(const char *label);
bool ggml_cuda_moe_stream_one_cache_ready(void);

bool ggml_cuda_moe_stream_handoff_upload(const float *host_ptr, int64_t ne01, int64_t dst_cols);

bool ggml_cuda_moe_stream_batch(
    int  src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    size_t nb02,
    const float *src1_f32,
    int64_t src1_ne1,
    size_t src1_nb1, size_t src1_nb2,
    const void *src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_q8_0_ne1,
    float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const int64_t *matrix_row_counts,
    const ggml_moe_row_mapping *matrix_rows,
    int64_t rows_stride);

int ggml_cuda_moe_stream_batch_preload_gate_updown(const char *gate_name, int expert_idx, size_t expert_bytes);
int ggml_cuda_moe_stream_batch_flush_gate_updown_cosubmit(void);
int ggml_cuda_moe_stream_batch_preload_active_from_pack(
    int src0_type_int,
    const char *src0_name,
    int64_t n_as,
    size_t expert_bytes,
    const int64_t *matrix_row_counts);
int ggml_cuda_moe_stream_q4_colibri_pilot_prefetch_from_gate(
    int src0_type_int,
    const char *src0_gate_name,
    int64_t n_as,
    size_t expert_bytes,
    const int64_t *matrix_row_counts);
int ggml_cuda_moe_stream_route_group_preload_updown_from_gate(
    const char *src0_gate_name,
    int64_t n_as,
    const int64_t *matrix_row_counts);
const void * ggml_cuda_moe_stream_cache_dev_ptr(const char *src0_name, size_t expert_bytes, int expert_idx);

bool ggml_cuda_moe_stream_preload_tensor(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes);
bool ggml_cuda_moe_stream_preload_tensor_prompt(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes);
bool ggml_cuda_moe_stream_register_tensor(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes);

bool ggml_cuda_moe_stream_cache_contains(
    const char *src0_name,
    size_t expert_bytes,
    int expert_idx);

bool ggml_cuda_moe_stream_mmvq_dev(
    int src0_type_int,
    const void *d_src0,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const float *d_src1_f32,
    void *d_src1_q8,
    float *d_dst,
    cudaStream_t stream);
bool ggml_cuda_moe_stream_mmvq_batch_dev(
    int src0_type_int,
    const void *d_src0,
    int64_t ne01,
    int64_t ne00,
    const float *d_src1_f32,
    void *d_src1_q8,
    float *d_dst,
    const int32_t *d_x_ids,
    int64_t n_active,
    int64_t src0_stride,
    cudaStream_t stream);

}

using pinned_stage_slot = ggml_moe_common::object_stage_slot;
using pinned_stage_ring = ggml_moe_common::object_stage_ring;


struct batch_ctx {
    cudaStream_t stream = nullptr;
    cudaStream_t prefetch_stream = nullptr;
    cudaStream_t up_stream = nullptr;
    cudaStream_t gate_stream = nullptr;
    cudaStream_t up_copy_stream = nullptr;
    cudaStream_t gate_copy_stream = nullptr;
    void * d_src0 = nullptr;     size_t d_src0_sz = 0;
    void * d_src1_f32 = nullptr; size_t d_src1_f32_sz = 0;
    void * d_src1_q8 = nullptr;  size_t d_src1_q8_sz = 0;
    void * d_src1_q8k = nullptr; size_t d_src1_q8k_sz = 0;
    void * d_src1_q8_up = nullptr;   size_t d_src1_q8_up_sz = 0;
    void * d_src1_q8_gate = nullptr; size_t d_src1_q8_gate_sz = 0;
    void * d_src1_q8_one = nullptr; size_t d_src1_q8_one_sz = 0;
    void * d_dst = nullptr;      size_t d_dst_sz = 0;
    void * d_up = nullptr;       size_t d_up_sz = 0;
    void * d_gate = nullptr;     size_t d_gate_sz = 0;
    void * d_handoff = nullptr;  size_t d_handoff_sz = 0;
    ggml_moe_common::device_descriptor_buffer q4_graph_descriptor_buffer;
    int32_t * d_ids_src1 = nullptr; size_t d_ids_src1_sz = 0;
    int32_t * d_ids_dst = nullptr; size_t d_ids_dst_sz = 0;
    int32_t * d_x_ids = nullptr; size_t d_x_ids_sz = 0;
    int32_t * d_x_ids_up = nullptr;   size_t d_x_ids_up_sz = 0;
    int32_t * d_x_ids_gate = nullptr; size_t d_x_ids_gate_sz = 0;
    int32_t * d_bounds = nullptr;  size_t d_bounds_sz = 0;
    void * h_src1 = nullptr;     size_t h_src1_sz = 0;
    void * h_src1_shadow = nullptr; size_t h_src1_shadow_sz = 0;
    void * h_dst = nullptr;      size_t h_dst_sz = 0;
    void * h_dst_shadow = nullptr; size_t h_dst_shadow_sz = 0;
    pinned_stage_ring stage_ring;
    pinned_stage_ring stage_ring_gate;
    pinned_stage_ring stage_ring_up_aux;
    pinned_stage_ring stage_ring_gate_aux;
    pinned_stage_ring stage_ring_down_aux;
    cudaEvent_t ev_stage_ready = nullptr;
    cudaEvent_t ev_up_done = nullptr;
    cudaEvent_t ev_gate_done = nullptr;
    cudaEvent_t ev_up_copy_aux_done = nullptr;
    cudaEvent_t ev_gate_copy_aux_done = nullptr;
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stage = nullptr;
    cudaEvent_t ev_quant = nullptr;
    cudaEvent_t ev_kernel = nullptr;
    cudaEvent_t ev_d2h = nullptr;
    cudaEvent_t ev_up_start = nullptr;
    cudaEvent_t ev_gate_work_start = nullptr;
    cudaEvent_t ev_up_compute_start = nullptr;
    cudaEvent_t ev_gate_compute_start = nullptr;
    cudaEvent_t ev_gate_start = nullptr;
    cudaEvent_t ev_up = nullptr;
    cudaEvent_t ev_gate = nullptr;
    int32_t h_ids_dst[MOE_STREAM_MAX_ACTIVE] = {};
    int32_t h_up_gate_ids_dst[MOE_STREAM_MAX_ACTIVE] = {};
    int32_t h_ids_src1[MOE_STREAM_MAX_ACTIVE] = {};
    int32_t h_x_ids[MOE_STREAM_MAX_ACTIVE] = {};
    int32_t h_x_ids_up[MOE_STREAM_MAX_ACTIVE] = {};
    int32_t h_x_ids_gate[MOE_STREAM_MAX_ACTIVE] = {};
    int32_t h_bounds[MOE_STREAM_MAX_ACTIVE + 1] = {};
};

static batch_ctx g_batch;
static std::mutex g_batch_mu;
static std::atomic<bool> g_batch_inited{false};
static std::atomic<bool> g_pinned_stage_report_registered{false};



struct registered_tensor {
    int type = 0;
    char name[128] = {};
    const void *data = nullptr;
    int64_t n_as = 0;
    size_t nb02 = 0;
    size_t expert_bytes = 0;
};

static std::mutex g_registered_mu;
static std::vector<registered_tensor> g_registered_tensors;

struct moe_gpu_handoff {
    const float * host_ptr = nullptr;
    const float * d_data = nullptr;
    size_t bytes = 0;
    int64_t ne01 = 0;
    int64_t dst_cols = 0;
    uint64_t serial = 0;
};

static moe_gpu_handoff g_handoff;
static bool init_batch_once();

struct batch_profile {
    bool enabled = false;
    uint64_t calls = 0;
    uint64_t active_experts = 0;
    uint64_t up_stage_jobs = 0;
    uint64_t gate_stage_jobs = 0;
    uint64_t max_up_stage_jobs = 0;
    uint64_t max_gate_stage_jobs = 0;
    uint64_t max_combined_stage_jobs = 0;
    uint64_t up_stage_jobs_hist[7] = {};
    uint64_t gate_stage_jobs_hist[7] = {};
    uint64_t combined_stage_jobs_hist[7] = {};
    double stage_ms = 0.0;
    double quant_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double scatter_ms = 0.0;
    double up_ms = 0.0;
    double gate_ms = 0.0;
    double up_wait_ms = 0.0;
    double gate_wait_ms = 0.0;
    double up_compute_ms = 0.0;
    double gate_compute_ms = 0.0;
    double fuse_ms = 0.0;
    double wall_ms = 0.0;
};

static batch_profile g_bprof;
static batch_profile g_uprof;
static batch_profile g_uprof_by_type[2][GGML_TYPE_COUNT][GGML_TYPE_COUNT];

static bool q4_graph_compute_region_profile_enabled() {
    const char *env = std::getenv("GGML_MOE_Q4_GRAPH_COMPUTE_REGION_PROFILE");
    return env && env[0] && env[0] != '0';
}



struct up_gate_layer_profile_entry {
    bool used = false;
    bool prompt_mode = false;
    ggml_type up_type = GGML_TYPE_COUNT;
    ggml_type gate_type = GGML_TYPE_COUNT;
    char up_tensor[128] = {};
    char gate_tensor[128] = {};
    batch_profile profile;
};

static bool g_up_gate_layer_profile_enabled = false;
static std::atomic<uint64_t> g_up_gate_layer_profile_records{0};
static std::mutex g_up_gate_layer_profile_mu;
static up_gate_layer_profile_entry g_up_gate_layer_profiles[256];







struct current_down_overlap_profile {
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> planned_jobs{0};
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> missing_tensor{0};
    std::atomic<uint64_t> missing_pack{0};
    std::atomic<uint64_t> submitted_batches{0};
    std::atomic<uint64_t> completed_jobs{0};
    std::atomic<uint64_t> failed_batches{0};
    std::atomic<uint64_t> mark_failed{0};
    std::atomic<uint64_t> max_jobs{0};
    std::atomic<uint64_t> batch_hist_1{0};
    std::atomic<uint64_t> batch_hist_2_4{0};
    std::atomic<uint64_t> batch_hist_5_8{0};
    std::atomic<uint64_t> batch_hist_9_16{0};
    std::atomic<uint64_t> batch_hist_17_32{0};
    std::atomic<uint64_t> batch_hist_gt32{0};
    std::atomic<uint64_t> worker_us{0};
};

static current_down_overlap_profile g_current_down_overlap;

struct current_down_overlap_tensor_profile {
    uint64_t calls = 0;
    uint64_t planned_jobs = 0;
    uint64_t cache_hits = 0;
    uint64_t missing_tensor = 0;
    uint64_t missing_pack = 0;
    uint64_t completed_jobs = 0;
};

static std::mutex g_current_down_overlap_tensor_profile_mu;
static std::unordered_map<std::string, current_down_overlap_tensor_profile> g_current_down_overlap_tensor_profile;

#if defined(GGML_SSD_STORE)
using expert_pack_entry = c0::expert_pack_entry;
using expert_pack_source = c0::expert_pack_source;
#else
struct expert_pack_entry {
    char tensor[128] = {};
    int32_t expert_idx = -1;
    int32_t source_idx = 0;
    int32_t layer_idx = -1;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
    uint64_t object_offset = 0;
    uint64_t object_nbytes = 0;
    uint64_t object_relative_offset = 0;
    bool alias_source = false;
};

struct expert_pack_source {
    FILE *file = nullptr;
#if !defined(_WIN32)
    int fd_direct = -1;
#endif
    void *mmap_base = nullptr;
    size_t mmap_size = 0;
    bool mmap_attempted = false;
    bool mmap_enabled = false;
    bool page_cache = false;
    bool direct_required = false;
    uint64_t file_size = 0;
    char path[512] = {};
};

#endif

struct expert_pack_adjacent_op_stats {
    std::atomic<uint64_t> groups{0};
    std::atomic<uint64_t> slices{0};
    std::atomic<uint64_t> extents_saved{0};
    std::atomic<uint64_t> physical_bytes{0};
    std::atomic<uint64_t> payload_bytes{0};
    std::atomic<uint64_t> max_group_jobs{0};
    std::atomic<uint64_t> max_span_bytes{0};
};

enum class q4_lossless_transport_mode : uint8_t {
    disabled = 0,
    verify = 1,
    cuda = 2,
};

struct expert_pack_state {
#if defined(GGML_SSD_STORE)
    FILE * & file = c0::shared_catalog().file;
    int & fd_direct = c0::shared_catalog().fd_direct;
    std::vector<expert_pack_source> & sources = c0::shared_catalog().sources;
    std::vector<expert_pack_entry> & entries = c0::shared_catalog().entries;
#else
    FILE *file = nullptr;
#if !defined(_WIN32)
    int fd_direct = -1;
#endif
    std::vector<expert_pack_source> sources;
    std::vector<expert_pack_entry> entries;
#endif
    std::atomic<uint64_t> mmap_hits{0};
    std::atomic<uint64_t> mmap_misses{0};
    std::atomic<uint64_t> mmap_bytes{0};
#if defined(GGML_SSD_STORE)
    bool & inited = c0::shared_catalog().inited;
    bool & enabled = c0::shared_catalog().enabled;
#else
    bool inited = false;
    bool enabled = false;
#endif
    bool q4_colibri = false;
    bool glm52_colibri = false;
    std::atomic<uint64_t> lossless_object_lookups{0};
    std::atomic<uint64_t> lossless_object_hits{0};
    std::atomic<uint64_t> lossless_object_fallbacks{0};
    std::atomic<uint64_t> lossless_ssd_bytes{0};
    std::atomic<uint64_t> lossless_raw_bytes{0};
    std::atomic<uint64_t> lossless_h2d_bytes{0};
    std::atomic<uint64_t> lossless_decoded_bytes{0};
    std::atomic<uint64_t> lossless_decode_us{0};
    std::atomic<uint64_t> lossless_decode_launches{0};
    std::atomic<uint64_t> lossless_decode_objects{0};
    std::atomic<uint64_t> lossless_decode_max_batch{0};
    std::atomic<uint64_t> lossless_async_decode_batches{0};
    std::atomic<uint64_t> lossless_async_decode_syncs{0};
    std::atomic<uint64_t> lossless_async_decode_sync_us{0};
    std::atomic<uint64_t> lossless_async_stage_releases{0};
    std::atomic<uint64_t> lossless_async_cache_invalidations{0};
    std::atomic<uint64_t> lossless_async_verified_batches{0};
    std::atomic<uint64_t> lossless_async_verified_objects{0};
    std::atomic<uint64_t> lossless_offline_trusted_records{0};
    std::atomic<uint64_t> lossless_crc_objects{0};
    std::atomic<uint64_t> lossless_crc_reuses{0};
    std::atomic<uint64_t> lossless_crc_failures{0};
    std::atomic<uint64_t> lossless_parse_failures{0};
    std::atomic<uint64_t> lossless_cuda_failures{0};
    std::atomic<uint64_t> lossless_status_failures[6]{};
    std::atomic<uint64_t> lossless_publish_failures{0};
    std::atomic<uint64_t> lossless_enqueue_failures[6]{};
    std::atomic<uint64_t> lossless_enqueue_h2d_detail[3]{};
    std::atomic<uint64_t> lossless_raw_fallback_bytes{0};
    std::atomic<uint64_t> lossless_raw_host_refill_requests{0};
    std::atomic<uint64_t> lossless_raw_host_refill_skips{0};
    std::atomic<uint64_t> lossless_host_cache_admissions{0};
    std::atomic<uint64_t> lossless_host_cache_hits{0};
    std::atomic<uint64_t> lossless_host_cache_ssd_bytes_avoided{0};
    std::atomic<uint64_t> lossless_host_cache_h2d_bytes{0};
    std::atomic<uint64_t> lossless_host_cache_fallbacks{0};
    std::atomic<uint64_t> lossless_standalone_raw_host_discards{0};
    std::atomic<uint64_t> lossless_standalone_single_role_bridges{0};
    bool q4_colibri_debug_partial_fallback = false;
    uint64_t q4_manifest_generation = 0;
    std::atomic<uint64_t> q4_colibri_partial_fallback_lookups{0};
    std::atomic<uint64_t> q4_colibri_partial_fallback_tensors{0};
    std::atomic<uint64_t> q4_pipe_submitted{0};
    std::atomic<uint64_t> q4_pipe_completed{0};
    std::atomic<uint64_t> q4_pipe_failed{0};
    std::atomic<uint64_t> q4_pipe_queue_full{0};
    std::atomic<uint64_t> q4_pipe_queue_wait_us{0};
    std::atomic<uint64_t> q4_pipe_worker_us{0};
    std::atomic<uint64_t> q4_pipe_ready_latency_us{0};
    std::atomic<uint64_t> q4_pipe_stale_generation{0};
    std::atomic<uint64_t> q4_pipe_key_mismatch{0};
    std::atomic<uint64_t> q4_pipe_shutdown_drain{0};
    std::atomic<uint64_t> q4_common_refill_batches{0};
    std::atomic<uint64_t> q4_common_refill_submit_rounds{0};
    std::atomic<uint64_t> q4_common_refill_submitted{0};
    std::atomic<uint64_t> q4_common_refill_completions{0};
    std::atomic<uint64_t> q4_common_refill_completed_batches{0};
    std::atomic<uint64_t> q4_common_refill_failures{0};
    std::atomic<uint64_t> q4_common_refill_plan_checks{0};
    std::atomic<uint64_t> q4_common_refill_plan_matches{0};
    std::atomic<uint64_t> q4_common_refill_plan_mismatches{0};
    std::atomic<uint64_t> q4_common_refill_owned_dispatches{0};
    std::atomic<uint64_t> q4_common_refill_owner_fallbacks{0};
    std::atomic<uint64_t> q4_scheduler_batches{0};
    std::atomic<uint64_t> q4_scheduler_active_experts{0};
    std::atomic<uint64_t> q4_scheduler_requested_objects{0};
    std::atomic<uint64_t> q4_scheduler_physical_jobs{0};
    std::atomic<uint64_t> q4_scheduler_role_slices{0};
    std::atomic<uint64_t> q4_scheduler_duplicates_eliminated{0};
    std::atomic<uint64_t> q4_scheduler_physical_bytes{0};
    std::atomic<uint64_t> q4_scheduler_logical_bytes{0};
    std::atomic<uint64_t> q4_scheduler_wait_us{0};
    std::atomic<uint64_t> q4_scheduler_read_us{0};
    std::atomic<uint64_t> q4_scheduler_h2d_us{0};
    std::atomic<uint64_t> q4_cache_pending_wait_calls{0};
    std::atomic<uint64_t> q4_cache_pending_wait_slots{0};
    std::atomic<uint64_t> q4_cache_pending_wait_us{0};
    std::atomic<uint64_t> q4_cache_pending_batch_wait_calls{0};
    std::atomic<uint64_t> q4_cache_pending_batch_wait_slots{0};
    std::atomic<uint64_t> q4_cache_pending_batch_wait_us{0};
    std::atomic<uint64_t> q4_cache_pending_batch_wait_representative_syncs{0};
    std::atomic<uint64_t> q4_cache_pending_batch_wait_extra_syncs{0};
    std::atomic<uint64_t> q4_object_h2d_batches{0};
    std::atomic<uint64_t> q4_object_h2d_ram_batches{0};
    std::atomic<uint64_t> q4_object_h2d_ssd_batches{0};
    std::atomic<uint64_t> q4_object_h2d_objects{0};
    std::atomic<uint64_t> q4_object_h2d_roles{0};
    std::atomic<uint64_t> q4_object_h2d_bytes{0};
    std::atomic<uint64_t> q4_object_h2d_enqueue_us{0};
    std::atomic<uint64_t> q4_object_adjacent_groups{0};
    std::atomic<uint64_t> q4_object_adjacent_slices{0};
    std::atomic<uint64_t> q4_object_adjacent_extents_saved{0};
    std::atomic<uint64_t> q4_object_adjacent_physical_bytes{0};
    std::atomic<uint64_t> q4_object_adjacent_payload_bytes{0};
    std::atomic<uint64_t> q4_object_adjacent_max_group_jobs{0};
    std::atomic<uint64_t> q4_object_adjacent_max_span_bytes{0};
    std::atomic<uint64_t> q4_object_adjacent_reject_source{0};
    std::atomic<uint64_t> q4_object_adjacent_reject_gap{0};
    std::atomic<uint64_t> q4_object_adjacent_reject_span{0};
    std::atomic<uint64_t> q4_scheduler_queue_depth_sum{0};
    std::atomic<uint64_t> q4_scheduler_queue_depth_samples{0};
    std::atomic<uint64_t> q4_scheduler_queue_depth_max{0};
    std::atomic<uint64_t> q4_scheduler_failures{0};
    std::atomic<uint64_t> q4_layer_io_batches[128]{};
    std::atomic<uint64_t> q4_layer_io_active_experts[128]{};
    std::atomic<uint64_t> q4_layer_io_requested_objects[128]{};
    std::atomic<uint64_t> q4_layer_io_physical_jobs[128]{};
    std::atomic<uint64_t> q4_layer_io_role_slices[128]{};
    std::atomic<uint64_t> q4_layer_io_physical_bytes[128]{};
    std::atomic<uint64_t> q4_layer_io_logical_bytes[128]{};
    std::atomic<uint64_t> q4_layer_io_wait_us[128]{};
    std::atomic<uint64_t> q4_layer_io_read_us[128]{};
    std::atomic<uint64_t> q4_layer_io_h2d_us[128]{};
    std::atomic<uint64_t> q4_layer_io_failures[128]{};
    std::atomic<uint64_t> q4_pilot_prefetch_calls{0};
    std::atomic<uint64_t> q4_pilot_prefetch_candidates{0};
    std::atomic<uint64_t> q4_pilot_prefetch_submitted_objects{0};
    std::atomic<uint64_t> q4_pilot_prefetch_submitted_roles{0};
    std::atomic<uint64_t> q4_pilot_prefetch_completed_objects{0};
    std::atomic<uint64_t> q4_pilot_prefetch_failed_objects{0};
    std::atomic<uint64_t> q4_pilot_prefetch_used_roles{0};
    std::atomic<uint64_t> q4_pilot_prefetch_skipped_missing_current{0};
    std::atomic<uint64_t> q4_pilot_prefetch_skipped_missing_next{0};
    std::atomic<uint64_t> q4_pilot_prefetch_skipped_last_layer{0};
    std::atomic<uint64_t> q4_pilot_prefetch_skipped_vram_present_objects{0};
    std::atomic<uint64_t> q4_pilot_prefetch_skipped_layer_filtered_objects{0};
    std::atomic<uint64_t> q4_pilot_prefetch_queue_full{0};
    std::atomic<uint64_t> q4_pilot_prefetch_bytes{0};
    std::atomic<uint64_t> q4_pilot_prefetch_precision_submitted_objects{0};
    std::atomic<uint64_t> q4_pilot_prefetch_precision_submitted_bytes{0};
    std::atomic<uint64_t> q4_pilot_prefetch_precision_hit_objects{0};
    std::atomic<uint64_t> q4_pilot_prefetch_precision_hit_roles{0};
    std::atomic<uint64_t> q4_pilot_prefetch_precision_hit_bytes{0};
    std::atomic<uint64_t> q4_pilot_prefetch_precision_layer_submitted_objects[128]{};
    std::atomic<uint64_t> q4_pilot_prefetch_precision_layer_submitted_bytes[128]{};
    std::atomic<uint64_t> q4_pilot_prefetch_precision_layer_hit_objects[128]{};
    std::atomic<uint64_t> q4_pilot_prefetch_precision_layer_hit_roles[128]{};
    std::atomic<uint64_t> q4_pilot_prefetch_precision_layer_hit_bytes[128]{};
    std::atomic<uint64_t> q4_object_ram_hits{0};
    std::atomic<uint64_t> q4_object_ram_misses{0};
    std::atomic<uint64_t> q4_object_ram_admissions{0};
    std::atomic<uint64_t> q4_object_ram_repins{0};
    std::atomic<uint64_t> q4_object_ram_demotions{0};
    std::atomic<uint64_t> q4_object_ram_evictions{0};
    std::atomic<uint64_t> q4_object_ram_bytes_avoided{0};
    std::atomic<uint64_t> q4_object_ram_wait_us_avoided{0};
    std::atomic<uint64_t> q4_object_ram_h2d_us{0};
    std::atomic<uint64_t> q4_object_ram_h2d_bytes{0};
    std::atomic<uint64_t> q4_object_ssd_reads{0};
    std::atomic<uint64_t> q4_object_ssd_bytes{0};
    std::atomic<uint64_t> q4_object_vram_hits{0};
    std::atomic<uint64_t> q4_object_vram_misses{0};
    std::atomic<uint64_t> q4_object_full_expert_hits{0};
    std::atomic<uint64_t> q4_object_role_hits[3];
    std::atomic<uint64_t> q4_object_role_misses[3];
    std::atomic<uint64_t> q4_object_layer_ram_hits[64];
    std::atomic<uint64_t> q4_object_layer_ram_misses[64];
    std::atomic<uint64_t> q4_object_layer_ram_bytes_avoided[64];
    std::atomic<uint64_t> q4_object_layer_ram_wait_us_avoided[64];
    std::atomic<uint64_t> q4_object_layer_ram_h2d_us[64];
    std::atomic<uint64_t> q4_object_layer_ram_h2d_bytes[64];
    std::atomic<uint64_t> q4_object_layer_vram_hits[64];
    std::atomic<uint64_t> q4_object_layer_vram_misses[64];
    std::atomic<uint64_t> q4_host_object_read_admissions{0};
    std::atomic<uint64_t> q4_host_object_read_admit_bytes{0};
    std::atomic<uint64_t> q4_host_object_read_admit_evictions{0};
    std::atomic<uint64_t> q4_host_object_read_admit_alloc_failures{0};
    bool reported_io_backend = false;
#if defined(GGML_SSD_STORE)
    int & io_backend = c0::shared_catalog().io_backend;
#else
    int io_backend = 0; // 0=buffered, 1=direct, 2=io_uring
#endif
    std::mutex mu;
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> read_failures{0};
    std::atomic<uint64_t> direct_reads{0};
    std::atomic<uint64_t> direct_fallbacks{0};
    ggml_moe_common::object_io_runtime_counters io_stats;
    std::atomic<uint64_t> iouring_h2d_enqueues{0};
    std::atomic<uint64_t> iouring_batch_hist_1{0};
    std::atomic<uint64_t> iouring_batch_hist_2_4{0};
    std::atomic<uint64_t> iouring_batch_hist_5_8{0};
    std::atomic<uint64_t> iouring_batch_hist_9_16{0};
    std::atomic<uint64_t> iouring_batch_hist_17_32{0};
    std::atomic<uint64_t> iouring_batch_hist_gt32{0};
    std::atomic<uint64_t> adjacent_coalesce_groups{0};
    std::atomic<uint64_t> adjacent_coalesce_slices{0};
    std::atomic<uint64_t> adjacent_coalesce_extents_saved{0};
    std::atomic<uint64_t> adjacent_coalesce_physical_bytes{0};
    std::atomic<uint64_t> adjacent_coalesce_payload_bytes{0};
    std::atomic<uint64_t> adjacent_coalesce_max_group_jobs{0};
    std::atomic<uint64_t> adjacent_coalesce_max_span_bytes{0};
    std::atomic<uint64_t> adjacent_coalesce_reject_source{0};
    std::atomic<uint64_t> adjacent_coalesce_reject_tensor{0};
    std::atomic<uint64_t> adjacent_coalesce_reject_gap{0};
    std::atomic<uint64_t> adjacent_coalesce_reject_span{0};
    expert_pack_adjacent_op_stats adjacent_coalesce_op[3];
    // RAM hot tier
    void *ram_tier_base = nullptr;
    size_t ram_tier_bytes = 0;
    struct ram_tier_entry { size_t offset; size_t nbytes; };
    std::vector<ram_tier_entry> ram_tier_index; // parallel to entries[], non-zero offset = resident
    size_t ram_tier_pinned_bytes = 0;
    std::atomic<uint64_t> ram_tier_hits{0};
    std::atomic<uint64_t> ram_tier_total{0};
    std::atomic<uint64_t> ram_tier_fadvise_bytes{0};
    std::atomic<uint64_t> ram_tier_fadvise_failures{0};
    std::atomic<uint64_t> ram_tier_direct_bytes{0};
    std::atomic<uint64_t> ram_tier_direct_failures{0};
    std::atomic<uint64_t> ram_tier_direct_fallbacks{0};
    std::atomic<uint64_t> ram_tier_h2d_enqueues{0};
    std::atomic<uint64_t> ram_tier_h2d_bytes{0};
    std::atomic<uint64_t> ram_tier_batch_calls{0};
    std::atomic<uint64_t> ram_tier_batch_jobs{0};
    std::atomic<uint64_t> ram_tier_batch_hits{0};
    std::atomic<uint64_t> ram_tier_batch_bytes{0};
    std::atomic<uint64_t> page_cache_reads{0};
    std::atomic<uint64_t> page_cache_bytes{0};
    std::atomic<uint64_t> page_cache_preload_bytes{0};
    std::atomic<uint64_t> page_cache_preload_failures{0};
    std::atomic<uint64_t> page_cache_sources{0};
};

static expert_pack_state g_expert_pack;

static bool expert_pack_catalog_ready() {
#if defined(GGML_SSD_STORE)
    return c0::catalog_available();
#else
    return g_expert_pack.enabled;
#endif
}

static size_t expert_pack_published_entry_count() {
#if defined(GGML_SSD_STORE)
    if (!c0::catalog_available()) return 0;
#endif
    return g_expert_pack.entries.size();
}


static void q4_object_layer_add(std::atomic<uint64_t> (&counters)[64], int layer, uint64_t value) {
    if (value == 0 || layer < 0 || layer >= 64) return;
    counters[layer].fetch_add(value, std::memory_order_relaxed);
}

static size_t expert_pack_iouring_batch_bucket(size_t jobs);

struct global_sched_shadow_key {
    int32_t source_idx = 0;
    uint64_t offset = 0;
    uint64_t nbytes = 0;

    bool operator==(const global_sched_shadow_key &other) const {
        return source_idx == other.source_idx && offset == other.offset && nbytes == other.nbytes;
    }
};

struct global_sched_shadow_key_hash {
    size_t operator()(const global_sched_shadow_key &key) const {
        uint64_t h = (uint64_t)(uint32_t)key.source_idx;
        h ^= key.offset + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= key.nbytes + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return (size_t)h;
    }
};

struct global_sched_shadow_active {
    uint32_t demand = 0;
    uint32_t prefetch = 0;
};

struct global_sched_shadow_profile {
    std::atomic<uint64_t> batches{0};
    std::atomic<uint64_t> tasks{0};
    std::atomic<uint64_t> demand_tasks{0};
    std::atomic<uint64_t> prefetch_tasks{0};
    std::atomic<uint64_t> active_duplicates{0};
    std::atomic<uint64_t> demand_hits_active_prefetch{0};
    std::atomic<uint64_t> prefetch_hits_active_demand{0};
    std::atomic<uint64_t> demand_hits_active_demand{0};
    std::atomic<uint64_t> prefetch_hits_active_prefetch{0};
    std::atomic<uint64_t> max_active{0};
    std::atomic<uint64_t> max_batch{0};
    std::atomic<uint64_t> batch_hist_1{0};
    std::atomic<uint64_t> batch_hist_2_4{0};
    std::atomic<uint64_t> batch_hist_5_8{0};
    std::atomic<uint64_t> batch_hist_9_16{0};
    std::atomic<uint64_t> batch_hist_17_32{0};
    std::atomic<uint64_t> batch_hist_gt32{0};
};

static global_sched_shadow_profile g_global_sched_shadow;
static std::mutex g_global_sched_shadow_mu;
static std::unordered_map<global_sched_shadow_key, global_sched_shadow_active, global_sched_shadow_key_hash> g_global_sched_shadow_active;
static std::atomic<bool> g_global_sched_shadow_report_registered{false};

static bool global_sched_shadow_enabled() {
    const char *env = std::getenv("GGML_MOE_GLOBAL_EXPERT_SCHED_SHADOW");
    return env && env[0] && env[0] != '0';
}

static bool global_sched_shadow_is_prefetch(const char *trace_op) {
    return trace_op && (
        std::strstr(trace_op, "prefetch") ||
        std::strstr(trace_op, "overlap"));
}

static void global_sched_shadow_atomic_max(std::atomic<uint64_t> &target, uint64_t value) {
    uint64_t current = target.load(std::memory_order_relaxed);
    while (current < value &&
            !target.compare_exchange_weak(current, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

static void global_sched_shadow_report_atexit() {
    const uint64_t batches = g_global_sched_shadow.batches.load();
    if (batches == 0) return;
    std::fprintf(stderr,
            "[moe_stream_batch] global expert sched shadow: batches=%lu tasks=%lu demand=%lu prefetch=%lu "
            "active_duplicates=%lu demand_hit_prefetch=%lu prefetch_hit_demand=%lu "
            "demand_hit_demand=%lu prefetch_hit_prefetch=%lu max_active=%lu max_batch=%lu "
            "batch_hist=1:%lu,2-4:%lu,5-8:%lu,9-16:%lu,17-32:%lu,gt32:%lu\n",
            (unsigned long)batches,
            (unsigned long)g_global_sched_shadow.tasks.load(),
            (unsigned long)g_global_sched_shadow.demand_tasks.load(),
            (unsigned long)g_global_sched_shadow.prefetch_tasks.load(),
            (unsigned long)g_global_sched_shadow.active_duplicates.load(),
            (unsigned long)g_global_sched_shadow.demand_hits_active_prefetch.load(),
            (unsigned long)g_global_sched_shadow.prefetch_hits_active_demand.load(),
            (unsigned long)g_global_sched_shadow.demand_hits_active_demand.load(),
            (unsigned long)g_global_sched_shadow.prefetch_hits_active_prefetch.load(),
            (unsigned long)g_global_sched_shadow.max_active.load(),
            (unsigned long)g_global_sched_shadow.max_batch.load(),
            (unsigned long)g_global_sched_shadow.batch_hist_1.load(),
            (unsigned long)g_global_sched_shadow.batch_hist_2_4.load(),
            (unsigned long)g_global_sched_shadow.batch_hist_5_8.load(),
            (unsigned long)g_global_sched_shadow.batch_hist_9_16.load(),
            (unsigned long)g_global_sched_shadow.batch_hist_17_32.load(),
            (unsigned long)g_global_sched_shadow.batch_hist_gt32.load());
}

struct global_sched_shadow_scope {
    struct entry {
        global_sched_shadow_key key;
        bool prefetch = false;
    };

    bool enabled = false;
    std::vector<entry> entries;

    global_sched_shadow_scope() = default;

    global_sched_shadow_scope(const global_sched_shadow_scope &) = delete;
    global_sched_shadow_scope &operator=(const global_sched_shadow_scope &) = delete;

    ~global_sched_shadow_scope() {
        if (!enabled || entries.empty()) return;
        std::lock_guard<std::mutex> lk(g_global_sched_shadow_mu);
        for (const entry &e : entries) {
            auto it = g_global_sched_shadow_active.find(e.key);
            if (it == g_global_sched_shadow_active.end()) {
                continue;
            }
            if (e.prefetch) {
                if (it->second.prefetch > 0) --it->second.prefetch;
            } else {
                if (it->second.demand > 0) --it->second.demand;
            }
            if (it->second.demand == 0 && it->second.prefetch == 0) {
                g_global_sched_shadow_active.erase(it);
            }
        }
    }

    void begin(const std::vector<global_sched_shadow_key> &keys, bool prefetch) {
        if (keys.empty() || !global_sched_shadow_enabled()) return;
        enabled = true;
        if (!g_global_sched_shadow_report_registered.exchange(true)) {
            std::atexit(global_sched_shadow_report_atexit);
        }
        ++g_global_sched_shadow.batches;
        g_global_sched_shadow.tasks.fetch_add(keys.size());
        if (prefetch) {
            g_global_sched_shadow.prefetch_tasks.fetch_add(keys.size());
        } else {
            g_global_sched_shadow.demand_tasks.fetch_add(keys.size());
        }
        global_sched_shadow_atomic_max(g_global_sched_shadow.max_batch, keys.size());
        switch (expert_pack_iouring_batch_bucket(keys.size())) {
            case 0: ++g_global_sched_shadow.batch_hist_1; break;
            case 1: ++g_global_sched_shadow.batch_hist_2_4; break;
            case 2: ++g_global_sched_shadow.batch_hist_5_8; break;
            case 3: ++g_global_sched_shadow.batch_hist_9_16; break;
            case 4: ++g_global_sched_shadow.batch_hist_17_32; break;
            default: ++g_global_sched_shadow.batch_hist_gt32; break;
        }

        std::lock_guard<std::mutex> lk(g_global_sched_shadow_mu);
        entries.reserve(keys.size());
        for (const global_sched_shadow_key &key : keys) {
            global_sched_shadow_active &active = g_global_sched_shadow_active[key];
            const bool has_demand = active.demand > 0;
            const bool has_prefetch = active.prefetch > 0;
            if (has_demand || has_prefetch) {
                ++g_global_sched_shadow.active_duplicates;
                if (prefetch) {
                    if (has_demand) ++g_global_sched_shadow.prefetch_hits_active_demand;
                    if (has_prefetch) ++g_global_sched_shadow.prefetch_hits_active_prefetch;
                } else {
                    if (has_prefetch) ++g_global_sched_shadow.demand_hits_active_prefetch;
                    if (has_demand) ++g_global_sched_shadow.demand_hits_active_demand;
                }
            }
            if (prefetch) {
                ++active.prefetch;
            } else {
                ++active.demand;
            }
            entries.push_back({key, prefetch});
        }
        global_sched_shadow_atomic_max(g_global_sched_shadow.max_active, g_global_sched_shadow_active.size());
    }
};

struct expert_pack_v2_entry {
    char tensor[128] = {};
    int32_t expert_idx = -1;
    int32_t packed_type = 0;
    int32_t source_idx = 0;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
    int64_t ne00 = 0;
    int64_t ne01 = 0;
    uint64_t nb01 = 0;
    uint64_t reserved = 0;
};

struct expert_pack_v2_state {
    bool inited = false;
    bool enabled = false;
    std::mutex mu;
    std::vector<expert_pack_source> sources;
    std::vector<expert_pack_v2_entry> entries;
};

static expert_pack_v2_state g_expert_pack_v2;

struct direct_read_site_stat {
    uint64_t calls = 0;
    uint64_t failures = 0;
    uint64_t bytes = 0;
    double wall_ms = 0.0;
};

static std::mutex g_direct_read_site_mu;
static std::unordered_map<std::string, direct_read_site_stat> g_direct_read_site_stats;
static std::atomic<bool> g_direct_read_site_registered{false};
static thread_local const char *g_direct_read_site_op = nullptr;
static thread_local const char *g_direct_read_site_tensor = nullptr;

struct direct_read_site_context_scope {
    const char *prev_op = nullptr;
    const char *prev_tensor = nullptr;

    direct_read_site_context_scope(const char *op, const char *tensor) {
        prev_op = g_direct_read_site_op;
        prev_tensor = g_direct_read_site_tensor;
        g_direct_read_site_op = op;
        g_direct_read_site_tensor = tensor;
    }

    ~direct_read_site_context_scope() {
        g_direct_read_site_op = prev_op;
        g_direct_read_site_tensor = prev_tensor;
    }
};

static const char *direct_read_site_profile_path() {
    static std::once_flag once;
    static const char *cached_path = nullptr;
    std::call_once(once, []() {
        const char *path = std::getenv("GGML_MOE_DIRECT_READ_SITE_PROFILE_OUT");
        if (path && path[0]) {
            cached_path = ::strdup(path);
            return;
        }
        const char *ttft_path = std::getenv("GGML_MOE_TTFT_TRACE_OUT");
        if (ttft_path && ttft_path[0]) {
            std::string derived_path = ttft_path;
            const size_t slash = derived_path.find_last_of('/');
            if (slash == std::string::npos) {
                derived_path = "direct-read-site-profile.csv";
            } else {
                derived_path.resize(slash + 1);
                derived_path += "direct-read-site-profile.csv";
            }
            cached_path = ::strdup(derived_path.c_str());
        }
    });
    return cached_path;
}

static const char *direct_read_site_role(const char *tensor) {
    if (!tensor) return "unknown";
    if (std::strstr(tensor, "ffn_up")) return "up";
    if (std::strstr(tensor, "ffn_gate")) return "gate";
    if (std::strstr(tensor, "ffn_down")) return "down";
    return "other";
}

static void direct_read_site_profile_write() {
    const char *path = direct_read_site_profile_path();
    if (!path) return;
    std::lock_guard<std::mutex> lk(g_direct_read_site_mu);
    FILE *f = std::fopen(path, "w");
    if (!f) return;
    std::fprintf(f, "op,role,tensor,expert_bytes,calls,failures,bytes,wall_ms\n");
    for (const auto &it : g_direct_read_site_stats) {
        const direct_read_site_stat &s = it.second;
        std::fprintf(f, "%s,%lu,%lu,%lu,%.3f\n",
                it.first.c_str(),
                (unsigned long)s.calls,
                (unsigned long)s.failures,
                (unsigned long)s.bytes,
                s.wall_ms);
    }
    std::fclose(f);
}

static void direct_read_site_profile_record(
        const char *op, const char *tensor, size_t nbytes, bool ok, double wall_ms) {
    if (!direct_read_site_profile_path()) return;
    if (!g_direct_read_site_registered.exchange(true)) {
        std::atexit(direct_read_site_profile_write);
    }
    char key[384];
    std::snprintf(key, sizeof(key), "%s,%s,%s,%lu",
            op && op[0] ? op : "unknown",
            direct_read_site_role(tensor),
            tensor && tensor[0] ? tensor : "",
            (unsigned long)nbytes);
    std::lock_guard<std::mutex> lk(g_direct_read_site_mu);
    direct_read_site_stat &s = g_direct_read_site_stats[key];
    ++s.calls;
    if (!ok) ++s.failures;
    if (ok) s.bytes += nbytes;
    s.wall_ms += wall_ms;
}

static void batch_profile_report_atexit() {
    if (!g_bprof.enabled || g_bprof.calls == 0) return;
    const double calls = (double)g_bprof.calls;
    const double accounted_ms =
        g_bprof.stage_ms + g_bprof.quant_ms + g_bprof.kernel_ms + g_bprof.d2h_ms + g_bprof.scatter_ms;
    const double wall_gap_ms = g_bprof.wall_ms - accounted_ms;
    std::fprintf(stderr,
        "[moe_stream_batch] profile: calls=%lu avg_active=%.2f "
        "stage=%.3f ms quant=%.3f ms kernel=%.3f ms d2h=%.3f ms scatter=%.3f ms "
        "total=%.3f ms/call wall=%.3f ms/call wall_gap=%.3f ms/call\n",
        g_bprof.calls,
        (double)g_bprof.active_experts / calls,
        g_bprof.stage_ms / calls,
        g_bprof.quant_ms / calls,
        g_bprof.kernel_ms / calls,
        g_bprof.d2h_ms / calls,
        g_bprof.scatter_ms / calls,
        accounted_ms / calls,
        g_bprof.wall_ms / calls,
        wall_gap_ms / calls);
}

static void up_gate_profile_report_atexit() {
    if (!g_uprof.enabled || g_uprof.calls == 0) return;
    const double calls = (double)g_uprof.calls;
    const double accounted_ms =
        g_uprof.stage_ms + g_uprof.quant_ms + g_uprof.kernel_ms + g_uprof.d2h_ms + g_uprof.scatter_ms;
    const double wall_gap_ms = g_uprof.wall_ms - accounted_ms;
    std::fprintf(stderr,
        "[moe_stream_batch] up/gate profile: calls=%lu avg_active=%.2f "
        "stage=%.3f ms quant=%.3f ms up=%.3f ms gate=%.3f ms "
        "up_stage_jobs=%.2f gate_stage_jobs=%.2f "
        "up_wait=%.3f ms gate_wait=%.3f ms up_compute=%.3f ms gate_compute=%.3f ms fuse=%.3f ms "
        "kernel=%.3f ms d2h=%.3f ms scatter=%.3f ms "
        "total=%.3f ms/call wall=%.3f ms/call wall_gap=%.3f ms/call\n",
        g_uprof.calls,
        (double)g_uprof.active_experts / calls,
        g_uprof.stage_ms / calls,
        g_uprof.quant_ms / calls,
        g_uprof.up_ms / calls,
        g_uprof.gate_ms / calls,
        (double)g_uprof.up_stage_jobs / calls,
        (double)g_uprof.gate_stage_jobs / calls,
        g_uprof.up_wait_ms / calls,
        g_uprof.gate_wait_ms / calls,
        g_uprof.up_compute_ms / calls,
        g_uprof.gate_compute_ms / calls,
        g_uprof.fuse_ms / calls,
        g_uprof.kernel_ms / calls,
        g_uprof.d2h_ms / calls,
        g_uprof.scatter_ms / calls,
        accounted_ms / calls,
        g_uprof.wall_ms / calls,
        wall_gap_ms / calls);
    std::fprintf(stderr,
        "[moe_stream_batch] up/gate stage-job hist: "
        "max_up=%lu max_gate=%lu max_combined=%lu "
        "up=0:%lu,1:%lu,2-4:%lu,5-8:%lu,9-16:%lu,17-32:%lu,gt32:%lu "
        "gate=0:%lu,1:%lu,2-4:%lu,5-8:%lu,9-16:%lu,17-32:%lu,gt32:%lu "
        "combined=0:%lu,1:%lu,2-4:%lu,5-8:%lu,9-16:%lu,17-32:%lu,gt32:%lu\n",
        g_uprof.max_up_stage_jobs,
        g_uprof.max_gate_stage_jobs,
        g_uprof.max_combined_stage_jobs,
        g_uprof.up_stage_jobs_hist[0], g_uprof.up_stage_jobs_hist[1],
        g_uprof.up_stage_jobs_hist[2], g_uprof.up_stage_jobs_hist[3],
        g_uprof.up_stage_jobs_hist[4], g_uprof.up_stage_jobs_hist[5],
        g_uprof.up_stage_jobs_hist[6],
        g_uprof.gate_stage_jobs_hist[0], g_uprof.gate_stage_jobs_hist[1],
        g_uprof.gate_stage_jobs_hist[2], g_uprof.gate_stage_jobs_hist[3],
        g_uprof.gate_stage_jobs_hist[4], g_uprof.gate_stage_jobs_hist[5],
        g_uprof.gate_stage_jobs_hist[6],
        g_uprof.combined_stage_jobs_hist[0], g_uprof.combined_stage_jobs_hist[1],
        g_uprof.combined_stage_jobs_hist[2], g_uprof.combined_stage_jobs_hist[3],
        g_uprof.combined_stage_jobs_hist[4], g_uprof.combined_stage_jobs_hist[5],
        g_uprof.combined_stage_jobs_hist[6]);
    for (int mode = 0; mode < 2; ++mode) {
        for (int up_type = 0; up_type < GGML_TYPE_COUNT; ++up_type) {
            for (int gate_type = 0; gate_type < GGML_TYPE_COUNT; ++gate_type) {
                const batch_profile &p = g_uprof_by_type[mode][up_type][gate_type];
                if (p.calls == 0) {
                    continue;
                }
                const double pcalls = (double)p.calls;
                const double p_accounted_ms =
                    p.stage_ms + p.quant_ms + p.kernel_ms + p.d2h_ms + p.scatter_ms;
                const double p_wall_gap_ms = p.wall_ms - p_accounted_ms;
                std::fprintf(stderr,
                    "[moe_stream_batch] up/gate type profile: mode=%s up_type=%d gate_type=%d "
                    "calls=%lu avg_active=%.2f stage=%.3f ms quant=%.3f ms "
                    "up=%.3f ms gate=%.3f ms up_stage_jobs=%.2f gate_stage_jobs=%.2f "
                    "up_wait=%.3f ms gate_wait=%.3f ms up_compute=%.3f ms gate_compute=%.3f ms "
                    "fuse=%.3f ms kernel=%.3f ms d2h=%.3f ms scatter=%.3f ms "
                    "total=%.3f ms/call wall=%.3f ms/call wall_gap=%.3f ms/call\n",
                    mode ? "prompt" : "decode",
                    up_type,
                    gate_type,
                    p.calls,
                    (double)p.active_experts / pcalls,
                    p.stage_ms / pcalls,
                    p.quant_ms / pcalls,
                    p.up_ms / pcalls,
                    p.gate_ms / pcalls,
                    (double)p.up_stage_jobs / pcalls,
                    (double)p.gate_stage_jobs / pcalls,
                    p.up_wait_ms / pcalls,
                    p.gate_wait_ms / pcalls,
                    p.up_compute_ms / pcalls,
                    p.gate_compute_ms / pcalls,
                    p.fuse_ms / pcalls,
                    p.kernel_ms / pcalls,
                    p.d2h_ms / pcalls,
                    p.scatter_ms / pcalls,
                    p_accounted_ms / pcalls,
                    p.wall_ms / pcalls,
                    p_wall_gap_ms / pcalls);
                std::fprintf(stderr,
                    "[moe_stream_batch] up/gate type stage-job hist: mode=%s up_type=%d gate_type=%d "
                    "max_up=%lu max_gate=%lu max_combined=%lu "
                    "up=0:%lu,1:%lu,2-4:%lu,5-8:%lu,9-16:%lu,17-32:%lu,gt32:%lu "
                    "gate=0:%lu,1:%lu,2-4:%lu,5-8:%lu,9-16:%lu,17-32:%lu,gt32:%lu "
                    "combined=0:%lu,1:%lu,2-4:%lu,5-8:%lu,9-16:%lu,17-32:%lu,gt32:%lu\n",
                    mode ? "prompt" : "decode",
                    up_type,
                    gate_type,
                    p.max_up_stage_jobs,
                    p.max_gate_stage_jobs,
                    p.max_combined_stage_jobs,
                    p.up_stage_jobs_hist[0], p.up_stage_jobs_hist[1],
                    p.up_stage_jobs_hist[2], p.up_stage_jobs_hist[3],
                    p.up_stage_jobs_hist[4], p.up_stage_jobs_hist[5],
                    p.up_stage_jobs_hist[6],
                    p.gate_stage_jobs_hist[0], p.gate_stage_jobs_hist[1],
                    p.gate_stage_jobs_hist[2], p.gate_stage_jobs_hist[3],
                    p.gate_stage_jobs_hist[4], p.gate_stage_jobs_hist[5],
                    p.gate_stage_jobs_hist[6],
                    p.combined_stage_jobs_hist[0], p.combined_stage_jobs_hist[1],
                    p.combined_stage_jobs_hist[2], p.combined_stage_jobs_hist[3],
                    p.combined_stage_jobs_hist[4], p.combined_stage_jobs_hist[5],
                    p.combined_stage_jobs_hist[6]);
            }
        }
    }
}

static double q4_profile_wall_gap_ms(const batch_profile &p) {
    const double accounted_ms =
        p.stage_ms + p.quant_ms + p.kernel_ms + p.d2h_ms + p.scatter_ms;
    return p.wall_ms - accounted_ms;
}

static double q4_graph_launch_overhead_us_assumption() {
    const char *env = std::getenv("GGML_MOE_Q4_GRAPH_LAUNCH_OVERHEAD_US");
    if (!env || !env[0]) return 20.0;
    const double value = std::atof(env);
    return value > 0.0 ? value : 20.0;
}

static void q4_graph_compute_region_profile_report_atexit() {
    if (!q4_graph_compute_region_profile_enabled()) return;
    const uint64_t up_calls = g_uprof.calls;
    const uint64_t down_calls = g_bprof.calls;
    if (up_calls + down_calls == 0) return;

    const double up_wall_gap_ms = q4_profile_wall_gap_ms(g_uprof);
    const double down_wall_gap_ms = q4_profile_wall_gap_ms(g_bprof);
    const double provider_stage_ms = g_uprof.stage_ms + g_bprof.stage_ms;
    const double cuda_compute_ms = g_uprof.kernel_ms + g_bprof.kernel_ms;
    const double graph_boundary_ms =
        g_uprof.d2h_ms + g_bprof.d2h_ms +
        g_uprof.scatter_ms + g_bprof.scatter_ms +
        up_wall_gap_ms + down_wall_gap_ms;
    const double launch_us = q4_graph_launch_overhead_us_assumption();
    const double replay_launch_bound_ms = (double)(up_calls + down_calls) * launch_us / 1000.0;
    const double total_wall_ms = g_uprof.wall_ms + g_bprof.wall_ms;
    const double total_calls = (double)(up_calls + down_calls);
    const double graph_unavailable_ms =
        provider_stage_ms + graph_boundary_ms + g_uprof.quant_ms + g_bprof.quant_ms;

    std::fprintf(stderr,
        "[moe_stream_batch] q4 graph compute-region profile: enabled=1 "
        "upgate_calls=%lu down_calls=%lu total_calls=%lu "
        "provider_stage_ms=%.3f cuda_compute_ms=%.3f graph_boundary_ms=%.3f "
        "quant_ms=%.3f wall_gap_ms=%.3f total_wall_ms=%.3f "
        "avg_provider_stage_ms=%.3f avg_cuda_compute_ms=%.3f avg_graph_boundary_ms=%.3f "
        "assumed_launch_us=%.2f replay_launch_bound_ms=%.3f "
        "non_replayable_ms=%.3f note=compute-only-graph-can-save-launch/scheduling-not-ssd-h2d-or-math\n",
        (unsigned long)up_calls,
        (unsigned long)down_calls,
        (unsigned long)(up_calls + down_calls),
        provider_stage_ms,
        cuda_compute_ms,
        graph_boundary_ms,
        g_uprof.quant_ms + g_bprof.quant_ms,
        up_wall_gap_ms + down_wall_gap_ms,
        total_wall_ms,
        total_calls > 0.0 ? provider_stage_ms / total_calls : 0.0,
        total_calls > 0.0 ? cuda_compute_ms / total_calls : 0.0,
        total_calls > 0.0 ? graph_boundary_ms / total_calls : 0.0,
        launch_us,
        replay_launch_bound_ms,
        graph_unavailable_ms);

    if (up_calls > 0) {
        const double calls = (double)up_calls;
        std::fprintf(stderr,
            "[moe_stream_batch] q4 graph compute-region upgate: calls=%lu "
            "stage_ms=%.3f quant_ms=%.3f up_ms=%.3f gate_ms=%.3f "
            "up_wait_ms=%.3f gate_wait_ms=%.3f up_compute_ms=%.3f gate_compute_ms=%.3f "
            "fuse_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f scatter_ms=%.3f wall_gap_ms=%.3f "
            "avg_kernel_ms=%.3f avg_wall_ms=%.3f\n",
            (unsigned long)up_calls,
            g_uprof.stage_ms,
            g_uprof.quant_ms,
            g_uprof.up_ms,
            g_uprof.gate_ms,
            g_uprof.up_wait_ms,
            g_uprof.gate_wait_ms,
            g_uprof.up_compute_ms,
            g_uprof.gate_compute_ms,
            g_uprof.fuse_ms,
            g_uprof.kernel_ms,
            g_uprof.d2h_ms,
            g_uprof.scatter_ms,
            up_wall_gap_ms,
            g_uprof.kernel_ms / calls,
            g_uprof.wall_ms / calls);
    }

    if (down_calls > 0) {
        const double calls = (double)down_calls;
        std::fprintf(stderr,
            "[moe_stream_batch] q4 graph compute-region down: calls=%lu "
            "stage_ms=%.3f quant_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f "
            "scatter_ms=%.3f wall_gap_ms=%.3f avg_kernel_ms=%.3f avg_wall_ms=%.3f\n",
            (unsigned long)down_calls,
            g_bprof.stage_ms,
            g_bprof.quant_ms,
            g_bprof.kernel_ms,
            g_bprof.d2h_ms,
            g_bprof.scatter_ms,
            down_wall_gap_ms,
            g_bprof.kernel_ms / calls,
            g_bprof.wall_ms / calls);
    }
}

static void up_gate_layer_profile_report_atexit() {
    if (!g_up_gate_layer_profile_enabled) {
        return;
    }
    std::fprintf(stderr,
        "[moe_stream_batch] up/gate layer profile summary: records=%lu\n",
        (unsigned long)g_up_gate_layer_profile_records.load(std::memory_order_relaxed));

    up_gate_layer_profile_entry entries[256];
    size_t n_entries = 0;
    {
        std::lock_guard<std::mutex> lk(g_up_gate_layer_profile_mu);
        for (const up_gate_layer_profile_entry &entry : g_up_gate_layer_profiles) {
            if (entry.used && entry.profile.calls > 0 && n_entries < 256) {
                entries[n_entries++] = entry;
            }
        }
    }

    if (n_entries == 0) {
        std::fprintf(stderr, "[moe_stream_batch] up/gate layer profile: no entries recorded\n");
        return;
    }

    std::sort(entries, entries + n_entries,
        [](const up_gate_layer_profile_entry &a, const up_gate_layer_profile_entry &b) {
            return a.profile.wall_ms > b.profile.wall_ms;
        });

    const char *limit_env = std::getenv("GGML_MOE_UP_GATE_LAYER_PROFILE_TOP");
    int limit = 32;
    if (limit_env && limit_env[0]) {
        limit = std::atoi(limit_env);
        if (limit <= 0) {
            limit = 32;
        }
    }

    int printed = 0;
    for (size_t i = 0; i < n_entries && printed < limit; ++i) {
        const up_gate_layer_profile_entry &entry = entries[i];
        const batch_profile &p = entry.profile;
        const double calls = (double)p.calls;
        const double accounted_ms = p.stage_ms + p.quant_ms + p.kernel_ms + p.d2h_ms + p.scatter_ms;
        const double wall_gap_ms = p.wall_ms - accounted_ms;
        std::fprintf(stderr,
            "[moe_stream_batch] up/gate layer profile: rank=%d mode=%s up_tensor=%s gate_tensor=%s "
            "up_type=%d gate_type=%d calls=%lu active=%lu avg_active=%.2f "
            "stage_total=%.3f quant_total=%.3f up_total=%.3f gate_total=%.3f "
            "up_wait_total=%.3f gate_wait_total=%.3f up_compute_total=%.3f gate_compute_total=%.3f "
            "fuse_total=%.3f kernel_total=%.3f d2h_total=%.3f scatter_total=%.3f wall_total=%.3f "
            "wall_per_call=%.3f kernel_per_call=%.3f wall_gap_total=%.3f "
            "up_stage_jobs=%lu gate_stage_jobs=%lu\n",
            printed + 1,
            entry.prompt_mode ? "prompt" : "decode",
            entry.up_tensor,
            entry.gate_tensor,
            (int)entry.up_type,
            (int)entry.gate_type,
            p.calls,
            p.active_experts,
            p.active_experts / calls,
            p.stage_ms,
            p.quant_ms,
            p.up_ms,
            p.gate_ms,
            p.up_wait_ms,
            p.gate_wait_ms,
            p.up_compute_ms,
            p.gate_compute_ms,
            p.fuse_ms,
            p.kernel_ms,
            p.d2h_ms,
            p.scatter_ms,
            p.wall_ms,
            p.wall_ms / calls,
            p.kernel_ms / calls,
            wall_gap_ms,
            p.up_stage_jobs,
            p.gate_stage_jobs);
        ++printed;
    }
}

struct batch_vram_cache {
    GGML_MOE_FIXED_ROLE_CACHE_IDENTITY_FIELDS(16384)
    std::unordered_map<uintptr_t, int> key_to_slot;
    uint64_t slot_used[16384] = {};
    uint32_t slot_hits[16384] = {};
    uint32_t slot_profile_count[16384] = {};
    char slot_tensor[16384][128] = {};
    int slot_expert[16384] = {};
    int slot_layer[16384] = {};
    bool slot_pinned[16384] = {};
    bool slot_prefetch_down[16384] = {};
    bool slot_q4_pilot[16384] = {};
    bool slot_object_admitted[16384] = {};
    GGML_MOE_FIXED_ROLE_CACHE_LIFECYCLE_FIELDS(16384)
    uint64_t clock = 1;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t preloads = 0;
    uint64_t pinned = 0;
    uint64_t down_prefetch_loads = 0;
    uint64_t down_prefetch_hits = 0;
    uint64_t down_prefetch_evicted = 0;
    uint64_t async_prefetch_waits = 0;
    uint64_t descriptor_acquires = 0;
    uint64_t descriptor_releases = 0;
    uint64_t descriptor_stale_releases = 0;
    uint64_t descriptor_lease_blocks = 0;
    uint64_t owner_invalidations = 0;
    uint64_t owner_invalidation_failures = 0;
};

static batch_vram_cache g_bcaches[2];
static bool g_bcache_inited[2] = {};

static int batch_cache_id_for_cache(const batch_vram_cache *cache) {
    for (int cache_id = 0; cache_id < 2; ++cache_id) {
        if (cache == &g_bcaches[cache_id]) return cache_id;
    }
    return -1;
}

static bool batch_cache_invalidate_role_owner(batch_vram_cache *cache, int slot) {
    if (!cache || slot < 0 || slot >= cache->n_slots) return false;
    if (cache->slot_key[slot] == 0) return true;
    const int cache_id = batch_cache_id_for_cache(cache);
    if (cache_id < 0) {
        ++cache->owner_invalidation_failures;
        return false;
    }
    ++cache->owner_invalidations;
    return true;
}

static ggml_moe_common::role_cache_view batch_cache_role_view(batch_vram_cache *cache) {
    ggml_moe_common::role_cache_view view;
    if (!cache) return view;
    view.pool = cache->pool;
    view.pool_bytes = cache->slot_sz * (size_t)std::max(cache->n_slots, 0);
    view.slot_bytes = cache->slot_sz;
    view.slot_count = cache->n_slots;
    view.slot_keys = cache->slot_key;
    view.slot_generations = cache->slot_generation;
    view.slot_payload_bytes = cache->slot_payload_bytes;
    view.slot_ready_markers = cache->slot_ready_marker;
    view.slot_states = cache->slot_state;
    view.slot_pending = cache->slot_pending;
    view.slot_filling = cache->slot_filling;
    view.slot_lease_counts = cache->slot_lease_count;
    view.slot_ready_events = cache->slot_ready;
    view.slot_lease_events = cache->slot_lease_ready;
    view.slot_lease_pending = cache->slot_lease_pending;
    return view;
}



static FILE * batch_vram_accounting_fp() {
    static FILE * fp = nullptr;
    static bool initialized = false;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    if (!initialized) {
        initialized = true;
        const char * path = std::getenv("GGML_MOE_VRAM_ACCOUNTING_OUT");
        if (path && path[0]) {
            fp = std::fopen(path, "a");
            if (fp) {
                std::setvbuf(fp, nullptr, _IOLBF, 0);
            } else {
                std::fprintf(stderr, "[moe_stream_batch] VRAM accounting: failed to open %s\n", path);
            }
        }
    }
    return fp;
}

static uint64_t batch_vram_accounting_now_us() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

static void batch_vram_accounting_record(const char * event, int cid, const batch_vram_cache * c, const char * detail = "") {
    FILE * fp = batch_vram_accounting_fp();
    if (!fp) return;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return;
    }
    const double mib = 1024.0 * 1024.0;
    const double pool_mib = c && c->pool ? ((double)c->n_slots * (double)c->slot_sz) / mib : 0.0;
    std::fprintf(fp,
        "%llu,batch_%s,%s,%.3f,%.3f,%.3f,%.3f,%d,%.3f,%s\n",
        (unsigned long long) batch_vram_accounting_now_us(),
        cid == 1 ? "upgate" : "down",
        event ? event : "",
        free_bytes / mib,
        total_bytes / mib,
        (total_bytes - free_bytes) / mib,
        pool_mib,
        c ? c->n_slots : 0,
        c ? c->slot_sz / mib : 0.0,
        detail ? detail : "");
}

struct cache_policy_diag {
    std::atomic<uint64_t> profile_count_lookups{0};
    std::atomic<uint64_t> profile_count_hits{0};
    std::atomic<uint64_t> inserted_profile_count_sum{0};
    std::atomic<uint64_t> inserted_profile_count_nonzero{0};
    std::atomic<uint64_t> profile_policy_evictions{0};
    std::atomic<uint64_t> profile_policy_victim_count_sum{0};
    std::atomic<uint64_t> profile_policy_victim_count_nonzero{0};
};

static cache_policy_diag g_cache_policy_diag;
static std::atomic<bool> g_cache_policy_diag_registered{false};

struct profile_entry {
    int expert_idx = -1;
    uint64_t count = 0;
    size_t expert_bytes = 0;
    char tensor[128] = {};
};

static std::vector<profile_entry> g_profile;
static std::unordered_map<uint64_t, uint32_t> g_profile_counts;
static std::unordered_set<uint64_t> g_profile_pinned_keys;
static bool g_profile_loaded = false;
static bool g_profile_enabled = false;
static std::vector<profile_entry> g_prompt_profile;
static bool g_prompt_profile_loaded = false;
static bool g_prompt_profile_enabled = false;
static std::mutex g_profile_mu;
static std::mutex g_profile_pinned_mu;
static char g_preloaded_tensors[256][128] = {};
static int g_n_preloaded_tensors = 0;
static std::atomic<int> g_profile_preload_calls{0};

struct batch_route_profile_entry {
    int expert_idx = -1;
    uint64_t count = 0;
    size_t expert_bytes = 0;
    char tensor[128] = {};
};

struct batch_route_trace_entry {
    uint64_t seq = 0;
    int expert_idx = -1;
    size_t expert_bytes = 0;
    char tensor[128] = {};
};

struct batch_route_detail_entry {
    uint64_t seq = 0;
    uint64_t call = 0;
    int layer = -1;
    int active_index = -1;
    int expert_idx = -1;
    int dst_id = -1;
    int flat_dst_id = -1;
    int token_id = -1;
    int n_active = 0;
    int src0_type = -1;
    int64_t ne01 = 0;
    int64_t ne00 = 0;
    size_t expert_bytes = 0;
    char mode[12] = {};
    char kind[12] = {};
    char tensor[128] = {};
};

struct trace_prefetch_state {
    bool inited = false;
    bool enabled = false;
    bool reported = false;
    size_t cursor = 0;
    size_t prefetch_cursor = 0;
    size_t lead_events = 0;
    size_t window = 96;
    size_t reuse_window = 96;
    int max_loads = 8;
    int min_future_uses = 1;
    char targets[32] = {};
    std::vector<batch_route_trace_entry> trace;
    std::mutex mu;
    uint64_t calls = 0;
    uint64_t matched = 0;
    uint64_t resync = 0;
    uint64_t loads = 0;
    uint64_t cached = 0;
    uint64_t missing_tensor = 0;
    uint64_t cache_unavailable = 0;
    uint64_t reuse_skips = 0;
};

struct host_prefetch_slot {
    const expert_pack_entry *entry = nullptr;
    size_t nbytes = 0;
    char tensor[128] = {};
    int expert_idx = -1;
    bool ready = false;
    bool reserved = false;
    bool in_use = false;
};

struct host_prefetch_state {
    bool inited = false;
    bool enabled = false;
    bool planned_enabled = false;
    bool stop = false;
    bool failed = false;
    size_t cursor = 0;
    size_t produce_cursor = 0;
    size_t skip_events = 0;
    size_t lead_events = 2048;
    size_t max_bytes = 512ULL * 1024ULL * 1024ULL;
    size_t used_bytes = 0;
    ggml_moe_common::object_stage_ring stage_ring;
    std::vector<batch_route_trace_entry> trace;
    std::deque<batch_route_trace_entry> planned;
    std::vector<host_prefetch_slot> slots;
    std::unordered_map<uint64_t, size_t> ready;
    std::unordered_map<uint64_t, size_t> planned_queued;
    std::mutex mu;
    std::condition_variable cv;
    std::thread worker;
    std::atomic<uint64_t> ready_count{0};
    std::atomic<uint64_t> empty_misses{0};
    uint64_t calls = 0;
    uint64_t matched = 0;
    uint64_t resync = 0;
    uint64_t submitted = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evicted = 0;
    uint64_t read_failures = 0;
    uint64_t alloc_failures = 0;
    uint64_t duplicate_skips = 0;
    uint64_t profile_pinned_skips = 0;
    uint64_t no_slot = 0;
    uint64_t scan_passes = 0;
    uint64_t reserved_skips = 0;
    uint64_t planned_enqueued = 0;
    uint64_t planned_dequeued = 0;
    uint64_t planned_duplicate_skips = 0;
};

struct predictive_prefetch_layer_state {
    int experts[16][8] = {};
    int n_active[16] = {};
    int cursor = 0;
    int filled = 0;
};

struct predictive_prefetch_state {
    bool inited = false;
    bool enabled = false;
    bool layers_all = false;
    bool layer_enabled[128] = {};
    bool role_up = true;
    bool role_gate = true;
    bool role_down = false;
    int window = 4;
    int topn = 1;
    int max_active = 8;
    int min_history = 4;
    predictive_prefetch_layer_state layers[128];
    std::mutex mu;
    uint64_t calls = 0;
    uint64_t layer_disabled = 0;
    uint64_t prompt_skips = 0;
    uint64_t no_history = 0;
    uint64_t predicted = 0;
    uint64_t submitted = 0;
    uint64_t skipped_vram = 0;
    uint64_t skipped_invalid = 0;
};

struct batch_ttft_trace_entry {
    uint64_t seq = 0;
    double t_ms = 0.0;
    double copy_ms = 0.0;
    int expert_idx = -1;
    int cache_hit = 0;
    int pack_hit = 0;
    int ram_hit = 0;
    size_t expert_bytes = 0;
    char op[24] = {};
    char tensor[128] = {};
};

static std::vector<batch_route_profile_entry> g_route_profile;
static std::vector<batch_route_trace_entry> g_route_trace;
static std::vector<batch_route_detail_entry> g_route_detail;
static trace_prefetch_state g_trace_prefetch;
static host_prefetch_state g_host_prefetch;
static predictive_prefetch_state g_predictive_prefetch;
static std::mutex g_route_profile_mu;
static std::mutex g_route_detail_mu;
static const char * g_route_profile_out = nullptr;
static const char * g_route_trace_out = nullptr;
static const char * g_route_detail_out = nullptr;
static bool g_route_profile_inited = false;
static std::atomic<bool> g_route_detail_inited{false};
static uint64_t g_route_trace_seq = 0;
static uint64_t g_route_detail_seq = 0;
static std::vector<batch_ttft_trace_entry> g_ttft_trace;
static std::mutex g_ttft_trace_mu;
static const char * g_ttft_trace_out = nullptr;
static std::atomic<bool> g_ttft_trace_inited{false};
static size_t g_ttft_trace_cap = 100000;
static uint64_t g_ttft_trace_seq = 0;
static std::chrono::steady_clock::time_point g_ttft_trace_start;







static void batch_route_detail_parse_name(const char *tensor_name, int &layer, char *kind, size_t kind_sz) {
    layer = -1;
    if (kind && kind_sz > 0) kind[0] = '\0';
    if (!tensor_name) return;

    const char *blk = std::strstr(tensor_name, "blk.");
    if (blk) {
        layer = std::atoi(blk + 4);
    }
    const char *parsed = "";
    if (std::strstr(tensor_name, ".ffn_up_exps.")) {
        parsed = "up";
    } else if (std::strstr(tensor_name, ".ffn_gate_exps.")) {
        parsed = "gate";
    } else if (std::strstr(tensor_name, ".ffn_down_exps.")) {
        parsed = "down";
    }
    if (kind && kind_sz > 0) {
        std::snprintf(kind, kind_sz, "%s", parsed);
    }
}













static void trim_csv_field_tail(char *s) {
    if (!s) return;
    size_t n = std::strlen(s);
    while (n > 0 && (s[n - 1] == '\r' || s[n - 1] == '\n')) {
        s[--n] = '\0';
    }
}

static bool load_route_trace_csv(const char *path, std::vector<batch_route_trace_entry> &trace, const char *label) {
    if (!path || !path[0]) return false;
    FILE *f = std::fopen(path, "r");
    if (!f) {
        std::fprintf(stderr, "[moe_stream_batch] %s: open failed: %s\n", label, path);
        return false;
    }
    char line[512];
    if (std::fgets(line, sizeof(line), f)) {
        while (std::fgets(line, sizeof(line), f)) {
            batch_route_trace_entry e;
            unsigned long long seq = 0;
            unsigned long long tensor_base = 0;
            if (std::sscanf(line, "%llu,%zu,0x%llx,%d,%127[^\n]",
                        &seq, &e.expert_bytes, &tensor_base, &e.expert_idx, e.tensor) == 5 &&
                    e.expert_idx >= 0 && e.expert_bytes > 0 && e.tensor[0]) {
                trim_csv_field_tail(e.tensor);
                e.seq = (uint64_t)seq;
                trace.push_back(e);
            }
        }
    }
    std::fclose(f);
    return !trace.empty();
}








static void batch_ttft_trace_report_atexit() {
    std::vector<batch_ttft_trace_entry> rows;
    {
        std::lock_guard<std::mutex> lk(g_ttft_trace_mu);
        rows = g_ttft_trace;
    }
    if (!g_ttft_trace_out || !g_ttft_trace_out[0] || rows.empty()) return;

    FILE *f = std::fopen(g_ttft_trace_out, "w");
    if (!f) {
        std::fprintf(stderr, "[moe_stream_batch] TTFT trace: open failed: %s\n", g_ttft_trace_out);
        return;
    }
    std::fprintf(f, "seq,t_ms,op,tensor,expert_idx,expert_bytes,cache_hit,pack_hit,ram_hit,copy_ms\n");
    for (const batch_ttft_trace_entry &e : rows) {
        std::fprintf(f, "%lu,%.3f,%s,%s,%d,%zu,%d,%d,%d,%.3f\n",
                     e.seq, e.t_ms, e.op, e.tensor, e.expert_idx, e.expert_bytes,
                     e.cache_hit, e.pack_hit, e.ram_hit, e.copy_ms);
    }
    std::fclose(f);
    std::fprintf(stderr, "[moe_stream_batch] TTFT trace written: %s (%zu events)\n",
                 g_ttft_trace_out, rows.size());
}

static bool batch_ttft_trace_enabled() {
    if (g_ttft_trace_inited.load(std::memory_order_acquire)) return g_ttft_trace_out && g_ttft_trace_out[0];
    std::lock_guard<std::mutex> lk(g_ttft_trace_mu);
    if (g_ttft_trace_inited.load(std::memory_order_acquire)) return g_ttft_trace_out && g_ttft_trace_out[0];
    g_ttft_trace_out = std::getenv("GGML_MOE_TTFT_TRACE_OUT");
    const char *cap_env = std::getenv("GGML_MOE_TTFT_TRACE_MAX_EVENTS");
    if (cap_env && cap_env[0]) {
        const long cap = std::atol(cap_env);
        if (cap > 0) g_ttft_trace_cap = (size_t)cap;
    }
    if (g_ttft_trace_out && g_ttft_trace_out[0]) {
        g_ttft_trace_start = std::chrono::steady_clock::now();
        std::atexit(batch_ttft_trace_report_atexit);
    }
    g_ttft_trace_inited.store(true, std::memory_order_release);
    return g_ttft_trace_out && g_ttft_trace_out[0];
}

static void batch_ttft_trace_record(
        const char *op,
        const char *tensor_name,
        int expert_idx,
        size_t expert_bytes,
        bool cache_hit,
        bool pack_hit,
        bool ram_hit,
        double copy_ms) {
    if (!batch_ttft_trace_enabled()) return;
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(g_ttft_trace_mu);
    const bool is_marker = op && std::strncmp(op, "mark_", 5) == 0;
    if (!is_marker && g_ttft_trace.size() >= g_ttft_trace_cap) return;

    batch_ttft_trace_entry e;
    e.seq = ++g_ttft_trace_seq;
    e.t_ms = std::chrono::duration<double, std::milli>(now - g_ttft_trace_start).count();
    e.copy_ms = copy_ms;
    e.expert_idx = expert_idx;
    e.cache_hit = cache_hit ? 1 : 0;
    e.pack_hit = pack_hit ? 1 : 0;
    e.ram_hit = ram_hit ? 1 : 0;
    e.expert_bytes = expert_bytes;
    std::snprintf(e.op, sizeof(e.op), "%s", op ? op : "");
    std::snprintf(e.tensor, sizeof(e.tensor), "%s", tensor_name ? tensor_name : "");
    g_ttft_trace.push_back(e);
}

extern "C" void ggml_cuda_moe_ttft_trace_mark(const char *label) {
    char op[24] = {};
    std::snprintf(op, sizeof(op), "mark_%s", label ? label : "");
    batch_ttft_trace_record(op, "", -1, 0, false, false, false, 0.0);
}

extern "C" void ggml_cuda_moe_ttft_trace_event(
        const char *op,
        const char *tensor_name,
        int marker_value,
        size_t expert_bytes,
        double elapsed_ms) {
    batch_ttft_trace_record(op, tensor_name, marker_value, expert_bytes, false, false, false, elapsed_ms);
}

struct batch_ttft_call_scope {
    const char *op = nullptr;
    const char *tensor = nullptr;
    int n_active = 0;
    size_t expert_bytes = 0;
    bool enabled = false;
    std::chrono::steady_clock::time_point start;

    batch_ttft_call_scope(const char *op_, const char *tensor_, int n_active_, size_t expert_bytes_)
        : op(op_), tensor(tensor_), n_active(n_active_), expert_bytes(expert_bytes_),
          enabled(batch_ttft_trace_enabled()),
          start(enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {
    }

    ~batch_ttft_call_scope() {
        if (!enabled) return;
        const auto end = std::chrono::steady_clock::now();
        batch_ttft_trace_record(
            op,
            tensor,
            n_active,
            expert_bytes,
            false,
            false,
            false,
            std::chrono::duration<double, std::milli>(end - start).count());
    }
};



static void batch_cache_report_atexit() {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t preloads = 0;
    uint64_t pinned = 0;
    uint64_t down_prefetch_loads = 0;
    uint64_t down_prefetch_hits = 0;
    uint64_t down_prefetch_evicted = 0;
    uint64_t async_prefetch_waits = 0;
    for (batch_vram_cache &c : g_bcaches) {
        hits += c.hits;
        misses += c.misses;
        preloads += c.preloads;
        pinned += c.pinned;
        down_prefetch_loads += c.down_prefetch_loads;
        down_prefetch_hits += c.down_prefetch_hits;
        down_prefetch_evicted += c.down_prefetch_evicted;
        async_prefetch_waits += c.async_prefetch_waits;
    }
    const uint64_t total = hits + misses;
    if (total == 0) return;
    if (pinned > 0) {
        std::fprintf(stderr, "[moe_stream_batch] VRAM cache: hits=%lu misses=%lu preloads=%lu pinned=%lu hit_rate=%.1f%%\n",
                     hits, misses, preloads, pinned, 100.0 * hits / total);
    } else {
        std::fprintf(stderr, "[moe_stream_batch] VRAM cache: hits=%lu misses=%lu preloads=%lu hit_rate=%.1f%%\n",
                     hits, misses, preloads, 100.0 * hits / total);
    }
    for (int ic = 0; ic < 2; ++ic) {
        const batch_vram_cache &c = g_bcaches[ic];
        const uint64_t c_total = c.hits + c.misses;
        if (c_total == 0) continue;
        const char *label = ic == 1 ? "upgate" : "down";
        std::fprintf(stderr,
            "[moe_stream_batch] VRAM cache %s: slots=%d slot=%.2f MiB "
            "hits=%lu misses=%lu preloads=%lu pinned=%lu hit_rate=%.1f%%\n",
            label, c.n_slots, c.slot_sz / (1024.0 * 1024.0),
            c.hits, c.misses, c.preloads, c.pinned, 100.0 * c.hits / c_total);
        if (c.descriptor_acquires || c.descriptor_releases ||
                c.descriptor_stale_releases || c.descriptor_lease_blocks ||
                c.owner_invalidations || c.owner_invalidation_failures) {
            std::fprintf(stderr,
                "[moe_stream_batch] VRAM descriptor %s: acquires=%lu releases=%lu "
                "stale_releases=%lu lease_blocks=%lu owner_invalidations=%lu "
                "owner_invalidation_failures=%lu\n",
                label, c.descriptor_acquires, c.descriptor_releases,
                c.descriptor_stale_releases, c.descriptor_lease_blocks,
                c.owner_invalidations, c.owner_invalidation_failures);
        }
    }
    if (down_prefetch_loads > 0) {
        std::fprintf(stderr, "[moe_stream_batch] down prefetch: loads=%lu hits=%lu evicted_unused=%lu useful_rate=%.1f%%\n",
                     down_prefetch_loads, down_prefetch_hits, down_prefetch_evicted,
                     100.0 * down_prefetch_hits / down_prefetch_loads);
    }
    if (async_prefetch_waits > 0) {
        std::fprintf(stderr, "[moe_stream_batch] async prefetch waits=%lu\n", async_prefetch_waits);
    }
}

static bool profile_protect_enabled() {
    const char *env = std::getenv("GGML_MOE_VRAM_PROFILE_PROTECT");
    return env && env[0] && env[0] != '0';
}

static void profile_pinned_key_record(uint64_t key) {
    if (key == 0) return;
    std::lock_guard<std::mutex> lk(g_profile_pinned_mu);
    g_profile_pinned_keys.insert(key);
}

static bool profile_pinned_key_contains(uint64_t key) {
    if (key == 0) return false;
    std::lock_guard<std::mutex> lk(g_profile_pinned_mu);
    return g_profile_pinned_keys.find(key) != g_profile_pinned_keys.end();
}

static size_t profile_reserve_slots(const batch_vram_cache *c) {
    if (!c || c->n_slots <= 1) return 0;

    const char *slots_env = std::getenv("GGML_MOE_VRAM_PROFILE_RESERVE_SLOTS");
    if (slots_env && slots_env[0]) {
        long reserve = std::atol(slots_env);
        if (reserve < 0) reserve = 0;
        if (reserve >= c->n_slots) reserve = c->n_slots - 1;
        return (size_t)reserve;
    }

    const char *pct_env = std::getenv("GGML_MOE_VRAM_PROFILE_RESERVE_PCT");
    long pct = (pct_env && pct_env[0]) ? std::atol(pct_env) : 20;
    if (pct < 0) pct = 0;
    if (pct > 95) pct = 95;

    size_t reserve = ((size_t)c->n_slots * (size_t)pct + 99) / 100;
    if (reserve >= (size_t)c->n_slots) reserve = (size_t)c->n_slots - 1;
    return reserve;
}

static size_t profile_preload_slot_budget(const batch_vram_cache *c) {
    if (!c || c->n_slots <= 0) return 0;
    if (!profile_protect_enabled()) return (size_t)c->n_slots;
    const size_t reserve = profile_reserve_slots(c);
    return (size_t)c->n_slots > reserve ? (size_t)c->n_slots - reserve : 1;
}





static bool profile_preload_enabled() {
    const char *env = std::getenv("GGML_MOE_VRAM_PROFILE_PRELOAD");
    return !env || !env[0] || env[0] != '0';
}

static bool profile_protect_profile_only_enabled() {
    const char *env = std::getenv("GGML_MOE_VRAM_PROFILE_PROTECT_PROFILE_ONLY");
    return env && env[0] && env[0] != '0';
}

static bool profile_pin_on_insert_enabled() {
    const char *env = std::getenv("GGML_MOE_VRAM_PROFILE_PIN_ON_INSERT");
    return env && env[0] && env[0] != '0';
}

static uint32_t profile_pin_min_count() {
    const char *env = std::getenv("GGML_MOE_VRAM_PROFILE_PIN_MIN_COUNT");
    if (!env || !env[0]) return 1;
    const unsigned long value = std::strtoul(env, nullptr, 10);
    return value > UINT32_MAX ? UINT32_MAX : (uint32_t)value;
}

static uint64_t batch_key_hash(const char *name, int expert_idx) {
    uint64_t h = 1469598103934665603ULL;
    if (name) {
        for (const unsigned char *p = (const unsigned char *)name; *p; ++p) {
            h ^= (uint64_t)*p;
            h *= 1099511628211ULL;
        }
    }
    h ^= (uint64_t)(uint32_t)expert_idx + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h ? h : 1;
}

static void profile_lookup_name(const char *name, char *out, size_t out_sz) {
    if (!name || out_sz == 0) return;
    std::snprintf(out, out_sz, "%s", name);
    char *p = std::strstr(out, ".ffn_up_exps.");
    if (!p) p = std::strstr(out, ".ffn_gate_exps.");
    if (p) {
        char tail[96];
        std::snprintf(tail, sizeof(tail), "%s", p + std::strlen(".ffn_up_exps."));
        if (std::strstr(p, ".ffn_gate_exps.")) {
            std::snprintf(tail, sizeof(tail), "%s", p + std::strlen(".ffn_gate_exps."));
        }
        *p = '\0';
        std::snprintf(out + std::strlen(out), out_sz - std::strlen(out), ".ffn_down_exps.%s", tail);
    }
}

static bool tensor_already_preloaded(const char *name) {
    if (!name || !name[0]) return true;
    for (int i = 0; i < g_n_preloaded_tensors; ++i) {
        if (std::strcmp(g_preloaded_tensors[i], name) == 0) return true;
    }
    static bool max_inited = false;
    static long max_tensors = 0;
    if (!max_inited) {
        const char *max_env = std::getenv("GGML_MOE_VRAM_PROFILE_PRELOAD_MAX_TENSORS");
        max_tensors = (max_env && max_env[0]) ? std::atol(max_env) : 0;
        max_inited = true;
    }
    if (max_tensors > 0 && g_n_preloaded_tensors >= max_tensors) return true;
    if (g_n_preloaded_tensors < (int)(sizeof(g_preloaded_tensors) / sizeof(g_preloaded_tensors[0]))) {
        std::snprintf(g_preloaded_tensors[g_n_preloaded_tensors++], sizeof(g_preloaded_tensors[0]), "%s", name);
    }
    return false;
}

static bool profile_has_tensor_locked(const std::vector<profile_entry> &entries, const char *name) {
    if (!name || !name[0]) return false;
    for (const profile_entry &e : entries) {
        if (std::strcmp(e.tensor, name) == 0) return true;
    }
    return false;
}

static bool load_profile_file(const char *path, std::vector<profile_entry> &entries, const char *label) {
    if (!path || !path[0]) {
        return false;
    }
    FILE *f = std::fopen(path, "r");
    if (!f) {
        std::fprintf(stderr, "[moe_stream_batch] %s preload: open failed: %s\n", label, path);
        return false;
    }
    char line[512];
    if (!std::fgets(line, sizeof(line), f)) {
        std::fclose(f);
        return false;
    }
    while (std::fgets(line, sizeof(line), f)) {
        profile_entry e;
        unsigned long long rank = 0, count = 0, cumulative = 0, tensor_base = 0;
        size_t expert_bytes = 0;
        if (std::sscanf(line, "%llu,%llu,%zu,%llu,0x%llx,%d,%127[^\n]",
                    &rank, &count, &expert_bytes, &cumulative, &tensor_base, &e.expert_idx, e.tensor) == 7 &&
                e.expert_idx >= 0 && e.tensor[0]) {
            trim_csv_field_tail(e.tensor);
            e.count = (uint64_t)count;
            e.expert_bytes = expert_bytes;
            entries.push_back(e);
        }
    }
    std::fclose(f);
    if (!entries.empty()) {
        std::fprintf(stderr, "[moe_stream_batch] %s preload: loaded %zu entries from %s\n",
                     label, entries.size(), path);
    }
    return !entries.empty();
}

static void rebuild_profile_count_index_locked() {
    g_profile_counts.clear();
    for (const profile_entry &e : g_profile) {
        const uint64_t count = e.count > UINT32_MAX ? UINT32_MAX : e.count;
        g_profile_counts[batch_key_hash(e.tensor, e.expert_idx)] = (uint32_t)count;
    }
}

static uint32_t profile_count_for_key(uintptr_t key) {
    std::lock_guard<std::mutex> lk(g_profile_mu);
    ++g_cache_policy_diag.profile_count_lookups;
    const auto it = g_profile_counts.find((uint64_t)key);
    if (it == g_profile_counts.end()) return 0;
    ++g_cache_policy_diag.profile_count_hits;
    return it->second;
}

static void load_profile_once() {
    if (g_profile_loaded) return;
    std::lock_guard<std::mutex> lk(g_profile_mu);
    if (g_profile_loaded) return;
    const char *path = std::getenv("GGML_MOE_VRAM_PROFILE");
    g_profile_enabled = load_profile_file(path, g_profile, "profile");
    if (g_profile_enabled) {
        rebuild_profile_count_index_locked();
    }
    g_profile_loaded = true;
}



static int batch_cache_id_for_size(size_t expert_sz) {
    const char *fused_env = std::getenv("GGML_MOE_STREAM_FUSED_UP_GATE");
    const char *split_env = std::getenv("GGML_MOE_VRAM_CACHE_SPLIT");
    const char *split_max_env = std::getenv("GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB");
    size_t split_max_mib = 4;
    if (split_max_env && split_max_env[0]) {
        split_max_mib = (size_t)std::strtoull(split_max_env, nullptr, 10);
        if (split_max_mib < 1) split_max_mib = 1;
        if (split_max_mib > 64) split_max_mib = 64;
    }
    if (fused_env && fused_env[0] && fused_env[0] != '0' &&
            split_env && split_env[0] && split_env[0] != '0' &&
            expert_sz <= split_max_mib*1024ULL*1024ULL) {
        return 1;
    }
    if (fused_env && fused_env[0] && fused_env[0] != '0' && expert_sz <= 2ULL*1024ULL*1024ULL) {
        return 1;
    }
    return 0;
}

static size_t batch_cache_budget_mib_for_id(size_t budget_mib, int cid) {
    const char *fused_env = std::getenv("GGML_MOE_STREAM_FUSED_UP_GATE");
    const bool fused = fused_env && fused_env[0] && fused_env[0] != '0';
    const char *split_env = std::getenv("GGML_MOE_VRAM_CACHE_SPLIT");
    const bool split = fused && split_env && split_env[0] && split_env[0] != '0';

    if (split) {
        const char *pct_env = std::getenv("GGML_MOE_VRAM_CACHE_UPGATE_PCT");
        long upgate_pct = (pct_env && pct_env[0]) ? std::atol(pct_env) : 63;
        if (upgate_pct < 1) upgate_pct = 1;
        if (upgate_pct > 99) upgate_pct = 99;
        const size_t upgate_mib = (budget_mib * (size_t)upgate_pct) / 100ULL;
        if (cid == 1) return upgate_mib > 0 ? upgate_mib : 1;
        return budget_mib > upgate_mib ? budget_mib - upgate_mib : 1;
    }

    if (fused && budget_mib > 8192 && cid == 1) {
        return 8192;
    }
    return budget_mib;
}

static size_t env_mib_or_default(const char *name, size_t fallback) {
    const char *env = std::getenv(name);
    if (!env || !env[0]) return fallback;
    return (size_t)std::strtoull(env, nullptr, 10);
}

static size_t q4_lossless_cuda_budget_reserve_mib() {
    const char *reserve_env = std::getenv(
        "GGML_MOE_LOSSLESS_VRAM_RESERVE_MIB");
    if (reserve_env && reserve_env[0]) {
        return static_cast<size_t>(std::strtoull(reserve_env, nullptr, 10));
    }
    {
        // The role cache is allocated before the deferred-pack manifest is
        // opened. Reserve the measured fixed CUDA decoder footprint at that
        // early point when CUDA lossless transport has already been requested.
        const char *mode = std::getenv("GGML_MOE_LOSSLESS_MODE");
        const char *pack = std::getenv("GGML_MOE_LOSSLESS_PACK");
        if (mode && std::strcmp(mode, "cuda") == 0 && pack && pack[0]) {
            return 512;
        }
        return 0;
    }
}

static size_t batch_cache_effective_budget_mib(size_t requested_mib) {
    static bool inited = false;
    static size_t effective_mib = 0;
    if (inited) return effective_mib;

    effective_mib = requested_mib;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t info_err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (info_err != cudaSuccess) {
        std::fprintf(stderr,
            "[moe_stream_batch] VRAM cache budget: requested=%zu MiB actual=%zu MiB cudaMemGetInfo failed: %s\n",
            requested_mib, effective_mib, cudaGetErrorString(info_err));
        cudaGetLastError();
        inited = true;
        return effective_mib;
    }

    const size_t free_mib = free_bytes / (1024ULL * 1024ULL);
    const size_t total_mib = total_bytes / (1024ULL * 1024ULL);
    const size_t graph_reserve_mib = env_mib_or_default("GGML_MOE_VRAM_CACHE_GRAPH_RESERVE_MIB", 0);
    const size_t safety_mib = env_mib_or_default("GGML_MOE_VRAM_CACHE_SAFETY_MIB", graph_reserve_mib > 0 ? 128 : 0);
    const size_t lossless_reserve_mib = q4_lossless_cuda_budget_reserve_mib();
    const char *clamp_env = std::getenv("GGML_MOE_VRAM_CACHE_AUTO_CLAMP");
    const bool clamp = (clamp_env && clamp_env[0] && clamp_env[0] != '0') || graph_reserve_mib > 0;

    if (clamp) {
        const size_t reserved_mib =
            graph_reserve_mib + safety_mib + lossless_reserve_mib;
        const size_t max_cache_mib = free_mib > reserved_mib ? free_mib - reserved_mib : 0;
        if (effective_mib > max_cache_mib) {
            effective_mib = max_cache_mib;
        }
    }

    std::fprintf(stderr,
        "[moe_stream_batch] VRAM cache budget: requested=%zu MiB actual=%zu MiB free=%zu MiB total=%zu MiB graph_reserve=%zu MiB safety=%zu MiB lossless_reserve=%zu MiB clamp=%d\n",
        requested_mib, effective_mib, free_mib, total_mib,
        graph_reserve_mib, safety_mib, lossless_reserve_mib, clamp ? 1 : 0);
    inited = true;
    return effective_mib;
}

static batch_vram_cache * batch_cache_get(size_t expert_sz) {
    const int cid = batch_cache_id_for_size(expert_sz);
    if (g_bcache_inited[cid]) {
        batch_vram_cache *c = &g_bcaches[cid];
        if (c->pool && expert_sz <= c->slot_sz) return c;
        if (!c->pool || expert_sz <= c->slot_sz) return nullptr;

        for (cudaEvent_t &ev : c->slot_lease_ready) {
            if (ev) {
                cudaEventSynchronize(ev);
                cudaEventDestroy(ev);
                ev = nullptr;
            }
        }
        cudaFree(c->pool);
        c->pool = nullptr;
        c->slot_sz = 0;
        c->n_slots = 0;
        c->key_to_slot.clear();
        std::fill_n(c->slot_key, 16384, (uintptr_t)0);
        std::fill_n(c->slot_used, 16384, (uint64_t)0);
        std::fill_n(c->slot_hits, 16384, (uint32_t)0);
        std::fill_n(c->slot_profile_count, 16384, (uint32_t)0);
        for (char (&name)[128] : c->slot_tensor) {
            name[0] = '\0';
        }
        std::fill_n(c->slot_expert, 16384, -1);
        std::fill_n(c->slot_layer, 16384, -1);
        std::fill_n(c->slot_pinned, 16384, false);
        std::fill_n(c->slot_prefetch_down, 16384, false);
        std::fill_n(c->slot_q4_pilot, 16384, false);
        std::fill_n(c->slot_object_admitted, 16384, false);
        std::fill_n(c->slot_filling, 16384, false);
        for (cudaEvent_t &ev : c->slot_ready) {
            if (ev) {
                cudaEventDestroy(ev);
                ev = nullptr;
            }
        }
        std::fill_n(c->slot_pending, 16384, false);
        std::fill_n(c->slot_lease_pending, 16384, false);
        std::fill_n(c->slot_lease_count, 16384, (uint32_t)0);
        std::fill_n(c->slot_payload_bytes, 16384, (size_t)0);
        std::fill_n(c->slot_ready_marker, 16384, q4_colibri_slot_marker{});
        std::fill_n(c->slot_state, 16384, q4_colibri_slot_state::free);
        c->clock = 1;
        c->hits = 0;
        c->misses = 0;
        c->preloads = 0;
        c->pinned = 0;
        c->down_prefetch_loads = 0;
        c->down_prefetch_hits = 0;
        c->down_prefetch_evicted = 0;
        c->async_prefetch_waits = 0;
        c->descriptor_acquires = 0;
        c->descriptor_releases = 0;
        c->descriptor_stale_releases = 0;
        c->descriptor_lease_blocks = 0;
        c->owner_invalidations = 0;
        c->owner_invalidation_failures = 0;
        g_bcache_inited[cid] = false;
    }
    const char *env = std::getenv("GGML_MOE_VRAM_CACHE_GB");
    const size_t budget_gb = env ? (size_t)std::atoi(env) : 16;
    const char *env_mib = std::getenv("GGML_MOE_VRAM_CACHE_MIB");
    size_t budget_mib = env_mib && env_mib[0] ? (size_t)std::strtoull(env_mib, nullptr, 10) : budget_gb * 1024ULL;
    if (budget_mib == 0) {
        g_bcache_inited[cid] = true;
        return nullptr;
    }
    budget_mib = batch_cache_effective_budget_mib(budget_mib);
    size_t this_budget_mib = batch_cache_budget_mib_for_id(budget_mib, cid);
    const size_t budget = this_budget_mib * 1024ULL * 1024ULL;
    batch_vram_cache *c = &g_bcaches[cid];
    c->slot_sz = expert_sz;
    c->n_slots = (int)(budget / expert_sz);
    if (c->n_slots > 16384) c->n_slots = 16384;
    if (c->n_slots < 1) {
        g_bcache_inited[cid] = true;
        return nullptr;
    }
    c->key_to_slot.clear();
    c->key_to_slot.reserve((size_t)c->n_slots * 2);
    batch_vram_accounting_record("batch_cache_alloc_before", cid, c);
    int alloc_slots = c->n_slots;
    size_t alloc = (size_t)alloc_slots * expert_sz;
    cudaError_t alloc_err = cudaMalloc(&c->pool, alloc);
    if (alloc_err != cudaSuccess) {
        std::fprintf(stderr, "[moe_stream_batch] VRAM cache: cudaMalloc %.1f GiB FAILED; retrying smaller pool\n",
                     alloc / (1024.0*1024.0*1024.0));
        cudaGetLastError();
        const char *retry_env = std::getenv("GGML_MOE_VRAM_CACHE_ALLOC_RETRY");
        const bool retry = !retry_env || !retry_env[0] || retry_env[0] != '0';
        while (retry && alloc_slots > 1) {
            int next_slots = (alloc_slots * 7) / 8;
            if (next_slots >= alloc_slots) next_slots = alloc_slots - 1;
            alloc_slots = next_slots;
            alloc = (size_t)alloc_slots * expert_sz;
            alloc_err = cudaMalloc(&c->pool, alloc);
            if (alloc_err == cudaSuccess) break;
            cudaGetLastError();
        }
        if (alloc_err != cudaSuccess) {
            std::fprintf(stderr, "[moe_stream_batch] VRAM cache: cudaMalloc retry failed; disabling %s cache\n",
                         cid == 1 ? "upgate" : "down");
            c->n_slots = 0;
            g_bcache_inited[cid] = true;
            batch_vram_accounting_record("batch_cache_alloc_failed", cid, c);
            return nullptr;
        }
        c->n_slots = alloc_slots;
    }
    batch_vram_accounting_record("batch_cache_alloc_after", cid, c);
    for (char (&name)[128] : c->slot_tensor) {
        name[0] = '\0';
    }
    std::fill_n(c->slot_expert, 16384, -1);
    std::fill_n(c->slot_layer, 16384, -1);
    std::fill_n(c->slot_object_admitted, 16384, false);
    std::fprintf(stderr, "[moe_stream_batch] VRAM cache: %.1f GiB, %d slots (%.2f MiB each)\n",
                 alloc / (1024.0*1024.0*1024.0), c->n_slots,
                 expert_sz / (1024.0*1024.0));
    std::atexit(batch_cache_report_atexit);
    g_bcache_inited[cid] = true;
    return c;
}

static bool batch_cache_wait_slot_ready(batch_vram_cache *c, int slot) {
    if (!c || slot < 0 || slot >= c->n_slots || !c->slot_pending[slot]) return true;
    const q4_colibri_slot_marker expected{
        c->slot_key[slot], c->slot_generation[slot], c->slot_payload_bytes[slot],
    };
    const q4_colibri_slot_marker ready = c->slot_ready_marker[slot];
    if (ready.generation != 0) {
        if (ready.generation != expected.generation) {
            ++g_expert_pack.q4_pipe_stale_generation;
            return false;
        }
        if (ready.key != expected.key) {
            ++g_expert_pack.q4_pipe_key_mismatch;
            return false;
        }
    }
    const auto wait_start = std::chrono::steady_clock::now();
    if (!ggml_moe_common::role_cache_synchronize_ready_event(
                batch_cache_role_view(c), slot)) {
        return false;
    }
    g_expert_pack.q4_cache_pending_wait_us.fetch_add(
        (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - wait_start).count(), std::memory_order_relaxed);
    ++c->async_prefetch_waits;
    ++g_expert_pack.q4_cache_pending_wait_calls;
    ++g_expert_pack.q4_cache_pending_wait_slots;
    return true;
}

static bool batch_cache_stream_wait_slot_lease(batch_vram_cache *c, int slot, cudaStream_t stream) {
    return c && ggml_moe_common::role_cache_stream_wait_lease_event(
        batch_cache_role_view(c), slot, stream);
}

static bool batch_cache_record_slot_lease(batch_vram_cache *c, int slot, cudaStream_t stream) {
    return c && ggml_moe_common::role_cache_record_lease_event(
        batch_cache_role_view(c), slot, stream);
}





static bool batch_cache_clear_slot(batch_vram_cache *c, int slot) {
    if (!c || slot < 0 || slot >= c->n_slots) return false;
    if (c->slot_lease_count[slot] != 0) {
        ++c->descriptor_lease_blocks;
        return false;
    }
    if (!batch_cache_wait_slot_ready(c, slot)) return false;
    if (!ggml_moe_common::role_cache_synchronize_lease_event(
                batch_cache_role_view(c), slot)) {
        return false;
    }
    const uintptr_t old_key = c->slot_key[slot];
    if (!batch_cache_invalidate_role_owner(c, slot)) return false;
    if (old_key != 0) {
        auto it = c->key_to_slot.find(old_key);
        if (it != c->key_to_slot.end() && it->second == slot) {
            c->key_to_slot.erase(it);
        }
    }
    c->slot_key[slot] = 0;
    c->slot_used[slot] = 0;
    c->slot_hits[slot] = 0;
    c->slot_profile_count[slot] = 0;
    c->slot_tensor[slot][0] = '\0';
    c->slot_expert[slot] = -1;
    c->slot_layer[slot] = -1;
    if (c->slot_pinned[slot] && c->pinned > 0) {
        --c->pinned;
    }
    c->slot_pinned[slot] = false;
    c->slot_prefetch_down[slot] = false;
    c->slot_q4_pilot[slot] = false;
    c->slot_object_admitted[slot] = false;
    c->slot_filling[slot] = false;
    c->slot_pending[slot] = false;
    ++c->slot_generation[slot];
    c->slot_payload_bytes[slot] = 0;
    c->slot_ready_marker[slot] = {};
    c->slot_state[slot] = q4_colibri_slot_state::free;
    return true;
}

static int batch_cache_find_slot(batch_vram_cache *c, uintptr_t key) {
    if (!c || !c->pool || c->n_slots == 0) return -1;
    auto it = c->key_to_slot.find(key);
    if (it != c->key_to_slot.end()) {
        const int slot = it->second;
        if (slot >= 0 && slot < c->n_slots && c->slot_key[slot] == key) {
            return c->slot_state[slot] != q4_colibri_slot_state::failed ? slot : -1;
        }
        c->key_to_slot.erase(it);
    }
    for (int slot = 0; slot < c->n_slots; ++slot) {
        if (c->slot_key[slot] == key) {
            c->key_to_slot[key] = slot;
            return c->slot_state[slot] != q4_colibri_slot_state::failed ? slot : -1;
        }
    }
    return -1;
}









enum {
    Q4_CACHE_LOCALITY_BUCKET_UNKNOWN = 0,
    Q4_CACHE_LOCALITY_BUCKET_CURRENT = 1,
    Q4_CACHE_LOCALITY_BUCKET_NEXT1 = 2,
    Q4_CACHE_LOCALITY_BUCKET_NEXT2 = 3,
    Q4_CACHE_LOCALITY_BUCKET_NEXT3_7 = 4,
    Q4_CACHE_LOCALITY_BUCKET_NEXT8_15 = 5,
    Q4_CACHE_LOCALITY_BUCKET_FAR_FUTURE = 6,
    Q4_CACHE_LOCALITY_BUCKET_PREV1 = 7,
    Q4_CACHE_LOCALITY_BUCKET_PREV2 = 8,
    Q4_CACHE_LOCALITY_BUCKET_PREV3_7 = 9,
    Q4_CACHE_LOCALITY_BUCKET_PREV8_15 = 10,
    Q4_CACHE_LOCALITY_BUCKET_FAR_PAST = 11,
    Q4_CACHE_LOCALITY_BUCKETS = 12,
};

static constexpr int Q4_TIER_REUSE_WINDOW_COUNT = 8;
static constexpr int Q4_TIER_REUSE_WINDOWS[Q4_TIER_REUSE_WINDOW_COUNT] = {1, 2, 4, 8, 16, 32, 64, 128};
static constexpr int Q4_TIER_REUSE_PROTECT_MAX_EXPERTS = 1024;

struct q4_tier_overlap_diag_state {
    std::mutex mu;
    uint64_t token_epoch = 0;
    std::atomic<uint64_t> token_epoch_public{0};
    int last_layer = -1;
    std::unordered_map<std::string, uint64_t> vram_evicted_epoch;
    std::unordered_map<std::string, uint64_t> vram_reuse_protect_until_epoch;
    std::unordered_set<std::string> vram_evicted_reuse_counted;
    std::unordered_map<std::string, uint64_t> covered_ram_evicted_epoch;
    std::atomic<uint64_t> vram_reuse_protect_until_fast[128][Q4_TIER_REUSE_PROTECT_MAX_EXPERTS]{};
    std::atomic<uint64_t> object_demands{0};
    std::atomic<uint64_t> vram_evictions{0};
    std::atomic<uint64_t> vram_evicted_reused_within_window{0};
    std::atomic<uint64_t> vram_evicted_reused_by_window[Q4_TIER_REUSE_WINDOW_COUNT]{};
    std::atomic<uint64_t> vram_reuse_protect_marks{0};
    std::atomic<uint64_t> vram_reuse_protect_layer_skips{0};
    std::atomic<uint64_t> vram_reuse_protect_penalized{0};
    std::atomic<uint64_t> vram_reuse_protect_selected{0};
    std::atomic<uint64_t> vram_reuse_protect_expired{0};
    std::atomic<uint64_t> vram_static_hotness_protected{0};
    std::atomic<uint64_t> vram_static_hotness_selected{0};
    std::atomic<uint64_t> ram_hits_after_vram_eviction{0};
    std::atomic<uint64_t> ram_hits_after_vram_eviction_within_window{0};
    std::atomic<uint64_t> ram_hit_vram_full_overlap{0};
    std::atomic<uint64_t> ram_hit_vram_partial_overlap_roles{0};
    std::atomic<uint64_t> covered_ram_evictions{0};
    std::atomic<uint64_t> covered_ram_eviction_misses_within_window{0};
    std::atomic<uint64_t> tier_delete_ram_on_h2d{0};
    std::atomic<uint64_t> tier_demote_attempts{0};
    std::atomic<uint64_t> tier_demote_success{0};
    std::atomic<uint64_t> tier_demote_skip_incomplete_vram{0};
    std::atomic<uint64_t> tier_demote_skip_already_ram{0};
    std::atomic<uint64_t> tier_demote_skip_no_slot{0};
    std::atomic<uint64_t> tier_demote_skip_low_value{0};
    std::atomic<uint64_t> tier_demote_bytes{0};
    std::atomic<uint64_t> tier_demote_d2h_us{0};
    std::atomic<uint64_t> tier_demote_async_submitted{0};
    std::atomic<uint64_t> tier_demote_async_completed{0};
    std::atomic<uint64_t> tier_demote_async_enqueue_failures{0};
    std::atomic<uint64_t> tier_demote_async_event_failures{0};
    std::atomic<uint64_t> tier_demote_async_pending_bytes{0};
    std::atomic<uint64_t> tier_demote_async_enqueue_us{0};
    std::atomic<uint64_t> tier_vram_score_victim_selections{0};
    std::atomic<uint64_t> tier_soft_dedup_victim_candidates{0};
    std::atomic<uint64_t> tier_soft_dedup_evictions{0};
    std::atomic<uint64_t> cache_locality_samples{0};
    std::atomic<uint64_t> cache_locality_skipped_lock{0};
    std::atomic<uint64_t> cache_locality_vram_roles[Q4_CACHE_LOCALITY_BUCKETS]{};
    std::atomic<uint64_t> cache_locality_vram_full_objects[Q4_CACHE_LOCALITY_BUCKETS]{};
    std::atomic<uint64_t> cache_locality_ram_objects[Q4_CACHE_LOCALITY_BUCKETS]{};
    std::atomic<uint64_t> cache_locality_ram_vram_full_overlap[Q4_CACHE_LOCALITY_BUCKETS]{};
    std::atomic<uint64_t> cache_locality_ram_vram_any_overlap[Q4_CACHE_LOCALITY_BUCKETS]{};
    std::atomic<uint64_t> cache_locality_ram_vram_partial_roles[Q4_CACHE_LOCALITY_BUCKETS]{};
    std::atomic<uint64_t> cache_locality_layer_samples[128]{};
    std::atomic<uint64_t> cache_locality_layer_vram_roles[128]{};
    std::atomic<uint64_t> cache_locality_layer_vram_near_roles[128]{};
    std::atomic<uint64_t> cache_locality_layer_vram_full_objects[128]{};
    std::atomic<uint64_t> cache_locality_layer_vram_near_full_objects[128]{};
    std::atomic<uint64_t> cache_locality_layer_ram_objects[128]{};
    std::atomic<uint64_t> cache_locality_layer_ram_near_objects[128]{};
    std::atomic<uint64_t> cache_locality_layer_ram_vram_full_overlap[128]{};
    std::atomic<uint64_t> cache_locality_layer_ram_vram_any_overlap[128]{};
    std::atomic<uint64_t> cache_locality_layer_ram_vram_partial_roles[128]{};
    std::atomic<uint64_t> tier_layer_object_demands[128]{};
    std::atomic<uint64_t> tier_layer_vram_evictions[128]{};
    std::atomic<uint64_t> tier_layer_vram_evicted_reused_within_window[128]{};
    std::atomic<uint64_t> tier_layer_vram_evicted_reused_by_window[Q4_TIER_REUSE_WINDOW_COUNT][128]{};
    std::atomic<uint64_t> tier_layer_ram_hits_after_vram_eviction[128]{};
    std::atomic<uint64_t> tier_layer_ram_hits_after_vram_eviction_within_window[128]{};
    std::atomic<uint64_t> tier_layer_ram_hit_vram_full_overlap[128]{};
    std::atomic<uint64_t> tier_layer_ram_hit_vram_partial_overlap_roles[128]{};
    std::atomic<uint64_t> tier_layer_covered_ram_evictions[128]{};
    std::atomic<uint64_t> tier_layer_covered_ram_eviction_misses_within_window[128]{};
    std::atomic<uint64_t> score_profile_ram_candidates{0};
    std::atomic<uint64_t> score_profile_ram_admitted{0};
    std::atomic<uint64_t> score_profile_ram_rejected{0};
    std::atomic<uint64_t> score_profile_ram_empty_slot{0};
    std::atomic<uint64_t> score_profile_ram_victim_events{0};
    std::atomic<uint64_t> score_profile_ram_candidate_gt_victim{0};
    std::atomic<uint64_t> score_profile_ram_candidate_eq_victim{0};
    std::atomic<uint64_t> score_profile_ram_candidate_lt_victim{0};
    std::atomic<uint64_t> score_profile_ram_candidate_score_sum{0};
    std::atomic<uint64_t> score_profile_ram_victim_score_sum{0};
    std::atomic<uint64_t> score_profile_vram_evictions{0};
    std::atomic<uint64_t> score_profile_vram_candidate_gt_victim{0};
    std::atomic<uint64_t> score_profile_vram_candidate_eq_victim{0};
    std::atomic<uint64_t> score_profile_vram_candidate_lt_victim{0};
    std::atomic<uint64_t> score_profile_vram_candidate_score_sum{0};
    std::atomic<uint64_t> score_profile_vram_victim_score_sum{0};
    std::atomic<uint64_t> score_profile_layer_ram_candidates[128]{};
    std::atomic<uint64_t> score_profile_layer_ram_admitted[128]{};
    std::atomic<uint64_t> score_profile_layer_ram_rejected[128]{};
    std::atomic<uint64_t> score_profile_layer_ram_empty_slot[128]{};
    std::atomic<uint64_t> score_profile_layer_ram_victim_events[128]{};
    std::atomic<uint64_t> score_profile_layer_ram_candidate_gt_victim[128]{};
    std::atomic<uint64_t> score_profile_layer_ram_candidate_eq_victim[128]{};
    std::atomic<uint64_t> score_profile_layer_ram_candidate_lt_victim[128]{};
    std::atomic<uint64_t> score_profile_layer_ram_candidate_score_sum[128]{};
    std::atomic<uint64_t> score_profile_layer_ram_victim_score_sum[128]{};
    std::atomic<uint64_t> score_profile_layer_vram_evictions[128]{};
    std::atomic<uint64_t> score_profile_layer_vram_candidate_gt_victim[128]{};
    std::atomic<uint64_t> score_profile_layer_vram_candidate_eq_victim[128]{};
    std::atomic<uint64_t> score_profile_layer_vram_candidate_lt_victim[128]{};
    std::atomic<uint64_t> score_profile_layer_vram_candidate_score_sum[128]{};
    std::atomic<uint64_t> score_profile_layer_vram_victim_score_sum[128]{};
};

static q4_tier_overlap_diag_state g_q4_tier_overlap_diag;



enum q4_host_object_origin : uint8_t {
    Q4_HOST_OBJECT_ORIGIN_UNKNOWN = 0,
    Q4_HOST_OBJECT_ORIGIN_PREFETCH = 1,
    Q4_HOST_OBJECT_ORIGIN_PILOT = 2,
    Q4_HOST_OBJECT_ORIGIN_READ_ADMIT = 3,
    Q4_HOST_OBJECT_ORIGIN_DEMOTE = 4,
    Q4_HOST_OBJECT_ORIGIN_COUNT = 5,
};



static bool q4_tier_layer_profile_enabled() {
    static const bool enabled = []() {
        const char *env = std::getenv("GGML_MOE_Q4_TIER_LAYER_PROFILE");
        return env && env[0] && env[0] != '0';
    }();
    return enabled;
}



static void q4_tier_layer_add(std::atomic<uint64_t> (&counters)[128], int layer, uint64_t value = 1) {
    if (!q4_tier_layer_profile_enabled()) return;
    if (value == 0 || layer < 0 || layer >= 128) return;
    counters[layer].fetch_add(value, std::memory_order_relaxed);
}



static bool q4_tier_vram_reuse_protect_enabled() {
    static bool enabled = []() {
        const char *env = std::getenv("GGML_MOE_Q4_VRAM_REUSE_PROTECT");
        return env && env[0] && env[0] != '0';
    }();
    return enabled;
}



static bool q4_tier_vram_reuse_protect_layer_allowed(int layer) {
    static bool inited = false;
    static bool allow_all = true;
    static bool allowed[128] = {};
    if (!inited) {
        const char *env = std::getenv("GGML_MOE_Q4_VRAM_REUSE_PROTECT_LAYERS");
        if (env && env[0] && std::strcmp(env, "all") != 0 && std::strcmp(env, "*") != 0) {
            allow_all = false;
            char buf[512];
            std::snprintf(buf, sizeof(buf), "%s", env);
            char *save = nullptr;
            for (char *tok = ::strtok_r(buf, ",;:", &save); tok; tok = ::strtok_r(nullptr, ",;:", &save)) {
                while (*tok == ' ' || *tok == '\t') ++tok;
                if (!*tok) continue;
                char *end = nullptr;
                long lo = std::strtol(tok, &end, 10);
                long hi = lo;
                if (end && *end == '-') {
                    char *end2 = nullptr;
                    hi = std::strtol(end + 1, &end2, 10);
                    end = end2;
                }
                if (end == tok || lo < 0 || hi < 0) continue;
                if (hi < lo) std::swap(lo, hi);
                lo = std::max<long>(0, std::min<long>(lo, 127));
                hi = std::max<long>(0, std::min<long>(hi, 127));
                for (long l = lo; l <= hi; ++l) {
                    allowed[l] = true;
                }
            }
        }
        inited = true;
    }
    if (layer < 0 || layer >= 128) return false;
    return allow_all || allowed[layer];
}



static bool q4_tier_demote_vram_evict_enabled() {
    const char *env = std::getenv("GGML_MOE_Q4_TIER_DEMOTE_VRAM_EVICT");
    return env && env[0] && env[0] != '0';
}

static bool q4_tier_demote_admit_layers_only_enabled() {
    const char *env = std::getenv("GGML_MOE_Q4_TIER_DEMOTE_ADMIT_LAYERS_ONLY");
    return env && env[0] && env[0] != '0';
}

static bool q4_vram_unified_score_enabled() {
    const char *env = std::getenv("GGML_MOE_Q4_VRAM_UNIFIED_SCORE");
    return env && env[0] && env[0] != '0';
}

static bool q4_vram_static_hotness_protect_enabled() {
    const char *env = std::getenv("GGML_MOE_Q4_VRAM_STATIC_HOTNESS_PROTECT");
    return env && env[0] && env[0] != '0';
}

static uint64_t q4_vram_static_hotness_protect_ticks() {
    const char *env = std::getenv("GGML_MOE_Q4_VRAM_STATIC_HOTNESS_PROTECT_TICKS");
    if (!env || !env[0]) return 8192;
    char *end = nullptr;
    unsigned long long value = std::strtoull(env, &end, 10);
    if (end == env) return 8192;
    return std::min<unsigned long long>(value, 1000000000ULL);
}

static bool q4_tier_demote_async_d2h_enabled() {
    const char *env = std::getenv("GGML_MOE_Q4_TIER_DEMOTE_ASYNC_D2H");
    return env && env[0] && env[0] != '0';
}

static uint32_t q4_tier_demote_min_hits() {
    const char *env = std::getenv("GGML_MOE_Q4_TIER_DEMOTE_MIN_HITS");
    if (!env || !env[0]) return 2;
    char *end = nullptr;
    unsigned long value = std::strtoul(env, &end, 10);
    if (end == env) return 2;
    if (value > 1024) value = 1024;
    return (uint32_t)value;
}

static std::string q4_tier_diag_object_key(int layer, int expert) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%d:%d", layer, expert);
    return std::string(buf);
}

static std::string q4_tier_diag_object_key(const q4_colibri_object_marker &marker) {
    return q4_tier_diag_object_key(marker.layer_idx, marker.expert_idx);
}

static bool q4_tier_reuse_protect_fast_in_range(int layer, int expert) {
    return layer >= 0 && layer < 128 && expert >= 0 && expert < Q4_TIER_REUSE_PROTECT_MAX_EXPERTS;
}



static void q4_tier_reuse_protect_clear_locked(const std::string &key, int layer, int expert) {
    if (q4_tier_reuse_protect_fast_in_range(layer, expert)) {
        g_q4_tier_overlap_diag.vram_reuse_protect_until_fast[layer][expert].store(
                0, std::memory_order_release);
    }
    g_q4_tier_overlap_diag.vram_reuse_protect_until_epoch.erase(key);
}



static int q4_tier_diag_cache_ready_slot(const batch_vram_cache &cache, uintptr_t key) {
    if (!cache.pool || cache.n_slots <= 0 || key == 0) return -1;
    auto indexed = cache.key_to_slot.find(key);
    if (indexed != cache.key_to_slot.end()) {
        const int slot = indexed->second;
        if (slot >= 0 && slot < cache.n_slots && cache.slot_key[slot] == key) {
            if (cache.slot_state[slot] == q4_colibri_slot_state::failed ||
                    cache.slot_state[slot] == q4_colibri_slot_state::free ||
                    cache.slot_state[slot] == q4_colibri_slot_state::reserved ||
                    cache.slot_state[slot] == q4_colibri_slot_state::reading ||
                    cache.slot_filling[slot] || cache.slot_pending[slot] ||
                    cache.slot_payload_bytes[slot] == 0) {
                return -1;
            }
            return slot;
        }
    }
    for (int slot = 0; slot < cache.n_slots; ++slot) {
        if (cache.slot_key[slot] != key) continue;
        if (cache.slot_state[slot] == q4_colibri_slot_state::failed ||
                cache.slot_state[slot] == q4_colibri_slot_state::free ||
                cache.slot_state[slot] == q4_colibri_slot_state::reserved ||
                cache.slot_state[slot] == q4_colibri_slot_state::reading ||
                cache.slot_filling[slot] || cache.slot_pending[slot] ||
                cache.slot_payload_bytes[slot] == 0) {
            return -1;
        }
        return slot;
    }
    return -1;
}





static bool q4_tier_role_family_names(
        const char *role_name,
        std::string &gate_name,
        std::string &up_name,
        std::string &down_name) {
    if (!role_name || !role_name[0]) return false;
    gate_name = role_name;
    const char *needles[3] = {
        ".ffn_gate_exps.",
        ".ffn_up_exps.",
        ".ffn_down_exps.",
    };
    size_t pos = std::string::npos;
    const char *matched = nullptr;
    for (const char *needle : needles) {
        pos = gate_name.find(needle);
        if (pos != std::string::npos) {
            matched = needle;
            break;
        }
    }
    if (!matched) return false;
    gate_name.replace(pos, std::strlen(matched), ".ffn_gate_exps.");
    up_name = gate_name;
    down_name = gate_name;
    up_name.replace(pos, std::strlen(".ffn_gate_exps."), ".ffn_up_exps.");
    down_name.replace(pos, std::strlen(".ffn_gate_exps."), ".ffn_down_exps.");
    return true;
}







static bool q4_tier_diag_vram_reuse_protected(int layer, int expert) {
    if (!q4_tier_vram_reuse_protect_enabled() || layer < 0 || expert < 0) return false;
    if (!q4_tier_vram_reuse_protect_layer_allowed(layer)) return false;
    if (q4_tier_reuse_protect_fast_in_range(layer, expert)) {
        const uint64_t until =
            g_q4_tier_overlap_diag.vram_reuse_protect_until_fast[layer][expert].load(
                    std::memory_order_acquire);
        if (until == 0) return false;
        const uint64_t epoch =
            g_q4_tier_overlap_diag.token_epoch_public.load(std::memory_order_acquire);
        if (epoch <= until) return true;
        uint64_t expected = until;
        if (g_q4_tier_overlap_diag.vram_reuse_protect_until_fast[layer][expert].compare_exchange_strong(
                    expected, 0, std::memory_order_acq_rel)) {
            ++g_q4_tier_overlap_diag.vram_reuse_protect_expired;
        }
        return false;
    }
    std::lock_guard<std::mutex> lock(g_q4_tier_overlap_diag.mu);
    const std::string key = q4_tier_diag_object_key(layer, expert);
    auto it = g_q4_tier_overlap_diag.vram_reuse_protect_until_epoch.find(key);
    if (it == g_q4_tier_overlap_diag.vram_reuse_protect_until_epoch.end()) return false;
    if (g_q4_tier_overlap_diag.token_epoch <= it->second) return true;
    g_q4_tier_overlap_diag.vram_reuse_protect_until_epoch.erase(it);
    ++g_q4_tier_overlap_diag.vram_reuse_protect_expired;
    return false;
}

static void q4_tier_diag_record_vram_eviction(int layer, int expert) {
    if (layer < 0 || expert < 0) return;
    std::lock_guard<std::mutex> lock(g_q4_tier_overlap_diag.mu);
    const uint64_t epoch = g_q4_tier_overlap_diag.token_epoch;
    const std::string key = q4_tier_diag_object_key(layer, expert);
    g_q4_tier_overlap_diag.vram_evicted_epoch[key] = epoch;
    g_q4_tier_overlap_diag.vram_evicted_reuse_counted.erase(key);
    q4_tier_reuse_protect_clear_locked(key, layer, expert);
    ++g_q4_tier_overlap_diag.vram_evictions;
    q4_tier_layer_add(g_q4_tier_overlap_diag.tier_layer_vram_evictions, layer);
}











static bool batch_cache_promote_profile_slot(batch_vram_cache *c, int slot, uint64_t profile_count) {
    if (!c || slot < 0 || slot >= c->n_slots || c->slot_key[slot] == 0) return false;
    if (c->slot_pinned[slot]) {
        if (profile_count > 0) {
            c->slot_profile_count[slot] = profile_count > UINT32_MAX ? UINT32_MAX : (uint32_t)profile_count;
        }
        return true;
    }
    if (c->pinned >= profile_preload_slot_budget(c)) return false;
    c->slot_pinned[slot] = true;
    ++c->pinned;
    c->slot_profile_count[slot] = profile_count > UINT32_MAX ? UINT32_MAX : (uint32_t)profile_count;
    return true;
}

static bool batch_slot_is_avoided(int slot, const int *avoid_slots, int n_avoid_slots) {
    if (!avoid_slots || n_avoid_slots <= 0) return false;
    for (int i = 0; i < n_avoid_slots; ++i) {
        if (avoid_slots[i] == slot) return true;
    }
    return false;
}

static bool cache_policy_lfu_lru_enabled() {
    const char *env = std::getenv("GGML_MOE_VRAM_CACHE_POLICY");
    return env && std::strcmp(env, "lfu_lru") == 0;
}

static bool q4_vram_object_admission_enabled() {
    static const bool enabled = []() {
        const char *env = std::getenv("GGML_MOE_VRAM_CACHE_POLICY");
        return env && std::strcmp(env, "object_admit_age_t2") == 0;
    }();
    return enabled;
}

static bool cache_policy_profile_lfu_lru_enabled() {
    const char *env = std::getenv("GGML_MOE_VRAM_CACHE_POLICY");
    return env && std::strcmp(env, "profile_lfu_lru") == 0;
}

static bool cache_policy_hybrid_profile_lfu_lru_enabled() {
    const char *env = std::getenv("GGML_MOE_VRAM_CACHE_POLICY");
    return env && std::strcmp(env, "hybrid_profile_lfu_lru") == 0;
}

static int cache_gate_preload_evict_updown_max_hits() {
    static int max_hits = []() {
        const char *env = std::getenv("GGML_MOE_GATE_PRELOAD_EVICT_UPDOWN_MAX_HITS");
        if (!env || !env[0]) return -1;
        return std::atoi(env);
    }();
    return max_hits;
}

static bool cache_tensor_is_gate(const char *name) {
    return name && std::strstr(name, ".ffn_gate_exps.") != nullptr;
}

static bool cache_tensor_is_updown(const char *name) {
    return name && (std::strstr(name, ".ffn_up_exps.") != nullptr ||
            std::strstr(name, ".ffn_down_exps.") != nullptr);
}

static uint64_t cache_policy_hybrid_after() {
    const char *env = std::getenv("GGML_MOE_VRAM_CACHE_PROFILE_AFTER");
    if (!env || !env[0]) return 12624;
    return std::strtoull(env, nullptr, 10);
}

static void cache_policy_diag_report_atexit() {
    const uint64_t lookups = g_cache_policy_diag.profile_count_lookups.load();
    const uint64_t evictions = g_cache_policy_diag.profile_policy_evictions.load();
    if (lookups == 0 && evictions == 0) return;
    const uint64_t lookup_hits = g_cache_policy_diag.profile_count_hits.load();
    const uint64_t inserted_nonzero = g_cache_policy_diag.inserted_profile_count_nonzero.load();
    const uint64_t inserted_sum = g_cache_policy_diag.inserted_profile_count_sum.load();
    const uint64_t victim_nonzero = g_cache_policy_diag.profile_policy_victim_count_nonzero.load();
    const uint64_t victim_sum = g_cache_policy_diag.profile_policy_victim_count_sum.load();
    std::fprintf(stderr,
        "[moe_stream_batch] cache policy diag: profile_count_lookups=%lu hits=%lu "
        "inserted_nonzero=%lu inserted_avg=%.2f evictions=%lu victim_nonzero=%lu victim_avg=%.2f\n",
        lookups, lookup_hits, inserted_nonzero,
        inserted_nonzero > 0 ? (double)inserted_sum / (double)inserted_nonzero : 0.0,
        evictions, victim_nonzero,
        victim_nonzero > 0 ? (double)victim_sum / (double)victim_nonzero : 0.0);
}

static bool cache_evict_profile_enabled() {
    static const bool enabled = []() {
        const char *env = std::getenv("GGML_MOE_CACHE_EVICT_PROFILE_OUT");
        return env && env[0];
    }();
    return enabled;
}

static void cache_evict_profile_record(
        const batch_vram_cache *c,
        int slot,
        uintptr_t victim_key,
        uint32_t victim_hits,
        uint32_t victim_profile_count,
        bool victim_pinned,
        bool victim_prefetch_down,
        uint64_t victim_used,
        uintptr_t incoming_key,
        const char *incoming_tensor,
        int incoming_expert,
        bool incoming_preload,
        bool incoming_prefetch_down,
        uint64_t incoming_profile_count) {
    const char *path = std::getenv("GGML_MOE_CACHE_EVICT_PROFILE_OUT");
    if (!path || !path[0] || !c || slot < 0 || slot >= c->n_slots) return;

    static std::mutex mu;
    static bool header_written = false;
    static uint64_t seq = 0;
    std::lock_guard<std::mutex> lk(mu);

    FILE *f = std::fopen(path, "a");
    if (!f) return;
    if (!header_written) {
        std::fprintf(f,
                "seq,slot_sz,slot,victim_tensor,victim_expert,victim_key,victim_hits,"
                "victim_profile_count,victim_pinned,victim_prefetch_down,victim_used,"
                "incoming_tensor,incoming_expert,incoming_key,incoming_preload,"
                "incoming_prefetch_down,incoming_profile_count,cache_clock\n");
        header_written = true;
    }
    std::fprintf(f,
            "%lu,%zu,%d,%s,%d,%lu,%u,%u,%d,%d,%lu,%s,%d,%lu,%d,%d,%lu,%lu\n",
            (unsigned long)++seq,
            c->slot_sz,
            slot,
            c->slot_tensor[slot],
            c->slot_expert[slot],
            (unsigned long)victim_key,
            victim_hits,
            victim_profile_count,
            victim_pinned ? 1 : 0,
            victim_prefetch_down ? 1 : 0,
            (unsigned long)victim_used,
            incoming_tensor ? incoming_tensor : "",
            incoming_expert,
            (unsigned long)incoming_key,
            incoming_preload ? 1 : 0,
            incoming_prefetch_down ? 1 : 0,
            (unsigned long)incoming_profile_count,
            (unsigned long)c->clock);
    std::fclose(f);
}

static int expert_pack_entry_cmp(const expert_pack_entry &e, const char *tensor_name, int expert_idx, size_t nbytes) {
    const int name_cmp = std::strcmp(e.tensor, tensor_name);
    if (name_cmp != 0) return name_cmp;
    if (e.expert_idx != expert_idx) return e.expert_idx < expert_idx ? -1 : 1;
    if (e.nbytes != nbytes) return e.nbytes < nbytes ? -1 : 1;
    return 0;
}

static int expert_pack_v2_entry_cmp(const expert_pack_v2_entry &e, const char *tensor_name, int expert_idx) {
    const int name_cmp = std::strcmp(e.tensor, tensor_name ? tensor_name : "");
    if (name_cmp != 0) return name_cmp;
    if (e.expert_idx < expert_idx) return -1;
    if (e.expert_idx > expert_idx) return 1;
    return 0;
}

static size_t expert_pack_direct_alignment() {
    return 4096;
}

static uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

static size_t expert_pack_env_size(const char *name, size_t default_value, size_t min_value, size_t max_value) {
    const char *env = std::getenv(name);
    if (!env || !env[0]) return default_value;
    const unsigned long long parsed = std::strtoull(env, nullptr, 10);
    if (parsed == 0) return default_value;
    size_t value = (size_t)parsed;
    if (value < min_value) value = min_value;
    if (value > max_value) value = max_value;
    return value;
}

static bool expert_pack_env_bool(const char *name, bool default_value) {
    const char *env = std::getenv(name);
    if (!env || !env[0]) return default_value;
    return env[0] != '0';
}

static bool common_object_cqe_enabled() {
    static const bool enabled = expert_pack_env_bool("GGML_MOE_COMMON_OBJECT_CQE", true);
    return enabled;
}



static const ggml_moe_common::object_pipeline_mode_config & expert_pack_pipeline_mode_config() {
    static const ggml_moe_common::object_pipeline_mode_config config =
        ggml_moe_common::object_pipeline_resolve_mode(
            ggml_moe_common::object_pipeline_mode::rolling_refill,
            expert_pack_env_bool("GGML_MOE_COMMON_ROLLING_REFILL", false));
    return config;
}

static const ggml_moe_common::object_pipeline_policy_config & expert_pack_pipeline_policy_config();





#if defined(__linux__)
static bool expert_pack_apply_current_thread_affinity(const char *env_name, const char *role) {
    const char *spec = std::getenv(env_name);
    if (!spec || !spec[0]) return true;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    bool any = false;
    bool valid = true;
    const char *cursor = spec;
    while (*cursor) {
        while (*cursor == ' ' || *cursor == ',') ++cursor;
        if (!*cursor) break;
        char *end = nullptr;
        const long first = std::strtol(cursor, &end, 10);
        if (end == cursor || first < 0 || first >= CPU_SETSIZE) {
            valid = false;
            break;
        }
        long last = first;
        cursor = end;
        if (*cursor == '-') {
            ++cursor;
            last = std::strtol(cursor, &end, 10);
            if (end == cursor || last < first || last >= CPU_SETSIZE) {
                valid = false;
                break;
            }
            cursor = end;
        }
        for (long cpu = first; cpu <= last; ++cpu) CPU_SET((int)cpu, &cpuset);
        any = true;
        while (*cursor == ' ') ++cursor;
        if (*cursor && *cursor != ',') {
            valid = false;
            break;
        }
    }

    if (!valid || !any) {
        std::fprintf(stderr,
                "[moe_stream_batch] thread affinity: role=%s env=%s cpus=%s status=invalid\n",
                role, env_name, spec);
        return false;
    }
    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    std::fprintf(stderr,
            "[moe_stream_batch] thread affinity: role=%s env=%s cpus=%s status=%s rc=%d\n",
            role, env_name, spec, rc == 0 ? "ok" : "failed", rc);
    return rc == 0;
}
#else
static bool expert_pack_apply_current_thread_affinity(const char *, const char *) {
    return true;
}
#endif



static bool stage_granularity_profile_enabled() {
    return expert_pack_env_bool("GGML_MOE_STAGE_GRANULARITY_PROFILE", false);
}

static size_t expert_pack_io_bytes() {
    return expert_pack_env_size("GGML_MOE_IO_BYTES", 2ULL * 1024ULL * 1024ULL,
            expert_pack_direct_alignment(), 64ULL * 1024ULL * 1024ULL);
}

static size_t expert_pack_legacy_io_depth() {
    return expert_pack_env_size("GGML_MOE_IO_DEPTH", 32, 1, 256);
}

static size_t expert_pack_legacy_io_refill_batch() {
    return expert_pack_env_size("GGML_MOE_IO_REFILL_BATCH", 1, 1, 256);
}

static const ggml_moe_common::object_pipeline_policy_config & expert_pack_pipeline_policy_config() {
    static const ggml_moe_common::object_pipeline_policy_config config = [] {
        const ggml_moe_common::object_pipeline_mode_config & mode =
            expert_pack_pipeline_mode_config();
        const size_t legacy_depth = expert_pack_legacy_io_depth();
        ggml_moe_common::object_pipeline_policy fallback;
        fallback.io_depth = legacy_depth;
        fallback.initial_batch = legacy_depth;
        fallback.min_flush_batch = 1;
        fallback.flush_timeout_us = 1;
        fallback.max_inflight_h2d = legacy_depth;
        switch (mode.mode) {
            case ggml_moe_common::object_pipeline_mode::sync_qd1:
                fallback.io_depth = 1;
                fallback.initial_batch = 1;
                fallback.refill_batch = 1;
                fallback.max_inflight_h2d = 1;
                break;
            case ggml_moe_common::object_pipeline_mode::persistent_demand:
                fallback.refill_batch = legacy_depth;
                break;
            case ggml_moe_common::object_pipeline_mode::rolling_refill:
                fallback.refill_batch = std::min(
                    expert_pack_legacy_io_refill_batch(), legacy_depth);
                break;
        }
        ggml_moe_common::object_pipeline_policy_config resolved =
            ggml_moe_common::object_pipeline_resolve_policy(fallback);
        if (!mode.valid) {
            resolved.policy = fallback;
            resolved.valid = false;
        }
        return resolved;
    }();
    return config;
}

static size_t expert_pack_io_depth() {
    return expert_pack_pipeline_policy_config().policy.io_depth;
}

static size_t expert_pack_io_refill_batch() {
    const ggml_moe_common::object_pipeline_policy & policy =
        expert_pack_pipeline_policy_config().policy;
    return std::min(policy.refill_batch, policy.io_depth);
}

// Preserve the original all-at-once default; explicit CHUNK overrides both
// states of the opt-in adaptive policy.
static size_t common_io_submit_chunk(size_t jobs) {
    static const size_t chunk = []() -> size_t {
        const char *env = std::getenv("GGML_MOE_COMMON_IO_SUBMIT_CHUNK");
        if (env && std::strcmp(env, "0") == 0) return 0;
        return expert_pack_env_size("GGML_MOE_COMMON_IO_SUBMIT_CHUNK", SIZE_MAX, 1, 256);
    }();
    static const bool adaptive = expert_pack_env_bool("GGML_MOE_COMMON_IO_SUBMIT_ADAPTIVE", false);
    return ggml_moe_common::object_io_select_submit_chunk(jobs, chunk, adaptive);
}

static bool expert_pack_io_eager_refill_enabled() {
    return expert_pack_env_bool("GGML_MOE_IO_EAGER_REFILL", false);
}

static bool expert_pack_adjacent_coalesce_enabled() {
    return expert_pack_env_bool("GGML_MOE_IO_ADJACENT_COALESCE", false);
}

static size_t expert_pack_adjacent_max_span() {
    return expert_pack_env_size("GGML_MOE_IO_ADJACENT_MAX_SPAN_MIB", 16, 1, 512) * 1024ULL * 1024ULL;
}

static size_t expert_pack_adjacent_max_gap() {
    const char *env = std::getenv("GGML_MOE_IO_ADJACENT_MAX_GAP");
    if (!env || !env[0]) return 0;
    const unsigned long long parsed = std::strtoull(env, nullptr, 10);
    return (size_t)std::min<unsigned long long>(parsed, 4096ULL);
}

static size_t expert_pack_adjacent_min_group() {
    return expert_pack_env_size("GGML_MOE_IO_ADJACENT_MIN_GROUP", 2, 2, 64);
}

static bool expert_pack_adjacent_op_allowed(const char *trace_op) {
    const char *filter = std::getenv("GGML_MOE_IO_ADJACENT_OP_FILTER");
    if (!filter || !filter[0]) return true;
    return trace_op && std::strstr(trace_op, filter) != nullptr;
}

static size_t expert_pack_adjacent_op_bucket(const char *trace_op) {
    if (trace_op && std::strstr(trace_op, "runtime_load") != nullptr) return 0;
    if (trace_op && std::strstr(trace_op, "current_down_overlap") != nullptr) return 1;
    return 2;
}

static const char *expert_pack_adjacent_op_name(size_t bucket) {
    switch (bucket) {
        case 0: return "runtime_load";
        case 1: return "current_down_overlap";
        default: return "other";
    }
}

static void expert_pack_atomic_max(std::atomic<uint64_t> &target, uint64_t value) {
    uint64_t current = target.load(std::memory_order_relaxed);
    while (current < value &&
            !target.compare_exchange_weak(current, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

static size_t expert_pack_iouring_batch_bucket(size_t jobs) {
    if (jobs <= 1) return 0;
    if (jobs <= 4) return 1;
    if (jobs <= 8) return 2;
    if (jobs <= 16) return 3;
    if (jobs <= 32) return 4;
    return 5;
}

static void expert_pack_record_iouring_batch(size_t jobs) {
    ++g_expert_pack.io_stats.batches;
    switch (expert_pack_iouring_batch_bucket(jobs)) {
        case 0: ++g_expert_pack.iouring_batch_hist_1; break;
        case 1: ++g_expert_pack.iouring_batch_hist_2_4; break;
        case 2: ++g_expert_pack.iouring_batch_hist_5_8; break;
        case 3: ++g_expert_pack.iouring_batch_hist_9_16; break;
        case 4: ++g_expert_pack.iouring_batch_hist_17_32; break;
        default: ++g_expert_pack.iouring_batch_hist_gt32; break;
    }
}























static bool up_gate_profile_csv_enabled() {
    const char *env = std::getenv("GGML_MOE_UP_GATE_PROFILE_OUT");
    return env && env[0];
}



































static void current_down_overlap_tensor_profile_write() {
    const char *path = std::getenv("GGML_MOE_CURRENT_DOWN_OVERLAP_PROFILE_OUT");
    if (!path || !path[0]) return;

    std::lock_guard<std::mutex> lk(g_current_down_overlap_tensor_profile_mu);
    if (g_current_down_overlap_tensor_profile.empty()) return;

    FILE *f = std::fopen(path, "w");
    if (!f) return;
    std::fprintf(f, "tensor,calls,planned_jobs,cache_hits,missing_tensor,missing_pack,completed_jobs\n");
    for (const auto &it : g_current_down_overlap_tensor_profile) {
        const current_down_overlap_tensor_profile &p = it.second;
        std::fprintf(f, "%s,%lu,%lu,%lu,%lu,%lu,%lu\n",
                it.first.c_str(),
                (unsigned long)p.calls,
                (unsigned long)p.planned_jobs,
                (unsigned long)p.cache_hits,
                (unsigned long)p.missing_tensor,
                (unsigned long)p.missing_pack,
                (unsigned long)p.completed_jobs);
    }
    std::fclose(f);
}

static void current_down_overlap_report_atexit() {
    const uint64_t calls = g_current_down_overlap.calls.load();
    const uint64_t submitted = g_current_down_overlap.submitted_batches.load();
    if (calls == 0 && submitted == 0) {
        current_down_overlap_tensor_profile_write();
        return;
    }
    std::fprintf(stderr,
        "[moe_stream_batch] current down overlap: calls=%lu planned_jobs=%lu completed_jobs=%lu "
        "cache_hits=%lu missing_tensor=%lu missing_pack=%lu submitted_batches=%lu failed_batches=%lu "
        "mark_failed=%lu max_jobs=%lu worker_us=%lu batch_hist=1:%lu,2-4:%lu,5-8:%lu,9-16:%lu,17-32:%lu,gt32:%lu\n",
        calls,
        g_current_down_overlap.planned_jobs.load(),
        g_current_down_overlap.completed_jobs.load(),
        g_current_down_overlap.cache_hits.load(),
        g_current_down_overlap.missing_tensor.load(),
        g_current_down_overlap.missing_pack.load(),
        submitted,
        g_current_down_overlap.failed_batches.load(),
        g_current_down_overlap.mark_failed.load(),
        g_current_down_overlap.max_jobs.load(),
        g_current_down_overlap.worker_us.load(),
        g_current_down_overlap.batch_hist_1.load(),
        g_current_down_overlap.batch_hist_2_4.load(),
        g_current_down_overlap.batch_hist_5_8.load(),
        g_current_down_overlap.batch_hist_9_16.load(),
        g_current_down_overlap.batch_hist_17_32.load(),
        g_current_down_overlap.batch_hist_gt32.load());
    current_down_overlap_tensor_profile_write();
}

static void expert_pack_report_atexit() {
    if (!expert_pack_catalog_ready()) return;
    if (const char *direct_path = direct_read_site_profile_path()) {
        std::lock_guard<std::mutex> lk(g_direct_read_site_mu);
        std::fprintf(stderr,
                "[moe_stream_batch] direct read site profile requested: path=%s entries=%zu\n",
                direct_path, g_direct_read_site_stats.size());
    }
    const ggml_moe_common::object_io_runtime_snapshot io_stats =
        ggml_moe_common::object_io_runtime_snapshot_load(g_expert_pack.io_stats);
    std::fprintf(stderr,
                 "[moe_stream_batch] expert pack: hits=%lu misses=%lu read_failures=%lu direct_reads=%lu direct_fallbacks=%lu "
                 "iouring_reads=%lu iouring_bytes=%lu iouring_fallbacks=%lu iouring_submit_us=%lu iouring_wait_us=%lu iouring_h2d_enqueues=%lu entries=%zu\n",
                 g_expert_pack.hits.load(), g_expert_pack.misses.load(),
                 g_expert_pack.read_failures.load(), g_expert_pack.direct_reads.load(),
                 g_expert_pack.direct_fallbacks.load(),
                 io_stats.reads, io_stats.bytes, io_stats.fallbacks, io_stats.submit_us,
                 io_stats.wait_us, g_expert_pack.iouring_h2d_enqueues.load(),
                 expert_pack_published_entry_count());
    if (g_expert_pack.q4_colibri) {
        std::fprintf(stderr,
                     "[moe_stream_batch] q4 colibri pack: partial_fallback_lookups=%lu partial_fallback_tensors=%lu debug_partial_fallback=%d\n",
                     g_expert_pack.q4_colibri_partial_fallback_lookups.load(),
                     g_expert_pack.q4_colibri_partial_fallback_tensors.load(),
                     g_expert_pack.q4_colibri_debug_partial_fallback ? 1 : 0);
        if (g_expert_pack.q4_object_h2d_batches.load() > 0) {
            std::fprintf(stderr,
                    "[moe_stream_batch] q4 object h2d batch: batches=%lu ram_batches=%lu ssd_batches=%lu "
                    "objects=%lu roles=%lu bytes=%lu enqueue_us=%lu\n",
                    g_expert_pack.q4_object_h2d_batches.load(),
                    g_expert_pack.q4_object_h2d_ram_batches.load(),
                    g_expert_pack.q4_object_h2d_ssd_batches.load(),
                    g_expert_pack.q4_object_h2d_objects.load(),
                    g_expert_pack.q4_object_h2d_roles.load(),
                    g_expert_pack.q4_object_h2d_bytes.load(),
                    g_expert_pack.q4_object_h2d_enqueue_us.load());
        }
    }
    const uint64_t inflight_samples = io_stats.inflight_samples;
    const double inflight_avg = inflight_samples > 0 ?
        (double)io_stats.inflight_sum / (double)inflight_samples : 0.0;
    if (io_stats.batches > 0 || io_stats.submit_calls > 0 || io_stats.wait_calls > 0) {
        std::fprintf(stderr,
                     "[moe_stream_batch] expert pack iouring detail: batches=%lu submit_calls=%lu wait_calls=%lu cqes=%lu "
                     "inflight_avg=%.2f inflight_max=%lu batch_hist=1:%lu,2-4:%lu,5-8:%lu,9-16:%lu,17-32:%lu,gt32:%lu\n",
                     io_stats.batches, io_stats.submit_calls, io_stats.wait_calls, io_stats.cqes,
                     inflight_avg, io_stats.inflight_max,
                     g_expert_pack.iouring_batch_hist_1.load(), g_expert_pack.iouring_batch_hist_2_4.load(),
                     g_expert_pack.iouring_batch_hist_5_8.load(), g_expert_pack.iouring_batch_hist_9_16.load(),
                     g_expert_pack.iouring_batch_hist_17_32.load(), g_expert_pack.iouring_batch_hist_gt32.load());
    }
    if (g_expert_pack.adjacent_coalesce_groups.load() > 0 ||
            g_expert_pack.adjacent_coalesce_reject_source.load() > 0 ||
            g_expert_pack.adjacent_coalesce_reject_tensor.load() > 0 ||
            g_expert_pack.adjacent_coalesce_reject_gap.load() > 0 ||
            g_expert_pack.adjacent_coalesce_reject_span.load() > 0) {
        std::fprintf(stderr,
                     "[moe_stream_batch] adjacent coalesce: groups=%lu slices=%lu extents_saved=%lu "
                     "physical_bytes=%lu payload_bytes=%lu max_group_jobs=%lu max_span_bytes=%lu "
                     "reject_source=%lu reject_tensor=%lu reject_gap=%lu reject_span=%lu\n",
                     g_expert_pack.adjacent_coalesce_groups.load(),
                     g_expert_pack.adjacent_coalesce_slices.load(),
                     g_expert_pack.adjacent_coalesce_extents_saved.load(),
                     g_expert_pack.adjacent_coalesce_physical_bytes.load(),
                     g_expert_pack.adjacent_coalesce_payload_bytes.load(),
                     g_expert_pack.adjacent_coalesce_max_group_jobs.load(),
                     g_expert_pack.adjacent_coalesce_max_span_bytes.load(),
                     g_expert_pack.adjacent_coalesce_reject_source.load(),
                     g_expert_pack.adjacent_coalesce_reject_tensor.load(),
                     g_expert_pack.adjacent_coalesce_reject_gap.load(),
                     g_expert_pack.adjacent_coalesce_reject_span.load());
        for (size_t i = 0; i < 3; ++i) {
            const expert_pack_adjacent_op_stats &s = g_expert_pack.adjacent_coalesce_op[i];
            if (s.groups.load() == 0 && s.slices.load() == 0 && s.extents_saved.load() == 0) {
                continue;
            }
            std::fprintf(stderr,
                         "[moe_stream_batch] adjacent coalesce op=%s: groups=%lu slices=%lu extents_saved=%lu "
                         "physical_bytes=%lu payload_bytes=%lu max_group_jobs=%lu max_span_bytes=%lu\n",
                         expert_pack_adjacent_op_name(i),
                         s.groups.load(), s.slices.load(), s.extents_saved.load(),
                         s.physical_bytes.load(), s.payload_bytes.load(),
                         s.max_group_jobs.load(), s.max_span_bytes.load());
        }
    }
    if (common_dynamic_tier_requested()) {
        common_dynamic_tier_report();
    } else if (g_expert_pack.ram_tier_base) {
        std::fprintf(stderr,
                     "[moe_stream_batch] RAM tier: hits=%lu total=%lu hit_rate=%.1f%% resident=%.2f MiB "
                     "pinned=%.2f MiB h2d_enqueues=%lu h2d_bytes=%lu "
                     "batch_calls=%lu batch_jobs=%lu batch_hits=%lu batch_bytes=%lu "
                     "fadvise_bytes=%lu fadvise_failures=%lu direct_bytes=%lu direct_failures=%lu direct_fallbacks=%lu\n",
                     g_expert_pack.ram_tier_hits.load(), g_expert_pack.ram_tier_total.load(),
                     g_expert_pack.ram_tier_total.load() > 0 ?
                         100.0 * g_expert_pack.ram_tier_hits.load() / g_expert_pack.ram_tier_total.load() : 0.0,
                     g_expert_pack.ram_tier_bytes / (1024.0 * 1024.0),
                     g_expert_pack.ram_tier_pinned_bytes / (1024.0 * 1024.0),
                     g_expert_pack.ram_tier_h2d_enqueues.load(),
                     g_expert_pack.ram_tier_h2d_bytes.load(),
                     g_expert_pack.ram_tier_batch_calls.load(),
                     g_expert_pack.ram_tier_batch_jobs.load(),
                     g_expert_pack.ram_tier_batch_hits.load(),
                     g_expert_pack.ram_tier_batch_bytes.load(),
                     g_expert_pack.ram_tier_fadvise_bytes.load(),
                     g_expert_pack.ram_tier_fadvise_failures.load(),
                     g_expert_pack.ram_tier_direct_bytes.load(),
                     g_expert_pack.ram_tier_direct_failures.load(),
                     g_expert_pack.ram_tier_direct_fallbacks.load());
    }
    if (g_expert_pack.page_cache_sources.load() > 0 ||
            g_expert_pack.page_cache_reads.load() > 0 ||
            g_expert_pack.page_cache_preload_bytes.load() > 0) {
        std::fprintf(stderr,
                     "[moe_stream_batch] page-cache pack: sources=%lu reads=%lu bytes=%lu preload_bytes=%lu preload_failures=%lu\n",
                     g_expert_pack.page_cache_sources.load(),
                     g_expert_pack.page_cache_reads.load(),
                     g_expert_pack.page_cache_bytes.load(),
                     g_expert_pack.page_cache_preload_bytes.load(),
                     g_expert_pack.page_cache_preload_failures.load());
    }
}

extern "C" void ggml_cuda_moe_stream_batch_report_counters(void) {
    batch_cache_report_atexit();
    expert_pack_report_atexit();
    std::fflush(stderr);
}

struct moe_phase_counter_snapshot {
    uint64_t pack_hits = 0;
    uint64_t pack_misses = 0;
    uint64_t pack_read_failures = 0;
    uint64_t pack_direct_reads = 0;
    uint64_t pack_direct_fallbacks = 0;
    uint64_t pack_iouring_reads = 0;
    uint64_t pack_iouring_bytes = 0;
    uint64_t pack_iouring_fallbacks = 0;
    uint64_t pack_iouring_submit_us = 0;
    uint64_t pack_iouring_wait_us = 0;
    uint64_t pack_iouring_h2d_enqueues = 0;
    uint64_t pack_iouring_batches = 0;
    uint64_t pack_iouring_submit_calls = 0;
    uint64_t pack_iouring_wait_calls = 0;
    uint64_t pack_iouring_cqes = 0;
    uint64_t current_down_calls = 0;
    uint64_t current_down_planned_jobs = 0;
    uint64_t current_down_completed_jobs = 0;
    uint64_t current_down_cache_hits = 0;
    uint64_t current_down_missing_tensor = 0;
    uint64_t current_down_missing_pack = 0;
    uint64_t current_down_submitted_batches = 0;
    uint64_t current_down_failed_batches = 0;
    uint64_t current_down_worker_us = 0;
    uint64_t cache_hits[2] = {};
    uint64_t cache_misses[2] = {};
    uint64_t cache_preloads[2] = {};
    uint64_t cache_pinned[2] = {};
    uint64_t cache_down_prefetch_loads[2] = {};
    uint64_t cache_down_prefetch_hits[2] = {};
    uint64_t cache_async_prefetch_waits[2] = {};
    uint64_t stage_copies[5] = {};
    uint64_t stage_waits[5] = {};
    uint64_t stage_fallbacks[5] = {};
    uint64_t stage_iouring_init_attempts[5] = {};
    uint64_t stage_iouring_init_reuses[5] = {};
    uint64_t stage_iouring_init_failures[5] = {};
    uint64_t stage_iouring_batches[5] = {};
    uint64_t stage_iouring_jobs[5] = {};
    uint64_t stage_iouring_wait_calls[5] = {};
    uint64_t stage_iouring_cqes[5] = {};
};

static moe_phase_counter_snapshot g_moe_phase_last;
static bool g_moe_phase_last_valid = false;
static std::atomic<uint64_t> g_moe_phase_seq{0};

static moe_phase_counter_snapshot moe_phase_snapshot_now() {
    moe_phase_counter_snapshot s;
    const ggml_moe_common::object_io_runtime_snapshot io_stats =
        ggml_moe_common::object_io_runtime_snapshot_load(g_expert_pack.io_stats);
    s.pack_hits = g_expert_pack.hits.load();
    s.pack_misses = g_expert_pack.misses.load();
    s.pack_read_failures = g_expert_pack.read_failures.load();
    s.pack_direct_reads = g_expert_pack.direct_reads.load();
    s.pack_direct_fallbacks = g_expert_pack.direct_fallbacks.load();
    s.pack_iouring_reads = io_stats.reads;
    s.pack_iouring_bytes = io_stats.bytes;
    s.pack_iouring_fallbacks = io_stats.fallbacks;
    s.pack_iouring_submit_us = io_stats.submit_us;
    s.pack_iouring_wait_us = io_stats.wait_us;
    s.pack_iouring_h2d_enqueues = g_expert_pack.iouring_h2d_enqueues.load();
    s.pack_iouring_batches = io_stats.batches;
    s.pack_iouring_submit_calls = io_stats.submit_calls;
    s.pack_iouring_wait_calls = io_stats.wait_calls;
    s.pack_iouring_cqes = io_stats.cqes;
    s.current_down_calls = g_current_down_overlap.calls.load();
    s.current_down_planned_jobs = g_current_down_overlap.planned_jobs.load();
    s.current_down_completed_jobs = g_current_down_overlap.completed_jobs.load();
    s.current_down_cache_hits = g_current_down_overlap.cache_hits.load();
    s.current_down_missing_tensor = g_current_down_overlap.missing_tensor.load();
    s.current_down_missing_pack = g_current_down_overlap.missing_pack.load();
    s.current_down_submitted_batches = g_current_down_overlap.submitted_batches.load();
    s.current_down_failed_batches = g_current_down_overlap.failed_batches.load();
    s.current_down_worker_us = g_current_down_overlap.worker_us.load();
    for (int i = 0; i < 2; ++i) {
        const batch_vram_cache &c = g_bcaches[i];
        s.cache_hits[i] = c.hits;
        s.cache_misses[i] = c.misses;
        s.cache_preloads[i] = c.preloads;
        s.cache_pinned[i] = c.pinned;
        s.cache_down_prefetch_loads[i] = c.down_prefetch_loads;
        s.cache_down_prefetch_hits[i] = c.down_prefetch_hits;
        s.cache_async_prefetch_waits[i] = c.async_prefetch_waits;
    }
    const pinned_stage_ring *rings[5] = {
        &g_batch.stage_ring,
        &g_batch.stage_ring_gate,
        &g_batch.stage_ring_up_aux,
        &g_batch.stage_ring_gate_aux,
        &g_batch.stage_ring_down_aux,
    };
    for (int i = 0; i < 5; ++i) {
        s.stage_copies[i] = rings[i]->copies;
        s.stage_waits[i] = rings[i]->waits;
        s.stage_fallbacks[i] = rings[i]->fallbacks;
        s.stage_iouring_init_attempts[i] = rings[i]->iouring_init_attempts;
        s.stage_iouring_init_reuses[i] = rings[i]->iouring_init_reuses;
        s.stage_iouring_init_failures[i] = rings[i]->iouring_init_failures;
        s.stage_iouring_batches[i] = rings[i]->iouring_batches;
        s.stage_iouring_jobs[i] = rings[i]->iouring_jobs;
        s.stage_iouring_wait_calls[i] = rings[i]->iouring_wait_calls;
        s.stage_iouring_cqes[i] = rings[i]->iouring_cqes;
    }
    return s;
}

static uint64_t moe_phase_delta_u64(uint64_t now, uint64_t prev) {
    return now >= prev ? now - prev : 0;
}

extern "C" void ggml_cuda_moe_stream_batch_phase_report(const char *label) {
    if (!label || !label[0]) {
        label = "unknown";
    }
    const moe_phase_counter_snapshot cur = moe_phase_snapshot_now();
    const moe_phase_counter_snapshot prev = g_moe_phase_last_valid ? g_moe_phase_last : moe_phase_counter_snapshot{};
    const uint64_t seq = g_moe_phase_seq.fetch_add(1, std::memory_order_relaxed);

    std::fprintf(stderr,
        "[moe_stream_batch_phase] label=%s seq=%lu "
        "pack_hits=%lu delta_pack_hits=%lu pack_misses=%lu delta_pack_misses=%lu "
        "read_failures=%lu delta_read_failures=%lu direct_reads=%lu delta_direct_reads=%lu "
        "iouring_reads=%lu delta_iouring_reads=%lu iouring_bytes=%lu delta_iouring_bytes=%lu "
        "iouring_submit_us=%lu delta_iouring_submit_us=%lu iouring_wait_us=%lu delta_iouring_wait_us=%lu "
        "iouring_h2d=%lu delta_iouring_h2d=%lu iouring_batches=%lu delta_iouring_batches=%lu "
        "iouring_submit_calls=%lu delta_iouring_submit_calls=%lu iouring_wait_calls=%lu delta_iouring_wait_calls=%lu "
        "iouring_cqes=%lu delta_iouring_cqes=%lu entries=%zu\n",
        label, (unsigned long)seq,
        cur.pack_hits, moe_phase_delta_u64(cur.pack_hits, prev.pack_hits),
        cur.pack_misses, moe_phase_delta_u64(cur.pack_misses, prev.pack_misses),
        cur.pack_read_failures, moe_phase_delta_u64(cur.pack_read_failures, prev.pack_read_failures),
        cur.pack_direct_reads, moe_phase_delta_u64(cur.pack_direct_reads, prev.pack_direct_reads),
        cur.pack_iouring_reads, moe_phase_delta_u64(cur.pack_iouring_reads, prev.pack_iouring_reads),
        cur.pack_iouring_bytes, moe_phase_delta_u64(cur.pack_iouring_bytes, prev.pack_iouring_bytes),
        cur.pack_iouring_submit_us, moe_phase_delta_u64(cur.pack_iouring_submit_us, prev.pack_iouring_submit_us),
        cur.pack_iouring_wait_us, moe_phase_delta_u64(cur.pack_iouring_wait_us, prev.pack_iouring_wait_us),
        cur.pack_iouring_h2d_enqueues, moe_phase_delta_u64(cur.pack_iouring_h2d_enqueues, prev.pack_iouring_h2d_enqueues),
        cur.pack_iouring_batches, moe_phase_delta_u64(cur.pack_iouring_batches, prev.pack_iouring_batches),
        cur.pack_iouring_submit_calls, moe_phase_delta_u64(cur.pack_iouring_submit_calls, prev.pack_iouring_submit_calls),
        cur.pack_iouring_wait_calls, moe_phase_delta_u64(cur.pack_iouring_wait_calls, prev.pack_iouring_wait_calls),
        cur.pack_iouring_cqes, moe_phase_delta_u64(cur.pack_iouring_cqes, prev.pack_iouring_cqes),
        expert_pack_published_entry_count());

    for (int i = 0; i < 2; ++i) {
        const char *cache_label = i == 1 ? "upgate" : "down";
        std::fprintf(stderr,
            "[moe_stream_batch_phase] label=%s seq=%lu cache=%s "
            "hits=%lu delta_hits=%lu misses=%lu delta_misses=%lu preloads=%lu delta_preloads=%lu "
            "pinned=%lu delta_pinned=%lu down_prefetch_loads=%lu delta_down_prefetch_loads=%lu "
            "down_prefetch_hits=%lu delta_down_prefetch_hits=%lu async_prefetch_waits=%lu delta_async_prefetch_waits=%lu\n",
            label, (unsigned long)seq, cache_label,
            cur.cache_hits[i], moe_phase_delta_u64(cur.cache_hits[i], prev.cache_hits[i]),
            cur.cache_misses[i], moe_phase_delta_u64(cur.cache_misses[i], prev.cache_misses[i]),
            cur.cache_preloads[i], moe_phase_delta_u64(cur.cache_preloads[i], prev.cache_preloads[i]),
            cur.cache_pinned[i], moe_phase_delta_u64(cur.cache_pinned[i], prev.cache_pinned[i]),
            cur.cache_down_prefetch_loads[i], moe_phase_delta_u64(cur.cache_down_prefetch_loads[i], prev.cache_down_prefetch_loads[i]),
            cur.cache_down_prefetch_hits[i], moe_phase_delta_u64(cur.cache_down_prefetch_hits[i], prev.cache_down_prefetch_hits[i]),
            cur.cache_async_prefetch_waits[i], moe_phase_delta_u64(cur.cache_async_prefetch_waits[i], prev.cache_async_prefetch_waits[i]));
    }

    const char *stage_names[5] = {"main", "gate", "up_aux", "gate_aux", "down_aux"};
    for (int i = 0; i < 5; ++i) {
        std::fprintf(stderr,
            "[moe_stream_batch_phase] label=%s seq=%lu stage=%s "
            "copies=%lu delta_copies=%lu waits=%lu delta_waits=%lu fallbacks=%lu delta_fallbacks=%lu "
            "iouring_init=%lu delta_iouring_init=%lu iouring_init_reuse=%lu delta_iouring_init_reuse=%lu "
            "iouring_init_fail=%lu delta_iouring_init_fail=%lu "
            "iouring_batches=%lu delta_iouring_batches=%lu iouring_jobs=%lu delta_iouring_jobs=%lu "
            "iouring_wait_calls=%lu delta_iouring_wait_calls=%lu iouring_cqes=%lu delta_iouring_cqes=%lu\n",
            label, (unsigned long)seq, stage_names[i],
            cur.stage_copies[i], moe_phase_delta_u64(cur.stage_copies[i], prev.stage_copies[i]),
            cur.stage_waits[i], moe_phase_delta_u64(cur.stage_waits[i], prev.stage_waits[i]),
            cur.stage_fallbacks[i], moe_phase_delta_u64(cur.stage_fallbacks[i], prev.stage_fallbacks[i]),
            cur.stage_iouring_init_attempts[i], moe_phase_delta_u64(cur.stage_iouring_init_attempts[i], prev.stage_iouring_init_attempts[i]),
            cur.stage_iouring_init_reuses[i], moe_phase_delta_u64(cur.stage_iouring_init_reuses[i], prev.stage_iouring_init_reuses[i]),
            cur.stage_iouring_init_failures[i], moe_phase_delta_u64(cur.stage_iouring_init_failures[i], prev.stage_iouring_init_failures[i]),
            cur.stage_iouring_batches[i], moe_phase_delta_u64(cur.stage_iouring_batches[i], prev.stage_iouring_batches[i]),
            cur.stage_iouring_jobs[i], moe_phase_delta_u64(cur.stage_iouring_jobs[i], prev.stage_iouring_jobs[i]),
            cur.stage_iouring_wait_calls[i], moe_phase_delta_u64(cur.stage_iouring_wait_calls[i], prev.stage_iouring_wait_calls[i]),
            cur.stage_iouring_cqes[i], moe_phase_delta_u64(cur.stage_iouring_cqes[i], prev.stage_iouring_cqes[i]));
    }

    std::fprintf(stderr,
        "[moe_stream_batch_phase] label=%s seq=%lu current_down "
        "calls=%lu delta_calls=%lu planned_jobs=%lu delta_planned_jobs=%lu "
        "completed_jobs=%lu delta_completed_jobs=%lu cache_hits=%lu delta_cache_hits=%lu "
        "missing_tensor=%lu delta_missing_tensor=%lu missing_pack=%lu delta_missing_pack=%lu "
        "submitted_batches=%lu delta_submitted_batches=%lu failed_batches=%lu delta_failed_batches=%lu "
        "worker_us=%lu delta_worker_us=%lu\n",
        label, (unsigned long)seq,
        cur.current_down_calls, moe_phase_delta_u64(cur.current_down_calls, prev.current_down_calls),
        cur.current_down_planned_jobs, moe_phase_delta_u64(cur.current_down_planned_jobs, prev.current_down_planned_jobs),
        cur.current_down_completed_jobs, moe_phase_delta_u64(cur.current_down_completed_jobs, prev.current_down_completed_jobs),
        cur.current_down_cache_hits, moe_phase_delta_u64(cur.current_down_cache_hits, prev.current_down_cache_hits),
        cur.current_down_missing_tensor, moe_phase_delta_u64(cur.current_down_missing_tensor, prev.current_down_missing_tensor),
        cur.current_down_missing_pack, moe_phase_delta_u64(cur.current_down_missing_pack, prev.current_down_missing_pack),
        cur.current_down_submitted_batches, moe_phase_delta_u64(cur.current_down_submitted_batches, prev.current_down_submitted_batches),
        cur.current_down_failed_batches, moe_phase_delta_u64(cur.current_down_failed_batches, prev.current_down_failed_batches),
        cur.current_down_worker_us, moe_phase_delta_u64(cur.current_down_worker_us, prev.current_down_worker_us));

    if (common_dynamic_tier_requested()) common_dynamic_tier_phase_report(label, seq);
    g_moe_phase_last = cur;
    g_moe_phase_last_valid = true;
    std::fflush(stderr);
}

static bool expert_pack_read_exact(FILE *file, void *dst, size_t sz) {
    char *out = (char *)dst;
    size_t done = 0;
    while (done < sz) {
        const size_t chunk = sz - done;
        const size_t got = std::fread(out + done, 1, chunk, file);
        if (got == 0) return false;
        done += got;
    }
    return true;
}

#if !defined(GGML_SSD_STORE)
static std::string expert_pack_key_string(const char *tensor_name, int expert_idx, uint64_t nbytes) {
    std::string key = tensor_name ? tensor_name : "";
    key.push_back('\n');
    key += std::to_string(expert_idx);
    key.push_back('\n');
    key += std::to_string((unsigned long long)nbytes);
    return key;
}
#endif

static expert_pack_source * expert_pack_source_for_entry(const expert_pack_entry *entry) {
#if defined(GGML_SSD_STORE)
    if (!c0::catalog_available()) return nullptr;
#endif
    if (!entry || entry->source_idx < 0 ||
            (size_t)entry->source_idx >= g_expert_pack.sources.size()) {
        return nullptr;
    }
    return &g_expert_pack.sources[(size_t)entry->source_idx];
}

#if defined(GGML_SSD_STORE)
static c0::source_open_hooks expert_pack_source_open_hooks() {
    c0::source_open_hooks hooks;
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    hooks.report_iouring_enabled = [](const char * label, const char * path) {
        std::fprintf(stderr, "[moe_stream_batch] %s: io_uring direct reads enabled: %s io_bytes=%zu depth=%zu\n",
                     label, path, expert_pack_io_bytes(), expert_pack_io_depth());
    };
#endif
    hooks.page_cache_source_opened = [] { ++g_expert_pack.page_cache_sources; };
    return hooks;
}
static bool expert_pack_open_data_source(const char * path, const char * label,
        bool page_cache_source = false, bool direct_required = false) {
    return c0::open_data_source(c0::shared_catalog(), path, label, page_cache_source, direct_required, expert_pack_source_open_hooks());
}
#else
static bool expert_pack_open_data_source(
        const char *path,
        const char *label,
        bool page_cache_source = false,
        bool direct_required = false) {
    if (!path || !path[0]) return false;

    FILE *file = std::fopen(path, "rb");
    if (!file) {
        std::fprintf(stderr, "[moe_stream_batch] %s: open failed: %s\n", label, path);
        return false;
    }

    expert_pack_source source;
    source.file = file;
    source.page_cache = page_cache_source;
    source.direct_required = direct_required;
    std::snprintf(source.path, sizeof(source.path), "%s", path);
#if !defined(_WIN32)
    struct stat source_status = {};
    if (fstat(fileno(file), &source_status) != 0 || source_status.st_size < 0) {
        std::fprintf(stderr, "[moe_stream_batch] %s: stat failed: %s\n", label, path);
        std::fclose(file);
        return false;
    }
    source.file_size = static_cast<uint64_t>(source_status.st_size);
    source.fd_direct = -1;
    if (!page_cache_source && (g_expert_pack.io_backend == 1 || g_expert_pack.io_backend == 2)) {
#if defined(O_DIRECT)
        source.fd_direct = ::open(path, O_RDONLY | O_DIRECT);
#endif
        if (source.fd_direct < 0) {
            if (direct_required) {
                std::fprintf(stderr, "[moe_stream_batch] %s: required O_DIRECT open failed: %s\n", label, path);
                std::fclose(file);
                return false;
            }
            std::fprintf(stderr, "[moe_stream_batch] %s: direct open failed; using buffered reads: %s\n", label, path);
            g_expert_pack.io_backend = 0;
        } else if (g_expert_pack.io_backend == 2) {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
            std::fprintf(stderr, "[moe_stream_batch] %s: io_uring direct reads enabled: %s io_bytes=%zu depth=%zu\n",
                         label, path, expert_pack_io_bytes(), expert_pack_io_depth());
#else
            const bool require_direct = expert_pack_env_bool("GGML_MOE_IO_REQUIRE_DIRECT", false);
            std::fprintf(stderr, "[moe_stream_batch] %s: io_uring requested but liburing headers are unavailable; %s\n",
                         label, require_direct ? "using direct fallback anyway" : "using direct fallback");
            g_expert_pack.io_backend = 1;
#endif
        } else if (g_expert_pack.io_backend == 1) {
            std::fprintf(stderr, "[moe_stream_batch] %s: direct reads enabled: %s\n", label, path);
        }
    } else if (page_cache_source) {
        ++g_expert_pack.page_cache_sources;
        std::fprintf(stderr, "[moe_stream_batch] %s: page-cache buffered source enabled: %s\n", label, path);
    }
#else
    if (g_expert_pack.io_backend == 1 || g_expert_pack.io_backend == 2) {
        if (direct_required) {
            std::fprintf(stderr, "[moe_stream_batch] %s: required O_DIRECT is unsupported on this platform\n", label);
            std::fclose(file);
            return false;
        }
        std::fprintf(stderr, "[moe_stream_batch] %s: direct/io_uring reads are not supported on this platform; using buffered reads\n", label);
        g_expert_pack.io_backend = 0;
    }
#endif

    g_expert_pack.sources.push_back(source);
    return true;
}
#endif

#if !defined(GGML_SSD_STORE)
static bool expert_pack_load_source(const char *path, int32_t source_idx, std::vector<expert_pack_entry> &entries, bool page_cache_source = false) {
    if (!path || !path[0]) return false;

    FILE *probe = std::fopen(path, "rb");
    if (!probe) {
        std::fprintf(stderr, "[moe_stream_batch] expert pack: open failed: %s\n", path);
        return false;
    }

    char magic[16] = {};
    uint32_t version = 0;
    uint32_t header_size = 0;
    uint64_t n_entries = 0;
    uint64_t data_start = 0;
    if (!expert_pack_read_exact(probe, magic, sizeof(magic)) ||
            !expert_pack_read_exact(probe, &version, sizeof(version)) ||
            !expert_pack_read_exact(probe, &header_size, sizeof(header_size)) ||
            !expert_pack_read_exact(probe, &n_entries, sizeof(n_entries)) ||
            !expert_pack_read_exact(probe, &data_start, sizeof(data_start)) ||
            std::memcmp(magic, "GGMLMOEPACKv1", 13) != 0 ||
            version != 1 || header_size < 40 || data_start < header_size || n_entries > 10000000ULL) {
        std::fprintf(stderr, "[moe_stream_batch] expert pack: invalid header: %s\n", path);
        std::fclose(probe);
        return false;
    }
    std::fclose(probe);

    if ((size_t)source_idx != g_expert_pack.sources.size()) {
        std::fprintf(stderr, "[moe_stream_batch] expert pack: internal source index mismatch for %s\n", path);
        return false;
    }
    if (!expert_pack_open_data_source(path, page_cache_source ? "page-cache expert pack" : "expert pack", page_cache_source)) {
        return false;
    }
    expert_pack_source &source = g_expert_pack.sources.back();
    if (::fseeko(source.file, 40, SEEK_SET) != 0) {
        std::fprintf(stderr, "[moe_stream_batch] expert pack: seek index failed: %s\n", path);
        return false;
    }

    const size_t before = entries.size();
    entries.resize(before + (size_t)n_entries);
    for (uint64_t i = 0; i < n_entries; ++i) {
        expert_pack_entry &e = entries[before + (size_t)i];
        uint32_t reserved = 0;
        if (!expert_pack_read_exact(source.file, e.tensor, sizeof(e.tensor)) ||
                !expert_pack_read_exact(source.file, &e.expert_idx, sizeof(e.expert_idx)) ||
                !expert_pack_read_exact(source.file, &reserved, sizeof(reserved)) ||
                !expert_pack_read_exact(source.file, &e.offset, sizeof(e.offset)) ||
                !expert_pack_read_exact(source.file, &e.nbytes, sizeof(e.nbytes))) {
            std::fprintf(stderr, "[moe_stream_batch] expert pack: short index: %s\n", path);
            return false;
        }
        e.tensor[sizeof(e.tensor) - 1] = '\0';
        e.source_idx = source_idx;
    }

    std::fprintf(stderr, "[moe_stream_batch] expert pack: loaded %lu entries from %s\n",
                 (unsigned long)n_entries, path);
    return true;
}
#endif

static bool expert_pack_v2_full_cover_down_iouring_enabled();

static bool expert_pack_v2_shadow_direct_read_enabled() {
    return expert_pack_env_bool("GGML_MOE_EXPERT_PACK_V2_SHADOW_DIRECT_READ", false);
}

static bool expert_pack_v2_shadow_iouring_read_enabled() {
    return expert_pack_env_bool("GGML_MOE_EXPERT_PACK_V2_SHADOW_IOURING_READ", false);
}

static bool expert_pack_v2_shadow_coalesce_same_layer_enabled() {
    return expert_pack_env_bool("GGML_MOE_EXPERT_PACK_V2_SHADOW_COALESCE_SAME_LAYER", false);
}

static bool expert_pack_v2_shadow_direct_open_allowed() {
    if (!expert_pack_v2_shadow_direct_read_enabled() &&
            !expert_pack_v2_shadow_iouring_read_enabled() &&
            !expert_pack_v2_full_cover_down_iouring_enabled()) return false;
    const char *io_backend_env = std::getenv("GGML_MOE_IO_BACKEND");
    return io_backend_env && (
        std::strcmp(io_backend_env, "direct") == 0 ||
        std::strcmp(io_backend_env, "iouring") == 0 ||
        std::strcmp(io_backend_env, "io_uring") == 0);
}

static bool expert_pack_v2_load_source(const char *path, int32_t source_idx, std::vector<expert_pack_v2_entry> &entries) {
    if (!path || !path[0]) return false;

    FILE *file = std::fopen(path, "rb");
    if (!file) {
        std::fprintf(stderr, "[moe_stream_batch] expert pack v2: open failed: %s\n", path);
        return false;
    }

    char magic[16] = {};
    uint32_t version = 0;
    uint32_t header_size = 0;
    uint64_t n_entries = 0;
    uint64_t data_start = 0;
    if (!expert_pack_read_exact(file, magic, sizeof(magic)) ||
            !expert_pack_read_exact(file, &version, sizeof(version)) ||
            !expert_pack_read_exact(file, &header_size, sizeof(header_size)) ||
            !expert_pack_read_exact(file, &n_entries, sizeof(n_entries)) ||
            !expert_pack_read_exact(file, &data_start, sizeof(data_start)) ||
            std::memcmp(magic, "GGMLMOEPACKv2", 13) != 0 ||
            version != 2 || header_size < 40 || data_start < header_size || n_entries > 10000000ULL) {
        std::fprintf(stderr, "[moe_stream_batch] expert pack v2: invalid header: %s\n", path);
        std::fclose(file);
        return false;
    }

    const size_t before = entries.size();
    entries.resize(before + (size_t)n_entries);
    for (uint64_t i = 0; i < n_entries; ++i) {
        expert_pack_v2_entry &e = entries[before + (size_t)i];
        if (!expert_pack_read_exact(file, e.tensor, sizeof(e.tensor)) ||
                !expert_pack_read_exact(file, &e.expert_idx, sizeof(e.expert_idx)) ||
                !expert_pack_read_exact(file, &e.packed_type, sizeof(e.packed_type)) ||
                !expert_pack_read_exact(file, &e.offset, sizeof(e.offset)) ||
                !expert_pack_read_exact(file, &e.nbytes, sizeof(e.nbytes)) ||
                !expert_pack_read_exact(file, &e.ne00, sizeof(e.ne00)) ||
                !expert_pack_read_exact(file, &e.ne01, sizeof(e.ne01)) ||
                !expert_pack_read_exact(file, &e.nb01, sizeof(e.nb01)) ||
                !expert_pack_read_exact(file, &e.reserved, sizeof(e.reserved))) {
            std::fprintf(stderr, "[moe_stream_batch] expert pack v2: short index: %s\n", path);
            std::fclose(file);
            return false;
        }
        e.tensor[sizeof(e.tensor) - 1] = '\0';
        e.source_idx = source_idx;
        if (!e.tensor[0] || e.expert_idx < 0 || e.packed_type <= 0 ||
                e.packed_type >= GGML_TYPE_COUNT || e.offset < data_start ||
                e.nbytes == 0 || e.ne00 <= 0 || e.ne01 <= 0 || e.nb01 == 0) {
            std::fprintf(stderr, "[moe_stream_batch] expert pack v2: invalid entry %lu in %s\n",
                         (unsigned long)i, path);
            std::fclose(file);
            return false;
        }
    }
    expert_pack_source source;
    source.file = file;
    std::snprintf(source.path, sizeof(source.path), "%s", path);
#if !defined(_WIN32)
    source.fd_direct = -1;
    if (expert_pack_v2_shadow_direct_open_allowed()) {
#if defined(O_DIRECT)
        source.fd_direct = ::open(path, O_RDONLY | O_DIRECT);
#endif
        if (source.fd_direct < 0) {
            std::fprintf(stderr,
                "[moe_stream_batch] expert pack v2: shadow direct open failed; using buffered debug reads: %s\n",
                path);
        } else {
            std::fprintf(stderr,
                "[moe_stream_batch] expert pack v2: shadow direct reads enabled: %s\n",
                path);
        }
    }
#endif
    g_expert_pack_v2.sources.push_back(source);
    std::fprintf(stderr, "[moe_stream_batch] expert pack v2: loaded %lu metadata entries from %s\n",
                 (unsigned long)n_entries, path);
    return true;
}

static void expert_pack_split_tsv_line(char *line, std::vector<char *> &fields) {
    fields.clear();
    char *p = line;
    while (p && *p) {
        fields.push_back(p);
        char *tab = std::strchr(p, '\t');
        if (!tab) break;
        *tab = '\0';
        p = tab + 1;
    }
    if (!fields.empty()) {
        char *last = fields.back();
        const size_t len = std::strlen(last);
        if (len > 0 && last[len - 1] == '\n') last[len - 1] = '\0';
        const size_t len2 = std::strlen(last);
        if (len2 > 0 && last[len2 - 1] == '\r') last[len2 - 1] = '\0';
    }
}

static int expert_pack_tsv_column(const std::vector<char *> &header, const char *name) {
    for (size_t i = 0; i < header.size(); ++i) {
        if (std::strcmp(header[i], name) == 0) return (int)i;
    }
    return -1;
}

// one alias index of GGML_MOE_EXPERT_GGUF_ALIAS_TSV (a ':'-separated list: the experts', then e.g. the dense weights
// homed on the drive); alias_sources: the data files the list's indexes opened so far, shared across them
#if !defined(GGML_SSD_STORE)
static bool expert_pack_load_gguf_alias_tsv(const char *path, std::vector<expert_pack_entry> &entries,
                                            std::unordered_map<std::string, int32_t> &alias_sources) {
    if (!path || !path[0]) return true;
    FILE *file = std::fopen(path, "r");
    if (!file) {
        std::fprintf(stderr, "[moe_stream_batch] GGUF alias: open failed: %s\n", path);
        return false;
    }

    std::unordered_set<std::string> existing_keys;
    existing_keys.reserve(entries.size() * 2 + 1024);
    for (const expert_pack_entry &e : entries) {
        existing_keys.insert(expert_pack_key_string(e.tensor, e.expert_idx, e.nbytes));
    }

    char line[4096];
    std::vector<char *> fields;
    if (!std::fgets(line, sizeof(line), file)) {
        std::fprintf(stderr, "[moe_stream_batch] GGUF alias: empty TSV: %s\n", path);
        std::fclose(file);
        return false;
    }
    expert_pack_split_tsv_line(line, fields);
    const int col_tensor = expert_pack_tsv_column(fields, "tensor");
    int col_expert = expert_pack_tsv_column(fields, "expert_idx");
    if (col_expert < 0) {
        col_expert = expert_pack_tsv_column(fields, "expert");
    }
    const int col_nbytes = expert_pack_tsv_column(fields, "nbytes");
    const int col_source = expert_pack_tsv_column(fields, "source_path");
    int col_offset = expert_pack_tsv_column(fields, "offset");
    if (col_offset < 0) {
        col_offset = expert_pack_tsv_column(fields, "model_offset");
    }
    if (col_offset < 0) {
        col_offset = expert_pack_tsv_column(fields, "offset_bytes");
    }
    if (col_tensor < 0 || col_expert < 0 || col_nbytes < 0 || col_source < 0 || col_offset < 0) {
        std::fprintf(stderr, "[moe_stream_batch] GGUF alias: required TSV columns missing: %s\n", path);
        std::fclose(file);
        return false;
    }

    uint64_t rows = 0;
    uint64_t loaded = 0;
    uint64_t skipped_duplicates = 0;
    uint64_t bad_rows = 0;
    while (std::fgets(line, sizeof(line), file)) {
        expert_pack_split_tsv_line(line, fields);
        ++rows;
        const int need_cols = std::max(std::max(col_tensor, col_expert), std::max(col_nbytes, std::max(col_source, col_offset))) + 1;
        if ((int)fields.size() < need_cols) {
            ++bad_rows;
            continue;
        }
        const char *tensor = fields[(size_t)col_tensor];
        const char *source_path = fields[(size_t)col_source];
        const int expert_idx = (int)std::strtol(fields[(size_t)col_expert], nullptr, 10);
        const uint64_t nbytes = (uint64_t)std::strtoull(fields[(size_t)col_nbytes], nullptr, 10);
        const uint64_t offset = (uint64_t)std::strtoull(fields[(size_t)col_offset], nullptr, 10);
        if (!tensor || !tensor[0] || !source_path || !source_path[0] ||
                expert_idx < 0 || nbytes == 0 || std::strlen(tensor) >= 128) {
            ++bad_rows;
            continue;
        }
        const std::string key = expert_pack_key_string(tensor, expert_idx, nbytes);
        if (existing_keys.find(key) != existing_keys.end()) {
            ++skipped_duplicates;
            continue;
        }

        int32_t source_idx = -1;
        auto it = alias_sources.find(source_path);
        if (it != alias_sources.end()) {
            source_idx = it->second;
        } else {
            source_idx = (int32_t)g_expert_pack.sources.size();
            if (!expert_pack_open_data_source(source_path, "GGUF alias", false,
                    expert_pack_env_bool("GGML_MOE_IO_REQUIRE_DIRECT", false))) {
                std::fclose(file);
                return false;
            }
            alias_sources[source_path] = source_idx;
        }

        expert_pack_entry e = {};
        std::snprintf(e.tensor, sizeof(e.tensor), "%s", tensor);
        e.expert_idx = expert_idx;
        e.source_idx = source_idx;
        e.offset = offset;
        e.nbytes = nbytes;
        e.alias_source = true;
        entries.push_back(e);
        existing_keys.insert(key);
        ++loaded;
    }
    std::fclose(file);

    std::fprintf(stderr,
            "[moe_stream_batch] GGUF alias: loaded %lu entries from %s sources=%zu skipped_duplicates=%lu bad_rows=%lu rows=%lu\n",
            (unsigned long)loaded, path, alias_sources.size(),
            (unsigned long)skipped_duplicates, (unsigned long)bad_rows, (unsigned long)rows);
    return bad_rows == 0;
}
#endif

static bool expert_pack_open_alias_source(const char *path, int32_t source_idx) {
    if (!path || !path[0]) return false;

    FILE *file = std::fopen(path, "rb");
    if (!file) {
        std::fprintf(stderr, "[moe_stream_batch] expert alias source: open failed: %s\n", path);
        return false;
    }

    expert_pack_source source;
    source.file = file;
    std::snprintf(source.path, sizeof(source.path), "%s", path);
#if !defined(_WIN32)
    source.fd_direct = -1;
    if (g_expert_pack.io_backend == 1 || g_expert_pack.io_backend == 2) {
#if defined(O_DIRECT)
        source.fd_direct = ::open(path, O_RDONLY | O_DIRECT);
#endif
        if (source.fd_direct < 0) {
            std::fprintf(stderr, "[moe_stream_batch] expert alias source: direct open failed; using buffered reads: %s\n", path);
            g_expert_pack.io_backend = 0;
        } else if (g_expert_pack.io_backend == 2) {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
            std::fprintf(stderr, "[moe_stream_batch] expert alias source: io_uring direct reads enabled: %s io_bytes=%zu depth=%zu\n",
                         path, expert_pack_io_bytes(), expert_pack_io_depth());
#else
            std::fprintf(stderr, "[moe_stream_batch] expert alias source: io_uring requested but liburing headers are unavailable; using direct fallback\n");
            g_expert_pack.io_backend = 1;
#endif
        } else {
            std::fprintf(stderr, "[moe_stream_batch] expert alias source: direct reads enabled: %s\n", path);
        }
    }
#else
    if (g_expert_pack.io_backend == 1 || g_expert_pack.io_backend == 2) {
        std::fprintf(stderr, "[moe_stream_batch] expert alias source: direct/io_uring reads are not supported on this platform; using buffered reads\n");
        g_expert_pack.io_backend = 0;
    }
#endif

    if ((size_t)source_idx != g_expert_pack.sources.size()) {
        std::fprintf(stderr, "[moe_stream_batch] expert alias source: internal source index mismatch for %s\n", path);
        std::fclose(file);
        return false;
    }
    g_expert_pack.sources.push_back(source);
    return true;
}

static void expert_pack_append_env_list(std::vector<std::string> &paths, const char *env_name) {
    const char *env = std::getenv(env_name);
    if (!env || !env[0]) return;

    const std::string value(env);
    size_t start = 0;
    while (start <= value.size()) {
        const size_t end = value.find(':', start);
        const size_t stop = end == std::string::npos ? value.size() : end;
        if (stop > start) {
            paths.emplace_back(value.substr(start, stop - start));
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
}

static void expert_pack_v2_init_once() {
    std::lock_guard<std::mutex> lk(g_expert_pack_v2.mu);
    if (g_expert_pack_v2.inited) return;

    std::vector<std::string> source_paths;
    expert_pack_append_env_list(source_paths, "GGML_MOE_EXPERT_PACK_V2");
    if (source_paths.empty()) {
        g_expert_pack_v2.inited = true;
        return;
    }

    std::vector<expert_pack_v2_entry> entries;
    for (const std::string &source_path : source_paths) {
        if (!expert_pack_v2_load_source(source_path.c_str(), (int32_t)g_expert_pack_v2.sources.size(), entries)) {
            g_expert_pack_v2.inited = true;
            return;
        }
    }

    std::sort(entries.begin(), entries.end(),
        [](const expert_pack_v2_entry &a, const expert_pack_v2_entry &b) {
            const int name_cmp = std::strcmp(a.tensor, b.tensor);
            if (name_cmp != 0) return name_cmp < 0;
            if (a.expert_idx != b.expert_idx) return a.expert_idx < b.expert_idx;
            return a.source_idx < b.source_idx;
        });

    for (size_t i = 1; i < entries.size(); ++i) {
        if (expert_pack_v2_entry_cmp(entries[i - 1], entries[i].tensor, entries[i].expert_idx) == 0) {
            std::fprintf(stderr, "[moe_stream_batch] expert pack v2: duplicate key across packs: %s expert=%d\n",
                         entries[i].tensor, entries[i].expert_idx);
            g_expert_pack_v2.inited = true;
            return;
        }
    }

    g_expert_pack_v2.entries = std::move(entries);
    g_expert_pack_v2.enabled = true;
    g_expert_pack_v2.inited = true;
    std::fprintf(stderr, "[moe_stream_batch] expert pack v2: total metadata entries=%zu sources=%zu\n",
                 g_expert_pack_v2.entries.size(), g_expert_pack_v2.sources.size());
}

static const expert_pack_v2_entry * expert_pack_v2_lookup(const char *tensor_name, int expert_idx) {
    expert_pack_v2_init_once();
    if (!g_expert_pack_v2.enabled || !tensor_name || !tensor_name[0] || expert_idx < 0) return nullptr;

    size_t lo = 0;
    size_t hi = g_expert_pack_v2.entries.size();
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const int cmp = expert_pack_v2_entry_cmp(g_expert_pack_v2.entries[mid], tensor_name, expert_idx);
        if (cmp < 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < g_expert_pack_v2.entries.size() &&
            expert_pack_v2_entry_cmp(g_expert_pack_v2.entries[lo], tensor_name, expert_idx) == 0) {
        return &g_expert_pack_v2.entries[lo];
    }
    return nullptr;
}

static expert_pack_source * expert_pack_v2_source_for_entry(const expert_pack_v2_entry *entry) {
    if (!entry || entry->source_idx < 0 ||
            (size_t)entry->source_idx >= g_expert_pack_v2.sources.size()) {
        return nullptr;
    }
    return &g_expert_pack_v2.sources[(size_t)entry->source_idx];
}

extern "C" bool ggml_cuda_moe_expert_pack_v2_lookup_debug(
        const char *tensor_name,
        int expert_idx,
        int *packed_type,
        int64_t *ne00,
        int64_t *ne01,
        size_t *nb01,
        size_t *nbytes) {
    const expert_pack_v2_entry *entry = expert_pack_v2_lookup(tensor_name, expert_idx);
    if (!entry) return false;
    if (packed_type) *packed_type = entry->packed_type;
    if (ne00) *ne00 = entry->ne00;
    if (ne01) *ne01 = entry->ne01;
    if (nb01) *nb01 = (size_t)entry->nb01;
    if (nbytes) *nbytes = (size_t)entry->nbytes;
    return true;
}

extern "C" bool ggml_cuda_moe_expert_pack_v2_read_debug(
        const char *tensor_name,
        int expert_idx,
        void *dst,
        size_t dst_capacity,
        size_t *nread) {
    if (nread) *nread = 0;
    if (!dst || dst_capacity == 0) return false;
    const expert_pack_v2_entry *entry = expert_pack_v2_lookup(tensor_name, expert_idx);
    if (!entry || entry->nbytes == 0 || entry->nbytes > dst_capacity) return false;
    expert_pack_source *source = expert_pack_v2_source_for_entry(entry);
    if (!source || !source->file) return false;

    std::lock_guard<std::mutex> lk(g_expert_pack_v2.mu);
#if defined(_WIN32)
    if (_fseeki64(source->file, (int64_t)entry->offset, SEEK_SET) != 0) {
#else
    if (::fseeko(source->file, (off_t)entry->offset, SEEK_SET) != 0) {
#endif
        return false;
    }
    const size_t sz = (size_t)entry->nbytes;
    if (!expert_pack_read_exact(source->file, dst, sz)) {
        return false;
    }
    if (nread) *nread = sz;
    return true;
}

static bool expert_pack_v2_packed_type_supported(int packed_type) {
    return packed_type == GGML_TYPE_IQ1_S ||
        packed_type == GGML_TYPE_IQ3_XXS || packed_type == GGML_TYPE_IQ3_S ||
        packed_type == GGML_TYPE_IQ2_XXS || packed_type == GGML_TYPE_IQ2_XS ||
        packed_type == GGML_TYPE_IQ2_S || packed_type == GGML_TYPE_Q2_K ||
        packed_type == GGML_TYPE_Q3_K || packed_type == GGML_TYPE_Q4_0 ||
        packed_type == GGML_TYPE_Q4_K || packed_type == GGML_TYPE_IQ4_XS;
}

static void expert_pack_v2_shadow_profile_record(
        const char *phase,
        const char *tensor_name,
        int expert_idx,
        int logical_type,
        size_t logical_nbytes,
        const char *cache_state) {
    const char *path = std::getenv("GGML_MOE_EXPERT_PACK_V2_SHADOW_PROFILE_OUT");
    if (!path || !path[0] || !tensor_name || !tensor_name[0] || expert_idx < 0) return;

    const expert_pack_v2_entry *entry = expert_pack_v2_lookup(tensor_name, expert_idx);
    const int v2_hit = entry ? 1 : 0;
    const int packed_type = entry ? entry->packed_type : -1;
    const size_t packed_nbytes = entry ? (size_t)entry->nbytes : 0;
    const int64_t ne00 = entry ? entry->ne00 : 0;
    const int64_t ne01 = entry ? entry->ne01 : 0;
    const size_t nb01 = entry ? (size_t)entry->nb01 : 0;
    const int supported = entry && expert_pack_v2_packed_type_supported(entry->packed_type) ? 1 : 0;
    const uint64_t byte_ratio_milli = (entry && logical_nbytes > 0) ?
        (uint64_t)((packed_nbytes * 1000ULL) / logical_nbytes) : 0;

    static std::mutex mu;
    static bool header_written = false;
    static uint64_t seq = 0;
    std::lock_guard<std::mutex> lk(mu);
    FILE *f = std::fopen(path, "a");
    if (!f) return;
    if (!header_written) {
        std::fprintf(f,
                "seq,phase,tensor,expert_idx,logical_type,logical_nbytes,cache_state,"
                "v2_hit,packed_type,packed_nbytes,ne00,ne01,nb01,supported,byte_ratio_milli\n");
        header_written = true;
    }
    std::fprintf(f,
            "%lu,%s,%s,%d,%d,%zu,%s,%d,%d,%zu,%ld,%ld,%zu,%d,%lu\n",
            (unsigned long)++seq,
            phase ? phase : "",
            tensor_name,
            expert_idx,
            logical_type,
            logical_nbytes,
            cache_state ? cache_state : "",
            v2_hit,
            packed_type,
            packed_nbytes,
            (long)ne00,
            (long)ne01,
            nb01,
            supported,
            (unsigned long)byte_ratio_milli);
    std::fclose(f);
}

struct expert_pack_v2_full_cover_down_state {
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> accepted_calls{0};
    std::atomic<uint64_t> rejected_not_decode_down{0};
    std::atomic<uint64_t> rejected_special_path{0};
    std::atomic<uint64_t> rejected_bad_cache_key{0};
    std::atomic<uint64_t> rejected_no_entry{0};
    std::atomic<uint64_t> rejected_unsupported{0};
    std::atomic<uint64_t> rejected_shape{0};
    std::atomic<uint64_t> rejected_homogeneous{0};
    std::atomic<uint64_t> rejected_not_smaller{0};
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> cache_misses{0};
    std::atomic<uint64_t> copied_entries{0};
    std::atomic<uint64_t> copied_bytes{0};
    std::atomic<uint64_t> saved_bytes{0};
    std::atomic<uint64_t> read_us{0};
    std::atomic<uint64_t> h2d_enqueue_us{0};
    std::atomic<uint64_t> total_us{0};
    std::atomic<uint64_t> insert_failures{0};
    std::atomic<uint64_t> read_failures{0};
    std::atomic<uint64_t> h2d_failures{0};
    std::atomic<uint64_t> overlap_calls{0};
    std::atomic<uint64_t> overlap_accepted{0};
    std::atomic<uint64_t> overlap_rejected{0};
    std::atomic<uint64_t> overlap_reject_bad_expert{0};
    std::atomic<uint64_t> overlap_reject_no_entry{0};
    std::atomic<uint64_t> overlap_reject_unsupported{0};
    std::atomic<uint64_t> overlap_reject_not_smaller{0};
    std::atomic<uint64_t> overlap_reject_shape{0};
    std::atomic<uint64_t> overlap_reject_homogeneous{0};
    std::atomic<uint64_t> overlap_reject_bad_cache_key{0};
    std::atomic<uint64_t> overlap_cache_hits{0};
    std::atomic<uint64_t> overlap_cache_misses{0};
    std::atomic<uint64_t> overlap_copied_entries{0};
    std::atomic<uint64_t> overlap_copied_bytes{0};
    std::atomic<uint64_t> overlap_read_us{0};
    std::atomic<uint64_t> overlap_h2d_enqueue_us{0};
    std::atomic<uint64_t> overlap_worker_us{0};
    std::atomic<uint64_t> overlap_failures{0};
    void *overlap_host = nullptr;
    size_t overlap_host_sz = 0;
    std::mutex overlap_mu;
    std::atomic<bool> report_registered{false};
};

static expert_pack_v2_full_cover_down_state g_expert_pack_v2_full_cover_down;

static bool expert_pack_v2_full_cover_down_enabled() {
    // The compressed override has a different payload/key domain. Dynamic
    // Q3 role caching keeps all three roles on the ordinary lossless path.
    return !common_dynamic_tier_requested() &&
        expert_pack_env_bool("GGML_MOE_EXPERT_PACK_V2_FULL_COVER_DOWN", false);
}

static bool expert_pack_v2_full_cover_down_overlap_enabled() {
    return expert_pack_env_bool("GGML_MOE_EXPERT_PACK_V2_FULL_COVER_DOWN_OVERLAP", false);
}

static bool expert_pack_v2_full_cover_down_iouring_enabled() {
    return expert_pack_env_bool("GGML_MOE_EXPERT_PACK_V2_FULL_COVER_DOWN_IOURING", false);
}

static const char * expert_pack_v2_full_cover_down_profile_path() {
    const char *path = std::getenv("GGML_MOE_EXPERT_PACK_V2_FULL_COVER_DOWN_PROFILE_OUT");
    return path && path[0] ? path : nullptr;
}

static void expert_pack_v2_full_cover_down_profile_record(
        const char *event,
        const char *reason,
        const char *tensor_name,
        int logical_type,
        int packed_type,
        int n_active,
        size_t logical_nbytes,
        size_t packed_nbytes,
        int cache_hits,
        int cache_misses,
        int copied_entries,
        uint64_t copied_bytes,
        uint64_t saved_bytes,
        uint64_t read_us,
        uint64_t h2d_enqueue_us,
        uint64_t total_us) {
    const char *path = expert_pack_v2_full_cover_down_profile_path();
    if (!path || !path[0]) return;

    static std::mutex mu;
    static bool header_written = false;
    static uint64_t seq = 0;
    std::lock_guard<std::mutex> lk(mu);

    FILE *f = std::fopen(path, "a");
    if (!f) return;
    if (!header_written) {
        std::fprintf(f,
                "seq,event,reason,tensor,logical_type,packed_type,n_active,"
                "logical_nbytes,packed_nbytes,cache_hits,cache_misses,"
                "copied_entries,copied_bytes,saved_bytes,read_us,h2d_enqueue_us,total_us\n");
        header_written = true;
    }
    std::fprintf(f,
            "%lu,%s,%s,%s,%d,%d,%d,%zu,%zu,%d,%d,%d,%lu,%lu,%lu,%lu,%lu\n",
            (unsigned long)++seq,
            event ? event : "",
            reason ? reason : "",
            tensor_name ? tensor_name : "",
            logical_type,
            packed_type,
            n_active,
            logical_nbytes,
            packed_nbytes,
            cache_hits,
            cache_misses,
            copied_entries,
            (unsigned long)copied_bytes,
            (unsigned long)saved_bytes,
            (unsigned long)read_us,
            (unsigned long)h2d_enqueue_us,
            (unsigned long)total_us);
    std::fclose(f);
}

static void expert_pack_v2_full_cover_down_report() {
    const expert_pack_v2_full_cover_down_state &s = g_expert_pack_v2_full_cover_down;
    const uint64_t calls = s.calls.load(std::memory_order_relaxed);
    if (calls == 0) return;
    std::fprintf(stderr,
            "[moe_stream_batch] expert pack v2 full-cover down: "
            "calls=%lu accepted=%lu reject_not_decode_down=%lu reject_special_path=%lu "
            "reject_bad_cache_key=%lu reject_no_entry=%lu reject_unsupported=%lu "
            "reject_shape=%lu reject_homogeneous=%lu reject_not_smaller=%lu "
            "cache_hits=%lu cache_misses=%lu copied_entries=%lu copied_bytes=%.3f GiB "
            "saved_bytes=%.3f GiB insert_failures=%lu read_failures=%lu h2d_failures=%lu "
            "read_ms=%.3f h2d_enqueue_ms=%.3f total_ms=%.3f "
            "overlap_calls=%lu overlap_accepted=%lu overlap_rejected=%lu "
            "overlap_reject_bad_expert=%lu overlap_reject_no_entry=%lu "
            "overlap_reject_unsupported=%lu overlap_reject_not_smaller=%lu "
            "overlap_reject_shape=%lu overlap_reject_homogeneous=%lu "
            "overlap_reject_bad_cache_key=%lu "
            "overlap_cache_hits=%lu overlap_cache_misses=%lu overlap_copied_entries=%lu "
            "overlap_copied_bytes=%.3f GiB overlap_read_ms=%.3f "
            "overlap_h2d_enqueue_ms=%.3f overlap_worker_ms=%.3f overlap_failures=%lu\n",
            (unsigned long)calls,
            (unsigned long)s.accepted_calls.load(std::memory_order_relaxed),
            (unsigned long)s.rejected_not_decode_down.load(std::memory_order_relaxed),
            (unsigned long)s.rejected_special_path.load(std::memory_order_relaxed),
            (unsigned long)s.rejected_bad_cache_key.load(std::memory_order_relaxed),
            (unsigned long)s.rejected_no_entry.load(std::memory_order_relaxed),
            (unsigned long)s.rejected_unsupported.load(std::memory_order_relaxed),
            (unsigned long)s.rejected_shape.load(std::memory_order_relaxed),
            (unsigned long)s.rejected_homogeneous.load(std::memory_order_relaxed),
            (unsigned long)s.rejected_not_smaller.load(std::memory_order_relaxed),
            (unsigned long)s.cache_hits.load(std::memory_order_relaxed),
            (unsigned long)s.cache_misses.load(std::memory_order_relaxed),
            (unsigned long)s.copied_entries.load(std::memory_order_relaxed),
            s.copied_bytes.load(std::memory_order_relaxed) / (1024.0 * 1024.0 * 1024.0),
            s.saved_bytes.load(std::memory_order_relaxed) / (1024.0 * 1024.0 * 1024.0),
            (unsigned long)s.insert_failures.load(std::memory_order_relaxed),
            (unsigned long)s.read_failures.load(std::memory_order_relaxed),
            (unsigned long)s.h2d_failures.load(std::memory_order_relaxed),
            s.read_us.load(std::memory_order_relaxed) / 1000.0,
            s.h2d_enqueue_us.load(std::memory_order_relaxed) / 1000.0,
            s.total_us.load(std::memory_order_relaxed) / 1000.0,
            (unsigned long)s.overlap_calls.load(std::memory_order_relaxed),
            (unsigned long)s.overlap_accepted.load(std::memory_order_relaxed),
            (unsigned long)s.overlap_rejected.load(std::memory_order_relaxed),
            (unsigned long)s.overlap_reject_bad_expert.load(std::memory_order_relaxed),
            (unsigned long)s.overlap_reject_no_entry.load(std::memory_order_relaxed),
            (unsigned long)s.overlap_reject_unsupported.load(std::memory_order_relaxed),
            (unsigned long)s.overlap_reject_not_smaller.load(std::memory_order_relaxed),
            (unsigned long)s.overlap_reject_shape.load(std::memory_order_relaxed),
            (unsigned long)s.overlap_reject_homogeneous.load(std::memory_order_relaxed),
            (unsigned long)s.overlap_reject_bad_cache_key.load(std::memory_order_relaxed),
            (unsigned long)s.overlap_cache_hits.load(std::memory_order_relaxed),
            (unsigned long)s.overlap_cache_misses.load(std::memory_order_relaxed),
            (unsigned long)s.overlap_copied_entries.load(std::memory_order_relaxed),
            s.overlap_copied_bytes.load(std::memory_order_relaxed) / (1024.0 * 1024.0 * 1024.0),
            s.overlap_read_us.load(std::memory_order_relaxed) / 1000.0,
            s.overlap_h2d_enqueue_us.load(std::memory_order_relaxed) / 1000.0,
            s.overlap_worker_us.load(std::memory_order_relaxed) / 1000.0,
            (unsigned long)s.overlap_failures.load(std::memory_order_relaxed));
}

static void expert_pack_v2_call_coverage_record(
        const char *phase,
        const char *role,
        const char *tensor_name,
        int logical_type,
        size_t logical_nbytes,
        int64_t logical_ne00,
        int64_t logical_ne01,
        size_t logical_nb01,
        batch_vram_cache *cache,
        const int *active_experts,
        int n_active) {
    const char *path = std::getenv("GGML_MOE_GPU_EXTENSION_COVERAGE_OUT");
    if (!path || !path[0] || !tensor_name || !tensor_name[0] ||
            !active_experts || n_active <= 0 || logical_nbytes == 0) {
        return;
    }

    int cache_hits = 0;
    int v2_hits = 0;
    int v2_supported_hits = 0;
    int v2_smaller_hits = 0;
    bool full_cover = true;
    bool homogeneous = true;
    int packed_type = -1;
    size_t packed_nbytes = 0;
    int64_t packed_ne00 = 0;
    int64_t packed_ne01 = 0;
    size_t packed_nb01 = 0;
    uint64_t split_possible_bytes = 0;
    uint64_t split_possible_saved_bytes = 0;

    for (int j = 0; j < n_active; ++j) {
        const int expert_idx = active_experts[j];
        if (cache && batch_cache_find_slot(cache, batch_key_hash(tensor_name, expert_idx)) >= 0) {
            ++cache_hits;
        }

        const expert_pack_v2_entry *entry = expert_pack_v2_lookup(tensor_name, expert_idx);
        if (!entry) {
            full_cover = false;
            continue;
        }

        ++v2_hits;
        const bool supported = expert_pack_v2_packed_type_supported(entry->packed_type);
        if (supported) {
            ++v2_supported_hits;
        } else {
            full_cover = false;
        }
        const size_t entry_nbytes = (size_t)entry->nbytes;
        split_possible_bytes += (uint64_t)entry_nbytes;
        if (entry_nbytes < logical_nbytes) {
            ++v2_smaller_hits;
            split_possible_saved_bytes += (uint64_t)(logical_nbytes - entry_nbytes);
        }

        if (packed_type < 0) {
            packed_type = entry->packed_type;
            packed_nbytes = entry_nbytes;
            packed_ne00 = entry->ne00;
            packed_ne01 = entry->ne01;
            packed_nb01 = (size_t)entry->nb01;
        } else if (packed_type != entry->packed_type ||
                packed_nbytes != entry_nbytes ||
                packed_ne00 != entry->ne00 ||
                packed_ne01 != entry->ne01 ||
                packed_nb01 != (size_t)entry->nb01) {
            homogeneous = false;
        }
    }

    const uint64_t logical_total_bytes = (uint64_t)logical_nbytes * (uint64_t)n_active;
    const bool full_homogeneous_cover = full_cover && homogeneous && v2_supported_hits == n_active;
    const uint64_t full_cover_bytes = full_homogeneous_cover ? (uint64_t)packed_nbytes * (uint64_t)n_active : 0;
    const uint64_t full_cover_saved_bytes =
        full_homogeneous_cover && logical_total_bytes > full_cover_bytes ?
        logical_total_bytes - full_cover_bytes : 0;

    static std::mutex mu;
    static bool header_written = false;
    static uint64_t seq = 0;
    std::lock_guard<std::mutex> lk(mu);

    FILE *f = std::fopen(path, "a");
    if (!f) return;
    if (!header_written) {
        std::fprintf(f,
                "seq,phase,role,tensor,logical_type,logical_nbytes,logical_ne00,logical_ne01,"
                "logical_nb01,n_active,cache_hits,cache_misses,v2_hits,v2_supported_hits,"
                "v2_smaller_hits,full_cover,homogeneous,full_homogeneous_cover,packed_type,"
                "packed_nbytes,packed_ne00,packed_ne01,packed_nb01,logical_total_bytes,"
                "split_possible_bytes,split_possible_saved_bytes,full_cover_bytes,"
                "full_cover_saved_bytes\n");
        header_written = true;
    }
    std::fprintf(f,
            "%lu,%s,%s,%s,%d,%zu,%ld,%ld,%zu,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%ld,%ld,%zu,%llu,%llu,%llu,%llu,%llu\n",
            (unsigned long)++seq,
            phase ? phase : "",
            role ? role : "",
            tensor_name,
            logical_type,
            logical_nbytes,
            (long)logical_ne00,
            (long)logical_ne01,
            logical_nb01,
            n_active,
            cache_hits,
            n_active - cache_hits,
            v2_hits,
            v2_supported_hits,
            v2_smaller_hits,
            full_cover ? 1 : 0,
            homogeneous ? 1 : 0,
            full_homogeneous_cover ? 1 : 0,
            packed_type,
            packed_nbytes,
            (long)packed_ne00,
            (long)packed_ne01,
            packed_nb01,
            (unsigned long long)logical_total_bytes,
            (unsigned long long)split_possible_bytes,
            (unsigned long long)split_possible_saved_bytes,
            (unsigned long long)full_cover_bytes,
            (unsigned long long)full_cover_saved_bytes);
    std::fclose(f);
}

struct expert_pack_v2_override_manifest_entry {
    int packed_type = -1;
    size_t packed_nbytes = 0;
    int64_t ne00 = 0;
    int64_t ne01 = 0;
    size_t nb01 = 0;
};

struct expert_pack_v2_override_preflight_state {
    std::mutex mu;
    bool inited = false;
    bool enabled = false;
    bool manifest_loaded = false;
    bool report_registered = false;
    std::unordered_map<std::string, expert_pack_v2_override_manifest_entry> manifest;
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> active_entries{0};
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> accepted_entries{0};
    std::atomic<uint64_t> full_cover_calls{0};
    std::atomic<uint64_t> reject_no_manifest{0};
    std::atomic<uint64_t> reject_not_allowed{0};
    std::atomic<uint64_t> reject_no_entry{0};
    std::atomic<uint64_t> reject_unsupported{0};
    std::atomic<uint64_t> reject_manifest_mismatch{0};
    std::atomic<uint64_t> reject_shape_mismatch{0};
    std::atomic<uint64_t> reject_not_smaller{0};
};

static expert_pack_v2_override_preflight_state g_expert_pack_v2_override_preflight;

static std::string expert_pack_v2_override_key(const char *tensor_name, int expert_idx) {
    return std::string(tensor_name ? tensor_name : "") + "#" + std::to_string(expert_idx);
}

static bool expert_pack_v2_override_preflight_env_enabled() {
    const char *preflight = std::getenv("GGML_MOE_EXPERT_PACK_V2_OVERRIDE_PREFLIGHT");
    if (preflight && preflight[0] && preflight[0] != '0') return true;
    const char *override = std::getenv("GGML_MOE_EXPERT_PACK_V2_OVERRIDE");
    if (override && override[0] && override[0] != '0') return true;
    const char *split_plan = std::getenv("GGML_MOE_EXPERT_PACK_V2_PARTIAL_SPLIT_PLAN");
    if (split_plan && split_plan[0] && split_plan[0] != '0') return true;
    const char *control = std::getenv("GGML_MOE_EXPERT_PACK_V2_PARTIAL_SPLIT_CONTROL_PROFILE");
    if (control && control[0] && control[0] != '0') return true;
    const char *shadow_stage = std::getenv("GGML_MOE_EXPERT_PACK_V2_PARTIAL_SPLIT_SHADOW_STAGE");
    return shadow_stage && shadow_stage[0] && shadow_stage[0] != '0';
}

static const char * expert_pack_v2_override_profile_path() {
    const char *path = std::getenv("GGML_MOE_EXPERT_PACK_V2_OVERRIDE_PROFILE_OUT");
    if (path && path[0]) return path;
    path = std::getenv("GGML_MOE_EXPERT_PACK_V2_PREFLIGHT_OUT");
    return path && path[0] ? path : nullptr;
}

static void expert_pack_v2_override_preflight_report() {
    if (!g_expert_pack_v2_override_preflight.enabled) return;
    std::fprintf(stderr,
        "[moe_stream_batch] expert pack v2 override preflight report: "
        "calls=%llu active=%llu cache_hits=%llu accepted=%llu full_cover_calls=%llu "
        "reject_no_manifest=%llu reject_not_allowed=%llu reject_no_entry=%llu "
        "reject_unsupported=%llu reject_manifest_mismatch=%llu "
        "reject_shape_mismatch=%llu reject_not_smaller=%llu manifest_rows=%zu\n",
        (unsigned long long)g_expert_pack_v2_override_preflight.calls.load(),
        (unsigned long long)g_expert_pack_v2_override_preflight.active_entries.load(),
        (unsigned long long)g_expert_pack_v2_override_preflight.cache_hits.load(),
        (unsigned long long)g_expert_pack_v2_override_preflight.accepted_entries.load(),
        (unsigned long long)g_expert_pack_v2_override_preflight.full_cover_calls.load(),
        (unsigned long long)g_expert_pack_v2_override_preflight.reject_no_manifest.load(),
        (unsigned long long)g_expert_pack_v2_override_preflight.reject_not_allowed.load(),
        (unsigned long long)g_expert_pack_v2_override_preflight.reject_no_entry.load(),
        (unsigned long long)g_expert_pack_v2_override_preflight.reject_unsupported.load(),
        (unsigned long long)g_expert_pack_v2_override_preflight.reject_manifest_mismatch.load(),
        (unsigned long long)g_expert_pack_v2_override_preflight.reject_shape_mismatch.load(),
        (unsigned long long)g_expert_pack_v2_override_preflight.reject_not_smaller.load(),
        g_expert_pack_v2_override_preflight.manifest.size());
}

static bool expert_pack_v2_override_manifest_parse_size(
        const std::vector<char *> &fields, int col, size_t &out) {
    if (col < 0 || (size_t)col >= fields.size() || !fields[(size_t)col]) return false;
    char *end = nullptr;
    const unsigned long long v = std::strtoull(fields[(size_t)col], &end, 10);
    if (!end || *end != '\0') return false;
    out = (size_t)v;
    return true;
}

static bool expert_pack_v2_override_manifest_parse_i64(
        const std::vector<char *> &fields, int col, int64_t &out) {
    if (col < 0 || (size_t)col >= fields.size() || !fields[(size_t)col]) return false;
    char *end = nullptr;
    const long long v = std::strtoll(fields[(size_t)col], &end, 10);
    if (!end || *end != '\0') return false;
    out = (int64_t)v;
    return true;
}

static void expert_pack_v2_override_preflight_init_once() {
    std::lock_guard<std::mutex> lk(g_expert_pack_v2_override_preflight.mu);
    if (g_expert_pack_v2_override_preflight.inited) return;

    g_expert_pack_v2_override_preflight.enabled = expert_pack_v2_override_preflight_env_enabled();
    g_expert_pack_v2_override_preflight.inited = true;
    if (!g_expert_pack_v2_override_preflight.enabled) return;

    const char *manifest_path = std::getenv("GGML_MOE_EXPERT_PACK_V2_OVERRIDE_MANIFEST");
    if (!manifest_path || !manifest_path[0]) {
        std::fprintf(stderr,
            "[moe_stream_batch] expert pack v2 override preflight active without manifest; "
            "all entries will be rejected for safety\n");
        g_expert_pack_v2_override_preflight.report_registered = true;
        std::atexit(expert_pack_v2_override_preflight_report);
        return;
    }

    FILE *file = std::fopen(manifest_path, "r");
    if (!file) {
        std::fprintf(stderr,
            "[moe_stream_batch] expert pack v2 override manifest open failed: %s\n",
            manifest_path);
        g_expert_pack_v2_override_preflight.report_registered = true;
        std::atexit(expert_pack_v2_override_preflight_report);
        return;
    }

    char line[4096];
    std::vector<char *> fields;
    if (!std::fgets(line, sizeof(line), file)) {
        std::fprintf(stderr,
            "[moe_stream_batch] expert pack v2 override manifest empty: %s\n",
            manifest_path);
        std::fclose(file);
        g_expert_pack_v2_override_preflight.report_registered = true;
        std::atexit(expert_pack_v2_override_preflight_report);
        return;
    }
    expert_pack_split_tsv_line(line, fields);
    const int col_tensor = expert_pack_tsv_column(fields, "tensor");
    const int col_expert = expert_pack_tsv_column(fields, "expert_idx");
    const int col_type = expert_pack_tsv_column(fields, "packed_type");
    const int col_nbytes = expert_pack_tsv_column(fields, "packed_nbytes");
    const int col_ne00 = expert_pack_tsv_column(fields, "packed_ne00");
    const int col_ne01 = expert_pack_tsv_column(fields, "packed_ne01");
    const int col_nb01 = expert_pack_tsv_column(fields, "packed_nb01");
    if (col_tensor < 0 || col_expert < 0 || col_type < 0 || col_nbytes < 0 ||
            col_ne00 < 0 || col_ne01 < 0 || col_nb01 < 0) {
        std::fprintf(stderr,
            "[moe_stream_batch] expert pack v2 override manifest missing required columns: %s\n",
            manifest_path);
        std::fclose(file);
        g_expert_pack_v2_override_preflight.report_registered = true;
        std::atexit(expert_pack_v2_override_preflight_report);
        return;
    }

    uint64_t rows = 0;
    uint64_t loaded = 0;
    uint64_t bad_rows = 0;
    while (std::fgets(line, sizeof(line), file)) {
        expert_pack_split_tsv_line(line, fields);
        ++rows;
        const int need_cols = std::max(
            std::max(col_tensor, col_expert),
            std::max(std::max(col_type, col_nbytes), std::max(col_ne00, std::max(col_ne01, col_nb01)))) + 1;
        if ((int)fields.size() < need_cols) {
            ++bad_rows;
            continue;
        }
        const char *tensor = fields[(size_t)col_tensor];
        char *end = nullptr;
        const long expert_long = std::strtol(fields[(size_t)col_expert], &end, 10);
        if (!tensor || !tensor[0] || std::strlen(tensor) >= 128 || !end || *end != '\0' ||
                expert_long < 0 || expert_long > INT_MAX) {
            ++bad_rows;
            continue;
        }
        int64_t packed_type_i64 = 0;
        int64_t ne00 = 0;
        int64_t ne01 = 0;
        size_t packed_nbytes = 0;
        size_t nb01 = 0;
        if (!expert_pack_v2_override_manifest_parse_i64(fields, col_type, packed_type_i64) ||
                !expert_pack_v2_override_manifest_parse_size(fields, col_nbytes, packed_nbytes) ||
                !expert_pack_v2_override_manifest_parse_i64(fields, col_ne00, ne00) ||
                !expert_pack_v2_override_manifest_parse_i64(fields, col_ne01, ne01) ||
                !expert_pack_v2_override_manifest_parse_size(fields, col_nb01, nb01) ||
                packed_type_i64 <= 0 || packed_type_i64 >= GGML_TYPE_COUNT ||
                packed_nbytes == 0 || ne00 <= 0 || ne01 <= 0 || nb01 == 0) {
            ++bad_rows;
            continue;
        }

        expert_pack_v2_override_manifest_entry entry;
        entry.packed_type = (int)packed_type_i64;
        entry.packed_nbytes = packed_nbytes;
        entry.ne00 = ne00;
        entry.ne01 = ne01;
        entry.nb01 = nb01;
        g_expert_pack_v2_override_preflight.manifest[
            expert_pack_v2_override_key(tensor, (int)expert_long)] = entry;
        ++loaded;
    }
    std::fclose(file);
    g_expert_pack_v2_override_preflight.manifest_loaded = loaded > 0 && bad_rows == 0;
    g_expert_pack_v2_override_preflight.report_registered = true;
    std::atexit(expert_pack_v2_override_preflight_report);
    std::fprintf(stderr,
        "[moe_stream_batch] expert pack v2 override preflight active: manifest=%s "
        "loaded=%llu bad_rows=%llu dispatch=disabled\n",
        manifest_path, (unsigned long long)loaded, (unsigned long long)bad_rows);
}

static void expert_pack_v2_override_preflight_record(
        const char *phase,
        const char *role,
        const char *tensor_name,
        int logical_type,
        size_t logical_nbytes,
        int64_t logical_ne00,
        int64_t logical_ne01,
        size_t logical_nb01,
        batch_vram_cache *cache,
        const int *active_experts,
        int n_active) {
    if (!tensor_name || !tensor_name[0] || !active_experts || n_active <= 0 || logical_nbytes == 0) {
        return;
    }
    expert_pack_v2_override_preflight_init_once();
    if (!g_expert_pack_v2_override_preflight.enabled) return;

    int cache_hits = 0;
    int accepted = 0;
    int reject_no_manifest = 0;
    int reject_not_allowed = 0;
    int reject_no_entry = 0;
    int reject_unsupported = 0;
    int reject_manifest_mismatch = 0;
    int reject_shape_mismatch = 0;
    int reject_not_smaller = 0;
    bool homogeneous = true;
    int packed_type = -1;
    size_t packed_nbytes = 0;
    int64_t packed_ne00 = 0;
    int64_t packed_ne01 = 0;
    size_t packed_nb01 = 0;
    uint64_t accepted_bytes = 0;
    uint64_t accepted_saved_bytes = 0;

    for (int j = 0; j < n_active; ++j) {
        const int expert_idx = active_experts[j];
        if (cache && batch_cache_find_slot(cache, batch_key_hash(tensor_name, expert_idx)) >= 0) {
            ++cache_hits;
        }

        if (!g_expert_pack_v2_override_preflight.manifest_loaded) {
            ++reject_no_manifest;
            continue;
        }

        const auto allowed_it = g_expert_pack_v2_override_preflight.manifest.find(
            expert_pack_v2_override_key(tensor_name, expert_idx));
        if (allowed_it == g_expert_pack_v2_override_preflight.manifest.end()) {
            ++reject_not_allowed;
            continue;
        }
        const expert_pack_v2_override_manifest_entry &allowed = allowed_it->second;
        const expert_pack_v2_entry *entry = expert_pack_v2_lookup(tensor_name, expert_idx);
        if (!entry) {
            ++reject_no_entry;
            continue;
        }
        if (entry->packed_type != allowed.packed_type ||
                (size_t)entry->nbytes != allowed.packed_nbytes ||
                entry->ne00 != allowed.ne00 ||
                entry->ne01 != allowed.ne01 ||
                (size_t)entry->nb01 != allowed.nb01) {
            ++reject_manifest_mismatch;
            continue;
        }
        if (!expert_pack_v2_packed_type_supported(entry->packed_type)) {
            ++reject_unsupported;
            continue;
        }
        if (entry->ne00 != logical_ne00 || entry->ne01 != logical_ne01) {
            ++reject_shape_mismatch;
            continue;
        }
        if ((size_t)entry->nbytes >= logical_nbytes) {
            ++reject_not_smaller;
            continue;
        }

        ++accepted;
        accepted_bytes += (uint64_t)entry->nbytes;
        accepted_saved_bytes += (uint64_t)(logical_nbytes - (size_t)entry->nbytes);
        if (packed_type < 0) {
            packed_type = entry->packed_type;
            packed_nbytes = (size_t)entry->nbytes;
            packed_ne00 = entry->ne00;
            packed_ne01 = entry->ne01;
            packed_nb01 = (size_t)entry->nb01;
        } else if (packed_type != entry->packed_type ||
                packed_nbytes != (size_t)entry->nbytes ||
                packed_ne00 != entry->ne00 ||
                packed_ne01 != entry->ne01 ||
                packed_nb01 != (size_t)entry->nb01) {
            homogeneous = false;
        }
    }

    const bool full_cover = accepted == n_active && homogeneous;
    g_expert_pack_v2_override_preflight.calls.fetch_add(1, std::memory_order_relaxed);
    g_expert_pack_v2_override_preflight.active_entries.fetch_add((uint64_t)n_active, std::memory_order_relaxed);
    g_expert_pack_v2_override_preflight.cache_hits.fetch_add((uint64_t)cache_hits, std::memory_order_relaxed);
    g_expert_pack_v2_override_preflight.accepted_entries.fetch_add((uint64_t)accepted, std::memory_order_relaxed);
    if (full_cover) {
        g_expert_pack_v2_override_preflight.full_cover_calls.fetch_add(1, std::memory_order_relaxed);
    }
    g_expert_pack_v2_override_preflight.reject_no_manifest.fetch_add((uint64_t)reject_no_manifest, std::memory_order_relaxed);
    g_expert_pack_v2_override_preflight.reject_not_allowed.fetch_add((uint64_t)reject_not_allowed, std::memory_order_relaxed);
    g_expert_pack_v2_override_preflight.reject_no_entry.fetch_add((uint64_t)reject_no_entry, std::memory_order_relaxed);
    g_expert_pack_v2_override_preflight.reject_unsupported.fetch_add((uint64_t)reject_unsupported, std::memory_order_relaxed);
    g_expert_pack_v2_override_preflight.reject_manifest_mismatch.fetch_add((uint64_t)reject_manifest_mismatch, std::memory_order_relaxed);
    g_expert_pack_v2_override_preflight.reject_shape_mismatch.fetch_add((uint64_t)reject_shape_mismatch, std::memory_order_relaxed);
    g_expert_pack_v2_override_preflight.reject_not_smaller.fetch_add((uint64_t)reject_not_smaller, std::memory_order_relaxed);

    const char *path = expert_pack_v2_override_profile_path();
    if (!path || !path[0]) return;

    static std::mutex profile_mu;
    static bool header_written = false;
    static uint64_t seq = 0;
    std::lock_guard<std::mutex> profile_lk(profile_mu);
    FILE *f = std::fopen(path, "a");
    if (!f) return;
    if (!header_written) {
        std::fprintf(f,
            "seq,phase,role,tensor,logical_type,logical_nbytes,logical_ne00,"
            "logical_ne01,logical_nb01,n_active,cache_hits,accepted,"
            "reject_no_manifest,reject_not_allowed,reject_no_entry,"
            "reject_unsupported,reject_manifest_mismatch,reject_shape_mismatch,"
            "reject_not_smaller,full_cover,homogeneous,packed_type,packed_nbytes,"
            "packed_ne00,packed_ne01,packed_nb01,logical_total_bytes,"
            "accepted_bytes,accepted_saved_bytes\n");
        header_written = true;
    }
    std::fprintf(f,
        "%llu,%s,%s,%s,%d,%zu,%ld,%ld,%zu,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%ld,%ld,%zu,%llu,%llu,%llu\n",
        (unsigned long long)++seq,
        phase ? phase : "",
        role ? role : "",
        tensor_name,
        logical_type,
        logical_nbytes,
        (long)logical_ne00,
        (long)logical_ne01,
        logical_nb01,
        n_active,
        cache_hits,
        accepted,
        reject_no_manifest,
        reject_not_allowed,
        reject_no_entry,
        reject_unsupported,
        reject_manifest_mismatch,
        reject_shape_mismatch,
        reject_not_smaller,
        full_cover ? 1 : 0,
        homogeneous ? 1 : 0,
        packed_type,
        packed_nbytes,
        (long)packed_ne00,
        (long)packed_ne01,
        packed_nb01,
        (unsigned long long)logical_nbytes * (unsigned long long)n_active,
        (unsigned long long)accepted_bytes,
        (unsigned long long)accepted_saved_bytes);
    std::fclose(f);
}

static bool expert_pack_v2_partial_split_plan_enabled() {
    const char *env = std::getenv("GGML_MOE_EXPERT_PACK_V2_PARTIAL_SPLIT_PLAN");
    return env && env[0] && env[0] != '0';
}

static const char * expert_pack_v2_partial_split_plan_path() {
    const char *path = std::getenv("GGML_MOE_EXPERT_PACK_V2_PARTIAL_SPLIT_PLAN_OUT");
    return path && path[0] ? path : nullptr;
}

static bool expert_pack_v2_partial_split_control_enabled() {
    const char *env = std::getenv("GGML_MOE_EXPERT_PACK_V2_PARTIAL_SPLIT_CONTROL_PROFILE");
    return env && env[0] && env[0] != '0';
}

static const char * expert_pack_v2_partial_split_control_path() {
    const char *path = std::getenv("GGML_MOE_EXPERT_PACK_V2_PARTIAL_SPLIT_CONTROL_OUT");
    return path && path[0] ? path : nullptr;
}

static uint64_t expert_pack_v2_partial_split_plan_max_calls() {
    const char *env = std::getenv("GGML_MOE_EXPERT_PACK_V2_PARTIAL_SPLIT_PLAN_MAX_CALLS");
    if (!env || !env[0]) return 0;
    char *end = nullptr;
    const unsigned long long v = std::strtoull(env, &end, 10);
    return end && *end == '\0' ? (uint64_t)v : 0;
}

static void expert_pack_v2_partial_split_plan_record(
        const char *phase,
        const char *role,
        const char *tensor_name,
        int logical_type,
        size_t logical_nbytes,
        int64_t logical_ne00,
        int64_t logical_ne01,
        size_t logical_nb01,
        batch_vram_cache *cache,
        const int *active_experts,
        const int32_t *dst_ids,
        const int32_t *flat_dst_ids,
        int n_active) {
    const bool plan_enabled = expert_pack_v2_partial_split_plan_enabled();
    const bool control_enabled = expert_pack_v2_partial_split_control_enabled();
    if (!plan_enabled && !control_enabled) return;
    if (!tensor_name || !tensor_name[0] || !active_experts || n_active <= 0 || logical_nbytes == 0) {
        return;
    }

    const char *path = expert_pack_v2_partial_split_plan_path();
    const char *control_path = expert_pack_v2_partial_split_control_path();
    if (plan_enabled && (!path || !path[0]) && (!control_enabled || !control_path || !control_path[0])) return;
    if (!plan_enabled && (!control_path || !control_path[0])) return;

    expert_pack_v2_override_preflight_init_once();
    if (!g_expert_pack_v2_override_preflight.enabled) return;

    static std::atomic<uint64_t> observed_calls{0};
    const uint64_t call_idx = observed_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    const uint64_t max_calls = expert_pack_v2_partial_split_plan_max_calls();
    if (max_calls > 0 && call_idx > max_calls) return;

    int cache_hits = 0;
    int covered = 0;
    int covered_cache_hits = 0;
    int reject_no_manifest = 0;
    int reject_not_allowed = 0;
    int reject_no_entry = 0;
    int reject_unsupported = 0;
    int reject_manifest_mismatch = 0;
    int reject_shape_mismatch = 0;
    int reject_not_smaller = 0;
    bool homogeneous = true;
    int packed_type = -1;
    size_t packed_nbytes = 0;
    int64_t packed_ne00 = 0;
    int64_t packed_ne01 = 0;
    size_t packed_nb01 = 0;
    uint64_t covered_bytes = 0;
    uint64_t covered_saved_bytes = 0;
    uint64_t covered_miss_bytes = 0;
    uint64_t covered_miss_saved_bytes = 0;
    std::vector<unsigned char> covered_flags((size_t)n_active, 0);

    const auto control_t0 = control_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    for (int j = 0; j < n_active; ++j) {
        const int expert_idx = active_experts[j];
        const bool cache_hit = cache && batch_cache_find_slot(cache, batch_key_hash(tensor_name, expert_idx)) >= 0;
        if (cache_hit) {
            ++cache_hits;
        }

        if (!g_expert_pack_v2_override_preflight.manifest_loaded) {
            ++reject_no_manifest;
            continue;
        }

        const auto allowed_it = g_expert_pack_v2_override_preflight.manifest.find(
            expert_pack_v2_override_key(tensor_name, expert_idx));
        if (allowed_it == g_expert_pack_v2_override_preflight.manifest.end()) {
            ++reject_not_allowed;
            continue;
        }
        const expert_pack_v2_override_manifest_entry &allowed = allowed_it->second;
        const expert_pack_v2_entry *entry = expert_pack_v2_lookup(tensor_name, expert_idx);
        if (!entry) {
            ++reject_no_entry;
            continue;
        }
        if (entry->packed_type != allowed.packed_type ||
                (size_t)entry->nbytes != allowed.packed_nbytes ||
                entry->ne00 != allowed.ne00 ||
                entry->ne01 != allowed.ne01 ||
                (size_t)entry->nb01 != allowed.nb01) {
            ++reject_manifest_mismatch;
            continue;
        }
        if (!expert_pack_v2_packed_type_supported(entry->packed_type)) {
            ++reject_unsupported;
            continue;
        }
        if (entry->ne00 != logical_ne00 || entry->ne01 != logical_ne01) {
            ++reject_shape_mismatch;
            continue;
        }
        if ((size_t)entry->nbytes >= logical_nbytes) {
            ++reject_not_smaller;
            continue;
        }

        covered_flags[(size_t)j] = 1;
        ++covered;
        if (cache_hit) {
            ++covered_cache_hits;
        } else {
            covered_miss_bytes += (uint64_t)logical_nbytes;
            covered_miss_saved_bytes += (uint64_t)(logical_nbytes - (size_t)entry->nbytes);
        }
        covered_bytes += (uint64_t)entry->nbytes;
        covered_saved_bytes += (uint64_t)(logical_nbytes - (size_t)entry->nbytes);
        if (packed_type < 0) {
            packed_type = entry->packed_type;
            packed_nbytes = (size_t)entry->nbytes;
            packed_ne00 = entry->ne00;
            packed_ne01 = entry->ne01;
            packed_nb01 = (size_t)entry->nb01;
        } else if (packed_type != entry->packed_type ||
                packed_nbytes != (size_t)entry->nbytes ||
                packed_ne00 != entry->ne00 ||
                packed_ne01 != entry->ne01 ||
                packed_nb01 != (size_t)entry->nb01) {
            homogeneous = false;
        }
    }
    const auto control_t1 = control_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    int covered_runs = 0;
    int uncovered_runs = 0;
    int transitions = 0;
    int prev = -1;
    int first_covered = -1;
    int last_covered = -1;
    int max_covered_run = 0;
    int max_uncovered_run = 0;
    int current_run = 0;
    for (int j = 0; j < n_active; ++j) {
        const int flag = covered_flags[(size_t)j] ? 1 : 0;
        if (flag && first_covered < 0) first_covered = j;
        if (flag) last_covered = j;
        if (flag != prev) {
            if (prev >= 0) ++transitions;
            if (flag) {
                ++covered_runs;
            } else {
                ++uncovered_runs;
            }
            current_run = 1;
            prev = flag;
        } else {
            ++current_run;
        }
        if (flag) {
            max_covered_run = std::max(max_covered_run, current_run);
        } else {
            max_uncovered_run = std::max(max_uncovered_run, current_run);
        }
    }
    const auto control_t2 = control_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    const int uncovered = n_active - covered;
    const bool full_cover = covered == n_active && homogeneous;
    const bool no_cover = covered == 0;
    const int compute_groups = (covered > 0 ? 1 : 0) + (uncovered > 0 ? 1 : 0);
    const int extra_compute_groups = compute_groups > 0 ? compute_groups - 1 : 0;
    const uint64_t logical_total_bytes = (uint64_t)logical_nbytes * (uint64_t)n_active;
    const uint64_t logical_miss_bytes = (uint64_t)logical_nbytes * (uint64_t)(n_active - cache_hits);
    const double partial_saved_ratio = logical_total_bytes > 0 ?
        (double)covered_saved_bytes / (double)logical_total_bytes : 0.0;
    const double partial_miss_saved_ratio = logical_miss_bytes > 0 ?
        (double)covered_miss_saved_bytes / (double)logical_miss_bytes : 0.0;

    double control_lookup_us = 0.0;
    double control_run_us = 0.0;
    double control_map_us = 0.0;
    double control_total_us = 0.0;
    int dst_nonmonotonic = 0;
    int covered_dst_nonmonotonic = 0;
    int uncovered_dst_nonmonotonic = 0;
    int covered_route_first = -1;
    int covered_route_last = -1;
    int uncovered_route_first = -1;
    int uncovered_route_last = -1;
    int covered_dst_first = -1;
    int covered_dst_last = -1;
    int uncovered_dst_first = -1;
    int uncovered_dst_last = -1;
    if (control_enabled) {
        std::vector<int32_t> covered_route_ids;
        std::vector<int32_t> uncovered_route_ids;
        std::vector<int32_t> covered_expert_ids;
        std::vector<int32_t> uncovered_expert_ids;
        std::vector<int32_t> covered_dst_rows;
        std::vector<int32_t> uncovered_dst_rows;
        covered_route_ids.reserve((size_t)covered);
        uncovered_route_ids.reserve((size_t)uncovered);
        covered_expert_ids.reserve((size_t)covered);
        uncovered_expert_ids.reserve((size_t)uncovered);
        covered_dst_rows.reserve((size_t)covered);
        uncovered_dst_rows.reserve((size_t)uncovered);

        int prev_dst = INT_MIN;
        int prev_covered_dst = INT_MIN;
        int prev_uncovered_dst = INT_MIN;
        for (int j = 0; j < n_active; ++j) {
            const int32_t dst = flat_dst_ids ? flat_dst_ids[j] : (dst_ids ? dst_ids[j] : (int32_t)j);
            if (prev_dst != INT_MIN && dst < prev_dst) {
                ++dst_nonmonotonic;
            }
            prev_dst = dst;
            if (covered_flags[(size_t)j]) {
                if (covered_route_first < 0) {
                    covered_route_first = j;
                    covered_dst_first = dst;
                }
                covered_route_last = j;
                covered_dst_last = dst;
                if (prev_covered_dst != INT_MIN && dst < prev_covered_dst) {
                    ++covered_dst_nonmonotonic;
                }
                prev_covered_dst = dst;
                covered_route_ids.push_back((int32_t)j);
                covered_expert_ids.push_back((int32_t)active_experts[j]);
                covered_dst_rows.push_back(dst);
            } else {
                if (uncovered_route_first < 0) {
                    uncovered_route_first = j;
                    uncovered_dst_first = dst;
                }
                uncovered_route_last = j;
                uncovered_dst_last = dst;
                if (prev_uncovered_dst != INT_MIN && dst < prev_uncovered_dst) {
                    ++uncovered_dst_nonmonotonic;
                }
                prev_uncovered_dst = dst;
                uncovered_route_ids.push_back((int32_t)j);
                uncovered_expert_ids.push_back((int32_t)active_experts[j]);
                uncovered_dst_rows.push_back(dst);
            }
        }
        const auto control_t3 = std::chrono::steady_clock::now();
        control_lookup_us = std::chrono::duration<double, std::micro>(control_t1 - control_t0).count();
        control_run_us = std::chrono::duration<double, std::micro>(control_t2 - control_t1).count();
        control_map_us = std::chrono::duration<double, std::micro>(control_t3 - control_t2).count();
        control_total_us = std::chrono::duration<double, std::micro>(control_t3 - control_t0).count();
    }

    if (plan_enabled && path && path[0]) {
        static std::mutex profile_mu;
        static bool header_written = false;
        static uint64_t seq = 0;
        std::lock_guard<std::mutex> profile_lk(profile_mu);
        FILE *f = std::fopen(path, "a");
        if (f) {
            if (!header_written) {
                std::fprintf(f,
                    "seq,call_idx,phase,role,tensor,logical_type,logical_nbytes,"
                    "logical_ne00,logical_ne01,logical_nb01,n_active,cache_hits,"
                    "covered,uncovered,covered_cache_hits,full_cover,no_cover,"
                    "homogeneous,covered_runs,uncovered_runs,transitions,"
                    "first_covered,last_covered,max_covered_run,max_uncovered_run,"
                    "packed_type,packed_nbytes,packed_ne00,packed_ne01,packed_nb01,"
                    "logical_total_bytes,logical_miss_bytes,covered_bytes,"
                    "covered_saved_bytes,covered_miss_bytes,covered_miss_saved_bytes,"
                    "partial_saved_ratio,partial_miss_saved_ratio,"
                    "reject_no_manifest,reject_not_allowed,reject_no_entry,"
                    "reject_unsupported,reject_manifest_mismatch,reject_shape_mismatch,"
                    "reject_not_smaller,compute_groups,extra_compute_groups\n");
                header_written = true;
            }
            std::fprintf(f,
                "%llu,%llu,%s,%s,%s,%d,%zu,%ld,%ld,%zu,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%ld,%ld,%zu,%llu,%llu,%llu,%llu,%llu,%llu,%.9f,%.9f,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                (unsigned long long)++seq,
                (unsigned long long)call_idx,
                phase ? phase : "",
                role ? role : "",
                tensor_name,
                logical_type,
                logical_nbytes,
                (long)logical_ne00,
                (long)logical_ne01,
                logical_nb01,
                n_active,
                cache_hits,
                covered,
                uncovered,
                covered_cache_hits,
                full_cover ? 1 : 0,
                no_cover ? 1 : 0,
                homogeneous ? 1 : 0,
                covered_runs,
                uncovered_runs,
                transitions,
                first_covered,
                last_covered,
                max_covered_run,
                max_uncovered_run,
                packed_type,
                packed_nbytes,
                (long)packed_ne00,
                (long)packed_ne01,
                packed_nb01,
                (unsigned long long)logical_total_bytes,
                (unsigned long long)logical_miss_bytes,
                (unsigned long long)covered_bytes,
                (unsigned long long)covered_saved_bytes,
                (unsigned long long)covered_miss_bytes,
                (unsigned long long)covered_miss_saved_bytes,
                partial_saved_ratio,
                partial_miss_saved_ratio,
                reject_no_manifest,
                reject_not_allowed,
                reject_no_entry,
                reject_unsupported,
                reject_manifest_mismatch,
                reject_shape_mismatch,
                reject_not_smaller,
                compute_groups,
                extra_compute_groups);
            std::fclose(f);
        }
    }

    if (control_enabled && control_path && control_path[0]) {
        static std::mutex control_mu;
        static bool control_header_written = false;
        static uint64_t control_seq = 0;
        std::lock_guard<std::mutex> control_lk(control_mu);
        FILE *f = std::fopen(control_path, "a");
        if (f) {
            if (!control_header_written) {
                std::fprintf(f,
                    "seq,call_idx,phase,role,tensor,n_active,covered,uncovered,"
                    "full_cover,no_cover,homogeneous,covered_runs,uncovered_runs,"
                    "transitions,extra_compute_groups,cache_hits,covered_cache_hits,"
                    "logical_nbytes,covered_miss_saved_bytes,partial_miss_saved_ratio,"
                    "lookup_us,run_us,map_us,total_us,us_per_active,us_per_extra_group,"
                    "dst_nonmonotonic,covered_dst_nonmonotonic,uncovered_dst_nonmonotonic,"
                    "covered_route_first,covered_route_last,uncovered_route_first,"
                    "uncovered_route_last,covered_dst_first,covered_dst_last,"
                    "uncovered_dst_first,uncovered_dst_last\n");
                control_header_written = true;
            }
            const double us_per_active = n_active > 0 ? control_total_us / (double)n_active : 0.0;
            const double us_per_extra_group = extra_compute_groups > 0 ?
                control_total_us / (double)extra_compute_groups : 0.0;
            std::fprintf(f,
                "%llu,%llu,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%llu,%.9f,%.3f,%.3f,%.3f,%.3f,%.6f,%.6f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                (unsigned long long)++control_seq,
                (unsigned long long)call_idx,
                phase ? phase : "",
                role ? role : "",
                tensor_name,
                n_active,
                covered,
                uncovered,
                full_cover ? 1 : 0,
                no_cover ? 1 : 0,
                homogeneous ? 1 : 0,
                covered_runs,
                uncovered_runs,
                transitions,
                extra_compute_groups,
                cache_hits,
                covered_cache_hits,
                logical_nbytes,
                (unsigned long long)covered_miss_saved_bytes,
                partial_miss_saved_ratio,
                control_lookup_us,
                control_run_us,
                control_map_us,
                control_total_us,
                us_per_active,
                us_per_extra_group,
                dst_nonmonotonic,
                covered_dst_nonmonotonic,
                uncovered_dst_nonmonotonic,
                covered_route_first,
                covered_route_last,
                uncovered_route_first,
                uncovered_route_last,
                covered_dst_first,
                covered_dst_last,
                uncovered_dst_first,
                uncovered_dst_last);
            std::fclose(f);
        }
    }
}

static bool expert_pack_page_cache_preload_source(expert_pack_source &source, const char *mode) {
    if (!source.page_cache || !source.file || !mode || !mode[0] || mode[0] == '0') {
        return true;
    }
#if !defined(_WIN32)
    const int fd = ::fileno(source.file);
    if (fd < 0) {
        ++g_expert_pack.page_cache_preload_failures;
        return false;
    }
    struct stat st = {};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ++g_expert_pack.page_cache_preload_failures;
        return false;
    }
    const size_t file_size = (size_t)st.st_size;
    if (std::strcmp(mode, "fadvise") == 0 || std::strcmp(mode, "willneed") == 0) {
        if (::posix_fadvise(fd, 0, (off_t)file_size, POSIX_FADV_WILLNEED) != 0) {
            ++g_expert_pack.page_cache_preload_failures;
            return false;
        }
        g_expert_pack.page_cache_preload_bytes += (uint64_t)file_size;
        std::fprintf(stderr, "[moe_stream_batch] page-cache preload: fadvise %.2f MiB path=%s\n",
                     file_size / (1024.0 * 1024.0), source.path);
        return true;
    }
    if (std::strcmp(mode, "mmap") == 0 || std::strcmp(mode, "touch") == 0) {
        void *base = ::mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) {
            ++g_expert_pack.page_cache_preload_failures;
            return false;
        }
        volatile uint8_t sink = 0;
        const uint8_t *ptr = (const uint8_t *)base;
        const size_t page = 4096;
        for (size_t off = 0; off < file_size; off += page) {
            sink ^= ptr[off];
        }
        if (file_size > 0) {
            sink ^= ptr[file_size - 1];
        }
        (void)sink;
        ::munmap(base, file_size);
        g_expert_pack.page_cache_preload_bytes += (uint64_t)file_size;
        std::fprintf(stderr, "[moe_stream_batch] page-cache preload: mmap-touch %.2f MiB path=%s\n",
                     file_size / (1024.0 * 1024.0), source.path);
        return true;
    }
#endif
    const size_t buf_sz = 8ULL * 1024ULL * 1024ULL;
    std::vector<char> buf(buf_sz);
    if (::fseeko(source.file, 0, SEEK_SET) != 0) {
        ++g_expert_pack.page_cache_preload_failures;
        return false;
    }
    uint64_t total = 0;
    while (true) {
        const size_t got = std::fread(buf.data(), 1, buf.size(), source.file);
        if (got > 0) {
            total += (uint64_t)got;
        }
        if (got < buf.size()) {
            if (std::ferror(source.file)) {
                ++g_expert_pack.page_cache_preload_failures;
                std::clearerr(source.file);
                return false;
            }
            break;
        }
    }
    g_expert_pack.page_cache_preload_bytes += total;
    std::fprintf(stderr, "[moe_stream_batch] page-cache preload: read %.2f MiB path=%s\n",
                 total / (1024.0 * 1024.0), source.path);
    return true;
}

static void expert_pack_page_cache_preload_sources() {
    const char *mode = std::getenv("GGML_MOE_PAGECACHE_PRELOAD");
    if (!mode || !mode[0] || mode[0] == '0') return;
    for (expert_pack_source &source : g_expert_pack.sources) {
        if (!source.page_cache) continue;
        (void)expert_pack_page_cache_preload_source(source, mode);
    }
}

static bool q4_colibri_routed_tensor_name(const char *tensor_name) {
    return tensor_name && std::strstr(tensor_name, "blk.") == tensor_name &&
        (std::strstr(tensor_name, ".ffn_gate_exps.weight") ||
         std::strstr(tensor_name, ".ffn_up_exps.weight") ||
         std::strstr(tensor_name, ".ffn_down_exps.weight"));
}

[[noreturn]] static void q4_colibri_fail(const std::string &message) {
    std::fprintf(stderr, "[moe_stream_batch] q4 colibri pack fatal: %s\n", message.c_str());
    std::fflush(stderr);
    std::abort();
}







static q4_lossless_transport_mode q4_lossless_transport_mode_from_env(
        const char *sidecar_path) {
    const char *mode = std::getenv("GGML_MOE_LOSSLESS_MODE");
    if (!mode || !mode[0]) {
        mode = std::getenv("GGML_MOE_GLM52_LOSSLESS_MODE");
    }
    if (!mode || !mode[0]) {
        return sidecar_path && sidecar_path[0] ?
            q4_lossless_transport_mode::verify :
            q4_lossless_transport_mode::disabled;
    }
    if (std::strcmp(mode, "0") == 0 || std::strcmp(mode, "off") == 0 ||
            std::strcmp(mode, "disabled") == 0) {
        return q4_lossless_transport_mode::disabled;
    }
    if (std::strcmp(mode, "verify") == 0) {
        return q4_lossless_transport_mode::verify;
    }
    if (std::strcmp(mode, "cuda") == 0) {
        return q4_lossless_transport_mode::cuda;
    }
    q4_colibri_fail(std::string("invalid GGML_MOE_LOSSLESS_MODE: ") + mode);
}



static const char *q4_lossless_sidecar_path_from_env() {
    const char *path = std::getenv("GGML_MOE_LOSSLESS_PACK");
    if (path && path[0]) return path;
    return std::getenv("GGML_MOE_GLM52_LOSSLESS_PACK");
}













#if defined(GGML_SSD_STORE)
static void expert_pack_initialize_legacy() {
#else
static void expert_pack_init_once() {
#endif
    std::lock_guard<std::mutex> lk(g_expert_pack.mu);
    if (g_expert_pack.inited) return;

    for (const char * name : {"GGML_MOE_Q4_COLIBRI_PACK", "GGML_MOE_Q4_COLIBRI_MANIFEST", "GGML_MOE_GLM52_COLIBRI_PACK", "GGML_MOE_GLM52_COLIBRI_MANIFEST"}) {
        const char * value = std::getenv(name);
        if (value && value[0]) {
            std::fprintf(stderr, "[moe_stream_batch] retired source family: %s\n", name);
            g_expert_pack.inited = true;
            return;
        }
    }
    const char * lossless_pack = q4_lossless_sidecar_path_from_env();
    if (q4_lossless_transport_mode_from_env(lossless_pack) !=
            q4_lossless_transport_mode::disabled) {
        std::fprintf(stderr, "[moe_stream_batch] retired source family: active lossless transport\n");
        g_expert_pack.inited = true;
        return;
    }

    const char *io_backend_env = std::getenv("GGML_MOE_IO_BACKEND");
    if (io_backend_env && io_backend_env[0]) {
        if (std::strcmp(io_backend_env, "direct") == 0) {
            g_expert_pack.io_backend = 1;
        } else if (std::strcmp(io_backend_env, "iouring") == 0 ||
                   std::strcmp(io_backend_env, "io_uring") == 0) {
            g_expert_pack.io_backend = 2;
        } else if (std::strcmp(io_backend_env, "mmap") == 0) {
            g_expert_pack.io_backend = 0;
        } else if (!g_expert_pack.reported_io_backend) {
            std::fprintf(stderr,
                "[moe_stream_batch] GGML_MOE_IO_BACKEND=%s requested; runtime supports mmap/buffered, direct, and iouring expert-pack reads\n",
                io_backend_env);
            g_expert_pack.reported_io_backend = true;
        }
    }

#if defined(GGML_SSD_STORE)
    const auto source_desc = c0::source_description_from_env();
    const auto & source_paths = source_desc.packs;
    const auto & page_cache_paths = source_desc.page_cache_packs;
#else
    std::vector<std::string> source_paths;
    const char *path = std::getenv("GGML_MOE_EXPERT_PACK");
    if (path && path[0]) {
        source_paths.emplace_back(path);
        const char *overlay_path = std::getenv("GGML_MOE_EXPERT_PACK_OVERLAY");
        if (overlay_path && overlay_path[0]) {
            source_paths.emplace_back(overlay_path);
        }
        const char *overlay_extra_path = std::getenv("GGML_MOE_EXPERT_PACK_OVERLAY_EXTRA");
        if (overlay_extra_path && overlay_extra_path[0]) {
            source_paths.emplace_back(overlay_extra_path);
        }
    }
    expert_pack_append_env_list(source_paths, "GGML_MOE_EXPERT_PACK_LIST");
    std::vector<std::string> page_cache_paths;
    const char *page_cache_path = std::getenv("GGML_MOE_PAGECACHE_PACK");
    if (page_cache_path && page_cache_path[0]) {
        page_cache_paths.emplace_back(page_cache_path);
    }
    expert_pack_append_env_list(page_cache_paths, "GGML_MOE_PAGECACHE_PACK_LIST");
#endif
    const char *gguf_alias_tsv = std::getenv("GGML_MOE_EXPERT_GGUF_ALIAS_TSV");
    const bool has_gguf_alias = gguf_alias_tsv && gguf_alias_tsv[0];

    if (source_paths.empty() && page_cache_paths.empty() && !has_gguf_alias) {
        g_expert_pack.inited = true;
        return;
    }

    std::vector<expert_pack_entry> entries;
#if defined(GGML_SSD_STORE)
    if (!c0::append_sources(c0::shared_catalog(), source_desc, entries, expert_pack_source_open_hooks())) {
        g_expert_pack.inited = true;
        return;
    }
#else
    for (const std::string &source_path : source_paths) {
        if (!expert_pack_load_source(source_path.c_str(), (int32_t)g_expert_pack.sources.size(), entries, false)) {
            g_expert_pack.inited = true;
            return;
        }
    }
    if (has_gguf_alias) {
        std::vector<std::string> alias_paths;
        expert_pack_append_env_list(alias_paths, "GGML_MOE_EXPERT_GGUF_ALIAS_TSV");
        std::unordered_map<std::string, int32_t> alias_sources;
        for (const std::string &alias_path : alias_paths) {
            if (!expert_pack_load_gguf_alias_tsv(alias_path.c_str(), entries, alias_sources)) {
                g_expert_pack.inited = true;
                return;
            }
        }
    }
    for (const std::string &source_path : page_cache_paths) {
        if (!expert_pack_load_source(source_path.c_str(), (int32_t)g_expert_pack.sources.size(), entries, true)) {
            g_expert_pack.inited = true;
            return;
        }
    }

#endif


    if (entries.empty()) {
        g_expert_pack.inited = true;
        return;
    }

#if defined(GGML_SSD_STORE)
    const bool replace_duplicates = source_desc.replace_duplicates ||
        !page_cache_paths.empty();
    if (!c0::finalize_entries(entries, replace_duplicates)) {
        g_expert_pack.inited = true;
        return;
    }
#else
    std::sort(entries.begin(), entries.end(),
        [](const expert_pack_entry &a, const expert_pack_entry &b) {
            const int name_cmp = std::strcmp(a.tensor, b.tensor);
            if (name_cmp != 0) return name_cmp < 0;
            if (a.expert_idx != b.expert_idx) return a.expert_idx < b.expert_idx;
            if (a.nbytes != b.nbytes) return a.nbytes < b.nbytes;
            return a.source_idx < b.source_idx;
        });
    const bool replace_duplicates =
        expert_pack_env_bool("GGML_MOE_EXPERT_PACK_REPLACE_DUPLICATES", false) ||
        !page_cache_paths.empty();
    uint64_t replaced_duplicates = 0;
    if (replace_duplicates && !entries.empty()) {
        std::vector<expert_pack_entry> deduped;
        deduped.reserve(entries.size());
        for (const expert_pack_entry &entry : entries) {
            if (!deduped.empty() &&
                    expert_pack_entry_cmp(deduped.back(), entry.tensor, entry.expert_idx, entry.nbytes) == 0) {
                if (entry.source_idx >= deduped.back().source_idx) {
                    deduped.back() = entry;
                }
                ++replaced_duplicates;
            } else {
                deduped.push_back(entry);
            }
        }
        entries = std::move(deduped);
        if (replaced_duplicates > 0) {
            std::fprintf(stderr,
                "[moe_stream_batch] expert pack: replaced %lu duplicate keys with later pack sources\n",
                (unsigned long)replaced_duplicates);
        }
    }
    for (size_t i = 1; i < entries.size(); ++i) {
        if (expert_pack_entry_cmp(entries[i - 1], entries[i].tensor, entries[i].expert_idx, entries[i].nbytes) == 0) {
            std::fprintf(stderr, "[moe_stream_batch] expert pack: duplicate key across packs: %s expert=%d bytes=%lu\n",
                         entries[i].tensor, entries[i].expert_idx, (unsigned long)entries[i].nbytes);
            g_expert_pack.inited = true;
            return;
        }
    }

#endif

    g_expert_pack.file = g_expert_pack.sources.empty() ? nullptr : g_expert_pack.sources[0].file;
#if !defined(_WIN32)
    g_expert_pack.fd_direct = g_expert_pack.sources.empty() ? -1 : g_expert_pack.sources[0].fd_direct;
#endif
    g_expert_pack.entries = std::move(entries);
    g_expert_pack.enabled = true;
    g_expert_pack.inited = true;
    expert_pack_page_cache_preload_sources();
    std::atexit(expert_pack_report_atexit);
    std::fprintf(stderr, "[moe_stream_batch] expert pack: total entries=%zu sources=%zu\n",
                 g_expert_pack.entries.size(), g_expert_pack.sources.size());
}

#if defined(GGML_SSD_STORE)
static void expert_pack_init_once() {
    (void)c0::initialize_shared_catalog();
}
#endif

static const expert_pack_entry * expert_pack_lookup_impl(const char *tensor_name, int expert_idx, size_t nbytes, bool count_stats) {
    expert_pack_init_once();
    const char *disable_env = std::getenv("GGML_MOE_EXPERT_PACK_RUNTIME_DISABLE");
    if (disable_env && disable_env[0] && disable_env[0] != '0') return nullptr;
    if (!expert_pack_catalog_ready() || !tensor_name || !tensor_name[0]) return nullptr;

    size_t lo = 0;
    size_t hi = g_expert_pack.entries.size();
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const int cmp = expert_pack_entry_cmp(g_expert_pack.entries[mid], tensor_name, expert_idx, nbytes);
        if (cmp < 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < g_expert_pack.entries.size() &&
            expert_pack_entry_cmp(g_expert_pack.entries[lo], tensor_name, expert_idx, nbytes) == 0) {
        if (count_stats) {
            ++g_expert_pack.hits;
        }
        return &g_expert_pack.entries[lo];
    }
    if (count_stats) {
        ++g_expert_pack.misses;
    }
    if (g_expert_pack.q4_colibri && q4_colibri_routed_tensor_name(tensor_name)) {
        if (g_expert_pack.q4_colibri_debug_partial_fallback) {
            ++g_expert_pack.q4_colibri_partial_fallback_lookups;
        } else {
            q4_colibri_fail(std::string("missing object key tensor=") + tensor_name +
                " expert=" + std::to_string(expert_idx) + " bytes=" + std::to_string(nbytes));
        }
    }
    return nullptr;
}

#if defined(GGML_SSD_STORE)
static const expert_pack_entry * c0_provider_lookup(const char * name, int expert, size_t bytes) {
    return expert_pack_lookup_impl(name, expert, bytes, false);
}
static const c0::provider c0_cuda_provider = {expert_pack_initialize_legacy, c0_provider_lookup};
struct c0_provider_registration {
    c0_provider_registration() {
        if (!c0::register_provider(&c0_cuda_provider)) {
            std::fprintf(stderr, "[c0_store_prototype] duplicate or late CUDA provider registration\n");
            std::abort();
        }
    }
    ~c0_provider_registration() {
        // Backend unload requires all readers to have stopped. Retire its
        // callbacks before this module disappears; never fall back to a new owner.
        if (!c0::unregister_provider(&c0_cuda_provider)) { std::abort(); }
    }
};
static c0_provider_registration c0_cuda_registration;
#endif

static const expert_pack_entry * expert_pack_lookup(const char *tensor_name, int expert_idx, size_t nbytes) {
    return expert_pack_lookup_impl(tensor_name, expert_idx, nbytes, true);
}

static const expert_pack_entry * expert_pack_lookup_any_size(const char *tensor_name, int expert_idx) {
    expert_pack_init_once();
    const char *disable_env = std::getenv("GGML_MOE_EXPERT_PACK_RUNTIME_DISABLE");
    if (disable_env && disable_env[0] && disable_env[0] != '0') return nullptr;
    if (!expert_pack_catalog_ready() || !tensor_name || !tensor_name[0]) return nullptr;

    expert_pack_entry probe = {};
    std::snprintf(probe.tensor, sizeof(probe.tensor), "%s", tensor_name);
    probe.expert_idx = expert_idx;
    probe.nbytes = 0;
    auto it = std::lower_bound(
            g_expert_pack.entries.begin(),
            g_expert_pack.entries.end(),
            probe,
            [](const expert_pack_entry &a, const expert_pack_entry &b) {
                const int name_cmp = std::strcmp(a.tensor, b.tensor);
                if (name_cmp != 0) return name_cmp < 0;
                if (a.expert_idx != b.expert_idx) return a.expert_idx < b.expert_idx;
                return a.nbytes < b.nbytes;
            });
    if (it != g_expert_pack.entries.end() &&
            std::strcmp(it->tensor, tensor_name) == 0 &&
            it->expert_idx == expert_idx) {
        return &*it;
    }
    return nullptr;
}




static expert_pack_source * expert_pack_mmap_ensure(const expert_pack_entry * entry) {
    expert_pack_init_once();
    expert_pack_source * source = expert_pack_source_for_entry(entry);
    if (!expert_pack_catalog_ready() || !source) {
        return nullptr;
    }
    if (source->mmap_base) {
        return source;
    }
    if (source->mmap_attempted) {
        return nullptr;
    }
    source->mmap_attempted = true;
#if !defined(_WIN32)
    const char *env = std::getenv("GGML_MOE_CPU_FALLBACK_PACK_MMAP");
    if (!source->page_cache && (!env || !env[0] || env[0] == '0')) {
        return nullptr;
    }
    if (!source->file) {
        return nullptr;
    }
    const int fd = fileno(source->file);
    if (fd < 0) {
        return nullptr;
    }
    struct stat st = {};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        return nullptr;
    }
    void *base = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        std::fprintf(stderr, "[moe_stream_batch] expert pack mmap: mmap failed: %s\n", source->path);
        return nullptr;
    }
    source->mmap_base = base;
    source->mmap_size = (size_t) st.st_size;
    source->mmap_enabled = true;
    std::fprintf(stderr, "[moe_stream_batch] expert pack mmap: enabled source=%d size=%.2f MiB path=%s\n",
                 entry ? entry->source_idx : -1,
                 source->mmap_size / (1024.0 * 1024.0),
                 source->path);
    return source;
#else
    return nullptr;
#endif
}

extern "C" const void * ggml_cuda_moe_expert_pack_mmap_ptr(const char *tensor_name, int expert_idx, size_t nbytes) {
    if (!tensor_name || !tensor_name[0] || nbytes == 0) {
        return nullptr;
    }
    const expert_pack_entry *entry = expert_pack_lookup(tensor_name, expert_idx, nbytes);
    if (!entry) {
        ++g_expert_pack.mmap_misses;
        return nullptr;
    }
    expert_pack_source * source = expert_pack_mmap_ensure(entry);
    if (!source) {
        return nullptr;
    }
    if (entry->offset > source->mmap_size || entry->nbytes != nbytes || entry->offset + entry->nbytes > source->mmap_size) {
        ++g_expert_pack.mmap_misses;
        return nullptr;
    }
    ++g_expert_pack.mmap_hits;
    g_expert_pack.mmap_bytes.fetch_add(nbytes);
    return (const char *) source->mmap_base + entry->offset;
}

extern "C" const void * ggml_cuda_moe_expert_pack_mmap_ptr_debug(
        const char * tensor_name,
        int expert_idx,
        size_t nbytes,
        const char ** reason,
        size_t * entry_nbytes,
        uint64_t * entry_offset) {
    if (reason) {
        *reason = "ok";
    }
    if (entry_nbytes) {
        *entry_nbytes = 0;
    }
    if (entry_offset) {
        *entry_offset = 0;
    }
    if (!tensor_name || !tensor_name[0] || nbytes == 0) {
        if (reason) {
            *reason = "invalid_args";
        }
        return nullptr;
    }
    expert_pack_init_once();
    if (!expert_pack_catalog_ready()) {
        if (reason) {
            *reason = "mmap_unavailable";
        }
        return nullptr;
    }

    const expert_pack_entry * entry = expert_pack_lookup_impl(tensor_name, expert_idx, nbytes, false);
    if (!entry) {
        ++g_expert_pack.mmap_misses;
        const expert_pack_entry * any_size = expert_pack_lookup_any_size(tensor_name, expert_idx);
        if (any_size) {
            if (reason) {
                *reason = "nbytes_mismatch";
            }
            if (entry_nbytes) {
                *entry_nbytes = (size_t) any_size->nbytes;
            }
            if (entry_offset) {
                *entry_offset = any_size->offset;
            }
        } else if (reason) {
            *reason = "entry_missing";
        }
        return nullptr;
    }
    expert_pack_source * source = expert_pack_mmap_ensure(entry);
    if (!source) {
        if (reason) {
            *reason = "mmap_unavailable";
        }
        return nullptr;
    }
    if (entry_nbytes) {
        *entry_nbytes = (size_t) entry->nbytes;
    }
    if (entry_offset) {
        *entry_offset = entry->offset;
    }
    if (entry->offset > source->mmap_size || entry->nbytes != nbytes || entry->offset + entry->nbytes > source->mmap_size) {
        ++g_expert_pack.mmap_misses;
        if (reason) {
            *reason = "range_invalid";
        }
        return nullptr;
    }
    ++g_expert_pack.mmap_hits;
    g_expert_pack.mmap_bytes.fetch_add(nbytes);
    return (const char *) source->mmap_base + entry->offset;
}

extern "C" const void * ggml_cuda_moe_expert_pack_mmap_ptr_source_debug(
        const char * tensor_name,
        int expert_idx,
        size_t nbytes,
        const char ** reason,
        size_t * entry_nbytes,
        uint64_t * entry_offset,
        int * source_idx,
        const char ** source_path,
        int * source_is_gguf) {
    if (source_idx) {
        *source_idx = -1;
    }
    if (source_path) {
        *source_path = nullptr;
    }
    if (source_is_gguf) {
        *source_is_gguf = 0;
    }
    if (reason) {
        *reason = "ok";
    }
    if (entry_nbytes) {
        *entry_nbytes = 0;
    }
    if (entry_offset) {
        *entry_offset = 0;
    }
    if (!tensor_name || !tensor_name[0] || nbytes == 0) {
        if (reason) {
            *reason = "invalid_args";
        }
        return nullptr;
    }
    expert_pack_init_once();
    if (!expert_pack_catalog_ready()) {
        if (reason) {
            *reason = "mmap_unavailable";
        }
        return nullptr;
    }

    const expert_pack_entry * entry = expert_pack_lookup_impl(tensor_name, expert_idx, nbytes, false);
    if (!entry) {
        ++g_expert_pack.mmap_misses;
        const expert_pack_entry * any_size = expert_pack_lookup_any_size(tensor_name, expert_idx);
        if (any_size) {
            if (reason) {
                *reason = "nbytes_mismatch";
            }
            if (entry_nbytes) {
                *entry_nbytes = (size_t) any_size->nbytes;
            }
            if (entry_offset) {
                *entry_offset = any_size->offset;
            }
        } else if (reason) {
            *reason = "entry_missing";
        }
        return nullptr;
    }
    expert_pack_source * source = expert_pack_mmap_ensure(entry);
    if (!source) {
        if (reason) {
            *reason = "mmap_unavailable";
        }
        return nullptr;
    }
    if (entry_nbytes) {
        *entry_nbytes = (size_t) entry->nbytes;
    }
    if (entry_offset) {
        *entry_offset = entry->offset;
    }
    if (source_idx) {
        *source_idx = entry->source_idx;
    }
    if (source_path) {
        *source_path = source->path;
    }
    if (source_is_gguf) {
        *source_is_gguf = std::strstr(source->path, ".gguf") != nullptr ? 1 : 0;
    }
    if (entry->offset > source->mmap_size || entry->nbytes != nbytes || entry->offset + entry->nbytes > source->mmap_size) {
        ++g_expert_pack.mmap_misses;
        if (reason) {
            *reason = "range_invalid";
        }
        return nullptr;
    }
    ++g_expert_pack.mmap_hits;
    g_expert_pack.mmap_bytes.fetch_add(nbytes);
    return (const char *) source->mmap_base + entry->offset;
}


// O_DIRECT may return EOF before unused page padding. Only a complete logical
// payload is accepted; an incomplete tensor remains an error.
static bool expert_pack_direct_pread_native(int fd, void * dst, size_t physical,
        size_t required, uint64_t offset) {
#if !defined(_WIN32)
    size_t done = 0;
    while (done < physical) {
        const size_t chunk = std::min(physical - done, size_t(64) * 1024 * 1024);
        ssize_t got;
        do { got = pread(fd, static_cast<char *>(dst) + done, chunk, off_t(offset + done)); }
        while (got < 0 && errno == EINTR);
        if (got <= 0) { return false; }
        done += size_t(got);
        if (done >= required) { return true; }
        if (size_t(got) != chunk) { return false; }
    }
    return done >= required;
#else
    (void) fd; (void) dst; (void) physical; (void) required; (void) offset;
    return false;
#endif
}

static bool expert_pack_direct_read_entry_to_host(const expert_pack_entry *entry, void *dst, size_t sz);

static uint64_t host_prefetch_key(const char *tensor_name, int expert_idx, size_t nbytes) {
    uint64_t h = 1469598103934665603ULL;
    if (tensor_name) {
        for (const unsigned char *p = (const unsigned char *)tensor_name; *p; ++p) {
            h ^= (uint64_t)(*p);
            h *= 1099511628211ULL;
        }
    }
    h ^= (uint64_t)(uint32_t)expert_idx;
    h *= 1099511628211ULL;
    h ^= (uint64_t)nbytes;
    h *= 1099511628211ULL;
    return h;
}

static void host_prefetch_report_atexit() {
    std::lock_guard<std::mutex> lk(g_host_prefetch.mu);
    if (!g_host_prefetch.enabled) return;
    std::fprintf(stderr,
        "[moe_stream_batch] host prefetch: calls=%lu matched=%lu resync=%lu submitted=%lu hits=%lu misses=%lu "
        "empty_misses=%lu ready=%lu evicted=%lu read_failures=%lu alloc_failures=%lu duplicate_skips=%lu profile_pinned_skips=%lu reserved_skips=%lu no_slot=%lu "
        "planned_enqueued=%lu planned_dequeued=%lu planned_duplicate_skips=%lu scan_passes=%lu "
        "cursor=%zu produce_cursor=%zu/%zu skip=%zu used=%.2f MiB slots=%zu\n",
        g_host_prefetch.calls,
        g_host_prefetch.matched,
        g_host_prefetch.resync,
        g_host_prefetch.submitted,
        g_host_prefetch.hits,
        g_host_prefetch.misses,
        (unsigned long)g_host_prefetch.empty_misses.load(),
        (unsigned long)g_host_prefetch.ready_count.load(),
        g_host_prefetch.evicted,
        g_host_prefetch.read_failures,
        g_host_prefetch.alloc_failures,
        g_host_prefetch.duplicate_skips,
        g_host_prefetch.profile_pinned_skips,
        g_host_prefetch.reserved_skips,
        g_host_prefetch.no_slot,
        g_host_prefetch.planned_enqueued,
        g_host_prefetch.planned_dequeued,
        g_host_prefetch.planned_duplicate_skips,
        g_host_prefetch.scan_passes,
        g_host_prefetch.cursor,
        g_host_prefetch.produce_cursor,
        g_host_prefetch.trace.size(),
        g_host_prefetch.skip_events,
        g_host_prefetch.used_bytes / (1024.0 * 1024.0),
        g_host_prefetch.slots.size());
}

static int host_prefetch_env_int(const char *name, int fallback, int lo, int hi) {
    const char *env = std::getenv(name);
    long value = (env && env[0]) ? std::atol(env) : fallback;
    if (value < lo) value = lo;
    if (value > hi) value = hi;
    return (int)value;
}

static size_t host_prefetch_env_mib(const char *name, size_t fallback_mib, size_t lo_mib, size_t hi_mib) {
    const char *env = std::getenv(name);
    size_t value = (env && env[0]) ? (size_t)std::strtoull(env, nullptr, 10) : fallback_mib;
    if (value < lo_mib) value = lo_mib;
    if (value > hi_mib) value = hi_mib;
    return value * 1024ULL * 1024ULL;
}

static bool predictive_host_prefetch_requested() {
    const char *env = std::getenv("GGML_MOE_PREDICTIVE_HOST_PREFETCH");
    return env && env[0] && env[0] != '0';
}

static bool host_prefetch_slot_alloc(
        ggml_moe_common::object_stage_slot &slot,
        size_t nbytes) {
    const size_t alloc_sz = (size_t)align_up_u64((uint64_t)nbytes, (uint64_t)expert_pack_direct_alignment());
    return ggml_moe_common::object_stage_slot_ensure_capacity(slot, alloc_sz, false);
}

static int host_prefetch_find_free_slot_locked(size_t nbytes) {
    int best = -1;
    for (size_t i = 0; i < g_host_prefetch.slots.size(); ++i) {
        const host_prefetch_slot &slot = g_host_prefetch.slots[i];
        if (slot.in_use) continue;
        if (slot.reserved) continue;
        if (!slot.ready) {
            best = (int)i;
            break;
        }
        if (best < 0) best = (int)i;
    }
    if (best >= 0 && g_host_prefetch.slots[(size_t)best].ready) {
        host_prefetch_slot &slot = g_host_prefetch.slots[(size_t)best];
        g_host_prefetch.ready.erase(host_prefetch_key(slot.tensor, slot.expert_idx, slot.nbytes));
        slot.ready = false;
        slot.reserved = false;
        slot.entry = nullptr;
        uint64_t ready = g_host_prefetch.ready_count.load(std::memory_order_relaxed);
        while (ready > 0 && !g_host_prefetch.ready_count.compare_exchange_weak(
                ready, ready - 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
        }
        ++g_host_prefetch.evicted;
    }
    const size_t alloc_sz = (size_t)align_up_u64((uint64_t)nbytes, (uint64_t)expert_pack_direct_alignment());
    const size_t cur_capacity = best >= 0 ?
        g_host_prefetch.stage_ring.slots[(size_t)best].host_capacity : 0;
    const size_t extra = alloc_sz > cur_capacity ? alloc_sz - cur_capacity : 0;
    if (best >= 0 && g_host_prefetch.used_bytes + extra > g_host_prefetch.max_bytes) {
        return -1;
    }
    return best;
}

static void host_prefetch_worker() {
    while (true) {
        batch_route_trace_entry e;
        size_t slot_idx = SIZE_MAX;
        const expert_pack_entry *entry = nullptr;
        {
            std::unique_lock<std::mutex> lk(g_host_prefetch.mu);
            g_host_prefetch.cv.wait(lk, [] {
                return g_host_prefetch.stop ||
                    (g_host_prefetch.enabled &&
                     (!g_host_prefetch.planned.empty() ||
                      g_host_prefetch.cursor < g_host_prefetch.trace.size()));
            });
            if (g_host_prefetch.stop) break;

            ++g_host_prefetch.scan_passes;
            if (!g_host_prefetch.planned.empty()) {
                e = g_host_prefetch.planned.front();
                g_host_prefetch.planned.pop_front();
                const uint64_t key = host_prefetch_key(e.tensor, e.expert_idx, e.expert_bytes);
                g_host_prefetch.planned_queued.erase(key);
                ++g_host_prefetch.planned_dequeued;
            } else {
                const size_t start = std::max(g_host_prefetch.cursor, g_host_prefetch.skip_events);
                const size_t end = std::min(g_host_prefetch.trace.size(), start + g_host_prefetch.lead_events);
                size_t selected = SIZE_MAX;
                for (size_t i = start; i < end; ++i) {
                    const batch_route_trace_entry &candidate = g_host_prefetch.trace[i];
                    const uint64_t key = host_prefetch_key(candidate.tensor, candidate.expert_idx, candidate.expert_bytes);
                    if (profile_pinned_key_contains(batch_key_hash(candidate.tensor, candidate.expert_idx))) {
                        ++g_host_prefetch.profile_pinned_skips;
                        continue;
                    }
                    auto it = g_host_prefetch.ready.find(key);
                    if (it != g_host_prefetch.ready.end()) {
                        const host_prefetch_slot &slot = g_host_prefetch.slots[it->second];
                        if (slot.ready) {
                            ++g_host_prefetch.duplicate_skips;
                        } else {
                            ++g_host_prefetch.reserved_skips;
                        }
                        continue;
                    }
                    selected = i;
                    break;
                }
                if (selected == SIZE_MAX) {
                    g_host_prefetch.cv.wait_for(lk, std::chrono::milliseconds(1));
                    continue;
                }

                e = g_host_prefetch.trace[selected];
                g_host_prefetch.produce_cursor = selected + 1;
            }
            const uint64_t key = host_prefetch_key(e.tensor, e.expert_idx, e.expert_bytes);
            entry = expert_pack_lookup(e.tensor, e.expert_idx, e.expert_bytes);
            if (!entry) {
                ++g_host_prefetch.read_failures;
                continue;
            }
            const int free_slot = host_prefetch_find_free_slot_locked(e.expert_bytes);
            if (free_slot < 0) {
                ++g_host_prefetch.no_slot;
                continue;
            }
            slot_idx = (size_t)free_slot;
            host_prefetch_slot &slot = g_host_prefetch.slots[slot_idx];
            slot.in_use = true;
            slot.reserved = true;
            slot.ready = false;
            slot.nbytes = e.expert_bytes;
            slot.expert_idx = e.expert_idx;
            std::snprintf(slot.tensor, sizeof(slot.tensor), "%s", e.tensor);
            g_host_prefetch.ready[key] = slot_idx;
        }

        bool ok = false;
        host_prefetch_slot *slot_ptr = &g_host_prefetch.slots[slot_idx];
        ggml_moe_common::object_stage_slot &stage = g_host_prefetch.stage_ring.slots[slot_idx];
        const size_t old_capacity = stage.host_capacity;
        if (ggml_moe_common::object_stage_slot_synchronize(g_host_prefetch.stage_ring, stage) &&
                host_prefetch_slot_alloc(stage, e.expert_bytes)) {
            const expert_pack_source *source = expert_pack_source_for_entry(entry);
            if (source && source->fd_direct >= 0) {
                const bool profile_direct = direct_read_site_profile_path() != nullptr;
                const auto direct_start = profile_direct ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                ok = expert_pack_direct_read_entry_to_host(entry, stage.host, e.expert_bytes);
                if (profile_direct) {
                    const auto direct_end = std::chrono::steady_clock::now();
                    direct_read_site_profile_record(
                            "host_prefetch", e.tensor, e.expert_bytes, ok,
                            std::chrono::duration<double, std::milli>(direct_end - direct_start).count());
                }
            } else if (source && source->file) {
                std::lock_guard<std::mutex> lk(g_expert_pack.mu);
                ok = ::fseeko(source->file, (off_t)entry->offset, SEEK_SET) == 0 &&
                    expert_pack_read_exact(source->file, stage.host, e.expert_bytes);
            }
        } else {
            std::lock_guard<std::mutex> lk(g_host_prefetch.mu);
            ++g_host_prefetch.alloc_failures;
        }

        {
            std::lock_guard<std::mutex> lk(g_host_prefetch.mu);
            host_prefetch_slot &slot = g_host_prefetch.slots[slot_idx];
            if (stage.host_capacity > old_capacity) {
                g_host_prefetch.used_bytes += stage.host_capacity - old_capacity;
            } else if (stage.host_capacity < old_capacity) {
                const size_t delta = old_capacity - stage.host_capacity;
                g_host_prefetch.used_bytes = delta < g_host_prefetch.used_bytes ? g_host_prefetch.used_bytes - delta : 0;
            }
            if (ok) {
                slot.entry = entry;
                slot.nbytes = e.expert_bytes;
                slot.expert_idx = e.expert_idx;
                std::snprintf(slot.tensor, sizeof(slot.tensor), "%s", e.tensor);
                slot.reserved = false;
                slot.ready = true;
                g_host_prefetch.ready[host_prefetch_key(slot.tensor, slot.expert_idx, slot.nbytes)] = slot_idx;
                g_host_prefetch.ready_count.fetch_add(1, std::memory_order_release);
                ++g_host_prefetch.submitted;
            } else {
                slot.ready = false;
                slot.reserved = false;
                slot.entry = nullptr;
                g_host_prefetch.ready.erase(host_prefetch_key(e.tensor, e.expert_idx, e.expert_bytes));
                ++g_host_prefetch.read_failures;
            }
            slot.in_use = false;
        }
    }
}

static void host_prefetch_shutdown() {
    {
        std::lock_guard<std::mutex> lk(g_host_prefetch.mu);
        g_host_prefetch.stop = true;
        g_host_prefetch.cv.notify_all();
    }
    if (g_host_prefetch.worker.joinable()) {
        g_host_prefetch.worker.join();
    }
    ggml_moe_common::object_stage_ring_release(g_host_prefetch.stage_ring);
}

static void host_prefetch_init_once() {
    if (g_host_prefetch.inited) return;
    std::lock_guard<std::mutex> lk(g_host_prefetch.mu);
    if (g_host_prefetch.inited) return;
    const char *path = std::getenv("GGML_MOE_HOST_PREFETCH");
    const char *planned_env = std::getenv("GGML_MOE_PLANNED_HOST_PREFETCH");
    const bool predictive_env = predictive_host_prefetch_requested();
    g_host_prefetch.planned_enabled = planned_env && planned_env[0] && planned_env[0] != '0';
    if ((!path || !path[0]) && !g_host_prefetch.planned_enabled && !predictive_env) {
        g_host_prefetch.inited = true;
        return;
    }
    if (path && path[0] && !load_route_trace_csv(path, g_host_prefetch.trace, "host prefetch")) {
        g_host_prefetch.inited = true;
        return;
    }
    const int slots = host_prefetch_env_int("GGML_MOE_HOST_PREFETCH_SLOTS", 64, 1, 512);
    g_host_prefetch.lead_events = (size_t)host_prefetch_env_int("GGML_MOE_HOST_PREFETCH_LEAD_EVENTS", 2048, 1, 32768);
    g_host_prefetch.skip_events = (size_t)host_prefetch_env_int("GGML_MOE_HOST_PREFETCH_SKIP_EVENTS", 0, 0, 100000000);
    if (g_host_prefetch.skip_events > g_host_prefetch.trace.size()) {
        g_host_prefetch.skip_events = g_host_prefetch.trace.size();
    }
    g_host_prefetch.produce_cursor = g_host_prefetch.skip_events;
    g_host_prefetch.max_bytes = host_prefetch_env_mib("GGML_MOE_HOST_PREFETCH_MAX_MIB", 512, 16, 8192);
    g_host_prefetch.slots.resize((size_t)slots);
    g_host_prefetch.stage_ring.slots.resize((size_t)slots);
    g_host_prefetch.enabled = true;
    g_host_prefetch.inited = true;
    std::atexit(host_prefetch_report_atexit);
    std::atexit(host_prefetch_shutdown);
    g_host_prefetch.worker = std::thread(host_prefetch_worker);
    g_host_prefetch.cv.notify_all();
    std::fprintf(stderr,
        "[moe_stream_batch] host prefetch: loaded %zu events from %s lead_events=%zu skip_events=%zu slots=%d max=%.2f MiB planned=%d predictive=%d\n",
        g_host_prefetch.trace.size(), (path && path[0]) ? path : "(none)",
        g_host_prefetch.lead_events, g_host_prefetch.skip_events, slots,
        g_host_prefetch.max_bytes / (1024.0 * 1024.0),
        g_host_prefetch.planned_enabled ? 1 : 0,
        predictive_env ? 1 : 0);
}

















struct predictive_prefetch_candidate {
    int expert = -1;
    int count = 0;
    int recent = 0;
};









#include "moe-common-dynamic-tier.inc"

static bool host_prefetch_copy_h2d(
        const expert_pack_entry *pack_entry,
        const char *tensor_name,
        int expert_idx,
        void *dst,
        size_t sz,
        cudaStream_t st) {
    host_prefetch_init_once();
    if (!g_host_prefetch.enabled || !pack_entry || !tensor_name || !tensor_name[0]) return false;
    if (g_host_prefetch.ready_count.load(std::memory_order_acquire) == 0) {
        g_host_prefetch.empty_misses.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    void *host = nullptr;
    size_t slot_idx = SIZE_MAX;
    {
        std::lock_guard<std::mutex> lk(g_host_prefetch.mu);
        const uint64_t key = host_prefetch_key(tensor_name, expert_idx, sz);
        auto it = g_host_prefetch.ready.find(key);
        if (it == g_host_prefetch.ready.end()) {
            ++g_host_prefetch.misses;
            return false;
        }
        host_prefetch_slot &slot = g_host_prefetch.slots[it->second];
        ggml_moe_common::object_stage_slot &stage = g_host_prefetch.stage_ring.slots[it->second];
        if (!slot.ready || slot.nbytes != sz || !stage.host) {
            if (slot.reserved) {
                ++g_host_prefetch.reserved_skips;
            } else {
                g_host_prefetch.ready.erase(it);
            }
            ++g_host_prefetch.misses;
            return false;
        }
        slot_idx = it->second;
        host = stage.host;
        slot.in_use = true;
        slot.ready = false;
        slot.reserved = false;
        slot.entry = nullptr;
        g_host_prefetch.ready.erase(it);
        uint64_t ready = g_host_prefetch.ready_count.load(std::memory_order_relaxed);
        while (ready > 0 && !g_host_prefetch.ready_count.compare_exchange_weak(
                ready, ready - 1, std::memory_order_acquire, std::memory_order_relaxed)) {
        }
        ++g_host_prefetch.hits;
    }
    common_dynamic_tier_fill(pack_entry, host, sz);
    const bool ok = host && cudaMemcpyAsync(dst, host, sz, cudaMemcpyHostToDevice, st) == cudaSuccess;
    {
        std::lock_guard<std::mutex> lk(g_host_prefetch.mu);
        if (slot_idx < g_host_prefetch.slots.size()) {
            if (ok && !ggml_moe_common::object_stage_slot_record(
                    g_host_prefetch.stage_ring.slots[slot_idx], st)) {
                cudaGetLastError();
                cudaStreamSynchronize(st);
            }
            g_host_prefetch.slots[slot_idx].in_use = false;
        }
    }
    return ok;
}

static std::once_flag g_ram_tier_once;

static bool expert_pack_direct_read_entry_to_host(const expert_pack_entry *entry, void *dst, size_t sz);

static void ram_tier_init() {
    if (!expert_pack_catalog_ready()) return;
    const char *ram_tier_env = std::getenv("GGML_MOE_RAM_TIER_MIB");
    if (!ram_tier_env || !ram_tier_env[0] || ram_tier_env[0] == '0') return;
    const size_t budget = (size_t)std::atol(ram_tier_env) * 1024ULL * 1024ULL;
    if (budget == 0) return;

    if (common_dynamic_tier_requested()) {
        common_dynamic_tier_init(budget);
        return;
    }

    std::vector<profile_entry> ram_profile;
    const char *ram_profile_path = std::getenv("GGML_MOE_RAM_TIER_PROFILE");
    if (ram_profile_path && ram_profile_path[0]) {
        if (!load_profile_file(ram_profile_path, ram_profile, "RAM tier profile")) {
            std::fprintf(stderr, "[moe_stream_batch] RAM tier: no RAM tier profile loaded, skipping\n");
            return;
        }
    } else {
        load_profile_once();
        ram_profile = g_profile;
    }
    if (ram_profile.empty()) {
        std::fprintf(stderr, "[moe_stream_batch] RAM tier: no profile loaded, skipping\n");
        return;
    }

    // Skip the top entries that are already in the VRAM cache.
    // The VRAM cache preloads ~n_slots entries per tensor from the profile.
    // Default skip = 500 (approximate VRAM slot count); override with GGML_MOE_RAM_TIER_SKIP.
    size_t skip = 500;
    const char *skip_env = std::getenv("GGML_MOE_RAM_TIER_SKIP");
    if (skip_env && skip_env[0]) skip = (size_t)std::atol(skip_env);

    g_expert_pack.ram_tier_index.resize(g_expert_pack.entries.size());
    void *region = ::mmap(nullptr, budget, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (region == MAP_FAILED) {
        std::fprintf(stderr, "[moe_stream_batch] RAM tier: mmap %zu MiB failed\n", budget / (1024*1024));
        return;
    }
    g_expert_pack.ram_tier_base = region;
    g_expert_pack.ram_tier_bytes = budget;
    const char *pin_mib_env = std::getenv("GGML_MOE_RAM_TIER_PIN_MIB");
    const bool pin_budget_set = pin_mib_env && pin_mib_env[0];
    const size_t pin_budget = pin_budget_set ?
        (size_t)std::atol(pin_mib_env) * 1024ULL * 1024ULL : budget;

    size_t loaded = 0, n_loaded = 0, n_skipped = 0, pin_prefix_bytes = 0, pin_prefix_entries = 0;
    const bool direct_preload = expert_pack_env_bool("GGML_MOE_RAM_TIER_PRELOAD_DIRECT", false);
    int preload_threads = 1;
    const char *preload_threads_env = std::getenv("GGML_MOE_RAM_TIER_PRELOAD_THREADS");
    if (preload_threads_env && preload_threads_env[0]) {
        preload_threads = std::max(1, std::atoi(preload_threads_env));
    }

    struct ram_tier_load_job {
        size_t pack_index;
        size_t offset;
        const expert_pack_entry *entry;
        bool ok;
    };
    std::vector<ram_tier_load_job> parallel_jobs;

    for (const profile_entry &pe : ram_profile) {
        if (loaded >= budget) break;
        if (n_skipped < skip) { ++n_skipped; continue; }
        size_t lo2 = 0, hi2 = g_expert_pack.entries.size();
        while (lo2 < hi2) {
            size_t mid2 = lo2 + (hi2 - lo2) / 2;
            int cmp2 = expert_pack_entry_cmp(g_expert_pack.entries[mid2], pe.tensor, pe.expert_idx, pe.expert_bytes);
            if (cmp2 < 0) lo2 = mid2 + 1; else hi2 = mid2;
        }
        if (lo2 >= g_expert_pack.entries.size() ||
            expert_pack_entry_cmp(g_expert_pack.entries[lo2], pe.tensor, pe.expert_idx, pe.expert_bytes) != 0)
            continue;
        const expert_pack_entry &ent = g_expert_pack.entries[lo2];
        if (loaded + ent.nbytes > budget) break;
        char *dst_ptr = (char *)region + loaded;
        if (direct_preload && preload_threads > 1) {
            parallel_jobs.push_back({lo2, loaded, &ent, false});
            loaded += (size_t)ent.nbytes;
            continue;
        }
        {
            const expert_pack_source *source = expert_pack_source_for_entry(&ent);
            if (!source || !source->file) continue;
            bool loaded_direct = false;
            if (direct_preload) {
                loaded_direct = expert_pack_direct_read_entry_to_host(&ent, dst_ptr, (size_t)ent.nbytes);
                if (loaded_direct) {
                    g_expert_pack.ram_tier_direct_bytes.fetch_add((uint64_t)ent.nbytes);
                } else {
                    ++g_expert_pack.ram_tier_direct_fallbacks;
                }
            }
            if (!loaded_direct) {
                std::lock_guard<std::mutex> lk2(g_expert_pack.mu);
                if (::fseeko(source->file, (off_t)ent.offset, SEEK_SET) != 0) continue;
                if (!expert_pack_read_exact(source->file, dst_ptr, (size_t)ent.nbytes)) continue;
            }
        }
        g_expert_pack.ram_tier_index[lo2] = {loaded, (size_t)ent.nbytes};
        loaded += (size_t)ent.nbytes;
        ++n_loaded;
        if (pin_prefix_bytes + (size_t)ent.nbytes <= pin_budget) {
            pin_prefix_bytes += (size_t)ent.nbytes;
            ++pin_prefix_entries;
        }
    }

    if (!parallel_jobs.empty()) {
        std::atomic<size_t> next_job{0};
        const int n_threads = std::min<int>(preload_threads, (int)parallel_jobs.size());
        std::vector<std::thread> workers;
        workers.reserve((size_t)n_threads);
        for (int ti = 0; ti < n_threads; ++ti) {
            workers.emplace_back([&]() {
                for (;;) {
                    const size_t ji = next_job.fetch_add(1);
                    if (ji >= parallel_jobs.size()) break;
                    ram_tier_load_job &job = parallel_jobs[ji];
                    char *dst_ptr = (char *)region + job.offset;
                    const bool ok = expert_pack_direct_read_entry_to_host(job.entry, dst_ptr, (size_t)job.entry->nbytes);
                    job.ok = ok;
                    if (ok) {
                        g_expert_pack.ram_tier_direct_bytes.fetch_add((uint64_t)job.entry->nbytes);
                    } else {
                        ++g_expert_pack.ram_tier_direct_fallbacks;
                    }
                }
            });
        }
        for (std::thread &worker : workers) worker.join();
        for (const ram_tier_load_job &job : parallel_jobs) {
            if (!job.ok) continue;
            g_expert_pack.ram_tier_index[job.pack_index] = {job.offset, (size_t)job.entry->nbytes};
            ++n_loaded;
            if (pin_prefix_bytes + (size_t)job.entry->nbytes <= pin_budget) {
                pin_prefix_bytes += (size_t)job.entry->nbytes;
                ++pin_prefix_entries;
            }
        }
    }

    std::fprintf(stderr, "[moe_stream_batch] RAM tier: loaded %zu entries (skipped %zu), %.2f MiB into anonymous mmap\n",
                 n_loaded, skip, loaded / (1024.0 * 1024.0));
    if (loaded > 0 && expert_pack_env_bool("GGML_MOE_RAM_TIER_PIN", true)) {
        const size_t pin_bytes = std::min(loaded, pin_prefix_bytes);
        if (pin_bytes == 0) {
            std::fprintf(stderr, "[moe_stream_batch] RAM tier: cudaHostRegister prefix budget is 0, using unpinned path (%.2f MiB)\n",
                         loaded / (1024.0 * 1024.0));
            return;
        }
        cudaError_t err = cudaHostRegister(region, pin_bytes, cudaHostRegisterDefault);
        if (err == cudaSuccess) {
            g_expert_pack.ram_tier_pinned_bytes = pin_bytes;
            if (pin_budget_set) {
                std::fprintf(stderr,
                             "[moe_stream_batch] RAM tier: cudaHostRegister succeeded (%.2f MiB pinned prefix, %zu entries, %.2f MiB resident)\n",
                             pin_bytes / (1024.0 * 1024.0), pin_prefix_entries, loaded / (1024.0 * 1024.0));
            } else {
                std::fprintf(stderr, "[moe_stream_batch] RAM tier: cudaHostRegister succeeded (%.2f MiB pinned)\n",
                             pin_bytes / (1024.0 * 1024.0));
            }
        } else {
            std::fprintf(stderr, "[moe_stream_batch] RAM tier: cudaHostRegister failed (%s), using unpinned path\n",
                         cudaGetErrorString(err));
        }
    } else if (loaded > 0) {
        std::fprintf(stderr, "[moe_stream_batch] RAM tier: cudaHostRegister disabled, using unpinned path (%.2f MiB)\n",
                     loaded / (1024.0 * 1024.0));
    }
}

static bool expert_pack_read_entry_iouring(const expert_pack_entry *entry, void *dst, size_t sz) {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING) && !defined(_WIN32)
    expert_pack_source *source = expert_pack_source_for_entry(entry);
    if (!entry || !source || source->fd_direct < 0 || entry->nbytes != sz) return false;
    const uint64_t alignment = expert_pack_direct_alignment();
    const bool aligned_alias = expert_pack_env_bool("GGML_MOE_IO_ALIGNED_ALIAS_BATCH", false) &&
        entry->alias_source && (entry->offset % alignment) != 0;
    const size_t payload_shift = (size_t)(entry->offset % alignment);
    const uint64_t read_offset = aligned_alias ? entry->offset - payload_shift : entry->offset;
    const size_t read_sz = aligned_alias ?
        (size_t)align_up_u64((uint64_t)payload_shift + (uint64_t)sz, alignment) :
        (size_t)align_up_u64((uint64_t)sz, alignment);
    void *read_dst = dst;
    if (aligned_alias) {
        if (posix_memalign(&read_dst, (size_t)alignment, read_sz) != 0 || !read_dst) {
            ++g_expert_pack.io_stats.fallbacks;
            return false;
        }
    } else if ((entry->offset % alignment) != 0 ||
            ((uintptr_t)dst % alignment) != 0 ||
            read_sz < sz) {
        ++g_expert_pack.io_stats.fallbacks;
        return false;
    }

    pinned_stage_ring ring;
    const unsigned int flags = expert_pack_env_bool("GGML_MOE_IO_SQPOLL", false) ?
        ggml_moe_common::object_io_sqpoll_setup_flag() : 0;
    if (!ggml_moe_common::object_io_ring_ensure(ring, 1, flags, nullptr)) {
        if (read_dst != dst) std::free(read_dst);
        ++g_expert_pack.io_stats.fallbacks;
        return false;
    }

    ggml_moe_common::object_io_window_slot io_window[1] = {};
    const bool common_object_cqe = common_object_cqe_enabled();
    bool read_prepared = false;
    if (common_object_cqe) {
        const ggml_moe_common::object_io_window_request request{
            source->fd_direct, read_dst, read_sz, read_offset, 0, false,
        };
        read_prepared = ggml_moe_common::object_io_window_prepare_read(
            ring, io_window, 1, 0, request);
    } else {
        read_prepared = ggml_moe_common::object_io_ring_prepare_read(
            ring, source->fd_direct, read_dst, read_sz, read_offset, 1);
    }
    if (!read_prepared) {
        ggml_moe_common::object_io_ring_release(ring);
        if (read_dst != dst) std::free(read_dst);
        ++g_expert_pack.io_stats.fallbacks;
        return false;
    }

    const auto submit_start = std::chrono::steady_clock::now();
    const bool submit_ok = ggml_moe_common::object_io_ring_submit(ring);
    const auto submit_end = std::chrono::steady_clock::now();
    g_expert_pack.io_stats.submit_us.fetch_add(
            (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(submit_end - submit_start).count());
    if (!submit_ok) {
        ggml_moe_common::object_io_ring_release(ring);
        if (read_dst != dst) std::free(read_dst);
        ++g_expert_pack.io_stats.fallbacks;
        return false;
    }

    ggml_moe_common::object_io_completion completion;
    const auto wait_start = std::chrono::steady_clock::now();
    const bool wait_ok = ggml_moe_common::object_io_ring_wait(ring, completion);
    const auto wait_end = std::chrono::steady_clock::now();
    g_expert_pack.io_stats.wait_us.fetch_add(
            (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(wait_end - wait_start).count());
    ggml_moe_common::object_io_completion_status status =
        ggml_moe_common::object_io_completion_status::invalid_user_data;
    if (wait_ok) {
        if (common_object_cqe) {
            ggml_moe_common::object_io_window_completion resolved;
            status = ggml_moe_common::object_io_window_resolve(
                ring, io_window, 1, completion, resolved);
            if (status == ggml_moe_common::object_io_completion_status::complete &&
                    (resolved.window_slot != 0 || resolved.request_id != 0)) {
                status = ggml_moe_common::object_io_completion_status::invalid_user_data;
            }
        } else {
            status = ggml_moe_common::object_io_completion_validate(
                ring, completion, completion.user_data == 1, read_sz);
        }
    }
    const bool ok = status == ggml_moe_common::object_io_completion_status::complete;
    ggml_moe_common::object_io_ring_release(ring);

    if (ok) {
        if (read_dst != dst) {
            std::memcpy(dst, (const char *)read_dst + payload_shift, sz);
            std::free(read_dst);
        }
        ++g_expert_pack.io_stats.reads;
        g_expert_pack.io_stats.bytes.fetch_add(sz);
        return true;
    }

    if (read_dst != dst) std::free(read_dst);
    ++g_expert_pack.io_stats.fallbacks;
    return false;
#else
    (void)entry;
    (void)dst;
    (void)sz;
    return false;
#endif
}

static bool expert_pack_direct_pread_all(int fd, void *dst, size_t sz, uint64_t offset) {
    return ggml_moe_common::object_pread_all(
        fd, dst, sz, offset, static_cast<size_t>(64) * 1024 * 1024);
}

static bool expert_pack_direct_read_entry_to_host(const expert_pack_entry *entry, void *dst, size_t sz) {
#if !defined(_WIN32)
    expert_pack_source *source = expert_pack_source_for_entry(entry);
    if (!entry || !source || source->fd_direct < 0 || entry->nbytes != sz) return false;
    const uint64_t alignment = expert_pack_direct_alignment();
    const uint64_t aligned_offset = (entry->offset / alignment) * alignment;
    const size_t prefix = (size_t)(entry->offset - aligned_offset);
    const size_t read_sz = (size_t)align_up_u64((uint64_t)prefix + (uint64_t)sz, alignment);
    if ((entry->offset % alignment) == 0 &&
            ((uintptr_t)dst % alignment) == 0 &&
            read_sz == (size_t)align_up_u64((uint64_t)sz, alignment)) {
        return expert_pack_direct_pread_native(source->fd_direct, dst, read_sz, sz, entry->offset);
    }
    if (source->direct_required && !entry->alias_source) {
        return false;
    }

    // Reuse one high-water buffer per reader. Repeated mmap-sized temporary
    // allocations otherwise fault every page on every unaligned native read.
    // Its resident pages are charged before subsequent expert-cache admission.
    struct aligned_bounce {
        void * data = nullptr;
        size_t capacity = 0, allocated_alignment = 0;
        ~aligned_bounce() { free(data); }
        bool reserve(size_t bytes, size_t align) {
            if (data && capacity >= bytes && allocated_alignment % align == 0) {
                return true;
            }
            free(data);
            data = nullptr;
            capacity = allocated_alignment = 0;
            if (posix_memalign(&data, align, bytes) != 0 || !data) {
                return false;
            }
            capacity = bytes;
            allocated_alignment = align;
            return true;
        }
    };
    static thread_local aligned_bounce scratch;
    if (!scratch.reserve(read_sz, alignment)) {
        return false;
    }
    const bool ok = expert_pack_direct_pread_native(source->fd_direct, scratch.data, read_sz, prefix + sz, aligned_offset);
    if (ok) {
        std::memcpy(dst, (const char *)scratch.data + prefix, sz);
    }
    return ok;
#else
    (void)entry;
    (void)dst;
    (void)sz;
    return false;
#endif
}

static bool expert_pack_read_entry(const expert_pack_entry *entry, void *dst, size_t sz) {
    expert_pack_source *source = expert_pack_source_for_entry(entry);
    if (!entry || !source || !source->file || entry->nbytes != sz) return false;

    std::call_once(g_ram_tier_once, ram_tier_init);

    if (source->page_cache) {
        expert_pack_source *mmap_source = expert_pack_mmap_ensure(entry);
        if (mmap_source && entry->offset <= mmap_source->mmap_size &&
                entry->offset + entry->nbytes <= mmap_source->mmap_size) {
            std::memcpy(dst, (const char *)mmap_source->mmap_base + entry->offset, sz);
            ++g_expert_pack.page_cache_reads;
            g_expert_pack.page_cache_bytes.fetch_add((uint64_t)sz);
            return true;
        }
        std::lock_guard<std::mutex> lk(g_expert_pack.mu);
#if defined(_WIN32)
        if (_fseeki64(source->file, (int64_t)entry->offset, SEEK_SET) != 0) {
#else
        if (::fseeko(source->file, (off_t)entry->offset, SEEK_SET) != 0) {
#endif
            ++g_expert_pack.read_failures;
            return false;
        }
        if (!expert_pack_read_exact(source->file, dst, sz)) {
            ++g_expert_pack.read_failures;
            return false;
        }
        ++g_expert_pack.page_cache_reads;
        g_expert_pack.page_cache_bytes.fetch_add((uint64_t)sz);
        return true;
    }

#if !defined(_WIN32)
    if (g_expert_pack.io_backend == 2 && source->fd_direct >= 0 &&
            expert_pack_env_bool("GGML_MOE_IO_URING_SINGLE", false)) {
        if (expert_pack_read_entry_iouring(entry, dst, sz)) {
            return true;
        }
    }
    if ((g_expert_pack.io_backend == 1 || g_expert_pack.io_backend == 2) && source->fd_direct >= 0) {
        const bool profile_direct = direct_read_site_profile_path() != nullptr;
        const auto direct_start = profile_direct ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const bool direct_ok = expert_pack_direct_read_entry_to_host(entry, dst, sz);
        if (profile_direct) {
            const auto direct_end = std::chrono::steady_clock::now();
            direct_read_site_profile_record(
                    g_direct_read_site_op ? g_direct_read_site_op : "expert_pack_read_entry",
                    g_direct_read_site_tensor ? g_direct_read_site_tensor : entry->tensor,
                    sz,
                    direct_ok,
                    std::chrono::duration<double, std::milli>(direct_end - direct_start).count());
        }
        if (direct_ok) {
            ++g_expert_pack.direct_reads;
            return true;
        }
        ++g_expert_pack.direct_fallbacks;
        if (source->direct_required) {
            ++g_expert_pack.read_failures;
            return false;
        }
    }
#endif

    if (source->direct_required) {
        ++g_expert_pack.read_failures;
        return false;
    }

    std::lock_guard<std::mutex> lk(g_expert_pack.mu);
#if defined(_WIN32)
    if (_fseeki64(source->file, (int64_t)entry->offset, SEEK_SET) != 0) {
#else
    if (::fseeko(source->file, (off_t)entry->offset, SEEK_SET) != 0) {
#endif
        ++g_expert_pack.read_failures;
        return false;
    }
    if (!expert_pack_read_exact(source->file, dst, sz)) {
        ++g_expert_pack.read_failures;
        return false;
    }
    return true;
}

// Borrow a stable pack fd without touching the mapped payload or initializing a
// GPU consumer. The caller owns every submitted I/O until its completion.
extern "C" bool ggml_cuda_moe_expert_pack_resolve_direct(
        const char * tensor_name,
        int expert_idx,
        size_t nbytes,
        int * fd_direct,
        uint64_t * offset,
        size_t * direct_nbytes) {
    if (fd_direct) *fd_direct = -1;
    if (offset) *offset = 0;
    if (direct_nbytes) *direct_nbytes = 0;
#if defined(__linux__)
    if (!tensor_name || !tensor_name[0] || expert_idx < 0 || !nbytes ||
            !fd_direct || !offset || !direct_nbytes || nbytes > uint64_t(INT64_MAX)) {
        return false;
    }
    const expert_pack_entry * entry = expert_pack_lookup_impl(tensor_name, expert_idx, nbytes, false);
    const expert_pack_source * source = expert_pack_source_for_entry(entry);
    const size_t padded_nbytes = static_cast<size_t>(align_up_u64(nbytes, 4096));
    if (!entry || !source || source->fd_direct < 0 || entry->offset % 4096 != 0 ||
            (padded_nbytes != nbytes && entry->alias_source) ||
            entry->offset > uint64_t(INT64_MAX) - padded_nbytes ||
            entry->offset > source->file_size || padded_nbytes > source->file_size - entry->offset) {
        return false;
    }
    const int flags = fcntl(source->fd_direct, F_GETFL);
    if (flags < 0 || !(flags & O_DIRECT)) {
        return false;
    }
    *fd_direct = source->fd_direct;
    *offset = entry->offset;
    *direct_nbytes = padded_nbytes;
    return true;
#else
    (void) tensor_name;
    (void) expert_idx;
    (void) nbytes;
    (void) direct_nbytes;
    return false;
#endif
}

// Logical native range for an aligned-read consumer. This API neither copies
// weights nor rounds their byte offsets; the CPU sink owns its bounded bounce.
extern "C" bool ggml_cuda_moe_common_expert_pack_resolve_raw_direct(
        const char * tensor_name, int expert_idx, size_t nbytes,
        int * fd_direct, uint64_t * offset, size_t * logical_nbytes) {
#if defined(__linux__)
    if (!tensor_name || expert_idx < 0 || !nbytes || !fd_direct || !offset || !logical_nbytes) { return false; }
    const expert_pack_entry * entry = expert_pack_lookup_impl(tensor_name, expert_idx, nbytes, false);
    const expert_pack_source * source = expert_pack_source_for_entry(entry);
    if (!entry || !source || source->fd_direct < 0 || entry->nbytes != nbytes ||
            entry->offset > source->file_size || nbytes > source->file_size - entry->offset) { return false; }
    const int flags = fcntl(source->fd_direct, F_GETFL);
    if (flags < 0 || !(flags & O_DIRECT)) { return false; }
    *fd_direct = source->fd_direct; *offset = entry->offset; *logical_nbytes = nbytes;
    return true;
#else
    (void) tensor_name; (void) expert_idx; (void) nbytes;
    (void) fd_direct; (void) offset; (void) logical_nbytes;
    return false;
#endif
}

extern "C" bool ggml_cuda_moe_expert_pack_read_to_host(
        const char *tensor_name,
        int expert_idx,
        size_t nbytes,
        void *dst,
        size_t dst_capacity,
        size_t *nread,
        int *source_is_gguf) {
    if (nread) {
        *nread = 0;
    }
    if (source_is_gguf) {
        *source_is_gguf = 0;
    }
    if (!tensor_name || !tensor_name[0] || expert_idx < 0 ||
            nbytes == 0 || !dst || dst_capacity < nbytes) {
        return false;
    }
    const expert_pack_entry *entry = expert_pack_lookup(tensor_name, expert_idx, nbytes);
    if (!entry) {
        return false;
    }
    const expert_pack_source *source = expert_pack_source_for_entry(entry);
    if (source_is_gguf && source && std::strstr(source->path, ".gguf")) {
        *source_is_gguf = 1;
    }
    if (!expert_pack_read_entry(entry, dst, nbytes)) {
        return false;
    }
    if (nread) {
        *nread = nbytes;
    }
    return true;
}

static bool stage_pinned_enabled() {
    const char *env = std::getenv("GGML_MOE_STAGE_PINNED");
    return env && env[0] && env[0] != '0';
}

static int stage_pinned_slot_count() {
    const char *env = std::getenv("GGML_MOE_STAGE_PINNED_SLOTS");
    long n = (env && env[0]) ? std::atol(env) : 16;
    if (n < 1) n = 1;
    if (n > 256) n = 256;
    return (int)n;
}





static bool pinned_stage_profile_enabled() {
    return g_bprof.enabled || g_uprof.enabled;
}

struct batch_copy_trace {
    bool pack_hit = false;
    bool ram_hit = false;
};

static bool copy_profile_enabled() {
    const char *env = std::getenv("GGML_MOE_COPY_PROFILE_OUT");
    return env && env[0];
}

static bool copy_profile_h2d_enabled() {
    return copy_profile_enabled() && expert_pack_env_bool("GGML_MOE_COPY_PROFILE_H2D", false);
}

static void copy_profile_record(
        const char *op,
        const char *tensor,
        int expert_idx,
        size_t bytes,
        bool pack_hit,
        bool ram_hit,
        bool iouring,
        double slot_wait_ms,
        double host_ms,
        double io_wait_ms,
        double enqueue_ms,
        double h2d_ms,
        double wall_ms) {
    const char *path = std::getenv("GGML_MOE_COPY_PROFILE_OUT");
    if (!path || !path[0]) return;

    static std::mutex mu;
    static bool header_written = false;
    static uint64_t seq = 0;
    std::lock_guard<std::mutex> lk(mu);

    FILE *f = std::fopen(path, "a");
    if (!f) return;
    if (!header_written) {
        std::fprintf(f,
                "seq,op,tensor,expert_idx,bytes,pack_hit,ram_hit,iouring,slot_wait_ms,host_ms,io_wait_ms,enqueue_ms,h2d_ms,wall_ms\n");
        header_written = true;
    }
    std::fprintf(f,
            "%lu,%s,%s,%d,%zu,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            (unsigned long)++seq,
            op ? op : "",
            tensor ? tensor : "",
            expert_idx,
            bytes,
            pack_hit ? 1 : 0,
            ram_hit ? 1 : 0,
            iouring ? 1 : 0,
            slot_wait_ms,
            host_ms,
            io_wait_ms,
            enqueue_ms,
            h2d_ms,
            wall_ms);
    std::fclose(f);
}

static bool ram_batch_profile_enabled() {
    const char *env = std::getenv("GGML_MOE_RAM_BATCH_PROFILE_OUT");
    return env && env[0];
}

static void ram_batch_profile_record(
        const char *op,
        const char *first_tensor,
        const char *last_tensor,
        size_t jobs,
        size_t ram_hit_jobs,
        size_t ram_hit_bytes,
        double enqueue_ms,
        double wall_ms) {
    const char *path = std::getenv("GGML_MOE_RAM_BATCH_PROFILE_OUT");
    if (!path || !path[0]) return;

    static std::mutex mu;
    static bool header_written = false;
    static uint64_t seq = 0;
    std::lock_guard<std::mutex> lk(mu);

    FILE *f = std::fopen(path, "a");
    if (!f) return;
    if (!header_written) {
        std::fprintf(f,
                "seq,op,first_tensor,last_tensor,jobs,ram_hit_jobs,ram_hit_bytes,enqueue_ms,wall_ms\n");
        header_written = true;
    }
    std::fprintf(f,
            "%lu,%s,%s,%s,%zu,%zu,%zu,%.6f,%.6f\n",
            (unsigned long)++seq,
            op ? op : "",
            first_tensor ? first_tensor : "",
            last_tensor ? last_tensor : "",
            jobs,
            ram_hit_jobs,
            ram_hit_bytes,
            enqueue_ms,
            wall_ms);
    std::fclose(f);
}

static bool io_batch_profile_enabled() {
    const char *env = std::getenv("GGML_MOE_IO_BATCH_PROFILE_OUT");
    return env && env[0];
}

static bool io_wait_trace_enabled() {
    const char *env = std::getenv("GGML_MOE_IO_WAIT_TRACE_OUT");
    return env && env[0];
}

static void io_wait_trace_record(
        uint64_t batch_seq,
        const char *op,
        const char *first_tensor,
        const char *last_tensor,
        size_t read_jobs,
        size_t completed_before,
        size_t inflight_before,
        size_t next_job_before,
        size_t depth,
        size_t refill_batch,
        double wait_ms,
        size_t drained_cqes,
        double enqueue_ms,
        size_t completed_after,
        size_t inflight_after) {
    const char *path = std::getenv("GGML_MOE_IO_WAIT_TRACE_OUT");
    if (!path || !path[0]) return;

    static std::mutex mu;
    static bool header_written = false;
    static uint64_t seq = 0;
    std::lock_guard<std::mutex> lk(mu);

    FILE *f = std::fopen(path, "a");
    if (!f) return;
    if (!header_written) {
        std::fprintf(f,
                "seq,batch_seq,op,first_tensor,last_tensor,read_jobs,completed_before,inflight_before,"
                "next_job,depth,refill_batch,wait_ms,drained_cqes,enqueue_ms,completed_after,inflight_after\n");
        header_written = true;
    }
    std::fprintf(f,
            "%lu,%lu,%s,%s,%s,%zu,%zu,%zu,%zu,%zu,%zu,%.6f,%zu,%.6f,%zu,%zu\n",
            (unsigned long)++seq,
            (unsigned long)batch_seq,
            op ? op : "",
            first_tensor ? first_tensor : "",
            last_tensor ? last_tensor : "",
            read_jobs,
            completed_before,
            inflight_before,
            next_job_before,
            depth,
            refill_batch,
            wait_ms,
            drained_cqes,
            enqueue_ms,
            completed_after,
            inflight_after);
    std::fclose(f);
}

static void io_batch_profile_record(
        const char *op,
        const char *first_tensor,
        const char *last_tensor,
        size_t jobs,
        size_t read_jobs,
        size_t ram_or_prefetch_jobs,
        size_t ram_or_prefetch_bytes,
        size_t depth,
        size_t slots,
        size_t refill_batch,
        size_t initial_submit_jobs,
        uint64_t submit_calls,
        uint64_t wait_calls,
        uint64_t cqes,
        uint64_t inflight_sum,
        uint64_t inflight_samples,
        uint64_t inflight_max,
        double slot_wait_ms,
        double submit_ms,
        double wait_ms,
        double enqueue_ms,
        double wall_ms,
        bool sort_by_offset) {
    const char *path = std::getenv("GGML_MOE_IO_BATCH_PROFILE_OUT");
    if (!path || !path[0]) return;

    static std::mutex mu;
    static bool header_written = false;
    static uint64_t seq = 0;
    std::lock_guard<std::mutex> lk(mu);

    FILE *f = std::fopen(path, "a");
    if (!f) return;
    if (!header_written) {
        std::fprintf(f,
                "seq,op,first_tensor,last_tensor,jobs,read_jobs,ram_or_prefetch_jobs,ram_or_prefetch_bytes,depth,slots,refill_batch,"
                "initial_submit_jobs,submit_calls,wait_calls,cqes,inflight_avg,inflight_max,"
                "slot_wait_ms,submit_ms,wait_ms,enqueue_ms,wall_ms,sort_by_offset\n");
        header_written = true;
    }

    const double inflight_avg = inflight_samples > 0 ?
        (double)inflight_sum / (double)inflight_samples : 0.0;
    std::fprintf(f,
            "%lu,%s,%s,%s,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%lu,%lu,%lu,%.6f,%lu,"
            "%.6f,%.6f,%.6f,%.6f,%.6f,%d\n",
            (unsigned long)++seq,
            op ? op : "",
            first_tensor ? first_tensor : "",
            last_tensor ? last_tensor : "",
            jobs,
            read_jobs,
            ram_or_prefetch_jobs,
            ram_or_prefetch_bytes,
            depth,
            slots,
            refill_batch,
            initial_submit_jobs,
            (unsigned long)submit_calls,
            (unsigned long)wait_calls,
            (unsigned long)cqes,
            inflight_avg,
            (unsigned long)inflight_max,
            slot_wait_ms,
            submit_ms,
            wait_ms,
            enqueue_ms,
            wall_ms,
            sort_by_offset ? 1 : 0);
    std::fclose(f);
}

struct io_locality_profile_item {
    int source_idx = -1;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
    char tensor[128] = {};
};

static bool io_locality_profile_enabled() {
    const char *env = std::getenv("GGML_MOE_IO_LOCALITY_PROFILE_OUT");
    return env && env[0];
}

static bool io_read_trace_enabled() {
    const char *env = std::getenv("GGML_MOE_IO_READ_TRACE_OUT");
    return env && env[0];
}

static bool h2d_coalesce_profile_enabled() {
    const char *env = std::getenv("GGML_MOE_H2D_COALESCE_PROFILE_OUT");
    return env && env[0];
}

static void h2d_coalesce_profile_record(
        const char *op,
        size_t jobs,
        size_t read_jobs,
        size_t copy_count,
        size_t coalesced_copy_count,
        size_t host_contiguous_pairs,
        size_t dst_contiguous_pairs,
        size_t both_contiguous_pairs,
        size_t expert_bytes) {
    const char *path = std::getenv("GGML_MOE_H2D_COALESCE_PROFILE_OUT");
    if (!path || !path[0] || copy_count == 0) return;

    static std::mutex mu;
    static bool header_written = false;
    static uint64_t seq = 0;
    std::lock_guard<std::mutex> lk(mu);

    FILE *f = std::fopen(path, "a");
    if (!f) return;
    if (!header_written) {
        std::fprintf(f,
                "seq,op,jobs,read_jobs,copy_count,coalesced_copy_count,"
                "host_contiguous_pairs,dst_contiguous_pairs,both_contiguous_pairs,"
                "expert_bytes,bytes,total_saved_copy_count\n");
        header_written = true;
    }
    const size_t saved = copy_count > coalesced_copy_count ? copy_count - coalesced_copy_count : 0;
    std::fprintf(f,
            "%lu,%s,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu\n",
            (unsigned long)++seq,
            op ? op : "",
            jobs,
            read_jobs,
            copy_count,
            coalesced_copy_count,
            host_contiguous_pairs,
            dst_contiguous_pairs,
            both_contiguous_pairs,
            expert_bytes,
            copy_count * expert_bytes,
            saved);
    std::fclose(f);
}

static void io_locality_profile_record(
        const char *op,
        size_t jobs,
        const std::vector<io_locality_profile_item> &items) {
    const char *path = std::getenv("GGML_MOE_IO_LOCALITY_PROFILE_OUT");
    if (!path || !path[0] || items.empty()) return;

    uint64_t read_bytes = 0;
    int source_switches = 0;
    bool same_tensor = true;
    for (size_t i = 0; i < items.size(); ++i) {
        read_bytes += items[i].nbytes;
        if (i > 0 && items[i].source_idx != items[i - 1].source_idx) {
            ++source_switches;
        }
        if (i > 0 && std::strcmp(items[i].tensor, items[0].tensor) != 0) {
            same_tensor = false;
        }
    }

    std::vector<io_locality_profile_item> sorted = items;
    std::stable_sort(sorted.begin(), sorted.end(),
        [](const io_locality_profile_item &a, const io_locality_profile_item &b) {
            if (a.source_idx != b.source_idx) return a.source_idx < b.source_idx;
            return a.offset < b.offset;
        });

    int unique_sources = 0;
    int last_source = INT_MIN;
    uint64_t span_bytes = 0;
    uint64_t gap_bytes = 0;
    uint64_t max_gap_bytes = 0;
    uint64_t adjacent_pairs = 0;
    uint64_t source_min = 0;
    uint64_t source_end = 0;
    bool have_source = false;

    for (const io_locality_profile_item &item : sorted) {
        const uint64_t item_end = item.offset + item.nbytes;
        if (!have_source || item.source_idx != last_source) {
            if (have_source) {
                span_bytes += source_end - source_min;
            }
            have_source = true;
            ++unique_sources;
            last_source = item.source_idx;
            source_min = item.offset;
            source_end = item_end;
            continue;
        }
        if (item.offset == source_end) {
            ++adjacent_pairs;
            source_end = item_end > source_end ? item_end : source_end;
        } else if (item.offset > source_end) {
            const uint64_t gap = item.offset - source_end;
            gap_bytes += gap;
            if (max_gap_bytes < gap) {
                max_gap_bytes = gap;
            }
            source_end = item_end;
        } else if (item_end > source_end) {
            source_end = item_end;
        }
    }
    if (have_source) {
        span_bytes += source_end - source_min;
    }

    static std::mutex mu;
    static bool header_written = false;
    static uint64_t seq = 0;
    std::lock_guard<std::mutex> lk(mu);

    FILE *f = std::fopen(path, "a");
    if (!f) return;
    if (!header_written) {
        std::fprintf(f,
                "seq,op,jobs,read_jobs,source_switches,unique_sources,read_bytes,"
                "span_bytes,gap_bytes,max_gap_bytes,adjacent_pairs,same_tensor,"
                "first_tensor,last_tensor\n");
        header_written = true;
    }
    std::fprintf(f,
            "%lu,%s,%zu,%zu,%d,%d,%lu,%lu,%lu,%lu,%lu,%d,%s,%s\n",
            (unsigned long)++seq,
            op ? op : "",
            jobs,
            items.size(),
            source_switches,
            unique_sources,
            (unsigned long)read_bytes,
            (unsigned long)span_bytes,
            (unsigned long)gap_bytes,
            (unsigned long)max_gap_bytes,
            (unsigned long)adjacent_pairs,
            same_tensor ? 1 : 0,
            items.front().tensor,
            items.back().tensor);
    std::fclose(f);
}

static uint64_t io_read_trace_next_batch_seq() {
    static std::atomic<uint64_t> seq{0};
    return ++seq;
}

static bool expert_pack_ram_tier_copy_h2d(
        const expert_pack_entry *pack_entry,
        void *dst,
        size_t sz,
        cudaStream_t st,
        batch_copy_trace *copy_trace = nullptr) {
    if (!pack_entry) return false;
    std::call_once(g_ram_tier_once, ram_tier_init);
    if (common_dynamic_tier_requested()) {
        const bool hit = common_dynamic_tier_copy_h2d(pack_entry, dst, sz, st);
        if (hit && copy_trace) copy_trace->ram_hit = true;
        return hit;
    }
    if (g_expert_pack.ram_tier_base == nullptr) return false;

    const size_t idx = (size_t)(pack_entry - g_expert_pack.entries.data());
    if (idx < g_expert_pack.ram_tier_index.size()) {
        const auto &rt = g_expert_pack.ram_tier_index[idx];
        if (rt.nbytes == sz) {
            const void *src = (const char *)g_expert_pack.ram_tier_base + rt.offset;
            if (cudaMemcpyAsync(dst, src, sz, cudaMemcpyHostToDevice, st) == cudaSuccess) {
                ++g_expert_pack.ram_tier_hits;
                ++g_expert_pack.ram_tier_total;
                ++g_expert_pack.ram_tier_h2d_enqueues;
                g_expert_pack.ram_tier_h2d_bytes.fetch_add(sz);
                if (copy_trace) copy_trace->ram_hit = true;
                return true;
            }
        }
    }

    ++g_expert_pack.ram_tier_total;
    return false;
}

static void pinned_stage_collect_timing(pinned_stage_ring &ring, pinned_stage_slot &slot) {
    float h2d_ms = 0.0f;
    if (ggml_moe_common::object_stage_slot_collect_timing(ring, slot, &h2d_ms) &&
            slot.object_ram_hit_timing_pending) {
        const uint64_t h2d_us = (uint64_t)(h2d_ms * 1000.0f);
        g_expert_pack.q4_object_ram_h2d_us.fetch_add(h2d_us, std::memory_order_relaxed);
        q4_object_layer_add(g_expert_pack.q4_object_layer_ram_h2d_us, slot.object_ram_hit_layer, h2d_us);
        q4_object_layer_add(
                g_expert_pack.q4_object_layer_ram_h2d_bytes,
                slot.object_ram_hit_layer,
                slot.object_ram_hit_bytes);
    }
    slot.object_ram_hit_timing_pending = false;
    slot.object_ram_hit_layer = -1;
    slot.object_ram_hit_bytes = 0;
}

static void pinned_stage_release(pinned_stage_ring &ring) {
    for (pinned_stage_slot &slot : ring.slots) {
        pinned_stage_collect_timing(ring, slot);
    }
    ggml_moe_common::object_stage_ring_release(ring);
}

static void pinned_stage_report_ring(const char *name, pinned_stage_ring &ring) {
    for (pinned_stage_slot &slot : ring.slots) {
        if (slot.pending && slot.done) {
            cudaEventSynchronize(slot.done);
            slot.pending = false;
        }
        pinned_stage_collect_timing(ring, slot);
    }
    if (ring.copies == 0 && ring.fallbacks == 0) return;
    if (ring.h2d_timed > 0 || ring.host_stage_ms > 0.0 || ring.enqueue_ms > 0.0 || ring.slot_wait_ms > 0.0) {
        std::fprintf(stderr,
            "[moe_stream_batch] pinned staging%s: copies=%lu waits=%lu fallbacks=%lu slots=%zu slot=%.2f MiB "
            "slot_wait=%.3f ms host_stage=%.3f ms enqueue=%.3f ms h2d=%.3f ms h2d_timed=%lu\n",
            name, ring.copies, ring.waits, ring.fallbacks, ring.slots.size(), ring.slot_sz / (1024.0 * 1024.0),
            ring.slot_wait_ms, ring.host_stage_ms, ring.enqueue_ms, ring.h2d_ms, ring.h2d_timed);
    } else {
        std::fprintf(stderr,
            "[moe_stream_batch] pinned staging%s: copies=%lu waits=%lu fallbacks=%lu slots=%zu slot=%.2f MiB\n",
            name, ring.copies, ring.waits, ring.fallbacks, ring.slots.size(), ring.slot_sz / (1024.0 * 1024.0));
    }
    if (ring.iouring_init_attempts > 0 || ring.iouring_init_reuses > 0 ||
            ring.iouring_init_failures > 0 || ring.iouring_batches > 0) {
        const double inflight_avg = ring.iouring_inflight_samples > 0 ?
            (double)ring.iouring_inflight_sum / (double)ring.iouring_inflight_samples : 0.0;
        std::fprintf(stderr,
            "[moe_stream_batch] pinned staging%s iouring: init=%lu init_reuse=%lu init_fail=%lu "
            "batches=%lu jobs=%lu submit_calls=%lu wait_calls=%lu cqes=%lu "
            "validated=%lu completed=%lu invalid_user_data=%lu invalid_expected_bytes=%lu "
            "io_errors=%lu short_reads=%lu oversized_reads=%lu inflight_avg=%.2f inflight_max=%lu "
            "sqpoll_affinity=%lu/%lu/%lu "
            "common_window_prepares=%lu common_window_resolves=%lu drained_cqes=%lu "
            "batch_hist=1:%lu,2-4:%lu,5-8:%lu,9-16:%lu,17-32:%lu,gt32:%lu\n",
            name, ring.iouring_init_attempts, ring.iouring_init_reuses, ring.iouring_init_failures,
            ring.iouring_batches, ring.iouring_jobs, ring.iouring_submit_calls,
            ring.iouring_wait_calls, ring.iouring_cqes,
            ring.completion_counters.validations, ring.completion_counters.completed,
            ring.completion_counters.invalid_user_data,
            ring.completion_counters.invalid_expected_bytes,
            ring.completion_counters.io_errors, ring.completion_counters.short_reads,
            ring.completion_counters.oversized_reads, inflight_avg, ring.iouring_inflight_max,
            ring.iouring_sqpoll_affinity_attempts,
            ring.iouring_sqpoll_affinity_successes,
            ring.iouring_sqpoll_affinity_failures,
            ring.window_prepares, ring.window_resolves, ring.iouring_drained_cqes,
            ring.iouring_batch_hist[0], ring.iouring_batch_hist[1], ring.iouring_batch_hist[2],
            ring.iouring_batch_hist[3], ring.iouring_batch_hist[4], ring.iouring_batch_hist[5]);
    }
    if (ring.granularity_calls > 0) {
        const double calls = (double)ring.granularity_calls;
        std::fprintf(stderr,
            "[moe_stream_batch] pinned staging%s granularity: calls=%lu avg_jobs=%.2f avg_read_jobs=%.2f "
            "avg_depth=%.2f avg_slots=%.2f max_jobs=%lu max_read_jobs=%lu max_depth=%lu\n",
            name, ring.granularity_calls,
            (double)ring.granularity_total_jobs / calls,
            (double)ring.granularity_read_jobs / calls,
            (double)ring.granularity_depth_sum / calls,
            (double)ring.granularity_slots_sum / calls,
            ring.granularity_max_jobs,
            ring.granularity_max_read_jobs,
            ring.granularity_max_depth);
    }
}

static void pinned_stage_report_atexit() {
    pinned_stage_report_ring("", g_batch.stage_ring);
    pinned_stage_report_ring(" gate", g_batch.stage_ring_gate);
    pinned_stage_report_ring(" up_aux", g_batch.stage_ring_up_aux);
    pinned_stage_report_ring(" gate_aux", g_batch.stage_ring_gate_aux);
    pinned_stage_report_ring(" down_aux", g_batch.stage_ring_down_aux);
}

static bool pinned_stage_ensure_slots(pinned_stage_ring &ring, size_t need, bool force, int n_slots) {
    if (!force && !stage_pinned_enabled()) return false;
    if (ring.failed) return false;
    const size_t alloc_need = (size_t)align_up_u64((uint64_t)need, (uint64_t)expert_pack_direct_alignment());

    if (n_slots < 1) n_slots = 1;
    if (!ring.slots.empty() && (int)ring.slots.size() == n_slots && ring.slot_sz >= alloc_need) {
        return true;
    }

    pinned_stage_release(ring);
    const bool profile_stage = pinned_stage_profile_enabled();
    const bool profile_copy_h2d = copy_profile_h2d_enabled();
    const bool create_timing_events = profile_stage || profile_copy_h2d;
    if (!ggml_moe_common::object_stage_ring_allocate(
                ring, alloc_need, (size_t)n_slots, create_timing_events)) {
        std::fprintf(stderr,
            "[moe_stream_batch] pinned staging: init failed for %d slots of %.2f MiB; falling back to direct H2D\n",
            n_slots, need / (1024.0 * 1024.0));
        return false;
    }

    if (!g_pinned_stage_report_registered.exchange(true)) {
        std::atexit(pinned_stage_report_atexit);
    }
    ring.report_registered = true;
    std::fprintf(stderr, "[moe_stream_batch] pinned staging: enabled, %d slots of %.2f MiB\n",
                 n_slots, need / (1024.0 * 1024.0));
    return true;
}

static bool pinned_stage_ensure(pinned_stage_ring &ring, size_t need, bool force) {
    return pinned_stage_ensure_slots(ring, need, force, stage_pinned_slot_count());
}

struct q4_direct_object_trace_state {
    std::mutex mu;
    bool header_written = false;
    uint64_t seq = 0;
};

static q4_direct_object_trace_state g_q4_direct_object_trace;

static const char *q4_direct_object_trace_path() {
    static const char *path = std::getenv("GGML_MOE_Q4_OBJECT_TIER_TRACE_OUT");
    return path && path[0] ? path : nullptr;
}

static bool q4_direct_object_trace_enabled() {
    return false;
}

static bool q4_direct_host_object_copy_h2d(
        const expert_pack_entry *pack_entry,
        void *dst,
        size_t role_bytes,
        cudaStream_t st);

static void q4_direct_host_object_submit_on_read(
        const expert_pack_entry *pack_entry,
        uint64_t read_us);

static void q4_colibri_host_object_preload_allowlist_once(size_t expert_bytes);

static void q4_direct_object_trace_record(
        const expert_pack_entry *pack_entry,
        const char *tensor_name,
        int expert_idx,
        size_t role_bytes,
        const char *demand_tier,
        const char *provider_path,
        bool ram_ready) {
    if (!q4_direct_object_trace_enabled() || !pack_entry) return;

    const int layer = pack_entry->layer_idx;
    const int expert = pack_entry->expert_idx >= 0 ? pack_entry->expert_idx : expert_idx;
    if (layer < 0 || expert < 0) return;

    std::lock_guard<std::mutex> lock(g_q4_direct_object_trace.mu);
    const char *path = q4_direct_object_trace_path();
    if (!path) return;
    FILE *f = std::fopen(path, "a");
    if (!f) return;
    if (!g_q4_direct_object_trace.header_written) {
        std::fprintf(f,
            "seq,epoch,layer,expert,source_idx,offset,object_bytes,active_rows,"
            "vram_present_roles,vram_ready_roles,ram_ready,missing_roles,scheduled_roles,"
            "logical_role_bytes,demand_tier,provider_path,ram_lookup_allowed,"
            "ram_admit_allowed,ram_net_allowed,after_vram_eviction,vram_eviction_age,"
            "current_layer,layer_distance,score_freq,object_score,"
            "ram_slot_present,ram_slot_ready,ram_slot_loading,ram_slot_delete_after_h2d,"
            "ram_slot_freq,ram_slot_score,ram_slot_used,ram_slot_origin,tensor,role_offset,trace_source\n");
        g_q4_direct_object_trace.header_written = true;
    }
    const uint64_t object_offset = pack_entry->object_nbytes > 0 ? pack_entry->object_offset : pack_entry->offset;
    const uint64_t object_bytes = pack_entry->object_nbytes > 0 ? pack_entry->object_nbytes : role_bytes;
    const uint64_t role_offset = pack_entry->object_nbytes > 0 ? pack_entry->object_relative_offset : 0;
    std::fprintf(f,
        "%llu,0,%d,%d,%d,%llu,%llu,1,0,0,%d,1,1,%llu,%s,%s,0,0,1,0,0,-1,0,1,0,0,0,0,0,0,0,0,%s,%s,%llu,direct_role_read\n",
        (unsigned long long)++g_q4_direct_object_trace.seq,
        layer,
        expert,
        (int)pack_entry->source_idx,
        (unsigned long long)object_offset,
        (unsigned long long)object_bytes,
        ram_ready ? 1 : 0,
        (unsigned long long)role_bytes,
        demand_tier ? demand_tier : "ssd",
        provider_path ? provider_path : "ssd_read_h2d",
        ram_ready ? "direct_ram_tier" : "",
        tensor_name ? tensor_name : pack_entry->tensor,
        (unsigned long long)role_offset);
    std::fclose(f);
}

static bool batch_cache_copy_h2d(
        pinned_stage_ring &ring,
        void *dst, const void *host_data, size_t sz, cudaStream_t st,
        const expert_pack_entry *pack_entry,
        batch_copy_trace *copy_trace = nullptr,
        const char *trace_op = nullptr,
        const char *tensor_name = nullptr,
        int expert_idx = -1) {
    if (copy_trace) {
        copy_trace->pack_hit = pack_entry != nullptr;
        copy_trace->ram_hit = false;
    }
    const bool profile_copy = copy_profile_enabled() && trace_op && trace_op[0];
    const auto copy_profile_start = profile_copy ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (g_expert_pack.q4_colibri && q4_colibri_routed_tensor_name(tensor_name) && !pack_entry) {
        q4_colibri_fail(std::string("routed tensor cannot fall back to host copy tensor=") +
            (tensor_name ? tensor_name : "") +
            " expert=" + std::to_string(expert_idx) +
            " bytes=" + std::to_string(sz));
    }

    if (pack_entry && pack_entry->object_nbytes > 0) {
        q4_colibri_host_object_preload_allowlist_once(sz);
    }

    if (q4_direct_host_object_copy_h2d(pack_entry, dst, sz, st)) {
        q4_direct_object_trace_record(
                pack_entry, tensor_name, expert_idx, sz,
                "ram", "ram_h2d", true);
        if (copy_trace) copy_trace->ram_hit = true;
        if (profile_copy) {
            const auto copy_profile_end = std::chrono::steady_clock::now();
            copy_profile_record(
                    trace_op, tensor_name, expert_idx, sz,
                    pack_entry != nullptr, true, false,
                    0.0, 0.0, 0.0, 0.0, -1.0,
                    std::chrono::duration<double, std::milli>(copy_profile_end - copy_profile_start).count());
        }
        return true;
    }

    // RAM tier fast path: if the entry is resident in the registered mmap,
    // do a direct H2D from the pinned region, bypassing the staging slot.
    if (expert_pack_ram_tier_copy_h2d(pack_entry, dst, sz, st, copy_trace)) {
        q4_direct_object_trace_record(
                pack_entry, tensor_name, expert_idx, sz,
                "ram", "ram_h2d", true);
        if (profile_copy) {
            const auto copy_profile_end = std::chrono::steady_clock::now();
            copy_profile_record(
                    trace_op, tensor_name, expert_idx, sz,
                    pack_entry != nullptr, true, false,
                    0.0, 0.0, 0.0, 0.0, -1.0,
                    std::chrono::duration<double, std::milli>(copy_profile_end - copy_profile_start).count());
        }
        return true;
    }

    const bool use_pinned_stage = stage_pinned_enabled() || pack_entry;
    if (use_pinned_stage) {
        if (pinned_stage_ensure(ring, sz, pack_entry != nullptr)) {
            const bool profile_stage = pinned_stage_profile_enabled();
            const bool profile_copy_h2d = copy_profile_h2d_enabled() && profile_copy;
            pinned_stage_slot &slot = ring.slots[ring.next++ % ring.slots.size()];
            double slot_wait_ms = 0.0;
            if (slot.pending) {
                const auto wait_start = profile_stage ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                const auto copy_wait_start = profile_copy && !profile_stage ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                if (!ggml_moe_common::object_stage_slot_synchronize(ring, slot)) return false;
                if (profile_stage) {
                    const auto wait_end = std::chrono::steady_clock::now();
                    slot_wait_ms = std::chrono::duration<double, std::milli>(wait_end - wait_start).count();
                    ring.slot_wait_ms += slot_wait_ms;
                } else if (profile_copy) {
                    const auto wait_end = std::chrono::steady_clock::now();
                    slot_wait_ms = std::chrono::duration<double, std::milli>(wait_end - copy_wait_start).count();
                }
            }
            pinned_stage_collect_timing(ring, slot);

            const bool measure_host = profile_stage || profile_copy;
            const auto host_start = measure_host ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (pack_entry) {
                direct_read_site_context_scope direct_scope(trace_op, tensor_name);
                const auto read_start = std::chrono::steady_clock::now();
                const bool read_ok = expert_pack_read_entry(pack_entry, slot.host, sz);
                const uint64_t read_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - read_start).count();
                if (!read_ok) {
                    return false;
                }
                common_dynamic_tier_fill(pack_entry, slot.host, sz);
                q4_direct_object_trace_record(
                        pack_entry, tensor_name, expert_idx, sz,
                        "ssd", "ssd_read_h2d", false);
                q4_direct_host_object_submit_on_read(pack_entry, read_us);
            } else {
                std::memcpy(slot.host, host_data, sz);
            }
            double host_ms = 0.0;
            if (profile_stage) {
                const auto host_end = std::chrono::steady_clock::now();
                host_ms = std::chrono::duration<double, std::milli>(host_end - host_start).count();
                ring.host_stage_ms += host_ms;
            } else if (profile_copy) {
                const auto host_end = std::chrono::steady_clock::now();
                host_ms = std::chrono::duration<double, std::milli>(host_end - host_start).count();
            }
            const bool measure_enqueue = profile_stage || profile_copy;
            const auto enqueue_start = measure_enqueue ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if ((profile_stage || profile_copy_h2d) && slot.copy_start) {
                if (cudaEventRecord(slot.copy_start, st) != cudaSuccess) return false;
            }
            if (cudaMemcpyAsync(dst, slot.host, sz, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
            if ((profile_stage || profile_copy_h2d) && slot.copy_done) {
                if (cudaEventRecord(slot.copy_done, st) != cudaSuccess) return false;
                if (profile_stage) {
                    slot.timing_pending = true;
                }
            }
            if (!ggml_moe_common::object_stage_slot_record(slot, st)) {
                cudaStreamSynchronize(st);
                return false;
            }
            double enqueue_ms = 0.0;
            if (profile_stage) {
                const auto enqueue_end = std::chrono::steady_clock::now();
                enqueue_ms = std::chrono::duration<double, std::milli>(enqueue_end - enqueue_start).count();
                ring.enqueue_ms += enqueue_ms;
            } else if (profile_copy) {
                const auto enqueue_end = std::chrono::steady_clock::now();
                enqueue_ms = std::chrono::duration<double, std::milli>(enqueue_end - enqueue_start).count();
            }
            ++ring.copies;
            if (profile_copy) {
                double h2d_ms = -1.0;
                if (profile_copy_h2d && slot.copy_start && slot.copy_done &&
                        cudaEventSynchronize(slot.copy_done) == cudaSuccess) {
                    float h2d_ms_f = 0.0f;
                    if (cudaEventElapsedTime(&h2d_ms_f, slot.copy_start, slot.copy_done) == cudaSuccess) {
                        h2d_ms = h2d_ms_f;
                    }
                }
                const auto copy_profile_end = std::chrono::steady_clock::now();
                copy_profile_record(
                        trace_op, tensor_name, expert_idx, sz,
                        pack_entry != nullptr, false, false,
                        slot_wait_ms, host_ms, 0.0, enqueue_ms, h2d_ms,
                        std::chrono::duration<double, std::milli>(copy_profile_end - copy_profile_start).count());
            }
            return true;
        }
        ++ring.fallbacks;
    }

    const auto enqueue_start = profile_copy ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool ok = cudaMemcpyAsync(dst, host_data, sz, cudaMemcpyHostToDevice, st) == cudaSuccess;
    if (profile_copy && ok) {
        const auto copy_profile_end = std::chrono::steady_clock::now();
        copy_profile_record(
                trace_op, tensor_name, expert_idx, sz,
                pack_entry != nullptr, false, false,
                0.0, 0.0, 0.0,
                std::chrono::duration<double, std::milli>(copy_profile_end - enqueue_start).count(),
                -1.0,
                std::chrono::duration<double, std::milli>(copy_profile_end - copy_profile_start).count());
    }
    return ok;
}

static bool batch_cache_copy_h2d(
        void *dst, const void *host_data, size_t sz, cudaStream_t st,
        const expert_pack_entry *pack_entry) {
    return batch_cache_copy_h2d(g_batch.stage_ring, dst, host_data, sz, st, pack_entry, nullptr);
}













































struct q4_colibri_host_object_prefetch_state {
    std::atomic<uint64_t> static_hotness_score_hits{0};
    std::atomic<uint64_t> static_hotness_score_bonus{0};
};

static q4_colibri_host_object_prefetch_state g_q4_host_object_prefetch;































struct q4_graph_hit_profile_state {
    std::mutex mu;
    bool report_registered = false;
    std::atomic<uint64_t> upgate_calls{0};
    std::atomic<uint64_t> down_calls{0};
    std::atomic<uint64_t> prompt_skips{0};
    std::atomic<uint64_t> layer_skips{0};
    std::atomic<uint64_t> descriptor_overflow{0};
    std::atomic<uint64_t> active_roles{0};
    std::atomic<uint64_t> vram_ready_roles{0};
    std::atomic<uint64_t> vram_miss_roles{0};
    std::atomic<uint64_t> full_hit_calls{0};
    std::atomic<uint64_t> upgate_full_hit_calls{0};
    std::atomic<uint64_t> down_full_hit_calls{0};
    std::atomic<uint64_t> role_active_roles[3]{};
    std::atomic<uint64_t> role_ready_roles[3]{};
    std::atomic<uint64_t> blocked_host_expanded_slots{0};
    std::atomic<uint64_t> would_replay_calls{0};
    std::atomic<uint64_t> upgate_miss_hist[17]{};
    std::atomic<uint64_t> down_miss_hist[9]{};
    std::atomic<uint64_t> layer_upgate_calls[128]{};
    std::atomic<uint64_t> layer_down_calls[128]{};
    std::atomic<uint64_t> layer_active_roles[128]{};
    std::atomic<uint64_t> layer_ready_roles[128]{};
    std::atomic<uint64_t> layer_full_hit_calls[128]{};
    std::atomic<uint64_t> layer_up_active_roles[128]{};
    std::atomic<uint64_t> layer_up_ready_roles[128]{};
    std::atomic<uint64_t> layer_gate_active_roles[128]{};
    std::atomic<uint64_t> layer_gate_ready_roles[128]{};
    std::atomic<uint64_t> layer_down_active_roles[128]{};
    std::atomic<uint64_t> layer_down_ready_roles[128]{};
    std::atomic<uint64_t> layer_upgate_full_hit_calls[128]{};
    std::atomic<uint64_t> layer_down_full_hit_calls[128]{};
    std::atomic<uint64_t> post_provider_calls{0};
    std::atomic<uint64_t> post_provider_active_objects{0};
    std::atomic<uint64_t> post_provider_present_roles{0};
    std::atomic<uint64_t> post_provider_pending_roles{0};
    std::atomic<uint64_t> post_provider_ready_roles{0};
    std::atomic<uint64_t> post_provider_full_present_calls{0};
    std::atomic<uint64_t> post_provider_full_ready_calls{0};
    std::atomic<uint64_t> post_wait_calls{0};
    std::atomic<uint64_t> post_wait_active_objects{0};
    std::atomic<uint64_t> post_wait_present_roles{0};
    std::atomic<uint64_t> post_wait_pending_roles{0};
    std::atomic<uint64_t> post_wait_ready_roles{0};
    std::atomic<uint64_t> post_wait_full_present_calls{0};
    std::atomic<uint64_t> post_wait_full_ready_calls{0};
    std::atomic<uint64_t> layer_post_provider_calls[128]{};
    std::atomic<uint64_t> layer_post_provider_active_objects[128]{};
    std::atomic<uint64_t> layer_post_provider_present_roles[128]{};
    std::atomic<uint64_t> layer_post_provider_pending_roles[128]{};
    std::atomic<uint64_t> layer_post_provider_ready_roles[128]{};
    std::atomic<uint64_t> layer_post_provider_full_present_calls[128]{};
    std::atomic<uint64_t> layer_post_provider_full_ready_calls[128]{};
    std::atomic<uint64_t> layer_post_wait_calls[128]{};
    std::atomic<uint64_t> layer_post_wait_active_objects[128]{};
    std::atomic<uint64_t> layer_post_wait_present_roles[128]{};
    std::atomic<uint64_t> layer_post_wait_pending_roles[128]{};
    std::atomic<uint64_t> layer_post_wait_ready_roles[128]{};
    std::atomic<uint64_t> layer_post_wait_full_present_calls[128]{};
    std::atomic<uint64_t> layer_post_wait_full_ready_calls[128]{};
    std::atomic<uint64_t> descriptor_dry_run_calls{0};
    std::atomic<uint64_t> descriptor_dry_run_objects{0};
    std::atomic<uint64_t> descriptor_dry_run_bytes{0};
    std::atomic<uint64_t> descriptor_dry_run_copy_us{0};
    std::atomic<uint64_t> descriptor_dry_run_failures{0};
};
static q4_graph_hit_profile_state g_q4_graph_hit_profile;





static bool q4_colibri_pipe_requested() {
    return false;
}









static int q4_graph_descriptor_max_experts() {
    return (int)expert_pack_env_size("GGML_MOE_Q4_GRAPH_DESCRIPTOR_MAX_EXPERTS", 384, 1, 4096);
}



static int q4_graph_layer_from_tensor(const char *tensor_name) {
    int layer = -1;
    char kind[12] = {};
    batch_route_detail_parse_name(tensor_name, layer, kind, sizeof(kind));
    return layer >= 0 && layer < 128 ? layer : -1;
}


struct vram_cache_layer_bias_config {
    bool inited = false;
    bool any = false;
    bool allow_all = false;
    bool allowed[128] = {};
    size_t ticks = 0;
    size_t layer_ticks[128] = {};
    bool dynamic_near = false;
    int dynamic_window = 0;
    size_t dynamic_ticks = 0;
};

static const vram_cache_layer_bias_config &vram_cache_layer_bias_config_get() {
    static vram_cache_layer_bias_config cfg;
    if (cfg.inited) return cfg;
    cfg.ticks = expert_pack_env_size("GGML_MOE_VRAM_CACHE_LAYER_BIAS_TICKS", 0, 0, 1000000000ULL);
    const char *env = std::getenv("GGML_MOE_VRAM_CACHE_LAYER_BIAS_LAYERS");
    if (env && env[0]) {
        if (std::strcmp(env, "all") == 0 || std::strcmp(env, "*") == 0) {
            cfg.allow_all = true;
            cfg.any = true;
        } else {
            char buf[512];
            std::snprintf(buf, sizeof(buf), "%s", env);
            char *save = nullptr;
            for (char *tok = ::strtok_r(buf, ",;:", &save); tok; tok = ::strtok_r(nullptr, ",;:", &save)) {
                while (*tok == ' ' || *tok == '\t') ++tok;
                if (!*tok) continue;
                char *endp = nullptr;
                long lo = std::strtol(tok, &endp, 10);
                long hi = lo;
                if (endp && *endp == '-') {
                    char *end2 = nullptr;
                    hi = std::strtol(endp + 1, &end2, 10);
                    endp = end2;
                }
                if (endp == tok || lo < 0 || hi < 0) continue;
                if (hi < lo) std::swap(lo, hi);
                lo = std::max<long>(0, std::min<long>(lo, 127));
                hi = std::max<long>(0, std::min<long>(hi, 127));
                for (long l = lo; l <= hi; ++l) {
                    cfg.allowed[l] = true;
                    cfg.any = true;
                }
            }
        }
    }

    const char *spec_env = std::getenv("GGML_MOE_VRAM_CACHE_LAYER_BIAS_SPEC");
    if (spec_env && spec_env[0]) {
        char buf[1024];
        std::snprintf(buf, sizeof(buf), "%s", spec_env);
        char *save = nullptr;
        for (char *tok = ::strtok_r(buf, ",;", &save); tok; tok = ::strtok_r(nullptr, ",;", &save)) {
            while (*tok == ' ' || *tok == '\t') ++tok;
            if (!*tok) continue;
            char *sep = std::strchr(tok, ':');
            if (!sep) sep = std::strchr(tok, '=');
            if (!sep) continue;
            *sep = '\0';
            char *tick_start = sep + 1;
            while (*tick_start == ' ' || *tick_start == '\t') ++tick_start;
            char *tick_end = nullptr;
            unsigned long long tick_value = std::strtoull(tick_start, &tick_end, 10);
            if (tick_end == tick_start || tick_value > 1000000000ULL) continue;

            char *endp = nullptr;
            long lo = std::strtol(tok, &endp, 10);
            long hi = lo;
            if (endp && *endp == '-') {
                char *end2 = nullptr;
                hi = std::strtol(endp + 1, &end2, 10);
                endp = end2;
            }
            if (endp == tok || lo < 0 || hi < 0) continue;
            if (hi < lo) std::swap(lo, hi);
            lo = std::max<long>(0, std::min<long>(lo, 127));
            hi = std::max<long>(0, std::min<long>(hi, 127));
            for (long l = lo; l <= hi; ++l) {
                cfg.allowed[l] = true;
                cfg.layer_ticks[l] = (size_t)tick_value;
                cfg.any = true;
            }
        }
    }

    cfg.dynamic_near = expert_pack_env_bool("GGML_MOE_VRAM_CACHE_DYNAMIC_NEAR_LAYER_BIAS", false);
    cfg.dynamic_window = (int)expert_pack_env_size(
            "GGML_MOE_VRAM_CACHE_DYNAMIC_NEAR_LAYER_WINDOW", 2, 0, 127);
    cfg.dynamic_ticks = expert_pack_env_size(
            "GGML_MOE_VRAM_CACHE_DYNAMIC_NEAR_LAYER_TICKS", 4096, 0, 1000000000ULL);

    if (cfg.any && (cfg.ticks > 0 || (spec_env && spec_env[0]))) {
        std::fprintf(stderr,
            "[moe_stream_batch] VRAM cache layer bias: layers=%s ticks=%zu spec=%s policy=lru-only\n",
            env ? env : "", cfg.ticks, spec_env ? spec_env : "");
    }
    if (cfg.dynamic_near && cfg.dynamic_ticks > 0) {
        std::fprintf(stderr,
            "[moe_stream_batch] VRAM cache dynamic near-layer bias: window=%d ticks=%zu policy=lru-only\n",
            cfg.dynamic_window, cfg.dynamic_ticks);
    }
    cfg.inited = true;
    return cfg;
}

static bool vram_cache_layer_bias_enabled_for(const vram_cache_layer_bias_config &cfg, int layer) {
    if (!cfg.any || layer < 0 || layer >= 128) return false;
    if (cfg.layer_ticks[layer] > 0) return true;
    if (cfg.ticks == 0) return false;
    return cfg.allow_all || cfg.allowed[layer];
}

static uint64_t vram_cache_add_lru_bias(uint64_t used, uint64_t bias) {
    if (bias == 0) return used;
    return used > UINT64_MAX - bias ? UINT64_MAX : used + bias;
}

static uint64_t vram_cache_layer_biased_lru_used(
        const batch_vram_cache *c, int slot, const vram_cache_layer_bias_config &cfg,
        int candidate_layer = -1) {
    const uint64_t used = c->slot_used[slot];
    const int layer = c->slot_layer[slot];
    uint64_t biased = used;
    if (vram_cache_layer_bias_enabled_for(cfg, layer)) {
        const uint64_t bias = (uint64_t)(cfg.layer_ticks[layer] > 0 ? cfg.layer_ticks[layer] : cfg.ticks);
        biased = vram_cache_add_lru_bias(biased, bias);
    }
    if (cfg.dynamic_near && cfg.dynamic_ticks > 0 &&
            candidate_layer >= 0 && layer >= 0 && layer < 128) {
        const int dist = layer - candidate_layer;
        if (dist >= 0 && dist <= cfg.dynamic_window) {
            const uint64_t scaled_bias =
                cfg.dynamic_window > 0 ?
                (uint64_t)((cfg.dynamic_ticks * (size_t)(cfg.dynamic_window + 1 - dist)) /
                    (size_t)(cfg.dynamic_window + 1)) :
                cfg.dynamic_ticks;
            biased = vram_cache_add_lru_bias(biased, scaled_bias);
        }
    }
    return biased;
}

struct vram_cache_layer_diag_state {
    std::atomic<bool> registered{false};
    std::atomic<uint64_t> inserts{0};
    std::atomic<uint64_t> replacements{0};
    std::atomic<uint64_t> candidate_unknown{0};
    std::atomic<uint64_t> victim_unknown{0};
    std::atomic<uint64_t> candidate_layer[128]{};
    std::atomic<uint64_t> victim_layer[128]{};
};

static vram_cache_layer_diag_state g_vram_cache_layer_diag;

static bool vram_cache_layer_diag_enabled() {
    return expert_pack_env_bool("GGML_MOE_VRAM_CACHE_LAYER_DIAG", false);
}

static void vram_cache_layer_diag_report() {
    if (!vram_cache_layer_diag_enabled()) return;
    const uint64_t inserts = g_vram_cache_layer_diag.inserts.load(std::memory_order_relaxed);
    const uint64_t replacements = g_vram_cache_layer_diag.replacements.load(std::memory_order_relaxed);
    std::fprintf(stderr,
        "[moe_stream_batch] vram cache layer diag: enabled=1 inserts=%lu replacements=%lu "
        "candidate_unknown=%lu victim_unknown=%lu\n",
        (unsigned long)inserts,
        (unsigned long)replacements,
        (unsigned long)g_vram_cache_layer_diag.candidate_unknown.load(std::memory_order_relaxed),
        (unsigned long)g_vram_cache_layer_diag.victim_unknown.load(std::memory_order_relaxed));
    for (int layer = 0; layer < 128; ++layer) {
        const uint64_t cand = g_vram_cache_layer_diag.candidate_layer[layer].load(std::memory_order_relaxed);
        const uint64_t vict = g_vram_cache_layer_diag.victim_layer[layer].load(std::memory_order_relaxed);
        if (cand == 0 && vict == 0) continue;
        std::fprintf(stderr,
            "[moe_stream_batch] vram cache layer diag row: layer=%d candidate_inserts=%lu victim_evictions=%lu\n",
            layer,
            (unsigned long)cand,
            (unsigned long)vict);
    }
}

static void vram_cache_layer_diag_register_once() {
    bool expected = false;
    if (g_vram_cache_layer_diag.registered.compare_exchange_strong(expected, true)) {
        std::atexit(vram_cache_layer_diag_report);
    }
}

static void vram_cache_layer_diag_record(int candidate_layer, int victim_layer, bool replacing) {
    if (!vram_cache_layer_diag_enabled()) return;
    vram_cache_layer_diag_register_once();
    ++g_vram_cache_layer_diag.inserts;
    if (candidate_layer >= 0 && candidate_layer < 128) {
        ++g_vram_cache_layer_diag.candidate_layer[candidate_layer];
    } else {
        ++g_vram_cache_layer_diag.candidate_unknown;
    }
    if (replacing) {
        ++g_vram_cache_layer_diag.replacements;
        if (victim_layer >= 0 && victim_layer < 128) {
            ++g_vram_cache_layer_diag.victim_layer[victim_layer];
        } else {
            ++g_vram_cache_layer_diag.victim_unknown;
        }
    }
}



struct q4_graph_slot_status {
    bool present = false;
    bool pending = false;
    bool ready = false;
    int slot = -1;
};

static bool q4_graph_descriptor_dry_run_enabled() {
    return expert_pack_env_bool("GGML_MOE_Q4_GRAPH_DESCRIPTOR_DRY_RUN", false);
}




struct q4_graph_readiness_io_snapshot {
    uint64_t scheduler_wait_us = 0;
    uint64_t scheduler_read_us = 0;
    uint64_t scheduler_h2d_us = 0;
    uint64_t iouring_batches = 0;
    uint64_t iouring_reads = 0;
    uint64_t iouring_bytes = 0;
};

struct q4_graph_readiness_provider_stats {
    uint64_t physical_jobs = 0;
    uint64_t role_slices = 0;
    uint64_t physical_bytes = 0;
    uint64_t logical_bytes = 0;
};

struct q4_graph_readiness_counts {
    uint64_t active_objects = 0;
    uint64_t active_roles = 0;
    uint64_t present_roles = 0;
    uint64_t pending_roles = 0;
    uint64_t ready_roles = 0;
    uint64_t vram_full_objects = 0;
    uint64_t vram_partial_objects = 0;
    uint64_t ram_ready_objects = 0;
    uint64_t ram_roles_to_h2d = 0;
    uint64_t ssd_objects = 0;
    uint64_t ssd_roles_to_read = 0;
};

struct q4_graph_readiness_csv_state {
    std::mutex mu;
    bool registered = false;
    bool header_written = false;
    uint64_t seq = 0;
};

static q4_graph_readiness_csv_state g_q4_graph_readiness_csv;















struct q4_scheduler_readiness_csv_state {
    std::mutex mu;
    bool header_written = false;
    uint64_t seq = 0;
};

static q4_scheduler_readiness_csv_state g_q4_scheduler_readiness_csv;























struct q4_object_reuse_trace_entry {
    uint64_t last_seq = 0;
    uint64_t count = 0;
};

struct q4_object_reuse_trace_state {
    std::mutex mu;
    bool header_written = false;
    uint64_t seq = 0;
    std::unordered_map<std::string, q4_object_reuse_trace_entry> objects;
};

static q4_object_reuse_trace_state g_q4_object_reuse_trace;







struct q4_object_tier_trace_state {
    std::mutex mu;
    bool header_written = false;
    uint64_t seq = 0;
};

static q4_object_tier_trace_state g_q4_object_tier_trace;





static uint64_t q4_colibri_host_object_score(
        const q4_colibri_object_marker &marker,
        int current_layer,
        uint32_t freq);



struct q4_object_tier_trace_ram_slot_snapshot {
    bool present = false;
    bool ready = false;
    bool loading = false;
    bool delete_after_h2d = false;
    uint64_t score = 0;
    uint32_t freq = 0;
    uint64_t used = 0;
};











static bool q4_graph_release_descriptor(
        batch_vram_cache *cache,
        uintptr_t key,
        const ggml_moe_expert_descriptor &descriptor,
        cudaStream_t stream,
        bool record_completion) {
    if (!cache) return false;
    bool last_lease = false;
    if (!ggml_moe_common::role_cache_release_descriptor(
            batch_cache_role_view(cache), key, descriptor, &last_lease)) {
        ++cache->descriptor_stale_releases;
        return false;
    }
    if (record_completion && last_lease &&
            !batch_cache_record_slot_lease(cache, descriptor.slot_id, stream)) {
        if (stream) cudaStreamSynchronize(stream);
        return false;
    }
    ++cache->descriptor_releases;
    return true;
}



















static bool q4_graph_descriptor_buffer_init() {
    if (!q4_graph_descriptor_dry_run_enabled()) return true;
    const size_t capacity = (size_t)q4_graph_descriptor_max_experts() * 3;
    if (g_batch.q4_graph_descriptor_buffer.device) {
        return g_batch.q4_graph_descriptor_buffer.capacity >= capacity;
    }
    if (!ggml_moe_common::device_descriptor_buffer_allocate_fixed(
            g_batch.q4_graph_descriptor_buffer, capacity)) {
        ++g_q4_graph_hit_profile.descriptor_dry_run_failures;
        return false;
    }
    return true;
}

























































struct q4_colibri_layer_set_config {
    bool enabled = false;
    bool allowed[128] = {};
};













struct q4_colibri_host_object_layer_quota_config {
    bool enabled = false;
    size_t quota[128] = {};
};

static const struct q4_colibri_host_object_layer_quota_config & q4_colibri_host_object_layer_quota_config() {
    static struct q4_colibri_host_object_layer_quota_config cfg = []() {
        struct q4_colibri_host_object_layer_quota_config out;
        const char *env = std::getenv("GGML_MOE_Q4_HOST_OBJECT_LAYER_QUOTAS");
        if (!env || !env[0]) return out;
        char buf[4096];
        std::snprintf(buf, sizeof(buf), "%s", env);
        char *save = nullptr;
        for (char *tok = ::strtok_r(buf, ",; \t\r\n", &save); tok; tok = ::strtok_r(nullptr, ",; \t\r\n", &save)) {
            while (*tok == ' ' || *tok == '\t') ++tok;
            if (!*tok) continue;
            char *sep = std::strchr(tok, ':');
            if (!sep) sep = std::strchr(tok, '=');
            if (!sep) continue;
            *sep = '\0';
            char *end_layer = nullptr;
            char *end_quota = nullptr;
            const long layer = std::strtol(tok, &end_layer, 10);
            const long quota = std::strtol(sep + 1, &end_quota, 10);
            if (end_layer == tok || end_quota == sep + 1 || layer < 0 || layer >= 128 || quota <= 0) {
                continue;
            }
            out.quota[layer] = (size_t)quota;
            out.enabled = true;
        }
        return out;
    }();
    return cfg;
}

































static bool q4_colibri_flow_score_cache_enabled() {
    return expert_pack_env_bool("GGML_MOE_Q4_FLOW_SCORE_CACHE", false);
}







static int q4_colibri_host_object_score_window() {
    return (int)expert_pack_env_size("GGML_MOE_Q4_HOST_OBJECT_SCORE_WINDOW", 2, 0, 128);
}



















static bool q4_colibri_flow_score_dynamic_hotness_enabled() {
    return expert_pack_env_bool("GGML_MOE_Q4_FLOW_SCORE_DYNAMIC_HOTNESS", true);
}





static bool q4_colibri_host_object_score_layer_hint_enabled(int layer) {
    static bool inited = false;
    static bool allow_all = false;
    static bool allowed[128] = {};
    if (!inited) {
        const char *env = std::getenv("GGML_MOE_Q4_HOST_OBJECT_SCORE_LAYER_HINTS");
        if (env && env[0]) {
            if (std::strcmp(env, "all") == 0 || std::strcmp(env, "*") == 0) {
                allow_all = true;
            } else {
                char buf[512];
                std::snprintf(buf, sizeof(buf), "%s", env);
                char *save = nullptr;
                for (char *tok = ::strtok_r(buf, ",;:", &save); tok; tok = ::strtok_r(nullptr, ",;:", &save)) {
                    while (*tok == ' ' || *tok == '\t') ++tok;
                    if (!*tok) continue;
                    char *endp = nullptr;
                    long lo = std::strtol(tok, &endp, 10);
                    long hi = lo;
                    if (endp && *endp == '-') {
                        char *end2 = nullptr;
                        hi = std::strtol(endp + 1, &end2, 10);
                        endp = end2;
                    }
                    if (endp == tok || lo < 0 || hi < 0) continue;
                    if (hi < lo) std::swap(lo, hi);
                    lo = std::max<long>(0, std::min<long>(lo, 127));
                    hi = std::max<long>(0, std::min<long>(hi, 127));
                    for (long l = lo; l <= hi; ++l) {
                        allowed[l] = true;
                    }
                }
            }
        }
        inited = true;
    }
    if (layer < 0 || layer >= 128) return false;
    return allow_all || allowed[layer];
}

static uint64_t q4_colibri_host_object_score_layer_bonus() {
    return expert_pack_env_size("GGML_MOE_Q4_HOST_OBJECT_SCORE_LAYER_BONUS", 0, 0, 1000000000ULL);
}

struct q4_colibri_layer_score_bonus_config {
    bool enabled = false;
    uint64_t bonus[128] = {};
};

static const q4_colibri_layer_score_bonus_config & q4_colibri_layer_score_bonus_config_get() {
    static const q4_colibri_layer_score_bonus_config cfg = []() {
        q4_colibri_layer_score_bonus_config out;
        const char *env = std::getenv("GGML_MOE_Q4_HOST_OBJECT_LAYER_SCORE_BONUS");
        if (!env || !env[0]) return out;
        char buf[4096];
        std::snprintf(buf, sizeof(buf), "%s", env);
        char *save = nullptr;
        for (char *tok = ::strtok_r(buf, ",; \t\r\n", &save);
                tok;
                tok = ::strtok_r(nullptr, ",; \t\r\n", &save)) {
            while (*tok == ' ' || *tok == '\t') ++tok;
            if (!*tok) continue;
            char *sep = std::strchr(tok, ':');
            if (!sep) sep = std::strchr(tok, '=');
            if (!sep) continue;
            *sep = '\0';
            char *end_layer = nullptr;
            char *end_bonus = nullptr;
            const long layer = std::strtol(tok, &end_layer, 10);
            const unsigned long long parsed_bonus = std::strtoull(sep + 1, &end_bonus, 10);
            if (end_layer == tok || end_bonus == sep + 1 || layer < 0 || layer >= 128) {
                continue;
            }
            out.bonus[layer] = std::min<unsigned long long>(parsed_bonus, 1000000000ULL);
            out.enabled = true;
        }
        return out;
    }();
    return cfg;
}



static uint64_t q4_colibri_layer_score_bonus(int layer) {
    if (layer < 0 || layer >= 128) return 0;
    return q4_colibri_layer_score_bonus_config_get().bonus[layer];
}

static uint64_t q4_colibri_host_object_static_hotness_key(int layer, int expert) {
    return ((uint64_t)(uint32_t)layer << 32) | (uint32_t)expert;
}

struct q4_colibri_static_hotness_state {
    bool enabled = false;
    std::unordered_map<uint64_t, uint64_t> scores;
};

static const q4_colibri_static_hotness_state & q4_colibri_static_hotness_config() {
    static const q4_colibri_static_hotness_state *cfg = []() {
        q4_colibri_static_hotness_state *out = new q4_colibri_static_hotness_state();
        const char *path = std::getenv("GGML_MOE_Q4_HOST_OBJECT_STATIC_HOTNESS_FILE");
        if (!path || !path[0]) return out;
        std::ifstream in(path);
        if (!in.good()) return out;

        const size_t max_entries = expert_pack_env_size(
                "GGML_MOE_Q4_HOST_OBJECT_STATIC_HOTNESS_MAX_ENTRIES", 4096, 1, 1048576);
        const uint64_t max_bonus = expert_pack_env_size(
                "GGML_MOE_Q4_HOST_OBJECT_STATIC_HOTNESS_MAX_BONUS", 250000, 1, 1000000000ULL);
        const uint64_t weight = expert_pack_env_size(
                "GGML_MOE_Q4_HOST_OBJECT_STATIC_HOTNESS_WEIGHT", 1, 0, 1000000ULL);

        std::string line;
        size_t rank = 0;
        while (rank < max_entries && std::getline(in, line)) {
            const char *p = line.c_str();
            while (*p == ' ' || *p == '\t') ++p;
            if (!*p || *p == '#') continue;

            char *end_layer = nullptr;
            const long layer = std::strtol(p, &end_layer, 10);
            if (end_layer == p || layer < 0 || layer >= 128) continue;
            while (*end_layer == ' ' || *end_layer == '\t' || *end_layer == ':' || *end_layer == ',' || *end_layer == ';') {
                ++end_layer;
            }
            char *end_expert = nullptr;
            const long expert = std::strtol(end_layer, &end_expert, 10);
            if (end_expert == end_layer || expert < 0 || expert >= 4096) continue;

            ++rank;
            uint64_t parsed_score = 0;
            const size_t score_pos = line.find("score=");
            if (score_pos != std::string::npos) {
                const char *score_text = line.c_str() + score_pos + 6;
                char *end_score = nullptr;
                const unsigned long long value = std::strtoull(score_text, &end_score, 10);
                if (end_score != score_text) {
                    parsed_score = (uint64_t)value;
                }
            }
            if (parsed_score == 0) {
                parsed_score = (uint64_t)(max_entries - rank + 1);
            }
            uint64_t bonus = parsed_score;
            if (weight > 0 && bonus > UINT64_MAX / weight) {
                bonus = UINT64_MAX;
            } else {
                bonus *= weight;
            }
            bonus = std::min<uint64_t>(bonus, max_bonus);
            out->scores[q4_colibri_host_object_static_hotness_key((int)layer, (int)expert)] = bonus;
        }
        out->enabled = !out->scores.empty();
        return out;
    }();
    return *cfg;
}

static bool q4_colibri_static_hotness_enabled() {
    return q4_colibri_static_hotness_config().enabled;
}



static uint64_t q4_colibri_static_hotness_bonus_raw(const q4_colibri_object_marker &marker) {
    if (marker.layer_idx < 0 || marker.expert_idx < 0) return 0;
    const q4_colibri_static_hotness_state &cfg = q4_colibri_static_hotness_config();
    if (!cfg.enabled) return 0;
    const auto it = cfg.scores.find(q4_colibri_host_object_static_hotness_key(marker.layer_idx, marker.expert_idx));
    if (it == cfg.scores.end() || it->second == 0) return 0;
    return it->second;
}

static uint64_t q4_colibri_static_hotness_bonus(const q4_colibri_object_marker &marker) {
    const uint64_t bonus = q4_colibri_static_hotness_bonus_raw(marker);
    if (bonus == 0) return 0;
    ++g_q4_host_object_prefetch.static_hotness_score_hits;
    g_q4_host_object_prefetch.static_hotness_score_bonus.fetch_add(bonus, std::memory_order_relaxed);
    return bonus;
}

static bool q4_object_continuous_score_enabled() {
    return expert_pack_env_bool("GGML_MOE_Q4_OBJECT_CONTINUOUS_SCORE", false) ||
        expert_pack_env_bool("GGML_MOE_Q4_FULL_DISTANCE_SCORE", false) ||
        q4_colibri_flow_score_cache_enabled() ||
        q4_vram_unified_score_enabled();
}

static bool q4_full_distance_score_enabled() {
    return expert_pack_env_bool("GGML_MOE_Q4_FULL_DISTANCE_SCORE", false);
}

static bool q4_score_vram_layer_bias_env_hint_enabled(int layer) {
    static bool inited = false;
    static bool allow_all = false;
    static bool allowed[128] = {};
    if (!inited) {
        const char *env = std::getenv("GGML_MOE_VRAM_CACHE_LAYER_BIAS_LAYERS");
        if (env && env[0]) {
            if (std::strcmp(env, "all") == 0 || std::strcmp(env, "*") == 0) {
                allow_all = true;
            } else {
                char buf[512];
                std::snprintf(buf, sizeof(buf), "%s", env);
                char *save = nullptr;
                for (char *tok = ::strtok_r(buf, ",;:", &save); tok; tok = ::strtok_r(nullptr, ",;:", &save)) {
                    while (*tok == ' ' || *tok == '\t') ++tok;
                    if (!*tok) continue;
                    char *endp = nullptr;
                    long lo = std::strtol(tok, &endp, 10);
                    long hi = lo;
                    if (endp && *endp == '-') {
                        char *end2 = nullptr;
                        hi = std::strtol(endp + 1, &end2, 10);
                        endp = end2;
                    }
                    if (endp == tok || lo < 0 || hi < 0) continue;
                    if (hi < lo) std::swap(lo, hi);
                    lo = std::max<long>(0, std::min<long>(lo, 127));
                    hi = std::max<long>(0, std::min<long>(hi, 127));
                    for (long l = lo; l <= hi; ++l) {
                        allowed[l] = true;
                    }
                }
            }
        }
        inited = true;
    }
    if (layer < 0 || layer >= 128) return false;
    return allow_all || allowed[layer];
}

static uint64_t q4_colibri_host_object_score(
        const q4_colibri_object_marker &marker,
        int current_layer,
        uint32_t freq) {
    const uint64_t freq_weight = expert_pack_env_size(
            "GGML_MOE_Q4_HOST_OBJECT_SCORE_FREQ_WEIGHT", 10000, 1, 1000000);
    if (!q4_object_continuous_score_enabled()) {
        const uint64_t near_weight = expert_pack_env_size(
                "GGML_MOE_Q4_HOST_OBJECT_SCORE_NEAR_WEIGHT", 1000, 0, 1000000);
        const uint64_t past_penalty = expert_pack_env_size(
                "GGML_MOE_Q4_HOST_OBJECT_SCORE_PAST_PENALTY", 2000, 0, 1000000);
        int64_t legacy_score = (int64_t)freq * (int64_t)freq_weight;
        if (current_layer >= 0 && marker.layer_idx >= 0) {
            const int dist = marker.layer_idx - current_layer;
            const int window = q4_colibri_host_object_score_window();
            if (dist >= 0 && dist <= window) {
                legacy_score += (int64_t)(window + 1 - dist) * (int64_t)near_weight;
            } else if (dist < 0) {
                legacy_score -= (int64_t)(-dist) * (int64_t)past_penalty;
            }
        }
        if (q4_colibri_host_object_score_layer_hint_enabled(marker.layer_idx)) {
            legacy_score += (int64_t)q4_colibri_host_object_score_layer_bonus();
        }
        legacy_score += (int64_t)q4_colibri_layer_score_bonus(marker.layer_idx);
        legacy_score += (int64_t)q4_colibri_static_hotness_bonus(marker);
        return legacy_score > 0 ? (uint64_t)legacy_score : 0;
    }

    const uint64_t future_weight = expert_pack_env_size(
            "GGML_MOE_Q4_OBJECT_SCORE_FUTURE_WEIGHT", 120000, 0, 1000000000ULL);
    const uint64_t next_token_weight = expert_pack_env_size(
            "GGML_MOE_Q4_OBJECT_SCORE_NEXT_TOKEN_WEIGHT", 16000, 0, 1000000000ULL);
    const uint64_t current_layer_bonus = expert_pack_env_size(
            "GGML_MOE_Q4_OBJECT_SCORE_CURRENT_BONUS", 120000, 0, 1000000000ULL);
    const uint64_t vram_bias_bonus = expert_pack_env_size(
            "GGML_MOE_Q4_OBJECT_SCORE_VRAM_BIAS_BONUS", 250000, 0, 1000000000ULL);
    const int max_layers = (int)expert_pack_env_size(
            "GGML_MOE_Q4_OBJECT_SCORE_MAX_LAYERS", 64, 1, 1024);
    const bool flow_score = q4_colibri_flow_score_cache_enabled();
    const bool full_distance_score = q4_full_distance_score_enabled();
    const bool dynamic_hotness = !flow_score || q4_colibri_flow_score_dynamic_hotness_enabled();
    int64_t score = dynamic_hotness ? (int64_t)freq * (int64_t)freq_weight : 0;
    if (current_layer >= 0 && marker.layer_idx >= 0) {
        const int dist = marker.layer_idx - current_layer;
        if (full_distance_score) {
            const uint64_t past_penalty = expert_pack_env_size(
                    "GGML_MOE_Q4_FULL_DISTANCE_PAST_PENALTY", 2000, 0, 1000000000ULL);
            if (dist == 0) {
                score += (int64_t)current_layer_bonus;
                score += (int64_t)future_weight;
            } else if (dist > 0) {
                const int clamped = std::min(dist, max_layers);
                const uint64_t numer = (uint64_t)std::max(1, max_layers + 1 - clamped);
                score += (int64_t)(future_weight * numer / (uint64_t)(max_layers + 1));
            } else {
                score -= (int64_t)(uint64_t)(-dist) * (int64_t)past_penalty;
            }
        } else if (flow_score) {
            const int layer_window = (int)expert_pack_env_size(
                    "GGML_MOE_Q4_FLOW_SCORE_LAYER_WINDOW", 8, 1, 128);
            const uint64_t past_penalty = expert_pack_env_size(
                    "GGML_MOE_Q4_FLOW_SCORE_PAST_PENALTY", 2000, 0, 1000000000ULL);
            if (dist == 0) {
                score += (int64_t)current_layer_bonus;
                score += (int64_t)future_weight;
            } else if (dist > 0 && dist <= layer_window) {
                score += (int64_t)(
                        future_weight * (uint64_t)(layer_window + 1 - dist) /
                        (uint64_t)(layer_window + 1));
            } else if (dist > 0) {
                score += (int64_t)(future_weight / (uint64_t)(dist + layer_window + 1));
            } else {
                score -= (int64_t)(uint64_t)(-dist) * (int64_t)past_penalty;
            }
        } else if (dist == 0) {
            score += (int64_t)current_layer_bonus;
            score += (int64_t)future_weight;
        } else if (dist > 0) {
            score += (int64_t)(future_weight / (uint64_t)(1 + dist));
        } else {
            const int wrapped_dist = std::max(1, max_layers + dist);
            score += (int64_t)(next_token_weight / (uint64_t)(1 + wrapped_dist));
        }
    }
    if (q4_colibri_host_object_score_layer_hint_enabled(marker.layer_idx)) {
        score += (int64_t)q4_colibri_host_object_score_layer_bonus();
    }
    if (q4_score_vram_layer_bias_env_hint_enabled(marker.layer_idx)) {
        score += (int64_t)vram_bias_bonus;
    }
    score += (int64_t)q4_colibri_layer_score_bonus(marker.layer_idx);
    score += (int64_t)q4_colibri_static_hotness_bonus(marker);
    return score > 0 ? (uint64_t)score : 0;
}



























static bool q4_object_score_profile_enabled() {
    return expert_pack_env_bool("GGML_MOE_Q4_OBJECT_SCORE_PROFILE", false);
}

static void q4_score_profile_add_layer(
        std::atomic<uint64_t> (&counters)[128],
        int layer,
        uint64_t value = 1) {
    if (value == 0 || layer < 0 || layer >= 128) return;
    counters[layer].fetch_add(value, std::memory_order_relaxed);
}



static void q4_score_profile_record_vram_eviction(
        int candidate_layer,
        uint64_t candidate_score,
        uint64_t victim_score) {
    if (!q4_object_score_profile_enabled()) return;
    ++g_q4_tier_overlap_diag.score_profile_vram_evictions;
    g_q4_tier_overlap_diag.score_profile_vram_candidate_score_sum.fetch_add(
            candidate_score, std::memory_order_relaxed);
    g_q4_tier_overlap_diag.score_profile_vram_victim_score_sum.fetch_add(
            victim_score, std::memory_order_relaxed);
    q4_score_profile_add_layer(g_q4_tier_overlap_diag.score_profile_layer_vram_evictions, candidate_layer);
    q4_score_profile_add_layer(
            g_q4_tier_overlap_diag.score_profile_layer_vram_candidate_score_sum,
            candidate_layer,
            candidate_score);
    q4_score_profile_add_layer(
            g_q4_tier_overlap_diag.score_profile_layer_vram_victim_score_sum,
            candidate_layer,
            victim_score);
    if (candidate_score > victim_score) {
        ++g_q4_tier_overlap_diag.score_profile_vram_candidate_gt_victim;
        q4_score_profile_add_layer(g_q4_tier_overlap_diag.score_profile_layer_vram_candidate_gt_victim, candidate_layer);
    } else if (candidate_score == victim_score) {
        ++g_q4_tier_overlap_diag.score_profile_vram_candidate_eq_victim;
        q4_score_profile_add_layer(g_q4_tier_overlap_diag.score_profile_layer_vram_candidate_eq_victim, candidate_layer);
    } else {
        ++g_q4_tier_overlap_diag.score_profile_vram_candidate_lt_victim;
        q4_score_profile_add_layer(g_q4_tier_overlap_diag.score_profile_layer_vram_candidate_lt_victim, candidate_layer);
    }
}
















static bool q4_direct_host_object_copy_h2d(
        const expert_pack_entry *pack_entry,
        void *dst,
        size_t role_bytes,
        cudaStream_t st) {
    return false;
}

static void q4_direct_host_object_submit_on_read(
        const expert_pack_entry *pack_entry,
        uint64_t read_us) {
    return;
}











static bool q4_colibri_host_object_demote_from_vram(
        const char *victim_tensor,
        int victim_layer,
        int victim_expert,
        uint32_t victim_hits,
        int current_layer) {
    (void)q4_tier_demote_vram_evict_enabled();
    return false;
}

































































































static bool q4_colibri_pipe_copy(
        batch_vram_cache *cache,
        int slot,
        uint64_t generation,
        uintptr_t key,
        void *dst,
        const expert_pack_entry *entry,
        size_t bytes,
        cudaStream_t consumer_stream,
        const char *tensor_name,
        int expert_idx) {
    return false;
}





template <typename Job>
static bool expert_pack_iouring_job_eligible(const Job &job, size_t expert_bytes) {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING) && !defined(_WIN32)
    if (g_expert_pack.io_backend != 2 || !job.pack_entry ||
            job.pack_entry->nbytes != expert_bytes) {
        return false;
    }
    const expert_pack_source *source = expert_pack_source_for_entry(job.pack_entry);
    return source && !source->page_cache && source->fd_direct >= 0;
#else
    (void)job;
    (void)expert_bytes;
    return false;
#endif
}


template <typename Job, typename JobSizeFn, typename JobStreamFn>
static bool expert_pack_iouring_copy_jobs_impl(
        const std::vector<Job> &jobs,
        size_t expert_bytes_hint,
        JobSizeFn job_size,
        JobStreamFn job_stream,
        cudaStream_t st,
        pinned_stage_ring &ring,
        const char *trace_op,
        bool cache_bypass = false,
        size_t refill_batch_override = 0) {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING) && !defined(_WIN32)
    if (jobs.empty()) return true;
    if (g_expert_pack.io_backend != 2) return false;

    const size_t alignment = expert_pack_direct_alignment();
    const bool aligned_alias_batch = expert_pack_env_bool("GGML_MOE_IO_ALIGNED_ALIAS_BATCH", false);
    const bool fixed_job_bytes = expert_bytes_hint != 0;
    const size_t default_read_sz = fixed_job_bytes ?
        (size_t)align_up_u64((uint64_t)expert_bytes_hint, (uint64_t)alignment) : 0;
    if (fixed_job_bytes && default_read_sz == 0) return false;
    const bool profile_io_batch = io_batch_profile_enabled();
    const auto io_batch_start = profile_io_batch ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    struct iouring_read_plan {
        size_t job_idx = 0;
        uint64_t offset = 0;
        size_t read_sz = 0;
        size_t prefix = 0;
        size_t payload_sz = 0;
        cudaStream_t copy_stream = nullptr;
        int32_t source_idx = 0;
        const char *tensor = nullptr;
    };

    struct iouring_group_slice {
        size_t plan_idx = 0;
        size_t payload_offset = 0;
    };

    struct iouring_group_plan {
        int32_t source_idx = 0;
        uint64_t offset = 0;
        size_t read_sz = 0;
        size_t expect_sz = 0;
        cudaStream_t copy_stream = nullptr;
        std::vector<iouring_group_slice> slices;
    };

    std::vector<iouring_read_plan> read_plans;
    read_plans.reserve(jobs.size());
    size_t max_read_sz = default_read_sz;
    size_t ram_or_prefetch_jobs = 0;
    size_t ram_or_prefetch_bytes = 0;
    size_t ram_hit_jobs = 0;
    size_t ram_hit_bytes = 0;
    double ram_hit_enqueue_ms = 0.0;
    const bool profile_ram_batch = ram_batch_profile_enabled();
    const auto ram_batch_start = profile_ram_batch ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    for (size_t i = 0; i < jobs.size(); ++i) {
        const Job &job = jobs[i];
        const size_t job_bytes = job_size(job);
        cudaStream_t copy_stream = job_stream(job);
        if (!copy_stream) copy_stream = st;
        if (!job.pack_entry ||
                job_bytes == 0 ||
                job.pack_entry->nbytes != job_bytes) {
            return false;
        }
        const expert_pack_source *source = expert_pack_source_for_entry(job.pack_entry);
        if (!source || source->fd_direct < 0) {
            return false;
        }
        batch_copy_trace copy_trace;
        copy_trace.pack_hit = true;
        const auto copy_start = batch_ttft_trace_enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const auto ram_enqueue_start = profile_ram_batch ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (!cache_bypass && expert_pack_ram_tier_copy_h2d(job.pack_entry, job.dst, job_bytes, copy_stream, &copy_trace)) {
            ++ram_or_prefetch_jobs;
            ram_or_prefetch_bytes += job_bytes;
            ++ram_hit_jobs;
            ram_hit_bytes += job_bytes;
            if (profile_ram_batch) {
                const auto ram_enqueue_end = std::chrono::steady_clock::now();
                ram_hit_enqueue_ms += std::chrono::duration<double, std::milli>(ram_enqueue_end - ram_enqueue_start).count();
            }
            if (copy_start != std::chrono::steady_clock::time_point{}) {
                const auto copy_end = std::chrono::steady_clock::now();
                batch_ttft_trace_record(
                    trace_op,
                    job.tensor,
                    job.expert_idx,
                    job_bytes,
                    false,
                    copy_trace.pack_hit,
                    copy_trace.ram_hit,
                    std::chrono::duration<double, std::milli>(copy_end - copy_start).count());
            }
            continue;
        }
        if (!cache_bypass && host_prefetch_copy_h2d(job.pack_entry, job.tensor, job.expert_idx, job.dst, job_bytes, st)) {
            ++ram_or_prefetch_jobs;
            ram_or_prefetch_bytes += job_bytes;
            copy_trace.pack_hit = true;
            copy_trace.ram_hit = false;
            if (copy_start != std::chrono::steady_clock::time_point{}) {
                const auto copy_end = std::chrono::steady_clock::now();
                batch_ttft_trace_record(
                    trace_op,
                    job.tensor,
                    job.expert_idx,
                    job_bytes,
                    false,
                    copy_trace.pack_hit,
                    copy_trace.ram_hit,
                    std::chrono::duration<double, std::milli>(copy_end - copy_start).count());
            }
            continue;
        }

        uint64_t read_offset = job.pack_entry->offset;
        size_t read_sz = fixed_job_bytes ?
            default_read_sz : (size_t)align_up_u64((uint64_t)job_bytes, (uint64_t)alignment);
        size_t prefix = 0;
        if ((job.pack_entry->offset % alignment) != 0) {
            if (!aligned_alias_batch) {
                return false;
            }
            read_offset = (job.pack_entry->offset / alignment) * alignment;
            prefix = (size_t)(job.pack_entry->offset - read_offset);
            read_sz = (size_t)align_up_u64((uint64_t)prefix + (uint64_t)job_bytes, (uint64_t)alignment);
        }
        if (read_sz < job_bytes || read_sz > (size_t)UINT_MAX) {
            return false;
        }
        max_read_sz = std::max(max_read_sz, read_sz);
        read_plans.push_back({i, read_offset, read_sz, prefix, job_bytes, copy_stream, job.pack_entry->source_idx, job.tensor});
    }
    const size_t adjacent_op_bucket = expert_pack_adjacent_op_bucket(trace_op);
    const bool coalesce_adjacent = expert_pack_adjacent_coalesce_enabled() &&
        expert_pack_adjacent_op_allowed(trace_op) && read_plans.size() > 1;
    const bool sort_by_offset = expert_pack_env_bool("GGML_MOE_IO_SORT_OFFSET", false) || coalesce_adjacent;
    std::vector<size_t> plan_order;
    if (sort_by_offset && read_plans.size() > 1) {
        plan_order.resize(read_plans.size());
        for (size_t i = 0; i < read_plans.size(); ++i) {
            plan_order[i] = i;
        }
        std::stable_sort(plan_order.begin(), plan_order.end(),
            [&](size_t a, size_t b) {
                const expert_pack_entry *ea = jobs[read_plans[a].job_idx].pack_entry;
                const expert_pack_entry *eb = jobs[read_plans[b].job_idx].pack_entry;
                if (ea->source_idx != eb->source_idx) return ea->source_idx < eb->source_idx;
                return read_plans[a].offset < read_plans[b].offset;
            });
    }
    auto ordered_plan_idx = [&](size_t seq_idx) -> size_t {
        return plan_order.empty() ? seq_idx : plan_order[seq_idx];
    };

    std::vector<iouring_group_plan> group_plans;
    group_plans.reserve(read_plans.size());
    const size_t adjacent_max_span = expert_pack_adjacent_max_span();
    const size_t adjacent_max_gap = expert_pack_adjacent_max_gap();
    const size_t adjacent_min_group = expert_pack_adjacent_min_group();
    size_t adjacent_groups = 0;
    size_t adjacent_slices = 0;
    size_t adjacent_extents_saved = 0;
    uint64_t adjacent_physical_bytes = 0;
    uint64_t adjacent_payload_bytes = 0;
    size_t adjacent_max_group_jobs = 0;
    size_t adjacent_max_span_bytes = 0;
    uint64_t adjacent_reject_source = 0;
    uint64_t adjacent_reject_tensor = 0;
    uint64_t adjacent_reject_gap = 0;
    uint64_t adjacent_reject_span = 0;

    auto append_single_group = [&](size_t plan_idx) {
        const iouring_read_plan &plan = read_plans[plan_idx];
        iouring_group_plan group;
        group.source_idx = plan.source_idx;
        group.offset = plan.offset;
        group.read_sz = plan.read_sz;
        group.copy_stream = plan.copy_stream;
        group.slices.push_back({plan_idx, plan.prefix});
        group_plans.push_back(std::move(group));
    };

    if (coalesce_adjacent) {
        size_t seq = 0;
        while (seq < read_plans.size()) {
            const size_t first_idx = ordered_plan_idx(seq);
            const iouring_read_plan &first = read_plans[first_idx];
            iouring_group_plan group;
            group.source_idx = first.source_idx;
            group.offset = first.offset;
            group.read_sz = first.read_sz;
            group.copy_stream = first.copy_stream;
            group.slices.push_back({first_idx, first.prefix});
            uint64_t group_end = first.offset + first.read_sz;
            size_t next_seq = seq + 1;
            while (next_seq < read_plans.size()) {
                const size_t next_idx = ordered_plan_idx(next_seq);
                const iouring_read_plan &next = read_plans[next_idx];
                if (next.source_idx != group.source_idx) {
                    ++adjacent_reject_source;
                    break;
                }
                if (std::strcmp(next.tensor, first.tensor) != 0) {
                    ++adjacent_reject_tensor;
                    break;
                }
                if (next.copy_stream != group.copy_stream) {
                    break;
                }
                if (next.offset < group_end || next.offset - group_end > adjacent_max_gap) {
                    ++adjacent_reject_gap;
                    break;
                }
                const uint64_t next_end = next.offset + next.read_sz;
                if (next_end < group.offset || next_end - group.offset > adjacent_max_span) {
                    ++adjacent_reject_span;
                    break;
                }
                group.slices.push_back({next_idx, (size_t)(next.offset - group.offset) + next.prefix});
                group_end = next_end;
                group.read_sz = (size_t)(group_end - group.offset);
                ++next_seq;
            }
            if (group.slices.size() >= adjacent_min_group) {
                max_read_sz = std::max(max_read_sz, group.read_sz);
                adjacent_groups += 1;
                adjacent_slices += group.slices.size();
                adjacent_extents_saved += group.slices.size() - 1;
                adjacent_physical_bytes += group.read_sz;
                for (const iouring_group_slice &payload_slice : group.slices) {
                    adjacent_payload_bytes += read_plans[payload_slice.plan_idx].payload_sz;
                }
                adjacent_max_group_jobs = std::max(adjacent_max_group_jobs, group.slices.size());
                adjacent_max_span_bytes = std::max(adjacent_max_span_bytes, group.read_sz);
                group_plans.push_back(std::move(group));
            } else {
                for (const iouring_group_slice &slice : group.slices) {
                    append_single_group(slice.plan_idx);
                }
            }
            seq = next_seq;
        }
    } else {
        for (size_t seq = 0; seq < read_plans.size(); ++seq) {
            append_single_group(ordered_plan_idx(seq));
        }
    }
    if (adjacent_groups > 0) {
        g_expert_pack.adjacent_coalesce_groups.fetch_add(adjacent_groups);
        g_expert_pack.adjacent_coalesce_slices.fetch_add(adjacent_slices);
        g_expert_pack.adjacent_coalesce_extents_saved.fetch_add(adjacent_extents_saved);
        g_expert_pack.adjacent_coalesce_physical_bytes.fetch_add(adjacent_physical_bytes);
        g_expert_pack.adjacent_coalesce_payload_bytes.fetch_add(adjacent_payload_bytes);
        expert_pack_atomic_max(g_expert_pack.adjacent_coalesce_max_group_jobs, adjacent_max_group_jobs);
        expert_pack_atomic_max(g_expert_pack.adjacent_coalesce_max_span_bytes, adjacent_max_span_bytes);
        expert_pack_adjacent_op_stats &op_stats = g_expert_pack.adjacent_coalesce_op[adjacent_op_bucket];
        op_stats.groups.fetch_add(adjacent_groups);
        op_stats.slices.fetch_add(adjacent_slices);
        op_stats.extents_saved.fetch_add(adjacent_extents_saved);
        op_stats.physical_bytes.fetch_add(adjacent_physical_bytes);
        op_stats.payload_bytes.fetch_add(adjacent_payload_bytes);
        expert_pack_atomic_max(op_stats.max_group_jobs, adjacent_max_group_jobs);
        expert_pack_atomic_max(op_stats.max_span_bytes, adjacent_max_span_bytes);
    }
    if (coalesce_adjacent) {
        g_expert_pack.adjacent_coalesce_reject_source.fetch_add(adjacent_reject_source);
        g_expert_pack.adjacent_coalesce_reject_tensor.fetch_add(adjacent_reject_tensor);
        g_expert_pack.adjacent_coalesce_reject_gap.fetch_add(adjacent_reject_gap);
        g_expert_pack.adjacent_coalesce_reject_span.fetch_add(adjacent_reject_span);
    }
    if (read_plans.empty()) {
        if (profile_io_batch) {
            const auto io_batch_end = std::chrono::steady_clock::now();
            io_batch_profile_record(
                    trace_op,
                    jobs.empty() ? "" : jobs.front().tensor,
                    jobs.empty() ? "" : jobs.back().tensor,
                    jobs.size(), read_plans.size(), ram_or_prefetch_jobs, ram_or_prefetch_bytes,
                    0, ring.slots.size(), 0, 0,
                    0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0,
                    std::chrono::duration<double, std::milli>(io_batch_end - io_batch_start).count(),
                    sort_by_offset);
        }
        if (ram_hit_jobs > 0) {
            ++g_expert_pack.ram_tier_batch_calls;
            g_expert_pack.ram_tier_batch_jobs.fetch_add(jobs.size());
            g_expert_pack.ram_tier_batch_hits.fetch_add(ram_hit_jobs);
            g_expert_pack.ram_tier_batch_bytes.fetch_add(ram_hit_bytes);
            if (profile_ram_batch) {
                const auto ram_batch_end = std::chrono::steady_clock::now();
                ram_batch_profile_record(
                        trace_op,
                        jobs.empty() ? "" : jobs.front().tensor,
                        jobs.empty() ? "" : jobs.back().tensor,
                        jobs.size(), ram_hit_jobs, ram_hit_bytes, ram_hit_enqueue_ms,
                        std::chrono::duration<double, std::milli>(ram_batch_end - ram_batch_start).count());
            }
        }
        return true;
    }
    // A widened (aligned alias) read of the last expert in a file runs past
    // EOF; O_DIRECT then completes with file_size - offset bytes. Expect that
    // exact size, as long as every payload slice still lies inside it.
    for (iouring_group_plan &group : group_plans) {
        const expert_pack_source *group_source =
            expert_pack_source_for_entry(jobs[read_plans[group.slices.front().plan_idx].job_idx].pack_entry);
        group.expect_sz = group.read_sz;
        if (group_source && group_source->file_size > group.offset &&
                group_source->file_size - group.offset < (uint64_t)group.read_sz) {
            group.expect_sz = (size_t)(group_source->file_size - group.offset);
        }
        for (const iouring_group_slice &slice : group.slices) {
            if (slice.payload_offset + read_plans[slice.plan_idx].payload_sz > group.expect_sz) {
                return false;
            }
        }
    }
    if (!pinned_stage_ensure(ring, max_read_sz, true)) {
        return false;
    }
    const size_t depth = std::min(expert_pack_io_depth(), ring.slots.size());
    if (depth == 0) {
        return false;
    }
    if (stage_granularity_profile_enabled()) {
        ++ring.granularity_calls;
        ring.granularity_total_jobs += jobs.size();
        ring.granularity_read_jobs += group_plans.size();
        ring.granularity_depth_sum += depth;
        ring.granularity_slots_sum += ring.slots.size();
        ring.granularity_max_jobs = std::max<uint64_t>(ring.granularity_max_jobs, (uint64_t)jobs.size());
        ring.granularity_max_read_jobs = std::max<uint64_t>(ring.granularity_max_read_jobs, (uint64_t)group_plans.size());
        ring.granularity_max_depth = std::max<uint64_t>(ring.granularity_max_depth, (uint64_t)depth);
    }
    global_sched_shadow_scope sched_shadow_scope;
    if (global_sched_shadow_enabled()) {
        std::vector<global_sched_shadow_key> sched_keys;
        sched_keys.reserve(read_plans.size());
        for (const iouring_read_plan &plan : read_plans) {
            const Job &job = jobs[plan.job_idx];
            if (!job.pack_entry) {
                continue;
            }
            sched_keys.push_back({
                job.pack_entry->source_idx,
                plan.offset,
                (uint64_t)plan.read_sz,
            });
        }
        sched_shadow_scope.begin(sched_keys, global_sched_shadow_is_prefetch(trace_op));
    }
    const bool profile_io_wait = io_wait_trace_enabled();
    const bool trace_io_read = io_read_trace_enabled();
    const uint64_t io_batch_seq = (profile_io_wait || trace_io_read || io_locality_profile_enabled()) ?
        io_read_trace_next_batch_seq() : 0;
    if (io_locality_profile_enabled()) {
        std::vector<io_locality_profile_item> locality_items;
        locality_items.reserve(read_plans.size());
        for (const iouring_read_plan &plan : read_plans) {
            const Job &job = jobs[plan.job_idx];
            if (!job.pack_entry) {
                continue;
            }
            io_locality_profile_item item;
            item.source_idx = job.pack_entry->source_idx;
            item.offset = plan.offset;
            item.nbytes = plan.read_sz;
            std::snprintf(item.tensor, sizeof(item.tensor), "%s", job.tensor);
            locality_items.push_back(item);
        }
        io_locality_profile_record(trace_op, jobs.size(), locality_items);
    }
    if (trace_io_read) {
        const char *path = std::getenv("GGML_MOE_IO_READ_TRACE_OUT");
        static std::mutex trace_mu;
        static bool header_written = false;
        std::lock_guard<std::mutex> lk(trace_mu);
        FILE *f = std::fopen(path, "a");
        if (f) {
            if (!header_written) {
                std::fprintf(f,
                        "batch_seq,batch_pos,op,jobs,read_jobs,tensor,expert_idx,"
                        "source_idx,offset,nbytes\n");
                header_written = true;
            }
            size_t batch_pos = 0;
            for (const iouring_read_plan &plan : read_plans) {
                const Job &job = jobs[plan.job_idx];
                if (!job.pack_entry) {
                    continue;
                }
                std::fprintf(f,
                        "%lu,%zu,%s,%zu,%zu,%s,%d,%d,%lu,%lu\n",
                        (unsigned long)io_batch_seq,
                        batch_pos++,
                        trace_op ? trace_op : "",
                        jobs.size(),
                        read_plans.size(),
                        job.tensor,
                        job.expert_idx,
                        job.pack_entry->source_idx,
                        (unsigned long)plan.offset,
                        (unsigned long)plan.read_sz);
            }
            std::fclose(f);
        }
    }
    if (h2d_coalesce_profile_enabled()) {
        size_t copy_count = 0;
        size_t coalesced_copy_count = 0;
        size_t host_contiguous_pairs = 0;
        size_t dst_contiguous_pairs = 0;
        size_t both_contiguous_pairs = 0;
        for (const iouring_group_plan &group : group_plans) {
            if (group.slices.empty()) {
                continue;
            }
            ++coalesced_copy_count;
            copy_count += group.slices.size();
            for (size_t i = 1; i < group.slices.size(); ++i) {
                const iouring_group_slice &prev_slice = group.slices[i - 1];
                const iouring_group_slice &cur_slice = group.slices[i];
                const iouring_read_plan &prev_plan = read_plans[prev_slice.plan_idx];
                const iouring_read_plan &cur_plan = read_plans[cur_slice.plan_idx];
                const Job &prev_job = jobs[prev_plan.job_idx];
                const Job &cur_job = jobs[cur_plan.job_idx];
                const bool host_contig =
                    prev_slice.payload_offset + prev_plan.payload_sz == cur_slice.payload_offset;
                const uintptr_t prev_dst = reinterpret_cast<uintptr_t>(prev_job.dst);
                const uintptr_t cur_dst = reinterpret_cast<uintptr_t>(cur_job.dst);
                const bool dst_contig = prev_dst + prev_plan.payload_sz == cur_dst;
                if (host_contig) {
                    ++host_contiguous_pairs;
                }
                if (dst_contig) {
                    ++dst_contiguous_pairs;
                }
                if (host_contig && dst_contig) {
                    ++both_contiguous_pairs;
                } else {
                    ++coalesced_copy_count;
                }
            }
        }
        h2d_coalesce_profile_record(
                trace_op,
                jobs.size(),
                group_plans.size(),
                copy_count,
                coalesced_copy_count,
                host_contiguous_pairs,
                dst_contiguous_pairs,
                both_contiguous_pairs,
                fixed_job_bytes ? expert_bytes_hint : 0);
    }
    expert_pack_record_iouring_batch(group_plans.size());
    ring.iouring_batches += 1;
    ring.iouring_jobs += group_plans.size();
    const size_t bucket = expert_pack_iouring_batch_bucket(group_plans.size());
    if (bucket < 6) {
        ring.iouring_batch_hist[bucket] += 1;
    }

    struct pending_job {
        size_t group_idx = 0;
        size_t slot_idx = 0;
        size_t bytes = 0;
        std::chrono::steady_clock::time_point copy_start;
    };

    const unsigned int flags = expert_pack_env_bool("GGML_MOE_IO_SQPOLL", false) ?
        ggml_moe_common::object_io_sqpoll_setup_flag() : 0;
    if (!ggml_moe_common::object_io_ring_ensure(ring, depth, flags, nullptr)) {
        ++g_expert_pack.io_stats.fallbacks;
        return false;
    }
    // Never return with reads in flight: the caller falls back to synchronous
    // reads, and the next batch reuses user_data 1..depth on this ring, so an
    // abandoned CQE would be matched to an unrelated request.
    struct ring_drain_guard {
        pinned_stage_ring &ring;
        ~ring_drain_guard() { (void)ggml_moe_common::object_io_ring_drain(ring); }
    } ring_drain{ring};
    std::vector<pending_job> pending(depth);
    std::vector<ggml_moe_common::object_io_window_slot> io_window(depth);
    const bool common_object_cqe = common_object_cqe_enabled();
    std::vector<size_t> free_pending;
    free_pending.reserve(depth);
    for (size_t i = 0; i < depth; ++i) {
        free_pending.push_back(depth - 1 - i);
    }

    const bool profile_stage = pinned_stage_profile_enabled();
    const bool profile_copy = copy_profile_enabled();
    const bool profile_copy_h2d = copy_profile_h2d_enabled() && profile_copy;
    double io_batch_slot_wait_ms = 0.0;
    double io_batch_submit_ms = 0.0;
    double io_batch_wait_ms = 0.0;
    double io_batch_enqueue_ms = 0.0;
    double io_wait_enqueue_ms = 0.0;
    uint64_t io_batch_submit_calls = 0;
    uint64_t io_batch_wait_calls = 0;
    uint64_t io_batch_cqes = 0;
    uint64_t io_batch_inflight_sum = 0;
    uint64_t io_batch_inflight_samples = 0;
    uint64_t io_batch_inflight_max = 0;
    auto submit_one = [&](size_t group_idx, size_t slot_idx, size_t pending_idx) -> bool {
        const iouring_group_plan &group = group_plans[group_idx];
        if (group.slices.empty()) return false;
        const iouring_read_plan &first_plan = read_plans[group.slices.front().plan_idx];
        const Job &first_job = jobs[first_plan.job_idx];
        pinned_stage_slot &slot = ring.slots[slot_idx];
        if (slot.pending) {
            const auto wait_start = (profile_stage || profile_io_batch) ?
                std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (!ggml_moe_common::object_stage_slot_synchronize(ring, slot)) return false;
            if (profile_stage || profile_io_batch) {
                const auto wait_end = std::chrono::steady_clock::now();
                const double slot_wait_ms = std::chrono::duration<double, std::milli>(wait_end - wait_start).count();
                if (profile_stage) ring.slot_wait_ms += slot_wait_ms;
                if (profile_io_batch) io_batch_slot_wait_ms += slot_wait_ms;
            }
        }
        pinned_stage_collect_timing(ring, slot);

        pending[pending_idx] = {
            group_idx,
            slot_idx,
            group.expect_sz,
            (batch_ttft_trace_enabled() || profile_copy) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}
        };
        const expert_pack_source *source = expert_pack_source_for_entry(first_job.pack_entry);
        if (!source || source->fd_direct < 0) return false;
        if (common_object_cqe) {
            const ggml_moe_common::object_io_window_request request{
                source->fd_direct,
                slot.host,
                group.read_sz,
                group.offset,
                group_idx,
                false,
                group.expect_sz,
            };
            return ggml_moe_common::object_io_window_prepare_read(
                ring, io_window.data(), io_window.size(), pending_idx, request);
        }
        return ggml_moe_common::object_io_ring_prepare_read(
            ring, source->fd_direct, slot.host, group.read_sz, group.offset,
            (uint64_t)pending_idx + 1);
    };

    size_t next_job = 0;
    size_t inflight = 0;
    const size_t refill_batch = std::min(
        refill_batch_override ? refill_batch_override : expert_pack_io_refill_batch(), depth);
    struct reusable_pending {
        size_t pending_idx = 0;
        size_t slot_idx = 0;
    };
    std::vector<reusable_pending> reusable_pending_slots;
    reusable_pending_slots.reserve(depth);
    const size_t initial_submit_jobs = std::min(group_plans.size(), depth);
    const size_t submit_chunk = common_io_submit_chunk(group_plans.size());
    const size_t chunk_jobs = submit_chunk == 0 ? initial_submit_jobs : std::min(submit_chunk, initial_submit_jobs);
    bool initial_submitted = false;

    size_t completed = 0;
    while (completed < group_plans.size()) {
        auto handle_cqe = [&](const ggml_moe_common::object_io_completion &completion) -> bool {
            size_t pending_idx = SIZE_MAX;
            ggml_moe_common::object_io_completion_status status;
            if (common_object_cqe) {
                ggml_moe_common::object_io_window_completion resolved;
                status = ggml_moe_common::object_io_window_resolve(
                    ring, io_window.data(), io_window.size(), completion, resolved);
                pending_idx = resolved.window_slot;
                if (status == ggml_moe_common::object_io_completion_status::complete &&
                        (pending_idx >= pending.size() ||
                         resolved.request_id != pending[pending_idx].group_idx)) {
                    status = ggml_moe_common::object_io_completion_status::invalid_user_data;
                }
            } else {
                const uint64_t data = completion.user_data;
                pending_idx = data == 0 ? SIZE_MAX : (size_t)data - 1;
                const bool pending_valid = pending_idx < pending.size();
                const size_t expected_bytes = pending_valid ? pending[pending_idx].bytes : 0;
                status = ggml_moe_common::object_io_completion_validate(
                    ring, completion, pending_valid, expected_bytes);
            }
            if (status != ggml_moe_common::object_io_completion_status::complete) {
                ++g_expert_pack.io_stats.fallbacks;
                return false;
            }

            const pending_job done = pending[pending_idx];
            const iouring_group_plan &group = group_plans[done.group_idx];
            pinned_stage_slot &slot = ring.slots[done.slot_idx];
            cudaStream_t copy_stream = group.copy_stream ? group.copy_stream : st;
            ++g_expert_pack.io_stats.cqes;

            const bool measure_enqueue = profile_stage || profile_copy || profile_io_batch || profile_io_wait;
            const auto enqueue_start = measure_enqueue ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if ((profile_stage || profile_copy_h2d) && slot.copy_start) {
                if (cudaEventRecord(slot.copy_start, copy_stream) != cudaSuccess) {
                    return false;
                }
            }
            for (const iouring_group_slice &slice : group.slices) {
                const iouring_read_plan &plan = read_plans[slice.plan_idx];
                const Job &job = jobs[plan.job_idx];
                const char *slot_src = (const char *)slot.host + slice.payload_offset;
                if (!cache_bypass) {
                    common_dynamic_tier_fill(job.pack_entry, slot_src, plan.payload_sz);
                }
                if (cudaMemcpyAsync(job.dst, slot_src, plan.payload_sz, cudaMemcpyHostToDevice, copy_stream) != cudaSuccess) {
                    return false;
                }
            }
            if ((profile_stage || profile_copy_h2d) && slot.copy_done) {
                if (cudaEventRecord(slot.copy_done, copy_stream) != cudaSuccess) {
                    return false;
                }
                if (profile_stage) {
                    slot.timing_pending = true;
                }
            }
            if (!ggml_moe_common::object_stage_slot_record(slot, copy_stream)) {
                cudaStreamSynchronize(copy_stream);
                return false;
            }
            double enqueue_ms = 0.0;
            if (profile_stage) {
                const auto enqueue_end = std::chrono::steady_clock::now();
                enqueue_ms = std::chrono::duration<double, std::milli>(enqueue_end - enqueue_start).count();
                ring.enqueue_ms += enqueue_ms;
            } else if (profile_copy) {
                const auto enqueue_end = std::chrono::steady_clock::now();
                enqueue_ms = std::chrono::duration<double, std::milli>(enqueue_end - enqueue_start).count();
            } else if (profile_io_batch) {
                const auto enqueue_end = std::chrono::steady_clock::now();
                enqueue_ms = std::chrono::duration<double, std::milli>(enqueue_end - enqueue_start).count();
            }
            if (profile_io_batch) io_batch_enqueue_ms += enqueue_ms;
            if (profile_io_wait) io_wait_enqueue_ms += enqueue_ms;
            if (profile_io_batch) ++io_batch_cqes;

            double h2d_ms = -1.0;
            if (done.copy_start != std::chrono::steady_clock::time_point{} &&
                    profile_copy_h2d && group.slices.size() == 1 &&
                    slot.copy_start && slot.copy_done &&
                    cudaEventSynchronize(slot.copy_done) == cudaSuccess) {
                float h2d_ms_f = 0.0f;
                if (cudaEventElapsedTime(&h2d_ms_f, slot.copy_start, slot.copy_done) == cudaSuccess) {
                    h2d_ms = h2d_ms_f;
                }
            }
            for (const iouring_group_slice &slice : group.slices) {
                const iouring_read_plan &plan = read_plans[slice.plan_idx];
                const Job &job = jobs[plan.job_idx];
                ++ring.copies;
                ++g_expert_pack.io_stats.reads;
                g_expert_pack.io_stats.bytes.fetch_add(plan.payload_sz);
                ++g_expert_pack.iouring_h2d_enqueues;
                if (done.copy_start != std::chrono::steady_clock::time_point{}) {
                    const auto copy_end = std::chrono::steady_clock::now();
                    const double wall_ms = std::chrono::duration<double, std::milli>(copy_end - done.copy_start).count();
                    if (batch_ttft_trace_enabled()) {
                        batch_ttft_trace_record(
                            trace_op,
                            job.tensor,
                            job.expert_idx,
                            plan.payload_sz,
                            false,
                            true,
                            false,
                            wall_ms);
                    }
                    if (profile_copy) {
                        copy_profile_record(
                                trace_op, job.tensor, job.expert_idx, plan.payload_sz,
                                true, false, true,
                                0.0, 0.0, wall_ms, enqueue_ms, h2d_ms, wall_ms);
                    }
                }
            }

            ++completed;
            --inflight;
            reusable_pending_slots.push_back({pending_idx, done.slot_idx});
            return true;
        };

        auto refill_pending = [&]() -> bool {
            size_t submitted = 0;
            while (next_job < group_plans.size() && inflight < depth && !reusable_pending_slots.empty() && submitted < refill_batch) {
                const reusable_pending reusable = reusable_pending_slots.back();
                reusable_pending_slots.pop_back();
                const size_t pending_idx = reusable.pending_idx;
                const size_t slot_idx = reusable.slot_idx;
                if (!submit_one(next_job, slot_idx, pending_idx)) {
                    ++g_expert_pack.io_stats.fallbacks;
                    return false;
                }
                ++next_job;
                ++inflight;
                ++submitted;
            }
            if (submitted > 0) {
                const auto refill_submit_start = profile_io_batch ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                if (!ggml_moe_common::object_io_ring_submit(ring)) {
                    ++g_expert_pack.io_stats.fallbacks;
                    return false;
                }
                if (profile_io_batch) {
                    const auto refill_submit_end = std::chrono::steady_clock::now();
                    io_batch_submit_ms += std::chrono::duration<double, std::milli>(refill_submit_end - refill_submit_start).count();
                    ++io_batch_submit_calls;
                }
                ++g_expert_pack.io_stats.submit_calls;
            }
            return true;
        };

        if (!initial_submitted) {
            static std::atomic<bool> reported{false};
            if (submit_chunk > 0 && !reported.exchange(true)) {
                std::fprintf(stderr,
                        "[moe_stream_batch] common I/O submit chunks: chunk=%zu initial_jobs=%zu depth=%zu\n",
                        submit_chunk, initial_submit_jobs, depth);
            }
            auto chunk_start = std::chrono::steady_clock::time_point{};
            const bool ok = ggml_moe_common::object_io_submit_initial(
                    group_plans.size(), depth, submit_chunk,
                    [&](size_t job_idx) -> bool {
                        if (job_idx % chunk_jobs == 0) chunk_start = std::chrono::steady_clock::now();
                        const size_t pending_idx = free_pending.back();
                        free_pending.pop_back();
                        const size_t slot_idx = ring.next++ % ring.slots.size();
                        if (!submit_one(job_idx, slot_idx, pending_idx)) return false;
                        ++next_job;
                        ++inflight;
                        return true;
                    },
                    [&](size_t remaining) -> int {
                        const int submitted = ggml_moe_common::object_io_submit_chunk_result(
                                ggml_moe_common::object_io_ring_submit_count(ring), remaining, flags != 0);
                        if (submitted > 0) {
                            ++g_expert_pack.io_stats.submit_calls;
                            if (profile_io_batch) ++io_batch_submit_calls;
                        }
                        if (submitted > 0 && (size_t)submitted == remaining) {
                            const double submit_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - chunk_start).count();
                            g_expert_pack.io_stats.submit_us.fetch_add((uint64_t)(submit_ms * 1000.0));
                            if (profile_io_batch) io_batch_submit_ms += submit_ms;
                        }
                        return submitted;
                    },
                    [&]() -> bool {
                        // Do not refill here: an earlier read's host slot can
                        // still be an H2D source even though its CQE is ready.
                        while (inflight > 0) {
                            ggml_moe_common::object_io_completion completion;
                            const auto result = ggml_moe_common::object_io_ring_peek(ring, completion);
                            if (result == ggml_moe_common::object_io_poll_result::empty) break;
                            if (result == ggml_moe_common::object_io_poll_result::error || !handle_cqe(completion)) {
                                return false;
                            }
                        }
                        return true;
                    });
            if (!ok) {
                ++g_expert_pack.io_stats.fallbacks;
                // Earlier chunks may already be in H2D. Never enter the
                // synchronous fallback with partially submitted live writers.
                if (submit_chunk > 0) GGML_ABORT("common I/O chunk submission failed; refusing partial-batch fallback");
                return false;
            }
            initial_submitted = true;
        }

        g_expert_pack.io_stats.inflight_sum.fetch_add(inflight);
        ++g_expert_pack.io_stats.inflight_samples;
        expert_pack_atomic_max(g_expert_pack.io_stats.inflight_max, inflight);
        ring.iouring_inflight_sum += inflight;
        ++ring.iouring_inflight_samples;
        if (ring.iouring_inflight_max < inflight) {
            ring.iouring_inflight_max = inflight;
        }
        if (profile_io_batch) {
            io_batch_inflight_sum += inflight;
            ++io_batch_inflight_samples;
            if (io_batch_inflight_max < inflight) {
                io_batch_inflight_max = inflight;
            }
        }
        const auto wait_start = std::chrono::steady_clock::now();
        const size_t wait_completed_before = completed;
        const size_t wait_inflight_before = inflight;
        const size_t wait_next_job_before = next_job;
        const double wait_enqueue_before = io_wait_enqueue_ms;
        ggml_moe_common::object_io_completion completion;
        ++g_expert_pack.io_stats.wait_calls;
        if (profile_io_batch) ++io_batch_wait_calls;
        const bool wait_ok = ggml_moe_common::object_io_ring_wait(ring, completion);
        const auto wait_end = std::chrono::steady_clock::now();
        const double wait_ms = std::chrono::duration<double, std::milli>(wait_end - wait_start).count();
        g_expert_pack.io_stats.wait_us.fetch_add((uint64_t)(wait_ms * 1000.0));
        if (profile_io_batch) io_batch_wait_ms += wait_ms;
        if (!wait_ok) {
            ++g_expert_pack.io_stats.fallbacks;
            return false;
        }
        size_t drained_cqes = 0;
        if (!handle_cqe(completion)) return false;
        ++drained_cqes;
        if (refill_batch == 1 && !refill_pending()) return false;

        while (completed < group_plans.size() && inflight > 0) {
            ggml_moe_common::object_io_completion extra_completion;
            const ggml_moe_common::object_io_poll_result peek_result =
                ggml_moe_common::object_io_ring_peek(ring, extra_completion);
            if (peek_result != ggml_moe_common::object_io_poll_result::ready) {
                break;
            }
            if (!handle_cqe(extra_completion)) return false;
            ++drained_cqes;
            if (refill_batch == 1 && !refill_pending()) return false;
        }
        if (profile_io_wait) {
            io_wait_trace_record(
                    io_batch_seq,
                    trace_op,
                    jobs.empty() ? "" : jobs.front().tensor,
                    jobs.empty() ? "" : jobs.back().tensor,
                    group_plans.size(),
                    wait_completed_before,
                    wait_inflight_before,
                    wait_next_job_before,
                    depth,
                    refill_batch,
                    wait_ms,
                    drained_cqes,
                    io_wait_enqueue_ms - wait_enqueue_before,
                    completed,
                    inflight);
        }
        if (refill_batch > 1 && !refill_pending()) {
            return false;
        }
    }

    if (profile_io_batch) {
        const auto io_batch_end = std::chrono::steady_clock::now();
        io_batch_profile_record(
                trace_op,
                jobs.empty() ? "" : jobs.front().tensor,
                jobs.empty() ? "" : jobs.back().tensor,
                jobs.size(), group_plans.size(), ram_or_prefetch_jobs, ram_or_prefetch_bytes,
                depth, ring.slots.size(), refill_batch, initial_submit_jobs,
                io_batch_submit_calls, io_batch_wait_calls, io_batch_cqes,
                io_batch_inflight_sum, io_batch_inflight_samples, io_batch_inflight_max,
                io_batch_slot_wait_ms, io_batch_submit_ms, io_batch_wait_ms, io_batch_enqueue_ms,
                std::chrono::duration<double, std::milli>(io_batch_end - io_batch_start).count(),
                sort_by_offset);
    }
    if (ram_hit_jobs > 0) {
        ++g_expert_pack.ram_tier_batch_calls;
        g_expert_pack.ram_tier_batch_jobs.fetch_add(jobs.size());
        g_expert_pack.ram_tier_batch_hits.fetch_add(ram_hit_jobs);
        g_expert_pack.ram_tier_batch_bytes.fetch_add(ram_hit_bytes);
        if (profile_ram_batch) {
            const auto ram_batch_end = std::chrono::steady_clock::now();
            ram_batch_profile_record(
                    trace_op,
                    jobs.empty() ? "" : jobs.front().tensor,
                    jobs.empty() ? "" : jobs.back().tensor,
                    jobs.size(), ram_hit_jobs, ram_hit_bytes, ram_hit_enqueue_ms,
                    std::chrono::duration<double, std::milli>(ram_batch_end - ram_batch_start).count());
        }
    }
    if (common_object_cqe) {
        ggml_moe_common::object_io_window_reset(io_window.data(), io_window.size());
    }
    return true;
#else
    (void)jobs;
    (void)expert_bytes_hint;
    (void)job_size;
    (void)job_stream;
    (void)st;
    (void)ring;
    (void)trace_op;
    (void)cache_bypass;
    (void)refill_batch_override;
    return false;
#endif
}

template <typename Job>
static bool expert_pack_iouring_copy_jobs(
        const std::vector<Job> &jobs,
        size_t expert_bytes,
        cudaStream_t st,
        pinned_stage_ring &ring,
        const char *trace_op) {
    auto fixed_job_size = [expert_bytes](const Job &) -> size_t {
        return expert_bytes;
    };
    auto fixed_job_stream = [st](const Job &) -> cudaStream_t {
        return st;
    };
    return expert_pack_iouring_copy_jobs_impl(jobs, expert_bytes, fixed_job_size, fixed_job_stream, st, ring, trace_op);
}

struct sched_pack_copy_job {
    void *dst = nullptr;
    const expert_pack_entry *pack_entry = nullptr;
    int expert_idx = -1;
    char tensor[128] = {};
};

extern "C" bool ggml_cuda_moe_expert_pack_copy_ranges_to_device(
        const char *tensor_name,
        const int32_t *range_first,
        const int32_t *range_last,
        size_t range_count,
        int64_t n_expert,
        size_t expert_bytes,
        size_t padding,
        bool cache_bypass,
        void *device_dst,
        size_t device_capacity,
        void *stream_ptr,
        uint64_t *experts_copied,
        uint64_t *iouring_reads,
        uint64_t *iouring_bytes,
        uint64_t *cache_lookups,
        uint64_t *cache_hits) {
    if (experts_copied) *experts_copied = 0;
    if (iouring_reads) *iouring_reads = 0;
    if (iouring_bytes) *iouring_bytes = 0;
    if (cache_lookups) *cache_lookups = 0;
    if (cache_hits) *cache_hits = 0;
    if (!tensor_name || !tensor_name[0] || !range_first || !range_last ||
            range_count == 0 || n_expert <= 0 || expert_bytes == 0 ||
            !device_dst || !stream_ptr ||
            (uint64_t)n_expert > SIZE_MAX / expert_bytes ||
            (size_t)n_expert * expert_bytes > device_capacity) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_batch_mu);
    std::vector<sched_pack_copy_job> jobs;
    int32_t previous_last = -1;
    for (size_t range = 0; range < range_count; ++range) {
        const int32_t first = range_first[range];
        const int32_t last = range_last[range];
        if (first < 0 || last < first || last >= n_expert || first <= previous_last) {
            return false;
        }
        previous_last = last;
        const size_t count = (size_t)(last - first) + 1;
        if (jobs.size() > SIZE_MAX - count) return false;
        jobs.reserve(jobs.size() + count);
        for (int32_t expert = first; expert <= last; ++expert) {
            const expert_pack_entry *entry = expert_pack_lookup_impl(
                tensor_name, expert, expert_bytes, false);
            if (!entry) return false;
            sched_pack_copy_job job;
            job.dst = (char *)device_dst + (size_t)expert * expert_bytes;
            job.pack_entry = entry;
            job.expert_idx = expert;
            std::snprintf(job.tensor, sizeof(job.tensor), "%s", tensor_name);
            if (!expert_pack_iouring_job_eligible(job, expert_bytes)) return false;
            jobs.push_back(job);
        }
    }

    const uint64_t reads_before = g_expert_pack.io_stats.reads.load(std::memory_order_relaxed);
    const uint64_t bytes_before = g_expert_pack.io_stats.bytes.load(std::memory_order_relaxed);
    const uint64_t lookups_before = g_expert_pack.ram_tier_total.load(std::memory_order_relaxed);
    const uint64_t hits_before = g_expert_pack.ram_tier_hits.load(std::memory_order_relaxed);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    auto fixed_job_size = [expert_bytes](const sched_pack_copy_job &) -> size_t {
        return expert_bytes;
    };
    auto fixed_job_stream = [stream](const sched_pack_copy_job &) -> cudaStream_t {
        return stream;
    };
    const size_t refill_batch_override = std::getenv("GGML_MOE_IO_REFILL_BATCH") ? 0 : 16;
    bool ok = expert_pack_iouring_copy_jobs_impl(
        jobs, expert_bytes, fixed_job_size, fixed_job_stream, stream,
        g_batch.stage_ring, "sched_prefill_pack", cache_bypass, refill_batch_override);
    if (ok && padding > 0) {
        for (size_t range = 0; range < range_count; ++range) {
            if (range_last[range] >= n_expert - 1) continue;
            const size_t offset = ((size_t)range_last[range] + 1) * expert_bytes;
            if (offset > device_capacity || padding > device_capacity - offset ||
                    cudaMemsetAsync((char *)device_dst + offset, 0, padding, stream) != cudaSuccess) {
                ok = false;
                break;
            }
        }
    }

    const uint64_t reads_after = g_expert_pack.io_stats.reads.load(std::memory_order_relaxed);
    const uint64_t bytes_after = g_expert_pack.io_stats.bytes.load(std::memory_order_relaxed);
    const uint64_t lookups_after = g_expert_pack.ram_tier_total.load(std::memory_order_relaxed);
    const uint64_t hits_after = g_expert_pack.ram_tier_hits.load(std::memory_order_relaxed);
    if (experts_copied && ok) *experts_copied = jobs.size();
    if (iouring_reads) *iouring_reads = reads_after - reads_before;
    if (iouring_bytes) *iouring_bytes = bytes_after - bytes_before;
    if (cache_lookups) *cache_lookups = lookups_after - lookups_before;
    if (cache_hits) *cache_hits = hits_after - hits_before;
    return ok;
}

// Dense weights homed on the drive (tools/adaptive DENSE_HOMES).  The alias index holds such a matrix as n equal
// blocks of whole rows, entries (tensor, block, block bytes), next to the routed experts.  The scheduler computes a
// MUL_MAT over such a weight on the GPU at any batch size and copies the weight through here: its blocks are read
// with O_DIRECT through io_uring into a pinned stage of their own (the expert copies keep their ring) and queued to
// the device on the backend's stream.  The stage holds two matrices: while one crosses the link, the next
// drive-homed matrix is read into the other half (ggml_cuda_moe_dense_read_ahead), so a run of drive-homed matrices
// costs its first read and then about the link's time a matrix.
struct dense_stage_half {
    std::string tensor;                 // the matrix read or being read into this half; empty: none
    size_t n_blocks = 0;
    size_t block_bytes = 0;
    std::vector<size_t> prefix;         // per block: where its bytes start in the slot (the read is page aligned)
    std::vector<size_t> expect;         // per block: the bytes the read must return
    std::vector<uint8_t> done;          // per block: read
    size_t in_flight = 0;
    bool failed = false;
};

struct dense_stage_state {
    std::mutex mu;
    pinned_stage_ring ring;             // 2 x blocks_per_half slots: block i of half h in slot h * blocks_per_half + i
    size_t blocks_per_half = 0;
    size_t need_blocks = 0;             // the largest matrix asked about (the load receipt, the scheduler): blocks
    size_t need_slot = 0;               // and slot bytes
    bool failed = false;                // the stage could not be allocated: every copy is left to the mapped weight
    dense_stage_half half[2];
    int last = 1;                       // the half of the latest copy; a read goes to the other one
};

static dense_stage_state g_dense_stage;

static size_t dense_slot_bytes(size_t block_bytes) {
    const size_t a = expert_pack_direct_alignment();
    return (size_t)align_up_u64((uint64_t)block_bytes + a, a);
}

// the blocks of a matrix the index holds as n_blocks x block_bytes == nbytes, n >= 1; 0 when it holds none
static size_t dense_index_blocks(const char *tensor_name, size_t nbytes, size_t *block_bytes) {
    if (!tensor_name || !tensor_name[0] || nbytes == 0) return 0;
    expert_pack_init_once();
    const char *disable_env = std::getenv("GGML_MOE_EXPERT_PACK_RUNTIME_DISABLE");
    if (!expert_pack_catalog_ready() || (disable_env && disable_env[0] && disable_env[0] != '0')) return 0;
    const auto &entries = g_expert_pack.entries;
    size_t lo = 0, hi = entries.size();
    while (lo < hi) {                   // the first entry of the tensor (block 0 sorts first)
        const size_t mid = lo + (hi - lo) / 2;
        if (std::strcmp(entries[mid].tensor, tensor_name) < 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo >= entries.size() || std::strcmp(entries[lo].tensor, tensor_name) != 0 || entries[lo].expert_idx != 0) {
        return 0;
    }
    const uint64_t bb = entries[lo].nbytes;
    if (bb == 0 || nbytes % bb != 0) return 0;
    const size_t n = nbytes / bb;
    if (lo + n > entries.size()) return 0;
    for (size_t i = 0; i < n; ++i) {
        const expert_pack_entry &e = entries[lo + i];
        if (std::strcmp(e.tensor, tensor_name) != 0 || e.expert_idx != (int32_t)i || e.nbytes != bb) return 0;
    }
    if (lo + n < entries.size() && std::strcmp(entries[lo + n].tensor, tensor_name) == 0) return 0;
    if (block_bytes) *block_bytes = (size_t)bb;
    return n;
}

#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING) && !defined(_WIN32)
// the ring failed: every read in flight is reaped unmatched (no read lands in a slot later) and its half failed
static void dense_stage_abort(dense_stage_state &s) {
    (void)ggml_moe_common::object_io_ring_drain(s.ring);
    for (dense_stage_half &half : s.half) {
        half.failed = half.failed || half.in_flight > 0;
        half.in_flight = 0;
    }
}

// takes one completion of the stage's ring and marks its block (a failed or short read fails its half); false when
// the ring gives none (dense_stage_abort)
static bool dense_stage_reap_one(dense_stage_state &s) {
    ggml_moe_common::object_io_completion c;
    if (!ggml_moe_common::object_io_ring_wait(s.ring, c)) {
        dense_stage_abort(s);
        return false;
    }
    const size_t slot = (size_t)c.user_data - 1;
    if (c.user_data == 0 || s.blocks_per_half == 0 || slot >= 2 * s.blocks_per_half) return true;
    dense_stage_half &half = s.half[slot / s.blocks_per_half];
    const size_t i = slot % s.blocks_per_half;
    if (half.in_flight > 0) --half.in_flight;
    if (i >= half.n_blocks || c.result < 0 || (size_t)c.result != half.expect[i]) {
        half.failed = true;
    } else {
        half.done[i] = 1;
    }
    return true;
}

// no read in flight in half h; false when one of its reads failed
static bool dense_stage_drain(dense_stage_state &s, int h) {
    while (s.half[h].in_flight > 0 && dense_stage_reap_one(s)) {
    }
    return !s.half[h].failed;
}

// the stage at the size of the largest matrix asked about: every read finished, every slot's copy done; a stage that
// could not be allocated is not tried again
static bool dense_stage_ensure(dense_stage_state &s, size_t n_blocks, size_t slot_bytes) {
    s.need_blocks = std::max(s.need_blocks, n_blocks);
    s.need_slot = std::max(s.need_slot, slot_bytes);
    if (s.blocks_per_half >= n_blocks && s.ring.slot_sz >= slot_bytes) {
        return true;
    }
    if (s.failed) {
        return false;
    }
    for (int h = 0; h < 2; ++h) {
        if (!s.ring.slots.empty()) (void)dense_stage_drain(s, h);
        s.half[h] = dense_stage_half();
    }
    ggml_moe_common::object_stage_ring_release(s.ring);
    s.ring = pinned_stage_ring();
    if (!ggml_moe_common::object_stage_ring_allocate(s.ring, s.need_slot, 2 * s.need_blocks, false) ||
            !ggml_moe_common::object_io_ring_ensure(s.ring, 2 * s.need_blocks, 0, nullptr)) {
        std::fprintf(stderr, "[moe_stream_batch] dense stage: init failed for 2 x %zu slots of %.2f MiB\n",
                     s.need_blocks, s.need_slot / (1024.0 * 1024.0));
        ggml_moe_common::object_stage_ring_release(s.ring);
        s.ring = pinned_stage_ring();
        s.blocks_per_half = 0;
        s.failed = true;
        return false;
    }
    s.blocks_per_half = s.need_blocks;
    std::fprintf(stderr, "[moe_stream_batch] dense stage: 2 x %zu slots of %.2f MiB pinned\n",
                 s.blocks_per_half, s.ring.slot_sz / (1024.0 * 1024.0));
    return true;
}

// submits the reads of a matrix into half h, its slots' earlier copies finished first
static bool dense_stage_submit(dense_stage_state &s, int h, const char *tensor_name, size_t n_blocks,
                               size_t block_bytes) {
    dense_stage_half &half = s.half[h];
    (void)dense_stage_drain(s, h);
    half = dense_stage_half();
    half.tensor = tensor_name;
    half.n_blocks = n_blocks;
    half.block_bytes = block_bytes;
    half.prefix.assign(n_blocks, 0);
    half.expect.assign(n_blocks, 0);
    half.done.assign(n_blocks, 0);
    const size_t a = expert_pack_direct_alignment();
    for (size_t i = 0; i < n_blocks; ++i) {
        const expert_pack_entry *e = expert_pack_lookup_impl(tensor_name, (int)i, block_bytes, false);
        const expert_pack_source *src = expert_pack_source_for_entry(e);
        pinned_stage_slot &slot = s.ring.slots[(size_t)h * s.blocks_per_half + i];
        if (!src || src->fd_direct < 0 || !ggml_moe_common::object_stage_slot_synchronize(s.ring, slot)) {
            half.failed = true;
            break;
        }
        const uint64_t off = e->offset / a * a;
        half.prefix[i] = (size_t)(e->offset - off);
        const size_t read_sz = (size_t)align_up_u64(half.prefix[i] + block_bytes, a);
        half.expect[i] = src->file_size > off && src->file_size - off < read_sz ? (size_t)(src->file_size - off) : read_sz;
        if (read_sz > slot.host_capacity || half.expect[i] < half.prefix[i] + block_bytes ||
                !ggml_moe_common::object_io_ring_prepare_read(s.ring, src->fd_direct, slot.host, read_sz, off,
                    (uint64_t)h * s.blocks_per_half + i + 1)) {
            half.failed = true;
            break;
        }
        ++half.in_flight;
    }
    if (!ggml_moe_common::object_io_ring_submit(s.ring)) {
        dense_stage_abort(s);
    }
    if (half.failed) {
        (void)dense_stage_drain(s, h);
        half = dense_stage_half();
        return false;
    }
    return true;
}
#endif

extern "C" bool ggml_cuda_moe_dense_has_tensor(const char *tensor_name, size_t nbytes, size_t *block_bytes) {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING) && !defined(_WIN32)
    if (!::ggml_moe::adapter_uses_common()) return false;
    size_t bb = 0;
    const size_t n = dense_index_blocks(tensor_name, nbytes, &bb);
    if (n == 0) return false;
    std::lock_guard<std::mutex> lock(g_dense_stage.mu);
    g_dense_stage.need_blocks = std::max(g_dense_stage.need_blocks, n);
    g_dense_stage.need_slot = std::max(g_dense_stage.need_slot, dense_slot_bytes(bb));
    if (block_bytes) *block_bytes = bb;
    return true;
#else
    (void)tensor_name;
    (void)nbytes;
    (void)block_bytes;
    return false;
#endif
}

// the stage for the largest matrix asked about so far, allocated now (the load receipt, once every weight homed on the
// drive was asked about): its pinned bytes; false when it cannot be
extern "C" bool ggml_cuda_moe_dense_stage_prepare(size_t *pinned_bytes) {
    if (pinned_bytes) *pinned_bytes = 0;
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING) && !defined(_WIN32)
    std::lock_guard<std::mutex> lock(g_dense_stage.mu);
    dense_stage_state &s = g_dense_stage;
    if (s.need_blocks == 0 || !dense_stage_ensure(s, s.need_blocks, s.need_slot)) {
        return false;
    }
    if (pinned_bytes) *pinned_bytes = 2 * s.blocks_per_half * s.ring.slot_sz;
    return true;
#else
    return false;
#endif
}

extern "C" bool ggml_cuda_moe_dense_read_ahead(const char *tensor_name, size_t nbytes) {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING) && !defined(_WIN32)
    size_t bb = 0;
    const size_t n = dense_index_blocks(tensor_name, nbytes, &bb);
    if (n == 0) return false;
    std::lock_guard<std::mutex> lock(g_dense_stage.mu);
    dense_stage_state &s = g_dense_stage;
    for (const dense_stage_half &h : s.half) {
        if (h.tensor == tensor_name) return true;
    }
    if (!dense_stage_ensure(s, n, dense_slot_bytes(bb))) return false;
    return dense_stage_submit(s, 1 - s.last, tensor_name, n, bb);
#else
    (void)tensor_name;
    (void)nbytes;
    return false;
#endif
}

extern "C" bool ggml_cuda_moe_dense_copy_to_device(
        const char *tensor_name,
        void *device_dst,
        size_t nbytes,
        void *stream_ptr,
        bool *read_ahead_hit) {
    if (read_ahead_hit) *read_ahead_hit = false;
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING) && !defined(_WIN32)
    size_t bb = 0;
    const size_t n = dense_index_blocks(tensor_name, nbytes, &bb);
    if (n == 0 || !device_dst || !stream_ptr) return false;
    std::lock_guard<std::mutex> lock(g_dense_stage.mu);
    dense_stage_state &s = g_dense_stage;
    if (!dense_stage_ensure(s, n, dense_slot_bytes(bb))) return false;
    int h = s.half[0].tensor == tensor_name ? 0 : s.half[1].tensor == tensor_name ? 1 : -1;
    if (h >= 0 && (s.half[h].n_blocks != n || s.half[h].block_bytes != bb || s.half[h].failed)) {
        (void)dense_stage_drain(s, h);
        s.half[h] = dense_stage_half();
        h = -1;
    }
    if (read_ahead_hit) *read_ahead_hit = h >= 0;
    if (h < 0) {
        h = 1 - s.last;
        if (!dense_stage_submit(s, h, tensor_name, n, bb)) return false;
    }
    dense_stage_half &half = s.half[h];
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    bool ok = true;
    for (size_t i = 0; i < n && ok; ++i) {
        while (!half.done[i] && !half.failed && dense_stage_reap_one(s)) {
        }
        pinned_stage_slot &slot = s.ring.slots[(size_t)h * s.blocks_per_half + i];
        ok = half.done[i] &&
            cudaMemcpyAsync((char *)device_dst + i * bb, (const char *)slot.host + half.prefix[i], bb,
                            cudaMemcpyHostToDevice, stream) == cudaSuccess &&
            ggml_moe_common::object_stage_slot_record(slot, stream);
    }
    (void)dense_stage_drain(s, h);
    half = dense_stage_half();
    s.last = h;
    return ok;
#else
    (void)tensor_name;
    (void)device_dst;
    (void)nbytes;
    (void)stream_ptr;
    return false;
#endif
}

template <typename Job>
static bool expert_pack_iouring_copy_jobs_varbytes(
        const std::vector<Job> &jobs,
        cudaStream_t st,
        pinned_stage_ring &ring,
        const char *trace_op) {
    auto per_job_size = [](const Job &job) -> size_t {
        return job.nbytes;
    };
    auto per_job_stream = [st](const Job &job) -> cudaStream_t {
        return job.stream ? job.stream : st;
    };
    return expert_pack_iouring_copy_jobs_impl(jobs, 0, per_job_size, per_job_stream, st, ring, trace_op);
}

struct expert_pack_v2_iouring_copy_job {
    int slot = -1;
    void *dst = nullptr;
    const expert_pack_v2_entry *entry = nullptr;
    int expert_idx = -1;
    char tensor[128] = {};
    char logical_tensor[128] = {};
};

static bool expert_pack_v2_iouring_copy_jobs(
        const std::vector<expert_pack_v2_iouring_copy_job> &jobs,
        size_t expert_bytes,
        cudaStream_t st,
        pinned_stage_ring &ring,
        const char *trace_op,
        uint64_t *read_us,
        uint64_t *h2d_enqueue_us) {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING) && !defined(_WIN32)
    if (read_us) *read_us = 0;
    if (h2d_enqueue_us) *h2d_enqueue_us = 0;
    if (jobs.empty()) return true;
    if (!expert_pack_v2_full_cover_down_iouring_enabled()) return false;
    if (g_expert_pack.io_backend != 2) return false;

    const size_t alignment = expert_pack_direct_alignment();
    const bool aligned_alias_batch = expert_pack_env_bool("GGML_MOE_IO_ALIGNED_ALIAS_BATCH", false);
    const size_t default_read_sz = (size_t)align_up_u64((uint64_t)expert_bytes, (uint64_t)alignment);
    if (default_read_sz == 0 || default_read_sz > (size_t)UINT_MAX) return false;

    struct read_plan {
        size_t job_idx = 0;
        uint64_t offset = 0;
        size_t read_sz = 0;
        size_t prefix = 0;
        int32_t source_idx = 0;
    };

    std::vector<read_plan> plans;
    plans.reserve(jobs.size());
    size_t max_read_sz = default_read_sz;
    for (size_t i = 0; i < jobs.size(); ++i) {
        const expert_pack_v2_iouring_copy_job &job = jobs[i];
        if (!job.entry || (size_t)job.entry->nbytes != expert_bytes || !job.dst) return false;
        const expert_pack_source *source = expert_pack_v2_source_for_entry(job.entry);
        if (!source || source->fd_direct < 0) return false;

        uint64_t read_offset = job.entry->offset;
        size_t read_sz = default_read_sz;
        size_t prefix = 0;
        if ((job.entry->offset % alignment) != 0) {
            if (!aligned_alias_batch) return false;
            read_offset = (job.entry->offset / alignment) * alignment;
            prefix = (size_t)(job.entry->offset - read_offset);
            read_sz = (size_t)align_up_u64((uint64_t)prefix + (uint64_t)expert_bytes, (uint64_t)alignment);
        }
        if (read_sz < expert_bytes || read_sz > (size_t)UINT_MAX) return false;
        max_read_sz = std::max(max_read_sz, read_sz);
        plans.push_back({i, read_offset, read_sz, prefix, job.entry->source_idx});
    }

    const bool sort_by_offset = expert_pack_env_bool("GGML_MOE_IO_SORT_OFFSET", false);
    if (sort_by_offset && plans.size() > 1) {
        std::stable_sort(plans.begin(), plans.end(),
            [&](const read_plan &a, const read_plan &b) {
                if (a.source_idx != b.source_idx) return a.source_idx < b.source_idx;
                return a.offset < b.offset;
            });
    }

    if (!pinned_stage_ensure(ring, max_read_sz, true)) return false;
    const size_t depth = std::min(expert_pack_io_depth(), ring.slots.size());
    if (depth == 0) return false;

    const unsigned int flags = expert_pack_env_bool("GGML_MOE_IO_SQPOLL", false) ?
        ggml_moe_common::object_io_sqpoll_setup_flag() : 0;
    if (!ggml_moe_common::object_io_ring_ensure(ring, depth, flags, nullptr)) {
        ++g_expert_pack.io_stats.fallbacks;
        return false;
    }
    struct pending_job {
        size_t plan_idx = 0;
        size_t slot_idx = 0;
        size_t bytes = 0;
        std::chrono::steady_clock::time_point copy_start;
    };
    std::vector<pending_job> pending(depth);
    std::vector<ggml_moe_common::object_io_window_slot> io_window(depth);
    const bool common_object_cqe = common_object_cqe_enabled();
    std::vector<size_t> free_pending;
    free_pending.reserve(depth);
    for (size_t i = 0; i < depth; ++i) {
        free_pending.push_back(depth - 1 - i);
    }
    struct reusable_pending {
        size_t pending_idx = 0;
        size_t slot_idx = 0;
    };
    std::vector<reusable_pending> reusable_pending_slots;
    reusable_pending_slots.reserve(depth);

    const bool profile_stage = pinned_stage_profile_enabled();
    const bool profile_copy = copy_profile_enabled();
    const bool profile_copy_h2d = copy_profile_h2d_enabled() && profile_copy;
    double local_h2d_enqueue_ms = 0.0;
    const auto read_start = std::chrono::steady_clock::now();

    auto submit_one = [&](size_t plan_idx, size_t slot_idx, size_t pending_idx) -> bool {
        const read_plan &plan = plans[plan_idx];
        const expert_pack_v2_iouring_copy_job &job = jobs[plan.job_idx];
        pinned_stage_slot &slot = ring.slots[slot_idx];
        if (slot.pending) {
            const auto wait_start = profile_stage ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (!ggml_moe_common::object_stage_slot_synchronize(ring, slot)) return false;
            if (profile_stage) {
                const auto wait_end = std::chrono::steady_clock::now();
                ring.slot_wait_ms += std::chrono::duration<double, std::milli>(wait_end - wait_start).count();
            }
        }
        pinned_stage_collect_timing(ring, slot);

        pending[pending_idx] = {
            plan_idx,
            slot_idx,
            plan.read_sz,
            (batch_ttft_trace_enabled() || profile_copy) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}
        };
        const expert_pack_source *source = expert_pack_v2_source_for_entry(job.entry);
        if (!source || source->fd_direct < 0) return false;
        if (common_object_cqe) {
            const ggml_moe_common::object_io_window_request request{
                source->fd_direct,
                slot.host,
                plan.read_sz,
                plan.offset,
                plan_idx,
                false,
            };
            return ggml_moe_common::object_io_window_prepare_read(
                ring, io_window.data(), io_window.size(), pending_idx, request);
        }
        return ggml_moe_common::object_io_ring_prepare_read(
            ring, source->fd_direct, slot.host, plan.read_sz, plan.offset,
            (uint64_t)pending_idx + 1);
    };

    ring.iouring_batches += 1;
    ring.iouring_jobs += plans.size();
    expert_pack_record_iouring_batch(plans.size());
    const size_t bucket = expert_pack_iouring_batch_bucket(plans.size());
    if (bucket < 6) {
        ring.iouring_batch_hist[bucket] += 1;
    }

    size_t next_plan = 0;
    size_t inflight = 0;
    const size_t refill_batch = std::min(expert_pack_io_refill_batch(), depth);
    while (next_plan < plans.size() && inflight < depth) {
        const size_t pending_idx = free_pending.back();
        free_pending.pop_back();
        const size_t slot_idx = ring.next++ % ring.slots.size();
        if (!submit_one(next_plan, slot_idx, pending_idx)) {
            ++g_expert_pack.io_stats.fallbacks;
            return false;
        }
        ++next_plan;
        ++inflight;
    }
    if (!ggml_moe_common::object_io_ring_submit(ring)) {
        ++g_expert_pack.io_stats.fallbacks;
        return false;
    }
    ++g_expert_pack.io_stats.submit_calls;

    size_t completed = 0;
    while (completed < plans.size()) {
        g_expert_pack.io_stats.inflight_sum.fetch_add(inflight);
        ++g_expert_pack.io_stats.inflight_samples;
        expert_pack_atomic_max(g_expert_pack.io_stats.inflight_max, inflight);
        ring.iouring_inflight_sum += inflight;
        ++ring.iouring_inflight_samples;
        if (ring.iouring_inflight_max < inflight) {
            ring.iouring_inflight_max = inflight;
        }

        auto handle_cqe = [&](const ggml_moe_common::object_io_completion &completion) -> bool {
            size_t pending_idx = SIZE_MAX;
            ggml_moe_common::object_io_completion_status status;
            if (common_object_cqe) {
                ggml_moe_common::object_io_window_completion resolved;
                status = ggml_moe_common::object_io_window_resolve(
                    ring, io_window.data(), io_window.size(), completion, resolved);
                pending_idx = resolved.window_slot;
                if (status == ggml_moe_common::object_io_completion_status::complete &&
                        (pending_idx >= pending.size() ||
                         resolved.request_id != pending[pending_idx].plan_idx)) {
                    status = ggml_moe_common::object_io_completion_status::invalid_user_data;
                }
            } else {
                const uint64_t data = completion.user_data;
                pending_idx = data == 0 ? SIZE_MAX : (size_t)data - 1;
                const bool pending_valid = pending_idx < pending.size();
                const size_t expected_bytes = pending_valid ? pending[pending_idx].bytes : 0;
                status = ggml_moe_common::object_io_completion_validate(
                    ring, completion, pending_valid, expected_bytes);
            }
            if (status != ggml_moe_common::object_io_completion_status::complete) {
                ++g_expert_pack.io_stats.fallbacks;
                return false;
            }

            const pending_job done = pending[pending_idx];
            const read_plan &plan = plans[done.plan_idx];
            const expert_pack_v2_iouring_copy_job &job = jobs[plan.job_idx];
            pinned_stage_slot &slot = ring.slots[done.slot_idx];
            ++g_expert_pack.io_stats.cqes;

            const bool measure_enqueue = profile_stage || profile_copy;
            const auto enqueue_start = measure_enqueue ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if ((profile_stage || profile_copy_h2d) && slot.copy_start) {
                if (cudaEventRecord(slot.copy_start, st) != cudaSuccess) return false;
            }
            const char *slot_src = (const char *)slot.host + plan.prefix;
            if (cudaMemcpyAsync(job.dst, slot_src, expert_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) {
                return false;
            }
            if ((profile_stage || profile_copy_h2d) && slot.copy_done) {
                if (cudaEventRecord(slot.copy_done, st) != cudaSuccess) return false;
                if (profile_stage) slot.timing_pending = true;
            }
            if (!ggml_moe_common::object_stage_slot_record(slot, st)) {
                cudaStreamSynchronize(st);
                return false;
            }
            double enqueue_ms = 0.0;
            if (measure_enqueue) {
                const auto enqueue_end = std::chrono::steady_clock::now();
                enqueue_ms = std::chrono::duration<double, std::milli>(enqueue_end - enqueue_start).count();
                if (profile_stage) ring.enqueue_ms += enqueue_ms;
                local_h2d_enqueue_ms += enqueue_ms;
            }
            ++ring.copies;
            ++g_expert_pack.io_stats.reads;
            g_expert_pack.io_stats.bytes.fetch_add(expert_bytes);
            ++g_expert_pack.iouring_h2d_enqueues;
            if (done.copy_start != std::chrono::steady_clock::time_point{}) {
                const auto copy_end = std::chrono::steady_clock::now();
                const double wall_ms = std::chrono::duration<double, std::milli>(copy_end - done.copy_start).count();
                if (batch_ttft_trace_enabled()) {
                    batch_ttft_trace_record(trace_op, job.logical_tensor[0] ? job.logical_tensor : job.tensor,
                            job.expert_idx, expert_bytes, false, true, false, wall_ms);
                }
                if (profile_copy) {
                    copy_profile_record(trace_op, job.logical_tensor[0] ? job.logical_tensor : job.tensor,
                            job.expert_idx, expert_bytes, true, false, true,
                            0.0, 0.0, wall_ms, enqueue_ms, -1.0, wall_ms);
                }
            }

            ++completed;
            --inflight;
            reusable_pending_slots.push_back({pending_idx, done.slot_idx});
            return true;
        };

        auto refill_pending = [&]() -> bool {
            size_t submitted = 0;
            while (next_plan < plans.size() && inflight < depth && !reusable_pending_slots.empty() && submitted < refill_batch) {
                const reusable_pending reusable = reusable_pending_slots.back();
                reusable_pending_slots.pop_back();
                if (!submit_one(next_plan, reusable.slot_idx, reusable.pending_idx)) {
                    ++g_expert_pack.io_stats.fallbacks;
                    return false;
                }
                ++next_plan;
                ++inflight;
                ++submitted;
            }
            if (submitted > 0) {
                if (!ggml_moe_common::object_io_ring_submit(ring)) {
                    ++g_expert_pack.io_stats.fallbacks;
                    return false;
                }
                ++g_expert_pack.io_stats.submit_calls;
            }
            return true;
        };

        ggml_moe_common::object_io_completion completion;
        const auto wait_start = std::chrono::steady_clock::now();
        ++g_expert_pack.io_stats.wait_calls;
        const bool wait_ok = ggml_moe_common::object_io_ring_wait(ring, completion);
        const auto wait_end = std::chrono::steady_clock::now();
        g_expert_pack.io_stats.wait_us.fetch_add(
                (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(wait_end - wait_start).count());
        if (!wait_ok) {
            ++g_expert_pack.io_stats.fallbacks;
            return false;
        }
        if (!handle_cqe(completion)) return false;
        if (refill_batch == 1 && !refill_pending()) return false;

        while (completed < plans.size() && inflight > 0) {
            ggml_moe_common::object_io_completion extra_completion;
            const ggml_moe_common::object_io_poll_result peek_result =
                ggml_moe_common::object_io_ring_peek(ring, extra_completion);
            if (peek_result != ggml_moe_common::object_io_poll_result::ready) break;
            if (!handle_cqe(extra_completion)) return false;
            if (refill_batch == 1 && !refill_pending()) return false;
        }
        if (refill_batch > 1 && !refill_pending()) return false;
    }

    const auto read_end = std::chrono::steady_clock::now();
    if (read_us) {
        *read_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(read_end - read_start).count();
    }
    if (h2d_enqueue_us) {
        *h2d_enqueue_us = (uint64_t)(local_h2d_enqueue_ms * 1000.0);
    }
    if (common_object_cqe) {
        ggml_moe_common::object_io_window_reset(io_window.data(), io_window.size());
    }
    return true;
#else
    (void)jobs;
    (void)expert_bytes;
    (void)st;
    (void)ring;
    (void)trace_op;
    (void)read_us;
    (void)h2d_enqueue_us;
    return false;
#endif
}

static int batch_cache_insert_slot(
        batch_vram_cache *c, uintptr_t key, const void *host_data, size_t sz, cudaStream_t st,
        bool allow_evict, bool preload, const int *avoid_slots = nullptr, int n_avoid_slots = 0,
        bool do_copy = true, const char *tensor_name = nullptr, int expert_idx = -1,
        bool prefetch_down = false, bool pin_preload = true, bool async_prefetch = false,
        uint64_t profile_count = 0, bool q4_pilot_prefetch = false,
        bool object_admitted = false) {
    auto insert_decline_debug_enabled = []() {
        const char *env = std::getenv("GGML_MOE_STREAM_DECLINE_DEBUG");
        return env && env[0] && std::strcmp(env, "0") != 0;
    };
    auto slot_state_name = [](q4_colibri_slot_state state) {
        switch (state) {
            case q4_colibri_slot_state::free: return "free";
            case q4_colibri_slot_state::reserved: return "reserved";
            case q4_colibri_slot_state::reading: return "reading";
            case q4_colibri_slot_state::h2d_ready: return "h2d_ready";
            case q4_colibri_slot_state::failed: return "failed";
        }
        return "unknown";
    };
    if (!c || !c->pool || c->n_slots == 0 || sz > c->slot_sz) {
        if (insert_decline_debug_enabled()) {
            std::fprintf(stderr,
                    "[moe_stream_batch] cache insert diag: reason=invalid_cache_or_size tensor=%s expert=%d "
                    "request_bytes=%zu slot_bytes=%zu slots=%d has_cache=%d has_pool=%d\n",
                    tensor_name ? tensor_name : "", expert_idx, sz, c ? c->slot_sz : 0,
                    c ? c->n_slots : 0, c ? 1 : 0, (c && c->pool) ? 1 : 0);
        }
        return -1;
    }
    const bool protect_profile_only = profile_protect_profile_only_enabled();
    bool pin_slot = preload && pin_preload && profile_protect_enabled() &&
        (!protect_profile_only || profile_count > 0 || profile_pinned_key_contains((uint64_t)key));
    if (!pin_slot && profile_pin_on_insert_enabled() && profile_protect_enabled()) {
        load_profile_once();
        if (g_profile_enabled) {
            if (profile_count == 0) {
                profile_count = profile_count_for_key(key);
            }
            if (profile_count >= profile_pin_min_count() && c->pinned < profile_preload_slot_budget(c)) {
                pin_slot = true;
            }
        }
    }
    if (pin_slot && c->pinned >= profile_preload_slot_budget(c)) return -1;

    int slot = -1;
    bool found_empty_slot = false;
    uint64_t oldest = UINT64_MAX;
    uint64_t lowest_unified_score = UINT64_MAX;
    uint32_t lowest_hits = UINT32_MAX;
    uint32_t lowest_profile_count = UINT32_MAX;
    const bool lfu_lru = cache_policy_lfu_lru_enabled();
    const bool profile_lfu_lru = cache_policy_profile_lfu_lru_enabled();
    const bool hybrid_profile_lfu_lru = cache_policy_hybrid_profile_lfu_lru_enabled();
    const bool object_admission = q4_vram_object_admission_enabled();
    const bool vram_unified_score = q4_vram_unified_score_enabled();
    const bool vram_static_hotness_protect =
        q4_vram_static_hotness_protect_enabled() && q4_colibri_static_hotness_enabled();
    const uint64_t vram_static_hotness_ticks =
        vram_static_hotness_protect ? q4_vram_static_hotness_protect_ticks() : 0;
    const bool evict_profile = cache_evict_profile_enabled();
    const vram_cache_layer_bias_config &vram_layer_bias_cfg = vram_cache_layer_bias_config_get();
    const bool use_profile_score =
        profile_lfu_lru || (hybrid_profile_lfu_lru && c->clock >= cache_policy_hybrid_after());
    const int candidate_layer = q4_graph_layer_from_tensor(tensor_name);
    const size_t q4_pilot_reserved_slots = g_expert_pack.glm52_colibri ?
        expert_pack_env_size("GGML_MOE_Q4_VRAM_PILOT_RESERVED_SLOTS", 0, 0, 4096) : 0;
    const int q4_pilot_reserved_start = q4_pilot_reserved_slots > 0 && c->n_slots > 0 ?
        std::max<int>(0, c->n_slots - (int)std::min<size_t>(q4_pilot_reserved_slots, (size_t)c->n_slots)) :
        c->n_slots;
    auto q4_slot_in_pilot_partition = [&](int slot_idx) {
        return q4_pilot_reserved_slots > 0 && slot_idx >= q4_pilot_reserved_start;
    };
    const size_t vram_score_candidate_limit = vram_unified_score ?
        expert_pack_env_size("GGML_MOE_Q4_VRAM_SCORE_CANDIDATE_LIMIT", 128, 1, 4096) : 0;
    std::vector<int> vram_score_candidates;
    if (vram_unified_score) {
        vram_score_candidates.reserve(vram_score_candidate_limit);
    }
    if (profile_lfu_lru || hybrid_profile_lfu_lru) {
        load_profile_once();
    }
    const int gate_preload_updown_max_hits = cache_gate_preload_evict_updown_max_hits();
    const bool gate_preload_protect_updown =
        preload && gate_preload_updown_max_hits >= 0 && cache_tensor_is_gate(tensor_name);
    auto emit_no_slot_diag = [&](const char *reason) {
        if (!insert_decline_debug_enabled()) return;
        int avoided = 0;
        int pilot_excluded = 0;
        int empty = 0;
        int filling = 0;
        int pending = 0;
        int reserved = 0;
        int reading = 0;
        int failed = 0;
        int pinned = 0;
        int leased = 0;
        int gate_protected = 0;
        int evictable = 0;
        int occupied = 0;
        for (int i = 0; i < c->n_slots; ++i) {
            if (batch_slot_is_avoided(i, avoid_slots, n_avoid_slots)) {
                ++avoided;
                continue;
            }
            const bool in_pilot_partition = q4_slot_in_pilot_partition(i);
            if (q4_pilot_reserved_slots > 0) {
                if (q4_pilot_prefetch) {
                    if (!in_pilot_partition || (c->slot_key[i] != 0 && !c->slot_q4_pilot[i])) {
                        ++pilot_excluded;
                        continue;
                    }
                } else if (in_pilot_partition) {
                    ++pilot_excluded;
                    continue;
                }
            }
            if (c->slot_key[i] == 0) {
                ++empty;
                continue;
            }
            ++occupied;
            if (c->slot_state[i] == q4_colibri_slot_state::failed) ++failed;
            if (c->slot_filling[i]) ++filling;
            if (c->slot_pending[i]) ++pending;
            if (c->slot_state[i] == q4_colibri_slot_state::reserved) ++reserved;
            if (c->slot_state[i] == q4_colibri_slot_state::reading) ++reading;
            if (c->slot_pinned[i]) ++pinned;
            if (c->slot_lease_count[i] != 0) ++leased;
            if (gate_preload_protect_updown && cache_tensor_is_updown(c->slot_tensor[i]) &&
                    (int)c->slot_hits[i] > gate_preload_updown_max_hits) {
                ++gate_protected;
            }
            if (allow_evict &&
                    !c->slot_filling[i] &&
                    !c->slot_pending[i] &&
                    c->slot_state[i] != q4_colibri_slot_state::reserved &&
                    c->slot_state[i] != q4_colibri_slot_state::reading &&
                    c->slot_lease_count[i] == 0 &&
                    !c->slot_pinned[i]) {
                ++evictable;
            }
        }
        std::fprintf(stderr,
                "[moe_stream_batch] cache insert diag: reason=%s tensor=%s expert=%d request_bytes=%zu "
                "slot_bytes=%zu slots=%d allow_evict=%d preload=%d do_copy=%d pin_slot=%d pinned=%lu "
                "budget=%zu occupied=%d empty=%d evictable=%d avoided=%d pilot_excluded=%d "
                "filling=%d pending=%d reserved=%d reading=%d failed=%d pinned_slots=%d gate_protected=%d "
                "leased=%d first_state=%s first_key=%lu\n",
                reason ? reason : "", tensor_name ? tensor_name : "", expert_idx, sz, c->slot_sz, c->n_slots,
                allow_evict ? 1 : 0, preload ? 1 : 0, do_copy ? 1 : 0, pin_slot ? 1 : 0,
                (unsigned long)c->pinned, profile_preload_slot_budget(c), occupied, empty, evictable,
                avoided, pilot_excluded, filling, pending, reserved, reading, failed, pinned, gate_protected, leased,
                c->n_slots > 0 ? slot_state_name(c->slot_state[0]) : "none",
                c->n_slots > 0 ? (unsigned long)c->slot_key[0] : 0UL);
    };
    if ((profile_lfu_lru || hybrid_profile_lfu_lru) && !g_cache_policy_diag_registered.exchange(true)) {
        std::atexit(cache_policy_diag_report_atexit);
    }
    for (int i = 0; i < c->n_slots; ++i) {
        if (batch_slot_is_avoided(i, avoid_slots, n_avoid_slots)) continue;
        const bool in_pilot_partition = q4_slot_in_pilot_partition(i);
        if (q4_pilot_reserved_slots > 0) {
            if (q4_pilot_prefetch) {
                if (!in_pilot_partition) continue;
                if (c->slot_key[i] != 0 && !c->slot_q4_pilot[i]) continue;
            } else if (in_pilot_partition) {
                continue;
            }
        }
        if (c->slot_lease_count[i] != 0) continue;
        if (c->slot_key[i] == 0) {
            slot = i;
            found_empty_slot = true;
            break;
        }
        // The K2.7 frequency-admission candidate uses the role cache as a
        // triplet store plus transient staging.  Only its object-level policy
        // may evict admitted objects; ordinary role-level selection is
        // restricted to transient slots.
        if (object_admission && c->slot_object_admitted[i]) continue;
        if (!allow_evict) continue;
        if (c->slot_filling[i]) continue;
        if (c->slot_pending[i] &&
                c->slot_state[i] == q4_colibri_slot_state::h2d_ready) {
            if (!batch_cache_wait_slot_ready(c, i)) continue;
        }
        if (c->slot_pending[i]) continue;
        if (c->slot_state[i] == q4_colibri_slot_state::reserved ||
                c->slot_state[i] == q4_colibri_slot_state::reading) {
            continue;
        }
        if (c->slot_pinned[i]) continue;
        if (gate_preload_protect_updown && cache_tensor_is_updown(c->slot_tensor[i]) &&
                (int)c->slot_hits[i] > gate_preload_updown_max_hits) {
            continue;
        }
        if (vram_unified_score) {
            const uint64_t effective_used =
                vram_cache_layer_biased_lru_used(c, i, vram_layer_bias_cfg, candidate_layer);
            if (vram_score_candidates.size() < vram_score_candidate_limit) {
                vram_score_candidates.push_back(i);
            } else {
                size_t replace_idx = SIZE_MAX;
                uint64_t newest_effective_used = 0;
                for (size_t cand_idx = 0; cand_idx < vram_score_candidates.size(); ++cand_idx) {
                    const int cand_slot = vram_score_candidates[cand_idx];
                    const uint64_t cand_used =
                        vram_cache_layer_biased_lru_used(c, cand_slot, vram_layer_bias_cfg, candidate_layer);
                    if (replace_idx == SIZE_MAX || cand_used > newest_effective_used) {
                        replace_idx = cand_idx;
                        newest_effective_used = cand_used;
                    }
                }
                if (replace_idx != SIZE_MAX && effective_used < newest_effective_used) {
                    vram_score_candidates[replace_idx] = i;
                }
            }
        } else if (use_profile_score) {
            if (c->slot_profile_count[i] < lowest_profile_count ||
                    (c->slot_profile_count[i] == lowest_profile_count &&
                     (c->slot_hits[i] < lowest_hits ||
                      (c->slot_hits[i] == lowest_hits && c->slot_used[i] < oldest)))) {
                lowest_profile_count = c->slot_profile_count[i];
                lowest_hits = c->slot_hits[i];
                oldest = c->slot_used[i];
                slot = i;
            }
        } else if (lfu_lru) {
            if (c->slot_hits[i] < lowest_hits ||
                    (c->slot_hits[i] == lowest_hits && c->slot_used[i] < oldest)) {
                lowest_hits = c->slot_hits[i];
                oldest = c->slot_used[i];
                slot = i;
            }
        } else {
            uint64_t effective_used =
                vram_cache_layer_biased_lru_used(c, i, vram_layer_bias_cfg, candidate_layer);
            if (q4_tier_diag_vram_reuse_protected(c->slot_layer[i], c->slot_expert[i])) {
                effective_used = UINT64_MAX - 1;
                ++g_q4_tier_overlap_diag.vram_reuse_protect_penalized;
            }
            if (vram_static_hotness_protect && vram_static_hotness_ticks > 0) {
                const q4_colibri_object_marker marker{
                    -1,
                    (int32_t)c->slot_layer[i],
                    (int32_t)c->slot_expert[i],
                    0,
                    0,
                    c->slot_payload_bytes[i],
                };
                if (q4_colibri_static_hotness_bonus_raw(marker) > 0) {
                    effective_used = effective_used > UINT64_MAX - vram_static_hotness_ticks ?
                        UINT64_MAX : effective_used + vram_static_hotness_ticks;
                    ++g_q4_tier_overlap_diag.vram_static_hotness_protected;
                }
            }
            if (effective_used < oldest) {
                oldest = effective_used;
                slot = i;
            }
        }
    }
    if (vram_unified_score && !found_empty_slot) {
        slot = -1;
        oldest = UINT64_MAX;
        lowest_hits = UINT32_MAX;
        for (int cand_slot : vram_score_candidates) {
            const q4_colibri_object_marker marker{
                -1, (int32_t)c->slot_layer[cand_slot], (int32_t)c->slot_expert[cand_slot], 0, 0, 0,
            };
            const uint64_t score = q4_colibri_host_object_score(
                    marker, candidate_layer, c->slot_hits[cand_slot]);
            if (slot < 0 || score < lowest_unified_score ||
                    (score == lowest_unified_score &&
                     (c->slot_hits[cand_slot] < lowest_hits ||
                      (c->slot_hits[cand_slot] == lowest_hits && c->slot_used[cand_slot] < oldest)))) {
                lowest_unified_score = score;
                lowest_hits = c->slot_hits[cand_slot];
                oldest = c->slot_used[cand_slot];
                slot = cand_slot;
            }
        }
    }
    if (slot < 0) {
        emit_no_slot_diag("no_slot");
        return -1;
    }
    if (!batch_cache_wait_slot_ready(c, slot)) {
        emit_no_slot_diag("wait_slot_ready_failed");
        return -1;
    }
    if (!batch_cache_stream_wait_slot_lease(c, slot, st)) {
        emit_no_slot_diag("wait_slot_lease_failed");
        return -1;
    }
    const bool replacing_slot = c->slot_key[slot] != 0;
    if (replacing_slot && !batch_cache_invalidate_role_owner(c, slot)) {
        emit_no_slot_diag("invalidate_role_owner_failed");
        return -1;
    }
    const uintptr_t victim_key = c->slot_key[slot];
    const uint32_t victim_hits = c->slot_hits[slot];
    const uint32_t victim_profile_count = c->slot_profile_count[slot];
    const bool victim_pinned = c->slot_pinned[slot];
    const bool victim_prefetch_down = c->slot_prefetch_down[slot];
    const uint64_t victim_used = c->slot_used[slot];
    const int victim_layer = replacing_slot ? c->slot_layer[slot] : -1;
    const int victim_expert = replacing_slot ? c->slot_expert[slot] : -1;
    char victim_tensor_name[128] = {};
    if (replacing_slot && c->slot_tensor[slot][0]) {
        std::snprintf(victim_tensor_name, sizeof(victim_tensor_name), "%s", c->slot_tensor[slot]);
    }
    if (replacing_slot && q4_tier_diag_vram_reuse_protected(victim_layer, victim_expert)) {
        ++g_q4_tier_overlap_diag.vram_reuse_protect_selected;
    }
    if (replacing_slot && vram_static_hotness_protect) {
        const q4_colibri_object_marker victim_marker{
            -1, (int32_t)victim_layer, (int32_t)victim_expert, 0, 0, c->slot_payload_bytes[slot],
        };
        if (q4_colibri_static_hotness_bonus_raw(victim_marker) > 0) {
            ++g_q4_tier_overlap_diag.vram_static_hotness_selected;
        }
    }
    vram_cache_layer_diag_record(candidate_layer, victim_layer, replacing_slot);
    if (replacing_slot) {
        if (vram_unified_score) {
            ++g_q4_tier_overlap_diag.tier_vram_score_victim_selections;
        }
        if (q4_object_score_profile_enabled()) {
            const q4_colibri_object_marker candidate_marker{
                -1, (int32_t)candidate_layer, (int32_t)expert_idx, 0, 0, sz,
            };
            const q4_colibri_object_marker victim_marker{
                -1, (int32_t)victim_layer, (int32_t)victim_expert, 0, 0, c->slot_payload_bytes[slot],
            };
            const uint64_t candidate_freq = profile_count > 0 ? profile_count : 1;
            const uint64_t candidate_score = q4_colibri_host_object_score(
                    candidate_marker, candidate_layer, (uint32_t)std::min<uint64_t>(candidate_freq, UINT32_MAX));
            const uint64_t victim_score = q4_colibri_host_object_score(
                    victim_marker, candidate_layer, victim_hits);
            q4_score_profile_record_vram_eviction(candidate_layer, candidate_score, victim_score);
        }
        (void)q4_colibri_host_object_demote_from_vram(
                victim_tensor_name, victim_layer, victim_expert, victim_hits, candidate_layer);
        q4_tier_diag_record_vram_eviction(victim_layer, victim_expert);
    }
    if (c->slot_pinned[slot] && !pin_slot && c->pinned > 0) {
        --c->pinned;
    }
    if (c->slot_key[slot] != 0 && c->slot_prefetch_down[slot]) {
        ++c->down_prefetch_evicted;
    }
    if (profile_count == 0 && (profile_lfu_lru || hybrid_profile_lfu_lru)) {
        profile_count = profile_count_for_key(key);
    }
    if ((profile_lfu_lru || hybrid_profile_lfu_lru) && c->slot_key[slot] != 0) {
        ++g_cache_policy_diag.profile_policy_evictions;
        if (c->slot_profile_count[slot] > 0) {
            ++g_cache_policy_diag.profile_policy_victim_count_nonzero;
            g_cache_policy_diag.profile_policy_victim_count_sum.fetch_add(c->slot_profile_count[slot]);
        }
    }
    if (replacing_slot && evict_profile) {
        cache_evict_profile_record(
                c, slot, victim_key, victim_hits, victim_profile_count, victim_pinned,
                victim_prefetch_down, victim_used, key, tensor_name, expert_idx, preload,
                prefetch_down, profile_count);
    }
    if (victim_key != 0) {
        auto it = c->key_to_slot.find(victim_key);
        if (it != c->key_to_slot.end() && it->second == slot) {
            c->key_to_slot.erase(it);
        }
    }
    c->slot_key[slot] = key;
    c->key_to_slot[key] = slot;
    c->slot_used[slot] = c->clock++;
    c->slot_hits[slot] = 0;
    c->slot_profile_count[slot] = profile_count > UINT32_MAX ? UINT32_MAX : (uint32_t)profile_count;
    c->slot_layer[slot] = candidate_layer;
    c->slot_expert[slot] = expert_idx;
    if (tensor_name && tensor_name[0]) {
        std::snprintf(c->slot_tensor[slot], sizeof(c->slot_tensor[slot]), "%s", tensor_name);
    } else {
        c->slot_tensor[slot][0] = '\0';
    }
    if ((profile_lfu_lru || hybrid_profile_lfu_lru) && c->slot_profile_count[slot] > 0) {
        ++g_cache_policy_diag.inserted_profile_count_nonzero;
        g_cache_policy_diag.inserted_profile_count_sum.fetch_add(c->slot_profile_count[slot]);
    }
    if (pin_slot && !c->slot_pinned[slot]) {
        ++c->pinned;
    }
    c->slot_pinned[slot] = pin_slot;
    if (pin_slot) {
        profile_pinned_key_record((uint64_t)key);
    }
    c->slot_prefetch_down[slot] = prefetch_down;
    c->slot_q4_pilot[slot] = q4_pilot_prefetch;
    c->slot_object_admitted[slot] = object_admitted;
    c->slot_filling[slot] = false;
    c->slot_pending[slot] = false;
    const uint64_t generation = ++c->slot_generation[slot];
    c->slot_payload_bytes[slot] = sz;
    c->slot_ready_marker[slot] = {};
    c->slot_state[slot] = q4_colibri_slot_state::reserved;
    void *dst = (char *)c->pool + (size_t)slot * c->slot_sz;
    auto clear_slot = [&]() {
        const uintptr_t old_key = c->slot_key[slot];
        if (old_key != 0) {
            auto it = c->key_to_slot.find(old_key);
            if (it != c->key_to_slot.end() && it->second == slot) {
                c->key_to_slot.erase(it);
            }
        }
        c->slot_key[slot] = 0;
        c->slot_used[slot] = 0;
        c->slot_hits[slot] = 0;
        c->slot_profile_count[slot] = 0;
        c->slot_tensor[slot][0] = '\0';
        c->slot_expert[slot] = -1;
        c->slot_layer[slot] = -1;
        if (c->slot_pinned[slot] && c->pinned > 0) {
            --c->pinned;
        }
        c->slot_pinned[slot] = false;
        c->slot_prefetch_down[slot] = false;
        c->slot_q4_pilot[slot] = false;
        c->slot_object_admitted[slot] = false;
        c->slot_filling[slot] = false;
        c->slot_pending[slot] = false;
        ++c->slot_generation[slot];
        c->slot_payload_bytes[slot] = 0;
        c->slot_ready_marker[slot] = {};
        c->slot_state[slot] = q4_colibri_slot_state::free;
    };
    if (do_copy) {
        const expert_pack_entry *pack_entry = expert_pack_lookup(tensor_name, expert_idx, sz);
        batch_copy_trace copy_trace;
        const auto copy_start = batch_ttft_trace_enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const char *copy_op = preload ? "preload_load" : "runtime_load";
        const bool q4_pipe_copy = pack_entry && g_expert_pack.q4_colibri && q4_colibri_pipe_requested();
        bool copied = q4_pipe_copy ? q4_colibri_pipe_copy(
            c, slot, generation, key, dst, pack_entry, sz, st, tensor_name, expert_idx) :
            batch_cache_copy_h2d(g_batch.stage_ring, dst, host_data, sz, st, pack_entry, &copy_trace,
                    copy_op, tensor_name, expert_idx);
        if (!copied) {
            if (q4_pipe_copy) {
                clear_slot();
                return -1;
            }
            if (g_expert_pack.glm52_colibri && q4_colibri_routed_tensor_name(tensor_name)) {
                clear_slot();
                return -1;
            }
            if (pack_entry && !batch_cache_copy_h2d(g_batch.stage_ring, dst, host_data, sz, st, nullptr, &copy_trace,
                        copy_op, tensor_name, expert_idx)) {
                clear_slot();
                return -1;
            }
            if (!pack_entry) {
                clear_slot();
                return -1;
            }
        }
        if (cudaGetLastError() != cudaSuccess) {
            clear_slot();
            return -1;
        }
        if (async_prefetch) {
            const ggml_moe_common::role_cache_view role_view = batch_cache_role_view(c);
            if (!ggml_moe_common::role_cache_ensure_ready_event(role_view, slot)) {
                clear_slot();
                return -1;
            }
            if (!ggml_moe_common::role_cache_record_ready_event(role_view, slot, st)) {
                clear_slot();
                return -1;
            }
            c->slot_ready_marker[slot] = {key, generation, sz};
            c->slot_state[slot] = q4_colibri_slot_state::h2d_ready;
            c->slot_pending[slot] = true;
        } else if (!q4_pipe_copy) {
            c->slot_state[slot] = q4_colibri_slot_state::h2d_ready;
        }
        if (copy_start != std::chrono::steady_clock::time_point{}) {
            const auto copy_end = std::chrono::steady_clock::now();
            batch_ttft_trace_record(
                preload ? "preload_load" : "runtime_load",
                tensor_name,
                expert_idx,
                sz,
                false,
                copy_trace.pack_hit,
                copy_trace.ram_hit,
                std::chrono::duration<double, std::milli>(copy_end - copy_start).count());
        }
    }
    if (preload) {
        ++c->preloads;
    } else {
        ++c->misses;
    }
    if (prefetch_down) {
        ++c->down_prefetch_loads;
    }
    return slot;
}







struct down_demand_prefill_job {
    batch_vram_cache *cache = nullptr;
    int slot = -1;
    uintptr_t key = 0;
    uint64_t generation = 0;
    void *dst = nullptr;
    const void *host_data = nullptr;
    const expert_pack_entry *pack_entry = nullptr;
    size_t expert_bytes = 0;
    int expert_idx = -1;
    char tensor[128] = {};
};

struct down_demand_prefill_queue_state {
    bool inited = false;
    bool enabled = false;
    bool stop = false;
    bool failed = false;
    cudaStream_t stream = nullptr;
    pinned_stage_ring ring;
    std::deque<down_demand_prefill_job> jobs;
    size_t max_jobs = 256;
    std::mutex mu;
    std::condition_variable cv;
    std::thread worker;
    std::atomic<uint64_t> submitted{0};
    std::atomic<uint64_t> completed{0};
    std::atomic<uint64_t> batches{0};
    std::atomic<uint64_t> failures{0};
    std::atomic<uint64_t> queue_full{0};
    std::atomic<uint64_t> no_slot{0};
    std::atomic<uint64_t> duplicate{0};
    std::atomic<uint64_t> evicted{0};
};

static down_demand_prefill_queue_state g_down_demand_queue;











static void preload_profile_entries_for_tensor(
        const std::vector<profile_entry> &entries,
        const char *label,
        const char *tensor_name,
        const void *src0_data,
        int64_t n_as,
        size_t nb02,
        size_t src0_bytes,
        cudaStream_t st,
        bool pin_preload,
        bool allow_evict,
        bool apply_skip,
        bool track_tensor,
        size_t scan_budget) {
    if (entries.empty() || !tensor_name || !tensor_name[0]) return;
    batch_vram_cache *cache = batch_cache_get(src0_bytes);
    if (!cache) return;
    static bool skip_inited = false;
    static long skip_calls = 0;
    if (apply_skip && !skip_inited) {
        const char *skip_env = std::getenv("GGML_MOE_VRAM_PROFILE_SKIP_FIRST_PRELOADS");
        skip_calls = (skip_env && skip_env[0]) ? std::atol(skip_env) : 0;
        if (skip_calls < 0) skip_calls = 0;
        skip_inited = true;
    }
    const int call_idx = g_profile_preload_calls.fetch_add(1);
    if (apply_skip && skip_calls > 0 && call_idx < skip_calls) return;
    if (track_tensor && tensor_already_preloaded(tensor_name)) return;
    char lookup_name[96] = {};
    profile_lookup_name(tensor_name, lookup_name, sizeof(lookup_name));
    int loaded = 0;
    const size_t preload_budget = scan_budget > 0 ? scan_budget : profile_preload_slot_budget(cache);
    const int cache_id = batch_cache_id_for_size(src0_bytes);
    const bool use_lookup = !profile_has_tensor_locked(entries, tensor_name);
    size_t seen_for_cache = 0;
    for (size_t ip = 0; ip < entries.size(); ++ip) {
        const profile_entry &e = entries[ip];
        if (e.expert_bytes != 0 && batch_cache_id_for_size(e.expert_bytes) != cache_id) continue;
        if (seen_for_cache++ >= preload_budget) break;
        if (std::strcmp(e.tensor, tensor_name) != 0 && std::strcmp(e.tensor, lookup_name) != 0) continue;
        if (!use_lookup && std::strcmp(e.tensor, tensor_name) != 0) continue;
        if (e.expert_idx < 0 || e.expert_idx >= n_as) continue;
        const uintptr_t key = batch_key_hash(tensor_name, e.expert_idx);
        const int existing_slot = batch_cache_find_slot(cache, key);
        if (existing_slot >= 0) {
            if (pin_preload && profile_protect_enabled()) {
                if (batch_cache_promote_profile_slot(cache, existing_slot, e.count)) {
                    profile_pinned_key_record((uint64_t)key);
                }
            }
            continue;
        }
        const char *expert_host = (const char *)src0_data + (size_t)e.expert_idx * nb02;
        if (batch_cache_insert_slot(cache, key, expert_host, src0_bytes, st, allow_evict, true,
                nullptr, 0, true, tensor_name, e.expert_idx, false, pin_preload, false, e.count) < 0) break;
        ++loaded;
    }
    if (loaded > 0) {
        std::fprintf(stderr, "[moe_stream_batch] %s preload: %s loaded=%d\n", label, tensor_name, loaded);
    }
}













static bool ensure_dev(void *&p, size_t &cur, size_t need) {
    if (cur >= need) return true;
    if (p) cudaFree(p);
    p = nullptr;
    cur = 0;
    if (cudaMalloc(&p, need) != cudaSuccess) return false;
    cur = need;
    return true;
}

static bool ensure_host_pinned(void *&p, size_t &cur, size_t need) {
    if (cur >= need) return true;
    if (p) cudaFreeHost(p);
    p = nullptr;
    cur = 0;
    if (cudaHostAlloc(&p, need, cudaHostAllocDefault) != cudaSuccess) return false;
    cur = need;
    return true;
}

struct expert_pack_v2_shadow_stage_state {
    std::mutex mu;
    void *host = nullptr;
    size_t host_sz = 0;
    void *dev = nullptr;
    size_t dev_sz = 0;
    cudaEvent_t copy_start = nullptr;
    cudaEvent_t copy_done = nullptr;
    bool event_failed = false;
    bool header_written = false;
    bool report_registered = false;
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> staged_calls{0};
    std::atomic<uint64_t> active_entries{0};
    std::atomic<uint64_t> covered_entries{0};
    std::atomic<uint64_t> staged_entries{0};
    std::atomic<uint64_t> staged_bytes{0};
    std::atomic<uint64_t> read_failures{0};
    std::atomic<uint64_t> h2d_failures{0};
    std::atomic<uint64_t> alloc_failures{0};
    std::atomic<uint64_t> direct_reads{0};
    std::atomic<uint64_t> direct_bytes{0};
    std::atomic<uint64_t> direct_physical_bytes{0};
    std::atomic<uint64_t> direct_fallbacks{0};
    std::atomic<uint64_t> buffered_reads{0};
    std::atomic<uint64_t> buffered_bytes{0};
    ggml_moe_common::object_io_runtime_counters io_stats;
    std::atomic<uint64_t> iouring_jobs{0};
    std::atomic<uint64_t> coalesced_groups{0};
    std::atomic<uint64_t> coalesced_source_calls{0};
    std::atomic<uint64_t> coalesced_entries{0};
    double alloc_ms = 0.0;
    double read_ms = 0.0;
    double h2d_ms = 0.0;
    double sync_ms = 0.0;
    double total_ms = 0.0;
};

static expert_pack_v2_shadow_stage_state g_expert_pack_v2_shadow_stage;

static bool expert_pack_v2_shadow_stage_enabled() {
    const char *env = std::getenv("GGML_MOE_EXPERT_PACK_V2_PARTIAL_SPLIT_SHADOW_STAGE");
    return env && env[0] && env[0] != '0';
}

static const char * expert_pack_v2_shadow_stage_path() {
    const char *path = std::getenv("GGML_MOE_EXPERT_PACK_V2_PARTIAL_SPLIT_SHADOW_STAGE_OUT");
    return path && path[0] ? path : nullptr;
}

static uint64_t expert_pack_v2_shadow_stage_max_calls() {
    const char *env = std::getenv("GGML_MOE_EXPERT_PACK_V2_PARTIAL_SPLIT_SHADOW_STAGE_MAX_CALLS");
    if (!env || !env[0]) return 0;
    char *end = nullptr;
    const unsigned long long v = std::strtoull(env, &end, 10);
    return end && *end == '\0' ? (uint64_t)v : 0;
}

static bool expert_pack_v2_shadow_stage_prompt_enabled() {
    const char *env = std::getenv("GGML_MOE_EXPERT_PACK_V2_PARTIAL_SPLIT_SHADOW_STAGE_PROMPT");
    return env && env[0] && env[0] != '0';
}

static void expert_pack_v2_shadow_stage_write_header(FILE *f, expert_pack_v2_shadow_stage_state &s) {
    if (!f || s.header_written) return;
    std::fprintf(f,
        "seq,phase,role,tensor,logical_type,logical_nbytes,logical_ne00,"
        "logical_ne01,logical_nb01,n_active,cache_hits,covered,"
        "covered_cache_hits,staged,reject_no_manifest,reject_not_allowed,"
        "reject_no_entry,reject_unsupported,reject_manifest_mismatch,"
        "reject_shape_mismatch,reject_not_smaller,covered_bytes,"
        "covered_saved_bytes,staged_bytes,staged_saved_bytes,"
        "max_entry_bytes,host_capacity,dev_capacity,alloc_grew,"
        "alloc_ms,read_ms,h2d_ms,sync_ms,total_ms,read_failures,"
        "h2d_failures,direct_reads,direct_bytes,"
        "direct_physical_bytes,direct_fallbacks,buffered_reads,"
        "buffered_bytes,iouring_batches,iouring_jobs,"
        "iouring_submit_calls,iouring_wait_calls,iouring_cqes,"
        "iouring_wait_us,iouring_inflight_sum,"
        "iouring_inflight_samples,iouring_inflight_max\n");
    s.header_written = true;
}

static void expert_pack_v2_shadow_stage_report() {
    expert_pack_v2_shadow_stage_state &s = g_expert_pack_v2_shadow_stage;
    const uint64_t calls = s.calls.load(std::memory_order_relaxed);
    if (calls == 0) return;
    const ggml_moe_common::object_io_runtime_snapshot io_stats =
        ggml_moe_common::object_io_runtime_snapshot_load(s.io_stats);
    std::fprintf(stderr,
        "[moe_stream_batch] expert pack v2 shadow stage report: "
        "calls=%llu staged_calls=%llu active=%llu covered=%llu staged=%llu "
        "staged_bytes=%.3f GiB read_failures=%llu h2d_failures=%llu alloc_failures=%llu "
        "direct_reads=%llu direct_bytes=%.3f GiB direct_physical=%.3f GiB "
        "direct_fallbacks=%llu buffered_reads=%llu buffered_bytes=%.3f GiB "
        "iouring_batches=%llu iouring_jobs=%llu iouring_submit_calls=%llu "
        "iouring_wait_calls=%llu iouring_cqes=%llu iouring_wait_ms=%.3f "
        "iouring_inflight_avg=%.2f iouring_inflight_max=%llu "
        "coalesced_groups=%llu coalesced_source_calls=%llu coalesced_entries=%llu "
        "alloc_ms=%.3f read_ms=%.3f h2d_ms=%.3f sync_ms=%.3f total_ms=%.3f "
        "host_capacity=%.3f MiB dev_capacity=%.3f MiB\n",
        (unsigned long long)calls,
        (unsigned long long)s.staged_calls.load(std::memory_order_relaxed),
        (unsigned long long)s.active_entries.load(std::memory_order_relaxed),
        (unsigned long long)s.covered_entries.load(std::memory_order_relaxed),
        (unsigned long long)s.staged_entries.load(std::memory_order_relaxed),
        s.staged_bytes.load(std::memory_order_relaxed) / (1024.0 * 1024.0 * 1024.0),
        (unsigned long long)s.read_failures.load(std::memory_order_relaxed),
        (unsigned long long)s.h2d_failures.load(std::memory_order_relaxed),
        (unsigned long long)s.alloc_failures.load(std::memory_order_relaxed),
        (unsigned long long)s.direct_reads.load(std::memory_order_relaxed),
        s.direct_bytes.load(std::memory_order_relaxed) / (1024.0 * 1024.0 * 1024.0),
        s.direct_physical_bytes.load(std::memory_order_relaxed) / (1024.0 * 1024.0 * 1024.0),
        (unsigned long long)s.direct_fallbacks.load(std::memory_order_relaxed),
        (unsigned long long)s.buffered_reads.load(std::memory_order_relaxed),
        s.buffered_bytes.load(std::memory_order_relaxed) / (1024.0 * 1024.0 * 1024.0),
        (unsigned long long)io_stats.batches,
        (unsigned long long)s.iouring_jobs.load(std::memory_order_relaxed),
        (unsigned long long)io_stats.submit_calls,
        (unsigned long long)io_stats.wait_calls,
        (unsigned long long)io_stats.cqes,
        io_stats.wait_us / 1000.0,
        io_stats.inflight_samples > 0 ?
            (double)io_stats.inflight_sum / (double)io_stats.inflight_samples : 0.0,
        (unsigned long long)io_stats.inflight_max,
        (unsigned long long)s.coalesced_groups.load(std::memory_order_relaxed),
        (unsigned long long)s.coalesced_source_calls.load(std::memory_order_relaxed),
        (unsigned long long)s.coalesced_entries.load(std::memory_order_relaxed),
        s.alloc_ms,
        s.read_ms,
        s.h2d_ms,
        s.sync_ms,
        s.total_ms,
        s.host_sz / (1024.0 * 1024.0),
        s.dev_sz / (1024.0 * 1024.0));
}

struct expert_pack_v2_shadow_candidate {
    int expert_idx = -1;
    const expert_pack_v2_entry *entry = nullptr;
    bool cache_hit = false;
};

struct expert_pack_v2_shadow_coalesce_group {
    bool active = false;
    int layer = -1;
    cudaStream_t stream = nullptr;
    uint64_t first_seq = 0;
    uint64_t last_seq = 0;
    uint64_t source_calls = 0;
    bool has_up = false;
    bool has_gate = false;
    bool has_down = false;
    int n_active = 0;
    int cache_hits = 0;
    int covered = 0;
    int covered_cache_hits = 0;
    int reject_no_manifest = 0;
    int reject_not_allowed = 0;
    int reject_no_entry = 0;
    int reject_unsupported = 0;
    int reject_manifest_mismatch = 0;
    int reject_shape_mismatch = 0;
    int reject_not_smaller = 0;
    uint64_t covered_bytes = 0;
    uint64_t covered_saved_bytes = 0;
    uint64_t staged_saved_bytes = 0;
    size_t max_entry_bytes = 0;
    std::vector<expert_pack_v2_shadow_candidate> candidates;
};

static expert_pack_v2_shadow_coalesce_group g_expert_pack_v2_shadow_coalesce;

static int expert_pack_v2_shadow_tensor_layer(const char *tensor_name) {
    if (!tensor_name || !tensor_name[0]) return -1;
    const char *p = std::strstr(tensor_name, "blk.");
    if (!p) return -1;
    p += 4;
    char *end = nullptr;
    const long layer = std::strtol(p, &end, 10);
    if (end == p || layer < 0 || layer > INT_MAX) return -1;
    return (int)layer;
}

static void expert_pack_v2_shadow_coalesce_reset(expert_pack_v2_shadow_coalesce_group &g) {
    g.active = false;
    g.layer = -1;
    g.stream = nullptr;
    g.first_seq = 0;
    g.last_seq = 0;
    g.source_calls = 0;
    g.has_up = false;
    g.has_gate = false;
    g.has_down = false;
    g.n_active = 0;
    g.cache_hits = 0;
    g.covered = 0;
    g.covered_cache_hits = 0;
    g.reject_no_manifest = 0;
    g.reject_not_allowed = 0;
    g.reject_no_entry = 0;
    g.reject_unsupported = 0;
    g.reject_manifest_mismatch = 0;
    g.reject_shape_mismatch = 0;
    g.reject_not_smaller = 0;
    g.covered_bytes = 0;
    g.covered_saved_bytes = 0;
    g.staged_saved_bytes = 0;
    g.max_entry_bytes = 0;
    g.candidates.clear();
}

static std::string expert_pack_v2_shadow_coalesce_role(const expert_pack_v2_shadow_coalesce_group &g) {
    std::string role;
    if (g.has_up) role += "up";
    if (g.has_gate) {
        if (!role.empty()) role += "+";
        role += "gate";
    }
    if (g.has_down) {
        if (!role.empty()) role += "+";
        role += "down";
    }
    if (role.empty()) role = "coalesced";
    return role;
}

struct expert_pack_v2_shadow_read_result {
    bool ok = false;
    bool direct = false;
    bool direct_attempted = false;
    size_t nread = 0;
    size_t physical_bytes = 0;
};

static expert_pack_v2_shadow_read_result expert_pack_v2_shadow_read_entry_to_host(
        const expert_pack_v2_entry *entry,
        const char *tensor_name,
        int expert_idx,
        void *dst,
        size_t dst_capacity) {
    expert_pack_v2_shadow_read_result result;
    if (!entry || !dst || dst_capacity < (size_t)entry->nbytes || entry->nbytes == 0) {
        return result;
    }

#if !defined(_WIN32)
    if (expert_pack_v2_shadow_direct_read_enabled()) {
        expert_pack_source *source = expert_pack_v2_source_for_entry(entry);
        if (source && source->fd_direct >= 0) {
            result.direct_attempted = true;
            const size_t sz = (size_t)entry->nbytes;
            const uint64_t alignment = expert_pack_direct_alignment();
            const uint64_t aligned_offset = (entry->offset / alignment) * alignment;
            const size_t prefix = (size_t)(entry->offset - aligned_offset);
            const size_t read_sz = (size_t)align_up_u64((uint64_t)prefix + (uint64_t)sz, alignment);
            result.physical_bytes = read_sz;
            if ((entry->offset % alignment) == 0 &&
                    ((uintptr_t)dst % alignment) == 0 &&
                    read_sz <= dst_capacity &&
                    read_sz == (size_t)align_up_u64((uint64_t)sz, alignment)) {
                if (expert_pack_direct_pread_all(source->fd_direct, dst, read_sz, entry->offset)) {
                    result.ok = true;
                    result.direct = true;
                    result.nread = sz;
                    return result;
                }
            } else {
                void *bounce = nullptr;
                if (posix_memalign(&bounce, alignment, read_sz) == 0 && bounce) {
                    const bool ok = expert_pack_direct_pread_all(source->fd_direct, bounce, read_sz, aligned_offset);
                    if (ok) {
                        std::memcpy(dst, (const char *)bounce + prefix, sz);
                        result.ok = true;
                        result.direct = true;
                        result.nread = sz;
                        free(bounce);
                        return result;
                    }
                    free(bounce);
                }
            }
        }
    }
#else
    (void)entry;
#endif

    size_t nread = 0;
    result.ok = ggml_cuda_moe_expert_pack_v2_read_debug(
        tensor_name, expert_idx, dst, dst_capacity, &nread);
    result.nread = nread;
    result.direct = false;
    return result;
}

struct expert_pack_v2_shadow_iouring_result {
    bool attempted = false;
    bool completed = false;
    uint64_t staged_bytes = 0;
    uint64_t read_failures = 0;
    uint64_t h2d_failures = 0;
    uint64_t direct_reads = 0;
    uint64_t direct_bytes = 0;
    uint64_t direct_physical_bytes = 0;
    uint64_t direct_fallbacks = 0;
    uint64_t batches = 0;
    uint64_t jobs = 0;
    uint64_t submit_calls = 0;
    uint64_t wait_calls = 0;
    uint64_t cqes = 0;
    uint64_t wait_us = 0;
    uint64_t inflight_sum = 0;
    uint64_t inflight_samples = 0;
    uint64_t inflight_max = 0;
    double read_ms = 0.0;
    double h2d_ms = 0.0;
    double sync_ms = 0.0;
};

static expert_pack_v2_shadow_iouring_result expert_pack_v2_shadow_iouring_stage_candidates(
        expert_pack_v2_shadow_stage_state &s,
        const std::vector<expert_pack_v2_shadow_candidate> &candidates,
        size_t slot_sz,
        cudaStream_t stream) {
    expert_pack_v2_shadow_iouring_result result;
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING) && !defined(_WIN32)
    if (!expert_pack_v2_shadow_iouring_read_enabled() || candidates.empty() ||
            !s.host || !s.dev || slot_sz == 0) {
        return result;
    }
    result.attempted = true;

    const size_t alignment = expert_pack_direct_alignment();
    if (((uintptr_t)s.host % alignment) != 0) {
        result.direct_fallbacks += candidates.size();
        return result;
    }
    const size_t depth = std::min(expert_pack_io_depth(), candidates.size());
    if (depth == 0) {
        return result;
    }

    struct read_plan {
        size_t candidate_idx = 0;
        int fd = -1;
        uint64_t offset = 0;
        size_t read_sz = 0;
        size_t prefix = 0;
        size_t payload_sz = 0;
        bool ok = false;
    };

    std::vector<read_plan> plans;
    plans.reserve(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        const expert_pack_v2_entry *entry = candidates[i].entry;
        expert_pack_source *source = expert_pack_v2_source_for_entry(entry);
        if (!entry || !source || source->fd_direct < 0 || entry->nbytes <= 0) {
            return result;
        }
        const size_t payload_sz = (size_t)entry->nbytes;
        const uint64_t aligned_offset = (entry->offset / alignment) * alignment;
        const size_t prefix = (size_t)(entry->offset - aligned_offset);
        const size_t read_sz = (size_t)align_up_u64((uint64_t)prefix + (uint64_t)payload_sz, alignment);
        if (read_sz == 0 || read_sz > slot_sz || read_sz > (size_t)UINT_MAX) {
            return result;
        }
        void *slot_ptr = (char *)s.host + i * slot_sz;
        if (((uintptr_t)slot_ptr % alignment) != 0) {
            return result;
        }
        plans.push_back({i, source->fd_direct, aligned_offset, read_sz, prefix, payload_sz, false});
    }
    if (plans.empty()) {
        result.completed = true;
        return result;
    }

    pinned_stage_ring local_ring;
    struct scoped_object_io_ring {
        pinned_stage_ring &ring;
        ~scoped_object_io_ring() {
            ggml_moe_common::object_io_ring_release(ring);
        }
    } local_ring_scope{local_ring};
    // Keep the shadow profiler isolated from the main expert-pack SQPOLL rings.
    if (!ggml_moe_common::object_io_ring_ensure(local_ring, depth, 0, nullptr)) {
        result.direct_fallbacks += plans.size();
        return result;
    }

    std::vector<ggml_moe_common::object_io_window_slot> io_window(depth);
    const bool common_object_cqe = common_object_cqe_enabled();

    auto submit_plan = [&](size_t plan_idx, size_t window_slot) -> bool {
        read_plan &plan = plans[plan_idx];
        void *dst = (char *)s.host + plan_idx * slot_sz;
        if (common_object_cqe) {
            const ggml_moe_common::object_io_window_request request{
                plan.fd, dst, plan.read_sz, plan.offset, plan_idx, false,
            };
            return ggml_moe_common::object_io_window_prepare_read(
                local_ring, io_window.data(), io_window.size(), window_slot, request);
        }
        return ggml_moe_common::object_io_ring_prepare_read(
                local_ring, plan.fd, dst, plan.read_sz, plan.offset,
                (uint64_t)plan_idx + 1);
    };

    const auto read_t0 = std::chrono::steady_clock::now();
    size_t next_plan = 0;
    size_t inflight = 0;
    while (next_plan < plans.size() && inflight < depth) {
        if (!submit_plan(next_plan, next_plan)) {
            result.direct_fallbacks += plans.size() - next_plan;
            return result;
        }
        ++next_plan;
        ++inflight;
    }
    if (inflight == 0) {
        return result;
    }
    if (!ggml_moe_common::object_io_ring_submit(local_ring)) {
        result.direct_fallbacks += plans.size();
        return result;
    }
    result.submit_calls += 1;

    size_t completed = 0;
    while (completed < plans.size()) {
        result.inflight_sum += inflight;
        ++result.inflight_samples;
        result.inflight_max = std::max<uint64_t>(result.inflight_max, (uint64_t)inflight);

        ggml_moe_common::object_io_completion completion;
        const auto wait_t0 = std::chrono::steady_clock::now();
        ++result.wait_calls;
        const bool wait_ok = ggml_moe_common::object_io_ring_wait_for(local_ring, completion, 5000);
        const auto wait_t1 = std::chrono::steady_clock::now();
        result.wait_us += (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(wait_t1 - wait_t0).count();
        if (!wait_ok) {
            result.read_failures += plans.size() - completed;
            return result;
        }

        size_t plan_idx = SIZE_MAX;
        size_t window_slot = SIZE_MAX;
        ggml_moe_common::object_io_completion_status status;
        if (common_object_cqe) {
            ggml_moe_common::object_io_window_completion resolved;
            status = ggml_moe_common::object_io_window_resolve(
                local_ring, io_window.data(), io_window.size(), completion, resolved);
            plan_idx = resolved.request_id;
            window_slot = resolved.window_slot;
        } else {
            const uint64_t data = completion.user_data;
            plan_idx = data == 0 ? SIZE_MAX : (size_t)data - 1;
            const bool plan_valid = plan_idx < plans.size();
            const size_t expected_bytes = plan_valid ? plans[plan_idx].read_sz : 0;
            status = ggml_moe_common::object_io_completion_validate(
                local_ring, completion, plan_valid, expected_bytes);
        }
        if (status == ggml_moe_common::object_io_completion_status::complete &&
                plan_idx < plans.size()) {
            plans[plan_idx].ok = true;
            ++result.direct_reads;
            result.direct_bytes += plans[plan_idx].payload_sz;
            result.direct_physical_bytes += plans[plan_idx].read_sz;
        } else {
            ++result.read_failures;
        }
        ++result.cqes;
        ++completed;
        --inflight;

        if (next_plan < plans.size()) {
            const size_t refill_window_slot = common_object_cqe ? window_slot : next_plan;
            if (!submit_plan(next_plan, refill_window_slot)) {
                result.direct_fallbacks += plans.size() - next_plan;
                return result;
            }
            ++next_plan;
            ++inflight;
            if (!ggml_moe_common::object_io_ring_submit(local_ring)) {
                result.direct_fallbacks += plans.size() - completed;
                return result;
            }
            ++result.submit_calls;
        }
    }
    const auto read_t1 = std::chrono::steady_clock::now();
    result.read_ms = std::chrono::duration<double, std::milli>(read_t1 - read_t0).count();
    result.batches = 1;
    result.jobs = plans.size();
    if (common_object_cqe) {
        ggml_moe_common::object_io_window_reset(io_window.data(), io_window.size());
    }

    for (size_t i = 0; i < plans.size(); ++i) {
        const read_plan &plan = plans[i];
        if (!plan.ok) continue;
        const char *payload = (const char *)s.host + i * slot_sz + plan.prefix;
        if (!s.event_failed) {
            cudaEventRecord(s.copy_start, stream);
        }
        if (cudaMemcpyAsync(s.dev, payload, plan.payload_sz, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            ++result.h2d_failures;
            continue;
        }
        if (!s.event_failed) {
            cudaEventRecord(s.copy_done, stream);
            const auto sync_t0 = std::chrono::steady_clock::now();
            if (cudaEventSynchronize(s.copy_done) != cudaSuccess) {
                ++result.h2d_failures;
                continue;
            }
            const auto sync_t1 = std::chrono::steady_clock::now();
            result.sync_ms += std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count();
            float copy_ms = 0.0f;
            if (cudaEventElapsedTime(&copy_ms, s.copy_start, s.copy_done) == cudaSuccess) {
                result.h2d_ms += (double)copy_ms;
            }
        } else {
            const auto sync_t0 = std::chrono::steady_clock::now();
            if (cudaStreamSynchronize(stream) != cudaSuccess) {
                ++result.h2d_failures;
                continue;
            }
            const auto sync_t1 = std::chrono::steady_clock::now();
            result.sync_ms += std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count();
        }
        result.staged_bytes += plan.payload_sz;
    }
    result.completed = true;
    return result;
#else
    (void)s;
    (void)candidates;
    (void)slot_sz;
    (void)stream;
    return result;
#endif
}

struct expert_pack_v2_shadow_stage_metrics {
    double alloc_ms = 0.0;
    double read_ms = 0.0;
    double h2d_ms = 0.0;
    double sync_ms = 0.0;
    double total_ms = 0.0;
    uint64_t staged_bytes = 0;
    uint64_t read_failures = 0;
    uint64_t h2d_failures = 0;
    uint64_t direct_reads = 0;
    uint64_t direct_bytes = 0;
    uint64_t direct_physical_bytes = 0;
    uint64_t direct_fallbacks = 0;
    uint64_t buffered_reads = 0;
    uint64_t buffered_bytes = 0;
    uint64_t iouring_batches = 0;
    uint64_t iouring_jobs = 0;
    uint64_t iouring_submit_calls = 0;
    uint64_t iouring_wait_calls = 0;
    uint64_t iouring_cqes = 0;
    uint64_t iouring_wait_us = 0;
    uint64_t iouring_inflight_sum = 0;
    uint64_t iouring_inflight_samples = 0;
    uint64_t iouring_inflight_max = 0;
    bool alloc_grew = false;
};

static bool expert_pack_v2_shadow_stage_candidates_locked(
        expert_pack_v2_shadow_stage_state &s,
        const std::vector<expert_pack_v2_shadow_candidate> &candidates,
        size_t max_entry_bytes,
        const char *fallback_tensor_name,
        cudaStream_t stream,
        expert_pack_v2_shadow_stage_metrics &m) {
    if (candidates.empty() || max_entry_bytes == 0) return false;

    const auto total_t0 = std::chrono::steady_clock::now();
    const auto alloc_t0 = std::chrono::steady_clock::now();
    const size_t old_host_sz = s.host_sz;
    const size_t old_dev_sz = s.dev_sz;
    const size_t shadow_alignment = expert_pack_direct_alignment();
    const bool use_iouring_shadow = expert_pack_v2_shadow_iouring_read_enabled();
    const size_t shadow_slot_sz = use_iouring_shadow ?
        (size_t)align_up_u64((uint64_t)max_entry_bytes, (uint64_t)shadow_alignment) :
        max_entry_bytes;
    if (shadow_slot_sz == 0 ||
            (use_iouring_shadow && candidates.size() > SIZE_MAX / shadow_slot_sz)) {
        s.alloc_failures.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    const size_t host_need = use_iouring_shadow ? shadow_slot_sz * candidates.size() : max_entry_bytes;
    if (!ensure_host_pinned(s.host, s.host_sz, host_need) ||
            !ensure_dev(s.dev, s.dev_sz, max_entry_bytes)) {
        s.alloc_failures.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    m.alloc_grew = s.host_sz != old_host_sz || s.dev_sz != old_dev_sz;
    if (!s.copy_start && cudaEventCreate(&s.copy_start) != cudaSuccess) {
        s.event_failed = true;
    }
    if (!s.copy_done && cudaEventCreate(&s.copy_done) != cudaSuccess) {
        s.event_failed = true;
    }
    const auto alloc_t1 = std::chrono::steady_clock::now();
    m.alloc_ms = std::chrono::duration<double, std::milli>(alloc_t1 - alloc_t0).count();

    bool used_iouring_shadow = false;
    if (use_iouring_shadow) {
        const expert_pack_v2_shadow_iouring_result io_result =
            expert_pack_v2_shadow_iouring_stage_candidates(s, candidates, shadow_slot_sz, stream);
        if (io_result.attempted && io_result.completed) {
            used_iouring_shadow = true;
            m.staged_bytes += io_result.staged_bytes;
            m.read_failures += io_result.read_failures;
            m.h2d_failures += io_result.h2d_failures;
            m.direct_reads += io_result.direct_reads;
            m.direct_bytes += io_result.direct_bytes;
            m.direct_physical_bytes += io_result.direct_physical_bytes;
            m.direct_fallbacks += io_result.direct_fallbacks;
            m.iouring_batches += io_result.batches;
            m.iouring_jobs += io_result.jobs;
            m.iouring_submit_calls += io_result.submit_calls;
            m.iouring_wait_calls += io_result.wait_calls;
            m.iouring_cqes += io_result.cqes;
            m.iouring_wait_us += io_result.wait_us;
            m.iouring_inflight_sum += io_result.inflight_sum;
            m.iouring_inflight_samples += io_result.inflight_samples;
            m.iouring_inflight_max = std::max<uint64_t>(m.iouring_inflight_max, io_result.inflight_max);
            m.read_ms += io_result.read_ms;
            m.h2d_ms += io_result.h2d_ms;
            m.sync_ms += io_result.sync_ms;
        }
    }

    if (!used_iouring_shadow) for (const expert_pack_v2_shadow_candidate &candidate : candidates) {
        const size_t entry_bytes = (size_t)candidate.entry->nbytes;
        const char *read_tensor_name = candidate.entry && candidate.entry->tensor[0] ?
            candidate.entry->tensor : fallback_tensor_name;
        const auto read_t0 = std::chrono::steady_clock::now();
        const expert_pack_v2_shadow_read_result read_result =
            expert_pack_v2_shadow_read_entry_to_host(
                candidate.entry, read_tensor_name, candidate.expert_idx, s.host, s.host_sz);
        const auto read_t1 = std::chrono::steady_clock::now();
        m.read_ms += std::chrono::duration<double, std::milli>(read_t1 - read_t0).count();
        if (!read_result.ok || read_result.nread != entry_bytes) {
            ++m.read_failures;
            continue;
        }
        if (read_result.direct) {
            ++m.direct_reads;
            m.direct_bytes += entry_bytes;
            m.direct_physical_bytes += read_result.physical_bytes;
        } else {
            ++m.buffered_reads;
            m.buffered_bytes += entry_bytes;
        }
        if (read_result.direct_attempted && !read_result.direct) {
            ++m.direct_fallbacks;
        }
        if (!s.event_failed) {
            cudaEventRecord(s.copy_start, stream);
        }
        if (cudaMemcpyAsync(s.dev, s.host, entry_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            ++m.h2d_failures;
            continue;
        }
        if (!s.event_failed) {
            cudaEventRecord(s.copy_done, stream);
            const auto sync_t0 = std::chrono::steady_clock::now();
            if (cudaEventSynchronize(s.copy_done) != cudaSuccess) {
                ++m.h2d_failures;
                continue;
            }
            const auto sync_t1 = std::chrono::steady_clock::now();
            m.sync_ms += std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count();
            float copy_ms = 0.0f;
            if (cudaEventElapsedTime(&copy_ms, s.copy_start, s.copy_done) == cudaSuccess) {
                m.h2d_ms += (double)copy_ms;
            }
        } else {
            const auto sync_t0 = std::chrono::steady_clock::now();
            if (cudaStreamSynchronize(stream) != cudaSuccess) {
                ++m.h2d_failures;
                continue;
            }
            const auto sync_t1 = std::chrono::steady_clock::now();
            m.sync_ms += std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count();
        }
        m.staged_bytes += (uint64_t)entry_bytes;
    }

    const auto total_t1 = std::chrono::steady_clock::now();
    m.total_ms = std::chrono::duration<double, std::milli>(total_t1 - total_t0).count();
    return true;
}

static void expert_pack_v2_shadow_stage_add_metrics_locked(
        expert_pack_v2_shadow_stage_state &s,
        size_t staged_entries,
        const expert_pack_v2_shadow_stage_metrics &m) {
    s.staged_calls.fetch_add(1, std::memory_order_relaxed);
    s.staged_entries.fetch_add((uint64_t)staged_entries, std::memory_order_relaxed);
    s.staged_bytes.fetch_add(m.staged_bytes, std::memory_order_relaxed);
    s.read_failures.fetch_add(m.read_failures, std::memory_order_relaxed);
    s.h2d_failures.fetch_add(m.h2d_failures, std::memory_order_relaxed);
    s.direct_reads.fetch_add(m.direct_reads, std::memory_order_relaxed);
    s.direct_bytes.fetch_add(m.direct_bytes, std::memory_order_relaxed);
    s.direct_physical_bytes.fetch_add(m.direct_physical_bytes, std::memory_order_relaxed);
    s.direct_fallbacks.fetch_add(m.direct_fallbacks, std::memory_order_relaxed);
    s.buffered_reads.fetch_add(m.buffered_reads, std::memory_order_relaxed);
    s.buffered_bytes.fetch_add(m.buffered_bytes, std::memory_order_relaxed);
    s.io_stats.batches.fetch_add(m.iouring_batches, std::memory_order_relaxed);
    s.iouring_jobs.fetch_add(m.iouring_jobs, std::memory_order_relaxed);
    s.io_stats.submit_calls.fetch_add(m.iouring_submit_calls, std::memory_order_relaxed);
    s.io_stats.wait_calls.fetch_add(m.iouring_wait_calls, std::memory_order_relaxed);
    s.io_stats.cqes.fetch_add(m.iouring_cqes, std::memory_order_relaxed);
    s.io_stats.wait_us.fetch_add(m.iouring_wait_us, std::memory_order_relaxed);
    s.io_stats.inflight_sum.fetch_add(m.iouring_inflight_sum, std::memory_order_relaxed);
    s.io_stats.inflight_samples.fetch_add(m.iouring_inflight_samples, std::memory_order_relaxed);
    expert_pack_atomic_max(s.io_stats.inflight_max, m.iouring_inflight_max);
    s.alloc_ms += m.alloc_ms;
    s.read_ms += m.read_ms;
    s.h2d_ms += m.h2d_ms;
    s.sync_ms += m.sync_ms;
    s.total_ms += m.total_ms;
}

static void expert_pack_v2_shadow_coalesce_flush_locked(expert_pack_v2_shadow_stage_state &s) {
    expert_pack_v2_shadow_coalesce_group &g = g_expert_pack_v2_shadow_coalesce;
    if (!g.active) return;
    if (g.candidates.empty()) {
        expert_pack_v2_shadow_coalesce_reset(g);
        return;
    }

    char tensor_name[128];
    std::snprintf(tensor_name, sizeof(tensor_name), "blk.%d.coalesced_v2_shadow", g.layer);
    const std::string role = expert_pack_v2_shadow_coalesce_role(g);
    const size_t staged_count = g.candidates.size();
    const uint64_t source_calls = g.source_calls;
    const uint64_t seq = g.first_seq;
    expert_pack_v2_shadow_stage_metrics m;
    if (!expert_pack_v2_shadow_stage_candidates_locked(
                s, g.candidates, g.max_entry_bytes, tensor_name, g.stream, m)) {
        expert_pack_v2_shadow_coalesce_reset(g);
        return;
    }

    expert_pack_v2_shadow_stage_add_metrics_locked(s, staged_count, m);
    s.coalesced_groups.fetch_add(1, std::memory_order_relaxed);
    s.coalesced_source_calls.fetch_add(source_calls, std::memory_order_relaxed);
    s.coalesced_entries.fetch_add((uint64_t)staged_count, std::memory_order_relaxed);

    const char *path = expert_pack_v2_shadow_stage_path();
    if (path && path[0]) {
        FILE *f = std::fopen(path, "a");
        if (f) {
            expert_pack_v2_shadow_stage_write_header(f, s);
            std::fprintf(f,
                "%llu,%s,%s,%s,%d,%zu,%ld,%ld,%zu,%d,%d,%d,%d,%zu,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%zu,%zu,%zu,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
                (unsigned long long)seq,
                "decode_coalesced_same_layer",
                role.c_str(),
                tensor_name,
                -1,
                (size_t)0,
                (long)g.layer,
                (long)source_calls,
                (size_t)0,
                g.n_active,
                g.cache_hits,
                g.covered,
                g.covered_cache_hits,
                staged_count,
                g.reject_no_manifest,
                g.reject_not_allowed,
                g.reject_no_entry,
                g.reject_unsupported,
                g.reject_manifest_mismatch,
                g.reject_shape_mismatch,
                g.reject_not_smaller,
                (unsigned long long)g.covered_bytes,
                (unsigned long long)g.covered_saved_bytes,
                (unsigned long long)m.staged_bytes,
                (unsigned long long)g.staged_saved_bytes,
                g.max_entry_bytes,
                s.host_sz,
                s.dev_sz,
                m.alloc_grew ? 1 : 0,
                m.alloc_ms,
                m.read_ms,
                m.h2d_ms,
                m.sync_ms,
                m.total_ms,
                (unsigned long long)m.read_failures,
                (unsigned long long)m.h2d_failures,
                (unsigned long long)m.direct_reads,
                (unsigned long long)m.direct_bytes,
                (unsigned long long)m.direct_physical_bytes,
                (unsigned long long)m.direct_fallbacks,
                (unsigned long long)m.buffered_reads,
                (unsigned long long)m.buffered_bytes,
                (unsigned long long)m.iouring_batches,
                (unsigned long long)m.iouring_jobs,
                (unsigned long long)m.iouring_submit_calls,
                (unsigned long long)m.iouring_wait_calls,
                (unsigned long long)m.iouring_cqes,
                (unsigned long long)m.iouring_wait_us,
                (unsigned long long)m.iouring_inflight_sum,
                (unsigned long long)m.iouring_inflight_samples,
                (unsigned long long)m.iouring_inflight_max);
            std::fclose(f);
        }
    }
    expert_pack_v2_shadow_coalesce_reset(g);
}

static bool expert_pack_v2_shadow_coalesce_observe(
        expert_pack_v2_shadow_stage_state &s,
        uint64_t call_idx,
        const char *phase,
        const char *role,
        const char *tensor_name,
        int n_active,
        int cache_hits,
        int covered,
        int covered_cache_hits,
        int reject_no_manifest,
        int reject_not_allowed,
        int reject_no_entry,
        int reject_unsupported,
        int reject_manifest_mismatch,
        int reject_shape_mismatch,
        int reject_not_smaller,
        uint64_t covered_bytes,
        uint64_t covered_saved_bytes,
        uint64_t staged_saved_bytes,
        size_t max_entry_bytes,
        const std::vector<expert_pack_v2_shadow_candidate> &candidates,
        cudaStream_t stream) {
    if (!expert_pack_v2_shadow_coalesce_same_layer_enabled() ||
            !expert_pack_v2_shadow_iouring_read_enabled() ||
            !phase || std::strncmp(phase, "decode", 6) != 0) {
        return false;
    }

    const int layer = expert_pack_v2_shadow_tensor_layer(tensor_name);
    const bool is_down = role && std::strcmp(role, "down") == 0;
    std::lock_guard<std::mutex> lk(s.mu);
    expert_pack_v2_shadow_coalesce_group &g = g_expert_pack_v2_shadow_coalesce;
    if (layer < 0) {
        expert_pack_v2_shadow_coalesce_flush_locked(s);
        return false;
    }
    if (g.active && (g.layer != layer || g.stream != stream)) {
        expert_pack_v2_shadow_coalesce_flush_locked(s);
    }
    if (candidates.empty()) {
        if (g.active && g.layer == layer && is_down) {
            expert_pack_v2_shadow_coalesce_flush_locked(s);
        }
        return false;
    }

    if (!g.active) {
        expert_pack_v2_shadow_coalesce_reset(g);
        g.active = true;
        g.layer = layer;
        g.stream = stream;
        g.first_seq = call_idx;
    }
    g.last_seq = call_idx;
    ++g.source_calls;
    if (role && std::strcmp(role, "up") == 0) {
        g.has_up = true;
    } else if (role && std::strcmp(role, "gate") == 0) {
        g.has_gate = true;
    } else if (is_down) {
        g.has_down = true;
    }
    g.n_active += n_active;
    g.cache_hits += cache_hits;
    g.covered += covered;
    g.covered_cache_hits += covered_cache_hits;
    g.reject_no_manifest += reject_no_manifest;
    g.reject_not_allowed += reject_not_allowed;
    g.reject_no_entry += reject_no_entry;
    g.reject_unsupported += reject_unsupported;
    g.reject_manifest_mismatch += reject_manifest_mismatch;
    g.reject_shape_mismatch += reject_shape_mismatch;
    g.reject_not_smaller += reject_not_smaller;
    g.covered_bytes += covered_bytes;
    g.covered_saved_bytes += covered_saved_bytes;
    g.staged_saved_bytes += staged_saved_bytes;
    g.max_entry_bytes = std::max(g.max_entry_bytes, max_entry_bytes);
    g.candidates.insert(g.candidates.end(), candidates.begin(), candidates.end());

    if (is_down) {
        expert_pack_v2_shadow_coalesce_flush_locked(s);
    }
    return true;
}

static void expert_pack_v2_shadow_stage_record(
        const char *phase,
        const char *role,
        const char *tensor_name,
        int logical_type,
        size_t logical_nbytes,
        int64_t logical_ne00,
        int64_t logical_ne01,
        size_t logical_nb01,
        batch_vram_cache *cache,
        const int *active_experts,
        int n_active,
        cudaStream_t stream) {
    if (!expert_pack_v2_shadow_stage_enabled()) return;
    if (!phase || !phase[0] || std::strncmp(phase, "decode", 6) != 0) {
        if (!expert_pack_v2_shadow_stage_prompt_enabled()) return;
    }
    if (!tensor_name || !tensor_name[0] || !active_experts || n_active <= 0 ||
            logical_nbytes == 0) {
        return;
    }

    static std::atomic<uint64_t> observed_calls{0};
    const uint64_t call_idx = observed_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    const uint64_t max_calls = expert_pack_v2_shadow_stage_max_calls();
    if (max_calls > 0 && call_idx > max_calls) return;

    expert_pack_v2_override_preflight_init_once();
    if (!g_expert_pack_v2_override_preflight.enabled) return;

    std::vector<expert_pack_v2_shadow_candidate> candidates;
    candidates.reserve((size_t)n_active);
    int cache_hits = 0;
    int covered = 0;
    int covered_cache_hits = 0;
    int reject_no_manifest = 0;
    int reject_not_allowed = 0;
    int reject_no_entry = 0;
    int reject_unsupported = 0;
    int reject_manifest_mismatch = 0;
    int reject_shape_mismatch = 0;
    int reject_not_smaller = 0;
    uint64_t covered_bytes = 0;
    uint64_t covered_saved_bytes = 0;
    uint64_t staged_saved_bytes = 0;
    size_t max_entry_bytes = 0;

    for (int j = 0; j < n_active; ++j) {
        const int expert_idx = active_experts[j];
        const bool cache_hit = cache && batch_cache_find_slot(cache, batch_key_hash(tensor_name, expert_idx)) >= 0;
        if (cache_hit) {
            ++cache_hits;
        }
        if (!g_expert_pack_v2_override_preflight.manifest_loaded) {
            ++reject_no_manifest;
            continue;
        }
        const auto allowed_it = g_expert_pack_v2_override_preflight.manifest.find(
            expert_pack_v2_override_key(tensor_name, expert_idx));
        if (allowed_it == g_expert_pack_v2_override_preflight.manifest.end()) {
            ++reject_not_allowed;
            continue;
        }
        const expert_pack_v2_override_manifest_entry &allowed = allowed_it->second;
        const expert_pack_v2_entry *entry = expert_pack_v2_lookup(tensor_name, expert_idx);
        if (!entry) {
            ++reject_no_entry;
            continue;
        }
        if (entry->packed_type != allowed.packed_type ||
                (size_t)entry->nbytes != allowed.packed_nbytes ||
                entry->ne00 != allowed.ne00 ||
                entry->ne01 != allowed.ne01 ||
                (size_t)entry->nb01 != allowed.nb01) {
            ++reject_manifest_mismatch;
            continue;
        }
        if (!expert_pack_v2_packed_type_supported(entry->packed_type)) {
            ++reject_unsupported;
            continue;
        }
        if (entry->ne00 != logical_ne00 || entry->ne01 != logical_ne01) {
            ++reject_shape_mismatch;
            continue;
        }
        if ((size_t)entry->nbytes >= logical_nbytes) {
            ++reject_not_smaller;
            continue;
        }

        ++covered;
        covered_bytes += (uint64_t)entry->nbytes;
        covered_saved_bytes += (uint64_t)(logical_nbytes - (size_t)entry->nbytes);
        if (cache_hit) {
            ++covered_cache_hits;
            continue;
        }
        expert_pack_v2_shadow_candidate candidate;
        candidate.expert_idx = expert_idx;
        candidate.entry = entry;
        candidate.cache_hit = false;
        candidates.push_back(candidate);
        max_entry_bytes = std::max(max_entry_bytes, (size_t)entry->nbytes);
        staged_saved_bytes += (uint64_t)(logical_nbytes - (size_t)entry->nbytes);
    }

    expert_pack_v2_shadow_stage_state &s = g_expert_pack_v2_shadow_stage;
    s.calls.fetch_add(1, std::memory_order_relaxed);
    s.active_entries.fetch_add((uint64_t)n_active, std::memory_order_relaxed);
    s.covered_entries.fetch_add((uint64_t)covered, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(s.mu);
        if (!s.report_registered) {
            std::atexit(expert_pack_v2_shadow_stage_report);
            s.report_registered = true;
        }
    }
    if (expert_pack_v2_shadow_coalesce_observe(
                s,
                call_idx,
                phase,
                role,
                tensor_name,
                n_active,
                cache_hits,
                covered,
                covered_cache_hits,
                reject_no_manifest,
                reject_not_allowed,
                reject_no_entry,
                reject_unsupported,
                reject_manifest_mismatch,
                reject_shape_mismatch,
                reject_not_smaller,
                covered_bytes,
                covered_saved_bytes,
                staged_saved_bytes,
                max_entry_bytes,
                candidates,
                stream)) {
        return;
    }
    if (candidates.empty()) {
        if (covered > 0) {
            std::lock_guard<std::mutex> lk(s.mu);
            const char *path = expert_pack_v2_shadow_stage_path();
            if (path && path[0]) {
                FILE *f = std::fopen(path, "a");
                if (f) {
                    expert_pack_v2_shadow_stage_write_header(f, s);
                    std::fprintf(f,
                        "%llu,%s,%s,%s,%d,%zu,%ld,%ld,%zu,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,0,%llu,%zu,%zu,%zu,0,0.000000,0.000000,0.000000,0.000000,0.000000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n",
                        (unsigned long long)call_idx,
                        phase ? phase : "",
                        role ? role : "",
                        tensor_name,
                        logical_type,
                        logical_nbytes,
                        (long)logical_ne00,
                        (long)logical_ne01,
                        logical_nb01,
                        n_active,
                        cache_hits,
                        covered,
                        covered_cache_hits,
                        0,
                        reject_no_manifest,
                        reject_not_allowed,
                        reject_no_entry,
                        reject_unsupported,
                        reject_manifest_mismatch,
                        reject_shape_mismatch,
                        reject_not_smaller,
                        (unsigned long long)covered_bytes,
                        (unsigned long long)covered_saved_bytes,
                        (unsigned long long)staged_saved_bytes,
                        max_entry_bytes,
                        s.host_sz,
                        s.dev_sz);
                    std::fclose(f);
                }
            }
        }
        return;
    }

    double alloc_ms = 0.0;
    double read_ms = 0.0;
    double h2d_ms = 0.0;
    double sync_ms = 0.0;
    double total_ms = 0.0;
    uint64_t staged_bytes = 0;
    uint64_t read_failures = 0;
    uint64_t h2d_failures = 0;
    uint64_t direct_reads = 0;
    uint64_t direct_bytes = 0;
    uint64_t direct_physical_bytes = 0;
    uint64_t direct_fallbacks = 0;
    uint64_t buffered_reads = 0;
    uint64_t buffered_bytes = 0;
    uint64_t iouring_batches = 0;
    uint64_t iouring_jobs = 0;
    uint64_t iouring_submit_calls = 0;
    uint64_t iouring_wait_calls = 0;
    uint64_t iouring_cqes = 0;
    uint64_t iouring_wait_us = 0;
    uint64_t iouring_inflight_sum = 0;
    uint64_t iouring_inflight_samples = 0;
    uint64_t iouring_inflight_max = 0;
    bool alloc_grew = false;

    const auto total_t0 = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lk(s.mu);
        const auto alloc_t0 = std::chrono::steady_clock::now();
        const size_t old_host_sz = s.host_sz;
        const size_t old_dev_sz = s.dev_sz;
        const size_t shadow_alignment = expert_pack_direct_alignment();
        const bool use_iouring_shadow = expert_pack_v2_shadow_iouring_read_enabled();
        const size_t shadow_slot_sz = use_iouring_shadow ?
            (size_t)align_up_u64((uint64_t)max_entry_bytes, (uint64_t)shadow_alignment) :
            max_entry_bytes;
        if (shadow_slot_sz == 0 ||
                (use_iouring_shadow && candidates.size() > SIZE_MAX / shadow_slot_sz)) {
            s.alloc_failures.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        const size_t host_need = use_iouring_shadow ? shadow_slot_sz * candidates.size() : max_entry_bytes;
        if (!ensure_host_pinned(s.host, s.host_sz, host_need) ||
                !ensure_dev(s.dev, s.dev_sz, max_entry_bytes)) {
            s.alloc_failures.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        alloc_grew = s.host_sz != old_host_sz || s.dev_sz != old_dev_sz;
        if (!s.copy_start && cudaEventCreate(&s.copy_start) != cudaSuccess) {
            s.event_failed = true;
        }
        if (!s.copy_done && cudaEventCreate(&s.copy_done) != cudaSuccess) {
            s.event_failed = true;
        }
        const auto alloc_t1 = std::chrono::steady_clock::now();
        alloc_ms = std::chrono::duration<double, std::milli>(alloc_t1 - alloc_t0).count();

        bool used_iouring_shadow = false;
        if (use_iouring_shadow) {
            const expert_pack_v2_shadow_iouring_result io_result =
                expert_pack_v2_shadow_iouring_stage_candidates(s, candidates, shadow_slot_sz, stream);
            if (io_result.attempted && io_result.completed) {
                used_iouring_shadow = true;
                staged_bytes += io_result.staged_bytes;
                read_failures += io_result.read_failures;
                h2d_failures += io_result.h2d_failures;
                direct_reads += io_result.direct_reads;
                direct_bytes += io_result.direct_bytes;
                direct_physical_bytes += io_result.direct_physical_bytes;
                direct_fallbacks += io_result.direct_fallbacks;
                iouring_batches += io_result.batches;
                iouring_jobs += io_result.jobs;
                iouring_submit_calls += io_result.submit_calls;
                iouring_wait_calls += io_result.wait_calls;
                iouring_cqes += io_result.cqes;
                iouring_wait_us += io_result.wait_us;
                iouring_inflight_sum += io_result.inflight_sum;
                iouring_inflight_samples += io_result.inflight_samples;
                iouring_inflight_max = std::max<uint64_t>(iouring_inflight_max, io_result.inflight_max);
                read_ms += io_result.read_ms;
                h2d_ms += io_result.h2d_ms;
                sync_ms += io_result.sync_ms;
            }
        }

        if (!used_iouring_shadow) for (const expert_pack_v2_shadow_candidate &candidate : candidates) {
            const size_t entry_bytes = (size_t)candidate.entry->nbytes;
            const auto read_t0 = std::chrono::steady_clock::now();
            const expert_pack_v2_shadow_read_result read_result =
                expert_pack_v2_shadow_read_entry_to_host(
                    candidate.entry, tensor_name, candidate.expert_idx, s.host, s.host_sz);
            const auto read_t1 = std::chrono::steady_clock::now();
            read_ms += std::chrono::duration<double, std::milli>(read_t1 - read_t0).count();
            if (!read_result.ok || read_result.nread != entry_bytes) {
                ++read_failures;
                continue;
            }
            if (read_result.direct) {
                ++direct_reads;
                direct_bytes += entry_bytes;
                direct_physical_bytes += read_result.physical_bytes;
            } else {
                ++buffered_reads;
                buffered_bytes += entry_bytes;
            }
            if (read_result.direct_attempted && !read_result.direct) {
                ++direct_fallbacks;
            }
            if (!s.event_failed) {
                cudaEventRecord(s.copy_start, stream);
            }
            if (cudaMemcpyAsync(s.dev, s.host, entry_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
                ++h2d_failures;
                continue;
            }
            if (!s.event_failed) {
                cudaEventRecord(s.copy_done, stream);
                const auto sync_t0 = std::chrono::steady_clock::now();
                if (cudaEventSynchronize(s.copy_done) != cudaSuccess) {
                    ++h2d_failures;
                    continue;
                }
                const auto sync_t1 = std::chrono::steady_clock::now();
                sync_ms += std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count();
                float copy_ms = 0.0f;
                if (cudaEventElapsedTime(&copy_ms, s.copy_start, s.copy_done) == cudaSuccess) {
                    h2d_ms += (double)copy_ms;
                }
            } else {
                const auto sync_t0 = std::chrono::steady_clock::now();
                if (cudaStreamSynchronize(stream) != cudaSuccess) {
                    ++h2d_failures;
                    continue;
                }
                const auto sync_t1 = std::chrono::steady_clock::now();
                sync_ms += std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count();
            }
            staged_bytes += (uint64_t)entry_bytes;
        }
    }
    const auto total_t1 = std::chrono::steady_clock::now();
    total_ms = std::chrono::duration<double, std::milli>(total_t1 - total_t0).count();

    s.staged_calls.fetch_add(1, std::memory_order_relaxed);
    s.staged_entries.fetch_add((uint64_t)candidates.size(), std::memory_order_relaxed);
    s.staged_bytes.fetch_add(staged_bytes, std::memory_order_relaxed);
    s.read_failures.fetch_add(read_failures, std::memory_order_relaxed);
    s.h2d_failures.fetch_add(h2d_failures, std::memory_order_relaxed);
    s.direct_reads.fetch_add(direct_reads, std::memory_order_relaxed);
    s.direct_bytes.fetch_add(direct_bytes, std::memory_order_relaxed);
    s.direct_physical_bytes.fetch_add(direct_physical_bytes, std::memory_order_relaxed);
    s.direct_fallbacks.fetch_add(direct_fallbacks, std::memory_order_relaxed);
    s.buffered_reads.fetch_add(buffered_reads, std::memory_order_relaxed);
    s.buffered_bytes.fetch_add(buffered_bytes, std::memory_order_relaxed);
    s.io_stats.batches.fetch_add(iouring_batches, std::memory_order_relaxed);
    s.iouring_jobs.fetch_add(iouring_jobs, std::memory_order_relaxed);
    s.io_stats.submit_calls.fetch_add(iouring_submit_calls, std::memory_order_relaxed);
    s.io_stats.wait_calls.fetch_add(iouring_wait_calls, std::memory_order_relaxed);
    s.io_stats.cqes.fetch_add(iouring_cqes, std::memory_order_relaxed);
    s.io_stats.wait_us.fetch_add(iouring_wait_us, std::memory_order_relaxed);
    s.io_stats.inflight_sum.fetch_add(iouring_inflight_sum, std::memory_order_relaxed);
    s.io_stats.inflight_samples.fetch_add(iouring_inflight_samples, std::memory_order_relaxed);
    expert_pack_atomic_max(s.io_stats.inflight_max, iouring_inflight_max);
    {
        std::lock_guard<std::mutex> lk(s.mu);
        s.alloc_ms += alloc_ms;
        s.read_ms += read_ms;
        s.h2d_ms += h2d_ms;
        s.sync_ms += sync_ms;
        s.total_ms += total_ms;
        const char *path = expert_pack_v2_shadow_stage_path();
        if (path && path[0]) {
            FILE *f = std::fopen(path, "a");
            if (f) {
                expert_pack_v2_shadow_stage_write_header(f, s);
                std::fprintf(f,
                    "%llu,%s,%s,%s,%d,%zu,%ld,%ld,%zu,%d,%d,%d,%d,%zu,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%zu,%zu,%zu,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
                    (unsigned long long)call_idx,
                    phase ? phase : "",
                    role ? role : "",
                    tensor_name,
                    logical_type,
                    logical_nbytes,
                    (long)logical_ne00,
                    (long)logical_ne01,
                    logical_nb01,
                    n_active,
                    cache_hits,
                    covered,
                    covered_cache_hits,
                    candidates.size(),
                    reject_no_manifest,
                    reject_not_allowed,
                    reject_no_entry,
                    reject_unsupported,
                    reject_manifest_mismatch,
                    reject_shape_mismatch,
                    reject_not_smaller,
                    (unsigned long long)covered_bytes,
                    (unsigned long long)covered_saved_bytes,
                    (unsigned long long)staged_bytes,
                    (unsigned long long)staged_saved_bytes,
                    max_entry_bytes,
                    s.host_sz,
                    s.dev_sz,
                    alloc_grew ? 1 : 0,
                    alloc_ms,
                    read_ms,
                    h2d_ms,
                    sync_ms,
                    total_ms,
                    (unsigned long long)read_failures,
                    (unsigned long long)h2d_failures,
                    (unsigned long long)direct_reads,
                    (unsigned long long)direct_bytes,
                    (unsigned long long)direct_physical_bytes,
                    (unsigned long long)direct_fallbacks,
                    (unsigned long long)buffered_reads,
                    (unsigned long long)buffered_bytes,
                    (unsigned long long)iouring_batches,
                    (unsigned long long)iouring_jobs,
                    (unsigned long long)iouring_submit_calls,
                    (unsigned long long)iouring_wait_calls,
                    (unsigned long long)iouring_cqes,
                    (unsigned long long)iouring_wait_us,
                    (unsigned long long)iouring_inflight_sum,
                    (unsigned long long)iouring_inflight_samples,
                    (unsigned long long)iouring_inflight_max);
                std::fclose(f);
            }
        }
    }
}




extern "C" bool ggml_cuda_moe_stream_handoff_upload(
        const float *host_ptr,
        int64_t ne01,
        int64_t dst_cols) {
    // The experimental batch producer/cache path is retired.
    return false;
}



static __device__ __forceinline__ int moe_q8k_nearest_even_int(float fval) {
    return __float2int_rn(fval);
}

static __global__ void moe_quantize_row_q8_k_kernel(const float * x, block_q8_K * y, int nblocks) {
    const int ib = blockIdx.x;
    if (ib >= nblocks || threadIdx.x != 0) {
        return;
    }

    const float * xb = x + (size_t)ib * QK_K;
    block_q8_K * yb = y + ib;

    float amax = 0.0f;
    for (int j = 0; j < QK_K; ++j) {
        const float ax = fabsf(xb[j]);
        if (ax > amax) {
            amax = ax;
        }
    }

    if (amax == 0.0f) {
        yb->d = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            yb->qs[j] = 0;
        }
        for (int j = 0; j < QK_K/16; ++j) {
            yb->bsums[j] = 0;
        }
        return;
    }

    const float d = amax / 127.0f;
    const float id = 127.0f / amax;
    for (int j = 0; j < QK_K; ++j) {
        int v = moe_q8k_nearest_even_int(id * xb[j]);
        if (v > 127) {
            v = 127;
        } else if (v < -128) {
            v = -128;
        }
        yb->qs[j] = (int8_t)v;
    }
    for (int j = 0; j < QK_K/16; ++j) {
        int sum = 0;
        for (int ii = 0; ii < 16; ++ii) {
            sum += yb->qs[j*16 + ii];
        }
        yb->bsums[j] = (int16_t)sum;
    }
    yb->d = d;
}

































static float moe_iq2_selftest_value_for_prompt(int col) {
    const int a = col * 19 + (col / 7) * 3;
    const float wave = std::cos(0.017f * (float)(col + 5));
    return 0.11f * (float)((a % 29) - 14) + wave;
}

static void moe_iq2_selftest_fill_row(uint8_t * dst, int row, int k) {
    const int n_blocks = k / QK_K;
    block_iq2_s * blocks = (block_iq2_s *)dst;
    for (int ib = 0; ib < n_blocks; ++ib) {
        block_iq2_s & b = blocks[ib];
        b.d = ggml_fp32_to_fp16(0.0625f * (float)(row + 2 + ib));
        for (int i = 0; i < QK_K/8; ++i) {
            b.qs[i] = (uint8_t)((row * 37 + ib * 17 + i * 11) & 0xff);
            b.qs[QK_K/8 + i] = (uint8_t)((row * 19 + ib * 13 + i * 7) & 0xff);
        }
        for (int i = 0; i < QK_K/32; ++i) {
            b.qh[i] = (uint8_t)((row * 29 + ib * 5 + i * 3) & 0xff);
            const int lo = (row + ib + i) & 0x0f;
            const int hi = (row * 3 + ib + i * 2) & 0x0f;
            b.scales[i] = (uint8_t)(lo | (hi << 4));
        }
    }
}









extern "C" bool ggml_cuda_moe_iq2_q8k_r8_selftest(void) {
    std::fprintf(stderr, "[moe_stream_batch] IQ2_S Q8_K_R8 selftest unavailable in vendor correctness path\n");
    return false;
}

extern "C" bool ggml_cuda_moe_iq2_xxs_q8k_selftest(void) {
    // The opt-in runtime streaming compute path is retired.
    return false;
}





extern "C" bool ggml_cuda_moe_iq2_prompt_replay(
        int src0_type_int,
        const void *up_expert,
        const void *gate_expert,
        int64_t ne01,
        int64_t ne00,
        size_t nb01,
        const float *src1_row_f32,
        int64_t col,
        int unary_op,
        float limit,
        ggml_moe_iq2_replay_result *out) {
    // The opt-in runtime streaming compute path is retired.
    return false;
}

static bool moe_q8k_selftest(cudaStream_t stream) {
    constexpr int nblocks = 3;
    std::vector<float> host_x((size_t)nblocks * QK_K);
    for (size_t i = 0; i < host_x.size(); ++i) {
        const int v = (int)((i * 37 + 11) % 257) - 128;
        host_x[i] = (float)v / 17.0f + ((i & 7) == 0 ? 0.03125f : -0.015625f);
    }

    std::vector<block_q8_K> expect(nblocks);
    std::vector<block_q8_K> got(nblocks);
    std::memset(expect.data(), 0, expect.size() * sizeof(block_q8_K));
    std::memset(got.data(), 0, got.size() * sizeof(block_q8_K));
    quantize_row_q8_K_ref(host_x.data(), expect.data(), (int64_t)host_x.size());

    float * d_x = nullptr;
    block_q8_K * d_y = nullptr;
    const size_t x_bytes = host_x.size() * sizeof(float);
    const size_t y_bytes = got.size() * sizeof(block_q8_K);
    if (cudaMalloc(&d_x, x_bytes) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&d_y, y_bytes) != cudaSuccess) {
        cudaFree(d_x);
        return false;
    }
    bool ok = true;
    ok = ok && cudaMemcpyAsync(d_x, host_x.data(), x_bytes, cudaMemcpyHostToDevice, stream) == cudaSuccess;
    ok = ok && cudaMemsetAsync(d_y, 0, y_bytes, stream) == cudaSuccess;
    if (ok) {
        moe_quantize_row_q8_k_kernel<<<nblocks, 1, 0, stream>>>(d_x, d_y, nblocks);
        ok = cudaGetLastError() == cudaSuccess;
    }
    ok = ok && cudaMemcpyAsync(got.data(), d_y, y_bytes, cudaMemcpyDeviceToHost, stream) == cudaSuccess;
    ok = ok && cudaStreamSynchronize(stream) == cudaSuccess;
    cudaFree(d_y);
    cudaFree(d_x);

    if (ok && std::memcmp(expect.data(), got.data(), y_bytes) != 0) {
        ok = false;
    }
    std::fprintf(stderr, "[moe_stream_batch] Q8_K CUDA quant selftest: %s\n", ok ? "ok" : "failed");
    return ok;
}







static std::atomic<int> g_q4_down_parity_calls{0};
static std::mutex g_q4_down_parity_seen_mu;
static std::unordered_set<std::string> g_q4_down_parity_seen_tensors;



























static std::atomic<int> g_mxfp4_down_probe_calls{0};
static std::mutex g_mxfp4_down_probe_out_mu;








































static bool init_batch_once() {
    if (g_batch_inited.load(std::memory_order_acquire)) return g_batch.stream != nullptr;
    std::lock_guard<std::mutex> lk(g_batch_mu);
    if (g_batch_inited.load(std::memory_order_acquire)) return g_batch.stream != nullptr;
    const char *env = std::getenv("GGML_MOE_STREAM");
    if (env && env[0] && env[0] != '0' && cudaSetDevice(0) == cudaSuccess) {
        if (cudaStreamCreate(&g_batch.stream) != cudaSuccess) {
            g_batch.stream = nullptr;
        }
        if (g_batch.stream && cudaStreamCreate(&g_batch.prefetch_stream) != cudaSuccess) {
            g_batch.prefetch_stream = nullptr;
        }
        if (g_batch.stream && cudaStreamCreateWithFlags(&g_batch.up_stream, cudaStreamNonBlocking) != cudaSuccess) {
            g_batch.up_stream = nullptr;
        }
        if (g_batch.stream && cudaStreamCreateWithFlags(&g_batch.gate_stream, cudaStreamNonBlocking) != cudaSuccess) {
            g_batch.gate_stream = nullptr;
        }
        if (g_batch.stream && cudaStreamCreateWithFlags(&g_batch.up_copy_stream, cudaStreamNonBlocking) != cudaSuccess) {
            g_batch.up_copy_stream = nullptr;
        }
        if (g_batch.stream && cudaStreamCreateWithFlags(&g_batch.gate_copy_stream, cudaStreamNonBlocking) != cudaSuccess) {
            g_batch.gate_copy_stream = nullptr;
        }
        if (g_batch.stream) {
            cudaEventCreateWithFlags(&g_batch.ev_stage_ready, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&g_batch.ev_up_done, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&g_batch.ev_gate_done, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&g_batch.ev_up_copy_aux_done, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&g_batch.ev_gate_copy_aux_done, cudaEventDisableTiming);
            if (!q4_graph_descriptor_buffer_init()) {
                std::fprintf(stderr,
                    "[moe_stream_batch] fixed-capacity q4 descriptor buffer unavailable; graph descriptor path disabled\n");
            }
        }
        const char *q8k_selftest_env = std::getenv("GGML_MOE_Q8K_SELFTEST");
        if (g_batch.stream && q8k_selftest_env && q8k_selftest_env[0] && q8k_selftest_env[0] != '0') {
            if (!moe_q8k_selftest(g_batch.stream)) {
                std::fprintf(stderr, "[moe_stream_batch] Q8_K CUDA quant selftest failed; exact prompt Q8_K path disabled\n");
            }
        }
        const char *prof_env = std::getenv("GGML_MOE_BATCH_PROFILE");
        const char *layer_prof_env = std::getenv("GGML_MOE_UP_GATE_LAYER_PROFILE");
        const bool graph_compute_profile = q4_graph_compute_region_profile_enabled();
        g_up_gate_layer_profile_enabled = layer_prof_env && layer_prof_env[0] && layer_prof_env[0] != '0';
        if (g_up_gate_layer_profile_enabled) {
            std::fprintf(stderr, "[moe_stream_batch] up/gate layer profile enabled\n");
        }
        g_bprof.enabled = (prof_env && prof_env[0] && prof_env[0] != '0') || graph_compute_profile;
        g_uprof.enabled = g_bprof.enabled || up_gate_profile_csv_enabled() || g_up_gate_layer_profile_enabled;
        std::atexit(current_down_overlap_report_atexit);
        if (g_batch.stream && (g_bprof.enabled || g_uprof.enabled)) {
            cudaEventCreate(&g_batch.ev_start);
            cudaEventCreate(&g_batch.ev_stage);
            cudaEventCreate(&g_batch.ev_quant);
            cudaEventCreate(&g_batch.ev_kernel);
            cudaEventCreate(&g_batch.ev_d2h);
            cudaEventCreate(&g_batch.ev_up_start);
            cudaEventCreate(&g_batch.ev_gate_work_start);
            cudaEventCreate(&g_batch.ev_up_compute_start);
            cudaEventCreate(&g_batch.ev_gate_compute_start);
            cudaEventCreate(&g_batch.ev_gate_start);
            cudaEventCreate(&g_batch.ev_up);
            cudaEventCreate(&g_batch.ev_gate);
            if (g_bprof.enabled) {
                std::atexit(batch_profile_report_atexit);
            }
            if (g_uprof.enabled) {
                std::atexit(up_gate_profile_report_atexit);
            }
            if (graph_compute_profile) {
                std::atexit(q4_graph_compute_region_profile_report_atexit);
            }
            if (g_up_gate_layer_profile_enabled) {
                std::atexit(up_gate_layer_profile_report_atexit);
            }
        }
    }
    g_batch_inited.store(true, std::memory_order_release);
    return g_batch.stream != nullptr;
}

extern "C" bool ggml_cuda_moe_stream_stage_active_experts(
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes,
    const int64_t *matrix_row_counts,
    void *dst_staged,
    size_t dst_staged_bytes,
    cudaStream_t stream) {
    // The experimental object/cache or probe path is retired.
    return false;
}

extern "C" bool ggml_cuda_moe_stream_preload_tensor(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes) {
    // The experimental batch producer/cache path is retired.
    return false;
}

extern "C" bool ggml_cuda_moe_stream_preload_tensor_prompt(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes) {
    // The experimental batch producer/cache path is retired.
    return false;
}

extern "C" bool ggml_cuda_moe_stream_preload_tensor_async(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes) {
    // The experimental batch producer/cache path is retired.
    return false;
}

extern "C" bool ggml_cuda_moe_stream_preload_expert_async(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes,
    int expert_idx) {
    // The experimental batch producer/cache path is retired.
    return false;
}

extern "C" bool ggml_cuda_moe_stream_preload_expert_from_pack_async(
    int src0_type_int,
    const char *src0_name,
    int64_t n_as,
    size_t expert_bytes,
    int expert_idx) {
    // The experimental batch producer/cache path is retired.
    return false;
}











struct q4_vram_object_admission_resident {
    uint64_t used = 0;
    int layer = -1;
    int expert = -1;
};

struct q4_vram_object_admission_state {
    std::unordered_map<uint64_t, uint32_t> frequency;
    std::unordered_map<uint64_t, q4_vram_object_admission_resident> resident;
    uint64_t clock = 0;
    uint64_t last_epoch = UINT64_MAX;
    uint64_t requests = 0;
    uint64_t resident_hits = 0;
    uint64_t admissions = 0;
    uint64_t promotions = 0;
    uint64_t evictions = 0;
    uint64_t bypasses = 0;
    uint64_t aging_events = 0;
    uint64_t rollbacks = 0;
    uint64_t transient_object_evictions = 0;
    size_t capacity = 0;
};

static q4_vram_object_admission_state g_q4_vram_object_admission;































































static void q4_colibri_host_object_preload_allowlist_once(size_t expert_bytes) {
    return;
}



struct q4_colibri_pilot_candidate {
    int expert = -1;
    int64_t rows = 0;
};

struct q4_colibri_pilot_layer_history {
    int experts[64] = {};
    int count = 0;
};

static q4_colibri_pilot_layer_history g_q4_pilot_layer_history[64];
static q4_colibri_pilot_layer_history g_q4_pilot_miss_layer_history[64];









extern "C" int ggml_cuda_moe_stream_batch_preload_active_from_pack(
    int src0_type_int,
    const char *src0_name,
    int64_t n_as,
    size_t expert_bytes,
    const int64_t *matrix_row_counts) {
    // The experimental batch producer/cache path is retired.
    return 0;
}

extern "C" int ggml_cuda_moe_stream_q4_colibri_pilot_prefetch_from_gate(
    int src0_type_int,
    const char *src0_gate_name,
    int64_t n_as,
    size_t expert_bytes,
    const int64_t *matrix_row_counts) {
    // The experimental batch producer/cache path is retired.
    return 0;
}

extern "C" bool ggml_cuda_moe_stream_preload_synchronize(void) {
    if (!init_batch_once()) return false;
    std::lock_guard<std::mutex> lk(g_batch_mu);
    if (!g_batch.stream) return false;
    return cudaStreamSynchronize(g_batch.stream) == cudaSuccess;
}

extern "C" bool ggml_cuda_moe_stream_cache_contains(
    const char *src0_name,
    size_t expert_bytes,
    int expert_idx) {
    // The experimental batch producer/cache path is retired.
    return false;
}

extern "C" const void * ggml_cuda_moe_stream_cache_dev_ptr(
    const char *src0_name,
    size_t expert_bytes,
    int expert_idx) {
    // The experimental batch producer/cache path is retired.
    return nullptr;
}

extern "C" bool ggml_cuda_moe_stream_register_tensor(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes) {
    // The experimental batch producer/cache path is retired.
    return false;
}





















































static std::atomic<uint64_t> g_route_group_calls{0};
static std::atomic<uint64_t> g_route_group_submitted{0};
static std::atomic<uint64_t> g_route_group_cache_hits{0};
static std::atomic<uint64_t> g_route_group_missing_pack{0};
static std::atomic<uint64_t> g_route_group_repeat_skips{0};
static std::atomic<uint64_t> g_route_group_invalid_gate{0};
static std::atomic<uint64_t> g_route_group_inactive_expert_skips{0};
static std::atomic<bool> g_route_group_report_registered{false};
static std::mutex g_route_group_seen_mu;
static std::unordered_map<uintptr_t, uint32_t> g_route_group_seen;

static std::atomic<uint64_t> g_route_group_native_calls{0};
static std::atomic<uint64_t> g_route_group_native_selected{0};
static std::atomic<uint64_t> g_route_group_native_gate_hits{0};
static std::atomic<uint64_t> g_route_group_native_gate_misses{0};
static std::atomic<uint64_t> g_route_group_native_up_hits{0};
static std::atomic<uint64_t> g_route_group_native_up_misses{0};
static std::atomic<uint64_t> g_route_group_native_down_hits{0};
static std::atomic<uint64_t> g_route_group_native_down_misses{0};
static std::atomic<uint64_t> g_route_group_native_stage_a_jobs{0};
static std::atomic<uint64_t> g_route_group_native_stage_a_batches{0};
static std::atomic<uint64_t> g_route_group_native_stage_a_bytes{0};
static std::atomic<uint64_t> g_route_group_native_stage_a_copy_us{0};
static std::atomic<uint64_t> g_route_group_native_stage_b_jobs{0};
static std::atomic<uint64_t> g_route_group_native_stage_b_batches{0};
static std::atomic<uint64_t> g_route_group_native_stage_b_bytes{0};
static std::atomic<uint64_t> g_route_group_native_stage_b_copy_us{0};
static std::atomic<uint64_t> g_route_group_native_missing_pack{0};
static std::atomic<uint64_t> g_route_group_native_no_slot{0};
static std::atomic<uint64_t> g_route_group_native_copy_fail{0};
static std::atomic<uint64_t> g_route_group_native_seen_skips{0};
static std::atomic<uint64_t> g_route_group_native_hotset_hits{0};
static std::atomic<uint64_t> g_route_group_native_hotset_skips{0};
static std::atomic<uint64_t> g_route_group_native_hotset_profile_missing{0};
static std::atomic<uint64_t> g_route_group_native_priority_down_enqueued{0};
static std::atomic<uint64_t> g_route_group_native_priority_down_queue_fail{0};
static std::atomic<uint64_t> g_route_group_native_priority_down_enqueue_us{0};
static std::mutex g_route_group_native_seen_mu;
static std::unordered_map<uintptr_t, uint32_t> g_route_group_native_seen;







struct route_group_native_job {
    batch_vram_cache *cache = nullptr;
    int slot = -1;
    uintptr_t key = 0;
    void *dst = nullptr;
    const void *host_data = nullptr;
    const expert_pack_entry *pack_entry = nullptr;
    size_t expert_bytes = 0;
    int expert_idx = -1;
    char tensor[128] = {};
    char role = '?';
};



















extern "C" int ggml_cuda_moe_stream_route_group_preload_updown_from_gate(
        const char *src0_gate_name,
        int64_t n_as,
        const int64_t *matrix_row_counts) {
    // The experimental batch producer/cache path is retired.
    return 0;
}






struct gate_updown_cosubmit_job {
    batch_vram_cache *cache = nullptr;
    int slot = -1;
    void *dst = nullptr;
    const void *host_data = nullptr;
    const expert_pack_entry *pack_entry = nullptr;
    size_t expert_bytes = 0;
    int expert_idx = -1;
    char tensor[128] = {};
};

static std::mutex g_gate_updown_cosubmit_pending_mu;
static std::vector<gate_updown_cosubmit_job> g_gate_updown_cosubmit_pending;

static std::atomic<uint64_t> g_gate_updown_cosubmit_calls{0};
static std::atomic<uint64_t> g_gate_updown_cosubmit_jobs{0};
static std::atomic<uint64_t> g_gate_updown_cosubmit_cache_hits{0};
static std::atomic<uint64_t> g_gate_updown_cosubmit_missing_pack{0};
static std::atomic<uint64_t> g_gate_updown_cosubmit_iouring_ok{0};
static std::atomic<uint64_t> g_gate_updown_cosubmit_fallback_ok{0};
static std::atomic<uint64_t> g_gate_updown_cosubmit_no_slot_skips{0};
static std::atomic<uint64_t> g_gate_updown_cosubmit_repeat_skips{0};
static std::atomic<uint64_t> g_gate_updown_cosubmit_profile_skips{0};
static std::atomic<uint64_t> g_gate_updown_cosubmit_failures{0};
static std::atomic<bool> g_gate_updown_cosubmit_report_registered{false};
static std::mutex g_gate_updown_cosubmit_seen_mu;
static std::unordered_map<uintptr_t, uint32_t> g_gate_updown_cosubmit_seen;



static bool gate_updown_cosubmit_env_enabled() {
    const char *env = std::getenv("GGML_MOE_GATE_UPDOWN_COSUBMIT");
    return env && env[0] && env[0] != '0';
}







extern "C" int ggml_cuda_moe_stream_batch_preload_gate_updown(const char *gate_name, int expert_idx, size_t expert_bytes) {
    // The experimental batch producer/cache path is retired.
    return 0;
}

extern "C" int ggml_cuda_moe_stream_batch_flush_gate_updown_cosubmit(void) {
    if (!gate_updown_cosubmit_env_enabled()) return 0;
    std::vector<gate_updown_cosubmit_job> jobs;
    {
        std::lock_guard<std::mutex> lk(g_gate_updown_cosubmit_pending_mu);
        if (g_gate_updown_cosubmit_pending.empty()) return 0;
        jobs.swap(g_gate_updown_cosubmit_pending);
    }

    size_t pos = 0;
    int ready_jobs = 0;
    while (pos < jobs.size()) {
        const size_t expert_bytes = jobs[pos].expert_bytes;
        size_t end = pos + 1;
        while (end < jobs.size() && jobs[end].expert_bytes == expert_bytes) {
            ++end;
        }
        std::vector<gate_updown_cosubmit_job> group(jobs.begin() + (ptrdiff_t)pos, jobs.begin() + (ptrdiff_t)end);
        const bool copied_iouring = expert_pack_iouring_copy_jobs(group, expert_bytes, g_batch.prefetch_stream,
                g_batch.stage_ring_up_aux, "gate_updown_cosubmit");
        bool copied = copied_iouring;
        if (!copied) {
            copied = true;
            for (const gate_updown_cosubmit_job &job : group) {
                batch_copy_trace copy_trace;
                if (!batch_cache_copy_h2d(g_batch.stage_ring_up_aux, job.dst, job.host_data, expert_bytes,
                            g_batch.prefetch_stream, job.pack_entry, &copy_trace,
                            "gate_updown_cosubmit", job.tensor, job.expert_idx)) {
                    copied = false;
                    break;
                }
            }
        }

        if (!copied || cudaGetLastError() != cudaSuccess) {
            for (const gate_updown_cosubmit_job &job : group) {
                batch_cache_clear_slot(job.cache, job.slot);
            }
            ++g_gate_updown_cosubmit_failures;
            pos = end;
            continue;
        }

        for (const gate_updown_cosubmit_job &job : group) {
            batch_vram_cache *cache = job.cache;
            if (!cache || job.slot < 0 || job.slot >= cache->n_slots) {
                ++g_gate_updown_cosubmit_failures;
                continue;
            }
            const ggml_moe_common::role_cache_view role_view = batch_cache_role_view(cache);
            if (!ggml_moe_common::role_cache_ensure_ready_event(role_view, job.slot)) {
                batch_cache_clear_slot(cache, job.slot);
                ++g_gate_updown_cosubmit_failures;
                continue;
            }
            if (!ggml_moe_common::role_cache_record_ready_event(
                        role_view, job.slot, g_batch.prefetch_stream)) {
                batch_cache_clear_slot(cache, job.slot);
                ++g_gate_updown_cosubmit_failures;
                continue;
            }
            cache->slot_pending[job.slot] = true;
            ++g_gate_updown_cosubmit_jobs;
            ++cache->preloads;
            ++ready_jobs;
        }
        if (copied_iouring) {
            ++g_gate_updown_cosubmit_iouring_ok;
        } else {
            ++g_gate_updown_cosubmit_fallback_ok;
        }
        pos = end;
    }
    return ready_jobs;
}















struct fused_upgate_down_shadow_profile {
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> rows{0};
    std::atomic<uint64_t> decode_calls{0};
    std::atomic<uint64_t> prompt_calls{0};
    std::atomic<uint64_t> up_cache_misses{0};
    std::atomic<uint64_t> gate_cache_misses{0};
    std::atomic<uint64_t> down_cache_misses{0};
    std::atomic<uint64_t> down_overlap_plannable{0};
    std::atomic<uint64_t> down_missing_tensor{0};
    std::atomic<uint64_t> down_missing_cache{0};
    std::atomic<uint64_t> down_missing_pack{0};
};

static fused_upgate_down_shadow_profile g_fused_upgate_down_shadow;
static std::atomic<bool> g_fused_upgate_down_shadow_report_registered{false};







struct q4_triplet_touch_state {
    std::atomic<bool> registered{false};
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> prompt_skips{0};
    std::atomic<uint64_t> missing_down_name{0};
    std::atomic<uint64_t> missing_cache{0};
    std::atomic<uint64_t> full_triplets{0};
    std::atomic<uint64_t> partial_triplets{0};
    std::atomic<uint64_t> touched_roles{0};
};

static q4_triplet_touch_state g_q4_triplet_touch;



























template<typename Launch>
static bool batch_cache_launch_with_role_leases(
        batch_vram_cache *cache,
        const char *key_name,
        ggml_moe_common::role kind,
        const int *experts,
        const int32_t *slots,
        int count,
        cudaStream_t run_stream,
        Launch &&launch) {
    if (!cache || !key_name || !experts || !slots || count <= 0 ||
            count > MOE_STREAM_MAX_ACTIVE || !run_stream) {
        return false;
    }

    uintptr_t keys[MOE_STREAM_MAX_ACTIVE] = {};
    ggml_moe_expert_descriptor descriptors[MOE_STREAM_MAX_ACTIVE] = {};
    int acquired = 0;
    for (; acquired < count; ++acquired) {
        keys[acquired] = batch_key_hash(key_name, experts[acquired]);
        if (!ggml_moe_common::role_cache_acquire_descriptor_for_stream(
                    batch_cache_role_view(cache), slots[acquired], keys[acquired],
                    (uint64_t)keys[acquired], kind, 0, run_stream,
                    descriptors[acquired])) {
            static std::atomic<int> first_role_lease_failure{0};
            if (first_role_lease_failure.fetch_add(1, std::memory_order_relaxed) == 0) {
                const int slot = slots[acquired];
                std::fprintf(stderr,
                    "[moe_stream_batch] role kernel lease acquire failed: tensor=%s "
                    "role=%d expert=%d slot=%d expected_key=%lu slot_key=%lu "
                    "generation=%lu state=%d pending=%d filling=%d payload_bytes=%zu\n",
                    key_name, (int)kind, experts[acquired], slot,
                    (unsigned long)keys[acquired],
                    (unsigned long)(slot >= 0 && slot < cache->n_slots ?
                        cache->slot_key[slot] : 0),
                    (unsigned long)(slot >= 0 && slot < cache->n_slots ?
                        cache->slot_generation[slot] : 0),
                    slot >= 0 && slot < cache->n_slots ?
                        (int)cache->slot_state[slot] : -1,
                    slot >= 0 && slot < cache->n_slots && cache->slot_pending[slot] ? 1 : 0,
                    slot >= 0 && slot < cache->n_slots && cache->slot_filling[slot] ? 1 : 0,
                    slot >= 0 && slot < cache->n_slots ?
                        cache->slot_payload_bytes[slot] : 0);
            }
            break;
        }
        ++cache->descriptor_acquires;
    }
    if (acquired != count) {
        for (int i = 0; i < acquired; ++i) {
            (void)q4_graph_release_descriptor(
                cache, keys[i], descriptors[i], nullptr, false);
        }
        return false;
    }

    const bool launched = launch();
    bool released = true;
    for (int i = 0; i < acquired; ++i) {
        released = q4_graph_release_descriptor(
            cache, keys[i], descriptors[i], run_stream, true) && released;
    }
    return launched && released;
}

extern "C" bool ggml_cuda_moe_stream_batch(
    int  src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    size_t nb02,
    const float *src1_f32,
    int64_t src1_ne1,
    size_t src1_nb1, size_t src1_nb2,
    const void *src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_q8_0_ne1,
    float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const int64_t *matrix_row_counts,
    const ggml_moe_row_mapping *matrix_rows,
    int64_t rows_stride) {
    // The opt-in runtime streaming compute path is retired.
    return false;
}


} // namespace ggml_moe_common_adapter

#endif // GGML_CUDA_MOE_STREAM_BATCH
