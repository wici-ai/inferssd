#include "moe-common-adapter-symbols.h"
#include "moe-expert-descriptor.h"

#ifndef GGML_CUDA_MOE_STREAM
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
extern "C" {
typedef struct { int32_t i1; int32_t i2; } ggml_moe_row_mapping;
void ggml_cuda_moe_stream_link_anchor(void) {}
bool ggml_cuda_moe_stream_available(void) { return false; }
int ggml_cuda_host_register(void *, size_t) { return 0; }
bool ggml_cuda_moe_stream_one(int, const char *, int64_t, const void *, int64_t, int64_t, size_t, const float *, size_t, size_t, int64_t, int64_t, const void *, size_t, float *, size_t, size_t, const ggml_moe_row_mapping *) { return false; }
void ggml_cuda_moe_stream_q80_probe(int, const char *, int64_t, const void *, int64_t, int64_t, size_t, const void *, size_t, int64_t, int64_t, const float *, size_t, size_t, const ggml_moe_row_mapping *) {}
void ggml_cuda_moe_stream_q80_write(int, const char *, int64_t, const void *, int64_t, int64_t, size_t, const void *, size_t, int64_t, int64_t, float *, size_t, size_t, const ggml_moe_row_mapping *) {}
bool ggml_cuda_moe_stream_q80_skip(int, const char *, int64_t, const void *, int64_t, int64_t, size_t, const void *, size_t, int64_t, int64_t, float *, size_t, size_t, const ggml_moe_row_mapping *) { return false; }
void ggml_cuda_moe_stream_q80_hot_batch_probe(int, const char *, int64_t, int64_t, int64_t, size_t, const void *, size_t, int64_t, const float *, size_t, size_t, const int64_t *, const ggml_moe_row_mapping *, int64_t, const float *, size_t, size_t) {}
bool ggml_cuda_moe_stream_one_cache_contains(const char *, const void *, size_t, int64_t) { return false; }
bool ggml_cuda_moe_stream_one_cache_ready(void) { return false; }
const void * ggml_cuda_moe_stream_one_cache_dev_ptr_or_insert(
    const char *, const void *, size_t, size_t, int64_t, cudaStream_t, int *) { return nullptr; }
bool ggml_cuda_moe_stream_one_cache_dev_ptrs_or_insert(
    const char *, const void *, size_t, size_t, const int *, const int32_t *, int, cudaStream_t,
    const void **, int *, ggml_moe_expert_descriptor *) { return false; }
void ggml_cuda_moe_stream_one_gate_up_prefetch(
    const char *, size_t, const int *, const int32_t *, int) {}
void ggml_cuda_moe_stream_one_cache_release_descriptors(
    const ggml_moe_expert_descriptor *, int, cudaStream_t) {}
void ggml_cuda_moe_stream_one_cache_release_descriptors_synchronized(
    const ggml_moe_expert_descriptor *, int) {}
const void * ggml_cuda_moe_stream_cache_dev_ptr(const char *, size_t, int) { return nullptr; }
bool ggml_cuda_moe_stream_mmvq_dev(int, const void *, int64_t, int64_t, size_t, const float *, void *, float *, cudaStream_t) { return false; }
bool ggml_cuda_moe_stream_mmvq_rows_dev(int, const void *, int64_t, int64_t, size_t, const float *, void *, const int32_t *, int64_t, float *, cudaStream_t) { return false; }
bool ggml_cuda_moe_stream_mmvq_batch_dev(int, const void *, int64_t, int64_t, const float *, void *, float *, const int32_t *, int64_t, int64_t, cudaStream_t) { return false; }
void ggml_cuda_moe_stream_sync(void) {}
}
#else
// ik_llama_fork: streaming MoE compute — invoked from ggml.c's CPU mul_mat_id
// path when GGML_MOE_STREAM=1.  Copies one expert's quantized weights H2D into
// a small VRAM staging buffer, runs the existing MMVQ kernel on GPU, copies
// the float32 result D2H, scatters into dst per the row-mapping.
//
// Uses a pool of N (default 8) CUDA streams with per-slot VRAM staging so
// multiple CPU threads can submit GPU work concurrently without a global lock.
//
// Author: Zili Meng <zilim@ieee.org>

#include "common.cuh"
#include "mmvq.cuh"
#include "moe-common-lossless-format.h"
#include "moe-common-lossless-nvcomp.h"
#include "moe-common-lossless-rans.h"
#include "moe-common-object-cache.h"
#include "moe-common-object-engine.h"
#include "moe-common-object-pipeline.h"
#include "moe-common-object-pipeline-runtime.h"
#include "moe-common-object-transport.h"
#include "moe-common-role-cache.h"
#include "moe-common-transfer-planner.h"
#include "quantize.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <cuda_runtime.h>

namespace ggml_moe_common_adapter {

extern "C" {

typedef struct {
    int32_t i1;
    int32_t i2;
} ggml_moe_row_mapping;

void ggml_cuda_moe_stream_link_anchor(void) {}

int ggml_cuda_moe_stream_batch_preload_gate_updown(const char *gate_name, int expert_idx, size_t expert_bytes);
int ggml_cuda_moe_stream_batch_flush_gate_updown_cosubmit(void);
const void * ggml_cuda_moe_stream_cache_dev_ptr(const char *src0_name, size_t expert_bytes, int expert_idx);

// src1_f32: F32 activations (will be Q8_1 quantized on GPU)
//           Layout: cne1 rows × ne00 cols, with stride row_size_f32 bytes per row.
//           If src1_f32 is NULL, falls back to caller-provided src1_q8_1.
bool ggml_cuda_moe_stream_one(
    int  src0_type_int,
    const char *src0_name,
    int64_t expert_index,
    const void *src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const float *src1_f32,
    size_t src1_nb1, size_t src1_nb2,
    int64_t src1_ne1,
    int64_t cne1,
    const void *src1_q8_1,             // optional pre-quantized (unused if src1_f32 set)
    size_t src1_padded_num_cols,
    float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const ggml_moe_row_mapping *rows);

void ggml_cuda_moe_stream_q80_probe(
    int src0_type_int,
    const char *src0_name,
    int64_t expert_index,
    const void *src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const void *src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_ne1,
    int64_t cne1,
    const float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const ggml_moe_row_mapping *rows);

void ggml_cuda_moe_stream_q80_write(
    int src0_type_int,
    const char *src0_name,
    int64_t expert_index,
    const void *src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const void *src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_ne1,
    int64_t cne1,
    float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const ggml_moe_row_mapping *rows);

bool ggml_cuda_moe_stream_q80_skip(
    int src0_type_int,
    const char *src0_name,
    int64_t expert_index,
    const void *src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const void *src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_ne1,
    int64_t cne1,
    float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const ggml_moe_row_mapping *rows);

void ggml_cuda_moe_stream_q80_hot_batch_probe(
    int src0_type_int,
    const char *src0_name,
    int64_t n_as,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const void *src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_ne1,
    const float *src1_f32,
    size_t src1_nb1,
    size_t src1_nb2,
    const int64_t *matrix_row_counts,
    const ggml_moe_row_mapping *matrix_rows,
    int64_t rows_per_expert,
    const float *dst,
    size_t dst_nb1,
    size_t dst_nb2);

bool ggml_cuda_moe_stream_available(void);
int  ggml_cuda_host_register(void *p, size_t n);
bool ggml_cuda_moe_stream_mmvf_dev(int, const void *, int64_t, int64_t, size_t, const float *, float *, int64_t, cudaStream_t);
bool ggml_cuda_moe_expert_pack_read_to_host(const char *, int, size_t, void *, size_t, size_t *, int *);
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
bool ggml_cuda_moe_stream_mmvq_rows_dev(
    int src0_type_int,
    const void *d_src0,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const float *d_src1_f32,
    void *d_src1_q8,
    const int32_t *d_x_ids,
    int64_t cne1,
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
const void * ggml_cuda_moe_stream_one_cache_dev_ptr_or_insert(
    const char *src0_name,
    const void *src0_base_data,
    size_t nb02,
    size_t src0_bytes,
    int64_t expert_index,
    cudaStream_t stream,
    int *cache_state);
// Sync all outstanding GPU work submitted by ggml_cuda_moe_stream_one and
// perform deferred dst scatter.  Call once at end of a mul_mat_id op (the
// sequential path's thread 0 invokes this after the per-expert loop).
void ggml_cuda_moe_stream_sync(void);

} // extern "C"

// === VRAM Expert Cache (PowerInfer pattern) =================================
//
// A fixed-size VRAM pool that caches recently-used expert weight tensors.
// On cache hit: kernel runs directly from VRAM (zero H2D latency).
// On cache miss: H2D into a cache slot, then kernel.  FIFO eviction.
//
// Budget: GGML_MOE_STREAM_ONE_CACHE_MIB / GB, or the shared GGML_MOE_VRAM_CACHE_GB.
// Each expert is ~4.6 MiB → ~3500 slots in 16 GiB.

// Hash table for O(1) lookup.  Power-of-2 sized, linear probing.
// Mapping: key (host pointer) → slot index in VRAM pool.
#define VRAM_CACHE_HT_SIZE 16384 *4  // power of 2, ~2× max slots
#define VRAM_CACHE_HT_MASK (VRAM_CACHE_HT_SIZE - 1)

struct vram_ht_entry {
    uintptr_t key;     // 0 = empty
    int       slot;
};

struct vram_cache {
    void *   pool       = nullptr;
    size_t   pool_sz    = 0;
    size_t   slot_sz    = 0;
    int      n_slots    = 0;
    int      next_evict = 0;
    int      used_slots = 0;
    bool     lru_frozen = false;
    uintptr_t slot_key[VRAM_CACHE_HT_SIZE] = {};
    uint64_t  slot_used[VRAM_CACHE_HT_SIZE] = {};
    uint64_t  slot_tensor_tag[VRAM_CACHE_HT_SIZE] = {};
    uint64_t  slot_generation[VRAM_CACHE_HT_SIZE] = {};
    uint8_t   slot_layout[VRAM_CACHE_HT_SIZE] = {};
    cudaEvent_t slot_ready[VRAM_CACHE_HT_SIZE] = {};
    bool     slot_pending[VRAM_CACHE_HT_SIZE] = {};
    bool     slot_filling[VRAM_CACHE_HT_SIZE] = {};
    cudaEvent_t slot_lease_ready[VRAM_CACHE_HT_SIZE] = {};
    bool     slot_lease_pending[VRAM_CACHE_HT_SIZE] = {};
    uint32_t slot_lease_count[VRAM_CACHE_HT_SIZE] = {};
    vram_ht_entry ht[VRAM_CACHE_HT_SIZE] = {};

    ggml_moe_common::object_host_cache ram_cache;

    std::atomic<uint64_t> clock{1};
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> ram_hits{0};
    std::atomic<uint64_t> ram_misses{0};
    std::atomic<uint64_t> ram_inserts{0};
    std::atomic<uint64_t> ram_evictions{0};
    std::atomic<uint64_t> freeze_rejections{0};
    std::atomic<uint64_t> descriptor_acquires{0};
    std::atomic<uint64_t> descriptor_releases{0};
    std::atomic<uint64_t> descriptor_cancels{0};
    std::atomic<uint64_t> descriptor_stale_releases{0};
    std::atomic<uint64_t> score_victim_selections{0};
    std::atomic<uint64_t> score_candidate_lte_victim{0};
    std::atomic<uint64_t> ram_score_admissions{0};
    std::atomic<uint64_t> ram_score_min_demand_rejects{0};
    std::atomic<uint64_t> ram_score_victim_rejects{0};
    std::atomic<uint64_t> ram_score_layer_rejects{0};
    std::atomic<uint64_t> ram_full_object_publishes{0};
    std::atomic<uint64_t> ram_non_object_stores{0};
    std::atomic<uint64_t> ram_host_store_attempts{0};
    std::atomic<uint64_t> ram_host_store_valid{0};
    std::atomic<uint64_t> ram_host_store_admitted{0};
    std::atomic<uint64_t> ram_host_store_slot_selected{0};
    std::atomic<uint64_t> ram_host_store_staging_requested{0};
    std::atomic<uint64_t> ram_host_store_common_eligible{0};
    std::atomic<uint64_t> ram_host_lease_acquires{0};
    std::atomic<uint64_t> ram_host_lease_releases{0};
    std::atomic<uint64_t> ram_host_lease_failures{0};
    std::mutex mu;

    // Appended to preserve every existing DeepSeek hot-field offset while
    // satisfying the shared generation-aware role-cache schema.
    GGML_MOE_FIXED_ROLE_CACHE_READINESS_FIELDS(VRAM_CACHE_HT_SIZE)
};

static vram_cache g_vcache;
static bool       g_vcache_inited = false;
static std::unordered_map<uintptr_t, uint32_t> g_vram_cache_demand_scores;


































struct ds4_ram_key_stat {
    int layer = -1;
    int expert = -1;
    uint32_t demands = 0;
    uint32_t ram_hits = 0;
};

static std::unordered_map<uintptr_t, ds4_ram_key_stat> g_ds4_ram_key_stats;

static uint64_t g_ds4_ram_layer_demands[43] = {};
static uint64_t g_ds4_ram_layer_hits[43] = {};
static uint64_t g_ds4_ram_layer_admissions[43] = {};
static uint64_t g_ds4_ram_layer_rejects[43] = {};

struct ds4_ram_prefetch_state {
    bool started = false;
    bool stop = false;
    bool failed = false;
    std::deque<uint32_t> queue;
    std::unordered_set<uintptr_t> queued_keys;
    std::unordered_set<uintptr_t> ready_keys;
    std::mutex mu;
    std::condition_variable cv;
    std::thread worker;
    std::atomic<uint64_t> submitted{0};
    std::atomic<uint64_t> completed{0};
    std::atomic<uint64_t> used{0};
    std::atomic<uint64_t> wasted{0};
    std::atomic<uint64_t> duplicate_skips{0};
    std::atomic<uint64_t> resident_skips{0};
    std::atomic<uint64_t> read_failures{0};
    std::atomic<uint64_t> bytes{0};
};

static ds4_ram_prefetch_state g_ds4_ram_prefetch;



















































































struct ds4_ram_slot_selection_context {
    int avoid_slot = -1;
    cudaStream_t stream = nullptr;
    bool require_ready_now = false;
    bool wait_each_candidate = false;
    ggml_moe_common::object_host_layer_quota_policy quota;
};

































struct ds4_vram_reservation {
    void * dst = nullptr;
    uintptr_t key = 0;
    uint64_t slot_generation = 0;
    int slot = -1;
};













// === Pool of GPU stream slots// === Pool of GPU stream slots ==============================================
//
// Each slot has its own stream + VRAM staging buffers.  A CPU thread atomically
// claims a slot via round-robin counter, does its H2D + kernel + D2H + sync on
// that slot, then releases.  Slots are independent — no shared resources.

#define MOE_STREAM_NSLOTS 8

struct deferred_scatter {
    int     slot;
    float * dst_base;
    size_t  dst_nb1;
    size_t  dst_nb2;
    int64_t ne01;
    int64_t cne1;
    int32_t i1[8];
    int32_t i2[8];
};

struct slot_ctx {
    cudaStream_t stream  = nullptr;
    void *  d_src0       = nullptr;
    size_t  d_src0_sz    = 0;
    void *  d_src1       = nullptr;       // Q8_1 quantized src1
    size_t  d_src1_sz    = 0;
    void *  d_src1_f32   = nullptr;       // F32 src1 staging (persistent)
    size_t  d_src1_f32_sz = 0;
    void *  d_dst        = nullptr;
    size_t  d_dst_sz     = 0;
    void *  d_ids        = nullptr;
    size_t  d_ids_sz     = 0;
    void *  h_scratch    = nullptr;
    size_t  h_scratch_sz = 0;
    void *  h_bounce     = nullptr;
    size_t  h_bounce_sz  = 0;
    void *  h_src0_pack  = nullptr;
    size_t  h_src0_pack_sz = 0;
    cudaEvent_t profile_weight_start = nullptr;
    cudaEvent_t profile_weight_done = nullptr;
    cudaEvent_t profile_input_done = nullptr;
    cudaEvent_t profile_kernel_done = nullptr;
    cudaEvent_t profile_d2h_done = nullptr;
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

// One queue of deferred scatters per CPU thread (thread_local).
static thread_local std::vector<deferred_scatter> tls_pending;

static slot_ctx          g_slots[MOE_STREAM_NSLOTS];
static std::atomic<int>  g_slot_rr{0};
static std::atomic<bool> g_avail{false};
static std::atomic<bool> g_inited{false};
static bool              g_defer_sync = false;
static std::mutex        g_init_mu;
// Slot resize is rare (only when needed > current); guard with one mutex
// rather than per-slot.  Compute path doesn't need it.
static std::mutex        g_resize_mu;


struct one_expert_pack_entry {
    char tensor[128] = {};
    int32_t expert_idx = -1;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
};

struct one_expert_pack_state {
    int fd = -1;
    int fd_direct = -1;
    bool direct_enabled = false;
    bool payload_sidecar_enabled = false;
    std::vector<one_expert_pack_entry> entries;
    bool inited = false;
    bool enabled = false;
    std::mutex mu;
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> reads{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> failures{0};
    std::atomic<uint64_t> direct_reads{0};
    std::atomic<uint64_t> direct_failures{0};
    std::atomic<uint64_t> direct_fallbacks{0};
};

static one_expert_pack_state g_one_pack;

struct one_pack_cqe_state {
    std::mutex mu;
    ggml_moe_common::object_stage_ring io_ring;
    ggml_moe_common::object_stage_ring import_ring;
    bool ring_inited = false;
    cudaStream_t stream = nullptr;
    bool init_attempted = false;
    bool ready = false;
    uint64_t batches = 0;
    uint64_t jobs = 0;
    uint64_t cqes = 0;
    uint64_t fallbacks = 0;
    uint64_t bytes = 0;
    uint64_t submit_us = 0;
    uint64_t wait_us = 0;
    uint64_t h2d_enqueue_us = 0;
};

static one_pack_cqe_state g_one_pack_cqe;

struct one_gate_up_prefetch_job {
    const one_expert_pack_entry * entry = nullptr;
    ds4_vram_reservation reservation;
    bool read_ok = false;
    bool published = false;
};

struct one_gate_up_prefetch_state {
    std::mutex mu;
    ggml_moe_common::object_stage_ring io_ring;
    ggml_moe_common::object_stage_ring import_ring;
    bool ring_inited = false;
    bool sqpoll = false;
    bool fixed_file = false;
    cudaStream_t stream = nullptr;
    bool init_attempted = false;
    int pending = 0;
    int submitted = 0;
    char tensor[128] = {};
    one_gate_up_prefetch_job jobs[8];
    uint64_t calls = 0;
    uint64_t scheduled = 0;
    uint64_t placement_skips = 0;
    uint64_t resident_skips = 0;
    uint64_t cqes = 0;
    uint64_t demand_fallbacks = 0;
    uint64_t bytes = 0;
    uint64_t submit_us = 0;
    uint64_t residual_wait_us = 0;
    uint64_t h2d_enqueue_us = 0;
};

static one_gate_up_prefetch_state g_one_gate_up_prefetch;

struct ds4_triplet_pack_entry {
    int32_t layer = -1;
    int32_t expert = -1;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
    uint64_t generation = 0;
};



struct ds4_triplet_pipe_batch {
    const ds4_triplet_pack_entry * entries[8] = {};
    void * host[8] = {};
    ds4_vram_reservation reservations[8] = {};
    ggml_moe_common::dependency_transfer_plan transfer_plans[8] = {};
    uint8_t caller_role_masks[8] = {};
    ds4_vram_reservation role_reservations[8][3] = {};
    void * role_ptrs[8][3] = {};
    bool host_cache_hits[8] = {};
    size_t host_cache_slots[8] = {
        SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX,
        SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX,
    };
    ggml_moe_common::object_request common_requests[8] = {};
    ggml_moe_common::object_transaction_ticket transaction_tickets[8] = {};
    bool common_transactions_owned = false;
    bool dependency_h2d = false;
    int count = 0;
    bool done = false;
    bool success = false;
    ggml_moe_common::object_pipeline_transaction_handle pipeline_handle;
    ggml_moe_common::object_stage_ring pipeline_stage_ring;
    ggml_moe_common::object_stage_ring * stage_ring = nullptr;
    size_t stage_slot_indices[8] = {
        SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX,
        SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX,
    };
    bool host_finalized = false;
    bool pipeline_submitted = false;
    bool pipeline_critical_returned = false;
    bool lossless_decode_all = false;
    bool lossless_objects[8] = {};
    bool lossless_planar[8] = {};
    bool lossless_scale3[8] = {};
    bool lossless_planar_scale3[8] = {};
    bool lossless_direct_quant[8] = {};
    bool lossless_nvcomp[8] = {};
    uint8_t lossless_nvcomp_role_masks[8] = {};
    bool lossless_crc_required[8] = {};
    uint8_t lossless_crc_role_masks[8] = {};
    size_t lossless_object_indices[8] = {
        SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX,
        SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX,
    };
    size_t lossless_crc_slots[8] = {
        SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX,
        SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX,
    };
    size_t lossless_crc_role_slots[8][3] = {};
    std::mutex mu;
    std::condition_variable cv;
};

static constexpr size_t DS4_TRIPLET_PIPE_QUEUE_CAPACITY = 64;

struct ds4_triplet_pack_state {
    int fd = -1;
    int fd_direct = -1;
    bool inited = false;
    bool enabled = false;
    uint64_t object_bytes = 0;
    uint64_t source_pack_bytes = 0;
    uint64_t role_bytes[3] = {};
    uint64_t manifest_generation = 0;
    std::vector<ds4_triplet_pack_entry> entries;
    std::mutex mu;
    std::mutex io_mu;
    std::mutex pipe_mu;
    std::condition_variable pipe_cv;
    std::deque<ds4_triplet_pipe_batch *> pipe_queue;
    std::thread pipe_worker;
    cudaStream_t pipe_stream = nullptr;
    cudaStream_t pipe_deferred_stream = nullptr;
    cudaEvent_t pipe_critical_done = nullptr;
    cudaEvent_t lossless_decode_done = nullptr;
    bool pipe_requested = false;
    bool pipe_started = false;
    bool pipe_stop = false;
    bool pipe_failed = false;
    bool pipe_common_worker = false;
    ggml_moe_common::object_stage_ring io_ring;
    bool ring_inited = false;
    bool ring_init_on_worker = false;
    bool ring_init_fallback = false;
    size_t ring_depth = 0;
    unsigned ring_requested_flags = 0;
    unsigned ring_active_flags = 0;
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> reads{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> failures{0};
    ggml_moe_common::object_io_runtime_counters io_stats;
    std::atomic<uint64_t> pipe_batches{0};
    std::atomic<uint64_t> pipe_cqes{0};
    std::atomic<uint64_t> pipe_submit_us{0};
    std::atomic<uint64_t> pipe_wait_us{0};
    std::atomic<uint64_t> pipe_process_us{0};
    std::atomic<uint64_t> pipe_caller_wait_us{0};
    std::atomic<uint64_t> pipe_h2d_bytes{0};
    std::atomic<uint64_t> pipe_h2d_enqueue_us{0};
    std::atomic<uint64_t> pipe_h2d_device_us{0};
    std::atomic<uint64_t> pipe_refill_batches{0};
    std::atomic<uint64_t> pipe_refill_submit_rounds{0};
    std::atomic<uint64_t> pipe_refill_submitted{0};
    std::atomic<uint64_t> pipe_refill_completions{0};
    std::atomic<uint64_t> pipe_refill_fallbacks{0};
    std::atomic<uint64_t> dependency_plans{0};
    std::atomic<uint64_t> dependency_full_misses{0};
    std::atomic<uint64_t> dependency_partial_misses{0};
    std::atomic<uint64_t> dependency_critical_copies{0};
    std::atomic<uint64_t> dependency_critical_bytes{0};
    std::atomic<uint64_t> dependency_deferred_copies{0};
    std::atomic<uint64_t> dependency_deferred_bytes{0};
    std::atomic<uint64_t> dependency_host_hits{0};
    std::atomic<uint64_t> dependency_host_misses{0};
    std::atomic<uint64_t> dependency_host_admissions{0};
    std::atomic<uint64_t> dependency_host_lease_failures{0};
    std::atomic<uint64_t> dependency_transaction_owners{0};
    std::atomic<uint64_t> dependency_transaction_joins{0};
    std::atomic<uint64_t> dependency_transaction_join_wait_us{0};
    std::atomic<uint64_t> dependency_transaction_failures{0};
    ggml_moe_common::object_pipeline_transaction_pool dependency_pipeline_pool;
    ds4_triplet_pipe_batch dependency_pipeline_batches[2];
    std::atomic<uint64_t> dependency_pipeline_acquires{0};
    std::atomic<uint64_t> dependency_pipeline_backpressure{0};
    std::atomic<uint64_t> dependency_pipeline_releases{0};
    std::atomic<uint64_t> dependency_pipeline_critical_publishes{0};
    std::atomic<uint64_t> dependency_pipeline_early_critical_publishes{0};
    std::atomic<uint64_t> dependency_pipeline_critical_returns{0};
    std::atomic<uint64_t> dependency_pipeline_wait_all_fallbacks{0};
    std::atomic<uint64_t> dependency_pipeline_critical_wait_us{0};
    std::atomic<uint64_t> dependency_worker_host_finalizes{0};
    std::atomic<uint64_t> dependency_stage_records{0};
    std::atomic<uint64_t> dependency_fallbacks{0};
    cudaEvent_t pipe_h2d_start_events[8] = {};
    cudaEvent_t pipe_h2d_stop_events[8] = {};
    bool pipe_h2d_events_initialized = false;
    std::atomic<uint64_t> host_handoff_lookups{0};
    std::atomic<uint64_t> host_handoff_hits{0};
    std::atomic<uint64_t> role_pair_reads{0};
    std::atomic<uint64_t> role_calls[3] = {};
    std::atomic<uint64_t> role_lookups[3] = {};
    std::atomic<uint64_t> role_hits[3] = {};
    std::atomic<uint64_t> role_misses[3] = {};
    std::atomic<uint64_t> role_ram_promotes[3] = {};
    std::atomic<uint64_t> role_physical_reads[3] = {};
    std::atomic<uint64_t> mixed_role_calls{0};
    std::atomic<uint64_t> common_transaction_fallbacks{0};
    ggml_moe_common::lossless_sidecar lossless_sidecar;
    ggml_moe_common::lossless_rans_cuda_context * lossless_critical_decoder = nullptr;
    ggml_moe_common::lossless_rans_cuda_context * lossless_deferred_decoder = nullptr;
    ggml_moe_common::lossless_nvcomp_context * lossless_nvcomp_decoder = nullptr;
    uint8_t * lossless_device_input = nullptr;
    uint32_t * lossless_critical_status_host = nullptr;
    uint32_t * lossless_deferred_status_host = nullptr;
    uint32_t * lossless_crc32_host = nullptr;
    size_t lossless_device_input_stride = 0;
    size_t lossless_crc32_capacity = 0;
    std::vector<uint8_t> lossless_crc_verified;
    bool lossless_requested = false;
    bool lossless_ready = false;
    bool lossless_failed = false;
    std::atomic<uint64_t> lossless_reads{0};
    std::atomic<uint64_t> lossless_physical_bytes{0};
    std::atomic<uint64_t> lossless_payload_bytes{0};
    std::atomic<uint64_t> lossless_raw_bytes_avoided{0};
    std::atomic<uint64_t> lossless_h2d_bytes{0};
    std::atomic<uint64_t> lossless_decode_batches{0};
    std::atomic<uint64_t> lossless_decode_objects{0};
    std::atomic<uint64_t> lossless_fallbacks{0};
    std::atomic<uint64_t> lossless_crc_objects{0};
    std::atomic<uint64_t> lossless_crc_reuses{0};
    std::atomic<uint64_t> lossless_crc_failures{0};
    std::atomic<uint64_t> lossless_decode_failures{0};
    std::atomic<uint64_t> lossless_candidate_batches[9] = {};
    std::atomic<uint64_t> lossless_policy_raw_objects{0};
};

static ds4_triplet_pack_state g_ds4_triplet_pack;
















struct ds4_pipeline_caller_guard {
    ggml_moe_common::object_pipeline_transaction_handle handle;
    bool active = false;
    bool worker_owned = false;

    ~ds4_pipeline_caller_guard() {
        if (!active) return;
        if (!worker_owned) {
            (void) g_ds4_triplet_pack.dependency_pipeline_pool.complete_worker(handle, false);
        }
        if (g_ds4_triplet_pack.dependency_pipeline_pool.release_caller(handle)) {
            g_ds4_triplet_pack.dependency_pipeline_releases.fetch_add(
                1, std::memory_order_relaxed);
        }
    }
};































struct ds4_triplet_trace_state {
    FILE * fp = nullptr;
    bool initialized = false;
    uint64_t layer_cycle[43] = {};
    std::atomic<uint64_t> next_call{1};
    std::atomic<uint64_t> next_event{1};
    std::mutex mu;
};

static ds4_triplet_trace_state g_ds4_triplet_trace;





struct ds4_triplet_host_handoff_entry {
    const void * object = nullptr;
    uint64_t generation = 0;
};

static ds4_triplet_host_handoff_entry g_ds4_triplet_host_handoffs[8];
static std::mutex g_ds4_triplet_host_handoff_mu;
static int g_ds4_triplet_host_handoff_next = 0;











static bool one_pack_read_exact_fd(int fd, void * dst, size_t sz, uint64_t off) {
    return ggml_moe_common::object_pread_all(fd, dst, sz, off);
}

template<typename T>
static bool ds4_triplet_read_value(int fd, uint64_t offset, T & value) {
    return one_pack_read_exact_fd(fd, &value, sizeof(value), offset);
}

















extern "C" const void * ggml_cuda_moe_stream_ds4_triplet_host_role_ptr(
        const char * src0_name,
        int64_t expert_index,
        size_t role_nbytes) {
    // The experimental object/cache or probe path is retired.
    return nullptr;
}





























































static std::atomic<uint64_t> g_one_pack_direct_read_time_us{0};
static std::atomic<uint64_t> g_one_pack_direct_read_max_us{0};
static std::atomic<uint64_t> g_one_pack_buffered_read_time_us{0};
static std::atomic<uint64_t> g_one_pack_buffered_read_max_us{0};













struct one_direct_manifest_state {
    int fd = -1;
    int fd_direct = -1;
    bool direct_enabled = false;
    bool inited = false;
    bool enabled = false;
    std::vector<one_expert_pack_entry> entries;
    std::vector<one_expert_pack_entry> prefill_entries;
    std::mutex mu;
    uint64_t manifest_bytes = 0;
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> reads{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> failures{0};
    std::atomic<uint64_t> direct_reads{0};
    std::atomic<uint64_t> direct_failures{0};
    std::atomic<uint64_t> direct_fallbacks{0};
};

static one_direct_manifest_state g_one_direct_manifest;


struct one_direct_repr_entry {
    char tensor[128] = {};
    int32_t expert_idx = -1;
    uint64_t row0 = 0;
    uint64_t row_count = 0;
    uint64_t model_offset = 0;
    uint64_t compressed_offset = 0;
    uint64_t compressed_nbytes = 0;
    uint64_t original_nbytes = 0;
    char repr_type[32] = {};
    char flags[64] = {};
};

struct one_direct_repr_state {
    bool inited = false;
    bool enabled = false;
    std::vector<one_direct_repr_entry> entries;
    std::vector<size_t> pool_offsets;
    std::vector<uint8_t> pool_ready;
    std::mutex mu;
    std::mutex pool_mu;
    std::mutex report_mu;
    FILE * report_fp = nullptr;
    int fd = -1;
    int fd_direct = -1;
    bool direct_enabled = false;
    bool payload_sidecar_enabled = false;
    void * pool = nullptr;
    size_t pool_sz = 0;
    bool pool_inited = false;
    bool pool_enabled = false;
    uint64_t compressed_bytes = 0;
    uint64_t original_bytes = 0;
    uint64_t exact_entries = 0;
    uint64_t partial_entries = 0;
    uint64_t compressed_entries = 0;
    uint64_t pool_attempted = 0;
    uint64_t pool_inserted = 0;
    uint64_t pool_read_failures = 0;
    uint64_t pool_copy_failures = 0;
    uint64_t pool_bytes = 0;
    double pool_read_ms = 0.0;
    double pool_h2d_ms = 0.0;
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
};

static one_direct_repr_state g_one_direct_repr;




































struct one_direct_repr_partial_hit {
    const one_direct_repr_entry * entry = nullptr;
    const char * d_src0 = nullptr;
    bool q2tern = false;
};



struct one_direct_repr_probe_match {
    int entries = 0;
    bool has_exact = false;
    bool has_partial = false;
    bool has_compressed = false;
    uint64_t resident_compressed_bytes = 0;
    uint64_t resident_original_bytes = 0;
    uint64_t src0_bytes_projected = 0;
};



static void one_direct_manifest_report_atexit() {
    if (!g_one_direct_manifest.enabled) {
        return;
    }
    std::fprintf(stderr,
        "[moe_stream] one direct manifest: entries=%zu manifest_bytes=%lu hits=%lu misses=%lu"
        " reads=%lu bytes=%lu failures=%lu direct_enabled=%d direct_reads=%lu"
        " direct_failures=%lu direct_fallbacks=%lu\n",
        g_one_direct_manifest.entries.size(),
        g_one_direct_manifest.manifest_bytes,
        g_one_direct_manifest.hits.load(),
        g_one_direct_manifest.misses.load(),
        g_one_direct_manifest.reads.load(),
        g_one_direct_manifest.bytes.load(),
        g_one_direct_manifest.failures.load(),
        g_one_direct_manifest.direct_enabled ? 1 : 0,
        g_one_direct_manifest.direct_reads.load(),
        g_one_direct_manifest.direct_failures.load(),
        g_one_direct_manifest.direct_fallbacks.load());
}

static bool one_direct_manifest_parse_row(char * line, one_expert_pack_entry & entry) {
    char tensor[128] = {};
    long long expert = -1;
    unsigned long long offset = 0;
    unsigned long long nbytes = 0;
    const int n = std::sscanf(line, "%127[^,],%lld,%llu,%llu", tensor, &expert, &offset, &nbytes);
    if (n != 4 || expert < 0 || nbytes == 0) {
        return false;
    }
    entry = one_expert_pack_entry{};
    std::snprintf(entry.tensor, sizeof(entry.tensor), "%s", tensor);
    entry.expert_idx = (int32_t) expert;
    entry.offset = (uint64_t) offset;
    entry.nbytes = (uint64_t) nbytes;
    return true;
}

static void one_direct_manifest_init_once() {
    std::lock_guard<std::mutex> lk(g_one_direct_manifest.mu);
    if (g_one_direct_manifest.inited) {
        return;
    }
    g_one_direct_manifest.inited = true;

    const char * manifest_path = std::getenv("GGML_MOE_STREAM_ONE_DIRECT_MANIFEST");
    if (!manifest_path || !manifest_path[0]) {
        return;
    }
    const char * model_path = std::getenv("GGML_MOE_STREAM_ONE_DIRECT_MODEL");
    if (!model_path || !model_path[0]) {
        std::fprintf(stderr,
            "[moe_stream] one direct manifest: GGML_MOE_STREAM_ONE_DIRECT_MODEL is required when manifest is set\n");
        return;
    }

    FILE * fp = std::fopen(manifest_path, "r");
    if (!fp) {
        std::fprintf(stderr, "[moe_stream] one direct manifest: failed to open manifest %s\n", manifest_path);
        return;
    }

    std::vector<one_expert_pack_entry> entries;
    uint64_t total_bytes = 0;
    char line[4096];
    while (std::fgets(line, sizeof(line), fp)) {
        one_expert_pack_entry entry;
        if (!one_direct_manifest_parse_row(line, entry)) {
            continue;
        }
        total_bytes += entry.nbytes;
        entries.push_back(entry);
    }
    std::fclose(fp);

    if (entries.empty()) {
        std::fprintf(stderr, "[moe_stream] one direct manifest: no entries loaded from %s\n", manifest_path);
        return;
    }

    g_one_direct_manifest.prefill_entries = entries;

    std::sort(entries.begin(), entries.end(), [](const one_expert_pack_entry & a, const one_expert_pack_entry & b) {
        const int name_cmp = std::strcmp(a.tensor, b.tensor);
        if (name_cmp != 0) {
            return name_cmp < 0;
        }
        if (a.expert_idx != b.expert_idx) {
            return a.expert_idx < b.expert_idx;
        }
        return a.nbytes < b.nbytes;
    });

    const int fd = ::open(model_path, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "[moe_stream] one direct manifest: model open failed: %s\n", model_path);
        return;
    }

    g_one_direct_manifest.fd = fd;
    const char * io_env = std::getenv("GGML_MOE_STREAM_ONE_DIRECT_IO");
    if (io_env && (std::strcmp(io_env, "direct") == 0 || std::strcmp(io_env, "odirect") == 0)) {
#if defined(__linux__) && defined(O_DIRECT)
        g_one_direct_manifest.fd_direct = ::open(model_path, O_RDONLY | O_DIRECT);
        if (g_one_direct_manifest.fd_direct >= 0) {
            g_one_direct_manifest.direct_enabled = true;
            std::fprintf(stderr, "[moe_stream] one direct manifest: O_DIRECT model reads enabled: %s\n", model_path);
        } else {
            std::fprintf(stderr, "[moe_stream] one direct manifest: O_DIRECT open failed; using buffered model reads: %s\n", model_path);
        }
#else
        std::fprintf(stderr, "[moe_stream] one direct manifest: O_DIRECT requested but unavailable; using buffered model reads\n");
#endif
    }

    g_one_direct_manifest.entries = std::move(entries);
    g_one_direct_manifest.manifest_bytes = total_bytes;
    g_one_direct_manifest.enabled = true;
    std::atexit(one_direct_manifest_report_atexit);
    std::fprintf(stderr, "[moe_stream] one direct manifest: loaded %zu entries bytes=%lu from %s\n",
            g_one_direct_manifest.entries.size(), g_one_direct_manifest.manifest_bytes, manifest_path);
}

[[maybe_unused]] static const one_expert_pack_entry * one_direct_manifest_lookup(const char * tensor_name, int64_t expert_idx, size_t nbytes) {
    one_direct_manifest_init_once();
    if (!g_one_direct_manifest.enabled || !tensor_name || !tensor_name[0]) {
        return nullptr;
    }
    size_t lo = 0;
    size_t hi = g_one_direct_manifest.entries.size();
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const one_expert_pack_entry & e = g_one_direct_manifest.entries[mid];
        int cmp = std::strcmp(e.tensor, tensor_name);
        if (cmp == 0) {
            if (e.expert_idx < expert_idx) cmp = -1;
            else if (e.expert_idx > expert_idx) cmp = 1;
            else if (e.nbytes < nbytes) cmp = -1;
            else if (e.nbytes > nbytes) cmp = 1;
        }
        if (cmp < 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < g_one_direct_manifest.entries.size()) {
        const one_expert_pack_entry & e = g_one_direct_manifest.entries[lo];
        if (std::strcmp(e.tensor, tensor_name) == 0 && e.expert_idx == expert_idx && e.nbytes == nbytes) {
            ++g_one_direct_manifest.hits;
            return &e;
        }
    }
    ++g_one_direct_manifest.misses;
    return nullptr;
}









struct one_direct_hot_pool_state {
    void * pool = nullptr;
    size_t pool_sz = 0;
    size_t slot_sz = 0;
    int n_slots = 0;
    bool inited = false;
    bool enabled = false;
    bool prefill_done = false;
    std::vector<one_expert_pack_entry> slot_entries;
    std::unordered_map<std::string, int> slot_lookup;
    std::mutex mu;
    uint64_t attempted = 0;
    uint64_t inserted = 0;
    uint64_t read_failures = 0;
    uint64_t copy_failures = 0;
    uint64_t transform_failures = 0;
    uint64_t bytes = 0;
    double prefill_elapsed_ms = 0.0;
    double read_elapsed_ms = 0.0;
    double h2d_elapsed_ms = 0.0;
    double transform_elapsed_ms = 0.0;
    bool transposed = false;
    std::atomic<uint64_t> lookups{0};
    std::atomic<uint64_t> hits{0};
    std::atomic<bool> async_started{false};
    std::atomic<bool> async_done{false};
    bool lookup_built = false;
    std::thread async_thread;
};

static one_direct_hot_pool_state g_one_direct_hot_pool;

































struct one_prefill_entry {
    std::string tensor;
    int64_t expert = -1;
    int64_t score = 0;
};

struct one_prefill_state {
    bool initialized = false;
    bool enabled = false;
    bool registered = false;
    bool done = false;
    size_t limit = 0;
    std::vector<one_prefill_entry> entries;
    std::mutex mu;
    uint64_t attempted = 0;
    uint64_t inserted = 0;
    uint64_t pack_misses = 0;
    uint64_t read_failures = 0;
    uint64_t bytes = 0;
    double elapsed_ms = 0.0;
};

static one_prefill_state g_one_prefill;
















































extern "C" void ggml_cuda_moe_stream_one_gate_up_prefetch(
        const char * src0_name,
        size_t src0_bytes,
        const int * expert_indices,
        const int32_t * route_ranks,
        int count) {
    // The experimental object/cache or probe path is retired.
}

extern "C" bool ggml_cuda_moe_stream_one_cache_dev_ptrs_or_insert(
        const char * src0_name,
        const void * src0_base_data,
        size_t nb02,
        size_t src0_bytes,
        const int * expert_indices,
        const int32_t * route_ranks,
        int count,
        cudaStream_t stream,
        const void ** output_ptrs,
        int * cache_states,
        ggml_moe_expert_descriptor * descriptors) {
    // The experimental object/cache or probe path is retired.
    return false;
}



extern "C" void ggml_cuda_moe_stream_one_cache_release_descriptors(
        const ggml_moe_expert_descriptor * descriptors,
        int count,
        cudaStream_t stream) {
    // The experimental object/cache or probe path is retired.
}

extern "C" void ggml_cuda_moe_stream_one_cache_release_descriptors_synchronized(
        const ggml_moe_expert_descriptor * descriptors,
        int count) {
    // The experimental object/cache or probe path is retired.
}

extern "C" bool ggml_cuda_moe_stream_one_cache_contains(
    const char *src0_name,
    const void *src0_data,
    size_t src0_bytes,
    int64_t expert_index)
{
    // The experimental object/cache or probe path is retired.
    return false;
}







extern "C" const void * ggml_cuda_moe_stream_one_cache_dev_ptr_or_insert(
    const char *src0_name,
    const void *src0_base_data,
    size_t nb02,
    size_t src0_bytes,
    int64_t expert_index,
    cudaStream_t st,
    int *cache_state)
{
    if (cache_state) {
        *cache_state = 0;
    }
    // The experimental object/cache or probe path is retired.
    return nullptr;
}

extern "C" bool ggml_cuda_moe_stream_one_cache_ready(void) {
    // The experimental object/cache or probe path is retired.
    return false;
}



struct one_trace_state {
    std::mutex mu;
    FILE * fp = nullptr;
    bool initialized = false;
    std::chrono::steady_clock::time_point start;
    std::atomic<uint64_t> seq{0};
};

static one_trace_state g_one_trace;

struct one_stage_profile_role {
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> cache_misses{0};
    std::atomic<uint64_t> src0_ns{0};
    std::atomic<uint64_t> host_prepare_ns{0};
    std::atomic<uint64_t> enqueue_ns{0};
    std::atomic<uint64_t> sync_ns{0};
    std::atomic<uint64_t> scatter_ns{0};
    std::atomic<uint64_t> total_ns{0};
    std::atomic<uint64_t> weight_h2d_us{0};
    std::atomic<uint64_t> input_h2d_us{0};
    std::atomic<uint64_t> kernel_us{0};
    std::atomic<uint64_t> d2h_us{0};
    std::atomic<uint64_t> weight_h2d_copies{0};
    std::atomic<uint64_t> weight_h2d_bytes{0};
    std::atomic<uint64_t> input_h2d_bytes{0};
    std::atomic<uint64_t> d2h_bytes{0};
};

static one_stage_profile_role g_one_stage_profile[3];

struct one_decode_weight_profile_slot {
    cudaEvent_t start = nullptr;
    cudaEvent_t done = nullptr;
    bool pending = false;
    int role = -1;
    size_t bytes = 0;
};

static one_decode_weight_profile_slot g_one_decode_weight_profile[16];




















static bool gate_updown_cosubmit_enabled() {
    const char *env = std::getenv("GGML_MOE_GATE_UPDOWN_COSUBMIT");
    return env && env[0] && env[0] != '0';
}









struct sparse_graph_probe_state {
    std::mutex mu;
    FILE * fp = nullptr;
    bool initialized = false;
    std::atomic<uint64_t> seq{0};
};

static sparse_graph_probe_state g_sparse_graph_probe;

















static const int8_t moe_stream_mxfp4_values[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};











static std::atomic<uint64_t> g_q80_probe_records{0};



static __device__ __forceinline__ int moe_stream_mxfp4_value_dev(int q) {
    switch (q & 0x0F) {
        case 0x0: return 0;
        case 0x1: return 1;
        case 0x2: return 2;
        case 0x3: return 3;
        case 0x4: return 4;
        case 0x5: return 6;
        case 0x6: return 8;
        case 0x7: return 12;
        case 0x8: return 0;
        case 0x9: return -1;
        case 0xA: return -2;
        case 0xB: return -3;
        case 0xC: return -4;
        case 0xD: return -6;
        case 0xE: return -8;
        default:  return -12;
    }
}



static __device__ __forceinline__ float moe_stream_q80_cpu_compat_hsum8(
        const float acc1[8], const float acc2[8]) {
    float x0 = acc1[0] + acc2[0];
    float x1 = acc1[1] + acc2[1];
    float x2 = acc1[2] + acc2[2];
    float x3 = acc1[3] + acc2[3];
    float x4 = acc1[4] + acc2[4];
    float x5 = acc1[5] + acc2[5];
    float x6 = acc1[6] + acc2[6];
    float x7 = acc1[7] + acc2[7];

    float r0 = x4 + x0;
    float r1 = x5 + x1;
    float r2 = x6 + x2;
    float r3 = x7 + x3;
    float s0 = r0 + r2;
    float s1 = r1 + r3;
    return s0 + s1;
}



static __device__ __forceinline__ int moe_stream_q80_cpu_compat_one_lane(
        const block_mxfp4 & x,
        const block_q8_0 & y,
        int lane) {
    const int group = lane & 3;
    const bool high = lane >= 4;
    int sum = 0;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int j = group * 4 + r;
        const uint8_t q = x.qs[j];
        const int q4 = high ? (q >> 4) : (q & 0x0F);
        const int q8_idx = high ? j + QK_MXFP4/2 : j;
        sum += y.qs[q8_idx] * moe_stream_mxfp4_value_dev(q4);
    }
    return sum;
}









template<bool TRANSPOSED>
static __global__ void moe_stream_q80_hot_batch_cpu_compat_rowtile_kernel(
        const char * const * __restrict__ src0_rows,
        size_t nb01,
        const char * __restrict__ q80,
        size_t q80_row_size,
        int64_t ne00,
        int rows_to_probe,
        int cols_to_probe,
        float * __restrict__ out) {
    constexpr int tile_cols = 16;
    __shared__ block_q8_0 sh_y[2];

    const int row = (int) blockIdx.x;
    const int tile_col0 = (int) blockIdx.y * tile_cols;
    const int col_group = threadIdx.x >> 4;
    const int sublane = threadIdx.x & 15;
    const int lane = threadIdx.x & 31;
    const int col = tile_col0 + col_group;
    const bool active = row < rows_to_probe && col < cols_to_probe;
    const int base_lane = (lane & 16) ? 16 : 0;
    const uint32_t half_mask = (lane & 16) ? 0xffff0000u : 0x0000ffffu;
    const int lane8 = sublane & 7;

    const char * src0 = active ? src0_rows[row] : nullptr;
    const block_mxfp4 * x_raw = (const block_mxfp4 *) (active ? src0 + (size_t) col * nb01 : nullptr);
    const block_mxfp4 * x_transposed = (const block_mxfp4 *) src0;
    const block_q8_0 * y = (const block_q8_0 *) (q80 + (size_t) row * q80_row_size);
    const int64_t nb = ne00 / QK_MXFP4;

    float acc = 0.0f;
    int64_t ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        char * sh_bytes = (char *) sh_y;
        const char * y_bytes = (const char *) (y + ib);
        for (int off = (int) threadIdx.x; off < (int) (2 * sizeof(block_q8_0)); off += (int) blockDim.x) {
            sh_bytes[off] = y_bytes[off];
        }
        __syncthreads();

        if (active) {
            const bool second = sublane >= 8;
            const int64_t block_idx = ib + (second ? 1 : 0);
            const block_mxfp4 & xb = TRANSPOSED ?
                x_transposed[block_idx * (int64_t) cols_to_probe + col] :
                x_raw[block_idx];
            const block_q8_0 & yb = sh_y[second ? 1 : 0];
            const int p = moe_stream_q80_cpu_compat_one_lane(xb, yb, lane8);
            const float scale = __half2float(yb.d) * (ggml_cuda_e8m0_to_fp32(xb.e) * 0.5f);
            acc = fmaf(scale, (float) p, acc);
        }
        __syncthreads();
    }

    float acc1[8];
    float acc2[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc1[i] = __shfl_sync(half_mask, acc, base_lane + i);
        acc2[i] = __shfl_sync(half_mask, acc, base_lane + 8 + i);
    }

    if (active && sublane == 0) {
        float sum = moe_stream_q80_cpu_compat_hsum8(acc1, acc2);
        for (; ib < nb; ++ib) {
            int sumi1 = 0;
            int sumi2 = 0;
            const block_mxfp4 & xb = TRANSPOSED ?
                x_transposed[ib * (int64_t) cols_to_probe + col] :
                x_raw[ib];
#pragma unroll
            for (int j = 0; j < QK_MXFP4/2; ++j) {
                const uint8_t q = xb.qs[j];
                sumi1 += y[ib].qs[j] * moe_stream_mxfp4_value_dev(q & 0x0F);
                sumi2 += y[ib].qs[j + QK_MXFP4/2] * moe_stream_mxfp4_value_dev(q >> 4);
            }
            const float scale = __half2float(y[ib].d) * (ggml_cuda_e8m0_to_fp32(xb.e) * 0.5f);
            sum = fmaf(scale, (float) (sumi1 + sumi2), sum);
        }
        out[(size_t) row * (size_t) cols_to_probe + (size_t) col] = sum;
    }
}







extern "C" void ggml_cuda_moe_stream_q80_probe(
    int src0_type_int,
    const char *src0_name,
    int64_t expert_index,
    const void *src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const void *src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_ne1,
    int64_t cne1,
    const float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const ggml_moe_row_mapping *rows) {
    // The experimental object/cache or probe path is retired.
}







static std::atomic<uint64_t> g_q80_write_calls{0};
static std::atomic<uint64_t> g_q80_write_reports{0};

extern "C" void ggml_cuda_moe_stream_q80_write(
    int src0_type_int,
    const char *src0_name,
    int64_t expert_index,
    const void *src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const void *src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_ne1,
    int64_t cne1,
    float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const ggml_moe_row_mapping *rows) {
    // The experimental object/cache or probe path is retired.
}







static std::atomic<uint64_t> g_q80_hot_batch_probe_seq{0};











extern "C" void ggml_cuda_moe_stream_q80_hot_batch_probe(
    int src0_type_int,
    const char *src0_name,
    int64_t n_as,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const void *src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_ne1,
    const float *src1_f32,
    size_t src1_nb1,
    size_t src1_nb2,
    const int64_t *matrix_row_counts,
    const ggml_moe_row_mapping *matrix_rows,
    int64_t rows_per_expert,
    const float *dst,
    size_t dst_nb1,
    size_t dst_nb2) {
    // The experimental object/cache or probe path is retired.
}



static std::atomic<uint64_t> g_q80_skip_calls{0};
static std::atomic<uint64_t> g_q80_skip_reports{0};

struct q80_skip_profile_state {
    std::atomic<uint64_t> calls_entered{0};
    std::atomic<uint64_t> calls_attempted{0};
    std::atomic<uint64_t> calls_ok{0};
    std::atomic<uint64_t> calls_failed{0};
    std::atomic<uint64_t> src0_bytes{0};
    std::atomic<uint64_t> q80_bytes{0};
    std::atomic<uint64_t> out_bytes{0};
    std::atomic<uint64_t> total_ns{0};
    std::atomic<uint64_t> host_src0_ns{0};
    std::atomic<uint64_t> host_q80_ns{0};
    std::atomic<uint64_t> cuda_alloc_ns{0};
    std::atomic<uint64_t> h2d_ns{0};
    std::atomic<uint64_t> kernel_sync_ns{0};
    std::atomic<uint64_t> d2h_ns{0};
    std::atomic<uint64_t> scatter_ns{0};
    std::atomic<uint64_t> report_ns{0};
    std::atomic<uint64_t> cuda_free_ns{0};
};

static q80_skip_profile_state g_q80_skip_profile;







extern "C" bool ggml_cuda_moe_stream_q80_skip(
    int src0_type_int,
    const char *src0_name,
    int64_t expert_index,
    const void *src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const void *src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_ne1,
    int64_t cne1,
    float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const ggml_moe_row_mapping *rows) {
    // The experimental object/cache or probe path is retired.
    return false;
}









extern "C" bool ggml_cuda_moe_stream_available(void) {
    // The opt-in runtime streaming compute path is retired.
    return false;
}

extern "C" int ggml_cuda_host_register(void *p, size_t n) {
    if (!p || n == 0) return 0;
    cudaError_t err = cudaHostRegister(p, n, cudaHostRegisterDefault);
    if (err == cudaSuccess) {
        std::fprintf(stderr, "[moe_stream] cudaHostRegister %.1f GiB OK\n",
                     n / (1024.0*1024.0*1024.0));
        return 1;
    }
    std::fprintf(stderr, "[moe_stream] cudaHostRegister %.1f GiB FAILED: %s\n",
                 n / (1024.0*1024.0*1024.0), cudaGetErrorString(err));
    return 0;
}

static void release_slot(int s);
static void flush_deferred_scatter();

// Acquire a slot via test-and-set on per-slot flag; flush deferred work if this
// thread filled all stream slots before the CPU-side op-level sync point.


static void release_slot(int s) {
    g_slots[s].in_use.clear(std::memory_order_release);
}

static void flush_deferred_scatter() {
    if (tls_pending.empty()) return;
    for (auto &ds : tls_pending) {
        slot_ctx &ctx = g_slots[ds.slot];
        cudaStreamSynchronize(ctx.stream);
        const float *src_buf = (const float *)ctx.h_scratch;
        for (int64_t k = 0; k < ds.cne1; ++k) {
            const int32_t i1 = ds.i1[k];
            const int32_t i2 = ds.i2[k];
            float *dst_row = (float *)((char *)ds.dst_base + i1 * ds.dst_nb1 + i2 * ds.dst_nb2);
            const float *src_row = src_buf + k * ds.ne01;
            std::memcpy(dst_row, src_row, ds.ne01 * sizeof(float));
        }
        release_slot(ds.slot);
    }
    tls_pending.clear();
}

extern "C" bool ggml_cuda_moe_stream_one(
    int  src0_type_int,
    const char *src0_name,
    int64_t expert_index,
    const void *src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const float *src1_f32,
    size_t src1_nb1, size_t src1_nb2,
    int64_t src1_ne1,
    int64_t cne1,
    const void *src1_q8_1,
    size_t src1_padded_num_cols,
    float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const ggml_moe_row_mapping *rows)
{
    // The opt-in runtime streaming compute path is retired.
    return false;
}

extern "C" void ggml_cuda_moe_stream_sync(void) {
    if (gate_updown_cosubmit_enabled()) {
        ggml_cuda_moe_stream_batch_flush_gate_updown_cosubmit();
    }
    flush_deferred_scatter();
}


} // namespace ggml_moe_common_adapter

#endif // GGML_CUDA_MOE_STREAM
