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
static bool ram_cache_finish_pending_event_locked(size_t slot_index);
static bool ram_cache_release_common_lease_locked(size_t slot_index);
static bool ram_cache_discard_identity_locked(size_t slot_index);

static ggml_moe_common::role_cache_view vram_cache_role_view() {
    ggml_moe_common::role_cache_view view;
    view.pool = g_vcache.pool;
    view.pool_bytes = g_vcache.pool_sz;
    view.slot_bytes = g_vcache.slot_sz;
    view.slot_count = g_vcache.n_slots;
    view.slot_keys = g_vcache.slot_key;
    view.slot_generations = g_vcache.slot_generation;
    view.slot_pending = g_vcache.slot_pending;
    view.slot_filling = g_vcache.slot_filling;
    view.slot_lease_counts = g_vcache.slot_lease_count;
    view.slot_ready_events = g_vcache.slot_ready;
    view.slot_lease_events = g_vcache.slot_lease_ready;
    view.slot_lease_pending = g_vcache.slot_lease_pending;
    return view;
}

static ggml_moe_common::role_cache_view vram_cache_readiness_view() {
    ggml_moe_common::role_cache_view view = vram_cache_role_view();
    view.slot_payload_bytes = g_vcache.slot_payload_bytes;
    view.slot_ready_markers = g_vcache.slot_ready_marker;
    view.slot_states = g_vcache.slot_state;
    return view;
}

static bool vram_cache_common_h2d_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_MOE_COMMON_ROLE_H2D");
        return !env || !env[0] || env[0] != '0';
    }();
    return enabled;
}

static bool ds4_common_object_cqe_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_MOE_COMMON_OBJECT_CQE");
        return !env || !env[0] || env[0] != '0';
    }();
    return enabled;
}

static bool ds4_common_object_staging_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_MOE_COMMON_OBJECT_STAGING");
        return !env || !env[0] || env[0] != '0';
    }();
    return enabled;
}

static bool ds4_dependency_h2d_requested() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_MOE_DEPENDENCY_H2D");
        return !env || !env[0] || env[0] != '0';
    }();
    return enabled;
}

static bool vram_cache_hash_lookup_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_ONE_HASH_LOOKUP");
        return env && env[0] && env[0] != '0';
    }();
    return enabled;
}

static size_t vram_cache_hash_bucket(uintptr_t key) {
    uint64_t value = (uint64_t) key;
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;
    return (size_t) value & VRAM_CACHE_HT_MASK;
}

static void vram_cache_hash_init_locked() {
    for (vram_ht_entry & entry : g_vcache.ht) {
        entry.key = 0;
        entry.slot = -1;
    }
}

static int vram_cache_hash_find_locked(uintptr_t key) {
    if (key == 0) return -1;
    size_t bucket = vram_cache_hash_bucket(key);
    for (size_t probe = 0; probe < VRAM_CACHE_HT_SIZE; ++probe) {
        const vram_ht_entry & entry = g_vcache.ht[bucket];
        if (entry.slot == -1) return -1;
        if (entry.slot >= 0 && entry.key == key) return entry.slot;
        bucket = (bucket + 1) & VRAM_CACHE_HT_MASK;
    }
    return -1;
}

static bool vram_cache_hash_insert_locked(uintptr_t key, int slot) {
    if (key == 0 || slot < 0) return false;
    size_t bucket = vram_cache_hash_bucket(key);
    size_t tombstone = VRAM_CACHE_HT_SIZE;
    for (size_t probe = 0; probe < VRAM_CACHE_HT_SIZE; ++probe) {
        vram_ht_entry & entry = g_vcache.ht[bucket];
        if (entry.slot >= 0 && entry.key == key) {
            entry.slot = slot;
            return true;
        }
        if (entry.slot == -2 && tombstone == VRAM_CACHE_HT_SIZE) tombstone = bucket;
        if (entry.slot == -1) {
            vram_ht_entry & dst = g_vcache.ht[tombstone == VRAM_CACHE_HT_SIZE ? bucket : tombstone];
            dst.key = key;
            dst.slot = slot;
            return true;
        }
        bucket = (bucket + 1) & VRAM_CACHE_HT_MASK;
    }
    if (tombstone != VRAM_CACHE_HT_SIZE) {
        g_vcache.ht[tombstone].key = key;
        g_vcache.ht[tombstone].slot = slot;
        return true;
    }
    return false;
}

static void vram_cache_hash_erase_locked(uintptr_t key, int expected_slot) {
    if (key == 0) return;
    size_t bucket = vram_cache_hash_bucket(key);
    for (size_t probe = 0; probe < VRAM_CACHE_HT_SIZE; ++probe) {
        vram_ht_entry & entry = g_vcache.ht[bucket];
        if (entry.slot == -1) return;
        if (entry.slot >= 0 && entry.key == key) {
            if (expected_slot < 0 || entry.slot == expected_slot) {
                entry.key = 0;
                entry.slot = -2;
            }
            return;
        }
        bucket = (bucket + 1) & VRAM_CACHE_HT_MASK;
    }
}

static bool vram_cache_score_victim_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_ONE_SCORE_VICTIM");
        return env && env[0] && env[0] != '0';
    }();
    return enabled;
}

static uint32_t vram_cache_note_demand_locked(uintptr_t key) {
    if (!vram_cache_score_victim_enabled() || key == 0) return 0;
    uint32_t & score = g_vram_cache_demand_scores[key];
    if (score != UINT32_MAX) ++score;
    return score;
}

static uint32_t vram_cache_demand_score_locked(uintptr_t key) {
    const auto it = g_vram_cache_demand_scores.find(key);
    return it == g_vram_cache_demand_scores.end() ? 0 : it->second;
}

struct ds4_ram_key_stat {
    int layer = -1;
    int expert = -1;
    uint32_t demands = 0;
    uint32_t ram_hits = 0;
};

static std::unordered_map<uintptr_t, ds4_ram_key_stat> g_ds4_ram_key_stats;
static bool ds4_ram_common_object_key(
        uintptr_t key,
        size_t bytes,
        ggml_moe_common::object_key &object);
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

static bool ds4_ram_score_policy_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_DS4_RAM_SCORE_POLICY");
        return env && env[0] && env[0] != '0';
    }();
    return enabled;
}

static uint32_t ds4_ram_score_min_demands() {
    static const uint32_t value = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_DS4_RAM_SCORE_MIN_DEMANDS");
        const unsigned long parsed = env && env[0] ? std::strtoul(env, nullptr, 10) : 2UL;
        return (uint32_t) std::max(1UL, std::min(parsed, 1000000UL));
    }();
    return value;
}

static int ds4_ram_layer_quota() {
    static const int value = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_DS4_RAM_LAYER_QUOTA");
        const long parsed = env && env[0] ? std::strtol(env, nullptr, 10) : 0L;
        return (int) std::max(0L, std::min(parsed, 16384L));
    }();
    return value;
}

static bool ds4_ram_layer_allowed(int layer) {
    static const std::vector<uint8_t> allowed = [] {
        std::vector<uint8_t> values(43, 1);
        const char * env = std::getenv("GGML_MOE_STREAM_DS4_RAM_ADMIT_LAYERS");
        if (!env || !env[0]) return values;
        std::fill(values.begin(), values.end(), 0);
        const char * p = env;
        while (*p) {
            while (*p == ' ' || *p == '\t' || *p == ',') ++p;
            if (!*p) break;
            char * end = nullptr;
            const long first = std::strtol(p, &end, 10);
            if (end == p) {
                while (*p && *p != ',') ++p;
                continue;
            }
            p = end;
            long last = first;
            if (*p == '-') {
                ++p;
                last = std::strtol(p, &end, 10);
                if (end != p) p = end;
            }
            const long lo = std::max(0L, std::min(first, last));
            const long hi = std::min(42L, std::max(first, last));
            for (long value = lo; value <= hi; ++value) values[(size_t) value] = 1;
            while (*p && *p != ',') ++p;
        }
        return values;
    }();
    return layer >= 0 && layer < (int) allowed.size() && allowed[(size_t) layer] != 0;
}

static FILE * vram_accounting_fp() {
    static FILE * fp = nullptr;
    static bool initialized = false;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    if (!initialized) {
        initialized = true;
        const char * path = std::getenv("GGML_MOE_VRAM_ACCOUNTING_OUT");
        if (path && path[0]) {
            fp = std::fopen(path, "w");
            if (fp) {
                std::setvbuf(fp, nullptr, _IOLBF, 0);
                std::fprintf(fp,
                    "t_us,component,event,free_mib,total_mib,used_mib,cache_pool_mib,cache_slots,slot_mib,detail\n");
            } else {
                std::fprintf(stderr, "[moe_stream] VRAM accounting: failed to open %s\n", path);
            }
        }
    }
    return fp;
}

static uint64_t vram_accounting_now_us() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

static void vram_accounting_record(const char * event, const char * detail = "") {
    FILE * fp = vram_accounting_fp();
    if (!fp) return;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return;
    }
    const double mib = 1024.0 * 1024.0;
    std::fprintf(fp,
        "%llu,one_stream,%s,%.3f,%.3f,%.3f,%.3f,%d,%.3f,%s\n",
        (unsigned long long) vram_accounting_now_us(),
        event ? event : "",
        free_bytes / mib,
        total_bytes / mib,
        (total_bytes - free_bytes) / mib,
        g_vcache.pool_sz / mib,
        g_vcache.n_slots,
        g_vcache.slot_sz / mib,
        detail ? detail : "");
}

static void vram_cache_report_atexit() {
    uint64_t h = g_vcache.hits.load();
    uint64_t m = g_vcache.misses.load();
    if (h + m != 0) {
        std::fprintf(stderr, "[moe_stream] VRAM cache: hits=%lu misses=%lu hit_rate=%.1f%%\n",
                     h, m, 100.0 * h / (h + m));
    }
    const uint64_t freeze_rejections = g_vcache.freeze_rejections.load();
    if (g_vcache.lru_frozen || freeze_rejections != 0) {
        std::fprintf(stderr, "[moe_stream] VRAM cache freeze: active=%d rejected_misses=%lu used_slots=%d/%d\n",
                     g_vcache.lru_frozen ? 1 : 0, freeze_rejections,
                     g_vcache.used_slots, g_vcache.n_slots);
    }
    uint64_t rh = g_vcache.ram_hits.load();
    uint64_t rm = g_vcache.ram_misses.load();
    if (rh + rm != 0 || g_vcache.ram_cache.slab.storage.host) {
        const uint64_t lease_acquires = g_vcache.ram_host_lease_acquires.load();
        const uint64_t lease_releases = g_vcache.ram_host_lease_releases.load();
        const double hit_rate = (rh + rm) ? (100.0 * rh / (rh + rm)) : 0.0;
        std::fprintf(stderr,
            "[moe_stream] RAM cache: hits=%lu misses=%lu hit_rate=%.1f%% inserts=%lu evictions=%lu pinned=%.2f MiB slots=%d\n",
            rh, rm, hit_rate, g_vcache.ram_inserts.load(), g_vcache.ram_evictions.load(),
            g_vcache.ram_cache.slab.storage.capacity / (1024.0 * 1024.0),
            (int)g_vcache.ram_cache.slots.size());
        std::fprintf(stderr,
            "[moe_stream] RAM common object: ready=%zu publishes=%lu non_object_stores=%lu "
            "host_store_attempts=%lu valid=%lu admitted=%lu slot_selected=%lu "
            "staging_requested=%lu staging_eligible=%lu "
            "lease_acquires=%lu lease_releases=%lu lease_outstanding=%lu "
            "lease_failures=%lu index_failures=%lu\n",
            ggml_moe_common::object_host_cache_ready_count(g_vcache.ram_cache),
            g_vcache.ram_full_object_publishes.load(),
            g_vcache.ram_non_object_stores.load(),
            g_vcache.ram_host_store_attempts.load(),
            g_vcache.ram_host_store_valid.load(),
            g_vcache.ram_host_store_admitted.load(),
            g_vcache.ram_host_store_slot_selected.load(),
            g_vcache.ram_host_store_staging_requested.load(),
            g_vcache.ram_host_store_common_eligible.load(),
            lease_acquires,
            lease_releases,
            lease_acquires >= lease_releases ? lease_acquires - lease_releases : 0,
            g_vcache.ram_host_lease_failures.load(),
            (unsigned long)g_vcache.ram_cache.ready_index_failures);
        const ggml_moe_common::object_host_staging_counters &staging =
            g_vcache.ram_cache.staging;
        const uint64_t staging_begins = staging.begins.load(std::memory_order_relaxed);
        const uint64_t staging_publishes = staging.publishes.load(std::memory_order_relaxed);
        const uint64_t staging_aborts = staging.aborts.load(std::memory_order_relaxed);
        std::fprintf(stderr,
            "[moe_stream] RAM common staging: begins=%lu publishes=%lu aborts=%lu "
            "failures=%lu outstanding=%lu\n",
            (unsigned long)staging_begins,
            (unsigned long)staging_publishes,
            (unsigned long)staging_aborts,
            (unsigned long)staging.failures.load(std::memory_order_relaxed),
            (unsigned long)(staging_begins >= staging_publishes + staging_aborts ?
                staging_begins - staging_publishes - staging_aborts : 0));
        const ggml_moe_common::object_host_completion_counters &completion =
            g_vcache.ram_cache.completion;
        const uint64_t common_lease_acquires =
            completion.h2d_lease_acquires.load(std::memory_order_relaxed);
        const uint64_t common_lease_releases =
            completion.h2d_lease_releases.load(std::memory_order_relaxed);
        std::fprintf(stderr,
            "[moe_stream] RAM common H2D: begins=%lu records=%lu completions=%lu "
            "cancels=%lu failures=%lu lease_acquires=%lu lease_releases=%lu "
            "lease_outstanding=%lu stream_waits=%lu\n",
            (unsigned long)completion.h2d_begins.load(std::memory_order_relaxed),
            (unsigned long)completion.h2d_records.load(std::memory_order_relaxed),
            (unsigned long)completion.h2d_completions.load(std::memory_order_relaxed),
            (unsigned long)completion.h2d_cancels.load(std::memory_order_relaxed),
            (unsigned long)completion.h2d_failures.load(std::memory_order_relaxed),
            (unsigned long)common_lease_acquires,
            (unsigned long)common_lease_releases,
            (unsigned long)(common_lease_acquires >= common_lease_releases ?
                common_lease_acquires - common_lease_releases : 0),
            (unsigned long)completion.h2d_stream_waits.load(std::memory_order_relaxed));
    }
    if (ds4_ram_score_policy_enabled()) {
        std::fprintf(stderr,
            "[moe_stream] DS4 RAM score: admissions=%lu min_demand_rejects=%lu "
            "victim_rejects=%lu layer_rejects=%lu min_demands=%u layer_quota=%d layers=%s\n",
            g_vcache.ram_score_admissions.load(),
            g_vcache.ram_score_min_demand_rejects.load(),
            g_vcache.ram_score_victim_rejects.load(),
            g_vcache.ram_score_layer_rejects.load(),
            ds4_ram_score_min_demands(), ds4_ram_layer_quota(),
            std::getenv("GGML_MOE_STREAM_DS4_RAM_ADMIT_LAYERS") ?
                std::getenv("GGML_MOE_STREAM_DS4_RAM_ADMIT_LAYERS") : "all");
        for (int layer = 0; layer < 43; ++layer) {
            if (g_ds4_ram_layer_demands[layer] == 0 && g_ds4_ram_layer_admissions[layer] == 0) continue;
            std::fprintf(stderr,
                "[moe_stream] DS4 RAM score layer: layer=%d demands=%lu hits=%lu admissions=%lu rejects=%lu\n",
                layer, g_ds4_ram_layer_demands[layer], g_ds4_ram_layer_hits[layer],
                g_ds4_ram_layer_admissions[layer], g_ds4_ram_layer_rejects[layer]);
        }
    }
    const uint64_t descriptor_acquires = g_vcache.descriptor_acquires.load();
    const uint64_t descriptor_releases = g_vcache.descriptor_releases.load();
    const uint64_t descriptor_cancels = g_vcache.descriptor_cancels.load();
    const uint64_t stale_releases = g_vcache.descriptor_stale_releases.load();
    if (descriptor_acquires != 0 || descriptor_releases != 0 || stale_releases != 0) {
        std::fprintf(stderr,
            "[moe_stream] expert descriptors: fixed_pool=1 acquires=%lu releases=%lu "
            "cancels=%lu outstanding=%lu stale_releases=%lu\n",
            descriptor_acquires, descriptor_releases, descriptor_cancels,
            descriptor_acquires >= descriptor_releases ?
                descriptor_acquires - descriptor_releases : 0,
            stale_releases);
    }
    if (vram_cache_score_victim_enabled()) {
        std::fprintf(stderr,
            "[moe_stream] VRAM score victim: selections=%lu candidate_lte_victim=%lu demand_keys=%zu\n",
            g_vcache.score_victim_selections.load(),
            g_vcache.score_candidate_lte_victim.load(),
            g_vram_cache_demand_scores.size());
    }
}

static bool moe_stream_dontneed_enabled() {
    static int enabled = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_DONTNEED");
        return env && env[0] && env[0] != '0';
    }();
    return enabled != 0;
}

static void moe_stream_dontneed_source_pages(const void * ptr, size_t size) {
#if defined(__linux__)
    if (!moe_stream_dontneed_enabled() || ptr == nullptr || size == 0) {
        return;
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return;
    }

    const uintptr_t begin = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t end = begin + size;
    const uintptr_t aligned_begin = begin & ~static_cast<uintptr_t>(page_size - 1);
    const uintptr_t aligned_end = (end + static_cast<uintptr_t>(page_size - 1)) & ~static_cast<uintptr_t>(page_size - 1);
    if (aligned_end <= aligned_begin) {
        return;
    }

    (void) madvise(reinterpret_cast<void *>(aligned_begin), aligned_end - aligned_begin, MADV_DONTNEED);
#else
    (void) ptr;
    (void) size;
#endif
}

static bool moe_stream_cache_admit_name_matches(const char * filter, const char * tensor) {
    if (!filter || !filter[0] || !tensor) {
        return false;
    }

    const char * p = filter;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == ',' || *p == ':') {
            ++p;
        }
        const char * start = p;
        while (*p && *p != ',' && *p != ':') {
            ++p;
        }
        const char * end = p;
        while (end > start && (end[-1] == ' ' || end[-1] == '\t')) {
            --end;
        }
        const size_t len = (size_t) (end - start);
        if (len > 0) {
            for (const char * hit = tensor; *hit; ++hit) {
                if (std::strncmp(hit, start, len) == 0) {
                    return true;
                }
            }
        }
    }
    return false;
}

static bool moe_stream_cache_admit_profile_applies(const char * tensor) {
    static const char * applies_filter = std::getenv("GGML_MOE_STREAM_CACHE_ADMIT_PROFILE_APPLIES_FILTER");
    return !applies_filter || !applies_filter[0] || moe_stream_cache_admit_name_matches(applies_filter, tensor);
}

static bool moe_stream_one_require_cache_admit(const char * tensor) {
    static const char * require_filter = std::getenv("GGML_MOE_STREAM_ONE_REQUIRE_CACHE_ADMIT_FILTER");
    return require_filter && require_filter[0] && moe_stream_cache_admit_name_matches(require_filter, tensor);
}

static int moe_stream_cache_partition_tail_slots() {
    static int slots = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_CACHE_PARTITION_TAIL_SLOTS");
        return (env && env[0]) ? std::atoi(env) : 0;
    }();
    return slots;
}

static bool moe_stream_cache_partition_tail_matches(const char * tensor) {
    static const char * filter = std::getenv("GGML_MOE_STREAM_CACHE_PARTITION_TAIL_FILTER");
    return filter && filter[0] && moe_stream_cache_admit_name_matches(filter, tensor);
}

static bool moe_stream_cache_partition_enabled() {
    static const char * filter = std::getenv("GGML_MOE_STREAM_CACHE_PARTITION_TAIL_FILTER");
    return filter && filter[0] && moe_stream_cache_partition_tail_slots() > 0;
}

static bool vram_cache_tensor_lru_enabled() {
    static int enabled = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_ONE_TENSOR_LRU");
        return (!env || !env[0] || env[0] != '0') ? 1 : 0;
    }();
    return enabled != 0;
}

static uint64_t vram_cache_tensor_tag(const char * tensor) {
    if (!tensor || !tensor[0]) {
        return 0;
    }
    uint64_t h = 1469598103934665603ULL;
    for (const unsigned char * p = (const unsigned char *) tensor; *p; ++p) {
        h ^= (uint64_t) *p;
        h *= 1099511628211ULL;
    }
    return h ? h : 1;
}

static bool moe_stream_cache_admit_allows(const char * tensor, int64_t expert) {
    static const char * name_filter = std::getenv("GGML_MOE_STREAM_CACHE_ADMIT_NAME_FILTER");
    if (name_filter && name_filter[0] && !moe_stream_cache_admit_name_matches(name_filter, tensor)) {
        return false;
    }
    if (!moe_stream_cache_admit_profile_applies(tensor)) {
        return true;
    }

    static std::mutex mu;
    static bool initialized = false;
    static bool enabled = false;
    static std::unordered_set<std::string> allow;

    if (!initialized) {
        std::lock_guard<std::mutex> lk(mu);
        if (!initialized) {
            const char * path = std::getenv("GGML_MOE_STREAM_CACHE_ADMIT_PROFILE");
            if (path && path[0]) {
                FILE * fp = std::fopen(path, "r");
                if (!fp) {
                    std::fprintf(stderr, "[moe_stream] cache admission: failed to open %s\n", path);
                } else {
                    char line[4096];
                    while (std::fgets(line, sizeof(line), fp)) {
                        char name[3072];
                        long long expert_id = -1;
                        if (std::sscanf(line, "%3071[^\t]\t%lld", name, &expert_id) == 2 && expert_id >= 0) {
                            allow.insert(std::string(name) + "\t" + std::to_string(expert_id));
                        }
                    }
                    std::fclose(fp);
                    enabled = true;
                    std::fprintf(stderr, "[moe_stream] cache admission: loaded %zu entries from %s\n",
                            allow.size(), path);
                }
            }
            initialized = true;
        }
    }

    if (!enabled) {
        return true;
    }
    if (!tensor) {
        return false;
    }
    const std::string key = std::string(tensor) + "\t" + std::to_string((long long) expert);
    return allow.find(key) != allow.end();
}

static void vram_cache_init(size_t expert_sz, size_t host_object_sz = 0) {
    if (g_vcache_inited) return;
    vram_accounting_record("one_cache_init_before");
    if (vram_cache_hash_lookup_enabled()) {
        std::lock_guard<std::mutex> lk(g_vcache.mu);
        vram_cache_hash_init_locked();
    }
    const char *env_one_mib = std::getenv("GGML_MOE_STREAM_ONE_CACHE_MIB");
    const char *env_one_gb = std::getenv("GGML_MOE_STREAM_ONE_CACHE_GB");
    const char *env_shared_mib = std::getenv("GGML_MOE_VRAM_CACHE_MIB");
    const char *env_shared_gb = std::getenv("GGML_MOE_VRAM_CACHE_GB");
    size_t budget_mib = 16ULL * 1024ULL;
    if (env_one_mib && env_one_mib[0]) {
        budget_mib = (size_t)std::strtoull(env_one_mib, nullptr, 10);
    } else if (env_one_gb && env_one_gb[0]) {
        budget_mib = (size_t)std::strtoull(env_one_gb, nullptr, 10) * 1024ULL;
    } else if (env_shared_mib && env_shared_mib[0]) {
        budget_mib = 0;
    } else if (env_shared_gb && env_shared_gb[0]) {
        budget_mib = (size_t)std::strtoull(env_shared_gb, nullptr, 10) * 1024ULL;
    }
    if (budget_mib == 0) { g_vcache_inited = true; return; }
    size_t budget = budget_mib * 1024ULL * 1024ULL;

    g_vcache.slot_sz = expert_sz;
    g_vcache.n_slots = (int)(budget / expert_sz);
    if (g_vcache.n_slots > VRAM_CACHE_HT_SIZE) g_vcache.n_slots = VRAM_CACHE_HT_SIZE;
    printf("budget:%zu,n_slot:%ld\n",budget,g_vcache.n_slots);
    if (g_vcache.n_slots < 1) { g_vcache_inited = true; return; }
    size_t alloc = (size_t)g_vcache.n_slots * expert_sz;
    if (cudaMalloc(&g_vcache.pool, alloc) != cudaSuccess) {
        std::fprintf(stderr, "[moe_stream] VRAM cache: cudaMalloc %.1f GiB FAILED\n",
                     alloc / (1024.0*1024.0*1024.0));
        cudaGetLastError();
        g_vcache.n_slots = 0;
        g_vcache_inited = true;
        vram_accounting_record("one_cache_alloc_failed");
        return;
    }
    g_vcache.pool_sz = alloc;
    const ggml_moe_common::role_cache_view schema_view = ggml_moe_common::fixed_role_cache_view(&g_vcache);
    if (!ggml_moe_common::role_cache_view_valid(schema_view) || schema_view.pool_bytes != g_vcache.pool_sz) {
        std::fprintf(stderr, "[moe_stream] VRAM cache: common role-cache schema validation failed\n");
        cudaFree(g_vcache.pool);
        g_vcache.pool = nullptr;
        g_vcache.pool_sz = 0;
        g_vcache.n_slots = 0;
        g_vcache_inited = true;
        vram_accounting_record("one_cache_schema_invalid");
        return;
    }
    std::fprintf(stderr, "[moe_stream] VRAM cache: common role-cache schema valid\n");
    std::fprintf(stderr, "[moe_stream] VRAM cache: %.1f GiB, %d slots (%.2f MiB each)\n",
                 alloc / (1024.0*1024.0*1024.0), g_vcache.n_slots,
                 expert_sz / (1024.0*1024.0));
    std::fprintf(stderr, "[moe_stream] VRAM cache eviction: %s\n",
                 vram_cache_tensor_lru_enabled() ? "tensor-local LRU" : "global LRU");
    std::fprintf(stderr, "[moe_stream] VRAM cache lookup: %s\n",
                 vram_cache_hash_lookup_enabled() ? "fixed hash table" : "linear scan");

    const char *env_ram_mib = std::getenv("GGML_MOE_STREAM_ONE_RAM_CACHE_MIB");
    const char *env_ram_gb  = std::getenv("GGML_MOE_STREAM_ONE_RAM_CACHE_GB");
    size_t ram_budget_mib = 0;
    if (env_ram_mib && env_ram_mib[0]) {
        ram_budget_mib = (size_t)std::strtoull(env_ram_mib, nullptr, 10);
    } else if (env_ram_gb && env_ram_gb[0]) {
        ram_budget_mib = (size_t)std::strtoull(env_ram_gb, nullptr, 10) * 1024ULL;
    }
    if (ram_budget_mib > 0) {
        if (host_object_sz == 0) host_object_sz = expert_sz;
        const size_t ram_budget = ram_budget_mib * 1024ULL * 1024ULL;
        size_t ram_slots = ram_budget / host_object_sz;
        if (ram_slots > VRAM_CACHE_HT_SIZE) ram_slots = VRAM_CACHE_HT_SIZE;
        std::fprintf(stderr,
                "[moe_stream] RAM cache request: %zu MiB, object=%.2f MiB, slots=%zu\n",
                ram_budget_mib,
                host_object_sz / (1024.0 * 1024.0),
                ram_slots);
        if (ram_slots > 0) {
            const size_t ram_alloc = ram_slots * host_object_sz;
            if (ggml_moe_common::object_host_cache_allocate_slab(
                    g_vcache.ram_cache, host_object_sz, ram_slots, false, 4096)) {
                if (!ggml_moe_common::object_host_cache_configure_ownership(
                            g_vcache.ram_cache, 0)) {
                    std::fprintf(stderr,
                            "[moe_stream] RAM cache: common Host ownership setup failed; RAM tier disabled\n");
                    ggml_moe_common::object_host_cache_release_slab(g_vcache.ram_cache);
                } else {
                    std::fprintf(stderr, "[moe_stream] RAM cache: %.1f GiB pinned, %d slots (%.2f MiB each)\n",
                                 ram_alloc / (1024.0*1024.0*1024.0), (int)ram_slots,
                                 host_object_sz / (1024.0*1024.0));
                }
            } else {
                std::fprintf(stderr, "[moe_stream] RAM cache: cudaHostAlloc %.1f GiB FAILED; RAM tier disabled\n",
                             ram_alloc / (1024.0*1024.0*1024.0));
                cudaGetLastError();
                ggml_moe_common::object_host_cache_release_slab(g_vcache.ram_cache);
            }
        }
    }

    vram_accounting_record("one_cache_alloc_after");
    std::atexit(vram_cache_report_atexit);
    g_vcache_inited = true;
}

static double vram_cache_freeze_hit_rate_threshold() {
    static const double threshold = []() {
        const char * env = std::getenv("GGML_MOE_STREAM_CACHE_FREEZE_HIT_RATE");
        if (!env || !env[0]) {
            return 0.0;
        }
        char * end = nullptr;
        double value = std::strtod(env, &end);
        if (end == env || value <= 0.0) {
            return 0.0;
        }
        if (value <= 1.0) {
            value *= 100.0;
        }
        return std::min(value, 100.0);
    }();
    return threshold;
}

static bool vram_cache_freeze_reject_miss() {
    const double threshold = vram_cache_freeze_hit_rate_threshold();
    if (threshold <= 0.0 || !g_vcache.pool || g_vcache.n_slots <= 0) {
        return false;
    }

    std::lock_guard<std::mutex> lk(g_vcache.mu);
    if (!g_vcache.lru_frozen) {
        if (g_vcache.used_slots < g_vcache.n_slots) {
            return false;
        }
        const uint64_t hits = g_vcache.hits.load(std::memory_order_relaxed);
        const uint64_t misses = g_vcache.misses.load(std::memory_order_relaxed);
        const uint64_t lookups = hits + misses;
        const double hit_rate = lookups == 0 ? 0.0 : 100.0 * (double) hits / (double) lookups;
        if (hit_rate < threshold) {
            return false;
        }
        g_vcache.lru_frozen = true;
        std::fprintf(stderr,
                "[moe_stream] VRAM cache LRU frozen: threshold=%.2f%% hit_rate=%.2f%% slots=%d/%d; misses use CPU fallback\n",
                threshold, hit_rate, g_vcache.used_slots, g_vcache.n_slots);
    }

    g_vcache.misses.fetch_add(1, std::memory_order_relaxed);
    g_vcache.freeze_rejections.fetch_add(1, std::memory_order_relaxed);
    return true;
}

static bool vram_cache_stream_wait_slot_ready_locked(int slot, cudaStream_t st) {
    return ggml_moe_common::role_cache_stream_wait_ready_event(
        vram_cache_role_view(), slot, st);
}

static bool vram_cache_stream_wait_slot_lease_locked(int slot, cudaStream_t st) {
    return ggml_moe_common::role_cache_stream_wait_lease_event(
        vram_cache_role_view(), slot, st);
}

static bool vram_cache_record_slot_lease_locked(int slot, cudaStream_t st) {
    return ggml_moe_common::role_cache_record_lease_event(
        vram_cache_role_view(), slot, st);
}

static bool ram_cache_slot_has_common_object(const ggml_moe_common::object_host_slot &slot) {
    return slot.marker.source_idx >= 0 && slot.marker.layer_idx >= 0 &&
        slot.marker.expert_idx >= 0 && slot.marker.bytes != 0 &&
        ggml_moe_common::object_key_effective_generation(slot.marker) != 0;
}

static bool ram_cache_release_common_lease_locked(size_t slot_index) {
    ggml_moe_common::object_host_slot &slot = g_vcache.ram_cache.slots[slot_index];
    if (!slot.pending_h2d) return true;
    const ggml_moe_common::object_host_completion_state state =
        ggml_moe_common::object_host_cache_synchronize_h2d(g_vcache.ram_cache, slot_index);
    if (state != ggml_moe_common::object_host_completion_state::complete &&
            state != ggml_moe_common::object_host_completion_state::idle) {
        g_vcache.ram_host_lease_failures.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    g_vcache.ram_host_lease_releases.fetch_add(1, std::memory_order_relaxed);
    return true;
}

static bool ram_cache_finish_pending_event_locked(size_t slot_index) {
    ggml_moe_common::object_host_slot &slot = g_vcache.ram_cache.slots[slot_index];
    slot.pending_d2h = false;
    slot.ready = true;
    return true;
}

static bool ram_cache_discard_identity_locked(size_t slot_index) {
    if (slot_index >= g_vcache.ram_cache.slots.size()) return false;
    ggml_moe_common::object_host_slot &slot = g_vcache.ram_cache.slots[slot_index];
    if (slot.pending_h2d) return false;
    const bool invalidated = ggml_moe_common::object_host_cache_invalidate_slot(
        g_vcache.ram_cache, slot_index);
    slot.cache_key = 0;
    slot.marker = {};
    slot.ready = false;
    slot.pending_h2d = false;
    slot.h2d_recorded = false;
    slot.pending_d2h = false;
    return invalidated;
}

static bool ram_cache_fail_pending_event_locked(size_t slot_index) {
    ggml_moe_common::object_host_slot &slot = g_vcache.ram_cache.slots[slot_index];
    slot.pending_d2h = false;
    return ram_cache_discard_identity_locked(slot_index);
}

static bool ram_cache_stream_wait_slot_ready_locked(int slot, cudaStream_t st) {
    if (slot < 0 || (size_t)slot >= g_vcache.ram_cache.slots.size()) return true;
    ggml_moe_common::object_host_slot &ram_slot = g_vcache.ram_cache.slots[(size_t)slot];
    if (ram_slot.pending_h2d) {
        const bool was_pending = ram_slot.pending_h2d;
        if (!ggml_moe_common::object_host_cache_stream_wait_h2d(
                    g_vcache.ram_cache, (size_t)slot, st)) {
            g_vcache.ram_host_lease_failures.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (was_pending && !ram_slot.pending_h2d) {
            g_vcache.ram_host_lease_releases.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (!ram_slot.pending_d2h) return true;
    cudaEvent_t ev = ram_slot.d2h_done;
    if (!ev) return ram_cache_finish_pending_event_locked((size_t)slot);
    const ggml_moe_common::object_host_transfer_status status =
        ggml_moe_common::object_host_slot_stream_wait_transfer(
            ram_slot, ggml_moe_common::object_host_transfer::d2h, st);
    if (status == ggml_moe_common::object_host_transfer_status::complete) {
        return ram_cache_finish_pending_event_locked((size_t)slot);
    }
    if (status == ggml_moe_common::object_host_transfer_status::error) {
        (void)ram_cache_fail_pending_event_locked((size_t)slot);
        return false;
    }
    return true;
}

static bool ram_cache_slot_ready_now_locked(int slot) {
    if (slot < 0 || (size_t)slot >= g_vcache.ram_cache.slots.size()) return false;
    ggml_moe_common::object_host_slot &ram_slot = g_vcache.ram_cache.slots[(size_t)slot];
    if (ram_slot.pending_h2d) {
        const ggml_moe_common::object_host_completion_state state =
            ggml_moe_common::object_host_cache_query_h2d(g_vcache.ram_cache, (size_t)slot);
        if (state == ggml_moe_common::object_host_completion_state::pending) return false;
        if (state == ggml_moe_common::object_host_completion_state::failed) {
            g_vcache.ram_host_lease_failures.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        g_vcache.ram_host_lease_releases.fetch_add(1, std::memory_order_relaxed);
    }
    if (!ram_slot.pending_d2h) return true;
    cudaEvent_t ev = ram_slot.d2h_done;
    if (!ev) return ram_cache_finish_pending_event_locked((size_t)slot);
    const ggml_moe_common::object_host_transfer_status status =
        ggml_moe_common::object_host_slot_query_transfer(
            ram_slot, ggml_moe_common::object_host_transfer::d2h);
    if (status == ggml_moe_common::object_host_transfer_status::complete) {
        return ram_cache_finish_pending_event_locked((size_t)slot);
    }
    if (status == ggml_moe_common::object_host_transfer_status::error) {
        (void)ram_cache_fail_pending_event_locked((size_t)slot);
    }
    return false;
}

static bool vram_cache_record_slot_ready_locked(int slot, cudaStream_t st) {
    if (!ggml_moe_common::role_cache_record_ready_event(
                vram_cache_role_view(), slot, st)) {
        return false;
    }
    g_vcache.slot_pending[slot] = true;
    return true;
}

static bool ram_cache_record_slot_ready_locked(int slot, cudaStream_t st) {
    if (slot < 0 || (size_t)slot >= g_vcache.ram_cache.slots.size()) return false;
    ggml_moe_common::object_host_slot &ram_slot = g_vcache.ram_cache.slots[(size_t)slot];
    return ggml_moe_common::object_host_slot_record_transfer(
        ram_slot, ggml_moe_common::object_host_transfer::d2h, st);
}

static bool ram_cache_prepare_identity_locked(
        int slot,
        uintptr_t key,
        size_t bytes) {
    if (slot < 0 || (size_t)slot >= g_vcache.ram_cache.slots.size()) return false;
    ggml_moe_common::object_host_slot &ram_slot = g_vcache.ram_cache.slots[(size_t)slot];
    if (ram_slot.pending_h2d && !ram_cache_slot_ready_now_locked(slot)) return false;
    if (!ggml_moe_common::object_host_cache_prepare_slot(
                g_vcache.ram_cache, (size_t)slot)) return false;
    ram_slot.cache_key = 0;
    ram_slot.marker = {};
    ram_slot.ready = false;
    ram_slot.pending_h2d = false;
    ram_slot.h2d_recorded = false;
    ggml_moe_common::object_key marker;
    if (ds4_ram_common_object_key(key, bytes, marker)) {
        ram_slot.marker = marker;
    }
    return true;
}

static bool ram_cache_publish_identity_locked(int slot) {
    if (slot < 0 || (size_t)slot >= g_vcache.ram_cache.slots.size()) return false;
    const ggml_moe_common::object_host_slot &ram_slot = g_vcache.ram_cache.slots[(size_t)slot];
    if (!ram_cache_slot_has_common_object(ram_slot)) {
        g_vcache.ram_non_object_stores.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    if (!ggml_moe_common::object_host_cache_publish_slot(
                g_vcache.ram_cache, (size_t)slot)) {
        return false;
    }
    g_vcache.ram_full_object_publishes.fetch_add(1, std::memory_order_relaxed);
    return true;
}

static int ram_cache_find_slot_locked(uintptr_t key) {
    if (!g_vcache.ram_cache.slab.storage.host || g_vcache.ram_cache.slots.empty()) return -1;
    ggml_moe_common::object_key marker;
    size_t common_slot = SIZE_MAX;
    if (ds4_ram_common_object_key(
            key, g_vcache.ram_cache.slab.slot_bytes, marker) &&
            ggml_moe_common::object_host_cache_find_ready_slot(
                g_vcache.ram_cache, marker, &common_slot) &&
            common_slot < g_vcache.ram_cache.slots.size()) {
        return (int)common_slot;
    }
    for (size_t slot = 0; slot < g_vcache.ram_cache.slots.size(); ++slot) {
        if (g_vcache.ram_cache.slots[slot].cache_key == key) return (int)slot;
    }
    return -1;
}

static ds4_ram_key_stat * ds4_ram_key_stat_locked(uintptr_t key) {
    auto it = g_ds4_ram_key_stats.find(key);
    return it == g_ds4_ram_key_stats.end() ? nullptr : &it->second;
}

static void ds4_ram_note_demand_locked(uintptr_t key) {
    if (!ds4_ram_score_policy_enabled()) return;
    ds4_ram_key_stat * stat = ds4_ram_key_stat_locked(key);
    if (!stat) return;
    if (stat->demands != UINT32_MAX) ++stat->demands;
    if (stat->layer >= 0 && stat->layer < 43) ++g_ds4_ram_layer_demands[stat->layer];
}

static bool ds4_ram_admission_allows_locked(uintptr_t key) {
    if (!ds4_ram_score_policy_enabled()) return true;
    ds4_ram_key_stat * stat = ds4_ram_key_stat_locked(key);
    if (!stat || stat->layer < 0 || stat->layer >= 43 || !ds4_ram_layer_allowed(stat->layer)) {
        g_vcache.ram_score_layer_rejects.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    ggml_moe_common::object_host_admission_policy policy;
    policy.min_frequency = ds4_ram_score_min_demands();
    if (ggml_moe_common::object_host_cache_evaluate_admission(
                policy, stat->demands, 0, false, 0) ==
            ggml_moe_common::object_host_admission_decision::reject_frequency) {
        g_vcache.ram_score_min_demand_rejects.fetch_add(1, std::memory_order_relaxed);
        ++g_ds4_ram_layer_rejects[stat->layer];
        return false;
    }
    return true;
}

static uint64_t ds4_ram_score_locked(uintptr_t key) {
    const ds4_ram_key_stat * stat = ds4_ram_key_stat_locked(key);
    return stat ? (uint64_t) stat->demands + 8ULL * stat->ram_hits : 0;
}

struct ds4_ram_slot_selection_context {
    int avoid_slot = -1;
    cudaStream_t stream = nullptr;
    bool require_ready_now = false;
    bool wait_each_candidate = false;
    ggml_moe_common::object_host_layer_quota_policy quota;
};

static ggml_moe_common::object_host_slot_candidate_state ds4_ram_classify_slot(
        ggml_moe_common::object_host_cache &cache,
        size_t slot_index,
        void *opaque) {
    ds4_ram_slot_selection_context &context =
        *static_cast<ds4_ram_slot_selection_context *>(opaque);
    if ((int)slot_index == context.avoid_slot) {
        return ggml_moe_common::object_host_slot_candidate_state::excluded;
    }
    ggml_moe_common::object_host_slot &slot = cache.slots[slot_index];
    if (slot.cache_key == 0) {
        return ggml_moe_common::object_host_cache_quota_allows_victim(
                    context.quota, false, false, -1, 0, 0) ?
            ggml_moe_common::object_host_slot_candidate_state::empty :
            ggml_moe_common::object_host_slot_candidate_state::excluded;
    }
    if (slot.pending_h2d && !ram_cache_slot_ready_now_locked((int)slot_index)) {
        return ggml_moe_common::object_host_slot_candidate_state::excluded;
    }
    if (context.require_ready_now) {
        if (!ram_cache_slot_ready_now_locked((int)slot_index)) {
            return ggml_moe_common::object_host_slot_candidate_state::excluded;
        }
    } else if (context.wait_each_candidate &&
            !ram_cache_stream_wait_slot_ready_locked((int)slot_index, context.stream)) {
        return ggml_moe_common::object_host_slot_candidate_state::excluded;
    }
    if (context.quota.enabled) {
        const ds4_ram_key_stat *resident = ds4_ram_key_stat_locked(slot.cache_key);
        const int resident_layer = resident ? resident->layer : -1;
        if (!ggml_moe_common::object_host_cache_quota_allows_victim(
                    context.quota, true, true, resident_layer, 0, 0)) {
            return ggml_moe_common::object_host_slot_candidate_state::excluded;
        }
    }
    return ggml_moe_common::object_host_slot_candidate_state::resident;
}

static uint64_t ds4_ram_slot_score(
        const ggml_moe_common::object_host_cache &cache,
        size_t slot_index,
        void *) {
    return ds4_ram_score_locked(cache.slots[slot_index].cache_key);
}

static int ram_cache_select_scored_slot_locked(
        uintptr_t candidate_key,
        int avoid_slot,
        cudaStream_t stream,
        bool require_ready_now) {
    ds4_ram_key_stat * candidate = ds4_ram_key_stat_locked(candidate_key);
    if (!candidate || candidate->layer < 0 || candidate->layer >= 43) return -1;
    const int quota = ds4_ram_layer_quota();
    int same_layer_count = 0;
    for (const ggml_moe_common::object_host_slot &slot : g_vcache.ram_cache.slots) {
        if (slot.cache_key == 0) continue;
        const ds4_ram_key_stat * resident = ds4_ram_key_stat_locked(slot.cache_key);
        if (resident && resident->layer == candidate->layer) ++same_layer_count;
    }

    ds4_ram_slot_selection_context context;
    context.avoid_slot = avoid_slot;
    context.stream = stream;
    context.require_ready_now = require_ready_now;
    context.wait_each_candidate = !require_ready_now;
    context.quota.enabled = quota > 0;
    context.quota.incoming_layer = candidate->layer;
    context.quota.incoming_occupancy = (size_t)same_layer_count;
    context.quota.incoming_quota = (size_t)quota;
    ggml_moe_common::object_host_slot_selection selection;
    ggml_moe_common::object_host_slot_selection_policy policy;
    policy.evaluate_score = true;
    policy.select_by_score = true;
    if (!ggml_moe_common::object_host_cache_select_slot(
                g_vcache.ram_cache,
                policy,
                ds4_ram_classify_slot,
                ds4_ram_slot_score,
                &context,
                selection)) {
        g_vcache.ram_score_layer_rejects.fetch_add(1, std::memory_order_relaxed);
        ++g_ds4_ram_layer_rejects[candidate->layer];
        return -1;
    }
    ggml_moe_common::object_host_admission_policy admission;
    admission.compare_victim = true;
    admission.allow_equal_victim = false;
    if (ggml_moe_common::object_host_cache_evaluate_admission(
                admission,
                0,
                ds4_ram_score_locked(candidate_key),
                !selection.empty,
                selection.score) == ggml_moe_common::object_host_admission_decision::reject_victim) {
        g_vcache.ram_score_victim_rejects.fetch_add(1, std::memory_order_relaxed);
        ++g_ds4_ram_layer_rejects[candidate->layer];
        return -1;
    }
    return (int)selection.slot_index;
}

static int ram_cache_select_slot_locked(int avoid_slot, cudaStream_t st) {
    if (!g_vcache.ram_cache.slab.storage.host || g_vcache.ram_cache.slots.empty()) return -1;
    ds4_ram_slot_selection_context context;
    context.avoid_slot = avoid_slot;
    ggml_moe_common::object_host_slot_selection selection;
    ggml_moe_common::object_host_slot_selection_policy policy;
    if (!ggml_moe_common::object_host_cache_select_slot(
                g_vcache.ram_cache,
                policy,
                ds4_ram_classify_slot,
                nullptr,
                &context,
                selection)) {
        return -1;
    }
    const int slot = (int)selection.slot_index;
    if (slot < 0 || !ram_cache_stream_wait_slot_ready_locked(slot, st)) return -1;
    return slot;
}

static int ram_cache_select_ready_slot_locked(int avoid_slot) {
    if (!g_vcache.ram_cache.slab.storage.host || g_vcache.ram_cache.slots.empty()) return -1;
    ds4_ram_slot_selection_context context;
    context.avoid_slot = avoid_slot;
    context.require_ready_now = true;
    ggml_moe_common::object_host_slot_selection selection;
    ggml_moe_common::object_host_slot_selection_policy policy;
    return ggml_moe_common::object_host_cache_select_slot(
            g_vcache.ram_cache,
            policy,
            ds4_ram_classify_slot,
            nullptr,
            &context,
            selection) ? (int)selection.slot_index : -1;
}

static void ds4_ram_prefetch_note_ram_eviction(uintptr_t key) {
    if (key == 0) return;
    std::lock_guard<std::mutex> lock(g_ds4_ram_prefetch.mu);
    if (g_ds4_ram_prefetch.ready_keys.erase(key) != 0) {
        g_ds4_ram_prefetch.wasted.fetch_add(1, std::memory_order_relaxed);
    }
}

static bool ram_cache_store_host_locked(uintptr_t key, const void *host_data, size_t sz, int avoid_slot) {
    g_vcache.ram_host_store_attempts.fetch_add(1, std::memory_order_relaxed);
    if (!g_vcache.ram_cache.slab.storage.host || g_vcache.ram_cache.slots.empty() ||
            !host_data || key == 0 || sz > g_vcache.ram_cache.slab.slot_bytes) return false;
    g_vcache.ram_host_store_valid.fetch_add(1, std::memory_order_relaxed);
    if (!ds4_ram_admission_allows_locked(key)) return false;
    g_vcache.ram_host_store_admitted.fetch_add(1, std::memory_order_relaxed);
    int slot = ram_cache_find_slot_locked(key);
    if (slot >= 0 && slot == avoid_slot) return true;
    if (slot >= 0) { if (!ram_cache_slot_ready_now_locked(slot)) return false; }
    else {
        slot = ds4_ram_score_policy_enabled() ?
            ram_cache_select_scored_slot_locked(key, avoid_slot, nullptr, true) :
            ram_cache_select_ready_slot_locked(avoid_slot);
        if (slot < 0) return false;
    }
    g_vcache.ram_host_store_slot_selected.fetch_add(1, std::memory_order_relaxed);
    ggml_moe_common::object_host_slot &ram_slot = g_vcache.ram_cache.slots[(size_t)slot];
    const bool replacing = ram_slot.cache_key != 0 && ram_slot.cache_key != key;
    if (replacing) ds4_ram_prefetch_note_ram_eviction(ram_slot.cache_key);
    ggml_moe_common::object_host_staging_transaction staging;
    ggml_moe_common::object_key marker;
    const bool common_object_staging_requested = ds4_common_object_staging_enabled();
    if (common_object_staging_requested) {
        g_vcache.ram_host_store_staging_requested.fetch_add(1, std::memory_order_relaxed);
    }
    const bool common_object_staging = common_object_staging_requested &&
        ds4_ram_common_object_key(key, sz, marker);
    if (common_object_staging_requested && !common_object_staging) {
        static std::atomic<bool> reported{false};
        if (!reported.exchange(true, std::memory_order_relaxed)) {
            const auto stat = g_ds4_ram_key_stats.find(key);
            std::fprintf(stderr,
                    "[moe_stream] RAM common staging: object key resolution failed "
                    "key=%p bytes=%zu slot_bytes=%zu key_known=%d\n",
                    (void *) key,
                    sz,
                    g_vcache.ram_cache.slab.slot_bytes,
                    stat != g_ds4_ram_key_stats.end() ? 1 : 0);
        }
    }
    if (common_object_staging) {
        g_vcache.ram_host_store_common_eligible.fetch_add(1, std::memory_order_relaxed);
        if (!ggml_moe_common::object_host_cache_begin_staging(
                    g_vcache.ram_cache, (size_t)slot, marker, staging)) {
            return false;
        }
    } else if (!ram_cache_prepare_identity_locked(slot, key, sz)) {
        return false;
    }
    void *dst = ram_slot.host;
    if (!dst) {
        if (common_object_staging) {
            (void)ggml_moe_common::object_host_cache_abort_staging(
                g_vcache.ram_cache, staging);
        }
        return false;
    }
    std::memcpy(dst, host_data, sz);
    ram_slot.cache_key = key;
    ram_slot.used = g_vcache.clock.fetch_add(1, std::memory_order_relaxed);
    ram_slot.pending_d2h = false;
    bool published = false;
    if (common_object_staging) {
        published = ggml_moe_common::object_host_cache_publish_staging(
            g_vcache.ram_cache, staging, 0);
        if (published) {
            g_vcache.ram_full_object_publishes.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        ram_slot.ready = true;
        published = ram_cache_publish_identity_locked(slot);
    }
    if (!published) {
        if (common_object_staging && staging.active) {
            (void)ggml_moe_common::object_host_cache_abort_staging(
                g_vcache.ram_cache, staging);
        }
        (void)ram_cache_discard_identity_locked((size_t)slot);
        return false;
    }
    g_vcache.ram_inserts.fetch_add(1, std::memory_order_relaxed);
    if (ds4_ram_score_policy_enabled()) {
        ds4_ram_key_stat * stat = ds4_ram_key_stat_locked(key);
        if (stat && stat->layer >= 0 && stat->layer < 43) ++g_ds4_ram_layer_admissions[stat->layer];
        g_vcache.ram_score_admissions.fetch_add(1, std::memory_order_relaxed);
    }
    if (replacing) g_vcache.ram_evictions.fetch_add(1, std::memory_order_relaxed);
    return true;
}

static bool ram_cache_store_victim_locked(uintptr_t key, const void *dev_data, size_t sz, cudaStream_t st, int avoid_slot) {
    if (!g_vcache.ram_cache.slab.storage.host || g_vcache.ram_cache.slots.empty() ||
            !dev_data || key == 0 || sz > g_vcache.ram_cache.slab.slot_bytes) return false;
    if (!ds4_ram_admission_allows_locked(key)) return false;
    int slot = ram_cache_find_slot_locked(key);
    if (slot >= 0 && slot != avoid_slot) {
        g_vcache.ram_cache.slots[(size_t)slot].used =
            g_vcache.clock.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    if (slot == avoid_slot) slot = -1;
    if (slot >= 0) { if (!ram_cache_stream_wait_slot_ready_locked(slot, st)) return false; }
    else {
        slot = ds4_ram_score_policy_enabled() ?
            ram_cache_select_scored_slot_locked(key, avoid_slot, st, false) :
            ram_cache_select_slot_locked(avoid_slot, st);
        if (slot < 0) return false;
    }
    ggml_moe_common::object_host_slot &ram_slot = g_vcache.ram_cache.slots[(size_t)slot];
    const bool replacing = ram_slot.cache_key != 0 && ram_slot.cache_key != key;
    if (replacing) ds4_ram_prefetch_note_ram_eviction(ram_slot.cache_key);
    if (!ram_cache_prepare_identity_locked(slot, key, sz)) return false;
    void *dst = ram_slot.host;
    if (!dst) return false;
    if (cudaMemcpyAsync(dst, dev_data, sz, cudaMemcpyDeviceToHost, st) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    ram_slot.cache_key = key;
    ram_slot.used = g_vcache.clock.fetch_add(1, std::memory_order_relaxed);
    ram_slot.ready = false;
    g_vcache.ram_inserts.fetch_add(1, std::memory_order_relaxed);
    if (ds4_ram_score_policy_enabled()) {
        ds4_ram_key_stat * stat = ds4_ram_key_stat_locked(key);
        if (stat && stat->layer >= 0 && stat->layer < 43) ++g_ds4_ram_layer_admissions[stat->layer];
        g_vcache.ram_score_admissions.fetch_add(1, std::memory_order_relaxed);
    }
    if (replacing) g_vcache.ram_evictions.fetch_add(1, std::memory_order_relaxed);
    if (!ram_cache_record_slot_ready_locked(slot, st)) {
        if (cudaStreamSynchronize(st) != cudaSuccess) {
            cudaGetLastError();
            ram_slot.cache_key = 0;
            ram_slot.marker = {};
            return false;
        }
        ram_slot.ready = true;
    }
    if (!ram_cache_publish_identity_locked(slot)) {
        if (ram_slot.pending_d2h && ram_slot.d2h_done) {
            (void)cudaEventSynchronize(ram_slot.d2h_done);
            ram_slot.pending_d2h = false;
        }
        (void)ram_cache_discard_identity_locked((size_t)slot);
        return false;
    }
    return true;
}

static void *vram_cache_insert_locked(uintptr_t key, const void *host_data, size_t sz, cudaStream_t st, bool count_miss, const char * tensor, int ram_src_slot) {
    if (!g_vcache.pool || g_vcache.n_slots == 0 || !host_data || sz > g_vcache.slot_sz) return nullptr;
    int range_begin = 0, range_end = g_vcache.n_slots;
    if (moe_stream_cache_partition_enabled()) {
        int tail_slots = moe_stream_cache_partition_tail_slots();
        if (tail_slots < 0) tail_slots = 0;
        if (tail_slots > g_vcache.n_slots) tail_slots = g_vcache.n_slots;
        const int tail_begin = g_vcache.n_slots - tail_slots;
        if (moe_stream_cache_partition_tail_matches(tensor)) { range_begin = tail_begin; range_end = g_vcache.n_slots; }
        else { range_begin = 0; range_end = tail_begin; }
        if (range_end <= range_begin) return nullptr;
    }
    const bool tensor_lru = vram_cache_tensor_lru_enabled() && tensor && tensor[0];
    const uint64_t tensor_tag = tensor_lru ? vram_cache_tensor_tag(tensor) : 0;
    int slot = -1;
    int same_tensor_slot = -1;
    int global_slot = -1;
    uint64_t same_tensor_oldest = UINT64_MAX;
    uint64_t global_oldest = UINT64_MAX;
    uint32_t same_tensor_lowest_score = UINT32_MAX;
    uint32_t global_lowest_score = UINT32_MAX;
    const bool score_victim = vram_cache_score_victim_enabled();
    for (int i = range_begin; i < range_end; ++i) {
        if (g_vcache.slot_lease_count[i] != 0 || g_vcache.slot_filling[i]) continue;
        if (g_vcache.slot_key[i] == 0) { slot = i; break; }
        const uint32_t score = score_victim ? vram_cache_demand_score_locked(g_vcache.slot_key[i]) : 0;
        if ((!score_victim && g_vcache.slot_used[i] < global_oldest) ||
                (score_victim && (score < global_lowest_score ||
                    (score == global_lowest_score && g_vcache.slot_used[i] < global_oldest)))) {
            global_lowest_score = score;
            global_oldest = g_vcache.slot_used[i];
            global_slot = i;
        }
        if (tensor_lru && g_vcache.slot_tensor_tag[i] == tensor_tag &&
                ((!score_victim && g_vcache.slot_used[i] < same_tensor_oldest) ||
                 (score_victim && (score < same_tensor_lowest_score ||
                    (score == same_tensor_lowest_score && g_vcache.slot_used[i] < same_tensor_oldest))))) {
            same_tensor_lowest_score = score;
            same_tensor_oldest = g_vcache.slot_used[i];
            same_tensor_slot = i;
        }
    }
    if (slot < 0) {
        if (g_vcache.lru_frozen) {
            return nullptr;
        }
        slot = (tensor_lru && same_tensor_slot >= 0) ? same_tensor_slot : global_slot;
    }
    if (slot < 0 ||
            !vram_cache_stream_wait_slot_ready_locked(slot, st) ||
            !vram_cache_stream_wait_slot_lease_locked(slot, st)) return nullptr;
    void *dst = (char *)g_vcache.pool + (size_t)slot * g_vcache.slot_sz;
    const uintptr_t evicted_key = g_vcache.slot_key[slot];
    const bool fills_empty_slot = evicted_key == 0;
    if (score_victim && evicted_key != 0 && evicted_key != key) {
        g_vcache.score_victim_selections.fetch_add(1, std::memory_order_relaxed);
        if (vram_cache_demand_score_locked(key) <= vram_cache_demand_score_locked(evicted_key)) {
            g_vcache.score_candidate_lte_victim.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (evicted_key != 0 && evicted_key != key) ram_cache_store_victim_locked(evicted_key, dst, sz, st, ram_src_slot);
    if (cudaMemcpyAsync(dst, host_data, sz, cudaMemcpyHostToDevice, st) != cudaSuccess) { cudaGetLastError(); return nullptr; }
    if (ram_src_slot < 0) ram_cache_store_host_locked(key, host_data, sz, -1);
    if (vram_cache_hash_lookup_enabled() && evicted_key != 0 && evicted_key != key) {
        vram_cache_hash_erase_locked(evicted_key, slot);
    }
    if (fills_empty_slot) ++g_vcache.used_slots;
    g_vcache.slot_key[slot] = key;
    if (vram_cache_hash_lookup_enabled() && !vram_cache_hash_insert_locked(key, slot)) {
        std::fprintf(stderr, "[moe_stream] VRAM cache hash insertion failed for key=%p\n", (void *) key);
    }
    g_vcache.slot_tensor_tag[slot] = tensor_lru ? tensor_tag : 0;
    g_vcache.slot_used[slot] = g_vcache.clock.fetch_add(1, std::memory_order_relaxed);
    if (++g_vcache.slot_generation[slot] == 0) ++g_vcache.slot_generation[slot];
    vram_cache_record_slot_ready_locked(slot, st);
    if (ram_src_slot >= 0) {
        ggml_moe_common::object_host_slot &ram_slot =
            g_vcache.ram_cache.slots[(size_t)ram_src_slot];
        if (ram_cache_slot_has_common_object(ram_slot)) {
            const bool was_pending = ram_slot.pending_h2d;
            if (!ggml_moe_common::object_host_cache_record_h2d(
                        g_vcache.ram_cache, (size_t)ram_src_slot, st)) {
                if (was_pending && !ram_slot.pending_h2d) {
                    g_vcache.ram_host_lease_releases.fetch_add(1, std::memory_order_relaxed);
                }
                g_vcache.ram_host_lease_failures.fetch_add(1, std::memory_order_relaxed);
                if (ram_slot.pending_h2d) return nullptr;
            }
        } else if (!ram_cache_record_slot_ready_locked(ram_src_slot, st)) {
            if (cudaStreamSynchronize(st) != cudaSuccess) {
                cudaGetLastError();
                return nullptr;
            }
            ram_slot.pending_d2h = false;
        }
    }
    if (count_miss) g_vcache.misses.fetch_add(1, std::memory_order_relaxed);
    return dst;
}

static void *vram_cache_lookup(uintptr_t key, cudaStream_t st) {
    if (!g_vcache.pool || g_vcache.n_slots == 0) return nullptr;
    std::lock_guard<std::mutex> lk(g_vcache.mu);
    vram_cache_note_demand_locked(key);
    ds4_ram_note_demand_locked(key);
    int slot = -1;
    if (vram_cache_hash_lookup_enabled()) {
        slot = vram_cache_hash_find_locked(key);
        if (slot >= g_vcache.n_slots || (slot >= 0 && g_vcache.slot_key[slot] != key)) {
            vram_cache_hash_erase_locked(key, slot);
            slot = -1;
        }
    } else {
        for (int candidate = 0; candidate < g_vcache.n_slots; ++candidate) {
            if (g_vcache.slot_key[candidate] == key) {
                slot = candidate;
                break;
            }
        }
    }
    if (slot < 0 || g_vcache.slot_filling[slot]) return nullptr;
    if (!vram_cache_stream_wait_slot_ready_locked(slot, st)) return nullptr;
    g_vcache.slot_used[slot] = g_vcache.clock.fetch_add(1, std::memory_order_relaxed);
    g_vcache.hits.fetch_add(1, std::memory_order_relaxed);
    return (char *)g_vcache.pool + (size_t)slot * g_vcache.slot_sz;
}

static bool vram_cache_contains_key(uintptr_t key) {
    if (!g_vcache.pool || g_vcache.n_slots == 0) return false;
    std::lock_guard<std::mutex> lk(g_vcache.mu);
    if (vram_cache_hash_lookup_enabled()) {
        const int slot = vram_cache_hash_find_locked(key);
        if (slot >= 0 && slot < g_vcache.n_slots && g_vcache.slot_key[slot] == key) return true;
        if (slot >= 0) vram_cache_hash_erase_locked(key, slot);
        return false;
    }
    for (int slot = 0; slot < g_vcache.n_slots; ++slot) if (g_vcache.slot_key[slot] == key) return true;
    return false;
}

static void *vram_cache_promote_from_ram(uintptr_t key, size_t sz, cudaStream_t st, const char * tensor) {
    if (!g_vcache.ram_cache.slab.storage.host || g_vcache.ram_cache.slots.empty() ||
            sz > g_vcache.slot_sz) return nullptr;
    std::lock_guard<std::mutex> lk(g_vcache.mu);
    int ram_slot = ram_cache_find_slot_locked(key);
    if (ram_slot < 0) { g_vcache.ram_misses.fetch_add(1, std::memory_order_relaxed); return nullptr; }
    if (g_vcache.ram_cache.slots[(size_t)ram_slot].pending_h2d &&
            !ram_cache_slot_ready_now_locked(ram_slot)) {
        g_vcache.ram_misses.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    if (!ram_cache_stream_wait_slot_ready_locked(ram_slot, st)) return nullptr;
    ggml_moe_common::object_host_slot &host_slot = g_vcache.ram_cache.slots[(size_t)ram_slot];
    host_slot.used = g_vcache.clock.fetch_add(1, std::memory_order_relaxed);
    g_vcache.ram_hits.fetch_add(1, std::memory_order_relaxed);
    if (ds4_ram_score_policy_enabled()) {
        ds4_ram_key_stat * stat = ds4_ram_key_stat_locked(key);
        if (stat) {
            if (stat->ram_hits != UINT32_MAX) ++stat->ram_hits;
            if (stat->layer >= 0 && stat->layer < 43) ++g_ds4_ram_layer_hits[stat->layer];
        }
    }
    {
        std::lock_guard<std::mutex> prefetch_lock(g_ds4_ram_prefetch.mu);
        if (g_ds4_ram_prefetch.ready_keys.erase(key) != 0) {
            g_ds4_ram_prefetch.used.fetch_add(1, std::memory_order_relaxed);
        }
    }
    const void *src = host_slot.host;
    if (!src) return nullptr;
    const bool common_object = ram_cache_slot_has_common_object(host_slot);
    if (common_object) {
        if (!ggml_moe_common::object_host_cache_begin_h2d(
                    g_vcache.ram_cache, (size_t)ram_slot)) {
            g_vcache.ram_host_lease_failures.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        g_vcache.ram_host_lease_acquires.fetch_add(1, std::memory_order_relaxed);
    }
    void *result = vram_cache_insert_locked(key, src, sz, st, true, tensor, ram_slot);
    if (!result && common_object && host_slot.pending_h2d) {
        if (ggml_moe_common::object_host_cache_cancel_h2d(
                    g_vcache.ram_cache, (size_t)ram_slot, st)) {
            g_vcache.ram_host_lease_releases.fetch_add(1, std::memory_order_relaxed);
        } else {
            g_vcache.ram_host_lease_failures.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return result;
}

static void *vram_cache_insert_impl(uintptr_t key, const void *host_data, size_t sz, cudaStream_t st, bool count_miss, const char * tensor) {
    std::lock_guard<std::mutex> lk(g_vcache.mu);
    return vram_cache_insert_locked(key, host_data, sz, st, count_miss, tensor, -1);
}

static void *vram_cache_insert(uintptr_t key, const void *host_data, size_t sz, cudaStream_t st, const char * tensor) {
    return vram_cache_insert_impl(key, host_data, sz, st, true, tensor);
}

static bool vram_cache_acquire_descriptor(
        const void * object_ptr,
        uintptr_t object_generation,
        int role,
        size_t role_offset,
        cudaStream_t st,
        ggml_moe_expert_descriptor & descriptor) {
    descriptor = {};
    descriptor.slot_id = -1;
    descriptor.role = role;
    if (!object_ptr || !g_vcache.pool || g_vcache.slot_sz == 0 ||
            role < 0 || role >= static_cast<int>(ggml_moe_common::role::count)) {
        return false;
    }

    std::lock_guard<std::mutex> lk(g_vcache.mu);
    const uintptr_t pool_begin = (uintptr_t) g_vcache.pool;
    const uintptr_t pool_end = pool_begin + g_vcache.pool_sz;
    const uintptr_t address = (uintptr_t) object_ptr;
    if (address < pool_begin || address >= pool_end) return false;
    const size_t delta = (size_t) (address - pool_begin);
    if (delta % g_vcache.slot_sz != 0) return false;
    const int slot = (int) (delta / g_vcache.slot_sz);
    if (slot < 0 || slot >= g_vcache.n_slots ||
            g_vcache.slot_key[slot] != object_generation) return false;
    if (!ggml_moe_common::role_cache_acquire_descriptor_for_stream(
            vram_cache_role_view(), slot, object_generation, (uint64_t) object_generation,
            static_cast<ggml_moe_common::role>(role), role_offset, st, descriptor)) {
        return false;
    }
    if (g_vcache.slot_layout[slot] == 1) {
        descriptor.role |= GGML_MOE_EXPERT_LAYOUT_PLANAR_MXFP4;
    } else if (g_vcache.slot_layout[slot] == 2) {
        descriptor.role |= GGML_MOE_EXPERT_LAYOUT_SCALE3_MXFP4;
    } else if (g_vcache.slot_layout[slot] == 3) {
        descriptor.role |= GGML_MOE_EXPERT_LAYOUT_PLANAR_SCALE3_MXFP4;
    }
    g_vcache.descriptor_acquires.fetch_add(1, std::memory_order_relaxed);
    return true;
}

static void vram_cache_cancel_descriptor(const ggml_moe_expert_descriptor & descriptor) {
    if (descriptor.slot_id < 0) return;
    std::lock_guard<std::mutex> lk(g_vcache.mu);
    if (ggml_moe_common::role_cache_release_descriptor(
            vram_cache_role_view(), (uintptr_t) descriptor.object_generation, descriptor)) {
        g_vcache.descriptor_releases.fetch_add(1, std::memory_order_relaxed);
        g_vcache.descriptor_cancels.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_vcache.descriptor_stale_releases.fetch_add(1, std::memory_order_relaxed);
    }
}

struct ds4_vram_reservation {
    void * dst = nullptr;
    uintptr_t key = 0;
    uint64_t slot_generation = 0;
    int slot = -1;
};

static bool vram_cache_reserve_triplet(
        uintptr_t key,
        size_t sz,
        cudaStream_t transfer_stream,
        ds4_vram_reservation & reservation,
        const char * tensor = "ds4_triplet",
        bool demote_victim = true) {
    reservation = {};
    reservation.slot = -1;
    if (!g_vcache.pool || g_vcache.n_slots <= 0 || sz > g_vcache.slot_sz || key == 0) return false;

    std::lock_guard<std::mutex> lk(g_vcache.mu);
    int range_begin = 0;
    int range_end = g_vcache.n_slots;
    if (moe_stream_cache_partition_enabled()) {
        int tail_slots = moe_stream_cache_partition_tail_slots();
        if (tail_slots < 0) tail_slots = 0;
        if (tail_slots > g_vcache.n_slots) tail_slots = g_vcache.n_slots;
        const int tail_begin = g_vcache.n_slots - tail_slots;
        if (moe_stream_cache_partition_tail_matches(tensor)) {
            range_begin = tail_begin;
        } else {
            range_end = tail_begin;
        }
        if (range_end <= range_begin) return false;
    }
    const bool tensor_lru = vram_cache_tensor_lru_enabled() && tensor && tensor[0];
    const uint64_t tensor_tag = tensor_lru ? vram_cache_tensor_tag(tensor) : 0;
    int slot = -1;
    int same_tensor_slot = -1;
    int global_slot = -1;
    uint64_t same_tensor_oldest = UINT64_MAX;
    uint64_t global_oldest = UINT64_MAX;
    uint32_t same_tensor_lowest_score = UINT32_MAX;
    uint32_t global_lowest_score = UINT32_MAX;
    const bool score_victim = vram_cache_score_victim_enabled();
    for (int i = range_begin; i < range_end; ++i) {
        if (g_vcache.slot_lease_count[i] != 0 || g_vcache.slot_filling[i]) continue;
        if (g_vcache.slot_key[i] == 0) {
            slot = i;
            break;
        }
        const uint32_t score = score_victim ? vram_cache_demand_score_locked(g_vcache.slot_key[i]) : 0;
        if ((!score_victim && g_vcache.slot_used[i] < global_oldest) ||
                (score_victim && (score < global_lowest_score ||
                    (score == global_lowest_score && g_vcache.slot_used[i] < global_oldest)))) {
            global_lowest_score = score;
            global_oldest = g_vcache.slot_used[i];
            global_slot = i;
        }
        if (tensor_lru && g_vcache.slot_tensor_tag[i] == tensor_tag &&
                ((!score_victim && g_vcache.slot_used[i] < same_tensor_oldest) ||
                 (score_victim && (score < same_tensor_lowest_score ||
                    (score == same_tensor_lowest_score && g_vcache.slot_used[i] < same_tensor_oldest))))) {
            same_tensor_lowest_score = score;
            same_tensor_oldest = g_vcache.slot_used[i];
            same_tensor_slot = i;
        }
    }
    if (slot < 0) {
        if (g_vcache.lru_frozen) return false;
        slot = tensor_lru && same_tensor_slot >= 0 ? same_tensor_slot : global_slot;
    }
    if (slot < 0 ||
            !vram_cache_stream_wait_slot_ready_locked(slot, transfer_stream) ||
            !vram_cache_stream_wait_slot_lease_locked(slot, transfer_stream)) return false;

    void * dst = (char *) g_vcache.pool + (size_t) slot * g_vcache.slot_sz;
    const uintptr_t evicted_key = g_vcache.slot_key[slot];
    const bool empty = evicted_key == 0;
    if (score_victim && evicted_key != 0 && evicted_key != key) {
        g_vcache.score_victim_selections.fetch_add(1, std::memory_order_relaxed);
        if (vram_cache_demand_score_locked(key) <= vram_cache_demand_score_locked(evicted_key)) {
            g_vcache.score_candidate_lte_victim.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (demote_victim && evicted_key != 0 && evicted_key != key &&
            g_vcache.ram_cache.slab.storage.host) {
        (void) ram_cache_store_victim_locked(evicted_key, dst, sz, transfer_stream, -1);
    }
    if (vram_cache_hash_lookup_enabled() && evicted_key != 0 && evicted_key != key) {
        vram_cache_hash_erase_locked(evicted_key, slot);
    }
    if (empty) ++g_vcache.used_slots;
    g_vcache.slot_key[slot] = key;
    if (vram_cache_hash_lookup_enabled() && !vram_cache_hash_insert_locked(key, slot)) {
        g_vcache.slot_key[slot] = evicted_key;
        if (empty) --g_vcache.used_slots;
        if (evicted_key != 0) (void) vram_cache_hash_insert_locked(evicted_key, slot);
        return false;
    }
    if (++g_vcache.slot_generation[slot] == 0) ++g_vcache.slot_generation[slot];
    if (!ggml_moe_common::role_cache_prepare_fill(
                vram_cache_readiness_view(), slot, key,
                g_vcache.slot_generation[slot], sz)) {
        if (vram_cache_hash_lookup_enabled()) {
            vram_cache_hash_erase_locked(key, slot);
            if (evicted_key != 0) (void) vram_cache_hash_insert_locked(evicted_key, slot);
        }
        g_vcache.slot_key[slot] = evicted_key;
        if (empty) --g_vcache.used_slots;
        return false;
    }
    g_vcache.slot_tensor_tag[slot] = tensor_tag;
    g_vcache.slot_layout[slot] = 0;
    g_vcache.slot_used[slot] = g_vcache.clock.fetch_add(1, std::memory_order_relaxed);
    ++g_vcache.slot_lease_count[slot];
    g_vcache.misses.fetch_add(1, std::memory_order_relaxed);
    reservation.dst = dst;
    reservation.key = key;
    reservation.slot_generation = g_vcache.slot_generation[slot];
    reservation.slot = slot;
    return true;
}

static bool vram_cache_set_reservation_layout(
        const ds4_vram_reservation &reservation,
        uint8_t layout) {
    std::lock_guard<std::mutex> lk(g_vcache.mu);
    const int slot = reservation.slot;
    if (slot < 0 || slot >= g_vcache.n_slots ||
            g_vcache.slot_key[slot] != reservation.key ||
            g_vcache.slot_generation[slot] != reservation.slot_generation ||
            g_vcache.slot_lease_count[slot] == 0) {
        return false;
    }
    g_vcache.slot_layout[slot] = layout;
    return true;
}

static bool vram_cache_publish_triplet(
        const ds4_vram_reservation & reservation,
        cudaStream_t transfer_stream) {
    std::lock_guard<std::mutex> lk(g_vcache.mu);
    const int slot = reservation.slot;
    if (slot < 0 || slot >= g_vcache.n_slots ||
            g_vcache.slot_key[slot] != reservation.key ||
            g_vcache.slot_generation[slot] != reservation.slot_generation ||
            g_vcache.slot_lease_count[slot] == 0) return false;
    const ggml_moe_common::role_cache_view readiness = vram_cache_readiness_view();
    if (!ggml_moe_common::role_cache_mark_reading(
                readiness, slot, reservation.key, reservation.slot_generation,
                g_vcache.slot_payload_bytes[slot]) ||
            !ggml_moe_common::role_cache_publish_ready(
                readiness, slot, reservation.key, reservation.slot_generation,
                g_vcache.slot_payload_bytes[slot], transfer_stream)) {
        return false;
    }
    --g_vcache.slot_lease_count[slot];
    return true;
}

static bool vram_cache_copy_triplet_from_host(
        const ds4_vram_reservation &reservation,
        const void *host_src,
        size_t bytes,
        cudaStream_t transfer_stream) {
    if (!vram_cache_common_h2d_enabled()) {
        return cudaMemcpyAsync(
                    reservation.dst, host_src, bytes,
                    cudaMemcpyHostToDevice, transfer_stream) == cudaSuccess &&
            vram_cache_publish_triplet(reservation, transfer_stream);
    }
    std::lock_guard<std::mutex> lk(g_vcache.mu);
    const int slot = reservation.slot;
    if (slot < 0 || slot >= g_vcache.n_slots ||
            g_vcache.slot_key[slot] != reservation.key ||
            g_vcache.slot_generation[slot] != reservation.slot_generation ||
            g_vcache.slot_lease_count[slot] == 0 ||
            g_vcache.slot_payload_bytes[slot] != bytes) {
        return false;
    }
    const ggml_moe_common::role_cache_h2d_copy copy{
        vram_cache_readiness_view(),
        slot,
        reservation.key,
        reservation.slot_generation,
        bytes,
        reservation.dst,
        host_src,
        bytes,
    };
    if (!ggml_moe_common::role_cache_copy_h2d_batch_async(
                &copy, 1, transfer_stream, nullptr)) {
        return false;
    }
    --g_vcache.slot_lease_count[slot];
    return true;
}

static bool vram_cache_copy_dependency_group_from_host(
        const ggml_moe_common::dependency_role_copy * copies,
        uint8_t copy_count,
        ds4_vram_reservation (&reservations)[3],
        const void * host_object,
        cudaStream_t transfer_stream,
        uint64_t & copied_bytes) {
    copied_bytes = 0;
    if ((!copies && copy_count != 0) || !host_object || !transfer_stream || copy_count > 3) {
        return false;
    }
    for (uint8_t copy_index = 0; copy_index < copy_count; ++copy_index) {
        const ggml_moe_common::dependency_role_copy & copy = copies[copy_index];
        const int role_index = static_cast<int>(copy.kind);
        if (role_index < 0 || role_index >= 3 || reservations[role_index].slot < 0 ||
                !vram_cache_copy_triplet_from_host(
                    reservations[role_index],
                    static_cast<const char *>(host_object) + copy.host_offset,
                    copy.bytes,
                    transfer_stream)) {
            return false;
        }
        copied_bytes += copy.bytes;
    }
    return true;
}

static void vram_cache_fail_triplet(const ds4_vram_reservation & reservation) {
    std::lock_guard<std::mutex> lk(g_vcache.mu);
    const int slot = reservation.slot;
    if (slot < 0 || slot >= g_vcache.n_slots ||
            g_vcache.slot_key[slot] != reservation.key ||
            g_vcache.slot_generation[slot] != reservation.slot_generation) return;
    const ggml_moe_common::role_cache_view readiness = vram_cache_readiness_view();
    (void) ggml_moe_common::role_cache_mark_failed(
        readiness, slot, reservation.key, reservation.slot_generation);
    if (g_vcache.slot_lease_count[slot] > 0) --g_vcache.slot_lease_count[slot];
    (void) ggml_moe_common::role_cache_clear_readiness(readiness, slot);
    if (vram_cache_hash_lookup_enabled()) {
        vram_cache_hash_erase_locked(reservation.key, slot);
    }
    g_vcache.slot_key[slot] = 0;
    if (g_vcache.used_slots > 0) --g_vcache.used_slots;
}

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

static uintptr_t ds4_triplet_role_key(const ds4_triplet_pack_entry & entry, int role) {
    return (uintptr_t) (entry.generation * 4 + (uint64_t) role + 1);
}

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
static size_t ds4_lossless_min_batch();

static bool ds4_common_async_pipeline_requested() {
    const char * env = std::getenv("GGML_MOE_COMMON_ASYNC_PIPELINE");
    return !env || !env[0] || env[0] != '0';
}

static bool ds4_common_async_critical_return_requested() {
    const char * env = std::getenv("GGML_MOE_COMMON_ASYNC_CRITICAL_RETURN");
    return !env || !env[0] || env[0] != '0';
}

static bool ds4_legacy_rolling_refill_requested() {
    const char * env = std::getenv("GGML_MOE_COMMON_ROLLING_REFILL");
    return env && env[0] && env[0] != '0';
}

static const ggml_moe_common::object_pipeline_mode_config & ds4_pipeline_mode_config() {
    static const ggml_moe_common::object_pipeline_mode_config config =
        ggml_moe_common::object_pipeline_resolve_mode(
            ggml_moe_common::object_pipeline_mode::rolling_refill,
            ds4_legacy_rolling_refill_requested());
    return config;
}

static const ggml_moe_common::object_pipeline_policy_config & ds4_pipeline_policy_config() {
    static const ggml_moe_common::object_pipeline_policy_config config = [] {
        const char * depth_env = std::getenv("GGML_MOE_IO_DEPTH");
        unsigned legacy_depth = depth_env && depth_env[0] ?
            (unsigned) std::strtoul(depth_env, nullptr, 10) : 16U;
        legacy_depth = std::max(8U, std::min(legacy_depth, 64U));

        ggml_moe_common::object_pipeline_policy fallback;
        fallback.io_depth = legacy_depth;
        fallback.max_inflight_h2d = 8;
        const ggml_moe_common::object_pipeline_mode_config & mode =
            ds4_pipeline_mode_config();
        switch (mode.mode) {
            case ggml_moe_common::object_pipeline_mode::sync_qd1:
                fallback.io_depth = 1;
                fallback.initial_batch = 1;
                fallback.refill_batch = 1;
                fallback.min_flush_batch = 1;
                fallback.flush_timeout_us = 1;
                fallback.max_inflight_h2d = 1;
                break;
            case ggml_moe_common::object_pipeline_mode::persistent_demand:
                fallback.initial_batch = std::min<size_t>(8, fallback.io_depth);
                fallback.refill_batch = fallback.initial_batch;
                fallback.min_flush_batch = 1;
                fallback.flush_timeout_us = 1;
                break;
            case ggml_moe_common::object_pipeline_mode::rolling_refill:
                fallback.initial_batch = std::min<size_t>(4, fallback.io_depth);
                fallback.refill_batch = std::min<size_t>(2, fallback.io_depth);
                fallback.min_flush_batch = 1;
                fallback.flush_timeout_us = 1;
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

static bool ds4_common_rolling_refill_requested() {
    const ggml_moe_common::object_pipeline_mode_config & mode =
        ds4_pipeline_mode_config();
    const ggml_moe_common::object_pipeline_policy_config & policy =
        ds4_pipeline_policy_config();
    return mode.valid && policy.valid &&
        (mode.mode == ggml_moe_common::object_pipeline_mode::rolling_refill ||
         mode.source != ggml_moe_common::object_pipeline_config_source::model_default ||
         policy.override_mask != 0);
}

static void ds4_triplet_pipe_batch_reset(ds4_triplet_pipe_batch & batch) {
    std::fill(batch.entries, batch.entries + 8, nullptr);
    std::fill(batch.host, batch.host + 8, nullptr);
    std::fill(batch.reservations, batch.reservations + 8, ds4_vram_reservation{});
    std::fill(batch.transfer_plans, batch.transfer_plans + 8,
        ggml_moe_common::dependency_transfer_plan{});
    for (int object_index = 0; object_index < 8; ++object_index) {
        for (int role = 0; role < 3; ++role) {
            batch.role_reservations[object_index][role] = {};
            batch.role_reservations[object_index][role].slot = -1;
            batch.role_ptrs[object_index][role] = nullptr;
        }
        batch.host_cache_hits[object_index] = false;
        batch.lossless_objects[object_index] = false;
        batch.lossless_planar[object_index] = false;
        batch.lossless_scale3[object_index] = false;
        batch.lossless_planar_scale3[object_index] = false;
        batch.lossless_direct_quant[object_index] = false;
        batch.lossless_crc_required[object_index] = false;
        batch.lossless_crc_role_masks[object_index] = 0;
        batch.lossless_object_indices[object_index] = SIZE_MAX;
        batch.lossless_crc_slots[object_index] = SIZE_MAX;
        for (int role = 0; role < 3; ++role) {
            batch.lossless_crc_role_slots[object_index][role] = SIZE_MAX;
        }
        batch.host_cache_slots[object_index] = SIZE_MAX;
        batch.stage_slot_indices[object_index] = SIZE_MAX;
        batch.caller_role_masks[object_index] = 0;
        batch.common_requests[object_index] = ggml_moe_common::object_request{};
        batch.transaction_tickets[object_index] =
            ggml_moe_common::object_transaction_ticket{};
    }
    batch.common_transactions_owned = false;
    batch.dependency_h2d = false;
    batch.count = 0;
    batch.done = false;
    batch.success = false;
    batch.stage_ring = nullptr;
    batch.host_finalized = false;
    batch.pipeline_submitted = false;
    batch.pipeline_critical_returned = false;
    batch.lossless_decode_all = false;
}

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

static bool ds4_dependency_h2d_eligible() {
    if (!ds4_dependency_h2d_requested()) return false;
    const size_t max_role_bytes = (size_t) std::max({
        g_ds4_triplet_pack.role_bytes[0],
        g_ds4_triplet_pack.role_bytes[1],
        g_ds4_triplet_pack.role_bytes[2],
    });
    const bool eligible = max_role_bytes != 0 && g_ds4_triplet_pack.object_bytes != 0;
    if (!eligible) {
        static std::atomic<bool> reported{false};
        if (!reported.exchange(true, std::memory_order_relaxed)) {
            std::fprintf(stderr,
                "[moe_stream] dependency H2D: falling back to object-level path "
                "(object_bytes=%lu max_role_bytes=%zu)\n",
                g_ds4_triplet_pack.object_bytes,
                max_role_bytes);
        }
    }
    return eligible;
}

static size_t ds4_dependency_role_slot_bytes() {
    return (size_t) std::max({
        g_ds4_triplet_pack.role_bytes[0],
        g_ds4_triplet_pack.role_bytes[1],
        g_ds4_triplet_pack.role_bytes[2],
    });
}

static const char * ds4_dependency_role_name(int role) {
    static const char * names[3] = {
        "ds4_dependency_gate",
        "ds4_dependency_up",
        "ds4_dependency_down",
    };
    return role >= 0 && role < 3 ? names[role] : "ds4_dependency";
}

static void ds4_dependency_fail_reservations(ds4_triplet_pipe_batch & batch) {
    for (int object_index = 0; object_index < batch.count; ++object_index) {
        for (int role = 0; role < 3; ++role) {
            if (batch.role_reservations[object_index][role].slot >= 0) {
                vram_cache_fail_triplet(batch.role_reservations[object_index][role]);
                batch.role_reservations[object_index][role] = {};
                batch.role_reservations[object_index][role].slot = -1;
                batch.role_ptrs[object_index][role] = nullptr;
            }
        }
    }
}

static void ds4_dependency_complete_owned_transactions(
        ds4_triplet_pipe_batch & batch,
        bool success) {
    if (!batch.common_transactions_owned || batch.count <= 0) return;
    if (!ggml_moe_common::object_engine_complete_batch(
            batch.common_requests, (size_t) batch.count, success)) {
        g_ds4_triplet_pack.common_transaction_fallbacks.fetch_add(
            1, std::memory_order_relaxed);
    }
    if (!success) {
        g_ds4_triplet_pack.dependency_transaction_failures.fetch_add(
            batch.count, std::memory_order_relaxed);
    }
    batch.common_transactions_owned = false;
}

static bool ds4_dependency_reserve_missing(
        ds4_triplet_pipe_batch & batch,
        cudaStream_t transfer_stream) {
    if (!batch.dependency_h2d || !transfer_stream || batch.count <= 0 || batch.count > 8) {
        return false;
    }
    for (int object_index = 0; object_index < batch.count; ++object_index) {
        const ggml_moe_common::dependency_transfer_plan & plan = batch.transfer_plans[object_index];
        for (int role = 0; role < 3; ++role) {
            const uint8_t bit = ggml_moe_common::role_bit(static_cast<ggml_moe_common::role>(role));
            if ((plan.missing_role_mask & bit) == 0) continue;
            if (!vram_cache_reserve_triplet(
                    ds4_triplet_role_key(*batch.entries[object_index], role),
                    (size_t) g_ds4_triplet_pack.role_bytes[role],
                    transfer_stream,
                    batch.role_reservations[object_index][role],
                    ds4_dependency_role_name(role),
                    false)) {
                ds4_dependency_fail_reservations(batch);
                return false;
            }
            batch.role_ptrs[object_index][role] =
                batch.role_reservations[object_index][role].dst;
        }
    }
    return true;
}

static bool ds4_dependency_acquire_host_object(
        const ds4_triplet_pack_entry & entry,
        ds4_triplet_pipe_batch & batch,
        int object_index) {
    if (object_index < 0 || object_index >= batch.count ||
            !g_vcache.ram_cache.slab.storage.host ||
            g_vcache.ram_cache.slab.slot_bytes < entry.nbytes) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_vcache.mu);
    const uintptr_t key = (uintptr_t) entry.generation;
    const int slot = ram_cache_find_slot_locked(key);
    if (slot < 0 || !ram_cache_slot_ready_now_locked(slot)) {
        g_vcache.ram_misses.fetch_add(1, std::memory_order_relaxed);
        g_ds4_triplet_pack.dependency_host_misses.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    ggml_moe_common::object_host_slot & host_slot =
        g_vcache.ram_cache.slots[(size_t) slot];
    if (!host_slot.host || host_slot.capacity < entry.nbytes ||
            !ggml_moe_common::object_host_cache_begin_h2d(
                g_vcache.ram_cache, (size_t) slot)) {
        g_ds4_triplet_pack.dependency_host_lease_failures.fetch_add(
            1, std::memory_order_relaxed);
        return false;
    }
    host_slot.used = g_vcache.clock.fetch_add(1, std::memory_order_relaxed);
    g_vcache.ram_hits.fetch_add(1, std::memory_order_relaxed);
    g_vcache.ram_host_lease_acquires.fetch_add(1, std::memory_order_relaxed);
    g_ds4_triplet_pack.dependency_host_hits.fetch_add(1, std::memory_order_relaxed);
    if (ds4_ram_score_policy_enabled()) {
        ds4_ram_key_stat * stat = ds4_ram_key_stat_locked(key);
        if (stat) {
            if (stat->ram_hits != UINT32_MAX) ++stat->ram_hits;
            if (stat->layer >= 0 && stat->layer < 43) {
                ++g_ds4_ram_layer_hits[stat->layer];
            }
        }
    }
    batch.host[object_index] = host_slot.host;
    batch.host_cache_hits[object_index] = true;
    batch.host_cache_slots[object_index] = (size_t) slot;
    return true;
}

static bool ds4_dependency_record_host_object(
        ds4_triplet_pipe_batch & batch,
        int object_index,
        cudaStream_t completion_stream) {
    if (object_index < 0 || object_index >= batch.count ||
            !batch.host_cache_hits[object_index] ||
            batch.host_cache_slots[object_index] == SIZE_MAX) {
        return true;
    }
    std::lock_guard<std::mutex> lock(g_vcache.mu);
    if (ggml_moe_common::object_host_cache_record_h2d(
            g_vcache.ram_cache,
            batch.host_cache_slots[object_index],
            completion_stream)) {
        return true;
    }
    g_ds4_triplet_pack.dependency_host_lease_failures.fetch_add(
        1, std::memory_order_relaxed);
    return false;
}

static void ds4_dependency_cancel_host_objects(
        ds4_triplet_pipe_batch & batch,
        cudaStream_t completion_stream) {
    for (int object_index = 0; object_index < batch.count; ++object_index) {
        if (!batch.host_cache_hits[object_index] ||
                batch.host_cache_slots[object_index] == SIZE_MAX) {
            continue;
        }
        std::lock_guard<std::mutex> lock(g_vcache.mu);
        if (ggml_moe_common::object_host_cache_cancel_h2d(
                g_vcache.ram_cache,
                batch.host_cache_slots[object_index],
                completion_stream)) {
            g_vcache.ram_host_lease_releases.fetch_add(1, std::memory_order_relaxed);
        } else {
            g_ds4_triplet_pack.dependency_host_lease_failures.fetch_add(
                1, std::memory_order_relaxed);
        }
    }
}

static bool ds4_dependency_admit_host_object(
        const ds4_triplet_pack_entry & entry,
        const void * host_object) {
    if (!host_object || !g_vcache.ram_cache.slab.storage.host ||
            g_vcache.ram_cache.slab.slot_bytes < entry.nbytes) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_vcache.mu);
    const uintptr_t key = (uintptr_t) entry.generation;
    if (ram_cache_find_slot_locked(key) >= 0) return false;
    if (!ram_cache_store_host_locked(key, host_object, (size_t) entry.nbytes, -1)) {
        return false;
    }
    g_ds4_triplet_pack.dependency_host_admissions.fetch_add(1, std::memory_order_relaxed);
    return true;
}

static uint64_t ds4_triplet_manifest_mix(uint64_t hash, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        hash ^= value & 0xff;
        hash *= UINT64_C(1099511628211);
        value >>= 8;
    }
    return hash;
}

static ggml_moe_common::object_request ds4_triplet_common_request(
        const ds4_triplet_pack_entry & entry,
        uint8_t role_mask) {
    ggml_moe_common::object_request request;
    request.key.source_idx = 0;
    request.key.layer_idx = entry.layer;
    request.key.expert_idx = entry.expert;
    request.key.manifest_generation = g_ds4_triplet_pack.manifest_generation;
    request.key.offset = entry.offset;
    request.key.bytes = (size_t) entry.nbytes;
    request.key.object_generation = entry.generation;
    uint64_t relative_offset = 0;
    for (int role_idx = 0; role_idx < 3; ++role_idx) {
        const ggml_moe_common::role kind = static_cast<ggml_moe_common::role>(role_idx);
        if (role_mask & ggml_moe_common::role_bit(kind)) {
            request.roles[request.role_count++] = {
                kind,
                relative_offset,
                (size_t) g_ds4_triplet_pack.role_bytes[role_idx],
            };
            request.required_role_mask |= ggml_moe_common::role_bit(kind);
        }
        relative_offset += g_ds4_triplet_pack.role_bytes[role_idx];
    }
    return request;
}

static bool ds4_ram_common_object_key(
        uintptr_t key,
        size_t bytes,
        ggml_moe_common::object_key &object) {
    if (key == 0 || !g_ds4_triplet_pack.enabled ||
            bytes != g_ds4_triplet_pack.object_bytes) {
        return false;
    }
    const auto stat = g_ds4_ram_key_stats.find(key);
    if (stat == g_ds4_ram_key_stats.end() || stat->second.layer < 0 ||
            stat->second.layer >= 43 || stat->second.expert < 0 ||
            stat->second.expert >= 256) {
        return false;
    }
    const ds4_triplet_pack_entry &entry = g_ds4_triplet_pack.entries[
        (size_t)stat->second.layer * 256 + (size_t)stat->second.expert];
    if ((uintptr_t)entry.generation != key || entry.nbytes != bytes) return false;
    const uint8_t full_role_mask =
        ggml_moe_common::role_bit(ggml_moe_common::role::gate) |
        ggml_moe_common::role_bit(ggml_moe_common::role::up) |
        ggml_moe_common::role_bit(ggml_moe_common::role::down);
    object = ds4_triplet_common_request(entry, full_role_mask).key;
    return true;
}

static bool ds4_triplet_common_begin(
        const ds4_triplet_pack_entry * const * entries,
        int count,
        uint8_t role_mask,
        ggml_moe_common::object_request (&requests)[8]) {
    if (!ggml_moe_common::object_engine_enabled()) return false;
    if (!entries || count <= 0 || count > 8) {
        g_ds4_triplet_pack.common_transaction_fallbacks.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    for (int i = 0; i < count; ++i) {
        if (!entries[i]) {
            g_ds4_triplet_pack.common_transaction_fallbacks.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        requests[i] = ds4_triplet_common_request(*entries[i], role_mask);
    }
    if (!ggml_moe_common::object_engine_begin_batch(requests, (size_t) count)) {
        g_ds4_triplet_pack.common_transaction_fallbacks.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

static void ds4_triplet_common_complete(
        ggml_moe_common::object_request (&requests)[8],
        int count,
        bool started,
        bool success) {
    if (started && !ggml_moe_common::object_engine_complete_batch(requests, (size_t) count, success)) {
        g_ds4_triplet_pack.common_transaction_fallbacks.fetch_add(1, std::memory_order_relaxed);
    }
}

struct ds4_triplet_trace_state {
    FILE * fp = nullptr;
    bool initialized = false;
    uint64_t layer_cycle[43] = {};
    std::atomic<uint64_t> next_call{1};
    std::atomic<uint64_t> next_event{1};
    std::mutex mu;
};

static ds4_triplet_trace_state g_ds4_triplet_trace;

static uint64_t ds4_triplet_trace_cycle(int layer, int role) {
    if (layer < 0 || layer >= 43) return 0;
    std::lock_guard<std::mutex> lock(g_ds4_triplet_trace.mu);
    if (role == 0) ++g_ds4_triplet_trace.layer_cycle[layer];
    return g_ds4_triplet_trace.layer_cycle[layer];
}

static void ds4_triplet_trace_record(
        uint64_t call_id,
        uint64_t layer_cycle,
        const ds4_triplet_pack_entry * entry,
        int role,
        int request_index,
        int route_rank,
        const char * event,
        int cache_state,
        int slot_id,
        uint64_t slot_generation) {
    const char * path = std::getenv("GGML_MOE_STREAM_DS4_TRIPLET_TRACE_OUT");
    if (!path || !path[0] || !entry) return;

    std::lock_guard<std::mutex> lock(g_ds4_triplet_trace.mu);
    if (!g_ds4_triplet_trace.initialized) {
        g_ds4_triplet_trace.initialized = true;
        g_ds4_triplet_trace.fp = std::fopen(path, "w");
        if (g_ds4_triplet_trace.fp) {
            std::setvbuf(g_ds4_triplet_trace.fp, nullptr, _IOLBF, 0);
            std::fprintf(g_ds4_triplet_trace.fp,
                    "event_seq,call_id,layer_cycle,layer,expert,role,request_index,route_rank,event,cache_state,slot_id,slot_generation,object_generation,offset,nbytes\n");
        } else {
            std::fprintf(stderr, "[moe_stream] DS4 triplet trace: failed to open %s\n", path);
        }
    }
    if (!g_ds4_triplet_trace.fp) return;
    const uint64_t event_seq = g_ds4_triplet_trace.next_event.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(g_ds4_triplet_trace.fp,
            "%lu,%lu,%lu,%d,%d,%d,%d,%d,%s,%d,%d,%lu,%lu,%lu,%lu\n",
            (unsigned long) event_seq,
            (unsigned long) call_id,
            (unsigned long) layer_cycle,
            entry->layer,
            entry->expert,
            role,
            request_index,
            route_rank,
            event ? event : "",
            cache_state,
            slot_id,
            (unsigned long) slot_generation,
            (unsigned long) entry->generation,
            (unsigned long) entry->offset,
            (unsigned long) entry->nbytes);
}

struct ds4_triplet_host_handoff_entry {
    const void * object = nullptr;
    uint64_t generation = 0;
};

static ds4_triplet_host_handoff_entry g_ds4_triplet_host_handoffs[8];
static std::mutex g_ds4_triplet_host_handoff_mu;
static int g_ds4_triplet_host_handoff_next = 0;

static void ds4_triplet_host_handoff_invalidate(const void * object) {
    if (!object) return;
    std::lock_guard<std::mutex> lock(g_ds4_triplet_host_handoff_mu);
    for (ds4_triplet_host_handoff_entry & entry : g_ds4_triplet_host_handoffs) {
        if (entry.object == object) entry = {};
    }
}

static void ds4_triplet_host_handoff_publish(const void * object, uint64_t generation) {
    if (!object || generation == 0) return;
    std::lock_guard<std::mutex> lock(g_ds4_triplet_host_handoff_mu);
    for (ds4_triplet_host_handoff_entry & entry : g_ds4_triplet_host_handoffs) {
        if (entry.object == object || entry.generation == generation) {
            entry.object = object;
            entry.generation = generation;
            return;
        }
    }
    ds4_triplet_host_handoff_entry & entry =
        g_ds4_triplet_host_handoffs[g_ds4_triplet_host_handoff_next++ % 8];
    entry.object = object;
    entry.generation = generation;
}

static void ds4_dependency_finalize_host_sources(
        ds4_triplet_pipe_batch & batch,
        cudaStream_t completion_stream) {
    if (batch.host_finalized || !completion_stream) return;
    bool host_record_failed = false;
    for (int object_index = 0; object_index < batch.count; ++object_index) {
        if (batch.host_cache_hits[object_index] &&
                !ds4_dependency_record_host_object(
                    batch, object_index, completion_stream)) {
            host_record_failed = true;
        }
    }
    if (host_record_failed) {
        ds4_dependency_cancel_host_objects(batch, completion_stream);
    }
    for (int object_index = 0; object_index < batch.count; ++object_index) {
        if (batch.host_cache_hits[object_index]) continue;
        const size_t slot_index = batch.stage_slot_indices[object_index];
        if (!batch.stage_ring || slot_index >= batch.stage_ring->slots.size()) continue;
        ggml_moe_common::object_stage_slot & slot = batch.stage_ring->slots[slot_index];
        if (!ggml_moe_common::object_stage_slot_record(slot, completion_stream)) {
            cudaGetLastError();
            cudaStreamSynchronize(completion_stream);
        } else {
            g_ds4_triplet_pack.dependency_stage_records.fetch_add(
                1, std::memory_order_relaxed);
        }
        if (!batch.lossless_objects[object_index]) {
            ds4_triplet_host_handoff_publish(
                slot.host, batch.entries[object_index]->generation);
            (void) ds4_dependency_admit_host_object(
                *batch.entries[object_index], slot.host);
        }
    }
    batch.host_finalized = true;
    g_ds4_triplet_pack.dependency_worker_host_finalizes.fetch_add(
        1, std::memory_order_relaxed);
}

static void ds4_triplet_pipe_worker_main();
static bool ds4_triplet_common_worker_init(void *);
static bool ds4_triplet_common_worker_process(void *, void *, uint32_t);

static bool one_pack_read_exact_fd(int fd, void * dst, size_t sz, uint64_t off) {
    return ggml_moe_common::object_pread_all(fd, dst, sz, off);
}

template<typename T>
static bool ds4_triplet_read_value(int fd, uint64_t offset, T & value) {
    return one_pack_read_exact_fd(fd, &value, sizeof(value), offset);
}

static bool ds4_triplet_requested() {
    const char * path = std::getenv("GGML_MOE_STREAM_DS4_TRIPLET_PACK");
    return path && path[0];
}

static bool ds4_triplet_role_pair_requested() {
    const char * env = std::getenv("GGML_MOE_STREAM_DS4_TRIPLET_ROLE_PAIR");
    return env && env[0] && env[0] != '0';
}

static bool ds4_lossless_load_sidecar() {
    const char * mode = std::getenv("GGML_MOE_DS4_LOSSLESS_MODE");
    const char * path = std::getenv("GGML_MOE_DS4_LOSSLESS_PACK");
    g_ds4_triplet_pack.lossless_requested =
        mode && std::strcmp(mode, "cuda") == 0 && path && path[0];
    if (!g_ds4_triplet_pack.lossless_requested) return true;
    if (!ggml_moe_common::lossless_sidecar_load(
            g_ds4_triplet_pack.lossless_sidecar,
            path,
            g_ds4_triplet_pack.source_pack_bytes) ||
            g_ds4_triplet_pack.lossless_sidecar.version !=
                ggml_moe_common::lossless_sidecar_version_mxfp4 ||
            g_ds4_triplet_pack.lossless_sidecar.layout !=
                ggml_moe_common::lossless_sidecar_mxfp4_layout ||
            g_ds4_triplet_pack.lossless_sidecar.table_count !=
                ggml_moe_common::lossless_rans_mxfp4_table_count ||
            g_ds4_triplet_pack.lossless_sidecar.objects.size() >
                g_ds4_triplet_pack.entries.size() ||
            g_ds4_triplet_pack.lossless_sidecar.max_payload_bytes == 0 ||
            g_ds4_triplet_pack.lossless_sidecar.max_physical_bytes >
                g_ds4_triplet_pack.object_bytes * 2) {
        std::fprintf(stderr,
            "[moe_stream] DS4 lossless: sidecar validation failed; raw fallback active\n");
        ggml_moe_common::lossless_sidecar_close(
            g_ds4_triplet_pack.lossless_sidecar);
        g_ds4_triplet_pack.lossless_failed = true;
        return false;
    }
    const auto sidecar_codec =
        g_ds4_triplet_pack.lossless_sidecar.objects[0].roles[0].codec;
    const bool nvcomp_sidecar = sidecar_codec ==
        ggml_moe_common::lossless_role_codec::nvcomp_mxfp4_rank;
    if (nvcomp_sidecar !=
            (g_ds4_triplet_pack.lossless_sidecar.nvcomp_chunk_bytes != 0) ||
            (nvcomp_sidecar && !ggml_moe_common::lossless_nvcomp_available())) {
        std::fprintf(stderr,
            "[moe_stream] DS4 lossless: nvCOMP sidecar/runtime mismatch; "
            "raw fallback active\n");
        ggml_moe_common::lossless_sidecar_close(
            g_ds4_triplet_pack.lossless_sidecar);
        g_ds4_triplet_pack.lossless_failed = true;
        return false;
    }
    for (size_t index = 0;
            index < g_ds4_triplet_pack.lossless_sidecar.objects.size();
            ++index) {
        const ggml_moe_common::lossless_object_record &object =
            g_ds4_triplet_pack.lossless_sidecar.objects[index];
        if (object.layer < 0 || object.expert < 0 || object.expert >= 256) {
            std::fprintf(stderr,
                "[moe_stream] DS4 lossless: invalid partial object identity at %zu; raw fallback active\n",
                index);
            ggml_moe_common::lossless_sidecar_close(
                g_ds4_triplet_pack.lossless_sidecar);
            g_ds4_triplet_pack.lossless_failed = true;
            return false;
        }
        const size_t source_index =
            static_cast<size_t>(object.layer) * 256 +
            static_cast<size_t>(object.expert);
        if (source_index >= g_ds4_triplet_pack.entries.size()) {
            std::fprintf(stderr,
                "[moe_stream] DS4 lossless: partial object out of range at %zu; raw fallback active\n",
                index);
            ggml_moe_common::lossless_sidecar_close(
                g_ds4_triplet_pack.lossless_sidecar);
            g_ds4_triplet_pack.lossless_failed = true;
            return false;
        }
        const ds4_triplet_pack_entry &entry =
            g_ds4_triplet_pack.entries[source_index];
        uint64_t raw_offset = entry.offset;
        bool valid = object.layer == entry.layer &&
            object.expert == entry.expert &&
            object.raw_bytes == entry.nbytes;
        for (size_t role = 0; valid && role < 3; ++role) {
            valid = object.roles[role].codec == sidecar_codec &&
                (sidecar_codec ==
                    ggml_moe_common::lossless_role_codec::rans_mxfp4 ||
                 sidecar_codec ==
                    ggml_moe_common::lossless_role_codec::compact_mxfp4_scale3 ||
                 sidecar_codec ==
                    ggml_moe_common::lossless_role_codec::nvcomp_mxfp4_rank) &&
                object.roles[role].raw_offset == raw_offset &&
                object.roles[role].raw_bytes ==
                    g_ds4_triplet_pack.role_bytes[role] &&
                object.roles[role].chunks ==
                    g_ds4_triplet_pack.lossless_sidecar.chunks_per_role;
            raw_offset += g_ds4_triplet_pack.role_bytes[role];
        }
        if (!valid || raw_offset != entry.offset + entry.nbytes) {
            std::fprintf(stderr,
                "[moe_stream] DS4 lossless: object identity mismatch at %zu; raw fallback active\n",
                index);
            ggml_moe_common::lossless_sidecar_close(
                g_ds4_triplet_pack.lossless_sidecar);
            g_ds4_triplet_pack.lossless_failed = true;
            return false;
        }
    }
    std::fprintf(stderr,
        "[moe_stream] DS4 lossless: loaded %zu/%zu MXFP4 objects from %s, "
        "max_payload=%.2f MiB max_physical=%.2f MiB chunks=%u context=%u\n",
        g_ds4_triplet_pack.lossless_sidecar.objects.size(),
        g_ds4_triplet_pack.entries.size(), path,
        g_ds4_triplet_pack.lossless_sidecar.max_payload_bytes /
            (1024.0 * 1024.0),
        g_ds4_triplet_pack.lossless_sidecar.max_physical_bytes /
            (1024.0 * 1024.0),
        g_ds4_triplet_pack.lossless_sidecar.chunks_per_role,
        g_ds4_triplet_pack.lossless_sidecar.quant_context_mode);
    return true;
}

static void ds4_lossless_cuda_release() {
    ggml_moe_common::lossless_nvcomp_destroy(
        g_ds4_triplet_pack.lossless_nvcomp_decoder);
    ggml_moe_common::lossless_rans_cuda_destroy(
        g_ds4_triplet_pack.lossless_deferred_decoder);
    ggml_moe_common::lossless_rans_cuda_destroy(
        g_ds4_triplet_pack.lossless_critical_decoder);
    g_ds4_triplet_pack.lossless_deferred_decoder = nullptr;
    g_ds4_triplet_pack.lossless_critical_decoder = nullptr;
    g_ds4_triplet_pack.lossless_nvcomp_decoder = nullptr;
    cudaFreeHost(g_ds4_triplet_pack.lossless_deferred_status_host);
    cudaFreeHost(g_ds4_triplet_pack.lossless_critical_status_host);
    cudaFreeHost(g_ds4_triplet_pack.lossless_crc32_host);
    cudaFree(g_ds4_triplet_pack.lossless_device_input);
    g_ds4_triplet_pack.lossless_deferred_status_host = nullptr;
    g_ds4_triplet_pack.lossless_critical_status_host = nullptr;
    g_ds4_triplet_pack.lossless_crc32_host = nullptr;
    g_ds4_triplet_pack.lossless_device_input = nullptr;
    g_ds4_triplet_pack.lossless_crc32_capacity = 0;
    g_ds4_triplet_pack.lossless_crc_verified.clear();
    g_ds4_triplet_pack.lossless_ready = false;
}

static bool ds4_lossless_cuda_init() {
    if (!g_ds4_triplet_pack.lossless_requested ||
            g_ds4_triplet_pack.lossless_failed) {
        return true;
    }
    constexpr size_t input_slots = 8;
    constexpr size_t max_roles = input_slots * 3;
    constexpr size_t crc32_capacity =
        max_roles * ggml_moe_common::lossless_crc32_segments;
    constexpr size_t lossless_input_alignment = 4096;
    const size_t stride =
        (g_ds4_triplet_pack.lossless_sidecar.max_payload_bytes +
            lossless_input_alignment - 1) &
        ~(lossless_input_alignment - 1);
    g_ds4_triplet_pack.lossless_device_input_stride = stride;
    g_ds4_triplet_pack.lossless_critical_decoder =
        ggml_moe_common::lossless_rans_cuda_create(
            g_ds4_triplet_pack.lossless_sidecar.tables.data(),
            g_ds4_triplet_pack.lossless_sidecar.table_count,
            max_roles);
    g_ds4_triplet_pack.lossless_deferred_decoder =
        ggml_moe_common::lossless_rans_cuda_create(
            g_ds4_triplet_pack.lossless_sidecar.tables.data(),
            g_ds4_triplet_pack.lossless_sidecar.table_count,
            max_roles);
    const bool nvcomp_sidecar =
        !g_ds4_triplet_pack.lossless_sidecar.objects.empty() &&
        g_ds4_triplet_pack.lossless_sidecar.objects[0].roles[0].codec ==
            ggml_moe_common::lossless_role_codec::nvcomp_mxfp4_rank;
    if (nvcomp_sidecar) {
        g_ds4_triplet_pack.lossless_nvcomp_decoder =
            ggml_moe_common::lossless_nvcomp_create(
                g_ds4_triplet_pack.lossless_sidecar.nvcomp_chunk_bytes,
                g_ds4_triplet_pack.object_bytes,
                input_slots,
                g_ds4_triplet_pack.lossless_sidecar.mxfp4_rank_to_symbol.data(),
                g_ds4_triplet_pack.lossless_sidecar.mxfp4_rank_to_symbol.size(),
                g_ds4_triplet_pack.pipe_stream);
    }
    const bool rank_remap_ready =
        g_ds4_triplet_pack.lossless_sidecar.quant_context_mode != 3 ||
        (g_ds4_triplet_pack.lossless_critical_decoder &&
         g_ds4_triplet_pack.lossless_deferred_decoder &&
         ggml_moe_common::lossless_rans_cuda_set_mxfp4_rank_remap(
            g_ds4_triplet_pack.lossless_critical_decoder,
            g_ds4_triplet_pack.lossless_sidecar.mxfp4_rank_to_symbol.data(),
            g_ds4_triplet_pack.lossless_sidecar.mxfp4_rank_to_symbol.size()) &&
         ggml_moe_common::lossless_rans_cuda_set_mxfp4_rank_remap(
            g_ds4_triplet_pack.lossless_deferred_decoder,
            g_ds4_triplet_pack.lossless_sidecar.mxfp4_rank_to_symbol.data(),
            g_ds4_triplet_pack.lossless_sidecar.mxfp4_rank_to_symbol.size()));
    if (!g_ds4_triplet_pack.lossless_critical_decoder ||
            !g_ds4_triplet_pack.lossless_deferred_decoder ||
            !rank_remap_ready ||
            (nvcomp_sidecar &&
             !g_ds4_triplet_pack.lossless_nvcomp_decoder) ||
            cudaMalloc(
                reinterpret_cast<void **>(
                    &g_ds4_triplet_pack.lossless_device_input),
                input_slots * stride) != cudaSuccess ||
            cudaMallocHost(
                reinterpret_cast<void **>(
                    &g_ds4_triplet_pack.lossless_critical_status_host),
                max_roles * sizeof(uint32_t)) != cudaSuccess ||
            cudaMallocHost(
                reinterpret_cast<void **>(
                    &g_ds4_triplet_pack.lossless_deferred_status_host),
                max_roles * sizeof(uint32_t)) != cudaSuccess ||
            cudaMallocHost(
                reinterpret_cast<void **>(
                    &g_ds4_triplet_pack.lossless_crc32_host),
                crc32_capacity * sizeof(uint32_t)) != cudaSuccess) {
        cudaGetLastError();
        ds4_lossless_cuda_release();
        g_ds4_triplet_pack.lossless_failed = true;
        std::fprintf(stderr,
            "[moe_stream] DS4 lossless: CUDA allocation failed; raw fallback active\n");
        return true;
    }
    g_ds4_triplet_pack.lossless_crc32_capacity = crc32_capacity;
    g_ds4_triplet_pack.lossless_crc_verified.assign(
        g_ds4_triplet_pack.lossless_sidecar.objects.size(), 0);
    g_ds4_triplet_pack.lossless_ready = true;
    return true;
}

static void ds4_triplet_report_atexit() {
    if (!g_ds4_triplet_pack.enabled) return;
    if (g_ds4_ram_prefetch.worker.joinable()) {
        {
            std::lock_guard<std::mutex> lock(g_ds4_ram_prefetch.mu);
            g_ds4_ram_prefetch.stop = true;
        }
        g_ds4_ram_prefetch.cv.notify_one();
        g_ds4_ram_prefetch.worker.join();
    }
    if (g_ds4_triplet_pack.pipe_common_worker) {
        ggml_moe_common::object_engine_stop_worker();
        if (g_ds4_triplet_pack.pipe_stream) {
            (void) cudaStreamSynchronize(g_ds4_triplet_pack.pipe_stream);
        }
        if (g_ds4_triplet_pack.pipe_deferred_stream) {
            (void) cudaStreamSynchronize(
                g_ds4_triplet_pack.pipe_deferred_stream);
        }
        ds4_lossless_cuda_release();
        if (g_ds4_triplet_pack.pipe_critical_done) {
            cudaEventDestroy(g_ds4_triplet_pack.pipe_critical_done);
            g_ds4_triplet_pack.pipe_critical_done = nullptr;
        }
        if (g_ds4_triplet_pack.lossless_decode_done) {
            cudaEventDestroy(g_ds4_triplet_pack.lossless_decode_done);
            g_ds4_triplet_pack.lossless_decode_done = nullptr;
        }
        if (g_ds4_triplet_pack.pipe_deferred_stream) {
            cudaStreamDestroy(g_ds4_triplet_pack.pipe_deferred_stream);
            g_ds4_triplet_pack.pipe_deferred_stream = nullptr;
        }
        if (g_ds4_triplet_pack.pipe_stream) {
            cudaStreamDestroy(g_ds4_triplet_pack.pipe_stream);
            g_ds4_triplet_pack.pipe_stream = nullptr;
        }
        g_ds4_triplet_pack.pipe_common_worker = false;
        g_ds4_triplet_pack.pipe_started = false;
    } else if (g_ds4_triplet_pack.pipe_worker.joinable()) {
        {
            std::lock_guard<std::mutex> lock(g_ds4_triplet_pack.pipe_mu);
            g_ds4_triplet_pack.pipe_stop = true;
        }
        g_ds4_triplet_pack.pipe_cv.notify_one();
        g_ds4_triplet_pack.pipe_worker.join();
    }
    for (ds4_triplet_pipe_batch & batch :
            g_ds4_triplet_pack.dependency_pipeline_batches) {
        ggml_moe_common::object_stage_ring_release(batch.pipeline_stage_ring);
    }
    {
        std::lock_guard<std::mutex> lock(g_vcache.mu);
        for (size_t slot_index = 0; slot_index < g_vcache.ram_cache.slots.size(); ++slot_index) {
            ggml_moe_common::object_host_slot &slot = g_vcache.ram_cache.slots[slot_index];
            if (slot.pending_h2d) {
                (void)ram_cache_release_common_lease_locked(slot_index);
            }
            if (!slot.pending_d2h) continue;
            if (!slot.d2h_done || cudaEventSynchronize(slot.d2h_done) == cudaSuccess) {
                (void)ram_cache_finish_pending_event_locked(slot_index);
            } else {
                cudaGetLastError();
                (void)ram_cache_fail_pending_event_locked(slot_index);
            }
        }
    }
    const ggml_moe_common::object_io_runtime_snapshot io_stats =
        ggml_moe_common::object_io_runtime_snapshot_load(g_ds4_triplet_pack.io_stats);
    std::fprintf(stderr,
            "[moe_stream] DS4 triplet pack: hits=%lu misses=%lu reads=%lu bytes=%lu failures=%lu "
            "objects=%zu object_mib=%.2f direct=%d iouring_batches=%lu iouring_reads=%lu "
            "iouring_submit_ms=%.3f iouring_wait_ms=%.3f iouring_fallbacks=%lu "
            "pipe_batches=%lu pipe_cqes=%lu pipe_submit_ms=%.3f pipe_wait_ms=%.3f "
            "pipe_process_ms=%.3f pipe_caller_wait_ms=%.3f pipe_h2d_bytes=%lu pipe_h2d_enqueue_ms=%.3f "
            "pipe_h2d_device_ms=%.3f pipe_refill_batches=%lu pipe_refill_submit_rounds=%lu "
            "pipe_refill_submitted=%lu pipe_refill_completions=%lu pipe_refill_fallbacks=%lu "
            "host_handoff_lookups=%lu host_handoff_hits=%lu host_handoff_bytes=%lu role_pair_reads=%lu "
            "common_transaction_fallbacks=%lu cqe_validations=%lu cqe_completed=%lu "
            "cqe_invalid_user_data=%lu cqe_invalid_expected_bytes=%lu cqe_io_errors=%lu "
            "cqe_short_reads=%lu cqe_oversized_reads=%lu common_window_prepares=%lu "
            "common_window_resolves=%lu sqpoll_affinity=%lu/%lu/%lu ring_init_on_worker=%d "
            "ring_init_fallback=%d ring_requested_flags=%u ring_active_flags=%u\n",
            g_ds4_triplet_pack.hits.load(),
            g_ds4_triplet_pack.misses.load(),
            g_ds4_triplet_pack.reads.load(),
            g_ds4_triplet_pack.bytes.load(),
            g_ds4_triplet_pack.failures.load(),
            g_ds4_triplet_pack.entries.size(),
            g_ds4_triplet_pack.object_bytes / (1024.0 * 1024.0),
            g_ds4_triplet_pack.fd_direct >= 0 ? 1 : 0,
            io_stats.batches,
            io_stats.reads,
            io_stats.submit_us / 1000.0,
            io_stats.wait_us / 1000.0,
            io_stats.fallbacks,
            g_ds4_triplet_pack.pipe_batches.load(),
            g_ds4_triplet_pack.pipe_cqes.load(),
            g_ds4_triplet_pack.pipe_submit_us.load() / 1000.0,
            g_ds4_triplet_pack.pipe_wait_us.load() / 1000.0,
            g_ds4_triplet_pack.pipe_process_us.load() / 1000.0,
            g_ds4_triplet_pack.pipe_caller_wait_us.load() / 1000.0,
            g_ds4_triplet_pack.pipe_h2d_bytes.load(),
            g_ds4_triplet_pack.pipe_h2d_enqueue_us.load() / 1000.0,
            g_ds4_triplet_pack.pipe_h2d_device_us.load() / 1000.0,
            g_ds4_triplet_pack.pipe_refill_batches.load(),
            g_ds4_triplet_pack.pipe_refill_submit_rounds.load(),
            g_ds4_triplet_pack.pipe_refill_submitted.load(),
            g_ds4_triplet_pack.pipe_refill_completions.load(),
            g_ds4_triplet_pack.pipe_refill_fallbacks.load(),
            g_ds4_triplet_pack.host_handoff_lookups.load(),
            g_ds4_triplet_pack.host_handoff_hits.load(),
            g_ds4_triplet_pack.host_handoff_hits.load() * g_ds4_triplet_pack.role_bytes[2],
            g_ds4_triplet_pack.role_pair_reads.load(),
            g_ds4_triplet_pack.common_transaction_fallbacks.load(),
            g_ds4_triplet_pack.io_ring.completion_counters.validations,
            g_ds4_triplet_pack.io_ring.completion_counters.completed,
            g_ds4_triplet_pack.io_ring.completion_counters.invalid_user_data,
            g_ds4_triplet_pack.io_ring.completion_counters.invalid_expected_bytes,
            g_ds4_triplet_pack.io_ring.completion_counters.io_errors,
            g_ds4_triplet_pack.io_ring.completion_counters.short_reads,
            g_ds4_triplet_pack.io_ring.completion_counters.oversized_reads,
            g_ds4_triplet_pack.io_ring.window_prepares,
            g_ds4_triplet_pack.io_ring.window_resolves,
            g_ds4_triplet_pack.io_ring.iouring_sqpoll_affinity_attempts,
            g_ds4_triplet_pack.io_ring.iouring_sqpoll_affinity_successes,
            g_ds4_triplet_pack.io_ring.iouring_sqpoll_affinity_failures,
            g_ds4_triplet_pack.ring_init_on_worker ? 1 : 0,
            g_ds4_triplet_pack.ring_init_fallback ? 1 : 0,
            g_ds4_triplet_pack.ring_requested_flags,
            g_ds4_triplet_pack.ring_active_flags);
    const ggml_moe_common::object_pipeline_mode_config & pipeline_mode =
        ds4_pipeline_mode_config();
    const ggml_moe_common::object_pipeline_policy_config & pipeline_policy =
        ds4_pipeline_policy_config();
    const ggml_moe_common::object_pipeline_dispatch_runtime_snapshot pipeline_dispatch =
        ggml_moe_common::object_pipeline_dispatch_runtime_snapshot_load();
    std::fprintf(stderr,
            "[moe_stream] DS4 common pipeline config: mode=%s source=%s valid=%d "
            "policy_valid=%d overrides=0x%x invalid=0x%x io_depth=%zu initial=%zu "
            "refill=%zu min_flush=%zu timeout_us=%lu max_h2d=%zu active=%d\n",
            ggml_moe_common::object_pipeline_mode_name(pipeline_mode.mode),
            ggml_moe_common::object_pipeline_config_source_name(pipeline_mode.source),
            pipeline_mode.valid ? 1 : 0,
            pipeline_policy.valid ? 1 : 0,
            pipeline_policy.override_mask,
            pipeline_policy.invalid_mask,
            pipeline_policy.policy.io_depth,
            pipeline_policy.policy.initial_batch,
            pipeline_policy.policy.refill_batch,
            pipeline_policy.policy.min_flush_batch,
            (unsigned long) pipeline_policy.policy.flush_timeout_us,
            pipeline_policy.policy.max_inflight_h2d,
            ds4_common_rolling_refill_requested() ? 1 : 0);
    ggml_moe_common::object_pipeline_metrics pipeline_metrics;
    pipeline_metrics.timing_mask =
        ggml_moe_common::object_pipeline_timing_lifetime |
        ggml_moe_common::object_pipeline_timing_foreground_wait |
        ggml_moe_common::object_pipeline_timing_ssd_read;
    if (g_ds4_triplet_pack.pipe_h2d_device_us.load() != 0) {
        pipeline_metrics.timing_mask |= ggml_moe_common::object_pipeline_timing_h2d;
    }
    pipeline_metrics.pipeline_lifetime_us =
        g_ds4_triplet_pack.pipe_process_us.load();
    pipeline_metrics.completed = g_ds4_triplet_pack.pipe_cqes.load();
    pipeline_metrics.failed = g_ds4_triplet_pack.failures.load();
    pipeline_metrics.submitted = pipeline_metrics.completed + pipeline_metrics.failed;
    pipeline_metrics.initial_submit_rounds = g_ds4_triplet_pack.pipe_batches.load();
    const uint64_t total_submit_rounds =
        g_ds4_triplet_pack.pipe_refill_submit_rounds.load();
    const uint64_t rolling_batches = g_ds4_triplet_pack.pipe_refill_batches.load();
    pipeline_metrics.refill_submit_rounds = total_submit_rounds > rolling_batches ?
        total_submit_rounds - rolling_batches : 0;
    pipeline_metrics.timeout_flush_rounds = pipeline_dispatch.timeout_flush_rounds;
    pipeline_metrics.timeout_wait_us = pipeline_dispatch.timeout_wait_us;
    pipeline_metrics.foreground_wait_us =
        g_ds4_triplet_pack.pipe_caller_wait_us.load();
    pipeline_metrics.ssd_reads = g_ds4_triplet_pack.reads.load();
    pipeline_metrics.ssd_bytes = g_ds4_triplet_pack.bytes.load();
    pipeline_metrics.ssd_read_us = g_ds4_triplet_pack.pipe_wait_us.load();
    pipeline_metrics.h2d_bytes = g_ds4_triplet_pack.pipe_h2d_bytes.load();
    pipeline_metrics.h2d_us = g_ds4_triplet_pack.pipe_h2d_device_us.load();
    pipeline_metrics.io_fallbacks = io_stats.fallbacks +
        g_ds4_triplet_pack.pipe_refill_fallbacks.load();
    pipeline_metrics.h2d_fallbacks =
        g_ds4_triplet_pack.dependency_fallbacks.load();
    ggml_moe_common::object_pipeline_metrics_report("deepseek", pipeline_metrics);
    if (g_ds4_triplet_pack.lossless_requested) {
        std::fprintf(stderr,
            "[moe_stream] DS4 lossless: ready=%d failed=%d reads=%lu "
            "physical_bytes=%lu payload_bytes=%lu raw_bytes_avoided=%lu "
            "h2d_bytes=%lu decode_batches=%lu decode_objects=%lu "
            "fallbacks=%lu crc_objects=%lu crc_reuses=%lu "
            "crc_failures=%lu decode_failures=%lu\n",
            g_ds4_triplet_pack.lossless_ready ? 1 : 0,
            g_ds4_triplet_pack.lossless_failed ? 1 : 0,
            g_ds4_triplet_pack.lossless_reads.load(),
            g_ds4_triplet_pack.lossless_physical_bytes.load(),
            g_ds4_triplet_pack.lossless_payload_bytes.load(),
            g_ds4_triplet_pack.lossless_raw_bytes_avoided.load(),
            g_ds4_triplet_pack.lossless_h2d_bytes.load(),
            g_ds4_triplet_pack.lossless_decode_batches.load(),
            g_ds4_triplet_pack.lossless_decode_objects.load(),
            g_ds4_triplet_pack.lossless_fallbacks.load(),
            g_ds4_triplet_pack.lossless_crc_objects.load(),
            g_ds4_triplet_pack.lossless_crc_reuses.load(),
            g_ds4_triplet_pack.lossless_crc_failures.load(),
            g_ds4_triplet_pack.lossless_decode_failures.load());
        std::fprintf(stderr,
            "[moe_stream] DS4 lossless policy: min_batch=%zu raw_objects=%lu "
            "batch_hist=0:%lu,1:%lu,2:%lu,3:%lu,4:%lu,5:%lu,6:%lu,7:%lu,8:%lu\n",
            ds4_lossless_min_batch(),
            g_ds4_triplet_pack.lossless_policy_raw_objects.load(),
            g_ds4_triplet_pack.lossless_candidate_batches[0].load(),
            g_ds4_triplet_pack.lossless_candidate_batches[1].load(),
            g_ds4_triplet_pack.lossless_candidate_batches[2].load(),
            g_ds4_triplet_pack.lossless_candidate_batches[3].load(),
            g_ds4_triplet_pack.lossless_candidate_batches[4].load(),
            g_ds4_triplet_pack.lossless_candidate_batches[5].load(),
            g_ds4_triplet_pack.lossless_candidate_batches[6].load(),
            g_ds4_triplet_pack.lossless_candidate_batches[7].load(),
            g_ds4_triplet_pack.lossless_candidate_batches[8].load());
    }
    static const char * role_names[3] = {"gate", "up", "down"};
    for (int role = 0; role < 3; ++role) {
        std::fprintf(stderr,
                "[moe_stream] DS4 triplet role: role=%s calls=%lu lookups=%lu hits=%lu misses=%lu "
                "ram_promotes=%lu physical_reads=%lu\n",
                role_names[role],
                g_ds4_triplet_pack.role_calls[role].load(),
                g_ds4_triplet_pack.role_lookups[role].load(),
                g_ds4_triplet_pack.role_hits[role].load(),
                g_ds4_triplet_pack.role_misses[role].load(),
                g_ds4_triplet_pack.role_ram_promotes[role].load(),
                g_ds4_triplet_pack.role_physical_reads[role].load());
    }
    if (ds4_dependency_h2d_requested() || g_ds4_triplet_pack.dependency_plans.load() != 0) {
        std::fprintf(stderr,
            "[moe_stream] DS4 dependency H2D: plans=%lu full_misses=%lu partial_misses=%lu "
            "critical_copies=%lu critical_bytes=%lu deferred_copies=%lu deferred_bytes=%lu "
            "host_hits=%lu host_misses=%lu host_admissions=%lu host_lease_failures=%lu "
            "transaction_owners=%lu transaction_joins=%lu join_wait_ms=%.3f "
            "transaction_failures=%lu pipeline_acquires=%lu pipeline_backpressure=%lu "
            "pipeline_releases=%lu pipeline_in_use=%zu critical_publishes=%lu "
            "early_critical_publishes=%lu critical_returns=%lu "
            "wait_all_fallbacks=%lu critical_wait_ms=%.3f "
            "worker_host_finalizes=%lu stage_records=%lu "
            "fallbacks=%lu\n",
            g_ds4_triplet_pack.dependency_plans.load(),
            g_ds4_triplet_pack.dependency_full_misses.load(),
            g_ds4_triplet_pack.dependency_partial_misses.load(),
            g_ds4_triplet_pack.dependency_critical_copies.load(),
            g_ds4_triplet_pack.dependency_critical_bytes.load(),
            g_ds4_triplet_pack.dependency_deferred_copies.load(),
            g_ds4_triplet_pack.dependency_deferred_bytes.load(),
            g_ds4_triplet_pack.dependency_host_hits.load(),
            g_ds4_triplet_pack.dependency_host_misses.load(),
            g_ds4_triplet_pack.dependency_host_admissions.load(),
            g_ds4_triplet_pack.dependency_host_lease_failures.load(),
            g_ds4_triplet_pack.dependency_transaction_owners.load(),
            g_ds4_triplet_pack.dependency_transaction_joins.load(),
            g_ds4_triplet_pack.dependency_transaction_join_wait_us.load() / 1000.0,
            g_ds4_triplet_pack.dependency_transaction_failures.load(),
            g_ds4_triplet_pack.dependency_pipeline_acquires.load(),
            g_ds4_triplet_pack.dependency_pipeline_backpressure.load(),
            g_ds4_triplet_pack.dependency_pipeline_releases.load(),
            g_ds4_triplet_pack.dependency_pipeline_pool.in_use(),
            g_ds4_triplet_pack.dependency_pipeline_critical_publishes.load(),
            g_ds4_triplet_pack.dependency_pipeline_early_critical_publishes.load(),
            g_ds4_triplet_pack.dependency_pipeline_critical_returns.load(),
            g_ds4_triplet_pack.dependency_pipeline_wait_all_fallbacks.load(),
            g_ds4_triplet_pack.dependency_pipeline_critical_wait_us.load() / 1000.0,
            g_ds4_triplet_pack.dependency_worker_host_finalizes.load(),
            g_ds4_triplet_pack.dependency_stage_records.load(),
            g_ds4_triplet_pack.dependency_fallbacks.load());
    }
    if (g_ds4_triplet_pack.mixed_role_calls.load() != 0) {
        std::fprintf(stderr, "[moe_stream] DS4 triplet invariant: mixed_role_calls=%lu\n",
                g_ds4_triplet_pack.mixed_role_calls.load());
    }
    if (g_ds4_ram_prefetch.started) {
        std::fprintf(stderr,
            "[moe_stream] DS4 RAM pilot prefetch: submitted=%lu completed=%lu used=%lu wasted=%lu "
            "duplicate_skips=%lu resident_skips=%lu read_failures=%lu bytes=%lu\n",
            g_ds4_ram_prefetch.submitted.load(), g_ds4_ram_prefetch.completed.load(),
            g_ds4_ram_prefetch.used.load(), g_ds4_ram_prefetch.wasted.load(),
            g_ds4_ram_prefetch.duplicate_skips.load(), g_ds4_ram_prefetch.resident_skips.load(),
            g_ds4_ram_prefetch.read_failures.load(), g_ds4_ram_prefetch.bytes.load());
    }
    if (g_ds4_triplet_pack.ring_inited) {
        ggml_moe_common::object_io_ring_release(g_ds4_triplet_pack.io_ring);
        g_ds4_triplet_pack.ring_inited = false;
    }
    ds4_lossless_cuda_release();
    ggml_moe_common::lossless_sidecar_close(
        g_ds4_triplet_pack.lossless_sidecar);
}

static void ds4_triplet_init_once() {
    std::lock_guard<std::mutex> lock(g_ds4_triplet_pack.mu);
    if (g_ds4_triplet_pack.inited) return;
    g_ds4_triplet_pack.inited = true;

    const char * path = std::getenv("GGML_MOE_STREAM_DS4_TRIPLET_PACK");
    if (!path || !path[0]) return;
    const int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "[moe_stream] DS4 triplet pack: open failed: %s\n", path);
        return;
    }

    char magic[16] = {};
    uint32_t version = 0;
    uint32_t metadata_bytes = 0;
    uint64_t object_count = 0;
    uint64_t data_start = 0;
    uint64_t object_bytes = 0;
    uint64_t role_bytes[3] = {};
    if (!one_pack_read_exact_fd(fd, magic, sizeof(magic), 0) ||
            !ds4_triplet_read_value(fd, 16, version) ||
            !ds4_triplet_read_value(fd, 20, metadata_bytes) ||
            !ds4_triplet_read_value(fd, 24, object_count) ||
            !ds4_triplet_read_value(fd, 32, data_start) ||
            !ds4_triplet_read_value(fd, 40, object_bytes) ||
            !ds4_triplet_read_value(fd, 48, role_bytes[0]) ||
            !ds4_triplet_read_value(fd, 56, role_bytes[1]) ||
            !ds4_triplet_read_value(fd, 64, role_bytes[2]) ||
            std::memcmp(magic, "DS4TRIPLETPACK1", 15) != 0 || version != 1 ||
            object_count != 43ULL * 256ULL || metadata_bytes != data_start ||
            metadata_bytes < 144 + object_count * 32 || (metadata_bytes & 4095U) != 0 ||
            object_bytes == 0 || object_bytes != role_bytes[0] + role_bytes[1] + role_bytes[2] ||
            role_bytes[0] != role_bytes[1] || role_bytes[1] != role_bytes[2]) {
        std::fprintf(stderr, "[moe_stream] DS4 triplet pack: invalid header: %s\n", path);
        ::close(fd);
        return;
    }

    struct stat st = {};
    const uint64_t expected_size = data_start + object_count * object_bytes;
    if (::fstat(fd, &st) != 0 || st.st_size < 0 || (uint64_t) st.st_size != expected_size) {
        std::fprintf(stderr, "[moe_stream] DS4 triplet pack: exact file size mismatch: %s\n", path);
        ::close(fd);
        return;
    }

    std::vector<ds4_triplet_pack_entry> entries((size_t) object_count);
    std::unordered_set<uint64_t> generations;
    generations.reserve((size_t) object_count);
    uint64_t manifest_generation = UINT64_C(1469598103934665603);
    manifest_generation = ds4_triplet_manifest_mix(manifest_generation, version);
    manifest_generation = ds4_triplet_manifest_mix(manifest_generation, metadata_bytes);
    manifest_generation = ds4_triplet_manifest_mix(manifest_generation, object_count);
    manifest_generation = ds4_triplet_manifest_mix(manifest_generation, data_start);
    manifest_generation = ds4_triplet_manifest_mix(manifest_generation, object_bytes);
    for (uint64_t role_size : role_bytes) {
        manifest_generation = ds4_triplet_manifest_mix(manifest_generation, role_size);
    }
    for (uint64_t ordinal = 0; ordinal < object_count; ++ordinal) {
        const uint64_t index_offset = 144 + ordinal * 32;
        ds4_triplet_pack_entry & entry = entries[(size_t) ordinal];
        if (!ds4_triplet_read_value(fd, index_offset + 0, entry.layer) ||
                !ds4_triplet_read_value(fd, index_offset + 4, entry.expert) ||
                !ds4_triplet_read_value(fd, index_offset + 8, entry.offset) ||
                !ds4_triplet_read_value(fd, index_offset + 16, entry.nbytes) ||
                !ds4_triplet_read_value(fd, index_offset + 24, entry.generation) ||
                entry.layer != (int32_t) (ordinal / 256) ||
                entry.expert != (int32_t) (ordinal % 256) ||
                entry.offset != data_start + ordinal * object_bytes ||
                entry.nbytes != object_bytes || entry.generation == 0 ||
                !generations.insert(entry.generation).second) {
            std::fprintf(stderr, "[moe_stream] DS4 triplet pack: invalid object index at %lu: %s\n",
                    (unsigned long) ordinal, path);
            ::close(fd);
            return;
        }
        manifest_generation = ds4_triplet_manifest_mix(manifest_generation, (uint32_t) entry.layer);
        manifest_generation = ds4_triplet_manifest_mix(manifest_generation, (uint32_t) entry.expert);
        manifest_generation = ds4_triplet_manifest_mix(manifest_generation, entry.offset);
        manifest_generation = ds4_triplet_manifest_mix(manifest_generation, entry.nbytes);
        manifest_generation = ds4_triplet_manifest_mix(manifest_generation, entry.generation);
    }
    if (manifest_generation == 0) manifest_generation = 1;
    g_ds4_ram_key_stats.clear();
    g_ds4_ram_key_stats.reserve(entries.size());
    for (const ds4_triplet_pack_entry & entry : entries) {
        g_ds4_ram_key_stats.emplace((uintptr_t) entry.generation,
            ds4_ram_key_stat{entry.layer, entry.expert, 0, 0});
    }

    const char * io = std::getenv("GGML_MOE_STREAM_DS4_TRIPLET_IO");
    const bool pipe_requested = io &&
        (std::strcmp(io, "pipe") == 0 || std::strcmp(io, "cqe_h2d") == 0);
    const bool iouring_requested = pipe_requested || (io &&
        (std::strcmp(io, "iouring") == 0 || std::strcmp(io, "io_uring") == 0));
    if (!io || !io[0] || std::strcmp(io, "direct") == 0 || std::strcmp(io, "odirect") == 0 || iouring_requested) {
#if defined(__linux__) && defined(O_DIRECT)
        g_ds4_triplet_pack.fd_direct = ::open(path, O_RDONLY | O_DIRECT);
        if (g_ds4_triplet_pack.fd_direct < 0) {
            std::fprintf(stderr, "[moe_stream] DS4 triplet pack: O_DIRECT open failed; buffered fallback active: %s\n", path);
        }
#endif
    }
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (iouring_requested && g_ds4_triplet_pack.fd_direct >= 0) {
        unsigned depth = (unsigned) std::max<size_t>(1,
            std::min<size_t>(ds4_pipeline_policy_config().policy.io_depth, 64));
        const char * defer_env =
            std::getenv("GGML_MOE_STREAM_DS4_TRIPLET_DEFER_TASKRUN");
        const bool defer_requested = defer_env && defer_env[0] && defer_env[0] != '0';
        g_ds4_triplet_pack.ring_depth = depth;
        g_ds4_triplet_pack.ring_requested_flags = defer_requested ?
            ggml_moe_common::object_io_deferred_taskrun_setup_flags() : 0;
        g_ds4_triplet_pack.ring_init_on_worker =
            defer_requested && g_ds4_triplet_pack.ring_requested_flags != 0;
        if (g_ds4_triplet_pack.ring_init_on_worker) {
            std::fprintf(stderr,
                "[moe_stream] DS4 triplet pack: deferred task-run ring will initialize on pipe worker\n");
        } else if (ggml_moe_common::object_io_ring_ensure(
                g_ds4_triplet_pack.io_ring, depth, 0, nullptr)) {
            g_ds4_triplet_pack.ring_inited = true;
        } else {
            std::fprintf(stderr, "[moe_stream] DS4 triplet pack: io_uring init failed; direct fallback active\n");
        }
    }
#else
    if (iouring_requested) {
        std::fprintf(stderr, "[moe_stream] DS4 triplet pack: io_uring unavailable; direct fallback active\n");
    }
#endif
    g_ds4_triplet_pack.fd = fd;
    g_ds4_triplet_pack.object_bytes = object_bytes;
    g_ds4_triplet_pack.source_pack_bytes = expected_size;
    std::copy(role_bytes, role_bytes + 3, g_ds4_triplet_pack.role_bytes);
    g_ds4_triplet_pack.manifest_generation = manifest_generation;
    g_ds4_triplet_pack.entries = std::move(entries);
    (void) ds4_lossless_load_sidecar();
    if (ggml_moe_common::object_engine_enabled()) {
        ggml_moe_common::object_layout layout;
        std::snprintf(layout.model, sizeof(layout.model), "%s", "deepseek-ds4");
        layout.source_idx = 0;
        layout.first_layer = 0;
        layout.layer_count = 43;
        layout.first_expert = 0;
        layout.experts_per_layer = 256;
        layout.top_k = 6;
        layout.object_count = object_count;
        layout.manifest_generation = manifest_generation;
        layout.shape_count = 1;
        layout.shapes[0].object_bytes = (size_t) object_bytes;
        uint64_t relative_offset = 0;
        for (int role_idx = 0; role_idx < 3; ++role_idx) {
            layout.shapes[0].roles[role_idx] = {
                static_cast<ggml_moe_common::role>(role_idx),
                relative_offset,
                (size_t) role_bytes[role_idx],
            };
            relative_offset += role_bytes[role_idx];
        }
        if (!ggml_moe_common::object_engine_configure(layout)) {
            std::fprintf(stderr, "[moe_stream] DS4 triplet pack: common object engine rejected layout\n");
            if (g_ds4_triplet_pack.ring_inited) {
                ggml_moe_common::object_io_ring_release(g_ds4_triplet_pack.io_ring);
                g_ds4_triplet_pack.ring_inited = false;
            }
            if (g_ds4_triplet_pack.fd_direct >= 0) {
                ::close(g_ds4_triplet_pack.fd_direct);
                g_ds4_triplet_pack.fd_direct = -1;
            }
            ::close(fd);
            g_ds4_triplet_pack.fd = -1;
            g_ds4_triplet_pack.entries.clear();
            return;
        }
    }
    g_ds4_triplet_pack.enabled = true;
    g_ds4_triplet_pack.pipe_requested = pipe_requested;
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (pipe_requested &&
            (g_ds4_triplet_pack.ring_inited || g_ds4_triplet_pack.ring_init_on_worker)) {
        if (ggml_moe_common::object_engine_enforcing()) {
            ggml_moe_common::object_worker_provider provider;
            provider.init = ds4_triplet_common_worker_init;
            provider.process = ds4_triplet_common_worker_process;
            provider.cpu_affinity_env = "GGML_MOE_CPU_PIPE";
            provider.legacy_cpu_affinity_env = "GGML_MOE_STREAM_DS4_PIPE_CPUSET";
            provider.worker_role = "deepseek-pipe";
            g_ds4_triplet_pack.pipe_started = ggml_moe_common::object_engine_start_worker(provider);
            g_ds4_triplet_pack.pipe_common_worker = g_ds4_triplet_pack.pipe_started;
            g_ds4_triplet_pack.pipe_failed = !g_ds4_triplet_pack.pipe_started;
        } else {
            g_ds4_triplet_pack.pipe_worker = std::thread(ds4_triplet_pipe_worker_main);
            std::unique_lock<std::mutex> pipe_lock(g_ds4_triplet_pack.pipe_mu);
            g_ds4_triplet_pack.pipe_cv.wait(pipe_lock, [] {
                return g_ds4_triplet_pack.pipe_started;
            });
        }
        if (g_ds4_triplet_pack.pipe_failed) {
            std::fprintf(stderr, "[moe_stream] DS4 triplet pipe: worker init failed; wait-all fallback active\n");
        }
    }
#endif
    std::atexit(ds4_triplet_report_atexit);
    std::fprintf(stderr,
            "[moe_stream] DS4 triplet pack: loaded %zu fixed objects from %s, object=%.2f MiB direct=%d\n",
            g_ds4_triplet_pack.entries.size(), path,
            object_bytes / (1024.0 * 1024.0), g_ds4_triplet_pack.fd_direct >= 0 ? 1 : 0);
}

static bool ds4_triplet_tensor_role(const char * tensor, int & layer, int & role) {
    if (!tensor) return false;
    char role_name[16] = {};
    int consumed = 0;
    if (std::sscanf(tensor, "blk.%d.ffn_%15[a-z]_exps.weight%n", &layer, role_name, &consumed) != 2 ||
            tensor[consumed] != '\0' || layer < 0 || layer >= 43) return false;
    if (std::strcmp(role_name, "gate") == 0) role = 0;
    else if (std::strcmp(role_name, "up") == 0) role = 1;
    else if (std::strcmp(role_name, "down") == 0) role = 2;
    else return false;
    return true;
}

extern "C" const void * ggml_cuda_moe_stream_ds4_triplet_host_role_ptr(
        const char * src0_name,
        int64_t expert_index,
        size_t role_nbytes) {
    const char * enabled = std::getenv("GGML_MOE_STREAM_DS4_TRIPLET_CPU_DOWN_HANDOFF");
    if (!enabled || !enabled[0] || enabled[0] == '0' || expert_index < 0 || expert_index >= 256) {
        return nullptr;
    }
    ds4_triplet_init_once();
    int layer = -1;
    int role = -1;
    if (!g_ds4_triplet_pack.enabled || !ds4_triplet_tensor_role(src0_name, layer, role) ||
            role != 2 || role_nbytes != g_ds4_triplet_pack.role_bytes[2]) {
        return nullptr;
    }
    g_ds4_triplet_pack.host_handoff_lookups.fetch_add(1, std::memory_order_relaxed);
    const uint64_t generation =
        g_ds4_triplet_pack.entries[(size_t) layer * 256 + (size_t) expert_index].generation;
    std::lock_guard<std::mutex> lock(g_ds4_triplet_host_handoff_mu);
    for (const ds4_triplet_host_handoff_entry & entry : g_ds4_triplet_host_handoffs) {
        if (entry.object && entry.generation == generation) {
            g_ds4_triplet_pack.host_handoff_hits.fetch_add(1, std::memory_order_relaxed);
            return (const char *) entry.object + g_ds4_triplet_pack.role_bytes[0] +
                g_ds4_triplet_pack.role_bytes[1];
        }
    }
    return nullptr;
}

static const ds4_triplet_pack_entry * ds4_triplet_lookup(
        const char * tensor, int64_t expert, size_t role_nbytes, int & role) {
    ds4_triplet_init_once();
    int layer = -1;
    if (!g_ds4_triplet_pack.enabled || expert < 0 || expert >= 256 ||
            !ds4_triplet_tensor_role(tensor, layer, role) ||
            role_nbytes != g_ds4_triplet_pack.role_bytes[role]) {
        g_ds4_triplet_pack.misses.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    const ds4_triplet_pack_entry & entry = g_ds4_triplet_pack.entries[(size_t) layer * 256 + (size_t) expert];
    g_ds4_triplet_pack.hits.fetch_add(1, std::memory_order_relaxed);
    return &entry;
}

static bool ds4_triplet_read_raw(const ds4_triplet_pack_entry & entry, void * dst) {
    bool ok = false;
    if (g_ds4_triplet_pack.fd_direct >= 0) {
        ok = one_pack_read_exact_fd(g_ds4_triplet_pack.fd_direct, dst, (size_t) entry.nbytes, entry.offset);
    }
    if (!ok) {
        ok = one_pack_read_exact_fd(g_ds4_triplet_pack.fd, dst, (size_t) entry.nbytes, entry.offset);
    }
    if (!ok) {
        g_ds4_triplet_pack.failures.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    g_ds4_triplet_pack.reads.fetch_add(1, std::memory_order_relaxed);
    g_ds4_triplet_pack.bytes.fetch_add(entry.nbytes, std::memory_order_relaxed);
    return true;
}

static bool ds4_triplet_read(const ds4_triplet_pack_entry & entry, void * dst) {
    const ds4_triplet_pack_entry * entries[8] = {&entry};
    ggml_moe_common::object_request requests[8];
    const uint8_t role_mask = ggml_moe_common::role_bit(ggml_moe_common::role::gate) |
        ggml_moe_common::role_bit(ggml_moe_common::role::up) |
        ggml_moe_common::role_bit(ggml_moe_common::role::down);
    const bool common_started = ds4_triplet_common_begin(entries, 1, role_mask, requests);
    const bool ok = ds4_triplet_read_raw(entry, dst);
    ds4_triplet_common_complete(requests, 1, common_started, ok);
    return ok;
}

static bool ds4_ram_pilot_prefetch_enabled() {
    const char * env = std::getenv("GGML_MOE_STREAM_DS4_RAM_PILOT_PREFETCH");
    return env && env[0] && env[0] != '0';
}

static int ds4_ram_pilot_prefetch_topk() {
    const char * env = std::getenv("GGML_MOE_STREAM_DS4_RAM_PILOT_TOPK");
    const long value = env && env[0] ? std::strtol(env, nullptr, 10) : 3L;
    return (int) std::max(1L, std::min(value, 8L));
}

static void ds4_ram_pilot_prefetch_worker() {
    void * buffer = nullptr;
    if (posix_memalign(&buffer, 4096, (size_t) g_ds4_triplet_pack.object_bytes) != 0 || !buffer) {
        std::lock_guard<std::mutex> lock(g_ds4_ram_prefetch.mu);
        g_ds4_ram_prefetch.failed = true;
        return;
    }
    for (;;) {
        uint32_t ordinal = 0;
        {
            std::unique_lock<std::mutex> lock(g_ds4_ram_prefetch.mu);
            g_ds4_ram_prefetch.cv.wait(lock, [] {
                return g_ds4_ram_prefetch.stop || !g_ds4_ram_prefetch.queue.empty();
            });
            if (g_ds4_ram_prefetch.stop && g_ds4_ram_prefetch.queue.empty()) break;
            ordinal = g_ds4_ram_prefetch.queue.front();
            g_ds4_ram_prefetch.queue.pop_front();
        }
        if (ordinal >= g_ds4_triplet_pack.entries.size()) continue;
        const ds4_triplet_pack_entry & entry = g_ds4_triplet_pack.entries[ordinal];
        const uintptr_t key = (uintptr_t) entry.generation;
        bool stored = false;
        if (ds4_triplet_read(entry, buffer)) {
            std::lock_guard<std::mutex> cache_lock(g_vcache.mu);
            if (ram_cache_find_slot_locked(key) >= 0) {
                stored = true;
            } else {
                stored = ram_cache_store_host_locked(key, buffer, (size_t) entry.nbytes, -1);
            }
        } else {
            g_ds4_ram_prefetch.read_failures.fetch_add(1, std::memory_order_relaxed);
        }
        {
            std::lock_guard<std::mutex> lock(g_ds4_ram_prefetch.mu);
            g_ds4_ram_prefetch.queued_keys.erase(key);
            if (stored) {
                g_ds4_ram_prefetch.ready_keys.insert(key);
            }
        }
        if (stored) {
            g_ds4_ram_prefetch.completed.fetch_add(1, std::memory_order_relaxed);
            g_ds4_ram_prefetch.bytes.fetch_add(entry.nbytes, std::memory_order_relaxed);
        } else {
            g_ds4_ram_prefetch.wasted.fetch_add(1, std::memory_order_relaxed);
        }
    }
    std::free(buffer);
}

static bool ds4_ram_pilot_prefetch_start() {
    if (!ds4_ram_pilot_prefetch_enabled() || !g_vcache.ram_cache.slab.storage.host ||
            g_vcache.ram_cache.slots.empty()) return false;
    std::lock_guard<std::mutex> lock(g_ds4_ram_prefetch.mu);
    if (!g_ds4_ram_prefetch.started) {
        g_ds4_ram_prefetch.started = true;
        g_ds4_ram_prefetch.worker = std::thread(ds4_ram_pilot_prefetch_worker);
    }
    return !g_ds4_ram_prefetch.failed;
}

static void ds4_ram_pilot_prefetch_enqueue(int layer, int expert) {
    if (!ds4_ram_pilot_prefetch_start() || layer < 0 || layer >= 43 || expert < 0 || expert >= 256) return;
    const uint32_t ordinal = (uint32_t) layer * 256U + (uint32_t) expert;
    const ds4_triplet_pack_entry & entry = g_ds4_triplet_pack.entries[ordinal];
    const uintptr_t key = (uintptr_t) entry.generation;
    {
        std::lock_guard<std::mutex> cache_lock(g_vcache.mu);
        bool resident = ram_cache_find_slot_locked(key) >= 0;
        for (int slot = 0; !resident && slot < g_vcache.n_slots; ++slot) {
            resident = g_vcache.slot_key[slot] == key;
        }
        if (resident) {
            g_ds4_ram_prefetch.resident_skips.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    {
        std::lock_guard<std::mutex> lock(g_ds4_ram_prefetch.mu);
        if (g_ds4_ram_prefetch.queued_keys.count(key) || g_ds4_ram_prefetch.ready_keys.count(key)) {
            g_ds4_ram_prefetch.duplicate_skips.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (g_ds4_ram_prefetch.queue.size() >= 64) {
            g_ds4_ram_prefetch.wasted.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        g_ds4_ram_prefetch.queue.push_back(ordinal);
        g_ds4_ram_prefetch.queued_keys.insert(key);
    }
    g_ds4_ram_prefetch.submitted.fetch_add(1, std::memory_order_relaxed);
    g_ds4_ram_prefetch.cv.notify_one();
}

static bool ds4_triplet_read_batch(
        const ds4_triplet_pack_entry * const * entries,
        void * const * destinations,
        int count) {
    if (!entries || !destinations || count <= 0 || count > 8) return false;
    ggml_moe_common::object_request common_requests[8];
    const uint8_t role_mask = ggml_moe_common::role_bit(ggml_moe_common::role::gate) |
        ggml_moe_common::role_bit(ggml_moe_common::role::up) |
        ggml_moe_common::role_bit(ggml_moe_common::role::down);
    const bool common_started = ds4_triplet_common_begin(entries, count, role_mask, common_requests);
    bool completed[8] = {};
    ggml_moe_common::object_io_window_slot io_window[8] = {};
    const bool common_object_cqe = ds4_common_object_cqe_enabled();
    bool iouring_attempted = false;

#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (g_ds4_triplet_pack.ring_inited && g_ds4_triplet_pack.fd_direct >= 0) {
        iouring_attempted = true;
        std::lock_guard<std::mutex> io_lock(g_ds4_triplet_pack.io_mu);
        const auto submit_start = std::chrono::steady_clock::now();
        int prepared = 0;
        for (; prepared < count; ++prepared) {
            bool read_prepared = false;
            if (common_object_cqe) {
                const ggml_moe_common::object_io_window_request read_request{
                    g_ds4_triplet_pack.fd_direct,
                    destinations[prepared],
                    (size_t) entries[prepared]->nbytes,
                    entries[prepared]->offset,
                    (size_t) prepared,
                    false,
                };
                read_prepared = ggml_moe_common::object_io_window_prepare_read(
                    g_ds4_triplet_pack.io_ring, io_window, 8,
                    (size_t) prepared, read_request);
            } else {
                read_prepared = ggml_moe_common::object_io_ring_prepare_read(
                    g_ds4_triplet_pack.io_ring,
                    g_ds4_triplet_pack.fd_direct,
                    destinations[prepared],
                    (size_t) entries[prepared]->nbytes,
                    entries[prepared]->offset,
                    (uint64_t) prepared);
            }
            if (!read_prepared) {
                break;
            }
        }
        const int submitted = prepared > 0 ?
            ggml_moe_common::object_io_ring_submit_count(g_ds4_triplet_pack.io_ring) : 0;
        const auto submit_done = std::chrono::steady_clock::now();
        g_ds4_triplet_pack.io_stats.submit_us.fetch_add((uint64_t)
            std::chrono::duration_cast<std::chrono::microseconds>(submit_done - submit_start).count());
        if (submitted > 0) {
            const auto wait_start = std::chrono::steady_clock::now();
            for (int seen = 0; seen < submitted; ++seen) {
                ggml_moe_common::object_io_completion completion;
                if (!ggml_moe_common::object_io_ring_wait(g_ds4_triplet_pack.io_ring, completion)) break;
                size_t index = 0;
                ggml_moe_common::object_io_completion_status status;
                if (common_object_cqe) {
                    ggml_moe_common::object_io_window_completion resolved;
                    status = ggml_moe_common::object_io_window_resolve(
                        g_ds4_triplet_pack.io_ring, io_window, 8,
                        completion, resolved);
                    index = resolved.request_id;
                } else {
                    const bool index_valid = completion.user_data < (uint64_t) count;
                    index = index_valid ? (size_t) completion.user_data : 0;
                    const size_t expected_bytes = index_valid ? (size_t) entries[index]->nbytes : 0;
                    status = ggml_moe_common::object_io_completion_validate(
                        g_ds4_triplet_pack.io_ring, completion, index_valid, expected_bytes);
                }
                if (status == ggml_moe_common::object_io_completion_status::complete &&
                        index < (size_t) count) {
                    completed[index] = true;
                    g_ds4_triplet_pack.reads.fetch_add(1, std::memory_order_relaxed);
                    g_ds4_triplet_pack.bytes.fetch_add(entries[index]->nbytes, std::memory_order_relaxed);
                    g_ds4_triplet_pack.io_stats.reads.fetch_add(1, std::memory_order_relaxed);
                }
            }
            const auto wait_done = std::chrono::steady_clock::now();
            g_ds4_triplet_pack.io_stats.wait_us.fetch_add((uint64_t)
                std::chrono::duration_cast<std::chrono::microseconds>(wait_done - wait_start).count());
            g_ds4_triplet_pack.io_stats.batches.fetch_add(1, std::memory_order_relaxed);
        }
        if (common_object_cqe) ggml_moe_common::object_io_window_reset(io_window, 8);
    }
#endif

    bool ok = true;
    for (int i = 0; i < count; ++i) {
        if (completed[i]) continue;
        if (iouring_attempted) g_ds4_triplet_pack.io_stats.fallbacks.fetch_add(1, std::memory_order_relaxed);
        if (!ds4_triplet_read_raw(*entries[i], destinations[i])) ok = false;
    }
    ds4_triplet_common_complete(common_requests, count, common_started, ok);
    return ok;
}

static const ggml_moe_common::lossless_object_record * ds4_lossless_object(
        const ds4_triplet_pack_entry &entry) {
    if (!g_ds4_triplet_pack.lossless_ready ||
            g_ds4_triplet_pack.lossless_failed) {
        return nullptr;
    }
    return ggml_moe_common::lossless_sidecar_find(
        g_ds4_triplet_pack.lossless_sidecar,
        entry.layer,
        entry.expert);
}

static bool ds4_lossless_trust_verified() {
    const char *env = std::getenv("GGML_MOE_LOSSLESS_ASYNC_TRUST_VERIFIED");
    return env && env[0] && env[0] != '0';
}

static size_t ds4_lossless_min_batch() {
    static const size_t value = [] {
        const char *env = std::getenv("GGML_MOE_DS4_LOSSLESS_MIN_BATCH");
        if (!env || !env[0]) return size_t{1};
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(env, &end, 10);
        return end != env && *end == '\0' ?
            std::clamp<size_t>(parsed, 1, 8) : size_t{1};
    }();
    return value;
}

static bool ds4_lossless_decode_all_requested() {
    const char *env = std::getenv("GGML_MOE_DS4_LOSSLESS_DECODE_ALL");
    return !env || !env[0] || env[0] != '0';
}

static size_t ds4_lossless_eager_decode_objects() {
    static const size_t value = [] {
        const char *env = std::getenv(
            "GGML_MOE_DS4_LOSSLESS_EAGER_DECODE_OBJECTS");
        if (!env || !env[0]) return size_t{0};
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(env, &end, 10);
        if (end == env || *end != '\0' || parsed == 0) return size_t{0};
        return std::clamp<size_t>(parsed, 1, 8);
    }();
    return value;
}

static bool ds4_lossless_append_role_descriptor(
        const ggml_moe_common::lossless_object_record &object,
        const void *host_payload,
        const uint8_t *mapped_host_payload,
        uint8_t *device_input,
        int role_index,
        void *destination,
        bool direct_quant,
        bool pitched_quant,
        bool planar_quant,
        bool compact_planar_direct,
        uint64_t &copied_bytes,
        std::vector<ggml_moe_common::lossless_mxfp4_decode_descriptor> &descriptors) {
    if (!host_payload || !device_input || !destination ||
            role_index < 0 || role_index >= 3) {
        return false;
    }
    const ggml_moe_common::lossless_role_record &role = object.roles[role_index];
    if (role.codec ==
            ggml_moe_common::lossless_role_codec::compact_mxfp4_scale3) {
        const bool sparse_exceptions = role.scale_codec ==
            ggml_moe_common::lossless_q40_scale_codec::
                fixed3_e8m0_group8_exceptions;
        if (sparse_exceptions && compact_planar_direct) return false;
        const uint64_t blocks = static_cast<uint64_t>(role.rows) *
            role.blocks_per_row;
        const bool planar_storage =
            role.scales_offset == blocks * 16;
        const uint8_t *host_role =
            static_cast<const uint8_t *>(host_payload) + role.payload_offset;
        uint8_t *device_role = device_input + role.payload_offset;
        if ((role.scale_codec !=
                ggml_moe_common::lossless_q40_scale_codec::fixed3_e8m0_group8 &&
             !sparse_exceptions) ||
                role.payload_bytes == 0 ||
                role.payload_offset > object.payload_bytes ||
                role.payload_bytes > object.payload_bytes - role.payload_offset) {
            return false;
        }
        if (planar_storage) {
            if (role.scales_bytes == 0 ||
                    role.scales_offset > role.payload_bytes ||
                    role.scales_bytes >
                        role.payload_bytes - role.scales_offset ||
                    (!sparse_exceptions && role.scales_bytes !=
                        role.payload_bytes - role.scales_offset) ||
                    (sparse_exceptions &&
                        (role.chunk_index_offset !=
                            role.scales_offset + role.scales_bytes ||
                         role.chunk_index_bytes !=
                            static_cast<uint64_t>(role.scale_records) * 8 ||
                         role.chunk_index_bytes != role.payload_bytes -
                            role.chunk_index_offset))) {
                return false;
            }
            if (cudaMemcpyAsync(
                        destination, host_role, role.scales_offset,
                        cudaMemcpyHostToDevice,
                        g_ds4_triplet_pack.pipe_stream) != cudaSuccess) {
                return false;
            }
            uint8_t *scale_destination = compact_planar_direct ?
                static_cast<uint8_t *>(destination) + role.scales_offset :
                device_role + role.scales_offset;
            if (cudaMemcpyAsync(
                        scale_destination,
                        host_role + role.scales_offset,
                        role.scales_bytes,
                        cudaMemcpyHostToDevice,
                        g_ds4_triplet_pack.pipe_stream) != cudaSuccess) {
                return false;
            }
            if (sparse_exceptions && role.chunk_index_bytes != 0 &&
                    cudaMemcpyAsync(
                        device_role + role.chunk_index_offset,
                        host_role + role.chunk_index_offset,
                        role.chunk_index_bytes,
                        cudaMemcpyHostToDevice,
                        g_ds4_triplet_pack.pipe_stream) != cudaSuccess) {
                return false;
            }
            if (compact_planar_direct) {
                copied_bytes += role.payload_bytes;
                return true;
            }
        } else if (cudaMemcpyAsync(
                device_role, host_role, role.payload_bytes,
                cudaMemcpyHostToDevice,
                g_ds4_triplet_pack.pipe_stream) != cudaSuccess) {
            return false;
        }
        ggml_moe_common::lossless_mxfp4_decode_descriptor descriptor;
        descriptor.records[0] = device_role +
            (planar_storage ? role.scales_offset : 0);
        descriptor.record_bytes[0] = planar_storage ?
            role.scales_bytes :
            (sparse_exceptions ? role.chunk_index_offset : role.payload_bytes);
        descriptor.destination = static_cast<uint8_t *>(destination);
        descriptor.destination_bytes = role.raw_bytes;
        descriptor.blocks = blocks;
        descriptor.role_index = static_cast<uint32_t>(role_index);
        descriptor.chunks = 1;
        descriptor.scale_records_count = 0;
        if (sparse_exceptions) {
            descriptor.scale_records[0] = device_role +
                role.chunk_index_offset;
            descriptor.scale_record_bytes[0] = role.chunk_index_bytes;
            descriptor.scale_records_count = role.scale_records;
        }
        descriptor.scale_codec = role.scale_codec;
        descriptor.quant_context_mode = planar_storage ? 6 : 5;
        descriptors.push_back(descriptor);
        copied_bytes += role.payload_bytes;
        return true;
    }
    std::array<ggml_moe_common::lossless_chunk_record,
        ggml_moe_common::lossless_rans_mxfp4_max_chunks> chunks{};
    std::array<ggml_moe_common::lossless_scale_record,
        ggml_moe_common::lossless_rans_mxfp4_max_scale_records> scale_records{};
    const bool supported_scale_codec = role.scale_codec ==
            ggml_moe_common::lossless_q40_scale_codec::rans_e8m0_nibble_planes ||
        role.scale_codec ==
            ggml_moe_common::lossless_q40_scale_codec::sparse_e8m0_binary;
    if (role.codec != ggml_moe_common::lossless_role_codec::rans_mxfp4 ||
            !supported_scale_codec ||
            role.chunks == 0 ||
            role.chunks > ggml_moe_common::lossless_rans_mxfp4_max_chunks ||
            !ggml_moe_common::lossless_sidecar_parse_chunks(
                role,
                static_cast<const uint8_t *>(host_payload),
                object.payload_bytes,
                chunks.data(),
                chunks.size()) ||
            !ggml_moe_common::lossless_sidecar_parse_scale_records(
                role,
                static_cast<const uint8_t *>(host_payload),
                object.payload_bytes,
                scale_records.data(),
                scale_records.size())) {
        return false;
    }
    ggml_moe_common::lossless_mxfp4_decode_descriptor descriptor;
    descriptor.chunks = role.chunks;
    for (uint32_t chunk = 0; chunk < role.chunks; ++chunk) {
        descriptor.records[chunk] = device_input +
            role.payload_offset + chunks[chunk].record_offset;
        descriptor.record_bytes[chunk] = chunks[chunk].record_bytes;
        descriptor.quant_byte_offsets[chunk] =
            chunks[chunk].quant_byte_offset;
        descriptor.packed_bytes[chunk] = chunks[chunk].packed_bytes;
    }
    for (uint32_t record = 0; record < role.scale_records; ++record) {
        descriptor.scale_records[record] = device_input +
            role.payload_offset + scale_records[record].record_offset;
        descriptor.scale_record_bytes[record] =
            scale_records[record].record_bytes;
    }
    descriptor.destination = static_cast<uint8_t *>(destination);
    descriptor.destination_bytes = role.raw_bytes;
    descriptor.blocks = static_cast<uint64_t>(role.rows) *
        role.blocks_per_row;
    descriptor.role_index = static_cast<uint32_t>(role_index);
    descriptor.scale_records_count = role.scale_records;
    descriptor.scale_codec = role.scale_codec;
    if (direct_quant || pitched_quant || planar_quant) {
        constexpr size_t quant_bytes_per_block = 16;
        constexpr size_t native_bytes_per_block = 17;
        uint64_t expected_quant_offset = 0;
        uint64_t expected_record_offset = chunks[0].record_offset;
        for (uint32_t chunk = 0; chunk < role.chunks; ++chunk) {
            if (chunks[chunk].record_bytes != chunks[chunk].packed_bytes ||
                    chunks[chunk].quant_byte_offset != expected_quant_offset ||
                    chunks[chunk].record_offset != expected_record_offset) {
                return false;
            }
            expected_quant_offset += chunks[chunk].packed_bytes;
            expected_record_offset += chunks[chunk].record_bytes;
        }
        if (expected_quant_offset != descriptor.blocks * quant_bytes_per_block ||
                expected_record_offset != role.scales_offset ||
                role.raw_bytes != descriptor.blocks * 17) {
            return false;
        }
        if (direct_quant && (!mapped_host_payload ||
                cudaMemcpyAsync(
                    device_input + role.payload_offset + role.scales_offset,
                    static_cast<const uint8_t *>(host_payload) +
                        role.payload_offset + role.scales_offset,
                    role.scales_bytes,
                    cudaMemcpyHostToDevice,
                    g_ds4_triplet_pack.pipe_stream) != cudaSuccess)) {
            return false;
        }
        if (direct_quant) {
            for (uint32_t chunk = 0; chunk < role.chunks; ++chunk) {
                descriptor.records[chunk] = mapped_host_payload +
                    role.payload_offset + chunks[chunk].record_offset;
            }
            descriptor.quant_context_mode = 2;
        } else if (pitched_quant) {
            const uint8_t *host_role =
                static_cast<const uint8_t *>(host_payload) +
                role.payload_offset;
            if (cudaMemcpy2DAsync(
                        static_cast<uint8_t *>(destination) + 1,
                        native_bytes_per_block,
                        host_role + chunks[0].record_offset,
                        quant_bytes_per_block,
                        quant_bytes_per_block,
                        descriptor.blocks,
                        cudaMemcpyHostToDevice,
                        g_ds4_triplet_pack.pipe_stream) != cudaSuccess ||
                    cudaMemcpyAsync(
                        device_input + role.payload_offset +
                            role.scales_offset,
                        host_role + role.scales_offset,
                        role.scales_bytes,
                        cudaMemcpyHostToDevice,
                        g_ds4_triplet_pack.pipe_stream) != cudaSuccess) {
                return false;
            }
            descriptor.quant_context_mode = 4;
        } else {
            const uint8_t *host_role =
                static_cast<const uint8_t *>(host_payload) +
                role.payload_offset;
            if (cudaMemcpyAsync(
                    destination,
                    host_role + chunks[0].record_offset,
                    expected_quant_offset,
                    cudaMemcpyHostToDevice,
                    g_ds4_triplet_pack.pipe_stream) != cudaSuccess ||
                    cudaMemcpyAsync(
                        device_input + role.payload_offset + role.scales_offset,
                        host_role + role.scales_offset,
                        role.scales_bytes,
                        cudaMemcpyHostToDevice,
                        g_ds4_triplet_pack.pipe_stream) != cudaSuccess) {
                return false;
            }
            descriptor.quant_context_mode = 4;
        }
        copied_bytes += expected_quant_offset + role.scales_bytes;
    } else {
        descriptor.quant_context_mode =
            g_ds4_triplet_pack.lossless_sidecar.quant_context_mode;
    }
    descriptors.push_back(descriptor);
    return true;
}

static bool ds4_lossless_enqueue_object(
        ds4_triplet_pipe_batch &batch,
        int object_index,
        const ggml_moe_common::lossless_object_record &object,
        std::vector<ggml_moe_common::lossless_mxfp4_decode_descriptor> &critical,
        std::vector<ggml_moe_common::lossless_mxfp4_decode_descriptor> &deferred) {
    if (object_index < 0 || object_index >= batch.count ||
            !batch.host[object_index] || object.payload_bytes == 0 ||
            object.payload_bytes >
                g_ds4_triplet_pack.lossless_device_input_stride) {
        return false;
    }
    uint8_t *device_input = g_ds4_triplet_pack.lossless_device_input +
        static_cast<size_t>(object_index) *
            g_ds4_triplet_pack.lossless_device_input_stride;
    const ggml_moe_common::lossless_object_record *objects_begin =
        g_ds4_triplet_pack.lossless_sidecar.objects.data();
    const size_t sidecar_object_index =
        static_cast<size_t>(&object - objects_begin);
    if (sidecar_object_index >=
            g_ds4_triplet_pack.lossless_sidecar.objects.size() ||
            sidecar_object_index >=
                g_ds4_triplet_pack.lossless_crc_verified.size()) {
        return false;
    }
    const bool trust_verified = ds4_lossless_trust_verified();
    const bool crc_required = !trust_verified &&
        g_ds4_triplet_pack.lossless_crc_verified[sidecar_object_index] == 0;
    const ggml_moe_common::dependency_transfer_plan &plan =
        batch.transfer_plans[object_index];
    const bool decode_all = ds4_lossless_decode_all_requested();
    const char *direct_env =
        std::getenv("GGML_MOE_DS4_LOSSLESS_DIRECT_QUANT_H2D");
    const bool direct_requested = direct_env && direct_env[0] &&
        direct_env[0] != '0';
    const bool direct_quant = direct_requested && decode_all &&
        g_ds4_triplet_pack.lossless_sidecar.quant_context_mode == 2;
    const char *pitched_env =
        std::getenv("GGML_MOE_DS4_LOSSLESS_PITCHED_QUANT_H2D");
    const bool pitched_quant = !direct_quant && decode_all &&
        ds4_lossless_trust_verified() && pitched_env && pitched_env[0] &&
        pitched_env[0] != '0' &&
        g_ds4_triplet_pack.lossless_sidecar.quant_context_mode == 2;
    const char *planar_env =
        std::getenv("GGML_MOE_DS4_LOSSLESS_PLANAR_RESIDENCY");
    const char *fast_mmvq_env =
        std::getenv("GGML_MOE_STREAM_DOWN_Q80_FAST_MMVQ");
    const bool fast_mmvq = fast_mmvq_env && fast_mmvq_env[0] &&
        fast_mmvq_env[0] != '0';
    const bool planar_quant = !direct_quant && !pitched_quant &&
        ds4_lossless_trust_verified() && fast_mmvq &&
        planar_env && planar_env[0] &&
        planar_env[0] != '0' &&
        g_ds4_triplet_pack.lossless_sidecar.quant_context_mode == 2;
    const bool compact_scale3 =
        object.roles[0].codec ==
            ggml_moe_common::lossless_role_codec::compact_mxfp4_scale3 &&
        object.roles[1].codec ==
            ggml_moe_common::lossless_role_codec::compact_mxfp4_scale3 &&
        object.roles[2].codec ==
            ggml_moe_common::lossless_role_codec::compact_mxfp4_scale3;
    const bool nvcomp_rank =
        object.roles[0].codec ==
            ggml_moe_common::lossless_role_codec::nvcomp_mxfp4_rank &&
        object.roles[1].codec ==
            ggml_moe_common::lossless_role_codec::nvcomp_mxfp4_rank &&
        object.roles[2].codec ==
            ggml_moe_common::lossless_role_codec::nvcomp_mxfp4_rank;
    const uint64_t first_role_blocks =
        static_cast<uint64_t>(object.roles[0].rows) *
        object.roles[0].blocks_per_row;
    const bool compact_scale3_planar = compact_scale3 &&
        object.roles[0].scales_offset == first_role_blocks * 16;
    const char *compact_direct_env = std::getenv(
        "GGML_MOE_DS4_LOSSLESS_SCALE3_DIRECT_RESIDENCY");
    const bool compact_planar_direct = compact_scale3_planar &&
        compact_direct_env && compact_direct_env[0] &&
        compact_direct_env[0] != '0';
    if ((compact_scale3 || nvcomp_rank) && !trust_verified) {
        return false;
    }
    void *mapped_host_device = nullptr;
    if (direct_quant && cudaHostGetDevicePointer(
            &mapped_host_device, batch.host[object_index], 0) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    const uint8_t *mapped_host_payload =
        static_cast<const uint8_t *>(mapped_host_device);
    const uint8_t verified_roles =
        g_ds4_triplet_pack.lossless_crc_verified[sidecar_object_index];
    const uint8_t direct_crc_mask = direct_quant && !trust_verified ?
        static_cast<uint8_t>(
            plan.missing_role_mask & static_cast<uint8_t>(~verified_roles)) : 0;
    if (direct_quant) {
        for (int role = 0; role < 3; ++role) {
            const uint8_t bit = ggml_moe_common::role_bit(
                static_cast<ggml_moe_common::role>(role));
            if ((plan.missing_role_mask & bit) != 0 &&
                    object.roles[role].raw_crc32 == 0) {
                return false;
            }
        }
    } else if (!pitched_quant && !planar_quant && !compact_scale3 &&
            cudaMemcpyAsync(
            device_input,
            batch.host[object_index],
            object.payload_bytes,
            cudaMemcpyHostToDevice,
            g_ds4_triplet_pack.pipe_stream) != cudaSuccess) {
        return false;
    }
    if ((direct_quant && direct_crc_mask == 0) ||
            (!direct_quant && !crc_required)) {
        g_ds4_triplet_pack.lossless_crc_reuses.fetch_add(
            1, std::memory_order_relaxed);
    }
    const size_t critical_begin = critical.size();
    const size_t deferred_begin = deferred.size();
    uint64_t copied_bytes = 0;
    if (nvcomp_rank) {
        if (!decode_all || !g_ds4_triplet_pack.lossless_nvcomp_decoder ||
                plan.missing_role_mask == 0) {
            return false;
        }
        batch.lossless_nvcomp[object_index] = true;
        batch.lossless_nvcomp_role_masks[object_index] =
            plan.missing_role_mask;
        copied_bytes = object.payload_bytes;
    } else {
        for (uint8_t copy_index = 0;
                copy_index < plan.critical_count;
                ++copy_index) {
            const int role_index = static_cast<int>(
                plan.critical[copy_index].kind);
            if (!ds4_lossless_append_role_descriptor(
                    object, batch.host[object_index], mapped_host_payload,
                    device_input, role_index,
                    batch.role_ptrs[object_index][role_index], direct_quant,
                    pitched_quant,
                    planar_quant, compact_planar_direct, copied_bytes,
                    critical)) {
                critical.resize(critical_begin);
                deferred.resize(deferred_begin);
                return false;
            }
        }
        for (uint8_t copy_index = 0;
                copy_index < plan.deferred_count;
                ++copy_index) {
            const int role_index = static_cast<int>(
                plan.deferred[copy_index].kind);
            if (!ds4_lossless_append_role_descriptor(
                    object, batch.host[object_index], mapped_host_payload,
                    device_input, role_index,
                    batch.role_ptrs[object_index][role_index], direct_quant,
                    pitched_quant,
                    planar_quant, compact_planar_direct, copied_bytes,
                    decode_all ? critical : deferred)) {
                critical.resize(critical_begin);
                deferred.resize(deferred_begin);
                return false;
            }
        }
    }
    batch.lossless_decode_all = batch.lossless_decode_all || decode_all;
    batch.lossless_objects[object_index] = true;
    batch.lossless_planar[object_index] = planar_quant ||
        (compact_scale3_planar && !compact_planar_direct);
    // Group-interleaved scale3 uses layout 2; planar scale3 uses layout 3.
    batch.lossless_scale3[object_index] = false;
    batch.lossless_planar_scale3[object_index] = compact_planar_direct;
    batch.lossless_direct_quant[object_index] = direct_quant;
    batch.lossless_crc_required[object_index] = direct_quant ?
        direct_crc_mask != 0 : crc_required;
    batch.lossless_crc_role_masks[object_index] = direct_crc_mask;
    batch.lossless_object_indices[object_index] = sidecar_object_index;
    if (!direct_quant && !planar_quant) copied_bytes = object.payload_bytes;
    g_ds4_triplet_pack.lossless_h2d_bytes.fetch_add(
        copied_bytes, std::memory_order_relaxed);
    g_ds4_triplet_pack.pipe_h2d_bytes.fetch_add(
        copied_bytes, std::memory_order_relaxed);
    return true;
}

static bool ds4_lossless_nvcomp_descriptor(
        ds4_triplet_pipe_batch &batch,
        int object_index,
        ggml_moe_common::lossless_nvcomp_object_descriptor &descriptor) {
    if (!g_ds4_triplet_pack.lossless_nvcomp_decoder || object_index < 0 ||
            object_index >= batch.count ||
            !batch.lossless_nvcomp[object_index]) {
        return false;
    }
    const size_t sidecar_object_index =
        batch.lossless_object_indices[object_index];
    if (sidecar_object_index >=
            g_ds4_triplet_pack.lossless_sidecar.objects.size()) {
        return false;
    }
    const auto &object =
        g_ds4_triplet_pack.lossless_sidecar.objects[sidecar_object_index];
    descriptor.compressed = g_ds4_triplet_pack.lossless_device_input +
        static_cast<size_t>(object_index) *
            g_ds4_triplet_pack.lossless_device_input_stride;
    descriptor.role_mask =
        batch.lossless_nvcomp_role_masks[object_index];
    for (size_t role = 0; role < 3; ++role) {
        descriptor.destinations[role] = static_cast<uint8_t *>(
            batch.role_ptrs[object_index][role]);
        descriptor.blocks[role] =
            static_cast<uint64_t>(object.roles[role].rows) *
            object.roles[role].blocks_per_row;
    }
    return true;
}

static bool ds4_lossless_enqueue_crc_batch(ds4_triplet_pipe_batch &batch) {
    std::vector<ggml_moe_common::lossless_crc32_descriptor> descriptors;
    descriptors.reserve(batch.count);
    for (int object_index = 0; object_index < batch.count; ++object_index) {
        if (!batch.lossless_objects[object_index] ||
                batch.lossless_direct_quant[object_index] ||
                !batch.lossless_crc_required[object_index]) {
            continue;
        }
        const size_t sidecar_object_index =
            batch.lossless_object_indices[object_index];
        if (sidecar_object_index >=
                g_ds4_triplet_pack.lossless_sidecar.objects.size()) {
            return false;
        }
        const ggml_moe_common::lossless_object_record &object =
            g_ds4_triplet_pack.lossless_sidecar.objects[sidecar_object_index];
        const size_t crc_slot = descriptors.size();
        batch.lossless_crc_slots[object_index] = crc_slot;
        descriptors.push_back({
            g_ds4_triplet_pack.lossless_device_input +
                static_cast<size_t>(object_index) *
                    g_ds4_triplet_pack.lossless_device_input_stride,
            object.payload_bytes,
        });
    }
    if (descriptors.empty()) return true;
    const size_t partial_count =
        descriptors.size() * ggml_moe_common::lossless_crc32_segments;
    if (partial_count > g_ds4_triplet_pack.lossless_crc32_capacity ||
            !ggml_moe_common::lossless_rans_cuda_crc32_batch_async(
                g_ds4_triplet_pack.lossless_critical_decoder,
                descriptors.data(), descriptors.size(),
                g_ds4_triplet_pack.pipe_stream) ||
            cudaMemcpyAsync(
                g_ds4_triplet_pack.lossless_crc32_host,
                ggml_moe_common::lossless_rans_cuda_crc32_partials_device(
                    g_ds4_triplet_pack.lossless_critical_decoder),
                partial_count * sizeof(uint32_t),
                cudaMemcpyDeviceToHost,
                g_ds4_triplet_pack.pipe_stream) != cudaSuccess) {
        return false;
    }
    return true;
}

static bool ds4_lossless_validate_payload_crc(
        ds4_triplet_pipe_batch &batch) {
    for (int object_index = 0; object_index < batch.count; ++object_index) {
        if (!batch.lossless_objects[object_index] ||
                batch.lossless_direct_quant[object_index] ||
                !batch.lossless_crc_required[object_index]) {
            continue;
        }
        const size_t sidecar_object_index =
            batch.lossless_object_indices[object_index];
        if (sidecar_object_index >=
                g_ds4_triplet_pack.lossless_sidecar.objects.size() ||
                sidecar_object_index >=
                    g_ds4_triplet_pack.lossless_crc_verified.size()) {
            return false;
        }
        const ggml_moe_common::lossless_object_record &object =
            g_ds4_triplet_pack.lossless_sidecar.objects[sidecar_object_index];
        const size_t crc_slot = batch.lossless_crc_slots[object_index];
        if (crc_slot == SIZE_MAX) return false;
        const size_t crc32_offset =
            crc_slot * ggml_moe_common::lossless_crc32_segments;
        const uint32_t actual = ggml_moe_common::lossless_crc32_combine_segments(
            g_ds4_triplet_pack.lossless_crc32_host + crc32_offset,
            object.payload_bytes);
        if (actual != object.payload_crc32) {
            g_ds4_triplet_pack.lossless_crc_failures.fetch_add(
                1, std::memory_order_relaxed);
            return false;
        }
        g_ds4_triplet_pack.lossless_crc_verified[sidecar_object_index] = 0x7;
        g_ds4_triplet_pack.lossless_crc_objects.fetch_add(
            1, std::memory_order_relaxed);
    }
    return true;
}

static bool ds4_lossless_enqueue_output_crc_batch(
        ds4_triplet_pipe_batch &batch) {
    std::vector<ggml_moe_common::lossless_crc32_descriptor> descriptors;
    descriptors.reserve(static_cast<size_t>(batch.count) * 3);
    for (int object_index = 0; object_index < batch.count; ++object_index) {
        const uint8_t crc_mask = batch.lossless_crc_role_masks[object_index];
        if (!batch.lossless_direct_quant[object_index] || crc_mask == 0) continue;
        const size_t sidecar_object_index =
            batch.lossless_object_indices[object_index];
        if (sidecar_object_index >=
                g_ds4_triplet_pack.lossless_sidecar.objects.size()) {
            return false;
        }
        const ggml_moe_common::lossless_object_record &object =
            g_ds4_triplet_pack.lossless_sidecar.objects[sidecar_object_index];
        for (int role = 0; role < 3; ++role) {
            const uint8_t bit = ggml_moe_common::role_bit(
                static_cast<ggml_moe_common::role>(role));
            if ((crc_mask & bit) == 0) continue;
            if (!batch.role_ptrs[object_index][role] ||
                    object.roles[role].raw_bytes == 0) {
                return false;
            }
            batch.lossless_crc_role_slots[object_index][role] =
                descriptors.size();
            descriptors.push_back({
                static_cast<const uint8_t *>(
                    batch.role_ptrs[object_index][role]),
                object.roles[role].raw_bytes,
            });
        }
    }
    if (descriptors.empty()) return true;
    const size_t partial_count =
        descriptors.size() * ggml_moe_common::lossless_crc32_segments;
    return partial_count <= g_ds4_triplet_pack.lossless_crc32_capacity &&
        ggml_moe_common::lossless_rans_cuda_crc32_batch_async(
            g_ds4_triplet_pack.lossless_critical_decoder,
            descriptors.data(), descriptors.size(),
            g_ds4_triplet_pack.pipe_stream) &&
        cudaMemcpyAsync(
            g_ds4_triplet_pack.lossless_crc32_host,
            ggml_moe_common::lossless_rans_cuda_crc32_partials_device(
                g_ds4_triplet_pack.lossless_critical_decoder),
            partial_count * sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            g_ds4_triplet_pack.pipe_stream) == cudaSuccess;
}

static bool ds4_lossless_validate_output_crc(
        ds4_triplet_pipe_batch &batch) {
    for (int object_index = 0; object_index < batch.count; ++object_index) {
        const uint8_t crc_mask = batch.lossless_crc_role_masks[object_index];
        if (!batch.lossless_direct_quant[object_index] || crc_mask == 0) continue;
        const size_t sidecar_object_index =
            batch.lossless_object_indices[object_index];
        if (sidecar_object_index >=
                g_ds4_triplet_pack.lossless_sidecar.objects.size() ||
                sidecar_object_index >=
                    g_ds4_triplet_pack.lossless_crc_verified.size()) {
            return false;
        }
        const ggml_moe_common::lossless_object_record &object =
            g_ds4_triplet_pack.lossless_sidecar.objects[sidecar_object_index];
        for (int role = 0; role < 3; ++role) {
            const uint8_t bit = ggml_moe_common::role_bit(
                static_cast<ggml_moe_common::role>(role));
            if ((crc_mask & bit) == 0) continue;
            const size_t crc_slot =
                batch.lossless_crc_role_slots[object_index][role];
            if (crc_slot == SIZE_MAX) return false;
            const size_t partial_offset =
                crc_slot * ggml_moe_common::lossless_crc32_segments;
            const uint32_t actual = ggml_moe_common::lossless_crc32_combine_segments(
                g_ds4_triplet_pack.lossless_crc32_host + partial_offset,
                object.roles[role].raw_bytes);
            if (actual != object.roles[role].raw_crc32) {
                g_ds4_triplet_pack.lossless_crc_failures.fetch_add(
                    1, std::memory_order_relaxed);
                return false;
            }
            g_ds4_triplet_pack.lossless_crc_verified[sidecar_object_index] |= bit;
            g_ds4_triplet_pack.lossless_crc_objects.fetch_add(
                1, std::memory_order_relaxed);
        }
    }
    return true;
}

static bool ds4_lossless_publish_group(
        ds4_triplet_pipe_batch &batch,
        bool critical,
        cudaStream_t stream) {
    for (int object_index = 0; object_index < batch.count; ++object_index) {
        if (!batch.lossless_objects[object_index]) continue;
        const ggml_moe_common::dependency_transfer_plan &plan =
            batch.transfer_plans[object_index];
        const ggml_moe_common::dependency_role_copy *copies = critical ?
            plan.critical : plan.deferred;
        const uint8_t copy_count = critical ?
            plan.critical_count : plan.deferred_count;
        for (uint8_t copy_index = 0;
                copy_index < copy_count;
                ++copy_index) {
            const int role_index = static_cast<int>(
                copies[copy_index].kind);
            if (batch.lossless_planar[object_index] &&
                    !vram_cache_set_reservation_layout(
                        batch.role_reservations[object_index][role_index], 1)) {
                return false;
            }
            if (batch.lossless_scale3[object_index] &&
                    !vram_cache_set_reservation_layout(
                        batch.role_reservations[object_index][role_index], 2)) {
                return false;
            }
            if (batch.lossless_planar_scale3[object_index] &&
                    !vram_cache_set_reservation_layout(
                        batch.role_reservations[object_index][role_index], 3)) {
                return false;
            }
            if (!vram_cache_publish_triplet(
                    batch.role_reservations[object_index][role_index],
                    stream)) {
                return false;
            }
        }
    }
    return true;
}

static bool ds4_triplet_pipe_process(ds4_triplet_pipe_batch & batch) {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (!g_ds4_triplet_pack.ring_inited || g_ds4_triplet_pack.fd_direct < 0 ||
            !g_ds4_triplet_pack.pipe_stream || batch.count <= 0 ||
            (batch.dependency_h2d &&
                (!g_ds4_triplet_pack.pipe_deferred_stream ||
                 !g_ds4_triplet_pack.pipe_critical_done))) {
        ds4_dependency_complete_owned_transactions(batch, false);
        return false;
    }

    const auto process_start = std::chrono::steady_clock::now();
    ggml_moe_common::object_request common_requests[8];
    const uint8_t role_mask = ggml_moe_common::role_bit(ggml_moe_common::role::gate) |
        ggml_moe_common::role_bit(ggml_moe_common::role::up) |
        ggml_moe_common::role_bit(ggml_moe_common::role::down);
    bool common_started = batch.common_transactions_owned;
    if (common_started) {
        std::copy(
            batch.common_requests,
            batch.common_requests + batch.count,
            common_requests);
    } else {
        common_started = ds4_triplet_common_begin(
            batch.entries, batch.count, role_mask, common_requests);
    }
    bool completed[8] = {};
    ggml_moe_common::object_io_window_slot io_window[8] = {};
    const bool common_object_cqe = ds4_common_object_cqe_enabled();
    bool h2d_profiled[8] = {};
    const char * profile_env = std::getenv("GGML_MOE_STREAM_DECODE_BATCH_PROFILE");
    bool profile_h2d = !batch.dependency_h2d &&
        profile_env && profile_env[0] && profile_env[0] != '0';
    if (profile_h2d && !g_ds4_triplet_pack.pipe_h2d_events_initialized) {
        for (int i = 0; i < 8; ++i) {
            if (cudaEventCreate(&g_ds4_triplet_pack.pipe_h2d_start_events[i]) != cudaSuccess ||
                    cudaEventCreate(&g_ds4_triplet_pack.pipe_h2d_stop_events[i]) != cudaSuccess) {
                cudaGetLastError();
                profile_h2d = false;
                break;
            }
        }
        g_ds4_triplet_pack.pipe_h2d_events_initialized = profile_h2d;
    }
    bool success = true;
    if (batch.dependency_h2d) {
        for (int i = 0; i < batch.count; ++i) {
            if (!batch.host_cache_hits[i]) continue;
            uint64_t copied_bytes = 0;
            if (!vram_cache_copy_dependency_group_from_host(
                    batch.transfer_plans[i].critical,
                    batch.transfer_plans[i].critical_count,
                    batch.role_reservations[i],
                    batch.host[i],
                    g_ds4_triplet_pack.pipe_stream,
                    copied_bytes)) {
                cudaGetLastError();
                success = false;
                continue;
            }
            completed[i] = true;
            g_ds4_triplet_pack.pipe_h2d_bytes.fetch_add(copied_bytes, std::memory_order_relaxed);
            g_ds4_triplet_pack.dependency_critical_copies.fetch_add(
                batch.transfer_plans[i].critical_count, std::memory_order_relaxed);
            g_ds4_triplet_pack.dependency_critical_bytes.fetch_add(
                copied_bytes, std::memory_order_relaxed);
        }
    }
    const ggml_moe_common::lossless_object_record *lossless_records[8] = {};
    std::vector<ggml_moe_common::lossless_mxfp4_decode_descriptor>
        lossless_critical_descriptors;
    std::vector<ggml_moe_common::lossless_mxfp4_decode_descriptor>
        lossless_deferred_descriptors;
    lossless_critical_descriptors.reserve(batch.count * 2);
    lossless_deferred_descriptors.reserve(batch.count);
    const size_t eager_decode_objects =
        ds4_lossless_eager_decode_objects();
    const bool eager_decode = eager_decode_objects != 0 &&
        ds4_lossless_trust_verified() &&
        ds4_lossless_decode_all_requested() &&
        !g_ds4_triplet_pack.lossless_nvcomp_decoder;
    const bool eager_nvcomp_completion = eager_decode_objects == 1 &&
        ds4_lossless_trust_verified() &&
        ds4_lossless_decode_all_requested() &&
        g_ds4_triplet_pack.lossless_nvcomp_decoder;
    size_t eager_pending_objects = 0;
    bool eager_decode_launched = false;
    auto flush_eager_decode = [&]() {
        if (lossless_critical_descriptors.empty()) return true;
        const size_t descriptor_count =
            lossless_critical_descriptors.size();
        if (cudaEventRecord(
                    g_ds4_triplet_pack.pipe_critical_done,
                    g_ds4_triplet_pack.pipe_stream) != cudaSuccess ||
                cudaStreamWaitEvent(
                    g_ds4_triplet_pack.pipe_deferred_stream,
                    g_ds4_triplet_pack.pipe_critical_done,
                    0) != cudaSuccess ||
                !ggml_moe_common::lossless_rans_cuda_decode_mxfp4_async(
                g_ds4_triplet_pack.lossless_critical_decoder,
                lossless_critical_descriptors.data(),
                descriptor_count,
                g_ds4_triplet_pack.pipe_deferred_stream) ||
                cudaEventRecord(
                    g_ds4_triplet_pack.lossless_decode_done,
                    g_ds4_triplet_pack.pipe_deferred_stream) != cudaSuccess) {
            return false;
        }
        g_ds4_triplet_pack.lossless_decode_batches.fetch_add(
            1, std::memory_order_relaxed);
        g_ds4_triplet_pack.lossless_decode_objects.fetch_add(
            eager_pending_objects, std::memory_order_relaxed);
        lossless_critical_descriptors.clear();
        eager_pending_objects = 0;
        eager_decode_launched = true;
        return true;
    };
    int read_indices[8] = {};
    size_t read_candidates = 0;
    for (int i = 0; i < batch.count; ++i) {
        if (!completed[i]) {
            read_indices[read_candidates++] = i;
        }
    }
    if (g_ds4_triplet_pack.lossless_requested) {
        g_ds4_triplet_pack.lossless_candidate_batches[
            std::min<size_t>(8, read_candidates)].fetch_add(
                1, std::memory_order_relaxed);
    }
    if (batch.dependency_h2d &&
            read_candidates >= ds4_lossless_min_batch()) {
        for (size_t candidate = 0; candidate < read_candidates; ++candidate) {
            const int index = read_indices[candidate];
            if (!batch.host_cache_hits[index]) {
                lossless_records[index] =
                    ds4_lossless_object(*batch.entries[index]);
            }
        }
    } else if (batch.dependency_h2d &&
            g_ds4_triplet_pack.lossless_requested) {
        g_ds4_triplet_pack.lossless_policy_raw_objects.fetch_add(
            read_candidates, std::memory_order_relaxed);
    }
    const bool rolling_refill = ds4_common_rolling_refill_requested();
    ggml_moe_common::object_pipeline_policy refill_policy =
        ds4_pipeline_policy_config().policy;
    refill_policy.io_depth = std::min<size_t>(8, refill_policy.io_depth);
    refill_policy.initial_batch = std::min(
        refill_policy.initial_batch, refill_policy.io_depth);
    refill_policy.refill_batch = std::min(
        refill_policy.refill_batch, refill_policy.io_depth);
    refill_policy.min_flush_batch = std::min(
        refill_policy.min_flush_batch, refill_policy.io_depth);
    refill_policy.max_inflight_h2d = std::min<size_t>(
        8, refill_policy.max_inflight_h2d);
    ggml_moe_common::object_pipeline_refill_cursor refill_cursor;
    if (read_candidates > 0) {
        if (!ggml_moe_common::object_pipeline_refill_reset(refill_cursor, read_candidates)) {
            success = false;
        } else if (rolling_refill) {
            g_ds4_triplet_pack.pipe_refill_batches.fetch_add(
                1, std::memory_order_relaxed);
        }
    }
    {
        std::lock_guard<std::mutex> io_lock(g_ds4_triplet_pack.io_mu);
        auto submit_refill = [&]() {
            const ggml_moe_common::object_pipeline_dispatch_plan dispatch =
                ggml_moe_common::object_pipeline_refill_plan_ready(
                    refill_cursor, refill_policy, 0, 0, rolling_refill);
            if (dispatch.reason == ggml_moe_common::object_pipeline_dispatch_reason::invalid) {
                return false;
            }
            if (dispatch.submit_count == 0) {
                return dispatch.reason == ggml_moe_common::object_pipeline_dispatch_reason::idle ||
                    dispatch.reason == ggml_moe_common::object_pipeline_dispatch_reason::io_backpressure;
            }
            const size_t first_candidate = refill_cursor.next();
            size_t prepared = 0;
            for (; prepared < dispatch.submit_count; ++prepared) {
                const int index = read_indices[first_candidate + prepared];
                const ggml_moe_common::lossless_object_record *lossless =
                    lossless_records[index];
                const int read_fd = lossless ?
                    g_ds4_triplet_pack.lossless_sidecar.fd_direct :
                    g_ds4_triplet_pack.fd_direct;
                const size_t read_bytes = lossless ?
                    static_cast<size_t>(lossless->physical_bytes) :
                    static_cast<size_t>(batch.entries[index]->nbytes);
                const uint64_t read_offset = lossless ?
                    lossless->file_offset : batch.entries[index]->offset;
                bool read_prepared = false;
                if (common_object_cqe) {
                    const ggml_moe_common::object_io_window_request read_request{
                        read_fd,
                        batch.host[index],
                        read_bytes,
                        read_offset,
                        (size_t) index,
                        false,
                    };
                    read_prepared = ggml_moe_common::object_io_window_prepare_read(
                        g_ds4_triplet_pack.io_ring, io_window, 8,
                        (size_t) index, read_request);
                } else {
                    read_prepared = ggml_moe_common::object_io_ring_prepare_read(
                        g_ds4_triplet_pack.io_ring,
                        read_fd,
                        batch.host[index],
                        read_bytes,
                        read_offset,
                        (uint64_t) (index + 1));
                }
                if (!read_prepared) break;
            }
            if (prepared == 0) return false;

            const auto submit_start = std::chrono::steady_clock::now();
            const int submit_rc =
                ggml_moe_common::object_io_ring_submit_count(g_ds4_triplet_pack.io_ring);
            g_ds4_triplet_pack.pipe_submit_us.fetch_add((uint64_t)
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - submit_start).count(),
                std::memory_order_relaxed);
            const size_t submitted = submit_rc > 0 ?
                std::min(prepared, (size_t) submit_rc) : 0;
            if (submitted == 0 ||
                    !ggml_moe_common::object_pipeline_refill_record_submit(
                        refill_cursor, submitted)) {
                return false;
            }
            if (rolling_refill) {
                g_ds4_triplet_pack.pipe_refill_submit_rounds.fetch_add(
                    1, std::memory_order_relaxed);
                g_ds4_triplet_pack.pipe_refill_submitted.fetch_add(
                    submitted, std::memory_order_relaxed);
            }
            return submitted == prepared;
        };

        if (read_candidates > 0 && success && !submit_refill()) success = false;

        const auto wait_start = std::chrono::steady_clock::now();
        while (refill_cursor.inflight_io() > 0) {
            ggml_moe_common::object_io_completion completion;
            if (!ggml_moe_common::object_io_ring_wait(g_ds4_triplet_pack.io_ring, completion)) {
                success = false;
                break;
            }
            size_t index = 0;
            ggml_moe_common::object_io_completion_status status;
            if (common_object_cqe) {
                ggml_moe_common::object_io_window_completion resolved;
                status = ggml_moe_common::object_io_window_resolve(
                    g_ds4_triplet_pack.io_ring, io_window, 8,
                    completion, resolved);
                index = resolved.request_id;
            } else {
                const bool index_valid = completion.user_data > 0 &&
                    completion.user_data <= (uint64_t) batch.count;
                index = index_valid ? (size_t) completion.user_data - 1 : 0;
                const size_t expected_bytes = index_valid ?
                    (lossless_records[index] ?
                        static_cast<size_t>(lossless_records[index]->physical_bytes) :
                        static_cast<size_t>(batch.entries[index]->nbytes)) : 0;
                status = ggml_moe_common::object_io_completion_validate(
                    g_ds4_triplet_pack.io_ring, completion, index_valid, expected_bytes);
            }
            if (!ggml_moe_common::object_pipeline_refill_record_io_completion(refill_cursor)) {
                success = false;
                break;
            }
            if (rolling_refill) {
                g_ds4_triplet_pack.pipe_refill_completions.fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (eager_nvcomp_completion && rolling_refill && success &&
                    refill_cursor.queued() > 0 && !submit_refill()) {
                success = false;
            }
            if (status == ggml_moe_common::object_io_completion_status::complete) {
                if (index >= (size_t) batch.count) {
                    success = false;
                    break;
                }
                const auto h2d_start = std::chrono::steady_clock::now();
                if (profile_h2d && cudaEventRecord(
                        g_ds4_triplet_pack.pipe_h2d_start_events[index],
                        g_ds4_triplet_pack.pipe_stream) != cudaSuccess) {
                    cudaGetLastError();
                    profile_h2d = false;
                }
                uint64_t copied_bytes = 0;
                const ggml_moe_common::lossless_object_record *lossless =
                    lossless_records[index];
                const bool copied = lossless && batch.dependency_h2d ?
                    ds4_lossless_enqueue_object(
                        batch, static_cast<int>(index), *lossless,
                        lossless_critical_descriptors,
                        lossless_deferred_descriptors) :
                    (batch.dependency_h2d ?
                        vram_cache_copy_dependency_group_from_host(
                            batch.transfer_plans[index].critical,
                            batch.transfer_plans[index].critical_count,
                            batch.role_reservations[index],
                            batch.host[index],
                            g_ds4_triplet_pack.pipe_stream,
                            copied_bytes) :
                        vram_cache_copy_triplet_from_host(
                            batch.reservations[index], batch.host[index],
                            (size_t) batch.entries[index]->nbytes,
                            g_ds4_triplet_pack.pipe_stream));
                if (lossless) copied_bytes = lossless->payload_bytes;
                if (!batch.dependency_h2d) {
                    copied_bytes = batch.entries[index]->nbytes;
                }
                if (copied && profile_h2d && cudaEventRecord(
                        g_ds4_triplet_pack.pipe_h2d_stop_events[index],
                        g_ds4_triplet_pack.pipe_stream) == cudaSuccess) {
                    h2d_profiled[index] = true;
                }
                g_ds4_triplet_pack.pipe_h2d_enqueue_us.fetch_add((uint64_t)
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - h2d_start).count(), std::memory_order_relaxed);
                if (copied) {
                    completed[index] = true;
                    const uint64_t physical_read_bytes = lossless ?
                        lossless->physical_bytes : batch.entries[index]->nbytes;
                    g_ds4_triplet_pack.reads.fetch_add(1, std::memory_order_relaxed);
                    g_ds4_triplet_pack.bytes.fetch_add(
                        physical_read_bytes, std::memory_order_relaxed);
                    g_ds4_triplet_pack.io_stats.reads.fetch_add(1, std::memory_order_relaxed);
                    g_ds4_triplet_pack.pipe_cqes.fetch_add(1, std::memory_order_relaxed);
                    if (!lossless) {
                        g_ds4_triplet_pack.pipe_h2d_bytes.fetch_add(
                            copied_bytes, std::memory_order_relaxed);
                    } else {
                        g_ds4_triplet_pack.lossless_reads.fetch_add(
                            1, std::memory_order_relaxed);
                        g_ds4_triplet_pack.lossless_physical_bytes.fetch_add(
                            lossless->physical_bytes, std::memory_order_relaxed);
                        g_ds4_triplet_pack.lossless_payload_bytes.fetch_add(
                            lossless->payload_bytes, std::memory_order_relaxed);
                        g_ds4_triplet_pack.lossless_raw_bytes_avoided.fetch_add(
                            batch.entries[index]->nbytes - lossless->physical_bytes,
                            std::memory_order_relaxed);
                    }
                    if (batch.dependency_h2d) {
                        uint64_t logical_critical_bytes = 0;
                        for (uint8_t copy = 0;
                                copy < batch.transfer_plans[index].critical_count;
                                ++copy) {
                            logical_critical_bytes +=
                                batch.transfer_plans[index].critical[copy].bytes;
                        }
                        g_ds4_triplet_pack.dependency_critical_copies.fetch_add(
                            batch.transfer_plans[index].critical_count, std::memory_order_relaxed);
                        g_ds4_triplet_pack.dependency_critical_bytes.fetch_add(
                            logical_critical_bytes, std::memory_order_relaxed);
                    }
                    if (lossless && eager_decode) {
                        ++eager_pending_objects;
                        if (eager_pending_objects >= eager_decode_objects &&
                                !flush_eager_decode()) {
                            g_ds4_triplet_pack.lossless_decode_failures.fetch_add(
                                1, std::memory_order_relaxed);
                            g_ds4_triplet_pack.lossless_failed = true;
                            success = false;
                        }
                    }
                    if (lossless && eager_nvcomp_completion &&
                            batch.lossless_nvcomp[index]) {
                        ggml_moe_common::lossless_nvcomp_object_descriptor descriptor;
                        const bool decoded =
                            ds4_lossless_nvcomp_descriptor(
                                batch, static_cast<int>(index), descriptor) &&
                            ggml_moe_common::lossless_nvcomp_decode_single_async(
                                g_ds4_triplet_pack.lossless_nvcomp_decoder,
                                descriptor, index,
                                g_ds4_triplet_pack.pipe_stream);
                        if (!decoded) {
                            cudaGetLastError();
                            g_ds4_triplet_pack.lossless_decode_failures.fetch_add(
                                1, std::memory_order_relaxed);
                            g_ds4_triplet_pack.lossless_failed = true;
                            success = false;
                        } else {
                            batch.lossless_nvcomp[index] = false;
                            g_ds4_triplet_pack.lossless_decode_batches.fetch_add(
                                1, std::memory_order_relaxed);
                            g_ds4_triplet_pack.lossless_decode_objects.fetch_add(
                                1, std::memory_order_relaxed);
                        }
                    }
                } else {
                    cudaGetLastError();
                    if (lossless) {
                        lossless_records[index] = nullptr;
                        g_ds4_triplet_pack.lossless_fallbacks.fetch_add(
                            1, std::memory_order_relaxed);
                    } else {
                        success = false;
                    }
                }
            } else {
                success = false;
            }
            if (rolling_refill && success && refill_cursor.queued() > 0 &&
                    !submit_refill()) {
                success = false;
            }
        }
        if (rolling_refill && read_candidates > 0 && !refill_cursor.done()) {
            g_ds4_triplet_pack.pipe_refill_fallbacks.fetch_add(
                1, std::memory_order_relaxed);
        }
        g_ds4_triplet_pack.pipe_wait_us.fetch_add((uint64_t)
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - wait_start).count(), std::memory_order_relaxed);
        if (common_object_cqe) ggml_moe_common::object_io_window_reset(io_window, 8);
    }

    for (int i = 0; i < batch.count; ++i) {
        if (completed[i]) continue;
        if (batch.host_cache_hits[i]) {
            success = false;
            continue;
        }
        g_ds4_triplet_pack.io_stats.fallbacks.fetch_add(1, std::memory_order_relaxed);
        uint64_t copied_bytes = 0;
        const bool read_ok = ds4_triplet_read_raw(*batch.entries[i], batch.host[i]);
        const bool copied = read_ok && (batch.dependency_h2d ?
            vram_cache_copy_dependency_group_from_host(
                batch.transfer_plans[i].critical,
                batch.transfer_plans[i].critical_count,
                batch.role_reservations[i],
                batch.host[i],
                g_ds4_triplet_pack.pipe_stream,
                copied_bytes) :
            vram_cache_copy_triplet_from_host(
                batch.reservations[i], batch.host[i],
                (size_t) batch.entries[i]->nbytes,
                g_ds4_triplet_pack.pipe_stream));
        if (!batch.dependency_h2d) copied_bytes = batch.entries[i]->nbytes;
        if (!copied) {
            cudaGetLastError();
            if (!batch.dependency_h2d) vram_cache_fail_triplet(batch.reservations[i]);
            success = false;
        } else {
            completed[i] = true;
            g_ds4_triplet_pack.pipe_h2d_bytes.fetch_add(copied_bytes, std::memory_order_relaxed);
            if (batch.dependency_h2d) {
                g_ds4_triplet_pack.dependency_critical_copies.fetch_add(
                    batch.transfer_plans[i].critical_count, std::memory_order_relaxed);
                g_ds4_triplet_pack.dependency_critical_bytes.fetch_add(
                    copied_bytes, std::memory_order_relaxed);
            }
        }
    }
    std::vector<ggml_moe_common::lossless_nvcomp_object_descriptor>
        nvcomp_descriptors;
    nvcomp_descriptors.reserve(batch.count);
    for (int object_index = 0; object_index < batch.count; ++object_index) {
        if (!batch.lossless_nvcomp[object_index]) continue;
        const size_t sidecar_object_index =
            batch.lossless_object_indices[object_index];
        if (sidecar_object_index >=
                g_ds4_triplet_pack.lossless_sidecar.objects.size()) {
            success = false;
            break;
        }
        const auto &object =
            g_ds4_triplet_pack.lossless_sidecar.objects[
                sidecar_object_index];
        ggml_moe_common::lossless_nvcomp_object_descriptor descriptor;
        descriptor.compressed = g_ds4_triplet_pack.lossless_device_input +
            static_cast<size_t>(object_index) *
                g_ds4_triplet_pack.lossless_device_input_stride;
        descriptor.role_mask =
            batch.lossless_nvcomp_role_masks[object_index];
        for (size_t role = 0; role < 3; ++role) {
            descriptor.destinations[role] = static_cast<uint8_t *>(
                batch.role_ptrs[object_index][role]);
            descriptor.blocks[role] =
                static_cast<uint64_t>(object.roles[role].rows) *
                object.roles[role].blocks_per_row;
        }
        nvcomp_descriptors.push_back(descriptor);
    }
    if (!nvcomp_descriptors.empty()) {
        if (!success || !ggml_moe_common::lossless_nvcomp_decode_async(
                g_ds4_triplet_pack.lossless_nvcomp_decoder,
                nvcomp_descriptors.data(), nvcomp_descriptors.size(),
                g_ds4_triplet_pack.pipe_stream)) {
            cudaGetLastError();
            g_ds4_triplet_pack.lossless_decode_failures.fetch_add(
                1, std::memory_order_relaxed);
            g_ds4_triplet_pack.lossless_failed = true;
            success = false;
        } else {
            g_ds4_triplet_pack.lossless_decode_batches.fetch_add(
                1, std::memory_order_relaxed);
            g_ds4_triplet_pack.lossless_decode_objects.fetch_add(
                nvcomp_descriptors.size(), std::memory_order_relaxed);
        }
    }

    bool lossless_payload_crc_pending = false;
    bool lossless_output_crc_pending = false;
    for (int object = 0; object < batch.count; ++object) {
        lossless_payload_crc_pending = lossless_payload_crc_pending ||
            (batch.lossless_objects[object] &&
             !batch.lossless_direct_quant[object] &&
             batch.lossless_crc_required[object]);
        lossless_output_crc_pending = lossless_output_crc_pending ||
            (batch.lossless_direct_quant[object] &&
             batch.lossless_crc_role_masks[object] != 0);
    }
    if (eager_decode && success && !flush_eager_decode()) {
        cudaGetLastError();
        g_ds4_triplet_pack.lossless_decode_failures.fetch_add(
            1, std::memory_order_relaxed);
        g_ds4_triplet_pack.lossless_failed = true;
        success = false;
    }
    if (eager_decode_launched && success &&
            cudaStreamWaitEvent(
                g_ds4_triplet_pack.pipe_stream,
                g_ds4_triplet_pack.lossless_decode_done,
                0) != cudaSuccess) {
        cudaGetLastError();
        g_ds4_triplet_pack.lossless_decode_failures.fetch_add(
            1, std::memory_order_relaxed);
        g_ds4_triplet_pack.lossless_failed = true;
        success = false;
    }
    if (lossless_payload_crc_pending && !ds4_lossless_enqueue_crc_batch(batch)) {
        cudaGetLastError();
        g_ds4_triplet_pack.lossless_failed = true;
        success = false;
    }
    if (!lossless_critical_descriptors.empty() && success) {
        const size_t descriptor_count =
            lossless_critical_descriptors.size();
        const bool trust_verified = ds4_lossless_trust_verified();
        const bool decode_launched =
            ggml_moe_common::lossless_rans_cuda_decode_mxfp4_async(
                g_ds4_triplet_pack.lossless_critical_decoder,
                lossless_critical_descriptors.data(),
                descriptor_count,
                g_ds4_triplet_pack.pipe_stream);
        bool decoded = decode_launched;
        bool status_copy_queued = trust_verified;
        if (decoded && !trust_verified) {
            status_copy_queued = cudaMemcpyAsync(
                g_ds4_triplet_pack.lossless_critical_status_host,
                ggml_moe_common::lossless_rans_cuda_status_device(
                    g_ds4_triplet_pack.lossless_critical_decoder),
                descriptor_count * sizeof(uint32_t),
                cudaMemcpyDeviceToHost,
                g_ds4_triplet_pack.pipe_stream) == cudaSuccess;
            decoded = status_copy_queued;
        }
        if (decoded && lossless_output_crc_pending) {
            decoded = ds4_lossless_enqueue_output_crc_batch(batch);
        }
        if (decoded && !trust_verified) {
            decoded = cudaStreamSynchronize(
                g_ds4_triplet_pack.pipe_stream) == cudaSuccess;
        }
        for (size_t descriptor = 0;
                decoded && !trust_verified && descriptor < descriptor_count;
                ++descriptor) {
            const uint32_t status =
                g_ds4_triplet_pack.lossless_critical_status_host[descriptor];
            if (status != 0) {
                std::fprintf(stderr,
                    "[moe_stream] DS4 lossless critical descriptor %zu status=%u\n",
                    descriptor, status);
                decoded = false;
            }
        }
        if (decoded && lossless_payload_crc_pending) {
            decoded = ds4_lossless_validate_payload_crc(batch);
        }
        if (decoded && lossless_output_crc_pending) {
            decoded = ds4_lossless_validate_output_crc(batch);
        }
        if (!decoded || !ds4_lossless_publish_group(
                batch, true, g_ds4_triplet_pack.pipe_stream)) {
            const cudaError_t error = cudaGetLastError();
            std::fprintf(stderr,
                "[moe_stream] DS4 lossless critical failure: decoded=%d "
                "launch=%d status_copy=%d descriptors=%zu capacity=%zu "
                "first_chunks=%u first_scale_records=%u first_scale_codec=%u "
                "first_context=%u cuda=%s\n",
                decoded ? 1 : 0, decode_launched ? 1 : 0,
                status_copy_queued ? 1 : 0, descriptor_count,
                ggml_moe_common::lossless_rans_cuda_capacity(
                    g_ds4_triplet_pack.lossless_critical_decoder),
                lossless_critical_descriptors[0].chunks,
                lossless_critical_descriptors[0].scale_records_count,
                static_cast<unsigned>(
                    lossless_critical_descriptors[0].scale_codec),
                lossless_critical_descriptors[0].quant_context_mode,
                cudaGetErrorString(error));
            if (!decode_launched) {
                for (size_t descriptor = 0;
                        descriptor < descriptor_count;
                        ++descriptor) {
                    const auto &item = lossless_critical_descriptors[descriptor];
                    std::fprintf(stderr,
                        "[moe_stream] DS4 lossless descriptor: index=%zu "
                        "chunks=%u scale_records=%u scale_codec=%u context=%u "
                        "blocks=%lu destination=%p\n",
                        descriptor, item.chunks, item.scale_records_count,
                        static_cast<unsigned>(item.scale_codec),
                        item.quant_context_mode, item.blocks,
                        static_cast<void *>(item.destination));
                }
            }
            g_ds4_triplet_pack.lossless_decode_failures.fetch_add(
                1, std::memory_order_relaxed);
            g_ds4_triplet_pack.lossless_failed = true;
            success = false;
        } else {
            size_t object_count = 0;
            for (int object = 0; object < batch.count; ++object) {
                if (batch.lossless_objects[object]) ++object_count;
            }
            g_ds4_triplet_pack.lossless_decode_batches.fetch_add(
                1, std::memory_order_relaxed);
            g_ds4_triplet_pack.lossless_decode_objects.fetch_add(
                object_count, std::memory_order_relaxed);
        }
    } else if (eager_decode_launched && success) {
        if (!ds4_lossless_publish_group(
                batch, true, g_ds4_triplet_pack.pipe_stream)) {
            cudaGetLastError();
            g_ds4_triplet_pack.lossless_decode_failures.fetch_add(
                1, std::memory_order_relaxed);
            g_ds4_triplet_pack.lossless_failed = true;
            success = false;
        }
    } else if (lossless_payload_crc_pending && success) {
        if (cudaStreamSynchronize(g_ds4_triplet_pack.pipe_stream) != cudaSuccess ||
                !ds4_lossless_validate_payload_crc(batch)) {
            cudaGetLastError();
            g_ds4_triplet_pack.lossless_failed = true;
            success = false;
        }
    } else if (success) {
        bool has_lossless_objects = false;
        for (int object = 0; object < batch.count; ++object) {
            has_lossless_objects = has_lossless_objects ||
                batch.lossless_objects[object];
        }
        if (has_lossless_objects && !ds4_lossless_publish_group(
                batch, true, g_ds4_triplet_pack.pipe_stream)) {
            cudaGetLastError();
            g_ds4_triplet_pack.lossless_decode_failures.fetch_add(
                1, std::memory_order_relaxed);
            g_ds4_triplet_pack.lossless_failed = true;
            success = false;
        }
    }
    if (batch.dependency_h2d) {
        bool pipeline_critical_published = false;
        if (cudaEventRecord(
                g_ds4_triplet_pack.pipe_critical_done,
                g_ds4_triplet_pack.pipe_stream) != cudaSuccess ||
                cudaStreamWaitEvent(
                    g_ds4_triplet_pack.pipe_deferred_stream,
                    g_ds4_triplet_pack.pipe_critical_done,
                    0) != cudaSuccess) {
            cudaGetLastError();
            success = false;
        }
        if (batch.common_transactions_owned && success) {
            for (int i = 0; i < batch.count; ++i) {
                if (!completed[i]) continue;
                const uint8_t ready_mask =
                    batch.transfer_plans[i].missing_role_mask &
                    batch.transfer_plans[i].critical_role_mask;
                if (ready_mask != 0 &&
                        !ggml_moe_common::object_engine_publish_transaction_roles(
                            batch.transaction_tickets[i], ready_mask)) {
                    success = false;
                }
            }
        }
        if (batch.pipeline_handle.generation != 0 && success) {
            bool caller_roles_ready = true;
            for (int i = 0; i < batch.count; ++i) {
                if (!completed[i] ||
                        (batch.caller_role_masks[i] &
                            batch.transfer_plans[i].deferred_role_mask) != 0) {
                    caller_roles_ready = false;
                    break;
                }
            }
            if (caller_roles_ready) {
                pipeline_critical_published =
                    g_ds4_triplet_pack.dependency_pipeline_pool.publish_critical(
                        batch.pipeline_handle, true);
                if (pipeline_critical_published) {
                    g_ds4_triplet_pack.dependency_pipeline_critical_publishes.fetch_add(
                        1, std::memory_order_relaxed);
                    g_ds4_triplet_pack.dependency_pipeline_early_critical_publishes.fetch_add(
                        1, std::memory_order_relaxed);
                } else {
                    success = false;
                }
            }
        }
        if (!lossless_deferred_descriptors.empty() && success) {
            const size_t descriptor_count =
                lossless_deferred_descriptors.size();
            bool decoded =
                ggml_moe_common::lossless_rans_cuda_decode_mxfp4_async(
                    g_ds4_triplet_pack.lossless_deferred_decoder,
                    lossless_deferred_descriptors.data(),
                    descriptor_count,
                    g_ds4_triplet_pack.pipe_deferred_stream) &&
                cudaMemcpyAsync(
                    g_ds4_triplet_pack.lossless_deferred_status_host,
                    ggml_moe_common::lossless_rans_cuda_status_device(
                        g_ds4_triplet_pack.lossless_deferred_decoder),
                    descriptor_count * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost,
                    g_ds4_triplet_pack.pipe_deferred_stream) == cudaSuccess;
            const char *async_env =
                std::getenv("GGML_MOE_LOSSLESS_ASYNC_TRUST_VERIFIED");
            const bool trust_verified = async_env && async_env[0] &&
                async_env[0] != '0';
            if (decoded && !trust_verified) {
                decoded = cudaStreamSynchronize(
                    g_ds4_triplet_pack.pipe_deferred_stream) == cudaSuccess;
                for (size_t descriptor = 0;
                        decoded && descriptor < descriptor_count;
                        ++descriptor) {
                    decoded =
                        g_ds4_triplet_pack.lossless_deferred_status_host[descriptor] == 0;
                }
            }
            if (!decoded || !ds4_lossless_publish_group(
                    batch, false,
                    g_ds4_triplet_pack.pipe_deferred_stream)) {
                cudaGetLastError();
                g_ds4_triplet_pack.lossless_decode_failures.fetch_add(
                    1, std::memory_order_relaxed);
                g_ds4_triplet_pack.lossless_failed = true;
                success = false;
            }
        } else if (batch.lossless_decode_all && success &&
                !ds4_lossless_publish_group(
                    batch, false,
                    g_ds4_triplet_pack.pipe_deferred_stream)) {
            cudaGetLastError();
            g_ds4_triplet_pack.lossless_decode_failures.fetch_add(
                1, std::memory_order_relaxed);
            g_ds4_triplet_pack.lossless_failed = true;
            success = false;
        }
        for (int i = 0; i < batch.count; ++i) {
            if (!completed[i]) continue;
            uint64_t copied_bytes = 0;
            if (batch.lossless_objects[i]) {
                for (uint8_t copy = 0;
                        copy < batch.transfer_plans[i].deferred_count;
                        ++copy) {
                    copied_bytes +=
                        batch.transfer_plans[i].deferred[copy].bytes;
                }
            } else if (!vram_cache_copy_dependency_group_from_host(
                        batch.transfer_plans[i].deferred,
                        batch.transfer_plans[i].deferred_count,
                        batch.role_reservations[i],
                        batch.host[i],
                        g_ds4_triplet_pack.pipe_deferred_stream,
                        copied_bytes)) {
                cudaGetLastError();
                success = false;
                continue;
            }
            if (!batch.lossless_objects[i]) {
                g_ds4_triplet_pack.pipe_h2d_bytes.fetch_add(
                    copied_bytes, std::memory_order_relaxed);
            }
            g_ds4_triplet_pack.dependency_deferred_copies.fetch_add(
                batch.transfer_plans[i].deferred_count, std::memory_order_relaxed);
            g_ds4_triplet_pack.dependency_deferred_bytes.fetch_add(
                copied_bytes, std::memory_order_relaxed);
        }
        if (batch.pipeline_handle.generation != 0 && !pipeline_critical_published) {
            const bool published =
                g_ds4_triplet_pack.dependency_pipeline_pool.publish_critical(
                    batch.pipeline_handle, success);
            if (published && success) {
                g_ds4_triplet_pack.dependency_pipeline_critical_publishes.fetch_add(
                    1, std::memory_order_relaxed);
            } else if (!published) {
                success = false;
            }
        }
        if (success) {
            ds4_dependency_finalize_host_sources(
                batch, g_ds4_triplet_pack.pipe_deferred_stream);
        }
    }
    if (profile_h2d) {
        for (int i = 0; i < batch.count; ++i) {
            if (!h2d_profiled[i]) continue;
            float elapsed_ms = 0.0f;
            if (cudaEventSynchronize(g_ds4_triplet_pack.pipe_h2d_stop_events[i]) == cudaSuccess &&
                    cudaEventElapsedTime(
                        &elapsed_ms,
                        g_ds4_triplet_pack.pipe_h2d_start_events[i],
                        g_ds4_triplet_pack.pipe_h2d_stop_events[i]) == cudaSuccess) {
                g_ds4_triplet_pack.pipe_h2d_device_us.fetch_add(
                    (uint64_t) (elapsed_ms * 1000.0f), std::memory_order_relaxed);
            } else {
                cudaGetLastError();
            }
        }
    }
    g_ds4_triplet_pack.pipe_batches.fetch_add(1, std::memory_order_relaxed);
    g_ds4_triplet_pack.pipe_process_us.fetch_add((uint64_t)
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - process_start).count(), std::memory_order_relaxed);
    if (batch.common_transactions_owned) {
        ds4_dependency_complete_owned_transactions(batch, success);
    } else {
        ds4_triplet_common_complete(common_requests, batch.count, common_started, success);
    }
    return success;
#else
    (void) batch;
    return false;
#endif
}

static bool ds4_triplet_pipe_worker_init() {
    if (cudaSetDevice(0) != cudaSuccess) {
        return false;
    }
    int least_priority = 0;
    int greatest_priority = 0;
    const bool dependency_h2d = ds4_dependency_h2d_requested();
    bool streams_ready = false;
    if (dependency_h2d) {
        if (cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority) != cudaSuccess) {
            cudaGetLastError();
            least_priority = 0;
            greatest_priority = 0;
        }
        streams_ready =
            cudaStreamCreateWithPriority(
                &g_ds4_triplet_pack.pipe_stream,
                cudaStreamNonBlocking,
                greatest_priority) == cudaSuccess &&
            cudaStreamCreateWithPriority(
                &g_ds4_triplet_pack.pipe_deferred_stream,
                cudaStreamNonBlocking,
                least_priority) == cudaSuccess &&
            cudaEventCreateWithFlags(
                &g_ds4_triplet_pack.pipe_critical_done,
                cudaEventDisableTiming) == cudaSuccess &&
            cudaEventCreateWithFlags(
                &g_ds4_triplet_pack.lossless_decode_done,
                cudaEventDisableTiming) == cudaSuccess;
    } else {
        streams_ready = cudaStreamCreateWithFlags(
            &g_ds4_triplet_pack.pipe_stream, cudaStreamNonBlocking) == cudaSuccess;
    }
    if (!streams_ready) {
        cudaGetLastError();
        if (g_ds4_triplet_pack.pipe_critical_done) {
            cudaEventDestroy(g_ds4_triplet_pack.pipe_critical_done);
            g_ds4_triplet_pack.pipe_critical_done = nullptr;
        }
        if (g_ds4_triplet_pack.lossless_decode_done) {
            cudaEventDestroy(g_ds4_triplet_pack.lossless_decode_done);
            g_ds4_triplet_pack.lossless_decode_done = nullptr;
        }
        if (g_ds4_triplet_pack.pipe_deferred_stream) {
            cudaStreamDestroy(g_ds4_triplet_pack.pipe_deferred_stream);
            g_ds4_triplet_pack.pipe_deferred_stream = nullptr;
        }
        if (g_ds4_triplet_pack.pipe_stream) {
            cudaStreamDestroy(g_ds4_triplet_pack.pipe_stream);
            g_ds4_triplet_pack.pipe_stream = nullptr;
        }
        return false;
    }
    if (!ds4_lossless_cuda_init()) return false;
    if (g_ds4_triplet_pack.ring_init_on_worker) {
        bool ring_ready = ggml_moe_common::object_io_ring_ensure(
            g_ds4_triplet_pack.io_ring,
            g_ds4_triplet_pack.ring_depth,
            g_ds4_triplet_pack.ring_requested_flags,
            nullptr);
        if (!ring_ready) {
            g_ds4_triplet_pack.ring_init_fallback = true;
            ring_ready = ggml_moe_common::object_io_ring_ensure(
                g_ds4_triplet_pack.io_ring,
                g_ds4_triplet_pack.ring_depth,
                0,
                nullptr);
        }
        g_ds4_triplet_pack.ring_inited = ring_ready;
        g_ds4_triplet_pack.ring_active_flags =
            ring_ready && !g_ds4_triplet_pack.ring_init_fallback ?
                g_ds4_triplet_pack.ring_requested_flags : 0;
        std::fprintf(stderr,
            "[moe_stream] DS4 triplet pipe ring: worker_init=1 requested_flags=%u "
            "active_flags=%u fallback=%d status=%s\n",
            g_ds4_triplet_pack.ring_requested_flags,
            g_ds4_triplet_pack.ring_active_flags,
            g_ds4_triplet_pack.ring_init_fallback ? 1 : 0,
            ring_ready ? "ok" : "failed");
        if (!ring_ready) {
            if (g_ds4_triplet_pack.pipe_critical_done) {
                cudaEventDestroy(g_ds4_triplet_pack.pipe_critical_done);
                g_ds4_triplet_pack.pipe_critical_done = nullptr;
            }
            if (g_ds4_triplet_pack.lossless_decode_done) {
                cudaEventDestroy(g_ds4_triplet_pack.lossless_decode_done);
                g_ds4_triplet_pack.lossless_decode_done = nullptr;
            }
            if (g_ds4_triplet_pack.pipe_deferred_stream) {
                cudaStreamDestroy(g_ds4_triplet_pack.pipe_deferred_stream);
                g_ds4_triplet_pack.pipe_deferred_stream = nullptr;
            }
            cudaStreamDestroy(g_ds4_triplet_pack.pipe_stream);
            g_ds4_triplet_pack.pipe_stream = nullptr;
            return false;
        }
    }
    return g_ds4_triplet_pack.ring_inited;
}

static void ds4_triplet_pipe_worker_main() {
    const bool device_ok = ds4_triplet_pipe_worker_init();
    {
        std::lock_guard<std::mutex> lock(g_ds4_triplet_pack.pipe_mu);
        g_ds4_triplet_pack.pipe_failed = !device_ok;
        g_ds4_triplet_pack.pipe_started = true;
    }
    g_ds4_triplet_pack.pipe_cv.notify_all();
    if (!device_ok) return;

    for (;;) {
        ds4_triplet_pipe_batch * batch = nullptr;
        {
            std::unique_lock<std::mutex> lock(g_ds4_triplet_pack.pipe_mu);
            g_ds4_triplet_pack.pipe_cv.wait(lock, [] {
                return g_ds4_triplet_pack.pipe_stop || !g_ds4_triplet_pack.pipe_queue.empty();
            });
            if (g_ds4_triplet_pack.pipe_stop && g_ds4_triplet_pack.pipe_queue.empty()) break;
            batch = g_ds4_triplet_pack.pipe_queue.front();
            g_ds4_triplet_pack.pipe_queue.pop_front();
        }
        const bool ok = batch && ds4_triplet_pipe_process(*batch);
        if (!batch) continue;
        {
            std::lock_guard<std::mutex> lock(batch->mu);
            batch->success = ok;
            batch->done = true;
        }
        batch->cv.notify_one();
        if (batch->pipeline_handle.generation != 0) {
            (void) g_ds4_triplet_pack.dependency_pipeline_pool.complete_worker(
                batch->pipeline_handle, ok);
        }
    }
    if (g_ds4_triplet_pack.pipe_critical_done) {
        cudaEventDestroy(g_ds4_triplet_pack.pipe_critical_done);
        g_ds4_triplet_pack.pipe_critical_done = nullptr;
    }
    if (g_ds4_triplet_pack.lossless_decode_done) {
        cudaEventDestroy(g_ds4_triplet_pack.lossless_decode_done);
        g_ds4_triplet_pack.lossless_decode_done = nullptr;
    }
    if (g_ds4_triplet_pack.pipe_deferred_stream) {
        cudaStreamDestroy(g_ds4_triplet_pack.pipe_deferred_stream);
        g_ds4_triplet_pack.pipe_deferred_stream = nullptr;
    }
    cudaStreamDestroy(g_ds4_triplet_pack.pipe_stream);
    g_ds4_triplet_pack.pipe_stream = nullptr;
}

static bool ds4_triplet_common_worker_init(void *) {
    return ds4_triplet_pipe_worker_init();
}

static bool ds4_triplet_common_worker_process(void *, void *context, uint32_t kind) {
    if (kind != 1 || !context) return false;
    ds4_triplet_pipe_batch *batch = static_cast<ds4_triplet_pipe_batch *>(context);
    const bool success = ds4_triplet_pipe_process(*batch);
    {
        std::lock_guard<std::mutex> lock(batch->mu);
        batch->success = success;
        batch->done = true;
    }
    batch->cv.notify_one();
    if (batch->pipeline_handle.generation != 0) {
        (void) g_ds4_triplet_pack.dependency_pipeline_pool.complete_worker(
            batch->pipeline_handle, success);
    }
    return success;
}

static bool ds4_triplet_pipe_submit(ds4_triplet_pipe_batch & batch) {
    if (!g_ds4_triplet_pack.pipe_requested || g_ds4_triplet_pack.pipe_failed ||
            !g_ds4_triplet_pack.pipe_started || batch.count <= 0) return false;
    const auto wait_start = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(g_ds4_triplet_pack.pipe_mu);
        if (g_ds4_triplet_pack.pipe_stop || g_ds4_triplet_pack.pipe_failed) return false;
        if (g_ds4_triplet_pack.pipe_common_worker) {
            if (!ggml_moe_common::object_engine_enqueue_work(
                    &batch, 1, 0, DS4_TRIPLET_PIPE_QUEUE_CAPACITY)) {
                return false;
            }
        } else {
            g_ds4_triplet_pack.pipe_queue.push_back(&batch);
        }
        batch.pipeline_submitted = true;
    }
    if (!g_ds4_triplet_pack.pipe_common_worker) {
        g_ds4_triplet_pack.pipe_cv.notify_one();
    }
    if (batch.pipeline_handle.generation != 0 &&
            ds4_common_async_critical_return_requested()) {
        const auto critical_wait_start = std::chrono::steady_clock::now();
        const ggml_moe_common::object_pipeline_transaction_wait_status status =
            g_ds4_triplet_pack.dependency_pipeline_pool.wait_critical(
                batch.pipeline_handle, UINT64_C(30000000));
        const uint64_t critical_wait_us = (uint64_t)
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - critical_wait_start).count();
        g_ds4_triplet_pack.dependency_pipeline_critical_wait_us.fetch_add(
            critical_wait_us, std::memory_order_relaxed);
        if (status == ggml_moe_common::object_pipeline_transaction_wait_status::critical_ready) {
            batch.pipeline_critical_returned = true;
            g_ds4_triplet_pack.dependency_pipeline_critical_returns.fetch_add(
                1, std::memory_order_relaxed);
            g_ds4_triplet_pack.pipe_caller_wait_us.fetch_add((uint64_t)
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - wait_start).count(),
                std::memory_order_relaxed);
            return true;
        }
        if (status == ggml_moe_common::object_pipeline_transaction_wait_status::completed) {
            g_ds4_triplet_pack.pipe_caller_wait_us.fetch_add((uint64_t)
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - wait_start).count(),
                std::memory_order_relaxed);
            return true;
        }
        g_ds4_triplet_pack.dependency_pipeline_wait_all_fallbacks.fetch_add(
            1, std::memory_order_relaxed);
    }
    std::unique_lock<std::mutex> lock(batch.mu);
    batch.cv.wait(lock, [&] { return batch.done; });
    g_ds4_triplet_pack.pipe_caller_wait_us.fetch_add((uint64_t)
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - wait_start).count(), std::memory_order_relaxed);
    return batch.success;
}

static bool one_pack_read_profile_enabled() {
    const char *env = std::getenv("GGML_MOE_ONE_PACK_READ_PROFILE_OUT");
    return env && env[0];
}

static bool one_pack_read_summary_enabled() {
    const char *env = std::getenv("GGML_MOE_ONE_PACK_READ_SUMMARY");
    return env && env[0] && env[0] != '0';
}

static std::atomic<uint64_t> g_one_pack_direct_read_time_us{0};
static std::atomic<uint64_t> g_one_pack_direct_read_max_us{0};
static std::atomic<uint64_t> g_one_pack_buffered_read_time_us{0};
static std::atomic<uint64_t> g_one_pack_buffered_read_max_us{0};

static void one_pack_read_summary_update(std::atomic<uint64_t> &total, std::atomic<uint64_t> &max_value, uint64_t value) {
    total.fetch_add(value, std::memory_order_relaxed);
    uint64_t prev = max_value.load(std::memory_order_relaxed);
    while (value > prev && !max_value.compare_exchange_weak(prev, value, std::memory_order_relaxed)) {
    }
}

static void one_pack_read_profile_record(
        const one_expert_pack_entry *entry,
        bool direct,
        bool direct_failed,
        bool ok,
        double elapsed_ms) {
    const char *path = std::getenv("GGML_MOE_ONE_PACK_READ_PROFILE_OUT");
    if (!path || !path[0] || !entry) return;

    static std::mutex mu;
    static bool header_written = false;
    static uint64_t seq = 0;
    std::lock_guard<std::mutex> lk(mu);

    FILE *f = std::fopen(path, "a");
    if (!f) return;
    if (!header_written) {
        std::fprintf(f,
                "seq,tensor,expert_idx,offset,nbytes,direct,direct_failed,ok,elapsed_ms\n");
        header_written = true;
    }
    std::fprintf(f,
            "%lu,%s,%d,%lu,%lu,%d,%d,%d,%.6f\n",
            (unsigned long)++seq,
            entry->tensor,
            entry->expert_idx,
            (unsigned long)entry->offset,
            (unsigned long)entry->nbytes,
            direct ? 1 : 0,
            direct_failed ? 1 : 0,
            ok ? 1 : 0,
            elapsed_ms);
    std::fclose(f);
}

static void one_pack_report_atexit() {
    if (!g_one_pack.enabled) {
        return;
    }
    std::fprintf(stderr,
        "[moe_stream] one expert pack: hits=%lu misses=%lu reads=%lu bytes=%lu failures=%lu entries=%zu"
        " direct_enabled=%d direct_reads=%lu direct_failures=%lu direct_fallbacks=%lu\n",
        g_one_pack.hits.load(), g_one_pack.misses.load(), g_one_pack.reads.load(),
        g_one_pack.bytes.load(), g_one_pack.failures.load(), g_one_pack.entries.size(),
        g_one_pack.direct_enabled ? 1 : 0,
        g_one_pack.direct_reads.load(), g_one_pack.direct_failures.load(),
        g_one_pack.direct_fallbacks.load());
    if (one_pack_read_summary_enabled()) {
        const uint64_t direct_reads = g_one_pack.direct_reads.load(std::memory_order_relaxed);
        const uint64_t direct_us = g_one_pack_direct_read_time_us.load(std::memory_order_relaxed);
        const uint64_t buffered_reads = g_one_pack.reads.load(std::memory_order_relaxed) - direct_reads;
        const uint64_t buffered_us = g_one_pack_buffered_read_time_us.load(std::memory_order_relaxed);
        std::fprintf(stderr,
            "[moe_stream] one expert pack read summary: direct_reads=%lu direct_total_ms=%.3f "
            "direct_avg_ms=%.6f direct_max_ms=%.6f buffered_reads=%lu buffered_total_ms=%.3f "
            "buffered_avg_ms=%.6f buffered_max_ms=%.6f\n",
            (unsigned long)direct_reads,
            direct_us / 1000.0,
            direct_reads ? (direct_us / 1000.0) / direct_reads : 0.0,
            g_one_pack_direct_read_max_us.load(std::memory_order_relaxed) / 1000.0,
            (unsigned long)buffered_reads,
            buffered_us / 1000.0,
            buffered_reads ? (buffered_us / 1000.0) / buffered_reads : 0.0,
            g_one_pack_buffered_read_max_us.load(std::memory_order_relaxed) / 1000.0);
    }
    if (g_one_pack_cqe.init_attempted) {
        std::fprintf(stderr,
            "[moe_stream] one expert pack CQE-H2D: ready=%d batches=%lu jobs=%lu cqes=%lu "
            "fallbacks=%lu bytes=%lu submit_ms=%.3f wait_ms=%.3f h2d_enqueue_ms=%.3f "
            "validated=%lu completed=%lu invalid_user_data=%lu invalid_expected_bytes=%lu "
            "io_errors=%lu short_reads=%lu oversized_reads=%lu\n",
            g_one_pack_cqe.ready ? 1 : 0, g_one_pack_cqe.batches, g_one_pack_cqe.jobs,
            g_one_pack_cqe.cqes, g_one_pack_cqe.fallbacks, g_one_pack_cqe.bytes,
            g_one_pack_cqe.submit_us / 1000.0, g_one_pack_cqe.wait_us / 1000.0,
            g_one_pack_cqe.h2d_enqueue_us / 1000.0,
            g_one_pack_cqe.io_ring.completion_counters.validations,
            g_one_pack_cqe.io_ring.completion_counters.completed,
            g_one_pack_cqe.io_ring.completion_counters.invalid_user_data,
            g_one_pack_cqe.io_ring.completion_counters.invalid_expected_bytes,
            g_one_pack_cqe.io_ring.completion_counters.io_errors,
            g_one_pack_cqe.io_ring.completion_counters.short_reads,
            g_one_pack_cqe.io_ring.completion_counters.oversized_reads);
        if (g_one_pack_cqe.ring_inited) {
            ggml_moe_common::object_io_ring_release(g_one_pack_cqe.io_ring);
            g_one_pack_cqe.ring_inited = false;
        }
        if (g_one_pack_cqe.stream) {
            cudaStreamDestroy(g_one_pack_cqe.stream);
            g_one_pack_cqe.stream = nullptr;
        }
    }
    if (g_one_gate_up_prefetch.init_attempted) {
        std::fprintf(stderr,
            "[moe_stream] one gate-up prefetch: sqpoll=%d fixed_file=%d calls=%lu scheduled=%lu placement_skips=%lu resident_skips=%lu "
            "cqes=%lu demand_fallbacks=%lu bytes=%lu submit_ms=%.3f residual_wait_ms=%.3f "
            "h2d_enqueue_ms=%.3f validated=%lu completed=%lu invalid_user_data=%lu "
            "invalid_expected_bytes=%lu io_errors=%lu short_reads=%lu oversized_reads=%lu\n",
            g_one_gate_up_prefetch.sqpoll ? 1 : 0,
            g_one_gate_up_prefetch.fixed_file ? 1 : 0,
            g_one_gate_up_prefetch.calls, g_one_gate_up_prefetch.scheduled,
            g_one_gate_up_prefetch.placement_skips, g_one_gate_up_prefetch.resident_skips,
            g_one_gate_up_prefetch.cqes,
            g_one_gate_up_prefetch.demand_fallbacks, g_one_gate_up_prefetch.bytes,
            g_one_gate_up_prefetch.submit_us / 1000.0,
            g_one_gate_up_prefetch.residual_wait_us / 1000.0,
            g_one_gate_up_prefetch.h2d_enqueue_us / 1000.0,
            g_one_gate_up_prefetch.io_ring.completion_counters.validations,
            g_one_gate_up_prefetch.io_ring.completion_counters.completed,
            g_one_gate_up_prefetch.io_ring.completion_counters.invalid_user_data,
            g_one_gate_up_prefetch.io_ring.completion_counters.invalid_expected_bytes,
            g_one_gate_up_prefetch.io_ring.completion_counters.io_errors,
            g_one_gate_up_prefetch.io_ring.completion_counters.short_reads,
            g_one_gate_up_prefetch.io_ring.completion_counters.oversized_reads);
        if (g_one_gate_up_prefetch.ring_inited) {
            ggml_moe_common::object_io_ring_release(g_one_gate_up_prefetch.io_ring);
            g_one_gate_up_prefetch.ring_inited = false;
        }
        if (g_one_gate_up_prefetch.stream) {
            cudaStreamDestroy(g_one_gate_up_prefetch.stream);
            g_one_gate_up_prefetch.stream = nullptr;
        }
    }
}

static void one_pack_init_once() {
    std::lock_guard<std::mutex> lk(g_one_pack.mu);
    if (g_one_pack.inited) {
        return;
    }
    const char * path = std::getenv("GGML_MOE_STREAM_ONE_EXPERT_PACK");
    if (!path || !path[0]) {
        g_one_pack.inited = true;
        return;
    }
    const int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "[moe_stream] one expert pack: open failed: %s\n", path);
        g_one_pack.inited = true;
        return;
    }

    char magic[16] = {};
    uint32_t version = 0;
    uint32_t header_size = 0;
    uint64_t n_entries = 0;
    uint64_t data_start = 0;
    uint64_t off = 0;
    if (!one_pack_read_exact_fd(fd, magic, sizeof(magic), off)) goto invalid_header;
    off += sizeof(magic);
    if (!one_pack_read_exact_fd(fd, &version, sizeof(version), off)) goto invalid_header;
    off += sizeof(version);
    if (!one_pack_read_exact_fd(fd, &header_size, sizeof(header_size), off)) goto invalid_header;
    off += sizeof(header_size);
    if (!one_pack_read_exact_fd(fd, &n_entries, sizeof(n_entries), off)) goto invalid_header;
    off += sizeof(n_entries);
    if (!one_pack_read_exact_fd(fd, &data_start, sizeof(data_start), off)) goto invalid_header;
    if (std::memcmp(magic, "GGMLMOEPACKv1", 13) != 0 ||
            version != 1 || header_size < 40 || data_start < header_size || n_entries > 10000000ULL) {
        goto invalid_header;
    }

    {
        std::vector<one_expert_pack_entry> entries;
        entries.resize((size_t)n_entries);
        uint64_t index_off = header_size;
        for (uint64_t i = 0; i < n_entries; ++i) {
            uint32_t reserved = 0;
            one_expert_pack_entry & e = entries[(size_t)i];
            if (!one_pack_read_exact_fd(fd, e.tensor, sizeof(e.tensor), index_off)) goto invalid_index;
            index_off += sizeof(e.tensor);
            if (!one_pack_read_exact_fd(fd, &e.expert_idx, sizeof(e.expert_idx), index_off)) goto invalid_index;
            index_off += sizeof(e.expert_idx);
            if (!one_pack_read_exact_fd(fd, &reserved, sizeof(reserved), index_off)) goto invalid_index;
            index_off += sizeof(reserved);
            if (!one_pack_read_exact_fd(fd, &e.offset, sizeof(e.offset), index_off)) goto invalid_index;
            index_off += sizeof(e.offset);
            if (!one_pack_read_exact_fd(fd, &e.nbytes, sizeof(e.nbytes), index_off)) goto invalid_index;
            index_off += sizeof(e.nbytes);
            e.tensor[sizeof(e.tensor) - 1] = '\0';
        }
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
        g_one_pack.fd = fd;
        const char * io_env = std::getenv("GGML_MOE_STREAM_ONE_EXPERT_PACK_IO");
        if (io_env && (std::strcmp(io_env, "direct") == 0 || std::strcmp(io_env, "odirect") == 0)) {
#if defined(__linux__) && defined(O_DIRECT)
            g_one_pack.fd_direct = ::open(path, O_RDONLY | O_DIRECT);
            if (g_one_pack.fd_direct >= 0) {
                g_one_pack.direct_enabled = true;
                std::fprintf(stderr, "[moe_stream] one expert pack: O_DIRECT payload reads enabled: %s\n", path);
            } else {
                std::fprintf(stderr, "[moe_stream] one expert pack: O_DIRECT open failed; using buffered payload reads: %s\n", path);
            }
#else
            std::fprintf(stderr, "[moe_stream] one expert pack: O_DIRECT requested but unavailable; using buffered payload reads\n");
#endif
        }
        g_one_pack.entries = std::move(entries);
        g_one_pack.enabled = true;
        g_one_pack.inited = true;
        std::atexit(one_pack_report_atexit);
        std::fprintf(stderr, "[moe_stream] one expert pack: loaded %zu entries from %s\n", g_one_pack.entries.size(), path);
        return;
    }

invalid_index:
    std::fprintf(stderr, "[moe_stream] one expert pack: short index: %s\n", path);
    ::close(fd);
    g_one_pack.inited = true;
    return;
invalid_header:
    std::fprintf(stderr, "[moe_stream] one expert pack: invalid header: %s\n", path);
    ::close(fd);
    g_one_pack.inited = true;
}

static const one_expert_pack_entry * one_pack_lookup(const char * tensor_name, int64_t expert_idx, size_t nbytes) {
    one_pack_init_once();
    if (!g_one_pack.enabled || !tensor_name || !tensor_name[0]) {
        return nullptr;
    }
    size_t lo = 0;
    size_t hi = g_one_pack.entries.size();
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const one_expert_pack_entry & e = g_one_pack.entries[mid];
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
    if (lo < g_one_pack.entries.size()) {
        const one_expert_pack_entry & e = g_one_pack.entries[lo];
        if (std::strcmp(e.tensor, tensor_name) == 0 && e.expert_idx == expert_idx && e.nbytes == nbytes) {
            ++g_one_pack.hits;
            return &e;
        }
    }
    ++g_one_pack.misses;
    return nullptr;
}

static bool one_pack_read_entry(const one_expert_pack_entry * entry, void * dst, size_t sz) {
    if (!entry || g_one_pack.fd < 0 || entry->nbytes != sz) {
        return false;
    }

    const bool profile = one_pack_read_profile_enabled();
    const bool summary = one_pack_read_summary_enabled();
    if (g_one_pack.fd_direct >= 0) {
        const auto t0 = (profile || summary) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

        const bool direct_ok = one_pack_read_exact_fd(g_one_pack.fd_direct, dst, sz, entry->offset);

        // sz = 4456448
        if (profile || summary) {
            const auto t1 = std::chrono::steady_clock::now();
            const uint64_t elapsed_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            if (summary) {
                one_pack_read_summary_update(g_one_pack_direct_read_time_us, g_one_pack_direct_read_max_us, elapsed_us);
            }
            one_pack_read_profile_record(
                    entry, true, !direct_ok, direct_ok,
                    elapsed_us / 1000.0);
        }
        if (direct_ok) {
            ++g_one_pack.reads;
            ++g_one_pack.direct_reads;
            g_one_pack.bytes.fetch_add(sz);
            return true;
        }
        ++g_one_pack.direct_failures;
        ++g_one_pack.direct_fallbacks;
    }
    const auto t0 = (profile || summary) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool buffered_ok = one_pack_read_exact_fd(g_one_pack.fd, dst, sz, entry->offset);
    if (profile || summary) {
        const auto t1 = std::chrono::steady_clock::now();
        const uint64_t elapsed_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        if (summary) {
            one_pack_read_summary_update(g_one_pack_buffered_read_time_us, g_one_pack_buffered_read_max_us, elapsed_us);
        }
        one_pack_read_profile_record(
                entry, false, g_one_pack.fd_direct >= 0, buffered_ok,
                elapsed_us / 1000.0);
    }
    if (!buffered_ok) {
        ++g_one_pack.failures;
        return false;
    }
    ++g_one_pack.reads;
    g_one_pack.bytes.fetch_add(sz);
    return true;
}

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

static char * one_direct_repr_trim(char * s) {
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') {
        ++s;
    }
    char * e = s + std::strlen(s);
    while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r' || e[-1] == '\n')) {
        *--e = '\0';
    }
    return s;
}

static bool one_direct_repr_parse_u64(const char * s, uint64_t & out) {
    if (!s || !s[0]) {
        return false;
    }
    char * end = nullptr;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s || (end && *one_direct_repr_trim(end) != '\0')) {
        return false;
    }
    out = (uint64_t) v;
    return true;
}

static bool one_direct_repr_parse_i32(const char * s, int32_t & out) {
    if (!s || !s[0]) {
        return false;
    }
    char * end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || v < 0 || v > INT32_MAX || (end && *one_direct_repr_trim(end) != '\0')) {
        return false;
    }
    out = (int32_t) v;
    return true;
}

static bool one_direct_repr_is_exact(const one_direct_repr_entry & e) {
    return std::strcmp(e.repr_type, "exact_mxfp4") == 0 && e.compressed_nbytes == e.original_nbytes;
}

static bool one_direct_repr_is_partial_exact(const one_direct_repr_entry & e) {
    return std::strcmp(e.repr_type, "exact_mxfp4_partial") == 0 &&
        e.compressed_nbytes == e.original_nbytes && e.row_count > 0;
}

static bool one_direct_repr_is_partial_q2tern(const one_direct_repr_entry & e) {
    return std::strcmp(e.repr_type, "mxfp4_q2tern_partial") == 0 &&
        e.compressed_nbytes > 0 && e.original_nbytes > 0 && e.row_count > 0;
}

static bool one_direct_repr_is_partial_resident_candidate(const one_direct_repr_entry & e) {
    return one_direct_repr_is_partial_exact(e) || one_direct_repr_is_partial_q2tern(e);
}

static uint64_t one_direct_repr_env_u64(const char * name, uint64_t fallback) {
    const char * env = std::getenv(name);
    if (!env || !env[0]) {
        return fallback;
    }
    return (uint64_t) std::strtoull(env, nullptr, 10);
}

static FILE * one_direct_repr_report_fp() {
    std::lock_guard<std::mutex> lk(g_one_direct_repr.report_mu);
    if (g_one_direct_repr.report_fp) {
        return g_one_direct_repr.report_fp;
    }
    const char * path = std::getenv("GGML_MOE_STREAM_ONE_DIRECT_REPR_REPORT");
    if (!path || !path[0]) {
        return nullptr;
    }
    g_one_direct_repr.report_fp = std::fopen(path, "w");
    if (!g_one_direct_repr.report_fp) {
        std::fprintf(stderr, "[moe_stream] one direct repr manifest: failed to open report: %s\n", path);
        return nullptr;
    }
    std::setvbuf(g_one_direct_repr.report_fp, nullptr, _IOLBF, 0);
    std::fprintf(g_one_direct_repr.report_fp,
            "event,seq,tensor,expert,row0,row_count,compressed_nbytes,original_nbytes,repr_type,flags,"
            "rows_total,repr_experts,repr_rows,repr_exact_rows,repr_partial_rows,repr_compressed_rows,"
            "repr_src0_bytes_projected,repr_resident_original_bytes,repr_resident_compressed_bytes\n");
    return g_one_direct_repr.report_fp;
}

static void one_direct_repr_report_entry(const one_direct_repr_entry & e) {
    if (FILE * fp = one_direct_repr_report_fp()) {
        flockfile(fp);
        std::fprintf(fp,
                "entry,0,%s,%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%s,%s,0,0,0,0,0,0,0,0,0\n",
                e.tensor,
                e.expert_idx,
                e.row0,
                e.row_count,
                e.compressed_nbytes,
                e.original_nbytes,
                e.repr_type,
                e.flags);
        funlockfile(fp);
    }
}

static void one_direct_repr_report_probe(
        uint64_t seq,
        const char * tensor,
        int64_t rows_total,
        int64_t repr_experts,
        int64_t repr_rows,
        int64_t repr_exact_rows,
        int64_t repr_partial_rows,
        int64_t repr_compressed_rows,
        uint64_t repr_src0_bytes_projected,
        uint64_t repr_resident_original_bytes,
        uint64_t repr_resident_compressed_bytes) {
    if (FILE * fp = one_direct_repr_report_fp()) {
        flockfile(fp);
        std::fprintf(fp,
                "probe,%" PRIu64 ",%s,-1,0,0,0,0,,,%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                seq,
                tensor ? tensor : "",
                rows_total,
                repr_experts,
                repr_rows,
                repr_exact_rows,
                repr_partial_rows,
                repr_compressed_rows,
                repr_src0_bytes_projected,
                repr_resident_original_bytes,
                repr_resident_compressed_bytes);
        funlockfile(fp);
    }
}

static void one_direct_repr_manifest_report_atexit() {
    if (!g_one_direct_repr.enabled) {
        return;
    }
    std::fprintf(stderr,
            "[moe_stream] one direct repr manifest: entries=%zu compressed_bytes=%lu original_bytes=%lu"
            " exact_entries=%lu partial_entries=%lu compressed_entries=%lu hits=%lu misses=%lu"
            " pool_enabled=%d pool_sz=%zu pool_attempted=%lu pool_inserted=%lu pool_bytes=%lu"
            " pool_read_failures=%lu pool_copy_failures=%lu pool_read_ms=%.3f pool_h2d_ms=%.3f direct_enabled=%d sidecar=%d\n",
            g_one_direct_repr.entries.size(),
            g_one_direct_repr.compressed_bytes,
            g_one_direct_repr.original_bytes,
            g_one_direct_repr.exact_entries,
            g_one_direct_repr.partial_entries,
            g_one_direct_repr.compressed_entries,
            g_one_direct_repr.hits.load(),
            g_one_direct_repr.misses.load(),
            g_one_direct_repr.pool_enabled ? 1 : 0,
            g_one_direct_repr.pool_sz,
            g_one_direct_repr.pool_attempted,
            g_one_direct_repr.pool_inserted,
            g_one_direct_repr.pool_bytes,
            g_one_direct_repr.pool_read_failures,
            g_one_direct_repr.pool_copy_failures,
            g_one_direct_repr.pool_read_ms,
            g_one_direct_repr.pool_h2d_ms,
            g_one_direct_repr.direct_enabled ? 1 : 0,
            g_one_direct_repr.payload_sidecar_enabled ? 1 : 0);
    if (g_one_direct_repr.report_fp) {
        std::fclose(g_one_direct_repr.report_fp);
        g_one_direct_repr.report_fp = nullptr;
    }
}

static bool one_direct_repr_manifest_parse_row(char * line, one_direct_repr_entry & entry) {
    char * tokens[16] = {};
    int ntok = 0;
    char * save = nullptr;
    for (char * tok = ::strtok_r(line, ",", &save); tok && ntok < 16; tok = ::strtok_r(nullptr, ",", &save)) {
        tokens[ntok++] = one_direct_repr_trim(tok);
    }
    if (ntok == 0 || !tokens[0][0] || tokens[0][0] == '#') {
        return false;
    }
    if (std::strcmp(tokens[0], "tensor") == 0) {
        return false;
    }

    entry = one_direct_repr_entry{};
    std::snprintf(entry.tensor, sizeof(entry.tensor), "%s", tokens[0]);
    if (!one_direct_repr_parse_i32(ntok > 1 ? tokens[1] : nullptr, entry.expert_idx)) {
        return false;
    }

    if (ntok == 4) {
        uint64_t offset = 0;
        uint64_t nbytes = 0;
        if (!one_direct_repr_parse_u64(tokens[2], offset) || !one_direct_repr_parse_u64(tokens[3], nbytes) || nbytes == 0) {
            return false;
        }
        entry.row0 = 0;
        entry.row_count = 0;
        entry.model_offset = offset;
        entry.compressed_offset = offset;
        entry.compressed_nbytes = nbytes;
        entry.original_nbytes = nbytes;
        std::snprintf(entry.repr_type, sizeof(entry.repr_type), "exact_mxfp4");
        std::snprintf(entry.flags, sizeof(entry.flags), "legacy_direct");
        return true;
    }

    if (ntok < 9) {
        return false;
    }
    if (!one_direct_repr_parse_u64(tokens[2], entry.row0) ||
            !one_direct_repr_parse_u64(tokens[3], entry.row_count) ||
            !one_direct_repr_parse_u64(tokens[4], entry.model_offset) ||
            !one_direct_repr_parse_u64(tokens[5], entry.compressed_offset) ||
            !one_direct_repr_parse_u64(tokens[6], entry.compressed_nbytes) ||
            !one_direct_repr_parse_u64(tokens[7], entry.original_nbytes) ||
            entry.compressed_nbytes == 0 || entry.original_nbytes == 0) {
        return false;
    }
    std::snprintf(entry.repr_type, sizeof(entry.repr_type), "%s", tokens[8]);
    std::snprintf(entry.flags, sizeof(entry.flags), "%s", ntok > 9 ? tokens[9] : "");
    return entry.repr_type[0] != '\0';
}

static void one_direct_repr_manifest_init_once() {
    std::lock_guard<std::mutex> lk(g_one_direct_repr.mu);
    if (g_one_direct_repr.inited) {
        return;
    }
    g_one_direct_repr.inited = true;

    const char * manifest_path = std::getenv("GGML_MOE_STREAM_ONE_DIRECT_REPR_MANIFEST");
    if (!manifest_path || !manifest_path[0]) {
        return;
    }

    FILE * fp = std::fopen(manifest_path, "r");
    if (!fp) {
        std::fprintf(stderr, "[moe_stream] one direct repr manifest: failed to open manifest %s\n", manifest_path);
        return;
    }

    std::vector<one_direct_repr_entry> entries;
    char line[4096];
    while (std::fgets(line, sizeof(line), fp)) {
        one_direct_repr_entry entry;
        if (!one_direct_repr_manifest_parse_row(line, entry)) {
            continue;
        }
        g_one_direct_repr.compressed_bytes += entry.compressed_nbytes;
        g_one_direct_repr.original_bytes += entry.original_nbytes;
        if (one_direct_repr_is_exact(entry)) {
            g_one_direct_repr.exact_entries++;
        } else if (entry.row_count > 0) {
            g_one_direct_repr.partial_entries++;
        } else {
            g_one_direct_repr.compressed_entries++;
        }
        entries.push_back(entry);
    }
    std::fclose(fp);

    if (entries.empty()) {
        std::fprintf(stderr, "[moe_stream] one direct repr manifest: no entries loaded from %s\n", manifest_path);
        return;
    }

    std::sort(entries.begin(), entries.end(), [](const one_direct_repr_entry & a, const one_direct_repr_entry & b) {
        const int name_cmp = std::strcmp(a.tensor, b.tensor);
        if (name_cmp != 0) {
            return name_cmp < 0;
        }
        if (a.expert_idx != b.expert_idx) {
            return a.expert_idx < b.expert_idx;
        }
        if (a.row0 != b.row0) {
            return a.row0 < b.row0;
        }
        return std::strcmp(a.repr_type, b.repr_type) < 0;
    });

    g_one_direct_repr.entries = std::move(entries);
    g_one_direct_repr.pool_offsets.assign(g_one_direct_repr.entries.size(), (size_t) -1);
    g_one_direct_repr.pool_ready.assign(g_one_direct_repr.entries.size(), 0);
    g_one_direct_repr.enabled = true;
    std::atexit(one_direct_repr_manifest_report_atexit);
    std::fprintf(stderr,
            "[moe_stream] one direct repr manifest: loaded %zu entries compressed_bytes=%lu original_bytes=%lu from %s\n",
            g_one_direct_repr.entries.size(),
            g_one_direct_repr.compressed_bytes,
            g_one_direct_repr.original_bytes,
            manifest_path);
    for (const one_direct_repr_entry & e : g_one_direct_repr.entries) {
        one_direct_repr_report_entry(e);
    }
}


static bool one_direct_repr_read_direct_aligned(int fd, uint64_t offset, void * dst, size_t sz) {
#if defined(__linux__) && defined(O_DIRECT)
    const uint64_t align = 4096;
    const uint64_t prefix = offset & (align - 1);
    const uint64_t read_off = offset - prefix;
    const size_t read_sz = (size_t) (((prefix + sz + align - 1) / align) * align);
    void * bounce = nullptr;
    if (posix_memalign(&bounce, (size_t) align, read_sz) != 0 || bounce == nullptr) {
        return false;
    }
    const bool ok = one_pack_read_exact_fd(fd, bounce, read_sz, read_off);
    if (ok) {
        std::memcpy(dst, (const char *) bounce + prefix, sz);
    }
    std::free(bounce);
    return ok;
#else
    (void) fd;
    (void) offset;
    (void) dst;
    (void) sz;
    return false;
#endif
}

static bool one_direct_repr_read_entry(const one_direct_repr_entry & entry, void * dst, size_t sz) {
    if (g_one_direct_repr.fd < 0 || entry.compressed_nbytes != (uint64_t) sz) {
        return false;
    }
    const uint64_t offset = g_one_direct_repr.payload_sidecar_enabled ? entry.compressed_offset : entry.model_offset;
    if (g_one_direct_repr.fd_direct >= 0) {
        if (one_direct_repr_read_direct_aligned(g_one_direct_repr.fd_direct, offset, dst, sz)) {
            return true;
        }
    }
    return one_pack_read_exact_fd(g_one_direct_repr.fd, dst, sz, offset);
}

static void one_direct_repr_pool_init_once() {
    one_direct_repr_manifest_init_once();
    std::lock_guard<std::mutex> lk(g_one_direct_repr.pool_mu);
    if (g_one_direct_repr.pool_inited) {
        return;
    }
    g_one_direct_repr.pool_inited = true;

    if (!g_one_direct_repr.enabled || g_one_direct_repr.entries.empty()) {
        return;
    }
    const uint64_t budget_mib = one_direct_repr_env_u64("GGML_MOE_STREAM_ONE_DIRECT_REPR_POOL_MIB", 0);
    if (budget_mib == 0) {
        return;
    }
    const char * payload_path = std::getenv("GGML_MOE_STREAM_ONE_DIRECT_REPR_PAYLOAD");
    const bool use_payload = payload_path && payload_path[0];
    const char * model_path = std::getenv("GGML_MOE_STREAM_ONE_DIRECT_MODEL");
    const char * source_path = use_payload ? payload_path : model_path;
    if (!source_path || !source_path[0]) {
        std::fprintf(stderr, "[moe_stream] one direct repr pool: GGML_MOE_STREAM_ONE_DIRECT_MODEL or GGML_MOE_STREAM_ONE_DIRECT_REPR_PAYLOAD is required\n");
        return;
    }

    const int fd = ::open(source_path, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "[moe_stream] one direct repr pool: payload/model open failed: %s\n", source_path);
        return;
    }
    g_one_direct_repr.fd = fd;
    g_one_direct_repr.payload_sidecar_enabled = use_payload;
    const char * io_env = std::getenv("GGML_MOE_STREAM_ONE_DIRECT_IO");
    if (io_env && (std::strcmp(io_env, "direct") == 0 || std::strcmp(io_env, "odirect") == 0)) {
#if defined(__linux__) && defined(O_DIRECT)
        g_one_direct_repr.fd_direct = ::open(source_path, O_RDONLY | O_DIRECT);
        if (g_one_direct_repr.fd_direct >= 0) {
            g_one_direct_repr.direct_enabled = true;
        }
#endif
    }

    const size_t budget = (size_t) budget_mib * 1024ULL * 1024ULL;
    void * pool = nullptr;
    if (cudaSetDevice(0) != cudaSuccess || cudaMalloc(&pool, budget) != cudaSuccess) {
        std::fprintf(stderr, "[moe_stream] one direct repr pool: cudaMalloc failed for %.2f MiB\n",
                budget / (1024.0 * 1024.0));
        return;
    }
    g_one_direct_repr.pool = pool;
    g_one_direct_repr.pool_sz = budget;
    g_one_direct_repr.pool_enabled = true;

    uint64_t limit = one_direct_repr_env_u64("GGML_MOE_STREAM_ONE_DIRECT_REPR_PREFILL_LIMIT", 0);
    if (limit == 0) {
        limit = UINT64_MAX;
    }
    size_t used = 0;
    void * h_tmp = nullptr;
    size_t h_tmp_sz = 0;
    for (size_t i = 0; i < g_one_direct_repr.entries.size() && g_one_direct_repr.pool_attempted < limit; ++i) {
        const one_direct_repr_entry & e = g_one_direct_repr.entries[i];
        if (!one_direct_repr_is_partial_resident_candidate(e)) {
            continue;
        }
        g_one_direct_repr.pool_attempted++;
        const size_t sz = (size_t) e.compressed_nbytes;
        if (sz == 0 || used + sz > budget) {
            break;
        }
        if (h_tmp_sz < sz) {
            std::free(h_tmp);
            h_tmp = std::malloc(sz);
            h_tmp_sz = h_tmp ? sz : 0;
        }
        if (!h_tmp) {
            g_one_direct_repr.pool_read_failures++;
            continue;
        }
        const auto t_read0 = std::chrono::steady_clock::now();
        if (!one_direct_repr_read_entry(e, h_tmp, sz)) {
            g_one_direct_repr.pool_read_failures++;
            continue;
        }
        const auto t_read1 = std::chrono::steady_clock::now();
        g_one_direct_repr.pool_read_ms += std::chrono::duration<double, std::milli>(t_read1 - t_read0).count();
        const auto t_h2d0 = std::chrono::steady_clock::now();
        if (cudaMemcpy((char *) g_one_direct_repr.pool + used, h_tmp, sz, cudaMemcpyHostToDevice) != cudaSuccess) {
            g_one_direct_repr.pool_copy_failures++;
            continue;
        }
        const auto t_h2d1 = std::chrono::steady_clock::now();
        g_one_direct_repr.pool_h2d_ms += std::chrono::duration<double, std::milli>(t_h2d1 - t_h2d0).count();
        g_one_direct_repr.pool_offsets[i] = used;
        g_one_direct_repr.pool_ready[i] = 1;
        g_one_direct_repr.pool_inserted++;
        g_one_direct_repr.pool_bytes += sz;
        used += sz;
    }
    std::free(h_tmp);
    std::fprintf(stderr,
            "[moe_stream] one direct repr pool: allocated %.2f MiB attempted=%lu inserted=%lu bytes=%lu read_ms=%.3f h2d_ms=%.3f direct_enabled=%d sidecar=%d\n",
            budget / (1024.0 * 1024.0),
            g_one_direct_repr.pool_attempted,
            g_one_direct_repr.pool_inserted,
            g_one_direct_repr.pool_bytes,
            g_one_direct_repr.pool_read_ms,
            g_one_direct_repr.pool_h2d_ms,
            g_one_direct_repr.direct_enabled ? 1 : 0,
            g_one_direct_repr.payload_sidecar_enabled ? 1 : 0);
}

struct one_direct_repr_partial_hit {
    const one_direct_repr_entry * entry = nullptr;
    const char * d_src0 = nullptr;
    bool q2tern = false;
};

static void one_direct_repr_find_partial_hits(
        const char * tensor_name,
        int64_t expert_idx,
        std::vector<one_direct_repr_partial_hit> & hits) {
    one_direct_repr_pool_init_once();
    if (!g_one_direct_repr.pool_enabled || !g_one_direct_repr.pool || !tensor_name || !tensor_name[0]) {
        return;
    }
    one_direct_repr_entry key;
    std::snprintf(key.tensor, sizeof(key.tensor), "%s", tensor_name);
    key.expert_idx = (int32_t) expert_idx;
    auto less_key = [](const one_direct_repr_entry & a, const one_direct_repr_entry & b) {
        const int name_cmp = std::strcmp(a.tensor, b.tensor);
        if (name_cmp != 0) {
            return name_cmp < 0;
        }
        if (a.expert_idx != b.expert_idx) {
            return a.expert_idx < b.expert_idx;
        }
        if (a.row0 != b.row0) {
            return a.row0 < b.row0;
        }
        return std::strcmp(a.repr_type, b.repr_type) < 0;
    };
    auto it = std::lower_bound(g_one_direct_repr.entries.begin(), g_one_direct_repr.entries.end(), key, less_key);
    for (; it != g_one_direct_repr.entries.end(); ++it) {
        const one_direct_repr_entry & e = *it;
        if (std::strcmp(e.tensor, tensor_name) != 0 || e.expert_idx != expert_idx) {
            break;
        }
        const size_t idx = (size_t) (it - g_one_direct_repr.entries.begin());
        if (one_direct_repr_is_partial_resident_candidate(e) && idx < g_one_direct_repr.pool_ready.size() && g_one_direct_repr.pool_ready[idx]) {
            hits.push_back({&e, (const char *) g_one_direct_repr.pool + g_one_direct_repr.pool_offsets[idx], one_direct_repr_is_partial_q2tern(e)});
        }
    }
}

struct one_direct_repr_probe_match {
    int entries = 0;
    bool has_exact = false;
    bool has_partial = false;
    bool has_compressed = false;
    uint64_t resident_compressed_bytes = 0;
    uint64_t resident_original_bytes = 0;
    uint64_t src0_bytes_projected = 0;
};

static bool one_direct_repr_manifest_match(
        const char * tensor_name,
        int64_t expert_idx,
        int64_t ne01,
        size_t nb01,
        int64_t cne1,
        one_direct_repr_probe_match & match) {
    one_direct_repr_manifest_init_once();
    if (!g_one_direct_repr.enabled || !tensor_name || !tensor_name[0]) {
        return false;
    }
    one_direct_repr_entry key;
    std::snprintf(key.tensor, sizeof(key.tensor), "%s", tensor_name);
    key.expert_idx = (int32_t) expert_idx;
    auto less_key = [](const one_direct_repr_entry & a, const one_direct_repr_entry & b) {
        const int name_cmp = std::strcmp(a.tensor, b.tensor);
        if (name_cmp != 0) {
            return name_cmp < 0;
        }
        if (a.expert_idx != b.expert_idx) {
            return a.expert_idx < b.expert_idx;
        }
        if (a.row0 != b.row0) {
            return a.row0 < b.row0;
        }
        return std::strcmp(a.repr_type, b.repr_type) < 0;
    };
    bool found = false;
    auto it = std::lower_bound(g_one_direct_repr.entries.begin(), g_one_direct_repr.entries.end(), key, less_key);
    for (; it != g_one_direct_repr.entries.end(); ++it) {
        const one_direct_repr_entry & e = *it;
        if (std::strcmp(e.tensor, tensor_name) != 0 || e.expert_idx != expert_idx) {
            break;
        }
        found = true;
        match.entries++;
        match.resident_compressed_bytes += e.compressed_nbytes;
        match.resident_original_bytes += e.original_nbytes;
        const uint64_t cols = e.row_count == 0 ? (uint64_t) ne01 : std::min<uint64_t>((uint64_t) ne01, e.row_count);
        match.src0_bytes_projected += (uint64_t) cne1 * cols * (uint64_t) nb01;
        if (one_direct_repr_is_exact(e)) {
            match.has_exact = true;
        } else if (e.row_count > 0) {
            match.has_partial = true;
        } else {
            match.has_compressed = true;
        }
        if (e.compressed_nbytes < e.original_nbytes || std::strcmp(e.repr_type, "exact_mxfp4") != 0) {
            match.has_compressed = true;
        }
    }
    if (found) {
        ++g_one_direct_repr.hits;
    } else {
        ++g_one_direct_repr.misses;
    }
    return found;
}

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

static bool one_direct_manifest_read_direct_aligned(const one_expert_pack_entry * entry, void * dst, size_t sz) {
#if defined(__linux__) && defined(O_DIRECT)
    const uint64_t align = 4096;
    const uint64_t prefix = entry->offset & (align - 1);
    const uint64_t read_off = entry->offset - prefix;
    const size_t read_sz = (size_t) (((prefix + sz + align - 1) / align) * align);
    void * bounce = nullptr;
    if (posix_memalign(&bounce, (size_t) align, read_sz) != 0 || bounce == nullptr) {
        return false;
    }
    const bool ok = one_pack_read_exact_fd(g_one_direct_manifest.fd_direct, bounce, read_sz, read_off);
    if (ok) {
        std::memcpy(dst, (const char *) bounce + prefix, sz);
    }
    std::free(bounce);
    return ok;
#else
    (void) entry;
    (void) dst;
    (void) sz;
    return false;
#endif
}

[[maybe_unused]] static bool one_direct_manifest_read_entry(const one_expert_pack_entry * entry, void * dst, size_t sz) {
    if (!entry || g_one_direct_manifest.fd < 0 || entry->nbytes != sz) {
        return false;
    }
    if (g_one_direct_manifest.fd_direct >= 0) {
        if (one_direct_manifest_read_direct_aligned(entry, dst, sz)) {
            ++g_one_direct_manifest.reads;
            ++g_one_direct_manifest.direct_reads;
            g_one_direct_manifest.bytes.fetch_add(sz);
            return true;
        }
        ++g_one_direct_manifest.direct_failures;
        ++g_one_direct_manifest.direct_fallbacks;
    }
    if (!one_pack_read_exact_fd(g_one_direct_manifest.fd, dst, sz, entry->offset)) {
        ++g_one_direct_manifest.failures;
        return false;
    }
    ++g_one_direct_manifest.reads;
    g_one_direct_manifest.bytes.fetch_add(sz);
    return true;
}

static bool ensure_host_pinned(void *&p, size_t &cur, size_t need);

static __global__ void moe_stream_mxfp4_transpose_blocks_kernel(
        const block_mxfp4 * __restrict__ src,
        block_mxfp4 * __restrict__ dst,
        int ne01,
        int nb) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ne01 * nb;
    if (idx >= total) {
        return;
    }
    const int ib = idx / ne01;
    const int col = idx - ib * ne01;
    dst[idx] = src[col * nb + ib];
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

static bool one_direct_hot_pool_async_prefill_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_MOE_STREAM_ONE_DIRECT_PREFILL_ASYNC");
        enabled = env && env[0] && std::strcmp(env, "0") != 0 ? 1 : 0;
    }
    return enabled != 0;
}

static bool one_direct_hot_pool_transpose_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_MOE_STREAM_ONE_DIRECT_TRANSPOSE");
        enabled = env && env[0] && std::strcmp(env, "0") != 0 ? 1 : 0;
    }
    return enabled != 0;
}

static void one_direct_hot_pool_join_async() {
    if (g_one_direct_hot_pool.async_thread.joinable()) {
        g_one_direct_hot_pool.async_thread.join();
    }
}

static void one_direct_hot_pool_report_atexit() {
    one_direct_hot_pool_join_async();
    if (!g_one_direct_hot_pool.enabled && g_one_direct_hot_pool.attempted == 0) {
        return;
    }
    std::fprintf(stderr,
        "[moe_stream] one direct hot pool: enabled=%d slots=%d slot_sz=%zu pool_sz=%zu"
        " attempted=%lu inserted=%lu read_failures=%lu copy_failures=%lu transform_failures=%lu bytes=%lu"
        " elapsed_ms=%.3f read_ms=%.3f h2d_ms=%.3f transform_ms=%.3f transposed=%d"
        " lookups=%lu hits=%lu async_started=%d async_done=%d\n",
        g_one_direct_hot_pool.enabled ? 1 : 0,
        g_one_direct_hot_pool.n_slots,
        g_one_direct_hot_pool.slot_sz,
        g_one_direct_hot_pool.pool_sz,
        g_one_direct_hot_pool.attempted,
        g_one_direct_hot_pool.inserted,
        g_one_direct_hot_pool.read_failures,
        g_one_direct_hot_pool.copy_failures,
        g_one_direct_hot_pool.transform_failures,
        g_one_direct_hot_pool.bytes,
        g_one_direct_hot_pool.prefill_elapsed_ms,
        g_one_direct_hot_pool.read_elapsed_ms,
        g_one_direct_hot_pool.h2d_elapsed_ms,
        g_one_direct_hot_pool.transform_elapsed_ms,
        g_one_direct_hot_pool.transposed ? 1 : 0,
        g_one_direct_hot_pool.lookups.load(),
        g_one_direct_hot_pool.hits.load(),
        g_one_direct_hot_pool.async_started.load(std::memory_order_acquire) ? 1 : 0,
        g_one_direct_hot_pool.async_done.load(std::memory_order_acquire) ? 1 : 0);
}

static uint64_t one_direct_hot_pool_env_u64(const char * name, uint64_t fallback) {
    const char * env = std::getenv(name);
    if (!env || !env[0]) {
        return fallback;
    }
    return (uint64_t) std::strtoull(env, nullptr, 10);
}

static bool gate_batch_prefetch_enabled() {
    static int enabled = [] {
        const char * env = std::getenv("GGML_MOE_GATE_BATCH_PREFETCH");
        return env && env[0] && env[0] != '0' ? 1 : 0;
    }();
    return enabled != 0;
}

static std::string one_direct_hot_pool_key(const char * tensor, int64_t expert) {
    std::string key = tensor ? tensor : "";
    key.push_back('\n');
    key += std::to_string((long long) expert);
    return key;
}

static bool one_direct_hot_pool_requested() {
    static int requested = [] {
        const char * pool_mib = std::getenv("GGML_MOE_STREAM_ONE_DIRECT_POOL_MIB");
        return pool_mib && pool_mib[0] && std::strcmp(pool_mib, "0") != 0 ? 1 : 0;
    }();
    return requested != 0;
}

static void one_direct_hot_pool_reset_lookup_locked() {
    g_one_direct_hot_pool.slot_lookup.clear();
    g_one_direct_hot_pool.lookup_built = false;
}

static void one_direct_hot_pool_init_once() {
    std::lock_guard<std::mutex> lk(g_one_direct_hot_pool.mu);
    if (g_one_direct_hot_pool.inited) {
        return;
    }
    g_one_direct_hot_pool.inited = true;
    std::atexit(one_direct_hot_pool_report_atexit);

    const uint64_t budget_mib = one_direct_hot_pool_env_u64("GGML_MOE_STREAM_ONE_DIRECT_POOL_MIB", 0);
    if (budget_mib == 0) {
        return;
    }

    one_direct_manifest_init_once();
    if (!g_one_direct_manifest.enabled || g_one_direct_manifest.entries.empty()) {
        std::fprintf(stderr, "[moe_stream] one direct hot pool: direct manifest is not enabled\n");
        return;
    }

    const std::vector<one_expert_pack_entry> & prefill_entries =
        g_one_direct_manifest.prefill_entries.empty() ? g_one_direct_manifest.entries : g_one_direct_manifest.prefill_entries;

    size_t slot_sz = 0;
    for (const one_expert_pack_entry & e : prefill_entries) {
        slot_sz = std::max(slot_sz, (size_t) e.nbytes);
    }
    if (slot_sz == 0) {
        return;
    }

    const size_t budget = (size_t) budget_mib * 1024ULL * 1024ULL;
    int n_slots = (int) std::min<uint64_t>(prefill_entries.size(), budget / slot_sz);
    if (n_slots <= 0) {
        std::fprintf(stderr, "[moe_stream] one direct hot pool: budget too small budget_mib=%lu slot_sz=%zu\n",
                budget_mib, slot_sz);
        return;
    }

    const size_t alloc = (size_t) n_slots * slot_sz;
    void * pool = nullptr;
    if (cudaMalloc(&pool, alloc) != cudaSuccess) {
        std::fprintf(stderr, "[moe_stream] one direct hot pool: cudaMalloc failed for %.2f MiB (%d slots)\n",
                alloc / (1024.0 * 1024.0), n_slots);
        return;
    }

    g_one_direct_hot_pool.pool = pool;
    g_one_direct_hot_pool.pool_sz = alloc;
    g_one_direct_hot_pool.slot_sz = slot_sz;
    g_one_direct_hot_pool.n_slots = n_slots;
    g_one_direct_hot_pool.slot_entries.resize((size_t) n_slots);
    g_one_direct_hot_pool.transposed = one_direct_hot_pool_transpose_enabled();
    one_direct_hot_pool_reset_lookup_locked();
    g_one_direct_hot_pool.enabled = true;
    std::fprintf(stderr, "[moe_stream] one direct hot pool: allocated %.2f MiB slots=%d slot_sz=%zu transposed=%d\n",
            alloc / (1024.0 * 1024.0), n_slots, slot_sz, g_one_direct_hot_pool.transposed ? 1 : 0);
}

static const void * one_direct_hot_pool_lookup_dev_ptr(const char * tensor, int64_t expert, bool * pool_ready) {
    if (!one_direct_hot_pool_requested()) {
        if (pool_ready) {
            *pool_ready = false;
        }
        return nullptr;
    }
    one_direct_hot_pool_init_once();
    std::lock_guard<std::mutex> lk(g_one_direct_hot_pool.mu);
    const bool ready = g_one_direct_hot_pool.enabled &&
        g_one_direct_hot_pool.prefill_done &&
        (!g_one_direct_hot_pool.async_started.load(std::memory_order_acquire) ||
         g_one_direct_hot_pool.async_done.load(std::memory_order_acquire));
    if (pool_ready) {
        *pool_ready = ready;
    }
    if (!ready || !g_one_direct_hot_pool.pool) {
        return nullptr;
    }
    g_one_direct_hot_pool.lookups.fetch_add(1, std::memory_order_relaxed);
    if (!g_one_direct_hot_pool.lookup_built) {
        g_one_direct_hot_pool.slot_lookup.clear();
        for (int i = 0; i < g_one_direct_hot_pool.n_slots; ++i) {
            const one_expert_pack_entry & e = g_one_direct_hot_pool.slot_entries[(size_t) i];
            if (e.tensor[0] && e.nbytes > 0) {
                g_one_direct_hot_pool.slot_lookup[one_direct_hot_pool_key(e.tensor, e.expert_idx)] = i;
            }
        }
        g_one_direct_hot_pool.lookup_built = true;
    }
    auto it = g_one_direct_hot_pool.slot_lookup.find(one_direct_hot_pool_key(tensor, expert));
    if (it == g_one_direct_hot_pool.slot_lookup.end()) {
        return nullptr;
    }
    g_one_direct_hot_pool.hits.fetch_add(1, std::memory_order_relaxed);
    return (const char *) g_one_direct_hot_pool.pool + (size_t) it->second * g_one_direct_hot_pool.slot_sz;
}

static const void * one_direct_hot_pool_lookup_stream_one_dev_ptr(const char * tensor, int64_t expert, bool * pool_ready) {
    const void * ptr = one_direct_hot_pool_lookup_dev_ptr(tensor, expert, pool_ready);
    if (!ptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lk(g_one_direct_hot_pool.mu);
    if (g_one_direct_hot_pool.transposed) {
        return nullptr;
    }
    return ptr;
}

static bool one_direct_hot_pool_entry_shape(const one_expert_pack_entry & e, int & ne01, int & nb) {
    if (std::strstr(e.tensor, "ffn_up_exps") != nullptr) {
        ne01 = 2048;
        nb = 4096 / QK_MXFP4;
    } else if (std::strstr(e.tensor, "ffn_down_exps") != nullptr) {
        ne01 = 4096;
        nb = 2048 / QK_MXFP4;
    } else {
        return false;
    }
    return e.nbytes == (uint64_t) ne01 * (uint64_t) nb * sizeof(block_mxfp4);
}

static bool one_direct_hot_pool_copy_entry(
        const one_expert_pack_entry & e,
        const void * h_src,
        void * dst,
        void * d_stage,
        cudaStream_t st,
        bool have_stream) {
    if (!g_one_direct_hot_pool.transposed) {
        const auto t0 = std::chrono::steady_clock::now();
        cudaError_t copy_err = cudaSuccess;
        if (have_stream) {
            copy_err = cudaMemcpyAsync(dst, h_src, (size_t) e.nbytes, cudaMemcpyHostToDevice, st);
            if (copy_err == cudaSuccess) {
                copy_err = cudaStreamSynchronize(st);
            }
        } else {
            copy_err = cudaMemcpy(dst, h_src, (size_t) e.nbytes, cudaMemcpyHostToDevice);
        }
        const auto t1 = std::chrono::steady_clock::now();
        g_one_direct_hot_pool.h2d_elapsed_ms +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        return copy_err == cudaSuccess;
    }

    int ne01 = 0;
    int nb = 0;
    if (!d_stage || !one_direct_hot_pool_entry_shape(e, ne01, nb)) {
        return false;
    }

    cudaError_t err = cudaSuccess;
    if (have_stream) {
        cudaEvent_t ev0 = nullptr;
        cudaEvent_t ev1 = nullptr;
        cudaEvent_t ev2 = nullptr;
        const bool have_events =
            cudaEventCreate(&ev0) == cudaSuccess &&
            cudaEventCreate(&ev1) == cudaSuccess &&
            cudaEventCreate(&ev2) == cudaSuccess;
        if (have_events) {
            cudaEventRecord(ev0, st);
        }
        err = cudaMemcpyAsync(d_stage, h_src, (size_t) e.nbytes, cudaMemcpyHostToDevice, st);
        if (err == cudaSuccess && have_events) {
            cudaEventRecord(ev1, st);
        }
        if (err == cudaSuccess) {
            const int threads = 256;
            const int blocks = (ne01 * nb + threads - 1) / threads;
            moe_stream_mxfp4_transpose_blocks_kernel<<<blocks, threads, 0, st>>>(
                    (const block_mxfp4 *) d_stage, (block_mxfp4 *) dst, ne01, nb);
            err = cudaGetLastError();
        }
        if (err == cudaSuccess && have_events) {
            cudaEventRecord(ev2, st);
        }
        if (err == cudaSuccess) {
            err = cudaStreamSynchronize(st);
        }
        if (err == cudaSuccess && have_events) {
            float h2d_ms = 0.0f;
            float transform_ms = 0.0f;
            if (cudaEventElapsedTime(&h2d_ms, ev0, ev1) == cudaSuccess) {
                g_one_direct_hot_pool.h2d_elapsed_ms += h2d_ms;
            }
            if (cudaEventElapsedTime(&transform_ms, ev1, ev2) == cudaSuccess) {
                g_one_direct_hot_pool.transform_elapsed_ms += transform_ms;
            }
        }
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        if (ev2) cudaEventDestroy(ev2);
    } else {
        const auto t0 = std::chrono::steady_clock::now();
        err = cudaMemcpy(d_stage, h_src, (size_t) e.nbytes, cudaMemcpyHostToDevice);
        const auto t1 = std::chrono::steady_clock::now();
        if (err == cudaSuccess) {
            const int threads = 256;
            const int blocks = (ne01 * nb + threads - 1) / threads;
            moe_stream_mxfp4_transpose_blocks_kernel<<<blocks, threads>>>(
                    (const block_mxfp4 *) d_stage, (block_mxfp4 *) dst, ne01, nb);
            err = cudaGetLastError();
        }
        if (err == cudaSuccess) {
            err = cudaDeviceSynchronize();
        }
        const auto t2 = std::chrono::steady_clock::now();
        g_one_direct_hot_pool.h2d_elapsed_ms +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        g_one_direct_hot_pool.transform_elapsed_ms +=
            std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
    return err == cudaSuccess;
}

static void one_direct_hot_pool_prefill_worker(uint64_t limit) {
    const auto t0 = std::chrono::steady_clock::now();

    void * h_tmp = nullptr;
    bool h_tmp_pinned = false;
    if (cudaHostAlloc(&h_tmp, g_one_direct_hot_pool.slot_sz, cudaHostAllocDefault) == cudaSuccess) {
        h_tmp_pinned = true;
    } else {
        h_tmp = std::malloc(g_one_direct_hot_pool.slot_sz);
    }

    cudaStream_t st = nullptr;
    const bool have_stream = cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) == cudaSuccess;
    void * d_stage = nullptr;
    if (g_one_direct_hot_pool.transposed) {
        if (cudaMalloc(&d_stage, g_one_direct_hot_pool.slot_sz) != cudaSuccess) {
            d_stage = nullptr;
        }
    }

    if (!h_tmp || (g_one_direct_hot_pool.transposed && !d_stage)) {
        g_one_direct_hot_pool.read_failures += limit;
    } else {
        const std::vector<one_expert_pack_entry> & prefill_entries =
            g_one_direct_manifest.prefill_entries.empty() ? g_one_direct_manifest.entries : g_one_direct_manifest.prefill_entries;
        for (uint64_t i = 0; i < limit; ++i) {
            const one_expert_pack_entry & e = prefill_entries[(size_t) i];
            g_one_direct_hot_pool.attempted++;
            if (e.nbytes > g_one_direct_hot_pool.slot_sz) {
                g_one_direct_hot_pool.read_failures++;
                continue;
            }
            const auto t_read0 = std::chrono::steady_clock::now();
            if (!one_direct_manifest_read_entry(&e, h_tmp, (size_t) e.nbytes)) {
                g_one_direct_hot_pool.read_failures++;
                continue;
            }
            const auto t_read1 = std::chrono::steady_clock::now();
            g_one_direct_hot_pool.read_elapsed_ms +=
                std::chrono::duration<double, std::milli>(t_read1 - t_read0).count();
            void * dst = (char *) g_one_direct_hot_pool.pool + (size_t) i * g_one_direct_hot_pool.slot_sz;
            if (!one_direct_hot_pool_copy_entry(e, h_tmp, dst, d_stage, st, have_stream)) {
                if (g_one_direct_hot_pool.transposed) {
                    g_one_direct_hot_pool.transform_failures++;
                } else {
                    g_one_direct_hot_pool.copy_failures++;
                }
                continue;
            }
            g_one_direct_hot_pool.slot_entries[(size_t) i] = e;
            g_one_direct_hot_pool.inserted++;
            g_one_direct_hot_pool.bytes += e.nbytes;
        }
    }

    if (d_stage) {
        cudaFree(d_stage);
    }
    if (have_stream) {
        cudaStreamDestroy(st);
    }
    if (h_tmp_pinned) {
        cudaFreeHost(h_tmp);
    } else {
        std::free(h_tmp);
    }

    const auto t1 = std::chrono::steady_clock::now();
    g_one_direct_hot_pool.prefill_elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    g_one_direct_hot_pool.async_done.store(true, std::memory_order_release);
    std::fprintf(stderr,
        "[moe_stream] one direct hot pool: async prefill completed attempted=%lu inserted=%lu bytes=%lu"
        " elapsed_ms=%.3f read_ms=%.3f h2d_ms=%.3f transform_ms=%.3f transposed=%d\n",
        g_one_direct_hot_pool.attempted,
        g_one_direct_hot_pool.inserted,
        g_one_direct_hot_pool.bytes,
        g_one_direct_hot_pool.prefill_elapsed_ms,
        g_one_direct_hot_pool.read_elapsed_ms,
        g_one_direct_hot_pool.h2d_elapsed_ms,
        g_one_direct_hot_pool.transform_elapsed_ms,
        g_one_direct_hot_pool.transposed ? 1 : 0);
}

static void one_direct_hot_pool_prefill_maybe(slot_ctx & ctx, cudaStream_t st) {
    one_direct_hot_pool_init_once();
    std::lock_guard<std::mutex> lk(g_one_direct_hot_pool.mu);
    if (g_one_direct_hot_pool.prefill_done || !g_one_direct_hot_pool.enabled) {
        return;
    }

    uint64_t limit = one_direct_hot_pool_env_u64("GGML_MOE_STREAM_ONE_DIRECT_PREFILL_LIMIT", 0);
    if (limit == 0) {
        return;
    }
    g_one_direct_hot_pool.prefill_done = true;
    if (limit > (uint64_t) g_one_direct_hot_pool.n_slots) {
        limit = (uint64_t) g_one_direct_hot_pool.n_slots;
    }
    const std::vector<one_expert_pack_entry> & prefill_entries =
        g_one_direct_manifest.prefill_entries.empty() ? g_one_direct_manifest.entries : g_one_direct_manifest.prefill_entries;
    if (limit > (uint64_t) prefill_entries.size()) {
        limit = (uint64_t) prefill_entries.size();
    }

    if (one_direct_hot_pool_async_prefill_enabled()) {
        g_one_direct_hot_pool.async_started.store(true, std::memory_order_release);
        g_one_direct_hot_pool.async_done.store(false, std::memory_order_release);
        g_one_direct_hot_pool.async_thread = std::thread(one_direct_hot_pool_prefill_worker, limit);
        std::fprintf(stderr,
            "[moe_stream] one direct hot pool: async prefill started limit=%lu\n",
            limit);
        return;
    }

    if (!ensure_host_pinned(ctx.h_src0_pack, ctx.h_src0_pack_sz, g_one_direct_hot_pool.slot_sz)) {
        g_one_direct_hot_pool.read_failures += limit;
        return;
    }

    void * d_stage = nullptr;
    if (g_one_direct_hot_pool.transposed) {
        if (cudaMalloc(&d_stage, g_one_direct_hot_pool.slot_sz) != cudaSuccess) {
            g_one_direct_hot_pool.transform_failures += limit;
            return;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < limit; ++i) {
        const one_expert_pack_entry & e = prefill_entries[(size_t) i];
        g_one_direct_hot_pool.attempted++;
        if (e.nbytes > g_one_direct_hot_pool.slot_sz) {
            g_one_direct_hot_pool.read_failures++;
            continue;
        }
        const auto t_read0 = std::chrono::steady_clock::now();
        if (!one_direct_manifest_read_entry(&e, ctx.h_src0_pack, (size_t) e.nbytes)) {
            g_one_direct_hot_pool.read_failures++;
            continue;
        }
        const auto t_read1 = std::chrono::steady_clock::now();
        g_one_direct_hot_pool.read_elapsed_ms +=
            std::chrono::duration<double, std::milli>(t_read1 - t_read0).count();
        void * dst = (char *) g_one_direct_hot_pool.pool + (size_t) i * g_one_direct_hot_pool.slot_sz;
        if (!one_direct_hot_pool_copy_entry(e, ctx.h_src0_pack, dst, d_stage, st, true)) {
            if (g_one_direct_hot_pool.transposed) {
                g_one_direct_hot_pool.transform_failures++;
            } else {
                g_one_direct_hot_pool.copy_failures++;
            }
            continue;
        }
        g_one_direct_hot_pool.slot_entries[(size_t) i] = e;
        g_one_direct_hot_pool.inserted++;
        g_one_direct_hot_pool.bytes += e.nbytes;
    }
    if (d_stage) {
        cudaFree(d_stage);
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) {
        g_one_direct_hot_pool.copy_failures++;
    }
    const auto t1 = std::chrono::steady_clock::now();
    g_one_direct_hot_pool.prefill_elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr,
        "[moe_stream] one direct hot pool: prefill attempted=%lu inserted=%lu bytes=%lu"
        " elapsed_ms=%.3f read_ms=%.3f h2d_ms=%.3f transform_ms=%.3f transposed=%d\n",
        g_one_direct_hot_pool.attempted,
        g_one_direct_hot_pool.inserted,
        g_one_direct_hot_pool.bytes,
        g_one_direct_hot_pool.prefill_elapsed_ms,
        g_one_direct_hot_pool.read_elapsed_ms,
        g_one_direct_hot_pool.h2d_elapsed_ms,
        g_one_direct_hot_pool.transform_elapsed_ms,
        g_one_direct_hot_pool.transposed ? 1 : 0);
}

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

static void one_prefill_report_atexit() {
    if (!g_one_prefill.enabled && g_one_prefill.attempted == 0) {
        return;
    }
    std::fprintf(stderr,
        "[moe_stream] one prefill: enabled=%d loaded=%zu limit=%zu attempted=%lu inserted=%lu"
        " pack_misses=%lu read_failures=%lu bytes=%lu elapsed_ms=%.3f\n",
        g_one_prefill.enabled ? 1 : 0,
        g_one_prefill.entries.size(),
        g_one_prefill.limit,
        g_one_prefill.attempted,
        g_one_prefill.inserted,
        g_one_prefill.pack_misses,
        g_one_prefill.read_failures,
        g_one_prefill.bytes,
        g_one_prefill.elapsed_ms);
}

static void one_prefill_init_once_locked() {
    if (g_one_prefill.initialized) {
        return;
    }
    g_one_prefill.initialized = true;

    const char * path = std::getenv("GGML_MOE_STREAM_ONE_PREFILL_PROFILE");
    if (!path || !path[0]) {
        return;
    }

    const char * limit_env = std::getenv("GGML_MOE_STREAM_ONE_PREFILL_LIMIT");
    g_one_prefill.limit = (limit_env && limit_env[0]) ? (size_t) std::strtoull(limit_env, nullptr, 10) : 0;

    FILE * fp = std::fopen(path, "r");
    if (!fp) {
        std::fprintf(stderr, "[moe_stream] one prefill: failed to open profile %s\n", path);
        return;
    }

    char line[4096];
    while (std::fgets(line, sizeof(line), fp)) {
        char tensor[3072];
        long long expert = -1;
        long long score = 0;
        const int n = std::sscanf(line, "%3071[^\t]\t%lld\t%lld", tensor, &expert, &score);
        if (n >= 2 && expert >= 0) {
            one_prefill_entry e;
            e.tensor = tensor;
            e.expert = expert;
            e.score = (n >= 3) ? score : 0;
            g_one_prefill.entries.push_back(std::move(e));
        }
    }
    std::fclose(fp);

    std::sort(g_one_prefill.entries.begin(), g_one_prefill.entries.end(),
        [](const one_prefill_entry & a, const one_prefill_entry & b) {
            if (a.score != b.score) {
                return a.score > b.score;
            }
            const int name_cmp = std::strcmp(a.tensor.c_str(), b.tensor.c_str());
            if (name_cmp != 0) {
                return name_cmp < 0;
            }
            return a.expert < b.expert;
        });

    if (g_one_prefill.limit == 0 || g_one_prefill.limit > g_one_prefill.entries.size()) {
        g_one_prefill.limit = g_one_prefill.entries.size();
    }
    g_one_prefill.enabled = !g_one_prefill.entries.empty();
    if (g_one_prefill.enabled && !g_one_prefill.registered) {
        g_one_prefill.registered = true;
        std::atexit(one_prefill_report_atexit);
    }
    std::fprintf(stderr, "[moe_stream] one prefill: loaded %zu entries from %s limit=%zu\n",
            g_one_prefill.entries.size(), path, g_one_prefill.limit);
}

static bool one_prefill_enabled() {
    std::lock_guard<std::mutex> lk(g_one_prefill.mu);
    one_prefill_init_once_locked();
    return g_one_prefill.enabled;
}

static uintptr_t one_named_cache_key(const char * tensor, int64_t expert) {
    uint64_t h = 1469598103934665603ULL;
    if (tensor) {
        for (const unsigned char * p = (const unsigned char *) tensor; *p; ++p) {
            h ^= (uint64_t) *p;
            h *= 1099511628211ULL;
        }
    }
    h ^= 0xff;
    h *= 1099511628211ULL;
    uint64_t x = (uint64_t) expert;
    for (int i = 0; i < 8; ++i) {
        h ^= (x >> (i * 8)) & 0xffU;
        h *= 1099511628211ULL;
    }
    if (h == 0) {
        h = 1;
    }
    return (uintptr_t) h;
}

static bool one_prefill_key_mode_enabled() {
    static int enabled = [] {
        const char * path = std::getenv("GGML_MOE_STREAM_ONE_PREFILL_PROFILE");
        return path && path[0] ? 1 : 0;
    }();
    return enabled != 0;
}

static uintptr_t one_cache_key_for(const char * tensor, int64_t expert, const void * src0_data) {
    const char * gate_up_prefetch = std::getenv("GGML_MOE_STREAM_ONE_GATE_UP_PREFETCH");
    const bool named_key = one_prefill_key_mode_enabled() ||
        (gate_up_prefetch && gate_up_prefetch[0] && gate_up_prefetch[0] != '0');
    return named_key ? one_named_cache_key(tensor, expert) : (uintptr_t) src0_data;
}

static const void * ds4_triplet_cache_dev_ptr_or_insert(
        const char * src0_name,
        size_t src0_bytes,
        int64_t expert_index,
        cudaStream_t stream,
        int * cache_state) {
    int role = -1;
    const ds4_triplet_pack_entry * entry = ds4_triplet_lookup(src0_name, expert_index, src0_bytes, role);
    if (!entry) return nullptr;

    if (!g_vcache_inited) {
        std::lock_guard<std::mutex> lock(g_init_mu);
        if (!g_vcache_inited) vram_cache_init((size_t) entry->nbytes);
    }
    if (!g_vcache.pool || g_vcache.n_slots <= 0 || entry->nbytes > g_vcache.slot_sz) return nullptr;

    const uintptr_t key = (uintptr_t) entry->generation;
    const size_t role_offset = role == 0 ? 0 :
        role == 1 ? (size_t) g_ds4_triplet_pack.role_bytes[0] :
        (size_t) (g_ds4_triplet_pack.role_bytes[0] + g_ds4_triplet_pack.role_bytes[1]);
    if (void * cached = vram_cache_lookup(key, stream)) {
        if (cache_state) *cache_state = 1;
        return (char *) cached + role_offset;
    }

    if (!moe_stream_cache_admit_allows(src0_name, expert_index)) {
        g_vcache.misses.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    if (vram_cache_freeze_reject_miss()) {
        if (cache_state) *cache_state = 5;
        return nullptr;
    }
    if (void * promoted = vram_cache_promote_from_ram(key, (size_t) entry->nbytes, stream, "ds4_triplet")) {
        if (cache_state) *cache_state = 2;
        return (char *) promoted + role_offset;
    }

    static ggml_moe_common::object_stage_ring import_ring;
    static std::mutex import_mu;

    std::lock_guard<std::mutex> import_lock(import_mu);
    if (import_ring.slots.empty()) {
        import_ring.slot_sz = (size_t) entry->nbytes;
        import_ring.slots.resize(8);
    }
    if (entry->nbytes > import_ring.slot_sz) return nullptr;
    ggml_moe_common::object_stage_slot & import =
        import_ring.slots[import_ring.next++ % import_ring.slots.size()];
    if (!ggml_moe_common::object_stage_slot_synchronize(import_ring, import)) return nullptr;
    if (!import.host &&
            !ggml_moe_common::object_stage_slot_allocate(import, import_ring.slot_sz, false)) {
        return nullptr;
    }
    ds4_triplet_host_handoff_invalidate(import.host);
    if (!ds4_triplet_read(*entry, import.host)) return nullptr;

    void * inserted = vram_cache_insert(key, import.host, (size_t) entry->nbytes, stream, "ds4_triplet");
    if (!inserted) return nullptr;
    if (!ggml_moe_common::object_stage_slot_record(import, stream)) {
        cudaGetLastError();
        cudaStreamSynchronize(stream);
    }
    ds4_triplet_host_handoff_publish(import.host, entry->generation);
    if (cache_state) *cache_state = 4;
    return (char *) inserted + role_offset;
}

static bool ds4_triplet_read_role_pair_raw(const ds4_triplet_pack_entry & entry, void * dst) {
    const size_t pair_bytes = (size_t) (g_ds4_triplet_pack.role_bytes[0] +
        g_ds4_triplet_pack.role_bytes[1]);
    const int fd = g_ds4_triplet_pack.fd_direct >= 0 ?
        g_ds4_triplet_pack.fd_direct : g_ds4_triplet_pack.fd;
    if (fd < 0 || !one_pack_read_exact_fd(fd, dst, pair_bytes, entry.offset)) {
        g_ds4_triplet_pack.failures.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    g_ds4_triplet_pack.reads.fetch_add(1, std::memory_order_relaxed);
    g_ds4_triplet_pack.role_pair_reads.fetch_add(1, std::memory_order_relaxed);
    g_ds4_triplet_pack.bytes.fetch_add(pair_bytes, std::memory_order_relaxed);
    return true;
}

static bool ds4_triplet_read_role_pair_batch(
        const ds4_triplet_pack_entry * const * entries,
        void * const * destinations,
        int count) {
    if (!entries || !destinations || count <= 0 || count > 8) return false;
    ggml_moe_common::object_request common_requests[8];
    const uint8_t role_mask = ggml_moe_common::role_bit(ggml_moe_common::role::gate) |
        ggml_moe_common::role_bit(ggml_moe_common::role::up);
    const bool common_started = ds4_triplet_common_begin(entries, count, role_mask, common_requests);
    const size_t pair_bytes = (size_t) (g_ds4_triplet_pack.role_bytes[0] +
        g_ds4_triplet_pack.role_bytes[1]);
    bool completed[8] = {};
    bool iouring_attempted = false;
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (g_ds4_triplet_pack.ring_inited && g_ds4_triplet_pack.fd_direct >= 0) {
        iouring_attempted = true;
        const auto submit_start = std::chrono::steady_clock::now();
        int prepared = 0;
        std::lock_guard<std::mutex> io_lock(g_ds4_triplet_pack.io_mu);
        for (; prepared < count; ++prepared) {
            if (!ggml_moe_common::object_io_ring_prepare_read(
                    g_ds4_triplet_pack.io_ring,
                    g_ds4_triplet_pack.fd_direct,
                    destinations[prepared],
                    pair_bytes,
                    entries[prepared]->offset,
                    (uint64_t) prepared)) {
                break;
            }
        }
        const int submitted = prepared > 0 ?
            ggml_moe_common::object_io_ring_submit_count(g_ds4_triplet_pack.io_ring) : 0;
        const auto submit_end = std::chrono::steady_clock::now();
        g_ds4_triplet_pack.io_stats.submit_us.fetch_add((uint64_t)
            std::chrono::duration_cast<std::chrono::microseconds>(submit_end - submit_start).count(),
            std::memory_order_relaxed);
        const auto wait_start = submit_end;
        int seen = 0;
        while (seen < submitted) {
            ggml_moe_common::object_io_completion completion;
            if (!ggml_moe_common::object_io_ring_wait(g_ds4_triplet_pack.io_ring, completion)) break;
            const bool index_valid = completion.user_data < (uint64_t) prepared;
            const size_t index = index_valid ? (size_t) completion.user_data : 0;
            if (ggml_moe_common::object_io_completion_validate(
                    g_ds4_triplet_pack.io_ring, completion, index_valid, pair_bytes) ==
                    ggml_moe_common::object_io_completion_status::complete) {
                completed[index] = true;
                g_ds4_triplet_pack.reads.fetch_add(1, std::memory_order_relaxed);
                g_ds4_triplet_pack.role_pair_reads.fetch_add(1, std::memory_order_relaxed);
                g_ds4_triplet_pack.bytes.fetch_add(pair_bytes, std::memory_order_relaxed);
                g_ds4_triplet_pack.io_stats.reads.fetch_add(1, std::memory_order_relaxed);
            }
            ++seen;
        }
        const auto wait_end = std::chrono::steady_clock::now();
        g_ds4_triplet_pack.io_stats.wait_us.fetch_add((uint64_t)
            std::chrono::duration_cast<std::chrono::microseconds>(wait_end - wait_start).count(),
            std::memory_order_relaxed);
        g_ds4_triplet_pack.io_stats.batches.fetch_add(1, std::memory_order_relaxed);
    }
#endif
    bool ok = true;
    for (int i = 0; i < count; ++i) {
        if (completed[i]) continue;
        if (iouring_attempted) {
            g_ds4_triplet_pack.io_stats.fallbacks.fetch_add(1, std::memory_order_relaxed);
        }
        if (!ds4_triplet_read_role_pair_raw(*entries[i], destinations[i])) ok = false;
    }
    ds4_triplet_common_complete(common_requests, count, common_started, ok);
    return ok;
}

static bool ds4_triplet_role_pair_cqe_h2d(
        const ds4_triplet_pack_entry * const * entries,
        void * const * host,
        int count,
        void ** inserted_gate,
        void ** inserted_up) {
#if !defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    (void) entries; (void) host; (void) count; (void) inserted_gate; (void) inserted_up;
    return false;
#else
    if (!g_ds4_triplet_pack.ring_inited || g_ds4_triplet_pack.fd_direct < 0 ||
            !g_ds4_triplet_pack.pipe_stream || count <= 0 || count > 8) return false;
    const size_t role_bytes = (size_t) g_ds4_triplet_pack.role_bytes[0];
    const size_t pair_bytes = role_bytes + (size_t) g_ds4_triplet_pack.role_bytes[1];
    ds4_vram_reservation gate_reservations[8];
    ds4_vram_reservation up_reservations[8];
    for (int i = 0; i < count; ++i) {
        if (!vram_cache_reserve_triplet(ds4_triplet_role_key(*entries[i], 0), role_bytes,
                g_ds4_triplet_pack.pipe_stream, gate_reservations[i], "ds4_pair_gate") ||
                !vram_cache_reserve_triplet(ds4_triplet_role_key(*entries[i], 1), role_bytes,
                g_ds4_triplet_pack.pipe_stream, up_reservations[i], "ds4_pair_up")) {
            for (int j = 0; j <= i; ++j) {
                if (gate_reservations[j].slot >= 0) vram_cache_fail_triplet(gate_reservations[j]);
                if (up_reservations[j].slot >= 0) vram_cache_fail_triplet(up_reservations[j]);
            }
            return false;
        }
    }
    ggml_moe_common::object_request common_requests[8];
    const uint8_t role_mask = ggml_moe_common::role_bit(ggml_moe_common::role::gate) |
        ggml_moe_common::role_bit(ggml_moe_common::role::up);
    const bool common_started = ds4_triplet_common_begin(entries, count, role_mask, common_requests);

    bool read_ok[8] = {};
    {
        std::lock_guard<std::mutex> io_lock(g_ds4_triplet_pack.io_mu);
        int prepared = 0;
        for (; prepared < count; ++prepared) {
            if (!ggml_moe_common::object_io_ring_prepare_read(
                    g_ds4_triplet_pack.io_ring,
                    g_ds4_triplet_pack.fd_direct,
                    host[prepared],
                    pair_bytes,
                    entries[prepared]->offset,
                    (uint64_t) prepared)) {
                break;
            }
        }
        const int submitted = prepared > 0 ?
            ggml_moe_common::object_io_ring_submit_count(g_ds4_triplet_pack.io_ring) : 0;
        int seen = 0;
        while (seen < submitted) {
            ggml_moe_common::object_io_completion completion;
            if (!ggml_moe_common::object_io_ring_wait(g_ds4_triplet_pack.io_ring, completion)) break;
            const bool index_valid = completion.user_data < (uint64_t) prepared;
            const size_t index = index_valid ? (size_t) completion.user_data : 0;
            if (ggml_moe_common::object_io_completion_validate(
                    g_ds4_triplet_pack.io_ring, completion, index_valid, pair_bytes) ==
                    ggml_moe_common::object_io_completion_status::complete) {
                read_ok[index] = true;
                g_ds4_triplet_pack.reads.fetch_add(1, std::memory_order_relaxed);
                g_ds4_triplet_pack.role_pair_reads.fetch_add(1, std::memory_order_relaxed);
                g_ds4_triplet_pack.bytes.fetch_add(pair_bytes, std::memory_order_relaxed);
                g_ds4_triplet_pack.io_stats.reads.fetch_add(1, std::memory_order_relaxed);
                g_ds4_triplet_pack.pipe_cqes.fetch_add(1, std::memory_order_relaxed);
            }
            ++seen;
        }
    }

    bool ok = true;
    for (int i = 0; i < count; ++i) {
        if (!read_ok[i]) {
            g_ds4_triplet_pack.io_stats.fallbacks.fetch_add(1, std::memory_order_relaxed);
            if (!ds4_triplet_read_role_pair_raw(*entries[i], host[i])) {
                ok = false;
                continue;
            }
        }
        const auto h2d_start = std::chrono::steady_clock::now();
        const bool copied =
            cudaMemcpyAsync(gate_reservations[i].dst, host[i], role_bytes,
                cudaMemcpyHostToDevice, g_ds4_triplet_pack.pipe_stream) == cudaSuccess &&
            cudaMemcpyAsync(up_reservations[i].dst, (char *) host[i] + role_bytes, role_bytes,
                cudaMemcpyHostToDevice, g_ds4_triplet_pack.pipe_stream) == cudaSuccess;
        const bool published = copied &&
            vram_cache_publish_triplet(gate_reservations[i], g_ds4_triplet_pack.pipe_stream) &&
            vram_cache_publish_triplet(up_reservations[i], g_ds4_triplet_pack.pipe_stream);
        const auto h2d_end = std::chrono::steady_clock::now();
        g_ds4_triplet_pack.pipe_h2d_enqueue_us.fetch_add((uint64_t)
            std::chrono::duration_cast<std::chrono::microseconds>(h2d_end - h2d_start).count(),
            std::memory_order_relaxed);
        if (!published) {
            cudaGetLastError();
            vram_cache_fail_triplet(gate_reservations[i]);
            vram_cache_fail_triplet(up_reservations[i]);
            ok = false;
            continue;
        }
        inserted_gate[i] = gate_reservations[i].dst;
        inserted_up[i] = up_reservations[i].dst;
        g_ds4_triplet_pack.pipe_h2d_bytes.fetch_add(pair_bytes, std::memory_order_relaxed);
    }
    g_ds4_triplet_pack.pipe_batches.fetch_add(1, std::memory_order_relaxed);
    ds4_triplet_common_complete(common_requests, count, common_started, ok);
    return ok;
#endif
}

static bool ds4_triplet_role_pair_dev_ptrs_or_insert(
        const char * src0_name,
        size_t src0_bytes,
        const int * expert_indices,
        int count,
        cudaStream_t stream,
        const void ** output_ptrs,
        int * cache_states,
        ggml_moe_expert_descriptor * descriptors) {
    const ds4_triplet_pack_entry * requested[8] = {};
    int roles[8] = {};
    for (int i = 0; i < count; ++i) {
        output_ptrs[i] = nullptr;
        cache_states[i] = 0;
        descriptors[i] = {};
        descriptors[i].slot_id = -1;
        requested[i] = ds4_triplet_lookup(src0_name, expert_indices[i], src0_bytes, roles[i]);
        if (!requested[i] || roles[i] < 0 || roles[i] > 1) return false;
    }

    if (g_ds4_triplet_pack.role_bytes[0] != g_ds4_triplet_pack.role_bytes[1]) return false;
    const size_t role_bytes = (size_t) g_ds4_triplet_pack.role_bytes[0];
    const size_t pair_bytes = role_bytes * 2;
    if (!g_vcache_inited) {
        std::lock_guard<std::mutex> lock(g_init_mu);
        if (!g_vcache_inited) vram_cache_init(role_bytes);
    }
    if (!g_vcache.pool || g_vcache.n_slots <= 0 || g_vcache.slot_sz < role_bytes) return false;

    int jobs_for_request[8];
    const ds4_triplet_pack_entry * jobs[8] = {};
    int job_count = 0;
    for (int i = 0; i < count; ++i) {
        jobs_for_request[i] = -1;
        const uintptr_t key = ds4_triplet_role_key(*requested[i], roles[i]);
        if (void * cached = vram_cache_lookup(key, stream)) {
            output_ptrs[i] = cached;
            cache_states[i] = 1;
            continue;
        }
        for (int job = 0; job < job_count; ++job) {
            if (jobs[job]->generation == requested[i]->generation) {
                jobs_for_request[i] = job;
                break;
            }
        }
        if (jobs_for_request[i] < 0) {
            jobs_for_request[i] = job_count;
            jobs[job_count++] = requested[i];
        }
    }

    static ggml_moe_common::object_stage_ring import_ring;
    static std::mutex import_mu;
    void * inserted_gate[8] = {};
    void * inserted_up[8] = {};
    if (job_count > 0) {
        std::lock_guard<std::mutex> import_lock(import_mu);
        if (import_ring.slots.empty()) {
            import_ring.slot_sz = pair_bytes;
            import_ring.slots.resize(8);
        }
        if (pair_bytes > import_ring.slot_sz) return false;
        void * destinations[8] = {};
        for (int job = 0; job < job_count; ++job) {
            ggml_moe_common::object_stage_slot & import = import_ring.slots[job];
            if (!ggml_moe_common::object_stage_slot_synchronize(import_ring, import)) return false;
            if (!import.host &&
                    !ggml_moe_common::object_stage_slot_allocate(import, import_ring.slot_sz, false)) {
                return false;
            }
            destinations[job] = import.host;
        }
        const bool cqe_h2d = g_ds4_triplet_pack.pipe_requested &&
            ds4_triplet_role_pair_cqe_h2d(
                jobs, destinations, job_count, inserted_gate, inserted_up);
        if (!cqe_h2d && !ds4_triplet_read_role_pair_batch(jobs, destinations, job_count)) return false;
        cudaStream_t transfer_stream = cqe_h2d ? g_ds4_triplet_pack.pipe_stream : stream;
        for (int job = 0; job < job_count; ++job) {
            ggml_moe_common::object_stage_slot & import = import_ring.slots[job];
            if (!cqe_h2d) {
                inserted_gate[job] = vram_cache_insert(
                    ds4_triplet_role_key(*jobs[job], 0), import.host, role_bytes, stream, "ds4_pair_gate");
                inserted_up[job] = vram_cache_insert(
                    ds4_triplet_role_key(*jobs[job], 1), (char *) import.host + role_bytes,
                    role_bytes, stream, "ds4_pair_up");
                if (!inserted_gate[job] || !inserted_up[job]) return false;
            }
            if (!ggml_moe_common::object_stage_slot_record(import, transfer_stream)) {
                cudaGetLastError();
                cudaStreamSynchronize(transfer_stream);
            }
        }
    }

    int acquired = 0;
    for (int i = 0; i < count; ++i) {
        const int job = jobs_for_request[i];
        if (job >= 0) {
            output_ptrs[i] = roles[i] == 0 ? inserted_gate[job] : inserted_up[job];
            cache_states[i] = 4;
        }
        const uintptr_t key = ds4_triplet_role_key(*requested[i], roles[i]);
        if (!output_ptrs[i] || !vram_cache_acquire_descriptor(
                output_ptrs[i], key, roles[i], 0, stream, descriptors[i])) {
            for (int j = 0; j < i; ++j) vram_cache_cancel_descriptor(descriptors[j]);
            return false;
        }
        ++acquired;
    }
    return acquired > 0;
}

static bool ds4_triplet_dependency_h2d_dev_ptrs_or_insert(
        const char * src0_name,
        size_t src0_bytes,
        const int * expert_indices,
        int count,
        cudaStream_t compute_stream,
        const void ** output_ptrs,
        int * cache_states,
        ggml_moe_expert_descriptor * descriptors) {
    if (!ds4_dependency_h2d_eligible() || !src0_name || !expert_indices ||
            count <= 0 || count > 8 || !compute_stream || !output_ptrs ||
            !cache_states || !descriptors) {
        return false;
    }

    const ds4_triplet_pack_entry * requested[8] = {};
    int roles[8] = {};
    int jobs_for_request[8] = {};
    const ds4_triplet_pack_entry * jobs[8] = {};
    int job_count = 0;
    for (int request = 0; request < count; ++request) {
        output_ptrs[request] = nullptr;
        cache_states[request] = 0;
        descriptors[request] = {};
        descriptors[request].slot_id = -1;
        jobs_for_request[request] = -1;
        requested[request] = ds4_triplet_lookup(
            src0_name, expert_indices[request], src0_bytes, roles[request]);
        if (!requested[request] || roles[request] < 0 || roles[request] >= 3) {
            return false;
        }
    }

    const size_t role_slot_bytes = ds4_dependency_role_slot_bytes();
    if (!g_vcache_inited) {
        std::lock_guard<std::mutex> lock(g_init_mu);
        if (!g_vcache_inited) {
            vram_cache_init(role_slot_bytes, (size_t) g_ds4_triplet_pack.object_bytes);
        }
    }
    if (!g_vcache.pool || g_vcache.n_slots <= 0 ||
            g_vcache.slot_sz != role_slot_bytes) {
        static std::atomic<bool> reported{false};
        if (!reported.exchange(true, std::memory_order_relaxed)) {
            std::fprintf(stderr,
                "[moe_stream] dependency H2D: incompatible VRAM cache geometry "
                "slot_bytes=%zu required=%zu; old path fallback\n",
                g_vcache.slot_sz, role_slot_bytes);
        }
        return false;
    }

    for (int request = 0; request < count; ++request) {
        {
            std::lock_guard<std::mutex> lock(g_vcache.mu);
            ds4_ram_note_demand_locked((uintptr_t) requested[request]->generation);
        }
        g_ds4_triplet_pack.role_calls[roles[request]].fetch_add(1, std::memory_order_relaxed);
        g_ds4_triplet_pack.role_lookups[roles[request]].fetch_add(1, std::memory_order_relaxed);
        const uintptr_t role_key = ds4_triplet_role_key(*requested[request], roles[request]);
        if (void * cached = vram_cache_lookup(role_key, compute_stream)) {
            output_ptrs[request] = cached;
            cache_states[request] = 1;
            g_ds4_triplet_pack.role_hits[roles[request]].fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        g_ds4_triplet_pack.role_misses[roles[request]].fetch_add(1, std::memory_order_relaxed);
        if (!moe_stream_cache_admit_allows(src0_name, expert_indices[request]) ||
                vram_cache_freeze_reject_miss()) {
            return false;
        }
        for (int job = 0; job < job_count; ++job) {
            if (jobs[job]->generation == requested[request]->generation) {
                jobs_for_request[request] = job;
                break;
            }
        }
        if (jobs_for_request[request] < 0) {
            jobs_for_request[request] = job_count;
            jobs[job_count++] = requested[request];
        }
    }

    auto acquire_descriptors = [&]() {
        int acquired = 0;
        for (int request = 0; request < count; ++request) {
            if (!output_ptrs[request] ||
                    !vram_cache_acquire_descriptor(
                        output_ptrs[request],
                        ds4_triplet_role_key(*requested[request], roles[request]),
                        roles[request],
                        0,
                        compute_stream,
                        descriptors[request])) {
                for (int previous = 0; previous < request; ++previous) {
                    vram_cache_cancel_descriptor(descriptors[previous]);
                }
                return false;
            }
            ++acquired;
        }
        return acquired == count;
    };
    if (job_count == 0) return acquire_descriptors();

    const bool pipe_available = g_ds4_triplet_pack.pipe_requested &&
        !g_ds4_triplet_pack.pipe_failed && g_ds4_triplet_pack.pipe_started &&
        g_ds4_triplet_pack.pipe_stream && g_ds4_triplet_pack.pipe_deferred_stream;
    const bool rendezvous_enabled = pipe_available &&
        ggml_moe_common::object_engine_enforcing();
    ds4_triplet_pipe_batch stack_batch;
    ds4_triplet_pipe_batch * batch_ptr = &stack_batch;
    ds4_pipeline_caller_guard pipeline_guard;
    if (rendezvous_enabled && ds4_common_async_pipeline_requested()) {
        if (!g_ds4_triplet_pack.dependency_pipeline_pool.configure(2) ||
                !g_ds4_triplet_pack.dependency_pipeline_pool.acquire(
                    pipeline_guard.handle)) {
            g_ds4_triplet_pack.dependency_pipeline_backpressure.fetch_add(
                1, std::memory_order_relaxed);
        } else {
            pipeline_guard.active = true;
            batch_ptr = &g_ds4_triplet_pack.dependency_pipeline_batches[
                pipeline_guard.handle.slot];
            g_ds4_triplet_pack.dependency_pipeline_acquires.fetch_add(
                1, std::memory_order_relaxed);
        }
    }
    ds4_triplet_pipe_batch & batch = *batch_ptr;
    ds4_triplet_pipe_batch_reset(batch);
    if (pipeline_guard.active) batch.pipeline_handle = pipeline_guard.handle;
    else batch.pipeline_handle = {};
    batch.dependency_h2d = true;
    int batch_index_for_job[8];
    ggml_moe_common::object_transaction_ticket joined_tickets[8] = {};
    uint8_t joined_role_masks[8] = {};
    uint8_t job_role_masks[8] = {};
    std::fill(batch_index_for_job, batch_index_for_job + 8, -1);
    for (int request = 0; request < count; ++request) {
        const int job = jobs_for_request[request];
        if (job < 0) continue;
        job_role_masks[job] |= ggml_moe_common::role_bit(
            static_cast<ggml_moe_common::role>(roles[request]));
    }
    ggml_moe_expert_descriptor resident_leases[8][3] = {};
    for (int job = 0; job < job_count; ++job) {
        const uint8_t full_role_mask =
            ggml_moe_common::role_bit(ggml_moe_common::role::gate) |
            ggml_moe_common::role_bit(ggml_moe_common::role::up) |
            ggml_moe_common::role_bit(ggml_moe_common::role::down);
        const ggml_moe_common::object_request object =
            ds4_triplet_common_request(*jobs[job], full_role_mask);
        ggml_moe_common::object_transaction_ticket ticket;
        if (rendezvous_enabled) {
            const ggml_moe_common::object_transaction_acquire_status acquire_status =
                ggml_moe_common::object_engine_acquire_transaction(object, ticket);
            if (acquire_status == ggml_moe_common::object_transaction_acquire_status::joined) {
                joined_tickets[job] = ticket;
                joined_role_masks[job] = job_role_masks[job];
                g_ds4_triplet_pack.dependency_transaction_joins.fetch_add(
                    1, std::memory_order_relaxed);
                continue;
            }
            if (acquire_status != ggml_moe_common::object_transaction_acquire_status::owner) {
                ds4_dependency_complete_owned_transactions(batch, false);
                g_ds4_triplet_pack.dependency_fallbacks.fetch_add(
                    1, std::memory_order_relaxed);
                return false;
            }
        }

        const int object_index = batch.count++;
        batch_index_for_job[job] = object_index;
        batch.entries[object_index] = jobs[job];
        batch.caller_role_masks[object_index] = job_role_masks[job];
        if (rendezvous_enabled) {
            batch.common_requests[object_index] = object;
            batch.transaction_tickets[object_index] = ticket;
            batch.common_transactions_owned = true;
            g_ds4_triplet_pack.dependency_transaction_owners.fetch_add(
                1, std::memory_order_relaxed);
        }
        uint8_t resident_mask = 0;
        for (int role = 0; role < 3; ++role) {
            resident_leases[object_index][role].slot_id = -1;
            const uintptr_t role_key = ds4_triplet_role_key(*jobs[job], role);
            void * resident = vram_cache_lookup(role_key, compute_stream);
            if (!resident) continue;
            if (!vram_cache_acquire_descriptor(
                    resident, role_key, role, 0, compute_stream,
                    resident_leases[object_index][role])) {
                for (int previous_object = 0; previous_object <= object_index; ++previous_object) {
                    for (int previous_role = 0; previous_role < 3; ++previous_role) {
                        vram_cache_cancel_descriptor(
                            resident_leases[previous_object][previous_role]);
                    }
                }
                ds4_dependency_complete_owned_transactions(batch, false);
                return false;
            }
            batch.role_ptrs[object_index][role] = resident;
            resident_mask |= ggml_moe_common::role_bit(static_cast<ggml_moe_common::role>(role));
        }
        const uint8_t critical_role_mask =
            ggml_moe_common::role_bit(ggml_moe_common::role::gate) |
            ggml_moe_common::role_bit(ggml_moe_common::role::up);
        if (ggml_moe_common::build_dependency_transfer_plan(
                object,
                resident_mask,
                critical_role_mask,
                batch.transfer_plans[object_index]) !=
                ggml_moe_common::dependency_transfer_status::planned) {
            for (int previous_object = 0; previous_object <= object_index; ++previous_object) {
                for (int previous_role = 0; previous_role < 3; ++previous_role) {
                    vram_cache_cancel_descriptor(
                        resident_leases[previous_object][previous_role]);
                }
            }
            ds4_dependency_complete_owned_transactions(batch, false);
            g_ds4_triplet_pack.dependency_fallbacks.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        g_ds4_triplet_pack.dependency_plans.fetch_add(1, std::memory_order_relaxed);
        if (batch.transfer_plans[object_index].all_required_roles_missing) {
            g_ds4_triplet_pack.dependency_full_misses.fetch_add(1, std::memory_order_relaxed);
        } else {
            g_ds4_triplet_pack.dependency_partial_misses.fetch_add(1, std::memory_order_relaxed);
        }
    }

    cudaStream_t reservation_stream = pipe_available ?
        g_ds4_triplet_pack.pipe_stream : compute_stream;
    const bool reserved = batch.count == 0 ||
        ds4_dependency_reserve_missing(batch, reservation_stream);
    for (int object_index = 0; object_index < batch.count; ++object_index) {
        for (int role = 0; role < 3; ++role) {
            vram_cache_cancel_descriptor(resident_leases[object_index][role]);
        }
    }
    if (!reserved) {
        ds4_dependency_complete_owned_transactions(batch, false);
        g_ds4_triplet_pack.dependency_fallbacks.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    static ggml_moe_common::object_stage_ring import_ring;
    static std::mutex import_mu;
    std::unique_lock<std::mutex> import_lock(import_mu, std::defer_lock);
    ggml_moe_common::object_stage_ring * stage_ring = &import_ring;
    if (pipeline_guard.active) {
        stage_ring = &batch.pipeline_stage_ring;
    } else if (batch.count > 0) {
        import_lock.lock();
    }
    batch.stage_ring = stage_ring;
    if (stage_ring->slots.empty()) {
        stage_ring->slot_sz = std::max<size_t>(
            (size_t) g_ds4_triplet_pack.object_bytes,
            g_ds4_triplet_pack.lossless_requested &&
                    !g_ds4_triplet_pack.lossless_failed ?
                g_ds4_triplet_pack.lossless_sidecar.max_physical_bytes : 0);
        stage_ring->slots.resize(8);
    }
    for (int object_index = 0; object_index < batch.count; ++object_index) {
        if (ds4_dependency_acquire_host_object(
                *batch.entries[object_index], batch, object_index)) {
            continue;
        }
        ggml_moe_common::object_stage_slot & import = stage_ring->slots[object_index];
        if (batch.entries[object_index]->nbytes > stage_ring->slot_sz ||
                !ggml_moe_common::object_stage_slot_synchronize(*stage_ring, import) ||
                (!import.host && !ggml_moe_common::object_stage_slot_allocate(
                    import, stage_ring->slot_sz, false))) {
            ds4_dependency_cancel_host_objects(batch, reservation_stream);
            ds4_dependency_fail_reservations(batch);
            ds4_dependency_complete_owned_transactions(batch, false);
            return false;
        }
        ds4_triplet_host_handoff_invalidate(import.host);
        batch.host[object_index] = import.host;
        batch.stage_slot_indices[object_index] = (size_t) object_index;
    }

    bool transferred = batch.count == 0;
    if (!transferred && pipe_available) {
        transferred = ds4_triplet_pipe_submit(batch);
        if (pipeline_guard.active && batch.pipeline_submitted) {
            pipeline_guard.worker_owned = true;
        }
    }
    cudaStream_t completion_stream = g_ds4_triplet_pack.pipe_deferred_stream;
    if (!transferred) {
        ds4_dependency_complete_owned_transactions(batch, false);
        g_ds4_triplet_pack.dependency_fallbacks.fetch_add(1, std::memory_order_relaxed);
        if (pipe_available) {
            cudaStreamSynchronize(g_ds4_triplet_pack.pipe_stream);
            cudaStreamSynchronize(g_ds4_triplet_pack.pipe_deferred_stream);
        }
        ds4_dependency_fail_reservations(batch);
        const ds4_triplet_pack_entry * read_entries[8] = {};
        void * read_destinations[8] = {};
        int read_count = 0;
        for (int object_index = 0; object_index < batch.count; ++object_index) {
            if (batch.host_cache_hits[object_index]) continue;
            read_entries[read_count] = batch.entries[object_index];
            read_destinations[read_count] = batch.host[object_index];
            ++read_count;
        }
        if (!ds4_dependency_reserve_missing(batch, compute_stream) ||
                (read_count > 0 && !ds4_triplet_read_batch(
                    read_entries, read_destinations, read_count))) {
            ds4_dependency_cancel_host_objects(batch, compute_stream);
            ds4_dependency_fail_reservations(batch);
            return false;
        }
        transferred = true;
        for (int object_index = 0; object_index < batch.count; ++object_index) {
            uint64_t critical_bytes = 0;
            uint64_t deferred_bytes = 0;
            const ggml_moe_common::dependency_transfer_plan & plan =
                batch.transfer_plans[object_index];
            if (!vram_cache_copy_dependency_group_from_host(
                    plan.critical, plan.critical_count,
                    batch.role_reservations[object_index], batch.host[object_index],
                    compute_stream, critical_bytes) ||
                    !vram_cache_copy_dependency_group_from_host(
                        plan.deferred, plan.deferred_count,
                        batch.role_reservations[object_index], batch.host[object_index],
                        compute_stream, deferred_bytes)) {
                cudaGetLastError();
                transferred = false;
                break;
            }
            g_ds4_triplet_pack.dependency_critical_copies.fetch_add(
                plan.critical_count, std::memory_order_relaxed);
            g_ds4_triplet_pack.dependency_critical_bytes.fetch_add(
                critical_bytes, std::memory_order_relaxed);
            g_ds4_triplet_pack.dependency_deferred_copies.fetch_add(
                plan.deferred_count, std::memory_order_relaxed);
            g_ds4_triplet_pack.dependency_deferred_bytes.fetch_add(
                deferred_bytes, std::memory_order_relaxed);
            g_ds4_triplet_pack.pipe_h2d_bytes.fetch_add(
                critical_bytes + deferred_bytes, std::memory_order_relaxed);
        }
        completion_stream = compute_stream;
    }
    if (!transferred) {
        cudaStreamSynchronize(completion_stream);
        ds4_dependency_cancel_host_objects(batch, completion_stream);
        ds4_dependency_fail_reservations(batch);
        ds4_dependency_complete_owned_transactions(batch, false);
        return false;
    }

    if (!batch.pipeline_critical_returned) {
        ds4_dependency_finalize_host_sources(batch, completion_stream);
    }
    for (int job = 0; job < job_count; ++job) {
        if (joined_tickets[job].transaction == 0) continue;
        const auto join_wait_start = std::chrono::steady_clock::now();
        const ggml_moe_common::object_transaction_wait_status wait_status =
            ggml_moe_common::object_engine_wait_transaction(
                joined_tickets[job], joined_role_masks[job], UINT64_C(30000000));
        g_ds4_triplet_pack.dependency_transaction_join_wait_us.fetch_add(
            (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - join_wait_start).count(),
            std::memory_order_relaxed);
        if (wait_status != ggml_moe_common::object_transaction_wait_status::ready &&
                wait_status != ggml_moe_common::object_transaction_wait_status::completed) {
            g_ds4_triplet_pack.dependency_transaction_failures.fetch_add(
                1, std::memory_order_relaxed);
            g_ds4_triplet_pack.dependency_fallbacks.fetch_add(
                1, std::memory_order_relaxed);
            return false;
        }
        for (int request = 0; request < count; ++request) {
            if (jobs_for_request[request] != job) continue;
            output_ptrs[request] = vram_cache_lookup(
                ds4_triplet_role_key(*requested[request], roles[request]),
                compute_stream);
            cache_states[request] = output_ptrs[request] ? 5 : 0;
            if (!output_ptrs[request]) {
                g_ds4_triplet_pack.dependency_transaction_failures.fetch_add(
                    1, std::memory_order_relaxed);
                return false;
            }
        }
    }
    for (int request = 0; request < count; ++request) {
        const int job = jobs_for_request[request];
        if (job < 0) continue;
        const int object_index = batch_index_for_job[job];
        if (object_index < 0) continue;
        output_ptrs[request] = batch.role_ptrs[object_index][roles[request]];
        cache_states[request] = batch.host_cache_hits[object_index] ? 2 : 4;
        if (!batch.host_cache_hits[object_index]) {
            g_ds4_triplet_pack.role_physical_reads[roles[request]].fetch_add(
                1, std::memory_order_relaxed);
        }
    }
    return acquire_descriptors();
}

static bool one_pack_cqe_h2d_enabled() {
    const char * env = std::getenv("GGML_MOE_STREAM_ONE_PACK_CQE_H2D");
    return env && env[0] && env[0] != '0';
}

static bool one_pack_cqe_h2d_dev_ptrs_or_insert(
        const char * src0_name,
        const void * src0_base_data,
        size_t nb02,
        size_t src0_bytes,
        const int * expert_indices,
        int count,
        cudaStream_t compute_stream,
        const void ** output_ptrs,
        int * cache_states,
        ggml_moe_expert_descriptor * descriptors) {
#if !defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    (void) src0_name; (void) src0_base_data; (void) nb02; (void) src0_bytes;
    (void) expert_indices; (void) count; (void) compute_stream; (void) output_ptrs;
    (void) cache_states; (void) descriptors;
    return false;
#else
    one_pack_init_once();
    if (!g_one_pack.enabled || g_one_pack.fd_direct < 0) return false;
    if (!g_vcache_inited) {
        std::lock_guard<std::mutex> lock(g_init_mu);
        if (!g_vcache_inited) vram_cache_init(src0_bytes);
    }
    if (!g_vcache.pool || g_vcache.n_slots <= 0 || src0_bytes > g_vcache.slot_sz) return false;

    struct plan {
        int expert = -1;
        uintptr_t key = 0;
        const one_expert_pack_entry * entry = nullptr;
        ds4_vram_reservation reservation;
        bool read_ok = false;
        bool published = false;
    };
    plan plans[8];
    int request_plan[8];
    int plan_count = 0;
    int acquired = 0;
    for (int i = 0; i < count; ++i) {
        output_ptrs[i] = nullptr;
        cache_states[i] = 0;
        descriptors[i] = {};
        descriptors[i].slot_id = -1;
        request_plan[i] = -1;
        const void * expert_data = (const char *) src0_base_data + (size_t) expert_indices[i] * nb02;
        const uintptr_t key = one_cache_key_for(src0_name, expert_indices[i], expert_data);
        if (void * cached = vram_cache_lookup(key, compute_stream)) {
            output_ptrs[i] = cached;
            cache_states[i] = 1;
        } else if (void * promoted = vram_cache_promote_from_ram(key, src0_bytes, compute_stream, src0_name)) {
            output_ptrs[i] = promoted;
            cache_states[i] = 2;
        } else {
            if (!moe_stream_cache_admit_allows(src0_name, expert_indices[i]) ||
                    vram_cache_freeze_reject_miss()) {
                for (int j = 0; j < i; ++j) vram_cache_cancel_descriptor(descriptors[j]);
                return false;
            }
            for (int p = 0; p < plan_count; ++p) {
                if (plans[p].key == key) {
                    request_plan[i] = p;
                    break;
                }
            }
            if (request_plan[i] < 0) {
                const one_expert_pack_entry * entry =
                    one_pack_lookup(src0_name, expert_indices[i], src0_bytes);
                if (!entry) {
                    for (int j = 0; j < i; ++j) vram_cache_cancel_descriptor(descriptors[j]);
                    return false;
                }
                request_plan[i] = plan_count;
                plans[plan_count].expert = expert_indices[i];
                plans[plan_count].key = key;
                plans[plan_count].entry = entry;
                ++plan_count;
            }
        }
        if (output_ptrs[i]) {
            if (!vram_cache_acquire_descriptor(
                    output_ptrs[i], key, 0, 0, compute_stream, descriptors[i])) {
                for (int j = 0; j < i; ++j) vram_cache_cancel_descriptor(descriptors[j]);
                return false;
            }
            ++acquired;
        }
    }
    if (plan_count == 0) return acquired == count;

    std::lock_guard<std::mutex> state_lock(g_one_pack_cqe.mu);
    if (!g_one_pack_cqe.init_attempted) {
        g_one_pack_cqe.init_attempted = true;
        if (cudaStreamCreateWithFlags(&g_one_pack_cqe.stream, cudaStreamNonBlocking) != cudaSuccess) {
            cudaGetLastError();
        }
    }
    if (!g_one_pack_cqe.ring_inited) {
        int depth = 8;
        if (const char * env = std::getenv("GGML_MOE_STREAM_ONE_PACK_CQE_DEPTH")) {
            depth = std::max(8, std::min(32, std::atoi(env)));
        }
        if (ggml_moe_common::object_io_ring_ensure(
                g_one_pack_cqe.io_ring, (size_t) depth, 0, nullptr)) {
            g_one_pack_cqe.ring_inited = true;
        }
    }
    g_one_pack_cqe.ready = g_one_pack_cqe.ring_inited && g_one_pack_cqe.stream;
    if (!g_one_pack_cqe.ready) {
        for (int i = 0; i < count; ++i) vram_cache_cancel_descriptor(descriptors[i]);
        return false;
    }
    if (g_one_pack_cqe.import_ring.slots.empty()) {
        g_one_pack_cqe.import_ring.slot_sz = src0_bytes;
        g_one_pack_cqe.import_ring.slots.resize(8);
    }
    if (src0_bytes > g_one_pack_cqe.import_ring.slot_sz) {
        for (int i = 0; i < count; ++i) vram_cache_cancel_descriptor(descriptors[i]);
        return false;
    }

    for (int p = 0; p < plan_count; ++p) {
        ggml_moe_common::object_stage_slot & import = g_one_pack_cqe.import_ring.slots[p];
        if (!ggml_moe_common::object_stage_slot_synchronize(g_one_pack_cqe.import_ring, import) ||
                (!import.host &&
                 !ggml_moe_common::object_stage_slot_allocate(import, g_one_pack_cqe.import_ring.slot_sz, false)) ||
                !vram_cache_reserve_triplet(plans[p].key, src0_bytes, g_one_pack_cqe.stream,
                    plans[p].reservation, src0_name)) {
            cudaGetLastError();
            for (int q = 0; q <= p; ++q) {
                if (plans[q].reservation.slot >= 0) vram_cache_fail_triplet(plans[q].reservation);
            }
            for (int i = 0; i < count; ++i) vram_cache_cancel_descriptor(descriptors[i]);
            return false;
        }
        if (!ggml_moe_common::object_io_ring_prepare_read(
                g_one_pack_cqe.io_ring,
                g_one_pack.fd_direct,
                import.host,
                src0_bytes,
                plans[p].entry->offset,
                (uint64_t) p)) {
            ggml_moe_common::object_io_ring_release(g_one_pack_cqe.io_ring);
            g_one_pack_cqe.ring_inited = false;
            g_one_pack_cqe.ready = false;
            for (int q = 0; q <= p; ++q) vram_cache_fail_triplet(plans[q].reservation);
            for (int i = 0; i < count; ++i) vram_cache_cancel_descriptor(descriptors[i]);
            return false;
        }
    }

    const auto submit_start = std::chrono::steady_clock::now();
    int submitted = 0;
    while (submitted < plan_count) {
        const int rc = ggml_moe_common::object_io_ring_submit_count(g_one_pack_cqe.io_ring);
        if (rc <= 0) break;
        submitted += rc;
    }
    const auto submit_end = std::chrono::steady_clock::now();
    g_one_pack_cqe.submit_us += (uint64_t)
        std::chrono::duration_cast<std::chrono::microseconds>(submit_end - submit_start).count();

    bool ok = true;
    auto publish_plan = [&](int p) {
        ggml_moe_common::object_stage_slot & import = g_one_pack_cqe.import_ring.slots[p];
        const auto h2d_start = std::chrono::steady_clock::now();
        const bool copied = plans[p].read_ok && cudaMemcpyAsync(
            plans[p].reservation.dst, import.host, src0_bytes,
            cudaMemcpyHostToDevice, g_one_pack_cqe.stream) == cudaSuccess;
        plans[p].published = copied &&
            vram_cache_publish_triplet(plans[p].reservation, g_one_pack_cqe.stream);
        g_one_pack_cqe.h2d_enqueue_us += (uint64_t)
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - h2d_start).count();
        if (!plans[p].published) {
            cudaGetLastError();
            if (copied) cudaStreamSynchronize(g_one_pack_cqe.stream);
            vram_cache_fail_triplet(plans[p].reservation);
            ok = false;
            return;
        }
        if (!ggml_moe_common::object_stage_slot_record(import, g_one_pack_cqe.stream)) {
            cudaGetLastError();
            cudaStreamSynchronize(g_one_pack_cqe.stream);
        }
    };

    int reaped = 0;
    for (; reaped < submitted; ++reaped) {
        ggml_moe_common::object_io_completion completion;
        const auto wait_start = std::chrono::steady_clock::now();
        if (!ggml_moe_common::object_io_ring_wait(g_one_pack_cqe.io_ring, completion)) break;
        g_one_pack_cqe.wait_us += (uint64_t)
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - wait_start).count();
        const bool plan_valid = completion.user_data < (uint64_t) plan_count;
        const int p = plan_valid ? (int) completion.user_data : 0;
        if (ggml_moe_common::object_io_completion_validate(
                g_one_pack_cqe.io_ring, completion, plan_valid, src0_bytes) ==
                ggml_moe_common::object_io_completion_status::complete) {
            plans[p].read_ok = true;
            g_one_pack.reads.fetch_add(1, std::memory_order_relaxed);
            g_one_pack.direct_reads.fetch_add(1, std::memory_order_relaxed);
            g_one_pack.bytes.fetch_add(src0_bytes, std::memory_order_relaxed);
            ++g_one_pack_cqe.cqes;
            g_one_pack_cqe.bytes += src0_bytes;
            publish_plan(p);
        }
    }
    if (submitted != plan_count || reaped != submitted) {
        ggml_moe_common::object_io_ring_release(g_one_pack_cqe.io_ring);
        g_one_pack_cqe.ring_inited = false;
        g_one_pack_cqe.ready = false;
    }

    for (int p = 0; p < plan_count; ++p) {
        if (plans[p].published) continue;
        if (plans[p].read_ok) continue;
        ++g_one_pack_cqe.fallbacks;
        plans[p].read_ok = one_pack_read_entry(
            plans[p].entry, g_one_pack_cqe.import_ring.slots[p].host, src0_bytes);
        publish_plan(p);
    }
    ++g_one_pack_cqe.batches;
    g_one_pack_cqe.jobs += (uint64_t) plan_count;
    if (!ok) {
        for (int i = 0; i < count; ++i) vram_cache_cancel_descriptor(descriptors[i]);
        return false;
    }

    for (int i = 0; i < count; ++i) {
        const int p = request_plan[i];
        if (p < 0) continue;
        output_ptrs[i] = plans[p].reservation.dst;
        cache_states[i] = 4;
        if (!vram_cache_acquire_descriptor(
                output_ptrs[i], plans[p].key, 0, 0, compute_stream, descriptors[i])) {
            for (int j = 0; j < count; ++j) vram_cache_cancel_descriptor(descriptors[j]);
            return false;
        }
        ++acquired;
    }
    return acquired == count;
#endif
}

static bool one_pack_sequential_dev_ptrs_or_insert(
        const char * src0_name,
        const void * src0_base_data,
        size_t nb02,
        size_t src0_bytes,
        const int * expert_indices,
        int count,
        cudaStream_t stream,
        const void ** output_ptrs,
        int * cache_states,
        ggml_moe_expert_descriptor * descriptors) {
    int acquired = 0;
    for (int i = 0; i < count; ++i) {
        output_ptrs[i] = nullptr;
        cache_states[i] = 0;
        descriptors[i] = {};
        descriptors[i].slot_id = -1;

        const void * expert_data =
            (const char *) src0_base_data + (size_t) expert_indices[i] * nb02;
        const uintptr_t key =
            one_cache_key_for(src0_name, expert_indices[i], expert_data);
        output_ptrs[i] = ggml_cuda_moe_stream_one_cache_dev_ptr_or_insert(
            src0_name, src0_base_data, nb02, src0_bytes, expert_indices[i],
            stream, &cache_states[i]);
        if (!output_ptrs[i] || !vram_cache_acquire_descriptor(
                output_ptrs[i], key, 0, 0, stream, descriptors[i])) {
            for (int j = 0; j < acquired; ++j) {
                vram_cache_cancel_descriptor(descriptors[j]);
            }
            return false;
        }
        ++acquired;
    }
    return acquired == count;
}

static bool one_gate_up_prefetch_enabled() {
    const char * env = std::getenv("GGML_MOE_STREAM_ONE_GATE_UP_PREFETCH");
    return env && env[0] && env[0] != '0';
}

static bool one_gate_up_prefetch_make_up_name(const char * gate_name, char * up_name, size_t up_name_size) {
    if (!gate_name || !up_name || up_name_size == 0) return false;
    const char * needle = std::strstr(gate_name, ".ffn_gate_exps.");
    if (!needle) return false;
    const size_t prefix = (size_t) (needle - gate_name);
    const char * suffix = needle + std::strlen(".ffn_gate_exps.");
    const int n = std::snprintf(up_name, up_name_size, "%.*s.ffn_up_exps.%s",
        (int) prefix, gate_name, suffix);
    return n > 0 && (size_t) n < up_name_size;
}

static int one_stage_profile_role_index(const char * name);
static bool one_stage_profile_enabled();
static bool one_decode_weight_profile_start(int slot, int role, cudaStream_t stream);
static void one_decode_weight_profile_done(int slot, int role, size_t bytes, cudaStream_t stream);

static int one_gate_up_prefetch_keep_count(const char * gate_name) {
    int keep = 0;
    if (const char * env = std::getenv("GGML_MOE_KEEP_TOPK_UPDOWN")) {
        keep = std::max(0, std::atoi(env));
    }
    const char * layer_text = gate_name ? std::strstr(gate_name, "blk.") : nullptr;
    const int layer = layer_text ? std::atoi(layer_text + 4) : -1;
    int range_begin = -1;
    int range_end = -1;
    const char * range = std::getenv("GGML_MOE_KEEP_TOPK_LAYER_RANGE");
    const char * value = std::getenv("GGML_MOE_KEEP_TOPK_LAYER_VALUE");
    if (layer >= 0 && range && value &&
            std::sscanf(range, "%d-%d", &range_begin, &range_end) == 2 &&
            layer >= range_begin && layer <= range_end) {
        keep = std::max(0, std::atoi(value));
    }
    return keep;
}

static bool one_gate_up_prefetch_init_locked() {
#if !defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    return false;
#else
    if (!g_one_gate_up_prefetch.init_attempted) {
        g_one_gate_up_prefetch.init_attempted = true;
        if (cudaStreamCreateWithFlags(
                &g_one_gate_up_prefetch.stream, cudaStreamNonBlocking) != cudaSuccess) {
            cudaGetLastError();
        }
    }
    if (!g_one_gate_up_prefetch.ring_inited) {
        const char * sqpoll_env = std::getenv("GGML_MOE_STREAM_ONE_GATE_UP_SQPOLL");
        const bool request_sqpoll = sqpoll_env && sqpoll_env[0] && sqpoll_env[0] != '0';
        bool ring_ready = false;
        if (request_sqpoll) {
            ring_ready = ggml_moe_common::object_io_ring_ensure(
                g_one_gate_up_prefetch.io_ring, 8,
                ggml_moe_common::object_io_sqpoll_setup_flag(), nullptr, 2000);
            g_one_gate_up_prefetch.sqpoll = ring_ready;
        }
        if (!ring_ready) {
            ring_ready = ggml_moe_common::object_io_ring_ensure(
                g_one_gate_up_prefetch.io_ring, 8, 0, nullptr);
            g_one_gate_up_prefetch.sqpoll = false;
        }
        if (ring_ready) {
            g_one_gate_up_prefetch.ring_inited = true;
            g_one_gate_up_prefetch.fixed_file =
                ggml_moe_common::object_io_ring_register_file(
                    g_one_gate_up_prefetch.io_ring, g_one_pack.fd_direct);
        }
    }
    return g_one_gate_up_prefetch.ring_inited && g_one_gate_up_prefetch.stream;
#endif
}

static void one_gate_up_prefetch_reap(const char * src0_name) {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (!one_gate_up_prefetch_enabled() || !src0_name ||
            !std::strstr(src0_name, ".ffn_up_exps.")) return;

    std::lock_guard<std::mutex> lock(g_one_gate_up_prefetch.mu);
    const int pending = g_one_gate_up_prefetch.pending;
    const int submitted = g_one_gate_up_prefetch.submitted;
    if (pending <= 0) return;

    int reaped = 0;
    for (; reaped < submitted; ++reaped) {
        ggml_moe_common::object_io_completion completion;
        const auto wait_start = std::chrono::steady_clock::now();
        if (!ggml_moe_common::object_io_ring_wait(
                g_one_gate_up_prefetch.io_ring, completion)) break;
        g_one_gate_up_prefetch.residual_wait_us += (uint64_t)
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - wait_start).count();
        const bool plan_valid = completion.user_data < (uint64_t) pending;
        const int p = plan_valid ? (int) completion.user_data : 0;
        const size_t expected_bytes = plan_valid ?
            (size_t) g_one_gate_up_prefetch.jobs[p].entry->nbytes : 0;
        if (ggml_moe_common::object_io_completion_validate(
                g_one_gate_up_prefetch.io_ring, completion, plan_valid, expected_bytes) ==
                ggml_moe_common::object_io_completion_status::complete) {
            one_gate_up_prefetch_job & job = g_one_gate_up_prefetch.jobs[p];
            ggml_moe_common::object_stage_slot & import =
                g_one_gate_up_prefetch.import_ring.slots[p];
            job.read_ok = true;
            const auto h2d_start = std::chrono::steady_clock::now();
            const int profile_role = one_stage_profile_enabled() ?
                one_stage_profile_role_index(g_one_gate_up_prefetch.tensor) : -1;
            const bool profile_weight = one_decode_weight_profile_start(
                p + 8, profile_role, g_one_gate_up_prefetch.stream);
            const bool copied = cudaMemcpyAsync(
                job.reservation.dst, import.host, (size_t) job.entry->nbytes,
                cudaMemcpyHostToDevice, g_one_gate_up_prefetch.stream) == cudaSuccess;
            if (copied && profile_weight) {
                one_decode_weight_profile_done(
                    p + 8, profile_role, (size_t) job.entry->nbytes,
                    g_one_gate_up_prefetch.stream);
            }
            job.published = copied &&
                vram_cache_publish_triplet(job.reservation, g_one_gate_up_prefetch.stream);
            g_one_gate_up_prefetch.h2d_enqueue_us += (uint64_t)
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - h2d_start).count();
            if (job.published) {
                g_one_pack.reads.fetch_add(1, std::memory_order_relaxed);
                g_one_pack.direct_reads.fetch_add(1, std::memory_order_relaxed);
                g_one_pack.bytes.fetch_add(job.entry->nbytes, std::memory_order_relaxed);
                ++g_one_gate_up_prefetch.cqes;
                g_one_gate_up_prefetch.bytes += job.entry->nbytes;
                if (!ggml_moe_common::object_stage_slot_record(
                        import, g_one_gate_up_prefetch.stream)) {
                    cudaGetLastError();
                    cudaStreamSynchronize(g_one_gate_up_prefetch.stream);
                }
            } else {
                cudaGetLastError();
                if (copied) cudaStreamSynchronize(g_one_gate_up_prefetch.stream);
            }
        }
    }

    for (int p = 0; p < pending; ++p) {
        one_gate_up_prefetch_job & job = g_one_gate_up_prefetch.jobs[p];
        if (!job.published) {
            vram_cache_fail_triplet(job.reservation);
            ++g_one_gate_up_prefetch.demand_fallbacks;
        }
        job = {};
    }
    g_one_gate_up_prefetch.pending = 0;
    g_one_gate_up_prefetch.submitted = 0;
    g_one_gate_up_prefetch.tensor[0] = '\0';
    if (submitted != pending || reaped != submitted) {
        ggml_moe_common::object_io_ring_release(g_one_gate_up_prefetch.io_ring);
        g_one_gate_up_prefetch.ring_inited = false;
        g_one_gate_up_prefetch.sqpoll = false;
        g_one_gate_up_prefetch.fixed_file = false;
    }
#else
    (void) src0_name;
#endif
}

static void one_gate_up_prefetch_schedule(
        const char * src0_name,
        size_t src0_bytes,
        const int * expert_indices,
        const int32_t * route_ranks,
        int count) {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (!one_gate_up_prefetch_enabled() || !src0_name ||
            !std::strstr(src0_name, ".ffn_gate_exps.") ||
            !expert_indices || count <= 0 || count > 8) return;

    one_pack_init_once();
    if (!g_one_pack.enabled || g_one_pack.fd_direct < 0) return;
    if (!g_vcache_inited) {
        std::lock_guard<std::mutex> init_lock(g_init_mu);
        if (!g_vcache_inited) vram_cache_init(src0_bytes);
    }
    if (!g_vcache.pool || g_vcache.n_slots <= 0 || src0_bytes > g_vcache.slot_sz) return;

    char up_name[128] = {};
    if (!one_gate_up_prefetch_make_up_name(src0_name, up_name, sizeof(up_name))) return;

    std::lock_guard<std::mutex> lock(g_one_gate_up_prefetch.mu);
    ++g_one_gate_up_prefetch.calls;
    if (g_one_gate_up_prefetch.pending > 0 || !one_gate_up_prefetch_init_locked()) return;
    if (g_one_gate_up_prefetch.import_ring.slots.empty()) {
        g_one_gate_up_prefetch.import_ring.slot_sz = src0_bytes;
        g_one_gate_up_prefetch.import_ring.slots.resize(8);
    }
    if (src0_bytes > g_one_gate_up_prefetch.import_ring.slot_sz) return;

    const int keep_count = one_gate_up_prefetch_keep_count(src0_name);
    int job_count = 0;
    for (int i = 0; i < count; ++i) {
        if (route_ranks && keep_count > 0 && route_ranks[i] >= keep_count) {
            ++g_one_gate_up_prefetch.placement_skips;
            continue;
        }
        bool duplicate = false;
        for (int j = 0; j < i; ++j) {
            if (expert_indices[j] == expert_indices[i]) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) continue;

        const uintptr_t key = one_named_cache_key(up_name, expert_indices[i]);
        if (vram_cache_contains_key(key)) {
            ++g_one_gate_up_prefetch.resident_skips;
            continue;
        }
        if (!moe_stream_cache_admit_allows(up_name, expert_indices[i]) ||
                vram_cache_freeze_reject_miss()) continue;
        const one_expert_pack_entry * entry =
            one_pack_lookup(up_name, expert_indices[i], src0_bytes);
        if (!entry) continue;

        ggml_moe_common::object_stage_slot & import =
            g_one_gate_up_prefetch.import_ring.slots[job_count];
        if (!ggml_moe_common::object_stage_slot_synchronize(
                    g_one_gate_up_prefetch.import_ring, import) ||
                (!import.host &&
                 !ggml_moe_common::object_stage_slot_allocate(
                     import, g_one_gate_up_prefetch.import_ring.slot_sz, false))) {
            cudaGetLastError();
            continue;
        }

        one_gate_up_prefetch_job & job = g_one_gate_up_prefetch.jobs[job_count];
        job = {};
        job.entry = entry;
        if (!vram_cache_reserve_triplet(
                key, src0_bytes, g_one_gate_up_prefetch.stream, job.reservation, up_name)) {
            job = {};
            continue;
        }
        if (!ggml_moe_common::object_io_ring_prepare_read(
                g_one_gate_up_prefetch.io_ring,
                g_one_pack.fd_direct,
                import.host,
                src0_bytes,
                entry->offset,
                (uint64_t) job_count,
                g_one_gate_up_prefetch.fixed_file)) {
            vram_cache_fail_triplet(job.reservation);
            job = {};
            for (int p = 0; p < job_count; ++p) {
                vram_cache_fail_triplet(g_one_gate_up_prefetch.jobs[p].reservation);
                g_one_gate_up_prefetch.jobs[p] = {};
            }
            ggml_moe_common::object_io_ring_release(g_one_gate_up_prefetch.io_ring);
            g_one_gate_up_prefetch.ring_inited = false;
            g_one_gate_up_prefetch.sqpoll = false;
            g_one_gate_up_prefetch.fixed_file = false;
            return;
        }
        ++job_count;
    }
    if (job_count == 0) return;

    const auto submit_start = std::chrono::steady_clock::now();
    int submitted = 0;
    while (submitted < job_count) {
        const int rc = ggml_moe_common::object_io_ring_submit_count(
            g_one_gate_up_prefetch.io_ring);
        if (rc <= 0) break;
        submitted += rc;
    }
    g_one_gate_up_prefetch.submit_us += (uint64_t)
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - submit_start).count();
    if (submitted <= 0) {
        ggml_moe_common::object_io_ring_release(g_one_gate_up_prefetch.io_ring);
        g_one_gate_up_prefetch.ring_inited = false;
        g_one_gate_up_prefetch.sqpoll = false;
        g_one_gate_up_prefetch.fixed_file = false;
        for (int p = 0; p < job_count; ++p) {
            vram_cache_fail_triplet(g_one_gate_up_prefetch.jobs[p].reservation);
            g_one_gate_up_prefetch.jobs[p] = {};
        }
        return;
    }
    g_one_gate_up_prefetch.pending = job_count;
    g_one_gate_up_prefetch.submitted = submitted;
    std::snprintf(g_one_gate_up_prefetch.tensor, sizeof(g_one_gate_up_prefetch.tensor), "%s", up_name);
    g_one_gate_up_prefetch.scheduled += (uint64_t) job_count;
#else
    (void) src0_name; (void) src0_bytes; (void) expert_indices; (void) route_ranks; (void) count;
#endif
}

extern "C" void ggml_cuda_moe_stream_one_gate_up_prefetch(
        const char * src0_name,
        size_t src0_bytes,
        const int * expert_indices,
        const int32_t * route_ranks,
        int count) {
    one_gate_up_prefetch_schedule(src0_name, src0_bytes, expert_indices, route_ranks, count);
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
    if (!src0_name || !src0_base_data || nb02 == 0 ||
            !expert_indices || count <= 0 || count > 8 || !output_ptrs || !cache_states || !descriptors) return false;

    if (!ds4_triplet_requested()) {
        one_gate_up_prefetch_reap(src0_name);
        if (one_pack_cqe_h2d_enabled()) {
            return one_pack_cqe_h2d_dev_ptrs_or_insert(
                src0_name, src0_base_data, nb02, src0_bytes, expert_indices, count,
                stream, output_ptrs, cache_states, descriptors);
        }
        return one_pack_sequential_dev_ptrs_or_insert(
            src0_name, src0_base_data, nb02, src0_bytes, expert_indices, count,
            stream, output_ptrs, cache_states, descriptors);
    }

    ds4_triplet_init_once();
    if (ds4_dependency_h2d_requested() && ds4_dependency_h2d_eligible()) {
        return ds4_triplet_dependency_h2d_dev_ptrs_or_insert(
            src0_name, src0_bytes, expert_indices, count, stream,
            output_ptrs, cache_states, descriptors);
    }

    if (ds4_triplet_role_pair_requested()) {
        return ds4_triplet_role_pair_dev_ptrs_or_insert(
            src0_name, src0_bytes, expert_indices, count, stream,
            output_ptrs, cache_states, descriptors);
    }

    const ds4_triplet_pack_entry * requested[8] = {};
    size_t role_offsets[8] = {};
    int roles[8] = {};
    int jobs_for_request[8];
    const ds4_triplet_pack_entry * jobs[8] = {};
    int job_count = 0;
    for (int i = 0; i < count; ++i) {
        output_ptrs[i] = nullptr;
        cache_states[i] = 0;
        descriptors[i] = {};
        descriptors[i].slot_id = -1;
        jobs_for_request[i] = -1;
        int role = -1;
        requested[i] = ds4_triplet_lookup(src0_name, expert_indices[i], src0_bytes, role);
        if (!requested[i]) return false;
        roles[i] = role;
        role_offsets[i] = role == 0 ? 0 : role == 1 ?
            (size_t) g_ds4_triplet_pack.role_bytes[0] :
            (size_t) (g_ds4_triplet_pack.role_bytes[0] + g_ds4_triplet_pack.role_bytes[1]);
    }
    const int call_role = roles[0];
    const int call_layer = requested[0]->layer;
    const uint64_t call_id = g_ds4_triplet_trace.next_call.fetch_add(1, std::memory_order_relaxed);
    const uint64_t layer_cycle = ds4_triplet_trace_cycle(call_layer, call_role);
    g_ds4_triplet_pack.role_calls[call_role].fetch_add(1, std::memory_order_relaxed);
    for (int i = 0; i < count; ++i) {
        g_ds4_triplet_pack.role_lookups[roles[i]].fetch_add(1, std::memory_order_relaxed);
        if (roles[i] != call_role || requested[i]->layer != call_layer) {
            g_ds4_triplet_pack.mixed_role_calls.fetch_add(1, std::memory_order_relaxed);
        }
    }

    if (roles[0] == 0 && ds4_ram_pilot_prefetch_enabled()) {
        static std::mutex route_mu;
        static int previous_routes[43][8] = {};
        static int previous_counts[43] = {};
        int predicted[8] = {};
        int predicted_count = 0;
        const int layer = requested[0]->layer;
        {
            std::lock_guard<std::mutex> route_lock(route_mu);
            if (layer + 1 < 43) {
                predicted_count = std::min(previous_counts[layer + 1], ds4_ram_pilot_prefetch_topk());
                std::copy(previous_routes[layer + 1], previous_routes[layer + 1] + predicted_count, predicted);
            }
            previous_counts[layer] = count;
            std::copy(expert_indices, expert_indices + count, previous_routes[layer]);
        }
        for (int i = 0; i < predicted_count; ++i) {
            ds4_ram_pilot_prefetch_enqueue(layer + 1, predicted[i]);
        }
    }

    if (!g_vcache_inited) {
        std::lock_guard<std::mutex> lock(g_init_mu);
        if (!g_vcache_inited) vram_cache_init((size_t) requested[0]->nbytes);
    }
    if (!g_vcache.pool || g_vcache.n_slots <= 0) return false;

    for (int i = 0; i < count; ++i) {
        const uintptr_t key = (uintptr_t) requested[i]->generation;
        if (void * cached = vram_cache_lookup(key, stream)) {
            output_ptrs[i] = (char *) cached + role_offsets[i];
            cache_states[i] = 1;
            g_ds4_triplet_pack.role_hits[roles[i]].fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        g_ds4_triplet_pack.role_misses[roles[i]].fetch_add(1, std::memory_order_relaxed);
        if (!moe_stream_cache_admit_allows(src0_name, expert_indices[i])) {
            g_vcache.misses.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (vram_cache_freeze_reject_miss()) {
            cache_states[i] = 5;
            continue;
        }
        if (void * promoted = vram_cache_promote_from_ram(key, (size_t) requested[i]->nbytes, stream, "ds4_triplet")) {
            output_ptrs[i] = (char *) promoted + role_offsets[i];
            cache_states[i] = 2;
            g_ds4_triplet_pack.role_ram_promotes[roles[i]].fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        for (int job = 0; job < job_count; ++job) {
            if (jobs[job]->generation == requested[i]->generation) {
                jobs_for_request[i] = job;
                break;
            }
        }
        if (jobs_for_request[i] < 0) {
            jobs_for_request[i] = job_count;
            jobs[job_count++] = requested[i];
        }
    }

    auto acquire_descriptors = [&]() {
        int acquired = 0;
        for (int i = 0; i < count; ++i) {
            if (!output_ptrs[i]) continue;
            const void * object_ptr = (const char *) output_ptrs[i] - role_offsets[i];
            if (!vram_cache_acquire_descriptor(
                    object_ptr, (uintptr_t) requested[i]->generation, roles[i], role_offsets[i],
                    stream, descriptors[i])) {
                for (int j = 0; j < i; ++j) vram_cache_cancel_descriptor(descriptors[j]);
                for (int request = 0; request < count; ++request) {
                    ds4_triplet_trace_record(call_id, layer_cycle, requested[request], roles[request], request,
                            route_ranks ? route_ranks[request] : request, "descriptor_failure",
                            cache_states[request], descriptors[request].slot_id,
                            descriptors[request].slot_generation);
                }
                return false;
            }
            ++acquired;
        }
        for (int i = 0; i < count; ++i) {
            const char * event = cache_states[i] == 1 ? "vram_hit" :
                cache_states[i] == 2 ? "ram_promote" :
                cache_states[i] == 4 ? "pack_insert" :
                cache_states[i] == 5 ? "frozen_miss" : "unresolved";
            ds4_triplet_trace_record(call_id, layer_cycle, requested[i], roles[i], i,
                    route_ranks ? route_ranks[i] : i, event, cache_states[i],
                    descriptors[i].slot_id, descriptors[i].slot_generation);
        }
        return acquired > 0;
    };

    if (job_count == 0) return acquire_descriptors();
    static ggml_moe_common::object_stage_ring import_ring;
    static std::mutex import_mu;
    std::lock_guard<std::mutex> import_lock(import_mu);

    if (import_ring.slots.empty()) {
        import_ring.slot_sz = (size_t) jobs[0]->nbytes;
        import_ring.slots.resize(8);
    }
    void * destinations[8] = {};
    for (int job = 0; job < job_count; ++job) {
        if (jobs[job]->nbytes > import_ring.slot_sz) return false;
        ggml_moe_common::object_stage_slot & import = import_ring.slots[job];
        if (!ggml_moe_common::object_stage_slot_synchronize(import_ring, import)) return false;
        if (!import.host &&
                !ggml_moe_common::object_stage_slot_allocate(import, import_ring.slot_sz, false)) {
            return false;
        }
        ds4_triplet_host_handoff_invalidate(import.host);
        destinations[job] = import.host;
    }
    void * inserted[8] = {};
    bool pipe_done = false;
    if (g_ds4_triplet_pack.pipe_requested && !g_ds4_triplet_pack.pipe_failed &&
            g_ds4_triplet_pack.pipe_stream) {
        ds4_triplet_pipe_batch pipe_batch;
        pipe_batch.count = job_count;
        bool reserved = true;
        for (int job = 0; job < job_count; ++job) {
            pipe_batch.entries[job] = jobs[job];
            pipe_batch.host[job] = destinations[job];
            if (!vram_cache_reserve_triplet(
                    (uintptr_t) jobs[job]->generation, (size_t) jobs[job]->nbytes,
                    g_ds4_triplet_pack.pipe_stream, pipe_batch.reservations[job])) {
                reserved = false;
                break;
            }
        }
        if (reserved && ds4_triplet_pipe_submit(pipe_batch)) {
            pipe_done = true;
            for (int job = 0; job < job_count; ++job) {
                inserted[job] = pipe_batch.reservations[job].dst;
                if (!ggml_moe_common::object_stage_slot_record(
                            import_ring.slots[job], g_ds4_triplet_pack.pipe_stream)) {
                    cudaGetLastError();
                    cudaStreamSynchronize(g_ds4_triplet_pack.pipe_stream);
                }
            }
        } else {
            cudaStreamSynchronize(g_ds4_triplet_pack.pipe_stream);
            for (int job = 0; job < job_count; ++job) {
                if (pipe_batch.reservations[job].slot >= 0) {
                    vram_cache_fail_triplet(pipe_batch.reservations[job]);
                }
            }
        }
    }

    if (!pipe_done) {
        if (!ds4_triplet_read_batch(jobs, destinations, job_count)) return false;
        for (int job = 0; job < job_count; ++job) {
            inserted[job] = vram_cache_insert(
                (uintptr_t) jobs[job]->generation, destinations[job], (size_t) jobs[job]->nbytes, stream, "ds4_triplet");
            if (!inserted[job]) return false;
            if (!ggml_moe_common::object_stage_slot_record(import_ring.slots[job], stream)) {
                cudaGetLastError();
                cudaStreamSynchronize(stream);
            }
        }
    }
    for (int i = 0; i < count; ++i) {
        const int job = jobs_for_request[i];
        if (job >= 0) {
            output_ptrs[i] = (char *) inserted[job] + role_offsets[i];
            cache_states[i] = 4;
        }
    }
    for (int job = 0; job < job_count; ++job) {
        int request = 0;
        while (request < count && jobs_for_request[request] != job) ++request;
        const int role = request < count ? roles[request] : call_role;
        g_ds4_triplet_pack.role_physical_reads[role].fetch_add(1, std::memory_order_relaxed);
        ds4_triplet_trace_record(call_id, layer_cycle, jobs[job], role, request,
                request < count && route_ranks ? route_ranks[request] : request,
                "physical_read", 4, -1, 0);
    }
    for (int job = 0; job < job_count; ++job) {
        ds4_triplet_host_handoff_publish(import_ring.slots[job].host, jobs[job]->generation);
    }
    return acquire_descriptors();
}

static void vram_cache_release_descriptors(
        const ggml_moe_expert_descriptor * descriptors,
        int count,
        cudaStream_t stream,
        bool stream_synchronized) {
    if (!descriptors || count <= 0 || (!stream && !stream_synchronized)) return;
    std::lock_guard<std::mutex> lk(g_vcache.mu);
    for (int i = 0; i < count; ++i) {
        const ggml_moe_expert_descriptor & descriptor = descriptors[i];
        const int slot = descriptor.slot_id;
        if (slot < 0) continue;
        bool last_lease = false;
        if (!ggml_moe_common::role_cache_release_descriptor(
                vram_cache_role_view(), (uintptr_t) descriptor.object_generation,
                descriptor, &last_lease)) {
            g_vcache.descriptor_stale_releases.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (last_lease && !stream_synchronized &&
                !vram_cache_record_slot_lease_locked(slot, stream)) {
            cudaStreamSynchronize(stream);
        }
        g_vcache.descriptor_releases.fetch_add(1, std::memory_order_relaxed);
    }
}

extern "C" void ggml_cuda_moe_stream_one_cache_release_descriptors(
        const ggml_moe_expert_descriptor * descriptors,
        int count,
        cudaStream_t stream) {
    vram_cache_release_descriptors(descriptors, count, stream, false);
}

extern "C" void ggml_cuda_moe_stream_one_cache_release_descriptors_synchronized(
        const ggml_moe_expert_descriptor * descriptors,
        int count) {
    vram_cache_release_descriptors(descriptors, count, nullptr, true);
}

extern "C" bool ggml_cuda_moe_stream_one_cache_contains(
    const char *src0_name,
    const void *src0_data,
    size_t src0_bytes,
    int64_t expert_index)
{
    (void) src0_bytes;
    if (!src0_name || !src0_data || expert_index < 0) {
        return false;
    }
    if (ds4_triplet_requested()) {
        int role = -1;
        const ds4_triplet_pack_entry * entry = ds4_triplet_lookup(src0_name, expert_index, src0_bytes, role);
        return entry && vram_cache_contains_key((uintptr_t) entry->generation);
    }
    const uintptr_t key = one_cache_key_for(src0_name, expert_index, src0_data);
    return vram_cache_contains_key(key);
}

static int one_stage_profile_role_index(const char * name);
static bool one_stage_profile_enabled();
static void one_decode_weight_profile_finalize(int slot);
static bool one_decode_weight_profile_start(int slot, int role, cudaStream_t stream);
static void one_decode_weight_profile_done(int slot, int role, size_t bytes, cudaStream_t stream);

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
    if (!src0_name || !src0_base_data || nb02 == 0 || src0_bytes == 0 || expert_index < 0) {
        return nullptr;
    }

    if (ds4_triplet_requested()) {
        return ds4_triplet_cache_dev_ptr_or_insert(
            src0_name, src0_bytes, expert_index, st, cache_state);
    }

    if (!g_vcache_inited) {
        std::lock_guard<std::mutex> lk(g_init_mu);
        if (!g_vcache_inited) {
            vram_cache_init(src0_bytes);
        }
    }
    if (!g_vcache.pool || g_vcache.n_slots <= 0 || src0_bytes > g_vcache.slot_sz) {
        return nullptr;
    }

    const void *expert_data = (const char *) src0_base_data + (size_t) expert_index * nb02;
    const uintptr_t key = one_cache_key_for(src0_name, expert_index, expert_data);

    if (void *cached = vram_cache_lookup(key, st)) {
        if (cache_state) {
            *cache_state = 1;
        }
        return cached;
    }

    const bool cache_admit = moe_stream_cache_admit_allows(src0_name, expert_index);
    if (!cache_admit) {
        g_vcache.misses.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    if (vram_cache_freeze_reject_miss()) {
        if (cache_state) {
            *cache_state = 5;
        }
        return nullptr;
    }

    if (void *promoted = vram_cache_promote_from_ram(key, src0_bytes, st, src0_name)) {
        if (cache_state) {
            *cache_state = 2;
        }
        return promoted;
    }

    const void *copy_src = expert_data;
    const int profile_role = one_stage_profile_enabled() ? one_stage_profile_role_index(src0_name) : -1;
    int pack_slot = -1;
    static ggml_moe_common::object_stage_ring import_ring;
    static std::mutex import_mu;

    const char *decode_one_pack_env = std::getenv("GGML_MOE_STREAM_DECODE_BATCH_ONE_PACK");
    if (!decode_one_pack_env || !decode_one_pack_env[0] || decode_one_pack_env[0] != '0') {
        if (const one_expert_pack_entry *pack_entry = one_pack_lookup(src0_name, expert_index, src0_bytes)) {
            std::lock_guard<std::mutex> lk(import_mu);
            if (import_ring.slots.empty()) {
                import_ring.slot_sz = src0_bytes;
                import_ring.slots.resize(8);
            }
            if (src0_bytes <= import_ring.slot_sz) {
                pack_slot = (int) (import_ring.next++ % import_ring.slots.size());
            }
            ggml_moe_common::object_stage_slot *slot =
                pack_slot >= 0 ? &import_ring.slots[pack_slot] : nullptr;
            if (!slot ||
                    !ggml_moe_common::object_stage_slot_synchronize(import_ring, *slot) ||
                    (!slot->host &&
                     !ggml_moe_common::object_stage_slot_allocate(*slot, import_ring.slot_sz, false))) {
                cudaGetLastError();
                pack_slot = -1;
            } else if (one_pack_read_entry(pack_entry, slot->host, src0_bytes)) {
                one_decode_weight_profile_finalize(pack_slot);
                copy_src = slot->host;
            } else {
                pack_slot = -1;
                copy_src = expert_data;
            }
        }
    }

    const bool profile_weight = pack_slot >= 0 &&
        one_decode_weight_profile_start(pack_slot, profile_role, st);
    void *inserted = vram_cache_insert(key, copy_src, src0_bytes, st, src0_name);
    if (inserted) {
        if (profile_weight) {
            one_decode_weight_profile_done(pack_slot, profile_role, src0_bytes, st);
        }
        if (pack_slot >= 0) {
            ggml_moe_common::object_stage_slot &slot = import_ring.slots[pack_slot];
            if (!ggml_moe_common::object_stage_slot_record(slot, st)) {
                cudaGetLastError();
                cudaStreamSynchronize(st);
            }
        }
        if (cache_state) {
            *cache_state = pack_slot >= 0 ? 4 : 3;
        }
        return inserted;
    }
    return nullptr;
}

extern "C" bool ggml_cuda_moe_stream_one_cache_ready(void) {
    return g_vcache_inited && g_vcache.pool && g_vcache.n_slots > 0;
}

static void one_prefill_maybe(slot_ctx & ctx, cudaStream_t st, size_t src0_bytes) {
    if (!one_prefill_key_mode_enabled() || !one_prefill_enabled()) {
        return;
    }

    std::lock_guard<std::mutex> lk(g_one_prefill.mu);
    if (g_one_prefill.done || !g_one_prefill.enabled || !g_vcache.pool || g_vcache.n_slots <= 0) {
        return;
    }
    g_one_prefill.done = true;

    size_t n = g_one_prefill.limit;
    if (n > (size_t) g_vcache.n_slots) {
        n = (size_t) g_vcache.n_slots;
    }
    if (n > g_one_prefill.entries.size()) {
        n = g_one_prefill.entries.size();
    }

    {
        std::lock_guard<std::mutex> rk(g_resize_mu);
        if (!ensure_host_pinned(ctx.h_src0_pack, ctx.h_src0_pack_sz, src0_bytes)) {
            g_one_prefill.read_failures += n;
            return;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < n; ++i) {
        const one_prefill_entry & e = g_one_prefill.entries[i];
        g_one_prefill.attempted++;
        const one_expert_pack_entry * pack_entry = one_pack_lookup(e.tensor.c_str(), e.expert, src0_bytes);
        if (!pack_entry) {
            g_one_prefill.pack_misses++;
            continue;
        }
        if (!one_pack_read_entry(pack_entry, ctx.h_src0_pack, src0_bytes)) {
            g_one_prefill.read_failures++;
            continue;
        }
        const uintptr_t key = one_named_cache_key(e.tensor.c_str(), e.expert);
        if (vram_cache_insert_impl(key, ctx.h_src0_pack, src0_bytes, st, false, e.tensor.c_str())) {
            g_one_prefill.inserted++;
            g_one_prefill.bytes += src0_bytes;
        } else {
            g_one_prefill.read_failures++;
        }
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) {
        g_one_prefill.read_failures++;
    }
    const auto t1 = std::chrono::steady_clock::now();
    g_one_prefill.elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr,
        "[moe_stream] one prefill: completed attempted=%lu inserted=%lu bytes=%lu elapsed_ms=%.3f\n",
        g_one_prefill.attempted,
        g_one_prefill.inserted,
        g_one_prefill.bytes,
        g_one_prefill.elapsed_ms);
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

static int one_stage_profile_role_index(const char * name) {
    if (!name) return -1;
    if (std::strstr(name, "ffn_gate_exps")) return 0;
    if (std::strstr(name, "ffn_up_exps")) return 1;
    if (std::strstr(name, "ffn_down_exps")) return 2;
    return -1;
}

static void one_decode_weight_profile_finalize(int slot) {
    one_decode_weight_profile_slot & event = g_one_decode_weight_profile[slot];
    if (!event.pending || event.role < 0) return;
    float elapsed_ms = 0.0f;
    if (cudaEventSynchronize(event.done) == cudaSuccess &&
            cudaEventElapsedTime(&elapsed_ms, event.start, event.done) == cudaSuccess) {
        one_stage_profile_role & p = g_one_stage_profile[event.role];
        p.weight_h2d_us.fetch_add((uint64_t) (elapsed_ms * 1000.0f), std::memory_order_relaxed);
        p.weight_h2d_copies.fetch_add(1, std::memory_order_relaxed);
        p.weight_h2d_bytes.fetch_add(event.bytes, std::memory_order_relaxed);
    } else {
        cudaGetLastError();
    }
    event.pending = false;
}

static bool one_decode_weight_profile_start(int slot, int role, cudaStream_t stream) {
    if (slot < 0 || slot >= 16 || role < 0) return false;
    one_decode_weight_profile_finalize(slot);
    one_decode_weight_profile_slot & event = g_one_decode_weight_profile[slot];
    if ((!event.start && cudaEventCreate(&event.start) != cudaSuccess) ||
            (!event.done && cudaEventCreate(&event.done) != cudaSuccess) ||
            cudaEventRecord(event.start, stream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

static void one_decode_weight_profile_done(
        int slot, int role, size_t bytes, cudaStream_t stream) {
    if (slot < 0 || slot >= 16 || role < 0) return;
    one_decode_weight_profile_slot & event = g_one_decode_weight_profile[slot];
    if (cudaEventRecord(event.done, stream) != cudaSuccess) {
        cudaGetLastError();
        return;
    }
    event.pending = true;
    event.role = role;
    event.bytes = bytes;
}

static void one_stage_profile_report() {
    for (int slot = 0; slot < 16; ++slot) one_decode_weight_profile_finalize(slot);
    static const char * names[] = {"gate", "up", "down"};
    auto ms = [](uint64_t ns) { return ns / 1000000.0; };
    for (int role = 0; role < 3; ++role) {
        const uint64_t calls = g_one_stage_profile[role].calls.load();
        const one_stage_profile_role & p = g_one_stage_profile[role];
        if (!calls && !p.weight_h2d_bytes.load()) continue;
        std::fprintf(stderr,
            "[moe_stream_one_profile] role=%s calls=%lu cache_hits=%lu cache_misses=%lu "
            "src0_ms=%.3f host_prepare_ms=%.3f enqueue_ms=%.3f sync_ms=%.3f "
            "scatter_ms=%.3f total_ms=%.3f weight_h2d_ms=%.3f input_h2d_ms=%.3f "
            "kernel_ms=%.3f d2h_ms=%.3f weight_h2d_copies=%lu weight_h2d_bytes=%lu "
            "input_h2d_bytes=%lu d2h_bytes=%lu\n",
            names[role], calls, p.cache_hits.load(), p.cache_misses.load(),
            ms(p.src0_ns.load()), ms(p.host_prepare_ns.load()), ms(p.enqueue_ns.load()),
            ms(p.sync_ns.load()), ms(p.scatter_ns.load()), ms(p.total_ns.load()),
            p.weight_h2d_us.load() / 1000.0, p.input_h2d_us.load() / 1000.0,
            p.kernel_us.load() / 1000.0, p.d2h_us.load() / 1000.0,
            p.weight_h2d_copies.load(), p.weight_h2d_bytes.load(),
            p.input_h2d_bytes.load(), p.d2h_bytes.load());
    }
}

static bool one_stage_profile_enabled() {
    static bool enabled = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_ONE_PROFILE");
        const bool value = env && env[0] && env[0] != '0';
        if (value) std::atexit(one_stage_profile_report);
        return value;
    }();
    return enabled;
}

static bool one_stage_profile_events_init(slot_ctx & ctx) {
    if (ctx.profile_weight_start) return true;
    cudaEvent_t * events[] = {
        &ctx.profile_weight_start,
        &ctx.profile_weight_done,
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

static double trace_elapsed_ms(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(t - g_one_trace.start).count();
}


static bool gate_updown_cosubmit_profile_enabled() {
    const char *path = std::getenv("GGML_MOE_GATE_UPDOWN_COSUBMIT_PROFILE_OUT");
    return path && path[0];
}

static bool gate_updown_cosubmit_enabled() {
    const char *env = std::getenv("GGML_MOE_GATE_UPDOWN_COSUBMIT");
    return env && env[0] && env[0] != '0';
}

static uint64_t gate_updown_profile_now_us() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

static void gate_updown_cosubmit_profile_write(
        const char *phase,
        const char *tensor,
        const char *pair_tensor,
        int64_t expert,
        size_t expert_bytes,
        int64_t n_active,
        uint64_t active_hash,
        int cache_hit,
        int cache_inserted,
        int cache_hits,
        int cache_misses,
        int pack_hit,
        int pack_hits,
        int pack_misses,
        double src0_ms,
        double total_ms) {
    const char *path = std::getenv("GGML_MOE_GATE_UPDOWN_COSUBMIT_PROFILE_OUT");
    if (!path || !path[0]) return;
    static std::mutex mu;
    static bool header = false;
    std::lock_guard<std::mutex> lk(mu);
    FILE *f = std::fopen(path, "a");
    if (!f) return;
    if (!header) {
        std::fprintf(f,
                "phase,time_us,tensor,pair_tensor,expert,expert_bytes,n_active,active_hash,"
                "cache_hit,cache_inserted,cache_hits,cache_misses,pack_hit,pack_hits,pack_misses,src0_ms,total_ms\n");
        header = true;
    }
    std::fprintf(f,
            "%s,%lu,%s,%s,%ld,%lu,%ld,%lu,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f\n",
            phase ? phase : "",
            (unsigned long)gate_updown_profile_now_us(),
            tensor ? tensor : "",
            pair_tensor ? pair_tensor : "",
            (long)expert,
            (unsigned long)expert_bytes,
            (long)n_active,
            (unsigned long)active_hash,
            cache_hit,
            cache_inserted,
            cache_hits,
            cache_misses,
            pack_hit,
            pack_hits,
            pack_misses,
            src0_ms,
            total_ms);
    std::fclose(f);
}

static FILE * one_trace_fp_locked(std::chrono::steady_clock::time_point start_time) {
    if (!g_one_trace.initialized) {
        g_one_trace.initialized = true;
        g_one_trace.start = start_time;
        const char * path = std::getenv("GGML_MOE_STREAM_ONE_TRACE_OUT");
        if (path && path[0]) {
            g_one_trace.fp = std::fopen(path, "w");
            if (g_one_trace.fp) {
                std::setvbuf(g_one_trace.fp, nullptr, _IOLBF, 0);
                std::fprintf(g_one_trace.fp,
                    "seq,t_ms,tensor,expert,src0_ptr,src0_bytes,cne1,cache_hit,cache_inserted,slot,src0_ms,src1_ms,kernel_ms,d2h_ms,sync_ms,scatter_ms,dontneed_ms,total_ms\n");
            } else {
                std::fprintf(stderr, "[moe_stream] failed to open one trace: %s\n", path);
            }
        }
    }
    return g_one_trace.fp;
}

static void one_trace_write(
    const char * src0_name,
    int64_t expert_index,
    const void * src0_data,
    size_t src0_bytes,
    int64_t cne1,
    bool cache_hit,
    bool cache_inserted,
    int slot,
    std::chrono::steady_clock::time_point t0,
    std::chrono::steady_clock::time_point t_src0,
    std::chrono::steady_clock::time_point t_src1,
    std::chrono::steady_clock::time_point t_kernel,
    std::chrono::steady_clock::time_point t_d2h,
    std::chrono::steady_clock::time_point t_sync,
    std::chrono::steady_clock::time_point t_scatter,
    std::chrono::steady_clock::time_point t_dontneed) {
    std::lock_guard<std::mutex> lk(g_one_trace.mu);
    FILE * fp = one_trace_fp_locked(t0);
    if (!fp) {
        return;
    }
    const uint64_t seq = g_one_trace.seq.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(fp,
        "%lu,%.3f,%s,%ld,%p,%zu,%ld,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
        (unsigned long) seq,
        trace_elapsed_ms(t0),
        src0_name ? src0_name : "",
        (long) expert_index,
        src0_data,
        src0_bytes,
        (long) cne1,
        cache_hit ? 1 : 0,
        cache_inserted ? 1 : 0,
        slot,
        std::chrono::duration<double, std::milli>(t_src0 - t0).count(),
        std::chrono::duration<double, std::milli>(t_src1 - t_src0).count(),
        std::chrono::duration<double, std::milli>(t_kernel - t_src1).count(),
        std::chrono::duration<double, std::milli>(t_d2h - t_kernel).count(),
        std::chrono::duration<double, std::milli>(t_sync - t_d2h).count(),
        std::chrono::duration<double, std::milli>(t_scatter - t_sync).count(),
        std::chrono::duration<double, std::milli>(t_dontneed - t_scatter).count(),
        std::chrono::duration<double, std::milli>(t_dontneed - t0).count());
}

struct sparse_graph_probe_state {
    std::mutex mu;
    FILE * fp = nullptr;
    bool initialized = false;
    std::atomic<uint64_t> seq{0};
};

static sparse_graph_probe_state g_sparse_graph_probe;

static bool sparse_graph_probe_enabled() {
    static int enabled = [] {
        const char * env = std::getenv("DS4_SPARSE_PAIR_GRAPH_PROBE");
        return (env && env[0] && env[0] != '0') ? 1 : 0;
    }();
    return enabled != 0;
}

static const char * sparse_graph_probe_role(const char * name) {
    if (!name) {
        return "other";
    }
    if (std::strstr(name, "ffn_gate_exps") != nullptr) {
        return "gate";
    }
    if (std::strstr(name, "ffn_up_exps") != nullptr) {
        return "up";
    }
    if (std::strstr(name, "ffn_down_exps") != nullptr) {
        return "down";
    }
    return "other";
}

static FILE * sparse_graph_probe_fp_locked() {
    if (!g_sparse_graph_probe.initialized) {
        g_sparse_graph_probe.initialized = true;
        if (sparse_graph_probe_enabled()) {
            const char * path = std::getenv("DS4_SPARSE_PAIR_GRAPH_PROBE_OUT");
            const char * profile = std::getenv("DS4_SPARSE_PAIR_PROFILE_JSON");
            if (path && path[0]) {
                g_sparse_graph_probe.fp = std::fopen(path, "w");
                if (g_sparse_graph_probe.fp) {
                    std::setvbuf(g_sparse_graph_probe.fp, nullptr, _IOLBF, 0);
                    std::fprintf(g_sparse_graph_probe.fp,
                        "seq,tensor_role,tensor,expert,cne1,src0_bytes,src1_f32_bytes,dst_bytes,"
                        "src0_from_cache,src0_cache_inserted,slot,deferred_sync,src1_h2d_bytes,dst_d2h_bytes,"
                        "scatter_to_cpu,retains_gpu_output,returns_gpu_handle,zero_transfer_ready,"
                        "kernel_src0_ptr,d_src1_f32_ptr,d_dst_ptr,h_scratch_ptr,dst_host_ptr\n");
                    std::fprintf(stderr,
                        "[moe_stream] sparse graph probe enabled out=%s profile=%s\n",
                        path, (profile && profile[0]) ? profile : "");
                } else {
                    std::fprintf(stderr, "[moe_stream] failed to open sparse graph probe: %s\n", path);
                }
            }
        }
    }
    return g_sparse_graph_probe.fp;
}

static void sparse_graph_probe_write(
    const char * src0_name,
    int64_t expert_index,
    int64_t cne1,
    size_t src0_bytes,
    size_t src1_f32_bytes,
    size_t dst_bytes,
    bool cache_hit,
    bool cache_inserted,
    int slot,
    bool deferred_sync,
    const void * kernel_src0,
    const void * d_src1_f32,
    const void * d_dst,
    const void * h_scratch,
    const void * dst_host) {
    if (!sparse_graph_probe_enabled()) {
        return;
    }

    std::lock_guard<std::mutex> lk(g_sparse_graph_probe.mu);
    FILE * fp = sparse_graph_probe_fp_locked();
    if (!fp) {
        return;
    }

    const bool src1_h2d = src1_f32_bytes > 0;
    const bool dst_d2h = dst_bytes > 0;
    const bool scatter_to_cpu = true;
    const bool retains_gpu_output = false;
    const bool returns_gpu_handle = false;
    const bool zero_transfer_ready =
        !src1_h2d && !dst_d2h && !scatter_to_cpu && retains_gpu_output && returns_gpu_handle;
    const uint64_t seq = g_sparse_graph_probe.seq.fetch_add(1, std::memory_order_relaxed);

    std::fprintf(fp,
        "%" PRIu64 ",%s,%s,%" PRId64 ",%" PRId64 ",%zu,%zu,%zu,%d,%d,%d,%d,%zu,%zu,%d,%d,%d,%d,%p,%p,%p,%p,%p\n",
        seq,
        sparse_graph_probe_role(src0_name),
        src0_name ? src0_name : "",
        expert_index,
        cne1,
        src0_bytes,
        src1_f32_bytes,
        dst_bytes,
        cache_hit ? 1 : 0,
        cache_inserted ? 1 : 0,
        slot,
        deferred_sync ? 1 : 0,
        src1_h2d ? src1_f32_bytes : (size_t) 0,
        dst_d2h ? dst_bytes : (size_t) 0,
        scatter_to_cpu ? 1 : 0,
        retains_gpu_output ? 1 : 0,
        returns_gpu_handle ? 1 : 0,
        zero_transfer_ready ? 1 : 0,
        kernel_src0,
        d_src1_f32,
        d_dst,
        h_scratch,
        dst_host);
}

static bool moe_stream_one_experimental_ds4_enabled() {
    static int enabled = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4");
        return env && env[0] && env[0] != '0';
    }();
    return enabled != 0;
}

static bool moe_stream_one_ds4_nonids_enabled() {
    static int enabled = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_ONE_DS4_NONIDS");
        return env && env[0] && env[0] != '0';
    }();
    return enabled != 0;
}

static bool moe_stream_name_filter_matches_any(const char * filter, const char * name) {
    if (!filter || !filter[0] || !name) {
        return false;
    }

    const char * p = filter;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == ',' || *p == ':') {
            ++p;
        }
        const char * start = p;
        while (*p && *p != ',' && *p != ':') {
            ++p;
        }
        const char * end = p;
        while (end > start && (end[-1] == ' ' || end[-1] == '\t')) {
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

static bool moe_stream_one_name_filter_allows(const char * name) {

    static const char * filter = std::getenv("GGML_MOE_STREAM_ONE_NAME_FILTER");
    if (!filter || !filter[0]) {
        return true;
    }
    return moe_stream_name_filter_matches_any(filter, name);
}

static const int8_t moe_stream_mxfp4_values[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

static int moe_stream_env_int(const char * name, int fallback) {
    const char * env = std::getenv(name);
    return env && env[0] ? std::atoi(env) : fallback;
}

static bool moe_stream_q80_cpu_compat_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_MOE_STREAM_Q80_CPU_COMPAT");
        enabled = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return enabled != 0;
}

static FILE * moe_stream_q8_debug_fp() {
    static FILE * fp = nullptr;
    static int initialized = 0;
    static std::mutex mu;

    std::lock_guard<std::mutex> lk(mu);
    if (!initialized) {
        initialized = 1;
        const char * path = std::getenv("GGML_MOE_STREAM_Q8_DEBUG_OUT");
        if (path && path[0]) {
            fp = std::fopen(path, "w");
            if (fp) {
                std::setvbuf(fp, nullptr, _IOLBF, 0);
                std::fprintf(fp,
                    "tensor,expert,row_id,i12,out_col,block,gpu_q81_part,gpu_q81_total,gpu_value,d_q,e,sumi_q81\n");
            } else {
                std::fprintf(stderr, "[moe_stream_q8_debug] failed to open trace: %s\n", path);
            }
        }
    }
    return fp;
}

static void moe_stream_q8_debug_maybe(
    const char * src0_name,
    int64_t expert_index,
    const void * src0_data,
    int64_t ne00,
    size_t nb01,
    const void * d_src1_q8,
    const void * d_dst,
    int64_t src1_padded,
    int64_t ne01,
    int64_t cne1,
    const ggml_moe_row_mapping * rows,
    cudaStream_t st) {
    FILE * fp = moe_stream_q8_debug_fp();
    if (!fp || !src0_data || !d_src1_q8 || !d_dst || !rows) {
        return;
    }

    const int target_expert = moe_stream_env_int("GGML_MOE_STREAM_Q8_DEBUG_EXPERT", -1);
    const int target_row    = moe_stream_env_int("GGML_MOE_STREAM_Q8_DEBUG_ROW", -1);
    const int target_i12    = moe_stream_env_int("GGML_MOE_STREAM_Q8_DEBUG_I12", -1);
    const int target_col    = moe_stream_env_int("GGML_MOE_STREAM_Q8_DEBUG_COL", -1);
    if (target_expert >= 0 && expert_index != target_expert) {
        return;
    }
    if (target_col < 0 || target_col >= ne01) {
        return;
    }

    int64_t target_k = -1;
    for (int64_t k = 0; k < cne1; ++k) {
        if ((target_row < 0 || rows[k].i1 == target_row) && (target_i12 < 0 || rows[k].i2 == target_i12)) {
            target_k = k;
            break;
        }
    }
    if (target_k < 0) {
        return;
    }

    const size_t src1_q8_row_bytes = (size_t)(src1_padded / QK8_1) * sizeof(block_q8_1);
    std::vector<uint8_t> q8_row(src1_q8_row_bytes);
    float gpu_value = 0.0f;
    const char * d_q8_row = (const char *) d_src1_q8 + (size_t) target_k * src1_q8_row_bytes;
    const float * d_dst_value = (const float *) d_dst + (size_t) target_k * ne01 + (size_t) target_col;

    if (cudaMemcpyAsync(q8_row.data(), d_q8_row, src1_q8_row_bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess) {
        return;
    }
    if (cudaMemcpyAsync(&gpu_value, d_dst_value, sizeof(float), cudaMemcpyDeviceToHost, st) != cudaSuccess) {
        return;
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) {
        return;
    }

    const block_q8_1 * y = (const block_q8_1 *) q8_row.data();
    const block_mxfp4 * x = (const block_mxfp4 *) ((const char *) src0_data + (size_t) target_col * nb01);
    const int64_t nb = ne00 / QK_MXFP4;
    double total = 0.0;

    flockfile(fp);
    for (int64_t ib = 0; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            const int v0 = moe_stream_mxfp4_values[x[ib].qs[j] & 0x0F];
            const int v1 = moe_stream_mxfp4_values[x[ib].qs[j] >> 4];
            sumi += y[ib].qs[j] * v0;
            sumi += y[ib].qs[j + QK_MXFP4/2] * v1;
        }
        const float d_q = __low2float(y[ib].ds);
        const double part = (double) GGML_E8M0_TO_FP32_HALF(x[ib].e) * (double) d_q * (double) sumi;
        total += part;
        std::fprintf(fp,
            "%s,%" PRId64 ",%d,%d,%d,%" PRId64 ",%.9g,%.9g,%.9g,%.9g,%d,%d\n",
            src0_name ? src0_name : "",
            expert_index,
            rows[target_k].i1,
            rows[target_k].i2,
            target_col,
            ib,
            part,
            total,
            (double) gpu_value,
            (double) d_q,
            (int) x[ib].e,
            sumi);
    }
    funlockfile(fp);
}

static FILE * moe_stream_q80_probe_fp() {
    static FILE * fp = nullptr;
    static int initialized = 0;
    static std::mutex mu;

    std::lock_guard<std::mutex> lk(mu);
    if (!initialized) {
        initialized = 1;
        const char * path = std::getenv("GGML_MOE_STREAM_Q80_PROBE_OUT");
        if (path && path[0]) {
            fp = std::fopen(path, "w");
            if (fp) {
                std::setvbuf(fp, nullptr, _IOLBF, 0);
                std::fprintf(fp,
                    "record,tensor,expert,k,row_id,i12,out_col,gpu_q80,cpu_dst,abs_diff\n");
            } else {
                std::fprintf(stderr, "[moe_stream_q80_probe] failed to open trace: %s\n", path);
            }
        }
    }
    return fp;
}

static std::atomic<uint64_t> g_q80_probe_records{0};

static bool moe_stream_q80_probe_name_allows(const char * name) {
    static const char * filter = std::getenv("GGML_MOE_STREAM_Q80_PROBE_NAME_FILTER");
    if (!filter || !filter[0]) {
        return true;
    }
    return name && std::strstr(name, filter) != nullptr;
}

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

static __global__ void moe_stream_q80_probe_kernel(
        const char * __restrict__ src0,
        size_t nb01,
        const char * __restrict__ q80,
        size_t q80_row_size,
        int64_t ne00,
        int rows_to_probe,
        int cols_to_probe,
        float * __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows_to_probe * cols_to_probe;
    if (idx >= total) {
        return;
    }

    const int k = idx / cols_to_probe;
    const int col = idx - k * cols_to_probe;
    const block_mxfp4 * x = (const block_mxfp4 *) (src0 + (size_t) col * nb01);
    const block_q8_0 * y = (const block_q8_0 *) (q80 + (size_t) k * q80_row_size);
    const int64_t nb = ne00 / QK_MXFP4;

    float sum = 0.0f;
    for (int64_t ib = 0; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            const uint8_t q = x[ib].qs[j];
            sumi += y[ib].qs[j] * moe_stream_mxfp4_value_dev(q & 0x0F);
            sumi += y[ib].qs[j + QK_MXFP4/2] * moe_stream_mxfp4_value_dev(q >> 4);
        }
        const float d = __half2float(y[ib].d) * ggml_cuda_e8m0_to_fp32(x[ib].e) * 0.5f;
        sum += d * (float) sumi;
    }
    out[idx] = sum;
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

static __device__ __forceinline__ void moe_stream_q80_cpu_compat_block_lanes(
        const block_mxfp4 & x,
        const block_q8_0 & y,
        int lanes[8]) {
#pragma unroll
    for (int group = 0; group < 4; ++group) {
        int lo = 0;
        int hi = 0;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int j = group * 4 + r;
            const uint8_t q = x.qs[j];
            lo += y.qs[j] * moe_stream_mxfp4_value_dev(q & 0x0F);
            hi += y.qs[j + QK_MXFP4/2] * moe_stream_mxfp4_value_dev(q >> 4);
        }
        lanes[group] = lo;
        lanes[group + 4] = hi;
    }
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

static __global__ void moe_stream_q80_probe_cpu_compat_kernel(
        const char * __restrict__ src0,
        size_t nb01,
        const char * __restrict__ q80,
        size_t q80_row_size,
        int64_t ne00,
        int rows_to_probe,
        int cols_to_probe,
        float * __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows_to_probe * cols_to_probe;
    if (idx >= total) {
        return;
    }

    const int k = idx / cols_to_probe;
    const int col = idx - k * cols_to_probe;
    const block_mxfp4 * x = (const block_mxfp4 *) (src0 + (size_t) col * nb01);
    const block_q8_0 * y = (const block_q8_0 *) (q80 + (size_t) k * q80_row_size);
    const int64_t nb = ne00 / QK_MXFP4;

    float acc1[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float acc2[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    int64_t ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        int p1[8];
        int p2[8];
        moe_stream_q80_cpu_compat_block_lanes(x[ib + 0], y[ib + 0], p1);
        moe_stream_q80_cpu_compat_block_lanes(x[ib + 1], y[ib + 1], p2);

        const float scale0 = __half2float(y[ib + 0].d) * (ggml_cuda_e8m0_to_fp32(x[ib + 0].e) * 0.5f);
        const float scale1 = __half2float(y[ib + 1].d) * (ggml_cuda_e8m0_to_fp32(x[ib + 1].e) * 0.5f);
#pragma unroll
        for (int lane = 0; lane < 8; ++lane) {
            acc1[lane] = fmaf(scale0, (float) p1[lane], acc1[lane]);
            acc2[lane] = fmaf(scale1, (float) p2[lane], acc2[lane]);
        }
    }

    float sum = moe_stream_q80_cpu_compat_hsum8(acc1, acc2);
    for (; ib < nb; ++ib) {
        int sumi1 = 0;
        int sumi2 = 0;
#pragma unroll
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            const uint8_t q = x[ib].qs[j];
            sumi1 += y[ib].qs[j] * moe_stream_mxfp4_value_dev(q & 0x0F);
            sumi2 += y[ib].qs[j + QK_MXFP4/2] * moe_stream_mxfp4_value_dev(q >> 4);
        }
        const float scale = __half2float(y[ib].d) * (ggml_cuda_e8m0_to_fp32(x[ib].e) * 0.5f);
        sum = fmaf(scale, (float) (sumi1 + sumi2), sum);
    }
    out[idx] = sum;
}

static __global__ void moe_stream_q80_hot_batch_cpu_compat_kernel(
        const char * const * __restrict__ src0_rows,
        size_t nb01,
        const char * __restrict__ q80,
        size_t q80_row_size,
        int64_t ne00,
        int rows_to_probe,
        int cols_to_probe,
        float * __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows_to_probe * cols_to_probe;
    if (idx >= total) {
        return;
    }

    const int k = idx / cols_to_probe;
    const int col = idx - k * cols_to_probe;
    const char * src0 = src0_rows[k];
    const block_mxfp4 * x = (const block_mxfp4 *) (src0 + (size_t) col * nb01);
    const block_q8_0 * y = (const block_q8_0 *) (q80 + (size_t) k * q80_row_size);
    const int64_t nb = ne00 / QK_MXFP4;

    float acc1[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float acc2[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    int64_t ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        int p1[8];
        int p2[8];
        moe_stream_q80_cpu_compat_block_lanes(x[ib + 0], y[ib + 0], p1);
        moe_stream_q80_cpu_compat_block_lanes(x[ib + 1], y[ib + 1], p2);

        const float scale0 = __half2float(y[ib + 0].d) * (ggml_cuda_e8m0_to_fp32(x[ib + 0].e) * 0.5f);
        const float scale1 = __half2float(y[ib + 1].d) * (ggml_cuda_e8m0_to_fp32(x[ib + 1].e) * 0.5f);
#pragma unroll
        for (int lane = 0; lane < 8; ++lane) {
            acc1[lane] = fmaf(scale0, (float) p1[lane], acc1[lane]);
            acc2[lane] = fmaf(scale1, (float) p2[lane], acc2[lane]);
        }
    }

    float sum = moe_stream_q80_cpu_compat_hsum8(acc1, acc2);
    for (; ib < nb; ++ib) {
        int sumi1 = 0;
        int sumi2 = 0;
#pragma unroll
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            const uint8_t q = x[ib].qs[j];
            sumi1 += y[ib].qs[j] * moe_stream_mxfp4_value_dev(q & 0x0F);
            sumi2 += y[ib].qs[j + QK_MXFP4/2] * moe_stream_mxfp4_value_dev(q >> 4);
        }
        const float scale = __half2float(y[ib].d) * (ggml_cuda_e8m0_to_fp32(x[ib].e) * 0.5f);
        sum = fmaf(scale, (float) (sumi1 + sumi2), sum);
    }
    out[idx] = sum;
}

static __global__ void moe_stream_q80_hot_batch_cpu_compat_warp2_kernel(
        const char * const * __restrict__ src0_rows,
        size_t nb01,
        const char * __restrict__ q80,
        size_t q80_row_size,
        int64_t ne00,
        int rows_to_probe,
        int cols_to_probe,
        float * __restrict__ out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int half = lane >> 4;
    const int sublane = lane & 15;
    const int out_idx = warp * 2 + half;
    const int total = rows_to_probe * cols_to_probe;
    if (out_idx >= total) {
        return;
    }

    const int k = out_idx / cols_to_probe;
    const int col = out_idx - k * cols_to_probe;
    const int base_lane = half ? 16 : 0;
    const uint32_t half_mask = half ? 0xffff0000u : 0x0000ffffu;
    const char * src0 = src0_rows[k];
    const block_mxfp4 * x = (const block_mxfp4 *) (src0 + (size_t) col * nb01);
    const block_q8_0 * y = (const block_q8_0 *) (q80 + (size_t) k * q80_row_size);
    const int64_t nb = ne00 / QK_MXFP4;

    float acc = 0.0f;
    const int lane8 = sublane & 7;
    int64_t ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const bool second = sublane >= 8;
        const block_mxfp4 & xb = x[ib + (second ? 1 : 0)];
        const block_q8_0 & yb = y[ib + (second ? 1 : 0)];
        const int p = moe_stream_q80_cpu_compat_one_lane(xb, yb, lane8);
        const float scale = __half2float(yb.d) * (ggml_cuda_e8m0_to_fp32(xb.e) * 0.5f);
        acc = fmaf(scale, (float) p, acc);
    }

    float acc1[8];
    float acc2[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc1[i] = __shfl_sync(half_mask, acc, base_lane + i);
        acc2[i] = __shfl_sync(half_mask, acc, base_lane + 8 + i);
    }

    if (sublane == 0) {
        float sum = moe_stream_q80_cpu_compat_hsum8(acc1, acc2);
        for (; ib < nb; ++ib) {
            int sumi1 = 0;
            int sumi2 = 0;
#pragma unroll
            for (int j = 0; j < QK_MXFP4/2; ++j) {
                const uint8_t q = x[ib].qs[j];
                sumi1 += y[ib].qs[j] * moe_stream_mxfp4_value_dev(q & 0x0F);
                sumi2 += y[ib].qs[j + QK_MXFP4/2] * moe_stream_mxfp4_value_dev(q >> 4);
            }
            const float scale = __half2float(y[ib].d) * (ggml_cuda_e8m0_to_fp32(x[ib].e) * 0.5f);
            sum = fmaf(scale, (float) (sumi1 + sumi2), sum);
        }
        out[out_idx] = sum;
    }
}

static __global__ void moe_stream_q80_hot_batch_cpu_compat_warp2_transposed_kernel(
        const char * const * __restrict__ src0_rows,
        const char * __restrict__ q80,
        size_t q80_row_size,
        int64_t ne00,
        int rows_to_probe,
        int cols_to_probe,
        float * __restrict__ out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int half = lane >> 4;
    const int sublane = lane & 15;
    const int out_idx = warp * 2 + half;
    const int total = rows_to_probe * cols_to_probe;
    if (out_idx >= total) {
        return;
    }

    const int k = out_idx / cols_to_probe;
    const int col = out_idx - k * cols_to_probe;
    const int base_lane = half ? 16 : 0;
    const uint32_t half_mask = half ? 0xffff0000u : 0x0000ffffu;
    const block_mxfp4 * x = (const block_mxfp4 *) src0_rows[k];
    const block_q8_0 * y = (const block_q8_0 *) (q80 + (size_t) k * q80_row_size);
    const int64_t nb = ne00 / QK_MXFP4;

    float acc = 0.0f;
    const int lane8 = sublane & 7;
    int64_t ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const bool second = sublane >= 8;
        const block_mxfp4 & xb = x[(ib + (second ? 1 : 0)) * (int64_t) cols_to_probe + col];
        const block_q8_0 & yb = y[ib + (second ? 1 : 0)];
        const int p = moe_stream_q80_cpu_compat_one_lane(xb, yb, lane8);
        const float scale = __half2float(yb.d) * (ggml_cuda_e8m0_to_fp32(xb.e) * 0.5f);
        acc = fmaf(scale, (float) p, acc);
    }

    float acc1[8];
    float acc2[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc1[i] = __shfl_sync(half_mask, acc, base_lane + i);
        acc2[i] = __shfl_sync(half_mask, acc, base_lane + 8 + i);
    }

    if (sublane == 0) {
        float sum = moe_stream_q80_cpu_compat_hsum8(acc1, acc2);
        for (; ib < nb; ++ib) {
            int sumi1 = 0;
            int sumi2 = 0;
            const block_mxfp4 & xb = x[ib * (int64_t) cols_to_probe + col];
#pragma unroll
            for (int j = 0; j < QK_MXFP4/2; ++j) {
                const uint8_t q = xb.qs[j];
                sumi1 += y[ib].qs[j] * moe_stream_mxfp4_value_dev(q & 0x0F);
                sumi2 += y[ib].qs[j + QK_MXFP4/2] * moe_stream_mxfp4_value_dev(q >> 4);
            }
            const float scale = __half2float(y[ib].d) * (ggml_cuda_e8m0_to_fp32(xb.e) * 0.5f);
            sum = fmaf(scale, (float) (sumi1 + sumi2), sum);
        }
        out[out_idx] = sum;
    }
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

static __device__ __forceinline__ int moe_stream_q2tern_value_dev(uint8_t code) {
    if (code == 1) {
        return 4;
    }
    if (code == 2) {
        return -4;
    }
    return 0;
}

static __global__ void moe_stream_q80_partial_q2tern_rowrange_kernel(
        const char * const * __restrict__ src0_rows,
        const uint32_t * __restrict__ row_counts,
        int q2_row_size,
        const char * __restrict__ q80,
        size_t q80_row_size,
        int64_t ne00,
        int records,
        int max_cols,
        float * __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = records * max_cols;
    if (idx >= total) {
        return;
    }
    const int row = idx / max_cols;
    const int local_col = idx - row * max_cols;
    if (local_col >= (int) row_counts[row]) {
        return;
    }

    const char * src0 = src0_rows[row] + (size_t) local_col * (size_t) q2_row_size;
    const block_q8_0 * y = (const block_q8_0 *) (q80 + (size_t) row * q80_row_size);
    const int64_t nb = ne00 / QK_MXFP4;

    float sum = 0.0f;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const uint8_t e = (uint8_t) src0[ib * 9];
        const uint8_t * q2 = (const uint8_t *) (src0 + ib * 9 + 1);
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < QK_MXFP4; ++j) {
            const uint8_t code = (q2[j >> 2] >> ((j & 3) * 2)) & 0x03;
            sumi += y[ib].qs[j] * moe_stream_q2tern_value_dev(code);
        }
        const float scale = __half2float(y[ib].d) * (ggml_cuda_e8m0_to_fp32(e) * 0.5f);
        sum = fmaf(scale, (float) sumi, sum);
    }
    out[(size_t) row * (size_t) max_cols + (size_t) local_col] = sum;
}

static __global__ void moe_stream_q80_partial_exact_rowrange_kernel(
        const char * const * __restrict__ src0_rows,
        const uint32_t * __restrict__ row_counts,
        size_t nb01,
        const char * __restrict__ q80,
        size_t q80_row_size,
        int64_t ne00,
        int records,
        int max_cols,
        float * __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = records * max_cols;
    if (idx >= total) {
        return;
    }
    const int row = idx / max_cols;
    const int local_col = idx - row * max_cols;
    if (local_col >= (int) row_counts[row]) {
        return;
    }

    const char * src0 = src0_rows[row];
    const block_mxfp4 * x = (const block_mxfp4 *) (src0 + (size_t) local_col * nb01);
    const block_q8_0 * y = (const block_q8_0 *) (q80 + (size_t) row * q80_row_size);
    const int64_t nb = ne00 / QK_MXFP4;

    float acc1[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float acc2[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    int64_t ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        int p1[8];
        int p2[8];
        moe_stream_q80_cpu_compat_block_lanes(x[ib + 0], y[ib + 0], p1);
        moe_stream_q80_cpu_compat_block_lanes(x[ib + 1], y[ib + 1], p2);

        const float scale0 = __half2float(y[ib + 0].d) * (ggml_cuda_e8m0_to_fp32(x[ib + 0].e) * 0.5f);
        const float scale1 = __half2float(y[ib + 1].d) * (ggml_cuda_e8m0_to_fp32(x[ib + 1].e) * 0.5f);
#pragma unroll
        for (int lane = 0; lane < 8; ++lane) {
            acc1[lane] = fmaf(scale0, (float) p1[lane], acc1[lane]);
            acc2[lane] = fmaf(scale1, (float) p2[lane], acc2[lane]);
        }
    }

    float sum = moe_stream_q80_cpu_compat_hsum8(acc1, acc2);
    for (; ib < nb; ++ib) {
        int sumi1 = 0;
        int sumi2 = 0;
#pragma unroll
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            const uint8_t q = x[ib].qs[j];
            sumi1 += y[ib].qs[j] * moe_stream_mxfp4_value_dev(q & 0x0F);
            sumi2 += y[ib].qs[j + QK_MXFP4/2] * moe_stream_mxfp4_value_dev(q >> 4);
        }
        const float scale = __half2float(y[ib].d) * (ggml_cuda_e8m0_to_fp32(x[ib].e) * 0.5f);
        sum = fmaf(scale, (float) (sumi1 + sumi2), sum);
    }
    out[(size_t) row * (size_t) max_cols + (size_t) local_col] = sum;
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
    if (!moe_stream_q80_probe_name_allows(src0_name)) {
        return;
    }
    FILE * fp = moe_stream_q80_probe_fp();
    if (!fp || src0_type_int != GGML_TYPE_MXFP4 || !src0_data || !src1_q8_0 || !dst || !rows) {
        return;
    }
    if (ne00 <= 0 || ne01 <= 0 || cne1 <= 0 || src1_ne1 <= 0 || src1_q8_0_row_size == 0) {
        return;
    }
    if (ne00 % QK_MXFP4 != 0) {
        return;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        return;
    }

    const int max_rows = std::max(1, moe_stream_env_int("GGML_MOE_STREAM_Q80_PROBE_ROWS", 1));
    const int max_cols = std::max(1, moe_stream_env_int("GGML_MOE_STREAM_Q80_PROBE_COLS", 4));
    const int max_records = std::max(1, moe_stream_env_int("GGML_MOE_STREAM_Q80_PROBE_MAX_RECORDS", 64));
    const int rows_to_probe = (int) std::min<int64_t>(cne1, max_rows);
    const int cols_to_probe = (int) std::min<int64_t>(ne01, max_cols);
    const int total = rows_to_probe * cols_to_probe;
    if (total <= 0) {
        return;
    }

    const uint64_t base_record = g_q80_probe_records.fetch_add((uint64_t) total, std::memory_order_relaxed);
    if (base_record >= (uint64_t) max_records) {
        return;
    }
    int records_to_write = total;
    if (base_record + (uint64_t) total > (uint64_t) max_records) {
        records_to_write = (int) ((uint64_t) max_records - base_record);
    }

    std::vector<uint8_t> h_src0((size_t) cols_to_probe * nb01);
    for (int col = 0; col < cols_to_probe; ++col) {
        std::memcpy(h_src0.data() + (size_t) col * nb01,
                (const char *) src0_data + (size_t) col * nb01,
                nb01);
    }

    std::vector<uint8_t> h_q80((size_t) rows_to_probe * src1_q8_0_row_size);
    for (int k = 0; k < rows_to_probe; ++k) {
        int64_t i11 = rows[k].i1 % src1_ne1;
        if (i11 < 0) {
            i11 += src1_ne1;
        }
        const int64_t i12 = rows[k].i2;
        if (i12 < 0) {
            return;
        }
        const char * q80_row = (const char *) src1_q8_0 + (size_t) (i11 + i12 * src1_ne1) * src1_q8_0_row_size;
        std::memcpy(h_q80.data() + (size_t) k * src1_q8_0_row_size, q80_row, src1_q8_0_row_size);
    }

    void * d_src0 = nullptr;
    void * d_q80 = nullptr;
    float * d_out = nullptr;
    std::vector<float> h_out((size_t) total);
    const size_t src0_bytes = h_src0.size();
    const size_t q80_bytes = h_q80.size();
    const size_t out_bytes = h_out.size() * sizeof(float);

    bool ok = cudaMalloc(&d_src0, src0_bytes) == cudaSuccess &&
        cudaMalloc(&d_q80, q80_bytes) == cudaSuccess &&
        cudaMalloc((void **) &d_out, out_bytes) == cudaSuccess;
    if (ok) {
        ok = cudaMemcpy(d_src0, h_src0.data(), src0_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_q80, h_q80.data(), q80_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    if (ok) {
        const int threads = 128;
        const int blocks = (total + threads - 1) / threads;
        if (moe_stream_q80_cpu_compat_enabled()) {
            moe_stream_q80_probe_cpu_compat_kernel<<<blocks, threads>>>(
                    (const char *) d_src0, nb01, (const char *) d_q80, src1_q8_0_row_size,
                    ne00, rows_to_probe, cols_to_probe, d_out);
        } else {
            moe_stream_q80_probe_kernel<<<blocks, threads>>>(
                    (const char *) d_src0, nb01, (const char *) d_q80, src1_q8_0_row_size,
                    ne00, rows_to_probe, cols_to_probe, d_out);
        }
        ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
            cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    if (ok) {
        flockfile(fp);
        for (int idx = 0; idx < records_to_write; ++idx) {
            const int k = idx / cols_to_probe;
            const int col = idx - k * cols_to_probe;
            const float * dst_row = (const float *) ((const char *) dst + (size_t) rows[k].i1 * dst_nb1 + (size_t) rows[k].i2 * dst_nb2);
            const float cpu = dst_row[col];
            const float gpu = h_out[(size_t) idx];
            const double abs_diff = std::fabs((double) gpu - (double) cpu);
            std::fprintf(fp,
                    "%" PRIu64 ",%s,%" PRId64 ",%d,%d,%d,%d,%.9g,%.9g,%.9g\n",
                    base_record + (uint64_t) idx,
                    src0_name ? src0_name : "",
                    expert_index,
                    k,
                    rows[k].i1,
                    rows[k].i2,
                    col,
                    (double) gpu,
                    (double) cpu,
                    abs_diff);
        }
        funlockfile(fp);
    } else {
        std::fprintf(stderr, "[moe_stream_q80_probe] CUDA probe failed for %s expert=%" PRId64 "\n",
                src0_name ? src0_name : "", expert_index);
    }

    if (d_src0) {
        cudaFree(d_src0);
    }
    if (d_q80) {
        cudaFree(d_q80);
    }
    if (d_out) {
        cudaFree(d_out);
    }
}

static bool moe_stream_q80_allow_down_enabled() {
    static int enabled = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_Q80_ALLOW_DOWN");
        return (env && env[0] && env[0] != '0') ? 1 : 0;
    }();
    return enabled != 0;
}

static bool moe_stream_q80_write_name_allows(const char * name) {
    static const char * filter = std::getenv("GGML_MOE_STREAM_Q80_WRITE_NAME_FILTER");
    if (!filter || !filter[0] || !name || std::strstr(name, filter) == nullptr) {
        return false;
    }
    if (std::strstr(name, "ffn_up_exps") != nullptr) {
        return true;
    }
    return moe_stream_q80_allow_down_enabled() &&
        std::strstr(name, "ffn_down_exps") != nullptr;
}

static FILE * moe_stream_q80_write_report_fp() {
    static FILE * fp = nullptr;
    static int initialized = 0;
    static std::mutex mu;

    std::lock_guard<std::mutex> lk(mu);
    if (!initialized) {
        initialized = 1;
        const char * path = std::getenv("GGML_MOE_STREAM_Q80_WRITE_REPORT");
        if (path && path[0]) {
            fp = std::fopen(path, "w");
            if (fp) {
                std::setvbuf(fp, nullptr, _IOLBF, 0);
                std::fprintf(fp,
                    "record,tensor,expert,cne1,ne01,max_abs,mean_abs\n");
            } else {
                std::fprintf(stderr, "[moe_stream_q80_write] failed to open report: %s\n", path);
            }
        }
    }
    return fp;
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
    if (!moe_stream_q80_write_name_allows(src0_name)) {
        return;
    }
    if (src0_type_int != GGML_TYPE_MXFP4 || !src0_data || !src1_q8_0 || !dst || !rows) {
        return;
    }
    if (ne00 <= 0 || ne01 <= 0 || cne1 <= 0 || src1_ne1 <= 0 || src1_q8_0_row_size == 0) {
        return;
    }
    if (ne00 % QK_MXFP4 != 0) {
        return;
    }

    const int max_cne1 = std::max(1, moe_stream_env_int("GGML_MOE_STREAM_Q80_WRITE_MAX_CNE1", 1));
    if (cne1 > max_cne1) {
        return;
    }

    const int max_calls = moe_stream_env_int("GGML_MOE_STREAM_Q80_WRITE_MAX_CALLS", 0);
    const uint64_t call_idx = g_q80_write_calls.fetch_add(1, std::memory_order_relaxed);
    if (max_calls > 0 && call_idx >= (uint64_t) max_calls) {
        return;
    }

    if (cudaSetDevice(0) != cudaSuccess) {
        return;
    }

    const int64_t total = cne1 * ne01;
    if (total <= 0) {
        return;
    }

    std::vector<uint8_t> h_src0((size_t) ne01 * nb01);
    std::memcpy(h_src0.data(), src0_data, h_src0.size());

    std::vector<uint8_t> h_q80((size_t) cne1 * src1_q8_0_row_size);
    for (int64_t k = 0; k < cne1; ++k) {
        int64_t i11 = rows[k].i1 % src1_ne1;
        if (i11 < 0) {
            i11 += src1_ne1;
        }
        const int64_t i12 = rows[k].i2;
        if (i12 < 0) {
            return;
        }
        const char * q80_row = (const char *) src1_q8_0 + (size_t) (i11 + i12 * src1_ne1) * src1_q8_0_row_size;
        std::memcpy(h_q80.data() + (size_t) k * src1_q8_0_row_size, q80_row, src1_q8_0_row_size);
    }

    void * d_src0 = nullptr;
    void * d_q80 = nullptr;
    float * d_out = nullptr;
    std::vector<float> h_out((size_t) total);
    const size_t src0_bytes = h_src0.size();
    const size_t q80_bytes = h_q80.size();
    const size_t out_bytes = h_out.size() * sizeof(float);

    bool ok = cudaMalloc(&d_src0, src0_bytes) == cudaSuccess &&
        cudaMalloc(&d_q80, q80_bytes) == cudaSuccess &&
        cudaMalloc((void **) &d_out, out_bytes) == cudaSuccess;
    if (ok) {
        ok = cudaMemcpy(d_src0, h_src0.data(), src0_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_q80, h_q80.data(), q80_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    if (ok) {
        const int threads = 128;
        const int blocks = ((int) total + threads - 1) / threads;
        if (moe_stream_q80_cpu_compat_enabled()) {
            moe_stream_q80_probe_cpu_compat_kernel<<<blocks, threads>>>(
                    (const char *) d_src0, nb01, (const char *) d_q80, src1_q8_0_row_size,
                    ne00, (int) cne1, (int) ne01, d_out);
        } else {
            moe_stream_q80_probe_kernel<<<blocks, threads>>>(
                    (const char *) d_src0, nb01, (const char *) d_q80, src1_q8_0_row_size,
                    ne00, (int) cne1, (int) ne01, d_out);
        }
        ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
            cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    if (ok) {
        double sum_abs = 0.0;
        double max_abs = 0.0;
        for (int64_t k = 0; k < cne1; ++k) {
            float * dst_row = (float *) ((char *) dst + (size_t) rows[k].i1 * dst_nb1 + (size_t) rows[k].i2 * dst_nb2);
            const float * out_row = h_out.data() + (size_t) k * ne01;
            for (int64_t col = 0; col < ne01; ++col) {
                const double diff = std::fabs((double) out_row[col] - (double) dst_row[col]);
                sum_abs += diff;
                max_abs = std::max(max_abs, diff);
            }
            std::memcpy(dst_row, out_row, (size_t) ne01 * sizeof(float));
        }

        if (FILE * fp = moe_stream_q80_write_report_fp()) {
            const uint64_t report_idx = g_q80_write_reports.fetch_add(1, std::memory_order_relaxed);
            flockfile(fp);
            std::fprintf(fp,
                    "%" PRIu64 ",%s,%" PRId64 ",%" PRId64 ",%" PRId64 ",%.9g,%.9g\n",
                    report_idx,
                    src0_name ? src0_name : "",
                    expert_index,
                    cne1,
                    ne01,
                    max_abs,
                    total > 0 ? sum_abs / (double) total : 0.0);
            funlockfile(fp);
        }
    } else {
        std::fprintf(stderr, "[moe_stream_q80_write] CUDA overwrite failed for %s expert=%" PRId64 "\n",
                src0_name ? src0_name : "", expert_index);
    }

    if (d_src0) {
        cudaFree(d_src0);
    }
    if (d_q80) {
        cudaFree(d_q80);
    }
    if (d_out) {
        cudaFree(d_out);
    }
}

static bool moe_stream_q80_skip_name_allows(const char * name) {
    static const char * filter = std::getenv("GGML_MOE_STREAM_Q80_SKIP_NAME_FILTER");
    if (!filter || !filter[0] || !name || std::strstr(name, filter) == nullptr) {
        return false;
    }
    if (std::strstr(name, "ffn_up_exps") != nullptr) {
        return true;
    }
    return moe_stream_q80_allow_down_enabled() &&
        std::strstr(name, "ffn_down_exps") != nullptr;
}

static FILE * moe_stream_q80_partial_repr_probe_fp() {
    static FILE * fp = nullptr;
    static int initialized = 0;
    static std::mutex mu;

    std::lock_guard<std::mutex> lk(mu);
    if (!initialized) {
        initialized = 1;
        const char * path = std::getenv("GGML_MOE_STREAM_Q80_PARTIAL_REPR_PROBE_OUT");
        if (path && path[0]) {
            fp = std::fopen(path, "w");
            if (fp) {
                std::setvbuf(fp, nullptr, _IOLBF, 0);
                std::fprintf(fp,
                    "seq,tensor,records,matched_experts,rows_total,max_cols,src0_bytes,q80_bytes,out_bytes,"
                    "compare_ran,compare_ok,max_abs,mean_abs,diff_count,alloc_us,h2d_us,kernel_us,d2h_us,compare_us,free_us,elapsed_us\n");
            } else {
                std::fprintf(stderr, "[moe_stream_q80_partial_repr_probe] failed to open report: %s\n", path);
            }
        }
    }
    return fp;
}

static FILE * moe_stream_q80_hot_batch_probe_fp() {
    static FILE * fp = nullptr;
    static int initialized = 0;
    static std::mutex mu;

    std::lock_guard<std::mutex> lk(mu);
    if (!initialized) {
        initialized = 1;
        const char * path = std::getenv("GGML_MOE_STREAM_Q80_HOT_BATCH_PROBE_OUT");
        if (path && path[0]) {
            fp = std::fopen(path, "w");
            if (fp) {
                std::setvbuf(fp, nullptr, _IOLBF, 0);
                std::fprintf(fp,
                    "seq,tensor,n_as,ne01,ne00,rows_total,ready_rows,not_in_manifest_rows,not_ready_rows,"
                    "ready_experts,src0_bytes_projected,q80_bytes_projected,out_bytes_projected,"
                    "repr_experts,repr_rows,repr_exact_rows,repr_partial_rows,repr_compressed_rows,"
                    "repr_src0_bytes_projected,repr_resident_original_bytes,repr_resident_compressed_bytes,"
                    "pool_ready,async_started,async_done,compare_enabled,compare_ran,compare_ok,"
                    "max_abs,mean_abs,diff_count,alloc_us,h2d_us,kernel_us,d2h_us,compare_us,free_us,elapsed_us\n");
            } else {
                std::fprintf(stderr, "[moe_stream_q80_hot_batch_probe] failed to open report: %s\n", path);
            }
        }
    }
    return fp;
}

static std::atomic<uint64_t> g_q80_hot_batch_probe_seq{0};

static bool moe_stream_q80_hot_batch_probe_compare_enabled() {
    static int enabled = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_Q80_HOT_BATCH_PROBE_COMPARE");
        return (env && env[0] && env[0] != '0') ? 1 : 0;
    }();
    return enabled != 0;
}

static bool moe_stream_q80_hot_batch_probe_warp_enabled() {
    static int enabled = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_Q80_HOT_BATCH_PROBE_WARP");
        return (env && env[0] && env[0] != '0') ? 1 : 0;
    }();
    return enabled != 0;
}

static bool moe_stream_q80_hot_batch_probe_transpose_enabled() {
    static int enabled = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_Q80_HOT_BATCH_PROBE_TRANSPOSE");
        return (env && env[0] && env[0] != '0') ? 1 : 0;
    }();
    return enabled != 0;
}

static bool moe_stream_q80_hot_batch_probe_row_tile_enabled() {
    static int enabled = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_Q80_HOT_BATCH_PROBE_ROW_TILE");
        return (env && env[0] && env[0] != '0') ? 1 : 0;
    }();
    return enabled != 0;
}

static bool moe_stream_mmvq_hot_batch_probe_enabled() {
    static int enabled = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_MMVQ_HOT_BATCH_PROBE");
        return (env && env[0] && env[0] != '0') ? 1 : 0;
    }();
    return enabled != 0;
}

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
    if (src0_type_int != GGML_TYPE_MXFP4 || !src0_name || !src1_q8_0 || !matrix_row_counts || !matrix_rows || !dst) {
        return;
    }
    if (n_as <= 0 || ne01 <= 0 || ne00 <= 0 || nb01 == 0 || src1_q8_0_row_size == 0 ||
            src1_ne1 <= 0 || rows_per_expert <= 0) {
        return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    int64_t rows_total = 0;
    int64_t ready_rows = 0;
    int64_t not_in_manifest_rows = 0;
    int64_t not_ready_rows = 0;
    int64_t ready_experts = 0;
    uint64_t src0_bytes_projected = 0;
    uint64_t q80_bytes_projected = 0;
    uint64_t out_bytes_projected = 0;
    int64_t repr_experts = 0;
    int64_t repr_rows = 0;
    int64_t repr_exact_rows = 0;
    int64_t repr_partial_rows = 0;
    int64_t repr_compressed_rows = 0;
    uint64_t repr_src0_bytes_projected = 0;
    uint64_t repr_resident_original_bytes = 0;
    uint64_t repr_resident_compressed_bytes = 0;
    bool any_pool_ready = false;
    std::vector<const char *> h_src0_rows;
    std::vector<const float *> h_dst_rows;
    std::vector<uint8_t> h_q80;
    struct mmvq_hot_group {
        const char * src0;
        int64_t start;
        int64_t count;
    };
    const bool mmvq_probe_enabled = moe_stream_mmvq_hot_batch_probe_enabled();
    const bool mmvq_probe_can_gather = mmvq_probe_enabled && src1_f32 && src1_nb1 > 0 && src1_nb2 > 0;
    std::vector<float> h_src1_f32;
    std::vector<mmvq_hot_group> mmvq_groups;
    std::vector<const char *> h_partial_src0_rows;
    std::vector<const float *> h_partial_dst_rows;
    std::vector<uint8_t> h_partial_q80;
    std::vector<uint32_t> h_partial_row0;
    std::vector<uint32_t> h_partial_row_count;
    std::vector<const char *> h_q2_src0_rows;
    std::vector<const float *> h_q2_dst_rows;
    std::vector<uint8_t> h_q2_q80;
    std::vector<uint32_t> h_q2_row0;
    std::vector<uint32_t> h_q2_row_count;
    int64_t partial_matched_experts = 0;
    uint32_t partial_max_cols = 0;
    uint32_t q2_max_cols = 0;
    uint64_t partial_src0_bytes = 0;
    uint64_t partial_q80_bytes = 0;
    uint64_t partial_out_bytes = 0;
    uint64_t q2_src0_bytes = 0;
    uint64_t q2_q80_bytes = 0;
    uint64_t q2_out_bytes = 0;
    const bool partial_probe_enabled = moe_stream_q80_partial_repr_probe_fp() != nullptr;

    for (int64_t expert = 0; expert < n_as; ++expert) {
        const int64_t cne1 = matrix_row_counts[expert];
        if (cne1 <= 0) {
            continue;
        }
        rows_total += cne1;
        one_direct_repr_probe_match repr_match;
        if (one_direct_repr_manifest_match(src0_name, expert, ne01, nb01, cne1, repr_match)) {
            repr_experts++;
            repr_rows += cne1;
            if (repr_match.has_exact) {
                repr_exact_rows += cne1;
            }
            if (repr_match.has_partial) {
                repr_partial_rows += cne1;
            }
            if (repr_match.has_compressed) {
                repr_compressed_rows += cne1;
            }
            repr_src0_bytes_projected += repr_match.src0_bytes_projected;
            repr_resident_original_bytes += repr_match.resident_original_bytes;
            repr_resident_compressed_bytes += repr_match.resident_compressed_bytes;
        }
        bool pool_ready = false;
        const void * d_src0 = one_direct_hot_pool_lookup_dev_ptr(src0_name, expert, &pool_ready);
        any_pool_ready = any_pool_ready || pool_ready;
        if (!pool_ready) {
            not_ready_rows += cne1;
            continue;
        }
        if (!d_src0) {
            not_in_manifest_rows += cne1;
            continue;
        }
        int64_t expert_ready_rows = 0;
        const int64_t expert_start = (int64_t) h_src0_rows.size();
        const ggml_moe_row_mapping * rows = matrix_rows + expert * rows_per_expert;
        std::vector<one_direct_repr_partial_hit> partial_hits;
        if (partial_probe_enabled) {
            one_direct_repr_find_partial_hits(src0_name, expert, partial_hits);
            if (!partial_hits.empty()) {
                partial_matched_experts++;
            }
        }
        for (int64_t k = 0; k < cne1; ++k) {
            int64_t i11 = rows[k].i1 % src1_ne1;
            if (i11 < 0) {
                i11 += src1_ne1;
            }
            const int64_t i12 = rows[k].i2;
            if (i12 < 0) {
                continue;
            }
            const char * q80_row = (const char *) src1_q8_0 + (size_t) (i11 + i12 * src1_ne1) * src1_q8_0_row_size;
            const float * dst_row = (const float *) ((const char *) dst + (size_t) rows[k].i1 * dst_nb1 + (size_t) rows[k].i2 * dst_nb2);
            h_src0_rows.push_back((const char *) d_src0);
            h_dst_rows.push_back(dst_row);
            const size_t old = h_q80.size();
            h_q80.resize(old + src1_q8_0_row_size);
            std::memcpy(h_q80.data() + old, q80_row, src1_q8_0_row_size);
            for (const one_direct_repr_partial_hit & hit : partial_hits) {
                if (!hit.entry || !hit.d_src0 || hit.entry->row_count == 0 ||
                        hit.entry->row0 + hit.entry->row_count > (uint64_t) ne01) {
                    continue;
                }
                if (hit.q2tern) {
                    h_q2_src0_rows.push_back(hit.d_src0);
                    h_q2_dst_rows.push_back(dst_row);
                    h_q2_row0.push_back((uint32_t) hit.entry->row0);
                    h_q2_row_count.push_back((uint32_t) hit.entry->row_count);
                    q2_max_cols = std::max<uint32_t>(q2_max_cols, (uint32_t) hit.entry->row_count);
                    const size_t q2_old = h_q2_q80.size();
                    h_q2_q80.resize(q2_old + src1_q8_0_row_size);
                    std::memcpy(h_q2_q80.data() + q2_old, q80_row, src1_q8_0_row_size);
                    q2_src0_bytes += hit.entry->compressed_nbytes;
                    q2_q80_bytes += src1_q8_0_row_size;
                    q2_out_bytes += hit.entry->row_count * sizeof(float);
                } else {
                    h_partial_src0_rows.push_back(hit.d_src0);
                    h_partial_dst_rows.push_back(dst_row);
                    h_partial_row0.push_back((uint32_t) hit.entry->row0);
                    h_partial_row_count.push_back((uint32_t) hit.entry->row_count);
                    partial_max_cols = std::max<uint32_t>(partial_max_cols, (uint32_t) hit.entry->row_count);
                    const size_t partial_old = h_partial_q80.size();
                    h_partial_q80.resize(partial_old + src1_q8_0_row_size);
                    std::memcpy(h_partial_q80.data() + partial_old, q80_row, src1_q8_0_row_size);
                    partial_src0_bytes += hit.entry->compressed_nbytes;
                    partial_q80_bytes += src1_q8_0_row_size;
                    partial_out_bytes += hit.entry->row_count * sizeof(float);
                }
            }
            if (mmvq_probe_can_gather) {
                const char * f32_row = (const char *) src1_f32 + (size_t) i11 * src1_nb1 + (size_t) i12 * src1_nb2;
                const size_t old_f32 = h_src1_f32.size();
                h_src1_f32.resize(old_f32 + (size_t) ne00);
                std::memcpy(h_src1_f32.data() + old_f32, f32_row, (size_t) ne00 * sizeof(float));
            }
            expert_ready_rows++;
        }
        if (expert_ready_rows > 0) {
            if (mmvq_probe_can_gather) {
                mmvq_groups.push_back({(const char *) d_src0, expert_start, expert_ready_rows});
            }
            ready_experts++;
            ready_rows += expert_ready_rows;
            src0_bytes_projected += (uint64_t) expert_ready_rows * (uint64_t) ne01 * (uint64_t) nb01;
            q80_bytes_projected += (uint64_t) expert_ready_rows * (uint64_t) src1_q8_0_row_size;
            out_bytes_projected += (uint64_t) expert_ready_rows * (uint64_t) ne01 * (uint64_t) sizeof(float);
        }
    }

    if (rows_total == 0) {
        return;
    }
    const uint64_t seq = g_q80_hot_batch_probe_seq.fetch_add(1, std::memory_order_relaxed);
    const bool compare_enabled = moe_stream_q80_hot_batch_probe_compare_enabled();
    const int compare_max_records = moe_stream_env_int("GGML_MOE_STREAM_Q80_HOT_BATCH_PROBE_COMPARE_MAX_RECORDS", 0);
    const bool compare_allowed_by_limit = compare_max_records <= 0 || seq < (uint64_t) compare_max_records;

    bool compare_ran = false;
    bool compare_ok = false;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    uint64_t diff_count = 0;
    uint64_t alloc_us = 0;
    uint64_t h2d_us = 0;
    uint64_t kernel_us = 0;
    uint64_t d2h_us = 0;
    uint64_t compare_us = 0;
    uint64_t free_us = 0;

    if (compare_enabled && compare_allowed_by_limit && ready_rows > 0 && ready_rows <= INT32_MAX && ne01 <= INT32_MAX) {
        const auto t_alloc0 = std::chrono::steady_clock::now();
        const bool use_mmvq_probe = mmvq_probe_enabled && mmvq_probe_can_gather &&
            !mmvq_groups.empty() && h_src1_f32.size() == (size_t) ready_rows * (size_t) ne00;
        const char ** d_src0_rows = nullptr;
        void * d_q80 = nullptr;
        float * d_src1_f32 = nullptr;
        void * d_src1_q8 = nullptr;
        int32_t * d_ids = nullptr;
        float * d_out = nullptr;
        const size_t ptr_bytes = (size_t) ready_rows * sizeof(const char *);
        const size_t q80_bytes = h_q80.size();
        const int64_t src1_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
        const size_t src1_q8_1_row_size = (size_t) src1_padded * sizeof(block_q8_1) / QK8_1;
        const size_t src1_f32_bytes = (size_t) ready_rows * (size_t) ne00 * sizeof(float);
        const size_t src1_q8_1_bytes = (size_t) ready_rows * src1_q8_1_row_size;
        const size_t ids_bytes = (size_t) ready_rows * sizeof(int32_t);
        const size_t out_elems = (size_t) ready_rows * (size_t) ne01;
        const size_t out_bytes = out_elems * sizeof(float);
        std::vector<float> h_out(out_elems);
        bool ok = cudaSetDevice(0) == cudaSuccess;
        if (ok && use_mmvq_probe) {
            ok = cudaMalloc((void **) &d_src1_f32, src1_f32_bytes) == cudaSuccess &&
                cudaMalloc(&d_src1_q8, src1_q8_1_bytes) == cudaSuccess &&
                cudaMalloc((void **) &d_ids, ids_bytes) == cudaSuccess &&
                cudaMalloc((void **) &d_out, out_bytes) == cudaSuccess;
        } else if (ok) {
            ok = cudaMalloc((void **) &d_src0_rows, ptr_bytes) == cudaSuccess &&
                cudaMalloc(&d_q80, q80_bytes) == cudaSuccess &&
                cudaMalloc((void **) &d_out, out_bytes) == cudaSuccess;
        }
        const auto t_alloc1 = std::chrono::steady_clock::now();
        if (ok && use_mmvq_probe) {
            ok = cudaMemcpy(d_src1_f32, h_src1_f32.data(), src1_f32_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                cudaMemset(d_ids, 0, ids_bytes) == cudaSuccess;
        } else if (ok) {
            ok = cudaMemcpy(d_src0_rows, h_src0_rows.data(), ptr_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                cudaMemcpy(d_q80, h_q80.data(), q80_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        }
        const auto t_h2d = std::chrono::steady_clock::now();
        if (ok) {
            if (use_mmvq_probe) {
                for (const mmvq_hot_group & group : mmvq_groups) {
                    for (int64_t off = 0; off < group.count; off += MMVQ_MAX_BATCH_SIZE) {
                        const int64_t chunk = std::min<int64_t>(MMVQ_MAX_BATCH_SIZE, group.count - off);
                        const int64_t row0 = group.start + off;
                        if (!ggml_cuda_moe_stream_mmvq_rows_dev(
                                    src0_type_int,
                                    group.src0,
                                    ne01, ne00, nb01,
                                    d_src1_f32 + (size_t) row0 * (size_t) ne00,
                                    (char *) d_src1_q8 + (size_t) row0 * src1_q8_1_row_size,
                                    d_ids + row0,
                                    chunk,
                                    d_out + (size_t) row0 * (size_t) ne01,
                                    0)) {
                            ok = false;
                            break;
                        }
                    }
                    if (!ok) {
                        break;
                    }
                }
            } else if (moe_stream_q80_hot_batch_probe_row_tile_enabled()) {
                const int threads = 256;
                const dim3 blocks(
                        (unsigned int) ready_rows,
                        (unsigned int) (((int64_t) ne01 + 15) / 16),
                        1);
                if (moe_stream_q80_hot_batch_probe_transpose_enabled()) {
                    moe_stream_q80_hot_batch_cpu_compat_rowtile_kernel<true><<<blocks, threads>>>(
                            d_src0_rows, nb01, (const char *) d_q80, src1_q8_0_row_size,
                            ne00, (int) ready_rows, (int) ne01, d_out);
                } else {
                    moe_stream_q80_hot_batch_cpu_compat_rowtile_kernel<false><<<blocks, threads>>>(
                            d_src0_rows, nb01, (const char *) d_q80, src1_q8_0_row_size,
                            ne00, (int) ready_rows, (int) ne01, d_out);
                }
            } else if (moe_stream_q80_hot_batch_probe_warp_enabled()) {
                const int threads = 128;
                const size_t warps = (out_elems + 1) / 2;
                const int blocks = (int) ((warps * 32 + (size_t) threads - 1) / (size_t) threads);
                if (moe_stream_q80_hot_batch_probe_transpose_enabled()) {
                    moe_stream_q80_hot_batch_cpu_compat_warp2_transposed_kernel<<<blocks, threads>>>(
                            d_src0_rows, (const char *) d_q80, src1_q8_0_row_size,
                            ne00, (int) ready_rows, (int) ne01, d_out);
                } else {
                    moe_stream_q80_hot_batch_cpu_compat_warp2_kernel<<<blocks, threads>>>(
                            d_src0_rows, nb01, (const char *) d_q80, src1_q8_0_row_size,
                            ne00, (int) ready_rows, (int) ne01, d_out);
                }
            } else {
                const int threads = 128;
                const int blocks = (int) ((out_elems + (size_t) threads - 1) / (size_t) threads);
                moe_stream_q80_hot_batch_cpu_compat_kernel<<<blocks, threads>>>(
                        d_src0_rows, nb01, (const char *) d_q80, src1_q8_0_row_size,
                        ne00, (int) ready_rows, (int) ne01, d_out);
            }
            if (ok) {
                ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
            }
        }
        const auto t_kernel = std::chrono::steady_clock::now();
        if (ok) {
            ok = cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        const auto t_d2h = std::chrono::steady_clock::now();
        if (ok) {
            double sum_abs = 0.0;
            for (int64_t row = 0; row < ready_rows; ++row) {
                const float * dst_row = h_dst_rows[(size_t) row];
                const float * out_row = h_out.data() + (size_t) row * (size_t) ne01;
                for (int64_t col = 0; col < ne01; ++col) {
                    const double diff = std::fabs((double) out_row[col] - (double) dst_row[col]);
                    if (diff != 0.0) {
                        diff_count++;
                    }
                    sum_abs += diff;
                    max_abs = std::max(max_abs, diff);
                }
            }
            mean_abs = out_elems ? sum_abs / (double) out_elems : 0.0;
        }
        const auto t_compare = std::chrono::steady_clock::now();
        if (d_src0_rows) {
            cudaFree(d_src0_rows);
        }
        if (d_q80) {
            cudaFree(d_q80);
        }
        if (d_src1_f32) {
            cudaFree(d_src1_f32);
        }
        if (d_src1_q8) {
            cudaFree(d_src1_q8);
        }
        if (d_ids) {
            cudaFree(d_ids);
        }
        if (d_out) {
            cudaFree(d_out);
        }
        const auto t_free = std::chrono::steady_clock::now();

        compare_ran = true;
        compare_ok = ok;
        alloc_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_alloc1 - t_alloc0).count();
        h2d_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_h2d - t_alloc1).count();
        kernel_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_kernel - t_h2d).count();
        d2h_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_d2h - t_kernel).count();
        compare_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_compare - t_d2h).count();
        free_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_free - t_compare).count();
    }

    bool partial_compare_ran = false;
    bool partial_compare_ok = false;
    double partial_max_abs = 0.0;
    double partial_mean_abs = 0.0;
    uint64_t partial_diff_count = 0;
    uint64_t partial_alloc_us = 0;
    uint64_t partial_h2d_us = 0;
    uint64_t partial_kernel_us = 0;
    uint64_t partial_d2h_us = 0;
    uint64_t partial_compare_us = 0;
    uint64_t partial_free_us = 0;
    const auto t_partial0 = std::chrono::steady_clock::now();

    if (partial_probe_enabled && !h_partial_src0_rows.empty() && partial_max_cols > 0 && partial_max_cols <= (uint32_t) ne01) {
        const auto t_alloc0 = std::chrono::steady_clock::now();
        const int partial_records = (int) h_partial_src0_rows.size();
        const size_t ptr_bytes = (size_t) partial_records * sizeof(const char *);
        const size_t row_meta_bytes = (size_t) partial_records * sizeof(uint32_t);
        const size_t q80_bytes = h_partial_q80.size();
        const size_t out_elems = (size_t) partial_records * (size_t) partial_max_cols;
        const size_t out_bytes = out_elems * sizeof(float);
        const char ** d_src0_rows = nullptr;
        uint32_t * d_row_count = nullptr;
        void * d_q80 = nullptr;
        float * d_out = nullptr;
        std::vector<float> h_out(out_elems, 0.0f);
        bool ok = cudaSetDevice(0) == cudaSuccess &&
            cudaMalloc((void **) &d_src0_rows, ptr_bytes) == cudaSuccess &&
            cudaMalloc((void **) &d_row_count, row_meta_bytes) == cudaSuccess &&
            cudaMalloc(&d_q80, q80_bytes) == cudaSuccess &&
            cudaMalloc((void **) &d_out, out_bytes) == cudaSuccess;
        const auto t_alloc1 = std::chrono::steady_clock::now();
        if (ok) {
            ok = cudaMemcpy(d_src0_rows, h_partial_src0_rows.data(), ptr_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                cudaMemcpy(d_row_count, h_partial_row_count.data(), row_meta_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                cudaMemcpy(d_q80, h_partial_q80.data(), q80_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                cudaMemset(d_out, 0, out_bytes) == cudaSuccess;
        }
        const auto t_h2d = std::chrono::steady_clock::now();
        if (ok) {
            const int threads = 128;
            const int blocks = (int) ((out_elems + (size_t) threads - 1) / (size_t) threads);
            moe_stream_q80_partial_exact_rowrange_kernel<<<blocks, threads>>>(
                    d_src0_rows, d_row_count, nb01,
                    (const char *) d_q80, src1_q8_0_row_size,
                    ne00, partial_records, (int) partial_max_cols, d_out);
            ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
        }
        const auto t_kernel = std::chrono::steady_clock::now();
        if (ok) {
            ok = cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        const auto t_d2h = std::chrono::steady_clock::now();
        if (ok) {
            double sum_abs = 0.0;
            uint64_t compared = 0;
            for (int row = 0; row < partial_records; ++row) {
                const float * dst_row = h_partial_dst_rows[(size_t) row];
                const uint32_t row0 = h_partial_row0[(size_t) row];
                const uint32_t row_count = h_partial_row_count[(size_t) row];
                const float * out_row = h_out.data() + (size_t) row * (size_t) partial_max_cols;
                for (uint32_t col = 0; col < row_count; ++col) {
                    const double diff = std::fabs((double) out_row[col] - (double) dst_row[row0 + col]);
                    if (diff != 0.0) {
                        partial_diff_count++;
                    }
                    sum_abs += diff;
                    partial_max_abs = std::max(partial_max_abs, diff);
                    compared++;
                }
            }
            partial_mean_abs = compared ? sum_abs / (double) compared : 0.0;
        }
        const auto t_compare = std::chrono::steady_clock::now();
        if (d_src0_rows) cudaFree(d_src0_rows);
        if (d_row_count) cudaFree(d_row_count);
        if (d_q80) cudaFree(d_q80);
        if (d_out) cudaFree(d_out);
        const auto t_free = std::chrono::steady_clock::now();
        partial_compare_ran = true;
        partial_compare_ok = ok;
        partial_alloc_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_alloc1 - t_alloc0).count();
        partial_h2d_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_h2d - t_alloc1).count();
        partial_kernel_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_kernel - t_h2d).count();
        partial_d2h_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_d2h - t_kernel).count();
        partial_compare_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_compare - t_d2h).count();
        partial_free_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_free - t_compare).count();
    }
    if (partial_probe_enabled && !h_q2_src0_rows.empty() && q2_max_cols > 0 && q2_max_cols <= (uint32_t) ne01) {
        const auto t_alloc0 = std::chrono::steady_clock::now();
        const int q2_records = (int) h_q2_src0_rows.size();
        const size_t ptr_bytes = (size_t) q2_records * sizeof(const char *);
        const size_t row_meta_bytes = (size_t) q2_records * sizeof(uint32_t);
        const size_t q80_bytes = h_q2_q80.size();
        const size_t out_elems = (size_t) q2_records * (size_t) q2_max_cols;
        const size_t out_bytes = out_elems * sizeof(float);
        const int q2_row_size = (int) ((ne00 / QK_MXFP4) * 9);
        const char ** d_src0_rows = nullptr;
        uint32_t * d_row_count = nullptr;
        void * d_q80 = nullptr;
        float * d_out = nullptr;
        std::vector<float> h_out(out_elems, 0.0f);
        bool ok = cudaSetDevice(0) == cudaSuccess &&
            cudaMalloc((void **) &d_src0_rows, ptr_bytes) == cudaSuccess &&
            cudaMalloc((void **) &d_row_count, row_meta_bytes) == cudaSuccess &&
            cudaMalloc(&d_q80, q80_bytes) == cudaSuccess &&
            cudaMalloc((void **) &d_out, out_bytes) == cudaSuccess;
        const auto t_alloc1 = std::chrono::steady_clock::now();
        if (ok) {
            ok = cudaMemcpy(d_src0_rows, h_q2_src0_rows.data(), ptr_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                cudaMemcpy(d_row_count, h_q2_row_count.data(), row_meta_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                cudaMemcpy(d_q80, h_q2_q80.data(), q80_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                cudaMemset(d_out, 0, out_bytes) == cudaSuccess;
        }
        const auto t_h2d = std::chrono::steady_clock::now();
        if (ok) {
            const int threads = 128;
            const int blocks = (int) ((out_elems + (size_t) threads - 1) / (size_t) threads);
            moe_stream_q80_partial_q2tern_rowrange_kernel<<<blocks, threads>>>(
                    d_src0_rows, d_row_count, q2_row_size,
                    (const char *) d_q80, src1_q8_0_row_size,
                    ne00, q2_records, (int) q2_max_cols, d_out);
            ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
        }
        const auto t_kernel = std::chrono::steady_clock::now();
        if (ok) {
            ok = cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        const auto t_d2h = std::chrono::steady_clock::now();
        double q2_mean_abs = 0.0;
        uint64_t q2_diff_count = 0;
        double q2_max_abs_local = 0.0;
        if (ok) {
            double sum_abs = 0.0;
            uint64_t compared = 0;
            for (int row = 0; row < q2_records; ++row) {
                const float * dst_row = h_q2_dst_rows[(size_t) row];
                const uint32_t row0 = h_q2_row0[(size_t) row];
                const uint32_t row_count = h_q2_row_count[(size_t) row];
                const float * out_row = h_out.data() + (size_t) row * (size_t) q2_max_cols;
                for (uint32_t col = 0; col < row_count; ++col) {
                    const double diff = std::fabs((double) out_row[col] - (double) dst_row[row0 + col]);
                    if (diff != 0.0) {
                        q2_diff_count++;
                    }
                    sum_abs += diff;
                    q2_max_abs_local = std::max(q2_max_abs_local, diff);
                    compared++;
                }
            }
            q2_mean_abs = compared ? sum_abs / (double) compared : 0.0;
        }
        const auto t_compare = std::chrono::steady_clock::now();
        if (d_src0_rows) cudaFree(d_src0_rows);
        if (d_row_count) cudaFree(d_row_count);
        if (d_q80) cudaFree(d_q80);
        if (d_out) cudaFree(d_out);
        const auto t_free = std::chrono::steady_clock::now();

        partial_compare_ok = partial_compare_ran ? (partial_compare_ok && ok) : ok;
        partial_compare_ran = true;
        partial_diff_count += q2_diff_count;
        partial_max_abs = std::max(partial_max_abs, q2_max_abs_local);
        partial_mean_abs = std::max(partial_mean_abs, q2_mean_abs);
        partial_alloc_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_alloc1 - t_alloc0).count();
        partial_h2d_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_h2d - t_alloc1).count();
        partial_kernel_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_kernel - t_h2d).count();
        partial_d2h_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_d2h - t_kernel).count();
        partial_compare_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_compare - t_d2h).count();
        partial_free_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_free - t_compare).count();
        partial_src0_bytes += q2_src0_bytes;
        partial_q80_bytes += q2_q80_bytes;
        partial_out_bytes += q2_out_bytes;
        partial_max_cols = std::max(partial_max_cols, q2_max_cols);
    }
    const size_t partial_total_records = h_partial_src0_rows.size() + h_q2_src0_rows.size();

    const auto t_partial1 = std::chrono::steady_clock::now();
    if (FILE * fp = moe_stream_q80_partial_repr_probe_fp()) {
        flockfile(fp);
        std::fprintf(fp,
                "%" PRIu64 ",%s,%zu,%" PRId64 ",%" PRId64 ",%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%d,%d,%.9g,%.9g,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                seq,
                src0_name,
                partial_total_records,
                partial_matched_experts,
                rows_total,
                partial_max_cols,
                partial_src0_bytes,
                partial_q80_bytes,
                partial_out_bytes,
                partial_compare_ran ? 1 : 0,
                partial_compare_ok ? 1 : 0,
                partial_max_abs,
                partial_mean_abs,
                partial_diff_count,
                partial_alloc_us,
                partial_h2d_us,
                partial_kernel_us,
                partial_d2h_us,
                partial_compare_us,
                partial_free_us,
                (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_partial1 - t_partial0).count());
        funlockfile(fp);
    }

    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t elapsed_us =
        (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    if (FILE * fp = moe_stream_q80_hot_batch_probe_fp()) {
        flockfile(fp);
        std::fprintf(fp,
            "%" PRIu64 ",%s,%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64
            ",%" PRId64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
            ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64
            ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%d,%d,%d,%d,%d,%d,%.9g,%.9g,%" PRIu64
            ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
            seq,
            src0_name,
            n_as,
            ne01,
            ne00,
            rows_total,
            ready_rows,
            not_in_manifest_rows,
            not_ready_rows,
            ready_experts,
            src0_bytes_projected,
            q80_bytes_projected,
            out_bytes_projected,
            repr_experts,
            repr_rows,
            repr_exact_rows,
            repr_partial_rows,
            repr_compressed_rows,
            repr_src0_bytes_projected,
            repr_resident_original_bytes,
            repr_resident_compressed_bytes,
            any_pool_ready ? 1 : 0,
            g_one_direct_hot_pool.async_started.load(std::memory_order_acquire) ? 1 : 0,
            g_one_direct_hot_pool.async_done.load(std::memory_order_acquire) ? 1 : 0,
            compare_enabled ? 1 : 0,
            compare_ran ? 1 : 0,
            compare_ok ? 1 : 0,
            max_abs,
            mean_abs,
            diff_count,
            alloc_us,
            h2d_us,
            kernel_us,
            d2h_us,
            compare_us,
            free_us,
            elapsed_us);
        funlockfile(fp);
    }
    one_direct_repr_report_probe(
            seq,
            src0_name,
            rows_total,
            repr_experts,
            repr_rows,
            repr_exact_rows,
            repr_partial_rows,
            repr_compressed_rows,
            repr_src0_bytes_projected,
            repr_resident_original_bytes,
            repr_resident_compressed_bytes);
}

static FILE * moe_stream_q80_skip_report_fp() {
    static FILE * fp = nullptr;
    static int initialized = 0;
    static std::mutex mu;

    std::lock_guard<std::mutex> lk(mu);
    if (!initialized) {
        initialized = 1;
        const char * path = std::getenv("GGML_MOE_STREAM_Q80_SKIP_REPORT");
        if (path && path[0]) {
            fp = std::fopen(path, "w");
            if (fp) {
                std::setvbuf(fp, nullptr, _IOLBF, 0);
                std::fprintf(fp, "record,tensor,expert,cne1,ne01\n");
            } else {
                std::fprintf(stderr, "[moe_stream_q80_skip] failed to open report: %s\n", path);
            }
        }
    }
    return fp;
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

static uint64_t q80_skip_profile_ns(
        std::chrono::steady_clock::time_point a,
        std::chrono::steady_clock::time_point b) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

static void q80_skip_profile_report_atexit() {
    const uint64_t calls = g_q80_skip_profile.calls_attempted.load();
    if (calls == 0) {
        return;
    }

    const uint64_t total_ns = g_q80_skip_profile.total_ns.load();
    const auto ns_to_ms = [](uint64_t ns) { return (double) ns / 1000000.0; };
    const auto avg_us = [calls](uint64_t ns) { return calls ? (double) ns / (double) calls / 1000.0 : 0.0; };

    char line[2048];
    std::snprintf(line, sizeof(line),
        "[moe_stream_q80_skip_profile] calls_entered=%lu calls_attempted=%lu calls_ok=%lu calls_failed=%lu"
        " src0_bytes=%lu q80_bytes=%lu out_bytes=%lu"
        " total_ms=%.3f host_src0_ms=%.3f host_q80_ms=%.3f cuda_alloc_ms=%.3f"
        " h2d_ms=%.3f kernel_sync_ms=%.3f d2h_ms=%.3f scatter_ms=%.3f report_ms=%.3f cuda_free_ms=%.3f"
        " avg_total_us=%.3f avg_alloc_us=%.3f avg_h2d_us=%.3f avg_kernel_sync_us=%.3f avg_d2h_us=%.3f avg_free_us=%.3f\n",
        g_q80_skip_profile.calls_entered.load(),
        calls,
        g_q80_skip_profile.calls_ok.load(),
        g_q80_skip_profile.calls_failed.load(),
        g_q80_skip_profile.src0_bytes.load(),
        g_q80_skip_profile.q80_bytes.load(),
        g_q80_skip_profile.out_bytes.load(),
        ns_to_ms(total_ns),
        ns_to_ms(g_q80_skip_profile.host_src0_ns.load()),
        ns_to_ms(g_q80_skip_profile.host_q80_ns.load()),
        ns_to_ms(g_q80_skip_profile.cuda_alloc_ns.load()),
        ns_to_ms(g_q80_skip_profile.h2d_ns.load()),
        ns_to_ms(g_q80_skip_profile.kernel_sync_ns.load()),
        ns_to_ms(g_q80_skip_profile.d2h_ns.load()),
        ns_to_ms(g_q80_skip_profile.scatter_ns.load()),
        ns_to_ms(g_q80_skip_profile.report_ns.load()),
        ns_to_ms(g_q80_skip_profile.cuda_free_ns.load()),
        avg_us(total_ns),
        avg_us(g_q80_skip_profile.cuda_alloc_ns.load()),
        avg_us(g_q80_skip_profile.h2d_ns.load()),
        avg_us(g_q80_skip_profile.kernel_sync_ns.load()),
        avg_us(g_q80_skip_profile.d2h_ns.load()),
        avg_us(g_q80_skip_profile.cuda_free_ns.load()));
    std::fputs(line, stderr);

    const char * path = std::getenv("GGML_MOE_STREAM_Q80_SKIP_PROFILE_OUT");
    if (path && path[0]) {
        if (FILE * fp = std::fopen(path, "w")) {
            std::fputs(line, fp);
            std::fclose(fp);
        } else {
            std::fprintf(stderr, "[moe_stream_q80_skip_profile] failed to open profile out: %s\n", path);
        }
    }
}

static bool q80_skip_profile_enabled() {
    static int enabled = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_Q80_SKIP_PROFILE");
        const int value = (env && env[0] && env[0] != '0') ? 1 : 0;
        if (value) {
            std::atexit(q80_skip_profile_report_atexit);
        }
        return value;
    }();
    return enabled != 0;
}

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
    const bool profile = q80_skip_profile_enabled();
    if (profile) {
        g_q80_skip_profile.calls_entered.fetch_add(1, std::memory_order_relaxed);
    }
    if (!moe_stream_q80_skip_name_allows(src0_name)) {
        return false;
    }
    if (src0_type_int != GGML_TYPE_MXFP4 || !src0_data || !src1_q8_0 || !dst || !rows) {
        return false;
    }
    if (ne00 <= 0 || ne01 <= 0 || cne1 <= 0 || src1_ne1 <= 0 || src1_q8_0_row_size == 0) {
        return false;
    }
    if (ne00 % QK_MXFP4 != 0) {
        return false;
    }

    const int max_cne1 = std::max(1, moe_stream_env_int("GGML_MOE_STREAM_Q80_SKIP_MAX_CNE1", 1));
    if (cne1 > max_cne1) {
        return false;
    }

    const int max_calls = moe_stream_env_int("GGML_MOE_STREAM_Q80_SKIP_MAX_CALLS", 0);
    const uint64_t call_idx = g_q80_skip_calls.fetch_add(1, std::memory_order_relaxed);
    if (max_calls > 0 && call_idx >= (uint64_t) max_calls) {
        return false;
    }

    if (cudaSetDevice(0) != cudaSuccess) {
        return false;
    }

    const int64_t total = cne1 * ne01;
    if (total <= 0) {
        return false;
    }

    const auto t0 = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (profile) {
        g_q80_skip_profile.calls_attempted.fetch_add(1, std::memory_order_relaxed);
    }

    std::vector<uint8_t> h_src0((size_t) ne01 * nb01);
    std::memcpy(h_src0.data(), src0_data, h_src0.size());
    const auto t_src0 = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    std::vector<uint8_t> h_q80((size_t) cne1 * src1_q8_0_row_size);
    for (int64_t k = 0; k < cne1; ++k) {
        int64_t i11 = rows[k].i1 % src1_ne1;
        if (i11 < 0) {
            i11 += src1_ne1;
        }
        const int64_t i12 = rows[k].i2;
        if (i12 < 0) {
            return false;
        }
        const char * q80_row = (const char *) src1_q8_0 + (size_t) (i11 + i12 * src1_ne1) * src1_q8_0_row_size;
        std::memcpy(h_q80.data() + (size_t) k * src1_q8_0_row_size, q80_row, src1_q8_0_row_size);
    }
    const auto t_q80 = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    void * d_src0 = nullptr;
    void * d_q80 = nullptr;
    float * d_out = nullptr;
    std::vector<float> h_out((size_t) total);
    const size_t src0_bytes = h_src0.size();
    const size_t q80_bytes = h_q80.size();
    const size_t out_bytes = h_out.size() * sizeof(float);

    bool ok = cudaMalloc(&d_src0, src0_bytes) == cudaSuccess &&
        cudaMalloc(&d_q80, q80_bytes) == cudaSuccess &&
        cudaMalloc((void **) &d_out, out_bytes) == cudaSuccess;
    const auto t_alloc = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (ok) {
        ok = cudaMemcpy(d_src0, h_src0.data(), src0_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_q80, h_q80.data(), q80_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    const auto t_h2d = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (ok) {
        const int threads = 128;
        const int blocks = ((int) total + threads - 1) / threads;
        if (moe_stream_q80_cpu_compat_enabled()) {
            moe_stream_q80_probe_cpu_compat_kernel<<<blocks, threads>>>(
                    (const char *) d_src0, nb01, (const char *) d_q80, src1_q8_0_row_size,
                    ne00, (int) cne1, (int) ne01, d_out);
        } else {
            moe_stream_q80_probe_kernel<<<blocks, threads>>>(
                    (const char *) d_src0, nb01, (const char *) d_q80, src1_q8_0_row_size,
                    ne00, (int) cne1, (int) ne01, d_out);
        }
        ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
    }
    const auto t_kernel = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (ok) {
        ok = cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    const auto t_d2h = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    if (ok) {
        for (int64_t k = 0; k < cne1; ++k) {
            float * dst_row = (float *) ((char *) dst + (size_t) rows[k].i1 * dst_nb1 + (size_t) rows[k].i2 * dst_nb2);
            const float * out_row = h_out.data() + (size_t) k * ne01;
            std::memcpy(dst_row, out_row, (size_t) ne01 * sizeof(float));
        }
    }
    const auto t_scatter = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    if (ok) {
        if (FILE * fp = moe_stream_q80_skip_report_fp()) {
            const uint64_t report_idx = g_q80_skip_reports.fetch_add(1, std::memory_order_relaxed);
            flockfile(fp);
            std::fprintf(fp,
                    "%" PRIu64 ",%s,%" PRId64 ",%" PRId64 ",%" PRId64 "\n",
                    report_idx,
                    src0_name ? src0_name : "",
                    expert_index,
                    cne1,
                    ne01);
            funlockfile(fp);
        }
    } else {
        std::fprintf(stderr, "[moe_stream_q80_skip] CUDA skip failed for %s expert=%" PRId64 "\n",
                src0_name ? src0_name : "", expert_index);
    }
    const auto t_report = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    if (d_src0) {
        cudaFree(d_src0);
    }
    if (d_q80) {
        cudaFree(d_q80);
    }
    if (d_out) {
        cudaFree(d_out);
    }
    const auto t_free = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    if (profile) {
        g_q80_skip_profile.src0_bytes.fetch_add((uint64_t) src0_bytes, std::memory_order_relaxed);
        g_q80_skip_profile.q80_bytes.fetch_add((uint64_t) q80_bytes, std::memory_order_relaxed);
        g_q80_skip_profile.out_bytes.fetch_add((uint64_t) out_bytes, std::memory_order_relaxed);
        g_q80_skip_profile.host_src0_ns.fetch_add(q80_skip_profile_ns(t0, t_src0), std::memory_order_relaxed);
        g_q80_skip_profile.host_q80_ns.fetch_add(q80_skip_profile_ns(t_src0, t_q80), std::memory_order_relaxed);
        g_q80_skip_profile.cuda_alloc_ns.fetch_add(q80_skip_profile_ns(t_q80, t_alloc), std::memory_order_relaxed);
        g_q80_skip_profile.h2d_ns.fetch_add(q80_skip_profile_ns(t_alloc, t_h2d), std::memory_order_relaxed);
        g_q80_skip_profile.kernel_sync_ns.fetch_add(q80_skip_profile_ns(t_h2d, t_kernel), std::memory_order_relaxed);
        g_q80_skip_profile.d2h_ns.fetch_add(q80_skip_profile_ns(t_kernel, t_d2h), std::memory_order_relaxed);
        g_q80_skip_profile.scatter_ns.fetch_add(q80_skip_profile_ns(t_d2h, t_scatter), std::memory_order_relaxed);
        g_q80_skip_profile.report_ns.fetch_add(q80_skip_profile_ns(t_scatter, t_report), std::memory_order_relaxed);
        g_q80_skip_profile.cuda_free_ns.fetch_add(q80_skip_profile_ns(t_report, t_free), std::memory_order_relaxed);
        g_q80_skip_profile.total_ns.fetch_add(q80_skip_profile_ns(t0, t_free), std::memory_order_relaxed);
        if (ok) {
            g_q80_skip_profile.calls_ok.fetch_add(1, std::memory_order_relaxed);
        } else {
            g_q80_skip_profile.calls_failed.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return ok;
}

static bool moe_stream_one_type_allowed(ggml_type type, const char * name) {
    if (type == GGML_TYPE_BF16 || type == GGML_TYPE_F16) {
        const char * flag = std::getenv("GGML_MOE_STREAM_FLOAT");
        return flag && flag[0] && flag[0] != '0' && moe_stream_one_name_filter_allows(name);
    }
    if (type == GGML_TYPE_Q8_0) {
        const char * q8 = std::getenv("GGML_MOE_STREAM_Q8_0");
        return q8 && q8[0] && q8[0] != '0' && moe_stream_one_name_filter_allows(name);
    }
    if (type == GGML_TYPE_IQ3_XXS) {
        return moe_stream_one_name_filter_allows(name);
    }
    if (type == GGML_TYPE_Q4_K) {
        const char * env = std::getenv("GGML_MOE_STREAM_ONE_Q4K");
        return env && env[0] && env[0] != '0' && moe_stream_one_name_filter_allows(name);
    }
    if (!moe_stream_one_experimental_ds4_enabled()) {
        return false;
    }
    if (type != GGML_TYPE_MXFP4 && type != GGML_TYPE_F8_E4M3_B128) {
        return false;
    }
    return moe_stream_one_name_filter_allows(name);
}

static bool ensure_dev(void *&p, size_t &cur, size_t need) {
    if (cur >= need) return true;
    if (p) cudaFree(p);
    p = nullptr; cur = 0;
    if (cudaMalloc(&p, need) != cudaSuccess) return false;
    cur = need;
    return true;
}

static bool ensure_host_pinned(void *&p, size_t &cur, size_t need) {
    if (cur >= need) return true;
    if (p) cudaFreeHost(p);
    p = nullptr; cur = 0;
    if (cudaHostAlloc(&p, need, cudaHostAllocDefault) != cudaSuccess) return false;
    cur = need;
    return true;
}

static void init_once() {
    if (g_inited.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lk(g_init_mu);
    if (g_inited.load(std::memory_order_acquire)) return;

    const char *env = std::getenv("GGML_MOE_STREAM");
    bool enabled = env && env[0] && env[0] != '0';
    const char *defer_env = std::getenv("GGML_MOE_STREAM_DEFER");
    g_defer_sync = defer_env && defer_env[0] && defer_env[0] != '0';
    if (enabled && cudaSetDevice(0) == cudaSuccess) {
        bool ok = true;
        for (int i = 0; i < MOE_STREAM_NSLOTS; ++i) {
            if (cudaStreamCreate(&g_slots[i].stream) != cudaSuccess) { ok = false; break; }
        }
        if (ok) {
            g_avail.store(true, std::memory_order_release);
            std::fprintf(stderr, "[moe_stream] enabled (%d streams, GPU expert compute, defer_sync=%d)\n",
                         MOE_STREAM_NSLOTS, g_defer_sync ? 1 : 0);
        }
    }
    g_inited.store(true, std::memory_order_release);
}

extern "C" bool ggml_cuda_moe_stream_available(void) {
    init_once();
    return g_avail.load(std::memory_order_acquire);
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
static int acquire_slot() {
    for (;;) {
        for (int attempt = 0; attempt < 1024; ++attempt) {
            int s = g_slot_rr.fetch_add(1, std::memory_order_relaxed) & (MOE_STREAM_NSLOTS - 1);
            if (!g_slots[s].in_use.test_and_set(std::memory_order_acquire)) {
                return s;
            }
        }
        if (g_defer_sync && !tls_pending.empty()) {
            flush_deferred_scatter();
            continue;
        }
        std::this_thread::yield();
    }
}

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
    // printf("k");
    init_once();

    if (!g_avail.load(std::memory_order_acquire)) return false;

    static std::atomic<int> first_accounting{0};
    if (first_accounting.fetch_add(1, std::memory_order_relaxed) == 0) {
        vram_accounting_record("first_one_call_entry", src0_name ? src0_name : "");
    }

    const ggml_type t0 = (ggml_type)src0_type_int;
    const bool native_float = t0 == GGML_TYPE_BF16 || t0 == GGML_TYPE_F16;


    if (!moe_stream_one_type_allowed(t0, src0_name)) return false;

    if (cne1 < 1 || cne1 > MMVQ_MAX_BATCH_SIZE) return false;
    if (src1_ne1 <= 0) return false;
    if (!src1_f32) return false;   // require F32 src1 for on-GPU quantization

    static std::atomic<int> first_call{0};
    if (first_call.fetch_add(1) == 0) {
        std::fprintf(stderr, "[moe_stream] first call: ne01=%ld ne00=%ld nb01=%zu src0_bytes=%zu cne1=%ld\n",
                     (long)ne01, (long)ne00, nb01, (size_t)ne01 * nb01, (long)cne1);
        std::fflush(stderr);
    }
    one_direct_manifest_init_once();

    const size_t src0_bytes      = (size_t)ne01 * nb01;
    const size_t src1_f32_bytes  = (size_t)cne1 * ne00 * sizeof(float);
    const int64_t src1_padded    = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    const size_t src1_q8_1_bytes = (size_t)cne1 * (src1_padded / QK8_1) * sizeof(block_q8_1);
    const size_t dst_bytes       = (size_t)ne01 * cne1 * sizeof(float);
    const auto ts0 = std::chrono::steady_clock::now();

    int s = acquire_slot();
    slot_ctx &ctx = g_slots[s];

    // d_src1_f32 = staging for F32 src1 (sized once); d_src1 = Q8_1 quantized.
    bool ok = true;
    {
        std::lock_guard<std::mutex> rk(g_resize_mu);
        ok = ensure_dev(ctx.d_src0, ctx.d_src0_sz, src0_bytes)
          && (native_float || ensure_dev(ctx.d_src1, ctx.d_src1_sz, src1_q8_1_bytes))
          && ensure_dev(ctx.d_dst,  ctx.d_dst_sz,  dst_bytes)
          && ensure_dev(ctx.d_ids,  ctx.d_ids_sz,  (size_t)cne1 * sizeof(int32_t))
          && ensure_host_pinned(ctx.h_scratch, ctx.h_scratch_sz, dst_bytes);
    }
    if (!ok) { release_slot(s); return false; }
    static std::atomic<int> first_workspace_accounting{0};
    if (first_workspace_accounting.fetch_add(1, std::memory_order_relaxed) == 0) {
        vram_accounting_record("first_one_workspace_after", src0_name ? src0_name : "");
    }

    cudaStream_t st = ctx.stream;
    const int profile_role = one_stage_profile_enabled() ? one_stage_profile_role_index(src0_name) : -1;
    const bool profile_cuda = profile_role >= 0 && !g_defer_sync && one_stage_profile_events_init(ctx);
    // =======start time=========
    const int64_t start_us = ggml_time_us();
    one_direct_hot_pool_prefill_maybe(ctx, st);

    // Initialize VRAM cache on first call (needs to know expert size)
    if (!g_vcache_inited) {
        std::lock_guard<std::mutex> lk(g_init_mu);
        if (!g_vcache_inited) vram_cache_init(src0_bytes);
    }
    static std::atomic<int> first_cache_ready_accounting{0};
    if (first_cache_ready_accounting.fetch_add(1, std::memory_order_relaxed) == 0) {
        vram_accounting_record("first_one_cache_ready", src0_name ? src0_name : "");
    }

    one_prefill_maybe(ctx, st, src0_bytes);
    if (profile_cuda && cudaEventRecord(ctx.profile_weight_start, st) != cudaSuccess) {
        cudaGetLastError();
    }
    // 1 us
    // VRAM cache lookup: if this expert is already in VRAM, skip H2D entirely.
    uintptr_t cache_key = one_cache_key_for(src0_name, expert_index, src0_data);
    const void * hot_pool_vram = one_direct_hot_pool_lookup_stream_one_dev_ptr(src0_name, expert_index, nullptr);
    const void * batch_cache_vram = (hot_pool_vram || !gate_batch_prefetch_enabled()) ? nullptr :
        ggml_cuda_moe_stream_cache_dev_ptr(src0_name, src0_bytes, (int)expert_index);
    void *cached_vram = (hot_pool_vram || batch_cache_vram) ? nullptr : vram_cache_lookup(cache_key, st);
    const bool cache_hit = cached_vram != nullptr;
    const bool hot_pool_hit = hot_pool_vram != nullptr;
    const bool batch_cache_hit = batch_cache_vram != nullptr;
    bool cache_inserted = false;
    bool weight_h2d_queued = false;
    const void *kernel_src0 = nullptr;
//1us
    // if(strstr(src0_name,"up"))
    // printf("k:%s ",src0_name);
    // uint64_t start_us = ggml_time_us();
    int gate_profile_pack_hit = 0;
    if (hot_pool_hit) {
        // printf("abcd");
        kernel_src0 = hot_pool_vram;
    } else if (batch_cache_hit) {
        // printf("efgh");
        kernel_src0 = batch_cache_vram;
    } else if (cached_vram) {
        // printf("hit ");
        kernel_src0 = cached_vram;
    } else {
        const bool cache_admit = moe_stream_cache_admit_allows(src0_name, expert_index);
        void *inserted = cache_admit ? vram_cache_promote_from_ram(cache_key, src0_bytes, st, src0_name) : nullptr;
        if (inserted) {
            cache_inserted = true;
            weight_h2d_queued = true;
            // end_us = ggml_time_us();
            kernel_src0 = inserted;
        } else {

            if (!cache_admit && moe_stream_one_require_cache_admit(src0_name)) {
                g_vcache.misses.fetch_add(1, std::memory_order_relaxed);
                release_slot(s);
                return false;
            }

            const void * copy_src = src0_data;
            const char * alias_file = std::getenv("GGML_MOE_EXPERT_GGUF_ALIAS_TSV");
            if (alias_file && alias_file[0]) {
                std::lock_guard<std::mutex> rk(g_resize_mu);
                if (!ensure_host_pinned(ctx.h_src0_pack, ctx.h_src0_pack_sz, src0_bytes)) {
                    release_slot(s); return false;
                }
                size_t read_bytes = 0; int transport = 0;
                if (!ggml_cuda_moe_expert_pack_read_to_host(src0_name, int(expert_index), src0_bytes,
                        ctx.h_src0_pack, src0_bytes, &read_bytes, &transport)) {
                    release_slot(s); return false;
                }
                gate_profile_pack_hit = 1;
                copy_src = ctx.h_src0_pack;
            } else if (const one_expert_pack_entry * pack_entry = one_pack_lookup(src0_name, expert_index, src0_bytes)) {
                gate_profile_pack_hit = 1;
                std::lock_guard<std::mutex> rk(g_resize_mu);
                if (!ensure_host_pinned(ctx.h_src0_pack, ctx.h_src0_pack_sz, src0_bytes)) {
                    release_slot(s);
                    return false;
                }
            //    uint64_t mid_us = ggml_time_us();
                if (one_pack_read_entry(pack_entry, ctx.h_src0_pack, src0_bytes)) {
                    copy_src = ctx.h_src0_pack;
                }
                // uint64_t end_us = ggml_time_us();
                // if (end_us-mid_us>2000){
                //     printf("%ld ",end_us-mid_us);
                // }
            }

            if (cache_admit) {
                inserted = vram_cache_insert(cache_key, copy_src, src0_bytes, st, src0_name);
            } else {
                g_vcache.misses.fetch_add(1, std::memory_order_relaxed);
            }

            if (inserted) {
                cache_inserted = true;
                weight_h2d_queued = true;
                kernel_src0 = inserted;
            } else {
                if (cudaMemcpyAsync(ctx.d_src0, copy_src, src0_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) { release_slot(s); return false; }
                weight_h2d_queued = true;
                kernel_src0 = ctx.d_src0;
            }
        }
    }
    if (profile_cuda && cudaEventRecord(ctx.profile_weight_done, st) != cudaSuccess) {
        cudaGetLastError();
    }
           //1200us左右
    const auto t_src0 = std::chrono::steady_clock::now();

    // src1 (F32) → persistent VRAM staging → quantize to Q8_1 on GPU
    {
        std::lock_guard<std::mutex> rk(g_resize_mu);
        if (!ensure_dev(ctx.d_src1_f32, ctx.d_src1_f32_sz, src1_f32_bytes)) {
            release_slot(s); return false;
        }
        if (!ctx.h_bounce || ctx.h_bounce_sz < src1_f32_bytes) {
            ensure_host_pinned(ctx.h_bounce, ctx.h_bounce_sz, src1_f32_bytes);
        }
    }
    if (ctx.h_bounce && ctx.h_bounce_sz >= src1_f32_bytes) {
        char *bounce = (char *)ctx.h_bounce;
        const char *src1_base = (const char *)src1_f32;
        const size_t src1_ne1 = (src1_nb1 > 0 && src1_nb2 >= src1_nb1) ? src1_nb2 / src1_nb1 : 1;
        for (int64_t k = 0; k < cne1; ++k) {
            const int32_t i1 = src1_ne1 > 0 ? rows[k].i1 % (int32_t) src1_ne1 : 0;
            const int32_t i2 = rows[k].i2;
            const char *src_row = src1_base + (size_t)i1 * src1_nb1 + (size_t)i2 * src1_nb2;
            std::memcpy(bounce + (size_t)k * ne00 * sizeof(float), src_row, (size_t)ne00 * sizeof(float));
        }
        // 这里会调用一次Async
        cudaMemcpyAsync(ctx.d_src1_f32, ctx.h_bounce, src1_f32_bytes, cudaMemcpyHostToDevice, st);
    } else {
        release_slot(s);
        return false;
    }
    if (profile_cuda && cudaEventRecord(ctx.profile_input_done, st) != cudaSuccess) {
        cudaGetLastError();
    }
    const auto t_src1 = std::chrono::steady_clock::now();
    static const bool q8_batch_rows = []() {
        const char * value = std::getenv("GGML_MOE_STREAM_Q8_0_BATCH_ROWS");
        return value && value[0] && value[0] != '0';
    }();
    // Reuse the native multi-row MMVQ path for one Q8 expert's gathered
    // prefill rows. Decode and other formats retain their existing dispatch.
    if (native_float) {
        if (!ggml_cuda_moe_stream_mmvf_dev(src0_type_int, kernel_src0, ne01, ne00, nb01,
                static_cast<const float *>(ctx.d_src1_f32), static_cast<float *>(ctx.d_dst), cne1, st)) {
            release_slot(s); return false;
        }
    } else if ((t0 == GGML_TYPE_Q8_0 && q8_batch_rows && cne1 > 1) ||
            ((t0 == GGML_TYPE_MXFP4 || t0 == GGML_TYPE_F8_E4M3_B128) &&
            moe_stream_one_experimental_ds4_enabled() &&
            !moe_stream_one_ds4_nonids_enabled())) {
            //会走这部分,此处有**cudaMemsetAsync**的调用
        if (cudaMemsetAsync(ctx.d_ids, 0, (size_t)cne1 * sizeof(int32_t), st) != cudaSuccess) {
            release_slot(s);
            return false;
        }
        if (!ggml_cuda_moe_stream_mmvq_rows_dev(src0_type_int, kernel_src0, ne01, ne00, nb01,
                    (const float *)ctx.d_src1_f32, ctx.d_src1, (const int32_t *)ctx.d_ids, cne1,
                    (float *)ctx.d_dst, st)) {
            release_slot(s);
            return false;
        }
    } else {
        // 不会走
        const int64_t src1_q8_row_bytes = src1_padded * (int64_t)sizeof(block_q8_1) / QK8_1;
        for (int64_t k = 0; k < cne1; ++k) {
            const float *d_src1_row = (const float *)((const char *)ctx.d_src1_f32 + (size_t)k * ne00 * sizeof(float));
            void *d_src1_q8_row = (char *)ctx.d_src1 + (size_t)k * src1_q8_row_bytes;
            float *d_dst_row = (float *)ctx.d_dst + k * ne01;
            if (!ggml_cuda_moe_stream_mmvq_dev(src0_type_int, kernel_src0, ne01, ne00, nb01, d_src1_row, d_src1_q8_row, d_dst_row, st)) {
                release_slot(s);
                return false;
            }
        }
    }
    if (profile_cuda && cudaEventRecord(ctx.profile_kernel_done, st) != cudaSuccess) {
        cudaGetLastError();
    }
    const auto t_kernel = std::chrono::steady_clock::now();

    if (t0 == GGML_TYPE_MXFP4) {
        //会执行
        moe_stream_q8_debug_maybe(src0_name, expert_index, src0_data, ne00, nb01, ctx.d_src1, ctx.d_dst,
                src1_padded, ne01, cne1, rows, st);
    }
    //会走这部分,此处有**cudaMemsetAsync**的调用，第三个
    if (cudaMemcpyAsync(ctx.h_scratch, ctx.d_dst, dst_bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess) { release_slot(s); return false; }
    if (profile_cuda && cudaEventRecord(ctx.profile_d2h_done, st) != cudaSuccess) {
        cudaGetLastError();
    }
    const auto t_d2h = std::chrono::steady_clock::now();
    if (gate_updown_cosubmit_enabled()) {

        ggml_cuda_moe_stream_batch_preload_gate_updown(src0_name, (int)expert_index, src0_bytes);
    }

    sparse_graph_probe_write(src0_name, expert_index, cne1, src0_bytes, src1_f32_bytes, dst_bytes,
            cache_hit, cache_inserted, s, g_defer_sync, kernel_src0,
            ctx.d_src1_f32, ctx.d_dst, ctx.h_scratch, dst);

    if (!g_defer_sync) {
        //会执行，此处调用**cudaStreamSynchronize**
        if (cudaStreamSynchronize(st) != cudaSuccess) { release_slot(s); return false; }
        const auto t_sync = std::chrono::steady_clock::now();
        const float *src_buf = (const float *)ctx.h_scratch;
        for (int64_t k = 0; k < cne1; ++k) {
            const int32_t i1 = rows[k].i1;
            const int32_t i2 = rows[k].i2;
            float *dst_row = (float *)((char *)dst + i1 * dst_nb1 + i2 * dst_nb2);
            const float *src_row = src_buf + k * ne01;
            std::memcpy(dst_row, src_row, (size_t)ne01 * sizeof(float));
        }
        const auto t_scatter = std::chrono::steady_clock::now();
        moe_stream_dontneed_source_pages(src0_data, src0_bytes);
        const auto t_dontneed = std::chrono::steady_clock::now();
        if (profile_role >= 0) {
            auto ns = [](auto a, auto b) {
                return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
            };
            one_stage_profile_role & p = g_one_stage_profile[profile_role];
            p.calls.fetch_add(1, std::memory_order_relaxed);
            (cache_hit ? p.cache_hits : p.cache_misses).fetch_add(1, std::memory_order_relaxed);
            p.src0_ns.fetch_add(ns(ts0, t_src0), std::memory_order_relaxed);
            p.host_prepare_ns.fetch_add(ns(t_src0, t_src1), std::memory_order_relaxed);
            p.enqueue_ns.fetch_add(ns(t_src1, t_d2h), std::memory_order_relaxed);
            p.sync_ns.fetch_add(ns(t_d2h, t_sync), std::memory_order_relaxed);
            p.scatter_ns.fetch_add(ns(t_sync, t_scatter), std::memory_order_relaxed);
            p.total_ns.fetch_add(ns(ts0, t_dontneed), std::memory_order_relaxed);
            p.input_h2d_bytes.fetch_add(src1_f32_bytes, std::memory_order_relaxed);
            p.d2h_bytes.fetch_add(dst_bytes, std::memory_order_relaxed);
            if (weight_h2d_queued) p.weight_h2d_bytes.fetch_add(src0_bytes, std::memory_order_relaxed);
            if (profile_cuda) {
                float weight_ms = 0.0f;
                float input_ms = 0.0f;
                float kernel_ms = 0.0f;
                float d2h_ms = 0.0f;
                if (cudaEventElapsedTime(&weight_ms, ctx.profile_weight_start, ctx.profile_weight_done) == cudaSuccess &&
                        cudaEventElapsedTime(&input_ms, ctx.profile_weight_done, ctx.profile_input_done) == cudaSuccess &&
                        cudaEventElapsedTime(&kernel_ms, ctx.profile_input_done, ctx.profile_kernel_done) == cudaSuccess &&
                        cudaEventElapsedTime(&d2h_ms, ctx.profile_kernel_done, ctx.profile_d2h_done) == cudaSuccess) {
                    if (weight_h2d_queued) p.weight_h2d_us.fetch_add((uint64_t) (weight_ms * 1000.0f), std::memory_order_relaxed);
                    p.input_h2d_us.fetch_add((uint64_t) (input_ms * 1000.0f), std::memory_order_relaxed);
                    p.kernel_us.fetch_add((uint64_t) (kernel_ms * 1000.0f), std::memory_order_relaxed);
                    p.d2h_us.fetch_add((uint64_t) (d2h_ms * 1000.0f), std::memory_order_relaxed);
                } else {
                    cudaGetLastError();
                }
            }
        }
        one_trace_write(src0_name, expert_index, src0_data, src0_bytes, cne1, cache_hit, cache_inserted, s,
                        ts0, t_src0, t_src1, t_kernel, t_d2h, t_sync, t_scatter, t_dontneed);
        if (gate_updown_cosubmit_profile_enabled()) {
            //不会执行
            gate_updown_cosubmit_profile_write("gate_one", src0_name, "", expert_index, src0_bytes, cne1, 0,
                    cache_hit ? 1 : 0, cache_inserted ? 1 : 0, cache_hit ? 1 : 0, cache_hit ? 0 : 1,
                    gate_profile_pack_hit, gate_profile_pack_hit, cache_hit ? 0 : (gate_profile_pack_hit ? 0 : 1),
                    std::chrono::duration<double, std::milli>(t_src0 - ts0).count(),
                    std::chrono::duration<double, std::milli>(t_dontneed - ts0).count());
        }
        release_slot(s);
        // printf("t:%ld ",end_us-start_us);
        return true;
    }

    // Defer scatter until ggml_cuda_moe_stream_sync().  Slot remains "in_use"
    // until then so its h_scratch isn't trampled.
    deferred_scatter ds;
    ds.slot     = s;
    ds.dst_base = dst;
    ds.dst_nb1  = dst_nb1;
    ds.dst_nb2  = dst_nb2;
    ds.ne01     = ne01;
    ds.cne1     = cne1;
    for (int64_t k = 0; k < cne1 && k < 8; ++k) {
        ds.i1[k] = rows[k].i1;
        ds.i2[k] = rows[k].i2;
    }
    tls_pending.push_back(ds);
    one_trace_write(src0_name, expert_index, src0_data, src0_bytes, cne1, cache_hit, cache_inserted, s,
                    ts0, t_src0, t_src1, t_kernel, t_d2h, t_d2h, t_d2h, t_d2h);
    return true;
}

extern "C" void ggml_cuda_moe_stream_sync(void) {
    if (gate_updown_cosubmit_enabled()) {
        ggml_cuda_moe_stream_batch_flush_gate_updown_cosubmit();
    }
    flush_deferred_scatter();
}


} // namespace ggml_moe_common_adapter

#endif // GGML_CUDA_MOE_STREAM
