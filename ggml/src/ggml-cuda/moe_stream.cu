#include "moe-adapter-dispatch.h"

#ifndef GGML_CUDA_MOE_STREAM
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
extern "C" {
typedef struct { int32_t i1; int32_t i2; } ggml_moe_row_mapping;
typedef struct {
    const void * role_ptr;
    uint64_t object_generation;
    uint64_t slot_generation;
    int32_t slot_id;
    int32_t role;
} ggml_moe_expert_descriptor;
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
#include "quantize.cuh"

#include <algorithm>
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
#if defined(__linux__) && !defined(GGML_MOE_DISABLE_LIBURING) && __has_include(<liburing.h>)
#include <liburing.h>
#define GGML_MOE_STREAM_HAS_LIBURING 1
static inline void ggml_moe_stream_io_uring_sqe_set_data64(struct io_uring_sqe * sqe, uint64_t data) {
    io_uring_sqe_set_data(sqe, reinterpret_cast<void *>(static_cast<uintptr_t>(data)));
}
static inline uint64_t ggml_moe_stream_io_uring_cqe_get_data64(const struct io_uring_cqe * cqe) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(io_uring_cqe_get_data(cqe)));
}
#define io_uring_sqe_set_data64 ggml_moe_stream_io_uring_sqe_set_data64
#define io_uring_cqe_get_data64 ggml_moe_stream_io_uring_cqe_get_data64
#endif
#endif

#include <cuda_runtime.h>

extern "C" {

typedef struct {
    int32_t i1;
    int32_t i2;
} ggml_moe_row_mapping;

typedef struct {
    const void * role_ptr;
    uint64_t object_generation;
    uint64_t slot_generation;
    int32_t slot_id;
    int32_t role;
} ggml_moe_expert_descriptor;

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
// Sync all outstanding GPU work submitted by ggml_cuda_moe_stream_one and
// perform deferred dst scatter.  Call once at end of a mul_mat_id op (the
// sequential path's thread 0 invokes this after the per-expert loop).
void ggml_cuda_moe_stream_sync(void);

// Restored main/common adapter.  Its implementation is link-isolated in
// moe_stream_common.cu and has the same C ABI as the local path.
bool ggml_cuda_moe_common_stream_available(void);
const void * ggml_cuda_moe_common_stream_ds4_triplet_host_role_ptr(
    const char *, int64_t, size_t);
bool ggml_cuda_moe_common_stream_one(
    int, const char *, int64_t, const void *, int64_t, int64_t, size_t,
    const float *, size_t, size_t, int64_t, int64_t, const void *, size_t,
    float *, size_t, size_t, const ggml_moe_row_mapping *);
void ggml_cuda_moe_common_stream_q80_probe(
    int, const char *, int64_t, const void *, int64_t, int64_t, size_t,
    const void *, size_t, int64_t, int64_t, const float *, size_t, size_t,
    const ggml_moe_row_mapping *);
void ggml_cuda_moe_common_stream_q80_write(
    int, const char *, int64_t, const void *, int64_t, int64_t, size_t,
    const void *, size_t, int64_t, int64_t, float *, size_t, size_t,
    const ggml_moe_row_mapping *);
bool ggml_cuda_moe_common_stream_q80_skip(
    int, const char *, int64_t, const void *, int64_t, int64_t, size_t,
    const void *, size_t, int64_t, int64_t, float *, size_t, size_t,
    const ggml_moe_row_mapping *);
void ggml_cuda_moe_common_stream_q80_hot_batch_probe(
    int, const char *, int64_t, int64_t, int64_t, size_t, const void *,
    size_t, int64_t, const float *, size_t, size_t, const int64_t *,
    const ggml_moe_row_mapping *, int64_t, const float *, size_t, size_t);
bool ggml_cuda_moe_common_stream_one_cache_contains(
    const char *, const void *, size_t, int64_t);
bool ggml_cuda_moe_common_stream_one_cache_ready(void);
const void * ggml_cuda_moe_common_stream_one_cache_dev_ptr_or_insert(
    const char *, const void *, size_t, size_t, int64_t, cudaStream_t, int *);
bool ggml_cuda_moe_common_stream_one_cache_dev_ptrs_or_insert(
    const char *, const void *, size_t, size_t, const int *, const int32_t *,
    int, cudaStream_t, const void **, int *, ggml_moe_expert_descriptor *);
void ggml_cuda_moe_common_stream_one_gate_up_prefetch(
    const char *, size_t, const int *, const int32_t *, int);
void ggml_cuda_moe_common_stream_one_cache_release_descriptors(
    const ggml_moe_expert_descriptor *, int, cudaStream_t);
void ggml_cuda_moe_common_stream_one_cache_release_descriptors_synchronized(
    const ggml_moe_expert_descriptor *, int);
void ggml_cuda_moe_common_stream_sync(void);

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
    cudaEvent_t slot_ready[VRAM_CACHE_HT_SIZE] = {};
    bool     slot_pending[VRAM_CACHE_HT_SIZE] = {};
    bool     slot_filling[VRAM_CACHE_HT_SIZE] = {};
    cudaEvent_t slot_lease_ready[VRAM_CACHE_HT_SIZE] = {};
    bool     slot_lease_pending[VRAM_CACHE_HT_SIZE] = {};
    uint32_t slot_lease_count[VRAM_CACHE_HT_SIZE] = {};
    vram_ht_entry ht[VRAM_CACHE_HT_SIZE] = {};

    void *   ram_pool    = nullptr;
    size_t   ram_pool_sz = 0;
    int      ram_slots   = 0;
    uintptr_t ram_slot_key[VRAM_CACHE_HT_SIZE] = {};
    uint64_t  ram_slot_used[VRAM_CACHE_HT_SIZE] = {};
    cudaEvent_t ram_slot_ready[VRAM_CACHE_HT_SIZE] = {};
    bool     ram_slot_pending[VRAM_CACHE_HT_SIZE] = {};

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
    std::atomic<uint64_t> descriptor_stale_releases{0};
    std::atomic<uint64_t> score_victim_selections{0};
    std::atomic<uint64_t> score_candidate_lte_victim{0};
    std::atomic<uint64_t> ram_score_admissions{0};
    std::atomic<uint64_t> ram_score_min_demand_rejects{0};
    std::atomic<uint64_t> ram_score_victim_rejects{0};
    std::atomic<uint64_t> ram_score_layer_rejects{0};
    std::mutex mu;
};

static vram_cache g_vcache;
static bool       g_vcache_inited = false;
static std::unordered_map<uintptr_t, uint32_t> g_vram_cache_demand_scores;

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
    if (rh + rm != 0 || g_vcache.ram_pool) {
        const double hit_rate = (rh + rm) ? (100.0 * rh / (rh + rm)) : 0.0;
        std::fprintf(stderr,
            "[moe_stream] RAM cache: hits=%lu misses=%lu hit_rate=%.1f%% inserts=%lu evictions=%lu pinned=%.2f MiB slots=%d\n",
            rh, rm, hit_rate, g_vcache.ram_inserts.load(), g_vcache.ram_evictions.load(),
            g_vcache.ram_pool_sz / (1024.0 * 1024.0), g_vcache.ram_slots);
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
    const uint64_t stale_releases = g_vcache.descriptor_stale_releases.load();
    if (descriptor_acquires != 0 || descriptor_releases != 0 || stale_releases != 0) {
        std::fprintf(stderr,
            "[moe_stream] expert descriptors: fixed_pool=1 acquires=%lu releases=%lu stale_releases=%lu\n",
            descriptor_acquires, descriptor_releases, stale_releases);
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

static void vram_cache_init(size_t expert_sz) {
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
        const size_t ram_budget = ram_budget_mib * 1024ULL * 1024ULL;
        g_vcache.ram_slots = (int)(ram_budget / expert_sz);
        if (g_vcache.ram_slots > VRAM_CACHE_HT_SIZE) g_vcache.ram_slots = VRAM_CACHE_HT_SIZE;
        if (g_vcache.ram_slots > 0) {
            const size_t ram_alloc = (size_t)g_vcache.ram_slots * expert_sz;
            if (cudaHostAlloc(&g_vcache.ram_pool, ram_alloc, cudaHostAllocDefault) == cudaSuccess) {
                g_vcache.ram_pool_sz = ram_alloc;
                std::fprintf(stderr, "[moe_stream] RAM cache: %.1f GiB pinned, %d slots (%.2f MiB each)\n",
                             ram_alloc / (1024.0*1024.0*1024.0), g_vcache.ram_slots,
                             expert_sz / (1024.0*1024.0));
            } else {
                std::fprintf(stderr, "[moe_stream] RAM cache: cudaHostAlloc %.1f GiB FAILED; RAM tier disabled\n",
                             ram_alloc / (1024.0*1024.0*1024.0));
                cudaGetLastError();
                g_vcache.ram_pool = nullptr;
                g_vcache.ram_pool_sz = 0;
                g_vcache.ram_slots = 0;
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
    if (slot < 0 || slot >= g_vcache.n_slots || !g_vcache.slot_pending[slot]) return true;
    cudaEvent_t ev = g_vcache.slot_ready[slot];
    if (!ev) { g_vcache.slot_pending[slot] = false; return true; }
    cudaError_t qerr = cudaEventQuery(ev);
    if (qerr == cudaSuccess) { g_vcache.slot_pending[slot] = false; return true; }
    if (qerr != cudaErrorNotReady) { cudaGetLastError(); return false; }
    if (cudaStreamWaitEvent(st, ev, 0) != cudaSuccess) { cudaGetLastError(); return false; }
    return true;
}

static bool vram_cache_stream_wait_slot_lease_locked(int slot, cudaStream_t st) {
    if (slot < 0 || slot >= g_vcache.n_slots || !g_vcache.slot_lease_pending[slot]) return true;
    cudaEvent_t ev = g_vcache.slot_lease_ready[slot];
    if (!ev) { g_vcache.slot_lease_pending[slot] = false; return true; }
    const cudaError_t qerr = cudaEventQuery(ev);
    if (qerr == cudaSuccess) { g_vcache.slot_lease_pending[slot] = false; return true; }
    if (qerr != cudaErrorNotReady) { cudaGetLastError(); return false; }
    if (cudaStreamWaitEvent(st, ev, 0) != cudaSuccess) { cudaGetLastError(); return false; }
    return true;
}

static bool vram_cache_record_slot_lease_locked(int slot, cudaStream_t st) {
    if (slot < 0 || slot >= g_vcache.n_slots) return false;
    if (!g_vcache.slot_lease_ready[slot] &&
            cudaEventCreateWithFlags(&g_vcache.slot_lease_ready[slot], cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        g_vcache.slot_lease_pending[slot] = false;
        return false;
    }
    if (cudaEventRecord(g_vcache.slot_lease_ready[slot], st) != cudaSuccess) {
        cudaGetLastError();
        g_vcache.slot_lease_pending[slot] = false;
        return false;
    }
    g_vcache.slot_lease_pending[slot] = true;
    return true;
}

static bool ram_cache_stream_wait_slot_ready_locked(int slot, cudaStream_t st) {
    if (slot < 0 || slot >= g_vcache.ram_slots || !g_vcache.ram_slot_pending[slot]) return true;
    cudaEvent_t ev = g_vcache.ram_slot_ready[slot];
    if (!ev) { g_vcache.ram_slot_pending[slot] = false; return true; }
    cudaError_t qerr = cudaEventQuery(ev);
    if (qerr == cudaSuccess) { g_vcache.ram_slot_pending[slot] = false; return true; }
    if (qerr != cudaErrorNotReady) { cudaGetLastError(); return false; }
    if (cudaStreamWaitEvent(st, ev, 0) != cudaSuccess) { cudaGetLastError(); return false; }
    return true;
}

static bool ram_cache_slot_ready_now_locked(int slot) {
    if (slot < 0 || slot >= g_vcache.ram_slots) return false;
    if (!g_vcache.ram_slot_pending[slot]) return true;
    cudaEvent_t ev = g_vcache.ram_slot_ready[slot];
    if (!ev) { g_vcache.ram_slot_pending[slot] = false; return true; }
    cudaError_t qerr = cudaEventQuery(ev);
    if (qerr == cudaSuccess) { g_vcache.ram_slot_pending[slot] = false; return true; }
    if (qerr != cudaErrorNotReady) cudaGetLastError();
    return false;
}

static bool vram_cache_record_slot_ready_locked(int slot, cudaStream_t st) {
    if (slot < 0 || slot >= g_vcache.n_slots) return false;
    if (!g_vcache.slot_ready[slot] && cudaEventCreateWithFlags(&g_vcache.slot_ready[slot], cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError(); g_vcache.slot_pending[slot] = false; return false;
    }
    if (cudaEventRecord(g_vcache.slot_ready[slot], st) != cudaSuccess) {
        cudaGetLastError(); g_vcache.slot_pending[slot] = false; return false;
    }
    g_vcache.slot_pending[slot] = true;
    return true;
}

static bool ram_cache_record_slot_ready_locked(int slot, cudaStream_t st) {
    if (slot < 0 || slot >= g_vcache.ram_slots) return false;
    if (!g_vcache.ram_slot_ready[slot] && cudaEventCreateWithFlags(&g_vcache.ram_slot_ready[slot], cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError(); g_vcache.ram_slot_pending[slot] = false; return false;
    }
    if (cudaEventRecord(g_vcache.ram_slot_ready[slot], st) != cudaSuccess) {
        cudaGetLastError(); g_vcache.ram_slot_pending[slot] = false; return false;
    }
    g_vcache.ram_slot_pending[slot] = true;
    return true;
}

static int ram_cache_find_slot_locked(uintptr_t key) {
    if (!g_vcache.ram_pool || g_vcache.ram_slots == 0) return -1;
    for (int slot = 0; slot < g_vcache.ram_slots; ++slot) if (g_vcache.ram_slot_key[slot] == key) return slot;
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
    if (stat->demands < ds4_ram_score_min_demands()) {
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

static int ram_cache_select_scored_slot_locked(
        uintptr_t candidate_key,
        int avoid_slot,
        cudaStream_t stream,
        bool require_ready_now) {
    ds4_ram_key_stat * candidate = ds4_ram_key_stat_locked(candidate_key);
    if (!candidate || candidate->layer < 0 || candidate->layer >= 43) return -1;
    const int quota = ds4_ram_layer_quota();
    int same_layer_count = 0;
    for (int i = 0; i < g_vcache.ram_slots; ++i) {
        if (g_vcache.ram_slot_key[i] == 0) continue;
        const ds4_ram_key_stat * resident = ds4_ram_key_stat_locked(g_vcache.ram_slot_key[i]);
        if (resident && resident->layer == candidate->layer) ++same_layer_count;
    }

    int slot = -1;
    uint64_t lowest_score = UINT64_MAX;
    uint64_t oldest = UINT64_MAX;
    const bool restrict_layer = quota > 0 && same_layer_count >= quota;
    for (int i = 0; i < g_vcache.ram_slots; ++i) {
        if (i == avoid_slot) continue;
        if (g_vcache.ram_slot_key[i] == 0) {
            if (!restrict_layer) return i;
            continue;
        }
        if (require_ready_now) {
            if (!ram_cache_slot_ready_now_locked(i)) continue;
        } else if (!ram_cache_stream_wait_slot_ready_locked(i, stream)) {
            continue;
        }
        const ds4_ram_key_stat * resident = ds4_ram_key_stat_locked(g_vcache.ram_slot_key[i]);
        if (restrict_layer && (!resident || resident->layer != candidate->layer)) continue;
        const uint64_t score = ds4_ram_score_locked(g_vcache.ram_slot_key[i]);
        if (score < lowest_score || (score == lowest_score && g_vcache.ram_slot_used[i] < oldest)) {
            lowest_score = score;
            oldest = g_vcache.ram_slot_used[i];
            slot = i;
        }
    }
    if (slot < 0) {
        g_vcache.ram_score_layer_rejects.fetch_add(1, std::memory_order_relaxed);
        ++g_ds4_ram_layer_rejects[candidate->layer];
        return -1;
    }
    if (ds4_ram_score_locked(candidate_key) <= lowest_score) {
        g_vcache.ram_score_victim_rejects.fetch_add(1, std::memory_order_relaxed);
        ++g_ds4_ram_layer_rejects[candidate->layer];
        return -1;
    }
    return slot;
}

static int ram_cache_select_slot_locked(int avoid_slot, cudaStream_t st) {
    if (!g_vcache.ram_pool || g_vcache.ram_slots == 0) return -1;
    int slot = -1; uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < g_vcache.ram_slots; ++i) {
        if (i == avoid_slot) continue;
        if (g_vcache.ram_slot_key[i] == 0) { slot = i; break; }
        if (g_vcache.ram_slot_used[i] < oldest) { oldest = g_vcache.ram_slot_used[i]; slot = i; }
    }
    if (slot < 0 || !ram_cache_stream_wait_slot_ready_locked(slot, st)) return -1;
    return slot;
}

static int ram_cache_select_ready_slot_locked(int avoid_slot) {
    if (!g_vcache.ram_pool || g_vcache.ram_slots == 0) return -1;
    int slot = -1; uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < g_vcache.ram_slots; ++i) {
        if (i == avoid_slot) continue;
        if (g_vcache.ram_slot_key[i] == 0) { slot = i; break; }
        if (!ram_cache_slot_ready_now_locked(i)) continue;
        if (g_vcache.ram_slot_used[i] < oldest) { oldest = g_vcache.ram_slot_used[i]; slot = i; }
    }
    return slot;
}

static void ds4_ram_prefetch_note_ram_eviction(uintptr_t key) {
    if (key == 0) return;
    std::lock_guard<std::mutex> lock(g_ds4_ram_prefetch.mu);
    if (g_ds4_ram_prefetch.ready_keys.erase(key) != 0) {
        g_ds4_ram_prefetch.wasted.fetch_add(1, std::memory_order_relaxed);
    }
}

static bool ram_cache_store_host_locked(uintptr_t key, const void *host_data, size_t sz, int avoid_slot) {
    if (!g_vcache.ram_pool || g_vcache.ram_slots == 0 || !host_data || key == 0 || sz > g_vcache.slot_sz) return false;
    if (!ds4_ram_admission_allows_locked(key)) return false;
    int slot = ram_cache_find_slot_locked(key);
    if (slot == avoid_slot) return true;
    if (slot >= 0) { if (!ram_cache_slot_ready_now_locked(slot)) return false; }
    else {
        slot = ds4_ram_score_policy_enabled() ?
            ram_cache_select_scored_slot_locked(key, avoid_slot, nullptr, true) :
            ram_cache_select_ready_slot_locked(avoid_slot);
        if (slot < 0) return false;
    }
    const bool replacing = g_vcache.ram_slot_key[slot] != 0 && g_vcache.ram_slot_key[slot] != key;
    if (replacing) ds4_ram_prefetch_note_ram_eviction(g_vcache.ram_slot_key[slot]);
    void *dst = (char *)g_vcache.ram_pool + (size_t)slot * g_vcache.slot_sz;
    std::memcpy(dst, host_data, sz);
    g_vcache.ram_slot_key[slot] = key;
    g_vcache.ram_slot_used[slot] = g_vcache.clock.fetch_add(1, std::memory_order_relaxed);
    g_vcache.ram_slot_pending[slot] = false;
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
    if (!g_vcache.ram_pool || g_vcache.ram_slots == 0 || !dev_data || key == 0 || sz > g_vcache.slot_sz) return false;
    if (!ds4_ram_admission_allows_locked(key)) return false;
    int slot = ram_cache_find_slot_locked(key);
    if (slot >= 0 && slot != avoid_slot) { g_vcache.ram_slot_used[slot] = g_vcache.clock.fetch_add(1, std::memory_order_relaxed); return true; }
    if (slot == avoid_slot) slot = -1;
    if (slot >= 0) { if (!ram_cache_stream_wait_slot_ready_locked(slot, st)) return false; }
    else {
        slot = ds4_ram_score_policy_enabled() ?
            ram_cache_select_scored_slot_locked(key, avoid_slot, st, false) :
            ram_cache_select_slot_locked(avoid_slot, st);
        if (slot < 0) return false;
    }
    const bool replacing = g_vcache.ram_slot_key[slot] != 0 && g_vcache.ram_slot_key[slot] != key;
    if (replacing) ds4_ram_prefetch_note_ram_eviction(g_vcache.ram_slot_key[slot]);
    void *dst = (char *)g_vcache.ram_pool + (size_t)slot * g_vcache.slot_sz;
    if (cudaMemcpyAsync(dst, dev_data, sz, cudaMemcpyDeviceToHost, st) != cudaSuccess) { cudaGetLastError(); return false; }
    g_vcache.ram_slot_key[slot] = key;
    g_vcache.ram_slot_used[slot] = g_vcache.clock.fetch_add(1, std::memory_order_relaxed);
    g_vcache.ram_inserts.fetch_add(1, std::memory_order_relaxed);
    if (ds4_ram_score_policy_enabled()) {
        ds4_ram_key_stat * stat = ds4_ram_key_stat_locked(key);
        if (stat && stat->layer >= 0 && stat->layer < 43) ++g_ds4_ram_layer_admissions[stat->layer];
        g_vcache.ram_score_admissions.fetch_add(1, std::memory_order_relaxed);
    }
    if (replacing) g_vcache.ram_evictions.fetch_add(1, std::memory_order_relaxed);
    ram_cache_record_slot_ready_locked(slot, st);
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
    if (ram_src_slot >= 0) ram_cache_record_slot_ready_locked(ram_src_slot, st);
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
    if (!g_vcache.ram_pool || g_vcache.ram_slots == 0 || sz > g_vcache.slot_sz) return nullptr;
    std::lock_guard<std::mutex> lk(g_vcache.mu);
    int ram_slot = ram_cache_find_slot_locked(key);
    if (ram_slot < 0) { g_vcache.ram_misses.fetch_add(1, std::memory_order_relaxed); return nullptr; }
    if (!ram_cache_stream_wait_slot_ready_locked(ram_slot, st)) return nullptr;
    g_vcache.ram_slot_used[ram_slot] = g_vcache.clock.fetch_add(1, std::memory_order_relaxed);
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
    const void *src = (const char *)g_vcache.ram_pool + (size_t)ram_slot * g_vcache.slot_sz;
    return vram_cache_insert_locked(key, src, sz, st, true, tensor, ram_slot);
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
    if (!object_ptr || !g_vcache.pool || g_vcache.slot_sz == 0) return false;

    std::lock_guard<std::mutex> lk(g_vcache.mu);
    const uintptr_t pool_begin = (uintptr_t) g_vcache.pool;
    const uintptr_t pool_end = pool_begin + g_vcache.pool_sz;
    const uintptr_t address = (uintptr_t) object_ptr;
    if (address < pool_begin || address >= pool_end) return false;
    const size_t delta = (size_t) (address - pool_begin);
    if (delta % g_vcache.slot_sz != 0) return false;
    const int slot = (int) (delta / g_vcache.slot_sz);
    if (slot < 0 || slot >= g_vcache.n_slots ||
            g_vcache.slot_key[slot] != object_generation ||
            !vram_cache_stream_wait_slot_ready_locked(slot, st)) return false;

    ++g_vcache.slot_lease_count[slot];
    descriptor.role_ptr = (const char *) object_ptr + role_offset;
    descriptor.object_generation = (uint64_t) object_generation;
    descriptor.slot_generation = g_vcache.slot_generation[slot];
    descriptor.slot_id = slot;
    g_vcache.descriptor_acquires.fetch_add(1, std::memory_order_relaxed);
    return true;
}

static void vram_cache_cancel_descriptor(const ggml_moe_expert_descriptor & descriptor) {
    if (descriptor.slot_id < 0) return;
    std::lock_guard<std::mutex> lk(g_vcache.mu);
    const int slot = descriptor.slot_id;
    if (slot < g_vcache.n_slots &&
            g_vcache.slot_key[slot] == (uintptr_t) descriptor.object_generation &&
            g_vcache.slot_generation[slot] == descriptor.slot_generation &&
            g_vcache.slot_lease_count[slot] > 0) {
        --g_vcache.slot_lease_count[slot];
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
        const char * tensor = "ds4_triplet") {
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
    if (evicted_key != 0 && evicted_key != key && g_vcache.ram_pool) {
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
    g_vcache.slot_tensor_tag[slot] = tensor_tag;
    g_vcache.slot_used[slot] = g_vcache.clock.fetch_add(1, std::memory_order_relaxed);
    if (++g_vcache.slot_generation[slot] == 0) ++g_vcache.slot_generation[slot];
    g_vcache.slot_filling[slot] = true;
    ++g_vcache.slot_lease_count[slot];
    g_vcache.misses.fetch_add(1, std::memory_order_relaxed);
    reservation.dst = dst;
    reservation.key = key;
    reservation.slot_generation = g_vcache.slot_generation[slot];
    reservation.slot = slot;
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
    if (!vram_cache_record_slot_ready_locked(slot, transfer_stream)) return false;
    g_vcache.slot_filling[slot] = false;
    --g_vcache.slot_lease_count[slot];
    return true;
}

static void vram_cache_fail_triplet(const ds4_vram_reservation & reservation) {
    std::lock_guard<std::mutex> lk(g_vcache.mu);
    const int slot = reservation.slot;
    if (slot < 0 || slot >= g_vcache.n_slots ||
            g_vcache.slot_key[slot] != reservation.key ||
            g_vcache.slot_generation[slot] != reservation.slot_generation) return;
    if (vram_cache_hash_lookup_enabled()) {
        vram_cache_hash_erase_locked(reservation.key, slot);
    }
    g_vcache.slot_key[slot] = 0;
    g_vcache.slot_filling[slot] = false;
    if (g_vcache.slot_lease_count[slot] > 0) --g_vcache.slot_lease_count[slot];
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
#if defined(GGML_MOE_STREAM_HAS_LIBURING)
    io_uring ring = {};
    bool ring_inited = false;
#endif
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

struct one_pack_cqe_import_slot {
    void * ptr = nullptr;
    size_t size = 0;
    cudaEvent_t ready = nullptr;
};

static one_pack_cqe_state g_one_pack_cqe;
static one_pack_cqe_import_slot g_one_pack_cqe_imports[8];

struct one_gate_up_prefetch_job {
    const one_expert_pack_entry * entry = nullptr;
    ds4_vram_reservation reservation;
    bool read_ok = false;
    bool published = false;
};

struct one_gate_up_prefetch_state {
    std::mutex mu;
#if defined(GGML_MOE_STREAM_HAS_LIBURING)
    io_uring ring = {};
    bool ring_inited = false;
    bool sqpoll = false;
    bool fixed_file = false;
#endif
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
static one_pack_cqe_import_slot g_one_gate_up_prefetch_imports[8];

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
    int count = 0;
    bool done = false;
    bool success = false;
    std::mutex mu;
    std::condition_variable cv;
};

struct ds4_triplet_pack_state {
    int fd = -1;
    int fd_direct = -1;
    bool inited = false;
    bool enabled = false;
    uint64_t object_bytes = 0;
    uint64_t role_bytes[3] = {};
    std::vector<ds4_triplet_pack_entry> entries;
    std::mutex mu;
    std::mutex io_mu;
    std::mutex pipe_mu;
    std::condition_variable pipe_cv;
    std::deque<ds4_triplet_pipe_batch *> pipe_queue;
    std::thread pipe_worker;
    cudaStream_t pipe_stream = nullptr;
    bool pipe_requested = false;
    bool pipe_started = false;
    bool pipe_stop = false;
    bool pipe_failed = false;
#if defined(GGML_MOE_STREAM_HAS_LIBURING)
    io_uring ring = {};
    bool ring_inited = false;
#endif
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> reads{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> failures{0};
    std::atomic<uint64_t> iouring_batches{0};
    std::atomic<uint64_t> iouring_reads{0};
    std::atomic<uint64_t> iouring_submit_us{0};
    std::atomic<uint64_t> iouring_wait_us{0};
    std::atomic<uint64_t> iouring_fallbacks{0};
    std::atomic<uint64_t> pipe_batches{0};
    std::atomic<uint64_t> pipe_cqes{0};
    std::atomic<uint64_t> pipe_submit_us{0};
    std::atomic<uint64_t> pipe_wait_us{0};
    std::atomic<uint64_t> pipe_process_us{0};
    std::atomic<uint64_t> pipe_caller_wait_us{0};
    std::atomic<uint64_t> pipe_h2d_bytes{0};
    std::atomic<uint64_t> pipe_h2d_enqueue_us{0};
    std::atomic<uint64_t> pipe_h2d_device_us{0};
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
};

static ds4_triplet_pack_state g_ds4_triplet_pack;

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

static void ds4_triplet_pipe_worker_main();

static bool one_pack_read_exact_fd(int fd, void * dst, size_t sz, uint64_t off) {
    char * out = (char *) dst;
    size_t done = 0;
    while (done < sz) {
        const ssize_t got = ::pread(fd, out + done, sz - done, (off_t)(off + done));
        if (got <= 0) {
            return false;
        }
        done += (size_t) got;
    }
    return true;
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
    if (g_ds4_triplet_pack.pipe_worker.joinable()) {
        {
            std::lock_guard<std::mutex> lock(g_ds4_triplet_pack.pipe_mu);
            g_ds4_triplet_pack.pipe_stop = true;
        }
        g_ds4_triplet_pack.pipe_cv.notify_one();
        g_ds4_triplet_pack.pipe_worker.join();
    }
    std::fprintf(stderr,
            "[moe_stream] DS4 triplet pack: hits=%lu misses=%lu reads=%lu bytes=%lu failures=%lu "
            "objects=%zu object_mib=%.2f direct=%d iouring_batches=%lu iouring_reads=%lu "
            "iouring_submit_ms=%.3f iouring_wait_ms=%.3f iouring_fallbacks=%lu "
            "pipe_batches=%lu pipe_cqes=%lu pipe_submit_ms=%.3f pipe_wait_ms=%.3f "
            "pipe_process_ms=%.3f pipe_caller_wait_ms=%.3f pipe_h2d_bytes=%lu pipe_h2d_enqueue_ms=%.3f "
            "pipe_h2d_device_ms=%.3f "
            "host_handoff_lookups=%lu host_handoff_hits=%lu host_handoff_bytes=%lu role_pair_reads=%lu\n",
            g_ds4_triplet_pack.hits.load(),
            g_ds4_triplet_pack.misses.load(),
            g_ds4_triplet_pack.reads.load(),
            g_ds4_triplet_pack.bytes.load(),
            g_ds4_triplet_pack.failures.load(),
            g_ds4_triplet_pack.entries.size(),
            g_ds4_triplet_pack.object_bytes / (1024.0 * 1024.0),
            g_ds4_triplet_pack.fd_direct >= 0 ? 1 : 0,
            g_ds4_triplet_pack.iouring_batches.load(),
            g_ds4_triplet_pack.iouring_reads.load(),
            g_ds4_triplet_pack.iouring_submit_us.load() / 1000.0,
            g_ds4_triplet_pack.iouring_wait_us.load() / 1000.0,
            g_ds4_triplet_pack.iouring_fallbacks.load(),
            g_ds4_triplet_pack.pipe_batches.load(),
            g_ds4_triplet_pack.pipe_cqes.load(),
            g_ds4_triplet_pack.pipe_submit_us.load() / 1000.0,
            g_ds4_triplet_pack.pipe_wait_us.load() / 1000.0,
            g_ds4_triplet_pack.pipe_process_us.load() / 1000.0,
            g_ds4_triplet_pack.pipe_caller_wait_us.load() / 1000.0,
            g_ds4_triplet_pack.pipe_h2d_bytes.load(),
            g_ds4_triplet_pack.pipe_h2d_enqueue_us.load() / 1000.0,
            g_ds4_triplet_pack.pipe_h2d_device_us.load() / 1000.0,
            g_ds4_triplet_pack.host_handoff_lookups.load(),
            g_ds4_triplet_pack.host_handoff_hits.load(),
            g_ds4_triplet_pack.host_handoff_hits.load() * g_ds4_triplet_pack.role_bytes[2],
            g_ds4_triplet_pack.role_pair_reads.load());
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
#if defined(GGML_MOE_STREAM_HAS_LIBURING)
    if (g_ds4_triplet_pack.ring_inited) {
        io_uring_queue_exit(&g_ds4_triplet_pack.ring);
        g_ds4_triplet_pack.ring_inited = false;
    }
#endif
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
    }
    g_ds4_ram_key_stats.clear();
    g_ds4_ram_key_stats.reserve(entries.size());
    for (const ds4_triplet_pack_entry & entry : entries) {
        g_ds4_ram_key_stats.emplace((uintptr_t) entry.generation,
            ds4_ram_key_stat{entry.layer, 0, 0});
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
#if defined(GGML_MOE_STREAM_HAS_LIBURING)
    if (iouring_requested && g_ds4_triplet_pack.fd_direct >= 0) {
        const char * depth_env = std::getenv("GGML_MOE_IO_DEPTH");
        unsigned depth = depth_env && depth_env[0] ? (unsigned) std::strtoul(depth_env, nullptr, 10) : 16U;
        depth = std::max(8U, std::min(depth, 64U));
        if (io_uring_queue_init(depth, &g_ds4_triplet_pack.ring, 0) == 0) {
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
    std::copy(role_bytes, role_bytes + 3, g_ds4_triplet_pack.role_bytes);
    g_ds4_triplet_pack.entries = std::move(entries);
    g_ds4_triplet_pack.enabled = true;
    g_ds4_triplet_pack.pipe_requested = pipe_requested;
#if defined(GGML_MOE_STREAM_HAS_LIBURING)
    if (pipe_requested && g_ds4_triplet_pack.ring_inited) {
        g_ds4_triplet_pack.pipe_worker = std::thread(ds4_triplet_pipe_worker_main);
        std::unique_lock<std::mutex> pipe_lock(g_ds4_triplet_pack.pipe_mu);
        g_ds4_triplet_pack.pipe_cv.wait(pipe_lock, [] {
            return g_ds4_triplet_pack.pipe_started;
        });
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
    if (ggml_moe::adapter_uses_common()) {
        return ggml_cuda_moe_common_stream_ds4_triplet_host_role_ptr(
            src0_name, expert_index, role_nbytes);
    }
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

static bool ds4_triplet_read(const ds4_triplet_pack_entry & entry, void * dst) {
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
    if (!ds4_ram_pilot_prefetch_enabled() || !g_vcache.ram_pool || g_vcache.ram_slots <= 0) return false;
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
    bool completed[8] = {};
    bool iouring_attempted = false;

#if defined(GGML_MOE_STREAM_HAS_LIBURING)
    if (g_ds4_triplet_pack.ring_inited && g_ds4_triplet_pack.fd_direct >= 0) {
        iouring_attempted = true;
        std::lock_guard<std::mutex> io_lock(g_ds4_triplet_pack.io_mu);
        const auto submit_start = std::chrono::steady_clock::now();
        int prepared = 0;
        for (; prepared < count; ++prepared) {
            io_uring_sqe * sqe = io_uring_get_sqe(&g_ds4_triplet_pack.ring);
            if (!sqe) break;
            io_uring_prep_read(
                sqe, g_ds4_triplet_pack.fd_direct, destinations[prepared],
                (unsigned) entries[prepared]->nbytes, (off_t) entries[prepared]->offset);
            io_uring_sqe_set_data(sqe, (void *) (uintptr_t) prepared);
        }
        const int submitted = prepared > 0 ? io_uring_submit(&g_ds4_triplet_pack.ring) : 0;
        const auto submit_done = std::chrono::steady_clock::now();
        g_ds4_triplet_pack.iouring_submit_us.fetch_add((uint64_t)
            std::chrono::duration_cast<std::chrono::microseconds>(submit_done - submit_start).count());
        if (submitted > 0) {
            const auto wait_start = std::chrono::steady_clock::now();
            for (int seen = 0; seen < submitted; ++seen) {
                io_uring_cqe * cqe = nullptr;
                if (io_uring_wait_cqe(&g_ds4_triplet_pack.ring, &cqe) != 0 || !cqe) break;
                const int index = (int) (uintptr_t) io_uring_cqe_get_data(cqe);
                if (index >= 0 && index < count && cqe->res == (int) entries[index]->nbytes) {
                    completed[index] = true;
                    g_ds4_triplet_pack.reads.fetch_add(1, std::memory_order_relaxed);
                    g_ds4_triplet_pack.bytes.fetch_add(entries[index]->nbytes, std::memory_order_relaxed);
                    g_ds4_triplet_pack.iouring_reads.fetch_add(1, std::memory_order_relaxed);
                }
                io_uring_cqe_seen(&g_ds4_triplet_pack.ring, cqe);
            }
            const auto wait_done = std::chrono::steady_clock::now();
            g_ds4_triplet_pack.iouring_wait_us.fetch_add((uint64_t)
                std::chrono::duration_cast<std::chrono::microseconds>(wait_done - wait_start).count());
            g_ds4_triplet_pack.iouring_batches.fetch_add(1, std::memory_order_relaxed);
        }
    }
#endif

    bool ok = true;
    for (int i = 0; i < count; ++i) {
        if (completed[i]) continue;
        if (iouring_attempted) g_ds4_triplet_pack.iouring_fallbacks.fetch_add(1, std::memory_order_relaxed);
        if (!ds4_triplet_read(*entries[i], destinations[i])) ok = false;
    }
    return ok;
}

static bool ds4_triplet_pipe_process(ds4_triplet_pipe_batch & batch) {
#if defined(GGML_MOE_STREAM_HAS_LIBURING)
    if (!g_ds4_triplet_pack.ring_inited || g_ds4_triplet_pack.fd_direct < 0 ||
            !g_ds4_triplet_pack.pipe_stream || batch.count <= 0) return false;

    const auto process_start = std::chrono::steady_clock::now();
    bool completed[8] = {};
    bool h2d_profiled[8] = {};
    const char * profile_env = std::getenv("GGML_MOE_STREAM_DECODE_BATCH_PROFILE");
    bool profile_h2d = profile_env && profile_env[0] && profile_env[0] != '0';
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
    int submitted = 0;
    {
        std::lock_guard<std::mutex> io_lock(g_ds4_triplet_pack.io_mu);
        for (int i = 0; i < batch.count; ++i) {
            io_uring_sqe * sqe = io_uring_get_sqe(&g_ds4_triplet_pack.ring);
            if (!sqe) break;
            io_uring_prep_read(
                sqe, g_ds4_triplet_pack.fd_direct, batch.host[i],
                (unsigned) batch.entries[i]->nbytes, (off_t) batch.entries[i]->offset);
            io_uring_sqe_set_data(sqe, (void *) (uintptr_t) (i + 1));
            ++submitted;
        }
        const auto submit_start = std::chrono::steady_clock::now();
        const int submit_rc = submitted > 0 ? io_uring_submit(&g_ds4_triplet_pack.ring) : 0;
        g_ds4_triplet_pack.pipe_submit_us.fetch_add((uint64_t)
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - submit_start).count(), std::memory_order_relaxed);
        if (submit_rc < 0) submitted = 0;
        else submitted = std::min(submitted, submit_rc);

        const auto wait_start = std::chrono::steady_clock::now();
        for (int seen = 0; seen < submitted; ++seen) {
            io_uring_cqe * cqe = nullptr;
            if (io_uring_wait_cqe(&g_ds4_triplet_pack.ring, &cqe) != 0 || !cqe) {
                success = false;
                break;
            }
            const int index = (int) (uintptr_t) io_uring_cqe_get_data(cqe) - 1;
            if (index >= 0 && index < batch.count &&
                    cqe->res == (int) batch.entries[index]->nbytes) {
                const auto h2d_start = std::chrono::steady_clock::now();
                if (profile_h2d && cudaEventRecord(
                        g_ds4_triplet_pack.pipe_h2d_start_events[index],
                        g_ds4_triplet_pack.pipe_stream) != cudaSuccess) {
                    cudaGetLastError();
                    profile_h2d = false;
                }
                const bool copied = cudaMemcpyAsync(
                    batch.reservations[index].dst, batch.host[index],
                    (size_t) batch.entries[index]->nbytes, cudaMemcpyHostToDevice,
                    g_ds4_triplet_pack.pipe_stream) == cudaSuccess;
                if (copied && profile_h2d && cudaEventRecord(
                        g_ds4_triplet_pack.pipe_h2d_stop_events[index],
                        g_ds4_triplet_pack.pipe_stream) == cudaSuccess) {
                    h2d_profiled[index] = true;
                }
                const bool published = copied && vram_cache_publish_triplet(
                    batch.reservations[index], g_ds4_triplet_pack.pipe_stream);
                g_ds4_triplet_pack.pipe_h2d_enqueue_us.fetch_add((uint64_t)
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - h2d_start).count(), std::memory_order_relaxed);
                if (published) {
                    completed[index] = true;
                    g_ds4_triplet_pack.reads.fetch_add(1, std::memory_order_relaxed);
                    g_ds4_triplet_pack.bytes.fetch_add(batch.entries[index]->nbytes, std::memory_order_relaxed);
                    g_ds4_triplet_pack.iouring_reads.fetch_add(1, std::memory_order_relaxed);
                    g_ds4_triplet_pack.pipe_cqes.fetch_add(1, std::memory_order_relaxed);
                    g_ds4_triplet_pack.pipe_h2d_bytes.fetch_add(
                        batch.entries[index]->nbytes, std::memory_order_relaxed);
                } else {
                    cudaGetLastError();
                    success = false;
                }
            }
            io_uring_cqe_seen(&g_ds4_triplet_pack.ring, cqe);
        }
        g_ds4_triplet_pack.pipe_wait_us.fetch_add((uint64_t)
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - wait_start).count(), std::memory_order_relaxed);
    }

    for (int i = 0; i < batch.count; ++i) {
        if (completed[i]) continue;
        g_ds4_triplet_pack.iouring_fallbacks.fetch_add(1, std::memory_order_relaxed);
        if (!ds4_triplet_read(*batch.entries[i], batch.host[i]) ||
                cudaMemcpyAsync(
                    batch.reservations[i].dst, batch.host[i],
                    (size_t) batch.entries[i]->nbytes, cudaMemcpyHostToDevice,
                    g_ds4_triplet_pack.pipe_stream) != cudaSuccess ||
                !vram_cache_publish_triplet(batch.reservations[i], g_ds4_triplet_pack.pipe_stream)) {
            cudaGetLastError();
            vram_cache_fail_triplet(batch.reservations[i]);
            success = false;
        } else {
            g_ds4_triplet_pack.pipe_h2d_bytes.fetch_add(
                batch.entries[i]->nbytes, std::memory_order_relaxed);
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
    return success;
#else
    (void) batch;
    return false;
#endif
}

static void ds4_triplet_pipe_worker_main() {
    const bool device_ok = cudaSetDevice(0) == cudaSuccess &&
        cudaStreamCreateWithFlags(&g_ds4_triplet_pack.pipe_stream, cudaStreamNonBlocking) == cudaSuccess;
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
        if (batch) {
            {
                std::lock_guard<std::mutex> lock(batch->mu);
                batch->success = ok;
                batch->done = true;
            }
            batch->cv.notify_one();
        }
    }
    cudaStreamDestroy(g_ds4_triplet_pack.pipe_stream);
    g_ds4_triplet_pack.pipe_stream = nullptr;
}

static bool ds4_triplet_pipe_submit(ds4_triplet_pipe_batch & batch) {
    if (!g_ds4_triplet_pack.pipe_requested || g_ds4_triplet_pack.pipe_failed ||
            !g_ds4_triplet_pack.pipe_started || batch.count <= 0) return false;
    const auto wait_start = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(g_ds4_triplet_pack.pipe_mu);
        if (g_ds4_triplet_pack.pipe_stop || g_ds4_triplet_pack.pipe_failed) return false;
        g_ds4_triplet_pack.pipe_queue.push_back(&batch);
    }
    g_ds4_triplet_pack.pipe_cv.notify_one();
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
            "fallbacks=%lu bytes=%lu submit_ms=%.3f wait_ms=%.3f h2d_enqueue_ms=%.3f\n",
            g_one_pack_cqe.ready ? 1 : 0, g_one_pack_cqe.batches, g_one_pack_cqe.jobs,
            g_one_pack_cqe.cqes, g_one_pack_cqe.fallbacks, g_one_pack_cqe.bytes,
            g_one_pack_cqe.submit_us / 1000.0, g_one_pack_cqe.wait_us / 1000.0,
            g_one_pack_cqe.h2d_enqueue_us / 1000.0);
#if defined(GGML_MOE_STREAM_HAS_LIBURING)
        if (g_one_pack_cqe.ring_inited) {
            io_uring_queue_exit(&g_one_pack_cqe.ring);
            g_one_pack_cqe.ring_inited = false;
        }
#endif
        if (g_one_pack_cqe.stream) {
            cudaStreamDestroy(g_one_pack_cqe.stream);
            g_one_pack_cqe.stream = nullptr;
        }
    }
    if (g_one_gate_up_prefetch.init_attempted) {
        std::fprintf(stderr,
            "[moe_stream] one gate-up prefetch: sqpoll=%d fixed_file=%d calls=%lu scheduled=%lu placement_skips=%lu resident_skips=%lu "
            "cqes=%lu demand_fallbacks=%lu bytes=%lu submit_ms=%.3f residual_wait_ms=%.3f "
            "h2d_enqueue_ms=%.3f\n",
#if defined(GGML_MOE_STREAM_HAS_LIBURING)
            g_one_gate_up_prefetch.sqpoll ? 1 : 0,
            g_one_gate_up_prefetch.fixed_file ? 1 : 0,
#else
            0, 0,
#endif
            g_one_gate_up_prefetch.calls, g_one_gate_up_prefetch.scheduled,
            g_one_gate_up_prefetch.placement_skips, g_one_gate_up_prefetch.resident_skips,
            g_one_gate_up_prefetch.cqes,
            g_one_gate_up_prefetch.demand_fallbacks, g_one_gate_up_prefetch.bytes,
            g_one_gate_up_prefetch.submit_us / 1000.0,
            g_one_gate_up_prefetch.residual_wait_us / 1000.0,
            g_one_gate_up_prefetch.h2d_enqueue_us / 1000.0);
#if defined(GGML_MOE_STREAM_HAS_LIBURING)
        if (g_one_gate_up_prefetch.ring_inited) {
            io_uring_queue_exit(&g_one_gate_up_prefetch.ring);
            g_one_gate_up_prefetch.ring_inited = false;
        }
#endif
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

    struct triplet_import_slot {
        void * ptr = nullptr;
        size_t size = 0;
        cudaEvent_t ready = nullptr;
    };
    static triplet_import_slot import_slots[8];
    static int import_next = 0;
    static std::mutex import_mu;

    std::lock_guard<std::mutex> import_lock(import_mu);
    triplet_import_slot & import = import_slots[import_next++ % 8];
    if (import.ready && cudaEventSynchronize(import.ready) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    if (!ensure_host_pinned(import.ptr, import.size, (size_t) entry->nbytes)) return nullptr;
    ds4_triplet_host_handoff_invalidate(import.ptr);
    if (!ds4_triplet_read(*entry, import.ptr)) return nullptr;

    void * inserted = vram_cache_insert(key, import.ptr, (size_t) entry->nbytes, stream, "ds4_triplet");
    if (!inserted) return nullptr;
    if (!import.ready && cudaEventCreateWithFlags(&import.ready, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        cudaStreamSynchronize(stream);
    } else if (cudaEventRecord(import.ready, stream) != cudaSuccess) {
        cudaGetLastError();
        cudaStreamSynchronize(stream);
    }
    ds4_triplet_host_handoff_publish(import.ptr, entry->generation);
    if (cache_state) *cache_state = 4;
    return (char *) inserted + role_offset;
}

static uintptr_t ds4_triplet_role_key(const ds4_triplet_pack_entry & entry, int role) {
    return (uintptr_t) (entry.generation * 4 + (uint64_t) role + 1);
}

static bool ds4_triplet_read_role_pair(const ds4_triplet_pack_entry & entry, void * dst) {
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
    const size_t pair_bytes = (size_t) (g_ds4_triplet_pack.role_bytes[0] +
        g_ds4_triplet_pack.role_bytes[1]);
    bool completed[8] = {};
    bool iouring_attempted = false;
#if defined(GGML_MOE_STREAM_HAS_LIBURING)
    if (g_ds4_triplet_pack.ring_inited && g_ds4_triplet_pack.fd_direct >= 0) {
        iouring_attempted = true;
        const auto submit_start = std::chrono::steady_clock::now();
        int prepared = 0;
        std::lock_guard<std::mutex> io_lock(g_ds4_triplet_pack.io_mu);
        for (; prepared < count; ++prepared) {
            io_uring_sqe * sqe = io_uring_get_sqe(&g_ds4_triplet_pack.ring);
            if (!sqe) break;
            io_uring_prep_read(sqe, g_ds4_triplet_pack.fd_direct,
                destinations[prepared], pair_bytes, (off_t) entries[prepared]->offset);
            io_uring_sqe_set_data64(sqe, (uint64_t) prepared);
        }
        const int submitted = prepared > 0 ? io_uring_submit(&g_ds4_triplet_pack.ring) : 0;
        const auto submit_end = std::chrono::steady_clock::now();
        g_ds4_triplet_pack.iouring_submit_us.fetch_add((uint64_t)
            std::chrono::duration_cast<std::chrono::microseconds>(submit_end - submit_start).count(),
            std::memory_order_relaxed);
        const auto wait_start = submit_end;
        int seen = 0;
        while (seen < submitted) {
            io_uring_cqe * cqe = nullptr;
            if (io_uring_wait_cqe(&g_ds4_triplet_pack.ring, &cqe) != 0 || !cqe) break;
            const int index = (int) io_uring_cqe_get_data64(cqe);
            if (index >= 0 && index < prepared && cqe->res == (int) pair_bytes) {
                completed[index] = true;
                g_ds4_triplet_pack.reads.fetch_add(1, std::memory_order_relaxed);
                g_ds4_triplet_pack.role_pair_reads.fetch_add(1, std::memory_order_relaxed);
                g_ds4_triplet_pack.bytes.fetch_add(pair_bytes, std::memory_order_relaxed);
                g_ds4_triplet_pack.iouring_reads.fetch_add(1, std::memory_order_relaxed);
            }
            io_uring_cqe_seen(&g_ds4_triplet_pack.ring, cqe);
            ++seen;
        }
        const auto wait_end = std::chrono::steady_clock::now();
        g_ds4_triplet_pack.iouring_wait_us.fetch_add((uint64_t)
            std::chrono::duration_cast<std::chrono::microseconds>(wait_end - wait_start).count(),
            std::memory_order_relaxed);
        g_ds4_triplet_pack.iouring_batches.fetch_add(1, std::memory_order_relaxed);
    }
#endif
    bool ok = true;
    for (int i = 0; i < count; ++i) {
        if (completed[i]) continue;
        if (iouring_attempted) {
            g_ds4_triplet_pack.iouring_fallbacks.fetch_add(1, std::memory_order_relaxed);
        }
        if (!ds4_triplet_read_role_pair(*entries[i], destinations[i])) ok = false;
    }
    return ok;
}

static bool ds4_triplet_role_pair_cqe_h2d(
        const ds4_triplet_pack_entry * const * entries,
        void * const * host,
        int count,
        void ** inserted_gate,
        void ** inserted_up) {
#if !defined(GGML_MOE_STREAM_HAS_LIBURING)
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

    bool read_ok[8] = {};
    {
        std::lock_guard<std::mutex> io_lock(g_ds4_triplet_pack.io_mu);
        int prepared = 0;
        for (; prepared < count; ++prepared) {
            io_uring_sqe * sqe = io_uring_get_sqe(&g_ds4_triplet_pack.ring);
            if (!sqe) break;
            io_uring_prep_read(sqe, g_ds4_triplet_pack.fd_direct,
                host[prepared], pair_bytes, (off_t) entries[prepared]->offset);
            io_uring_sqe_set_data64(sqe, (uint64_t) prepared);
        }
        const int submitted = prepared > 0 ? io_uring_submit(&g_ds4_triplet_pack.ring) : 0;
        int seen = 0;
        while (seen < submitted) {
            io_uring_cqe * cqe = nullptr;
            if (io_uring_wait_cqe(&g_ds4_triplet_pack.ring, &cqe) != 0 || !cqe) break;
            const int index = (int) io_uring_cqe_get_data64(cqe);
            if (index >= 0 && index < prepared && cqe->res == (int) pair_bytes) {
                read_ok[index] = true;
                g_ds4_triplet_pack.reads.fetch_add(1, std::memory_order_relaxed);
                g_ds4_triplet_pack.role_pair_reads.fetch_add(1, std::memory_order_relaxed);
                g_ds4_triplet_pack.bytes.fetch_add(pair_bytes, std::memory_order_relaxed);
                g_ds4_triplet_pack.iouring_reads.fetch_add(1, std::memory_order_relaxed);
                g_ds4_triplet_pack.pipe_cqes.fetch_add(1, std::memory_order_relaxed);
            }
            io_uring_cqe_seen(&g_ds4_triplet_pack.ring, cqe);
            ++seen;
        }
    }

    bool ok = true;
    for (int i = 0; i < count; ++i) {
        if (!read_ok[i]) {
            g_ds4_triplet_pack.iouring_fallbacks.fetch_add(1, std::memory_order_relaxed);
            if (!ds4_triplet_read_role_pair(*entries[i], host[i])) {
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

    struct pair_import_slot {
        void * ptr = nullptr;
        size_t size = 0;
        cudaEvent_t ready = nullptr;
    };
    static pair_import_slot imports[8];
    static std::mutex import_mu;
    void * inserted_gate[8] = {};
    void * inserted_up[8] = {};
    if (job_count > 0) {
        std::lock_guard<std::mutex> import_lock(import_mu);
        void * destinations[8] = {};
        for (int job = 0; job < job_count; ++job) {
            pair_import_slot & import = imports[job];
            if (import.ready && cudaEventSynchronize(import.ready) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            if (!ensure_host_pinned(import.ptr, import.size, pair_bytes)) return false;
            destinations[job] = import.ptr;
        }
        const bool cqe_h2d = g_ds4_triplet_pack.pipe_requested &&
            ds4_triplet_role_pair_cqe_h2d(
                jobs, destinations, job_count, inserted_gate, inserted_up);
        if (!cqe_h2d && !ds4_triplet_read_role_pair_batch(jobs, destinations, job_count)) return false;
        cudaStream_t transfer_stream = cqe_h2d ? g_ds4_triplet_pack.pipe_stream : stream;
        for (int job = 0; job < job_count; ++job) {
            pair_import_slot & import = imports[job];
            if (!cqe_h2d) {
                inserted_gate[job] = vram_cache_insert(
                    ds4_triplet_role_key(*jobs[job], 0), import.ptr, role_bytes, stream, "ds4_pair_gate");
                inserted_up[job] = vram_cache_insert(
                    ds4_triplet_role_key(*jobs[job], 1), (char *) import.ptr + role_bytes,
                    role_bytes, stream, "ds4_pair_up");
                if (!inserted_gate[job] || !inserted_up[job]) return false;
            }
            if (!import.ready && cudaEventCreateWithFlags(&import.ready, cudaEventDisableTiming) != cudaSuccess) {
                cudaGetLastError();
                cudaStreamSynchronize(transfer_stream);
            } else if (cudaEventRecord(import.ready, transfer_stream) != cudaSuccess) {
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
#if !defined(GGML_MOE_STREAM_HAS_LIBURING)
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
        if (io_uring_queue_init((unsigned) depth, &g_one_pack_cqe.ring, 0) == 0) {
            g_one_pack_cqe.ring_inited = true;
        }
    }
    g_one_pack_cqe.ready = g_one_pack_cqe.ring_inited && g_one_pack_cqe.stream;
    if (!g_one_pack_cqe.ready) {
        for (int i = 0; i < count; ++i) vram_cache_cancel_descriptor(descriptors[i]);
        return false;
    }

    for (int p = 0; p < plan_count; ++p) {
        one_pack_cqe_import_slot & import = g_one_pack_cqe_imports[p];
        if ((import.ready && cudaEventSynchronize(import.ready) != cudaSuccess) ||
                !ensure_host_pinned(import.ptr, import.size, src0_bytes) ||
                !vram_cache_reserve_triplet(plans[p].key, src0_bytes, g_one_pack_cqe.stream,
                    plans[p].reservation, src0_name)) {
            cudaGetLastError();
            for (int q = 0; q <= p; ++q) {
                if (plans[q].reservation.slot >= 0) vram_cache_fail_triplet(plans[q].reservation);
            }
            for (int i = 0; i < count; ++i) vram_cache_cancel_descriptor(descriptors[i]);
            return false;
        }
        io_uring_sqe * sqe = io_uring_get_sqe(&g_one_pack_cqe.ring);
        if (!sqe) {
            io_uring_queue_exit(&g_one_pack_cqe.ring);
            g_one_pack_cqe.ring_inited = false;
            g_one_pack_cqe.ready = false;
            for (int q = 0; q <= p; ++q) vram_cache_fail_triplet(plans[q].reservation);
            for (int i = 0; i < count; ++i) vram_cache_cancel_descriptor(descriptors[i]);
            return false;
        }
        io_uring_prep_read(sqe, g_one_pack.fd_direct, import.ptr, src0_bytes,
            (off_t) plans[p].entry->offset);
        io_uring_sqe_set_data64(sqe, (uint64_t) p);
    }

    const auto submit_start = std::chrono::steady_clock::now();
    int submitted = 0;
    while (submitted < plan_count) {
        const int rc = io_uring_submit(&g_one_pack_cqe.ring);
        if (rc <= 0) break;
        submitted += rc;
    }
    const auto submit_end = std::chrono::steady_clock::now();
    g_one_pack_cqe.submit_us += (uint64_t)
        std::chrono::duration_cast<std::chrono::microseconds>(submit_end - submit_start).count();

    bool ok = true;
    auto publish_plan = [&](int p) {
        one_pack_cqe_import_slot & import = g_one_pack_cqe_imports[p];
        const auto h2d_start = std::chrono::steady_clock::now();
        const bool copied = plans[p].read_ok && cudaMemcpyAsync(
            plans[p].reservation.dst, import.ptr, src0_bytes,
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
        if (!import.ready && cudaEventCreateWithFlags(&import.ready, cudaEventDisableTiming) != cudaSuccess) {
            cudaGetLastError();
            cudaStreamSynchronize(g_one_pack_cqe.stream);
        } else if (cudaEventRecord(import.ready, g_one_pack_cqe.stream) != cudaSuccess) {
            cudaGetLastError();
            cudaStreamSynchronize(g_one_pack_cqe.stream);
        }
    };

    int reaped = 0;
    for (; reaped < submitted; ++reaped) {
        io_uring_cqe * cqe = nullptr;
        const auto wait_start = std::chrono::steady_clock::now();
        if (io_uring_wait_cqe(&g_one_pack_cqe.ring, &cqe) != 0 || !cqe) break;
        g_one_pack_cqe.wait_us += (uint64_t)
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - wait_start).count();
        const int p = (int) io_uring_cqe_get_data64(cqe);
        if (p >= 0 && p < plan_count && cqe->res == (int) src0_bytes) {
            plans[p].read_ok = true;
            g_one_pack.reads.fetch_add(1, std::memory_order_relaxed);
            g_one_pack.direct_reads.fetch_add(1, std::memory_order_relaxed);
            g_one_pack.bytes.fetch_add(src0_bytes, std::memory_order_relaxed);
            ++g_one_pack_cqe.cqes;
            g_one_pack_cqe.bytes += src0_bytes;
            publish_plan(p);
        }
        io_uring_cqe_seen(&g_one_pack_cqe.ring, cqe);
    }
    if (submitted != plan_count || reaped != submitted) {
        io_uring_queue_exit(&g_one_pack_cqe.ring);
        g_one_pack_cqe.ring_inited = false;
        g_one_pack_cqe.ready = false;
    }

    for (int p = 0; p < plan_count; ++p) {
        if (plans[p].published) continue;
        if (plans[p].read_ok) continue;
        ++g_one_pack_cqe.fallbacks;
        plans[p].read_ok = one_pack_read_entry(
            plans[p].entry, g_one_pack_cqe_imports[p].ptr, src0_bytes);
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
#if !defined(GGML_MOE_STREAM_HAS_LIBURING)
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
        int rc = -1;
        if (request_sqpoll) {
            io_uring_params params = {};
            params.flags = IORING_SETUP_SQPOLL;
            params.sq_thread_idle = 2000;
            rc = io_uring_queue_init_params(8, &g_one_gate_up_prefetch.ring, &params);
            g_one_gate_up_prefetch.sqpoll = rc == 0;
        }
        if (rc != 0) {
            rc = io_uring_queue_init(8, &g_one_gate_up_prefetch.ring, 0);
            g_one_gate_up_prefetch.sqpoll = false;
        }
        if (rc == 0) {
            g_one_gate_up_prefetch.ring_inited = true;
            const int fd = g_one_pack.fd_direct;
            g_one_gate_up_prefetch.fixed_file =
                io_uring_register_files(&g_one_gate_up_prefetch.ring, &fd, 1) == 0;
        }
    }
    return g_one_gate_up_prefetch.ring_inited && g_one_gate_up_prefetch.stream;
#endif
}

static void one_gate_up_prefetch_reap(const char * src0_name) {
#if defined(GGML_MOE_STREAM_HAS_LIBURING)
    if (!one_gate_up_prefetch_enabled() || !src0_name ||
            !std::strstr(src0_name, ".ffn_up_exps.")) return;

    std::lock_guard<std::mutex> lock(g_one_gate_up_prefetch.mu);
    const int pending = g_one_gate_up_prefetch.pending;
    const int submitted = g_one_gate_up_prefetch.submitted;
    if (pending <= 0) return;

    int reaped = 0;
    for (; reaped < submitted; ++reaped) {
        io_uring_cqe * cqe = nullptr;
        const auto wait_start = std::chrono::steady_clock::now();
        if (io_uring_wait_cqe(&g_one_gate_up_prefetch.ring, &cqe) != 0 || !cqe) break;
        g_one_gate_up_prefetch.residual_wait_us += (uint64_t)
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - wait_start).count();
        const int p = (int) io_uring_cqe_get_data64(cqe);
        if (p >= 0 && p < pending && cqe->res == (int) g_one_gate_up_prefetch.jobs[p].entry->nbytes) {
            one_gate_up_prefetch_job & job = g_one_gate_up_prefetch.jobs[p];
            one_pack_cqe_import_slot & import = g_one_gate_up_prefetch_imports[p];
            job.read_ok = true;
            const auto h2d_start = std::chrono::steady_clock::now();
            const int profile_role = one_stage_profile_enabled() ?
                one_stage_profile_role_index(g_one_gate_up_prefetch.tensor) : -1;
            const bool profile_weight = one_decode_weight_profile_start(
                p + 8, profile_role, g_one_gate_up_prefetch.stream);
            const bool copied = cudaMemcpyAsync(
                job.reservation.dst, import.ptr, (size_t) job.entry->nbytes,
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
                if (!import.ready &&
                        cudaEventCreateWithFlags(&import.ready, cudaEventDisableTiming) != cudaSuccess) {
                    cudaGetLastError();
                    cudaStreamSynchronize(g_one_gate_up_prefetch.stream);
                } else if (cudaEventRecord(import.ready, g_one_gate_up_prefetch.stream) != cudaSuccess) {
                    cudaGetLastError();
                    cudaStreamSynchronize(g_one_gate_up_prefetch.stream);
                }
            } else {
                cudaGetLastError();
                if (copied) cudaStreamSynchronize(g_one_gate_up_prefetch.stream);
            }
        }
        io_uring_cqe_seen(&g_one_gate_up_prefetch.ring, cqe);
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
        io_uring_queue_exit(&g_one_gate_up_prefetch.ring);
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
#if defined(GGML_MOE_STREAM_HAS_LIBURING)
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

        one_pack_cqe_import_slot & import = g_one_gate_up_prefetch_imports[job_count];
        if ((import.ready && cudaEventSynchronize(import.ready) != cudaSuccess) ||
                !ensure_host_pinned(import.ptr, import.size, src0_bytes)) {
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
        io_uring_sqe * sqe = io_uring_get_sqe(&g_one_gate_up_prefetch.ring);
        if (!sqe) {
            vram_cache_fail_triplet(job.reservation);
            job = {};
            for (int p = 0; p < job_count; ++p) {
                vram_cache_fail_triplet(g_one_gate_up_prefetch.jobs[p].reservation);
                g_one_gate_up_prefetch.jobs[p] = {};
            }
            io_uring_queue_exit(&g_one_gate_up_prefetch.ring);
            g_one_gate_up_prefetch.ring_inited = false;
            g_one_gate_up_prefetch.sqpoll = false;
            g_one_gate_up_prefetch.fixed_file = false;
            return;
        }
        io_uring_prep_read(sqe, g_one_gate_up_prefetch.fixed_file ? 0 : g_one_pack.fd_direct,
            import.ptr, src0_bytes, (off_t) entry->offset);
        if (g_one_gate_up_prefetch.fixed_file) sqe->flags |= IOSQE_FIXED_FILE;
        io_uring_sqe_set_data64(sqe, (uint64_t) job_count);
        ++job_count;
    }
    if (job_count == 0) return;

    const auto submit_start = std::chrono::steady_clock::now();
    int submitted = 0;
    while (submitted < job_count) {
        const int rc = io_uring_submit(&g_one_gate_up_prefetch.ring);
        if (rc <= 0) break;
        submitted += rc;
    }
    g_one_gate_up_prefetch.submit_us += (uint64_t)
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - submit_start).count();
    if (submitted <= 0) {
        io_uring_queue_exit(&g_one_gate_up_prefetch.ring);
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
    if (ggml_moe::adapter_uses_common()) {
        ggml_cuda_moe_common_stream_one_gate_up_prefetch(
            src0_name, src0_bytes, expert_indices, route_ranks, count);
        return;
    }
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
    if (ggml_moe::adapter_uses_common()) {
        return ggml_cuda_moe_common_stream_one_cache_dev_ptrs_or_insert(
            src0_name, src0_base_data, nb02, src0_bytes, expert_indices,
            route_ranks, count, stream, output_ptrs, cache_states, descriptors);
    }
    if (!src0_name || !src0_base_data || nb02 == 0 ||
            !expert_indices || count <= 0 || count > 8 || !output_ptrs || !cache_states || !descriptors) return false;

    if (!ds4_triplet_requested()) {
        one_gate_up_prefetch_reap(src0_name);
        return one_pack_cqe_h2d_enabled() && one_pack_cqe_h2d_dev_ptrs_or_insert(
            src0_name, src0_base_data, nb02, src0_bytes, expert_indices, count,
            stream, output_ptrs, cache_states, descriptors);
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
    struct batch_import_slot {
        void * ptr = nullptr;
        size_t size = 0;
        cudaEvent_t ready = nullptr;
    };
    static batch_import_slot imports[8];
    static std::mutex import_mu;
    std::lock_guard<std::mutex> import_lock(import_mu);

    void * destinations[8] = {};
    for (int job = 0; job < job_count; ++job) {
        if (imports[job].ready && cudaEventSynchronize(imports[job].ready) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        if (!ensure_host_pinned(imports[job].ptr, imports[job].size, (size_t) jobs[job]->nbytes)) return false;
        ds4_triplet_host_handoff_invalidate(imports[job].ptr);
        destinations[job] = imports[job].ptr;
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
                if (!imports[job].ready &&
                        cudaEventCreateWithFlags(&imports[job].ready, cudaEventDisableTiming) != cudaSuccess) {
                    cudaGetLastError();
                    cudaStreamSynchronize(g_ds4_triplet_pack.pipe_stream);
                } else if (cudaEventRecord(imports[job].ready, g_ds4_triplet_pack.pipe_stream) != cudaSuccess) {
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
            if (!imports[job].ready && cudaEventCreateWithFlags(&imports[job].ready, cudaEventDisableTiming) != cudaSuccess) {
                cudaGetLastError();
                cudaStreamSynchronize(stream);
            } else if (cudaEventRecord(imports[job].ready, stream) != cudaSuccess) {
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
        ds4_triplet_host_handoff_publish(imports[job].ptr, jobs[job]->generation);
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
        if (slot >= g_vcache.n_slots ||
                g_vcache.slot_key[slot] != (uintptr_t) descriptor.object_generation ||
                g_vcache.slot_generation[slot] != descriptor.slot_generation ||
                g_vcache.slot_lease_count[slot] == 0) {
            g_vcache.descriptor_stale_releases.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        --g_vcache.slot_lease_count[slot];
        if (g_vcache.slot_lease_count[slot] == 0 && !stream_synchronized &&
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
    if (ggml_moe::adapter_uses_common()) {
        ggml_cuda_moe_common_stream_one_cache_release_descriptors(
            descriptors, count, stream);
        return;
    }
    vram_cache_release_descriptors(descriptors, count, stream, false);
}

extern "C" void ggml_cuda_moe_stream_one_cache_release_descriptors_synchronized(
        const ggml_moe_expert_descriptor * descriptors,
        int count) {
    if (ggml_moe::adapter_uses_common()) {
        ggml_cuda_moe_common_stream_one_cache_release_descriptors_synchronized(
            descriptors, count);
        return;
    }
    vram_cache_release_descriptors(descriptors, count, nullptr, true);
}

extern "C" bool ggml_cuda_moe_stream_one_cache_contains(
    const char *src0_name,
    const void *src0_data,
    size_t src0_bytes,
    int64_t expert_index)
{
    if (ggml_moe::adapter_uses_common()) {
        return ggml_cuda_moe_common_stream_one_cache_contains(
            src0_name, src0_data, src0_bytes, expert_index);
    }
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
    if (ggml_moe::adapter_uses_common()) {
        return ggml_cuda_moe_common_stream_one_cache_dev_ptr_or_insert(
            src0_name, src0_base_data, nb02, src0_bytes, expert_index, st,
            cache_state);
    }
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
    struct import_slot {
        void *ptr = nullptr;
        size_t sz = 0;
        cudaEvent_t ready = nullptr;
    };
    static import_slot import_slots[8];
    static int import_next = 0;
    static std::mutex import_mu;

    const char *decode_one_pack_env = std::getenv("GGML_MOE_STREAM_DECODE_BATCH_ONE_PACK");
    if (!decode_one_pack_env || !decode_one_pack_env[0] || decode_one_pack_env[0] != '0') {
        if (const one_expert_pack_entry *pack_entry = one_pack_lookup(src0_name, expert_index, src0_bytes)) {
            std::lock_guard<std::mutex> lk(import_mu);
            pack_slot = import_next++ % 8;
            import_slot &slot = import_slots[pack_slot];
            if (slot.ready && cudaEventSynchronize(slot.ready) != cudaSuccess) {
                cudaGetLastError();
                pack_slot = -1;
            } else if (ensure_host_pinned(slot.ptr, slot.sz, src0_bytes) && one_pack_read_entry(pack_entry, slot.ptr, src0_bytes)) {
                one_decode_weight_profile_finalize(pack_slot);
                copy_src = slot.ptr;
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
            import_slot &slot = import_slots[pack_slot];
            if (!slot.ready && cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming) != cudaSuccess) {
                cudaGetLastError();
                cudaStreamSynchronize(st);
            } else if (cudaEventRecord(slot.ready, st) != cudaSuccess) {
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
    if (ggml_moe::adapter_uses_common()) {
        return ggml_cuda_moe_common_stream_one_cache_ready();
    }
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
    if (ggml_moe::adapter_uses_common()) {
        ggml_cuda_moe_common_stream_q80_probe(
            src0_type_int, src0_name, expert_index, src0_data, ne01, ne00,
            nb01, src1_q8_0, src1_q8_0_row_size, src1_ne1, cne1, dst,
            dst_nb1, dst_nb2, rows);
        return;
    }
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
    if (ggml_moe::adapter_uses_common()) {
        ggml_cuda_moe_common_stream_q80_write(
            src0_type_int, src0_name, expert_index, src0_data, ne01, ne00,
            nb01, src1_q8_0, src1_q8_0_row_size, src1_ne1, cne1, dst,
            dst_nb1, dst_nb2, rows);
        return;
    }
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
    ggml_moe::adapter_select_model_shape(n_as, nb01, "q80-hot-batch");
    if (ggml_moe::adapter_uses_common()) {
        ggml_cuda_moe_common_stream_q80_hot_batch_probe(
            src0_type_int, src0_name, n_as, ne01, ne00, nb01, src1_q8_0,
            src1_q8_0_row_size, src1_ne1, src1_f32, src1_nb1, src1_nb2,
            matrix_row_counts, matrix_rows, rows_per_expert, dst, dst_nb1,
            dst_nb2);
        return;
    }
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
    if (ggml_moe::adapter_uses_common()) {
        return ggml_cuda_moe_common_stream_q80_skip(
            src0_type_int, src0_name, expert_index, src0_data, ne01, ne00,
            nb01, src1_q8_0, src1_q8_0_row_size, src1_ne1, cne1, dst,
            dst_nb1, dst_nb2, rows);
    }
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
    if (type != GGML_TYPE_MXFP4) {
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
    // Availability is queried before the first routed tensor exposes the
    // model's expert shape.  Report the compiled path without initializing
    // either process-global adapter; the first shaped call selects one.
    if (!ggml_moe::adapter_is_selected()) {
        return true;
    }
    if (ggml_moe::adapter_uses_common()) {
        return ggml_cuda_moe_common_stream_available();
    }
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
    if (ggml_moe::adapter_uses_common()) {
        return ggml_cuda_moe_common_stream_one(
            src0_type_int, src0_name, expert_index, src0_data, ne01, ne00,
            nb01, src1_f32, src1_nb1, src1_nb2, src1_ne1, cne1,
            src1_q8_1, src1_padded_num_cols, dst, dst_nb1, dst_nb2, rows);
    }
    // printf("k");
    init_once();

    if (!g_avail.load(std::memory_order_acquire)) return false;

    static std::atomic<int> first_accounting{0};
    if (first_accounting.fetch_add(1, std::memory_order_relaxed) == 0) {
        vram_accounting_record("first_one_call_entry", src0_name ? src0_name : "");
    }

    const ggml_type t0 = (ggml_type)src0_type_int;


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
          && ensure_dev(ctx.d_src1, ctx.d_src1_sz, src1_q8_1_bytes)
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
            if (const one_expert_pack_entry * pack_entry = one_pack_lookup(src0_name, expert_index, src0_bytes)) {
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
    if (t0 == GGML_TYPE_MXFP4 &&
            moe_stream_one_experimental_ds4_enabled() &&
            !moe_stream_one_ds4_nonids_enabled()) {
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
    if (ggml_moe::adapter_uses_common()) {
        ggml_cuda_moe_common_stream_sync();
        return;
    }
    if (gate_updown_cosubmit_enabled()) {
        ggml_cuda_moe_stream_batch_flush_gate_updown_cosubmit();
    }
    flush_deferred_scatter();
}

#endif // GGML_CUDA_MOE_STREAM
