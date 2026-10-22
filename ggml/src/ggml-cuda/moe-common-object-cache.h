#pragma once

#include "moe-common-object-transport.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

struct CUstream_st;

namespace ggml_moe_common {

enum class object_host_payload_format : uint8_t {
    raw_object = 0,
    lossless_sidecar = 1,
};

struct object_host_slot {
    void *host = nullptr;
    size_t capacity = 0;
    uintptr_t cache_key = 0;
    uint64_t generation = 0;
    bool pageable = false;
    bool host_registered = false;
    bool host_register_attempted = false;
    bool owns_memory = false;
    object_key marker;
    bool ready = false;
    bool loading = false;
    bool in_use = false;
    bool pending_h2d = false;
    bool h2d_recorded = false;
    bool pending_d2h = false;
    bool delete_after_h2d = false;
    cudaEvent_t h2d_done = nullptr;
    cudaEvent_t d2h_done = nullptr;
    uint64_t used = 0;
    uint64_t read_us = 0;
    uint64_t score = 0;
    uint32_t freq = 0;
    uint8_t origin = 0;
    bool pilot_submitted = false;
    object_host_payload_format payload_format = object_host_payload_format::raw_object;
    size_t payload_bytes = 0;
    size_t payload_transport_bytes = 0;
    uint32_t payload_crc32 = 0;
    uint64_t payload_record_identity = 0;
};

struct object_host_slab {
    object_host_slot storage;
    size_t slot_bytes = 0;
    size_t slot_count = 0;
};

enum class object_host_ready_state : uint8_t {
    empty,
    occupied,
    tombstone,
};

struct object_host_ready_entry {
    object_key key;
    size_t slot = SIZE_MAX;
    object_host_ready_state state = object_host_ready_state::empty;
};

enum class object_host_completion_state : uint8_t {
    idle,
    pending,
    complete,
    failed,
};

struct object_host_completion_counters {
    std::atomic<uint64_t> h2d_begins{0};
    std::atomic<uint64_t> h2d_records{0};
    std::atomic<uint64_t> h2d_completions{0};
    std::atomic<uint64_t> h2d_cancels{0};
    std::atomic<uint64_t> h2d_failures{0};
    std::atomic<uint64_t> h2d_lease_acquires{0};
    std::atomic<uint64_t> h2d_lease_releases{0};
    std::atomic<uint64_t> h2d_stream_waits{0};
};

// Stable identity carried across an SSD read while the cache mutex is not
// held. Publication and abort both verify the slot generation before changing
// readiness, so a stale completion cannot publish into a reused Host slot.
struct object_host_staging_transaction {
    object_key key;
    size_t slot_index = SIZE_MAX;
    size_t required_capacity_bytes = 0;
    uint64_t slot_generation = 0;
    bool active = false;
};

struct object_host_staging_counters {
    std::atomic<uint64_t> begins{0};
    std::atomic<uint64_t> publishes{0};
    std::atomic<uint64_t> aborts{0};
    std::atomic<uint64_t> failures{0};
};

struct object_host_cache {
    object_host_slab slab;
    int32_t ownership_cache_id = -1;
    bool ownership_configured = false;
    size_t max_slots = 0;
    size_t max_resident_bytes = 0;
    size_t resident_capacity_bytes = 0;
    size_t resident_capacity_peak_bytes = 0;
    std::vector<object_host_slot> slots;
    // Fixed-capacity after configure_ownership(). Callers serialize access with
    // the cache's owning mutex; no allocation occurs on lookup/publish/evict.
    std::vector<object_host_ready_entry> ready_index;
    std::vector<size_t> ready_entry_by_slot;
    size_t ready_count = 0;
    size_t ready_tombstones = 0;
    uint64_t ready_index_failures = 0;
    object_host_completion_counters completion;
    object_host_staging_counters staging;
};

enum class object_host_slot_candidate_state : uint8_t {
    excluded,
    empty,
    resident,
};

using object_host_slot_classify_fn = object_host_slot_candidate_state (*)(
        object_host_cache &cache,
        size_t slot_index,
        void *context);
using object_host_slot_score_fn = uint64_t (*)(
        const object_host_cache &cache,
        size_t slot_index,
        void *context);

struct object_host_slot_selection {
    size_t slot_index = SIZE_MAX;
    uint64_t score = 0;
    uint64_t used = 0;
    bool empty = false;
};

struct object_host_slot_selection_policy {
    bool prefer_empty = true;
    bool evaluate_score = false;
    bool select_by_score = false;
};

enum class object_host_admission_decision : uint8_t {
    admit,
    reject_frequency,
    reject_score,
    reject_victim,
};

struct object_host_admission_policy {
    uint64_t min_frequency = 0;
    uint64_t min_score = 0;
    uint64_t victim_margin = 0;
    bool compare_victim = false;
    bool allow_equal_victim = true;
};

struct object_host_layer_quota_policy {
    int32_t incoming_layer = -1;
    size_t incoming_occupancy = 0;
    size_t incoming_quota = 0;
    bool enabled = false;
    bool protect_victim_quota = false;
};

enum class object_host_transfer : uint8_t {
    h2d,
    d2h,
};

enum class object_host_transfer_status : uint8_t {
    idle,
    pending,
    complete,
    error,
};

bool object_host_slot_allocate_memory(
        object_host_slot &slot,
        size_t bytes,
        bool pageable,
        size_t alignment);
bool object_host_slot_register_pageable(object_host_slot &slot);
void object_host_slot_release_memory(object_host_slot &slot);
bool object_host_slab_allocate(
        object_host_slab &slab,
        size_t slot_bytes,
        size_t slot_count,
        bool pageable,
        size_t alignment);
void object_host_slab_release(object_host_slab &slab);
void *object_host_slab_slot_pointer(object_host_slab &slab, size_t slot);
const void *object_host_slab_slot_pointer(const object_host_slab &slab, size_t slot);
bool object_host_slab_bind_slot(
        object_host_slab &slab,
        size_t slot_index,
        object_host_slot &slot);
bool object_host_cache_allocate_slab(
        object_host_cache &cache,
        size_t slot_bytes,
        size_t slot_count,
        bool pageable,
        size_t alignment);
void object_host_cache_release_slab(object_host_cache &cache);
bool object_host_cache_configure_ownership(object_host_cache &cache, int32_t cache_id);
bool object_host_cache_prepare_slot(object_host_cache &cache, size_t slot_index);
bool object_host_cache_publish_slot(object_host_cache &cache, size_t slot_index);
bool object_host_cache_begin_staging(
        object_host_cache &cache,
        size_t slot_index,
        const object_key &key,
        object_host_staging_transaction &transaction,
        size_t required_capacity_bytes = 0);
bool object_host_cache_publish_staging(
        object_host_cache &cache,
        object_host_staging_transaction &transaction,
        uint64_t read_us);
bool object_host_cache_abort_staging(
        object_host_cache &cache,
        object_host_staging_transaction &transaction);
bool object_host_cache_find_ready_slot(
        const object_host_cache &cache,
        const object_key &key,
        size_t *slot_index);
bool object_host_cache_unpublish_ready_slot(object_host_cache &cache, size_t slot_index);
size_t object_host_cache_ready_count(const object_host_cache &cache);
bool object_host_cache_acquire_slot_lease(object_host_cache &cache, size_t slot_index);
bool object_host_cache_release_slot_lease(object_host_cache &cache, size_t slot_index);
bool object_host_cache_begin_h2d(object_host_cache &cache, size_t slot_index);
bool object_host_cache_record_h2d(
        object_host_cache &cache,
        size_t slot_index,
        cudaStream_t stream);
bool object_host_cache_cancel_h2d(
        object_host_cache &cache,
        size_t slot_index,
        cudaStream_t stream);
object_host_completion_state object_host_cache_query_h2d(
        object_host_cache &cache,
        size_t slot_index);
object_host_completion_state object_host_cache_synchronize_h2d(
        object_host_cache &cache,
        size_t slot_index);
bool object_host_cache_stream_wait_h2d(
        object_host_cache &cache,
        size_t slot_index,
        cudaStream_t stream);
bool object_host_cache_invalidate_slot(object_host_cache &cache, size_t slot_index);
bool object_host_cache_select_slot(
        object_host_cache &cache,
        const object_host_slot_selection_policy &policy,
        object_host_slot_classify_fn classify,
        object_host_slot_score_fn score,
        void *context,
        object_host_slot_selection &selection);
object_host_admission_decision object_host_cache_evaluate_admission(
        const object_host_admission_policy &policy,
        uint64_t frequency,
        uint64_t candidate_score,
        bool has_victim,
        uint64_t victim_score);
bool object_host_cache_quota_allows_victim(
        const object_host_layer_quota_policy &policy,
        bool victim_occupied,
        bool victim_ready,
        int32_t victim_layer,
        size_t victim_occupancy,
        size_t victim_quota);
// Cache-owned H2D operations use the cache-level transaction API above so the
// transfer event and ownership lease have the same lifetime. These slot-level
// helpers provide the shared event primitive used directly by D2H operations.
bool object_host_slot_ensure_transfer_event(
        object_host_slot &slot,
        object_host_transfer transfer);
bool object_host_slot_record_transfer(
        object_host_slot &slot,
        object_host_transfer transfer,
        CUstream_st *stream);
object_host_transfer_status object_host_slot_query_transfer(
        const object_host_slot &slot,
        object_host_transfer transfer);
object_host_transfer_status object_host_slot_stream_wait_transfer(
        const object_host_slot &slot,
        object_host_transfer transfer,
        CUstream_st *stream);
void object_host_slot_clear_transfer(
        object_host_slot &slot,
        object_host_transfer transfer);
bool object_host_cache_budget_enabled(const object_host_cache &cache);
bool object_host_cache_budget_fits(
        const object_host_cache &cache,
        const object_host_slot &slot,
        size_t bytes);
void object_host_cache_account_release(
        object_host_cache &cache,
        object_host_slot &slot);
void object_host_cache_account_allocate(
        object_host_cache &cache,
        object_host_slot &slot,
        size_t bytes);

} // namespace ggml_moe_common
