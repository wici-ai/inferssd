#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace ggml_moe_common {

enum class role : uint8_t {
    gate = 0,
    up = 1,
    down = 2,
    count = 3,
};

constexpr uint8_t role_bit(role value) {
    return uint8_t{1} << static_cast<uint8_t>(value);
}

enum class role_slot_state : uint8_t {
    free,
    reserved,
    reading,
    h2d_ready,
    failed,
};

struct role_slot_marker {
    uintptr_t key = 0;
    uint64_t generation = 0;
    size_t bytes = 0;
};

struct object_key {
    int32_t source_idx = -1;
    int32_t layer_idx = -1;
    int32_t expert_idx = -1;
    uint64_t manifest_generation = 0;
    uint64_t offset = 0;
    size_t bytes = 0;
    // Zero means the whole pack uses manifest_generation (the Kimi layout).
    // Packs such as DeepSeek may additionally version each object independently.
    uint64_t object_generation = 0;
};

struct role_extent {
    role kind = role::gate;
    uint64_t relative_offset = 0;
    size_t bytes = 0;
};

constexpr size_t max_object_shape_classes = 8;
constexpr size_t max_role_cache_classes = 8;
constexpr size_t max_host_cache_classes = 8;

struct object_shape {
    size_t object_bytes = 0;
    role_extent roles[3] = {};
};

struct object_layout {
    char model[32] = {};
    int32_t source_idx = -1;
    int32_t first_layer = 0;
    uint32_t layer_count = 0;
    int32_t first_expert = 0;
    uint32_t experts_per_layer = 0;
    uint32_t top_k = 0;
    uint64_t object_count = 0;
    uint64_t manifest_generation = 0;
    uint8_t shape_count = 0;
    object_shape shapes[max_object_shape_classes] = {};
    // Required when shape_count > 1. The engine copies this direct-indexed map
    // during configure, so the caller only needs to keep it alive for the call.
    const uint8_t *object_shape_indices = nullptr;
    size_t object_shape_index_count = 0;
};

struct object_request {
    object_key key;
    role_extent roles[3] = {};
    uint8_t required_role_mask = 0;
    uint8_t role_count = 0;
    bool speculative = false;
};

struct object_readiness_timing {
    std::chrono::steady_clock::time_point schedule_start;
    std::chrono::steady_clock::time_point build_done;
    std::chrono::steady_clock::time_point queued_at;
    std::chrono::steady_clock::time_point worker_start;
    std::chrono::steady_clock::time_point initial_submit_start;
    std::chrono::steady_clock::time_point initial_submit_done;
    std::chrono::steady_clock::time_point first_cqe;
    std::chrono::steady_clock::time_point last_cqe;
    std::chrono::steady_clock::time_point first_h2d_start;
    std::chrono::steady_clock::time_point last_h2d_done;
    std::chrono::steady_clock::time_point provider_done;
    std::chrono::steady_clock::time_point active_wait_start;
    std::chrono::steady_clock::time_point active_wait_done;
    uint64_t input_jobs = 0;
    uint64_t plans = 0;
    uint64_t read_groups = 0;
    uint64_t depth = 0;
    uint64_t refill_batch = 0;
    uint64_t initial_submitted = 0;
    uint64_t refill_submit_calls = 0;
    uint64_t refill_submitted = 0;
    uint64_t wait_calls = 0;
    uint64_t wait_with_inflight = 0;
    uint64_t wait_no_inflight = 0;
    uint64_t inflight_sum = 0;
    uint64_t inflight_samples = 0;
    uint64_t max_inflight = 0;
    uint64_t deferred_host_hits = 0;
    uint64_t deferred_host_roles = 0;
    uint64_t object_h2d_batches = 0;
    uint64_t object_h2d_objects = 0;
    uint64_t object_h2d_roles = 0;
    uint64_t object_h2d_bytes = 0;
    uint64_t inline_h2d_us = 0;
    uint64_t read_slot_wait_us = 0;
    uint64_t h2d_slot_wait_us = 0;
    uint64_t queue_depth_at_submit = 0;
};

using object_worker_init_fn = bool (*)(void *user_data);
using object_worker_process_fn = bool (*)(void *user_data, void *context, uint32_t kind);

struct object_worker_provider {
    void *user_data = nullptr;
    object_worker_init_fn init = nullptr;
    object_worker_process_fn process = nullptr;
    const char *cpu_affinity_env = nullptr;
    const char *legacy_cpu_affinity_env = nullptr;
    const char *worker_role = nullptr;
};

// Fixed-capacity transaction and residency record shared by model adapters.
// Host ownership is generation- and lease-aware. Role snapshots are published
// at completion and synchronized with role-slot eviction.
struct object_record {
    uint64_t generation = 0;
    uint64_t pack_offset = 0;
    uint64_t inflight_transaction = 0;
    int32_t host_cache_id = -1;
    int32_t host_slot = -1;
    uint64_t host_slot_generation = 0;
    uint32_t host_lease_count = 0;
    uint32_t resident_mask = 0;
    uint32_t pending_mask = 0;
    uint32_t ready_mask = 0;
    uint64_t last_transaction = 0;
    bool last_transaction_succeeded = false;
    int32_t role_cache_id[3] = {-1, -1, -1};
    int32_t role_slot[3] = {-1, -1, -1};
    uint64_t role_slot_generation[3] = {};
    uint32_t role_lease_count[3] = {};
};

enum class object_transaction_acquire_status : uint8_t {
    invalid,
    owner,
    joined,
};

enum class object_transaction_wait_status : uint8_t {
    invalid,
    ready,
    completed,
    failed,
    timeout,
};

struct object_transaction_ticket {
    object_key key;
    uint64_t transaction = 0;
    uint8_t requested_role_mask = 0;
    bool owner = false;
};

// Adapter snapshot published after a physical transaction has made selected
// role slots consumable. valid_role_mask identifies the entries being
// replaced; roles outside that mask retain their previous residency.
struct object_residency_update {
    uint8_t valid_role_mask = 0;
    uint8_t resident_role_mask = 0;
    int32_t role_cache_id[3] = {-1, -1, -1};
    int32_t role_slot[3] = {-1, -1, -1};
    uint64_t role_slot_generation[3] = {};
    uint32_t role_lease_count[3] = {};
};

inline bool object_key_matches(const object_key &current, const object_key &expected) {
    return current.source_idx == expected.source_idx &&
        current.layer_idx == expected.layer_idx &&
        current.expert_idx == expected.expert_idx &&
        current.manifest_generation == expected.manifest_generation &&
        current.offset == expected.offset &&
        current.bytes == expected.bytes &&
        current.object_generation == expected.object_generation;
}

inline uint64_t object_key_effective_generation(const object_key &key) {
    return key.object_generation != 0 ? key.object_generation : key.manifest_generation;
}

inline bool role_slot_marker_matches(
        const role_slot_marker &current,
        const role_slot_marker &expected) {
    return current.key == expected.key &&
        current.generation == expected.generation &&
        current.bytes == expected.bytes;
}

inline bool role_slot_transition_allowed(role_slot_state from, role_slot_state to) {
    switch (from) {
        case role_slot_state::free:
            return to == role_slot_state::reserved;
        case role_slot_state::reserved:
            return to == role_slot_state::reading ||
                to == role_slot_state::failed ||
                to == role_slot_state::free;
        case role_slot_state::reading:
            return to == role_slot_state::h2d_ready ||
                to == role_slot_state::failed;
        case role_slot_state::h2d_ready:
        case role_slot_state::failed:
            return to == role_slot_state::free;
    }
    return false;
}

bool object_engine_enabled();
bool object_engine_enforcing();
bool object_engine_configure(const object_layout &layout);
bool object_engine_configure_host_cache(int32_t cache_id, size_t slot_count);
bool object_engine_publish_host_slot(
        const object_key &key,
        int32_t cache_id,
        int32_t slot_id,
        uint64_t slot_generation);
bool object_engine_acquire_host_slot(
        const object_key &key,
        int32_t cache_id,
        int32_t slot_id,
        uint64_t slot_generation);
bool object_engine_release_host_slot(
        const object_key &key,
        int32_t cache_id,
        int32_t slot_id,
        uint64_t slot_generation);
bool object_engine_invalidate_host_slot(
        int32_t cache_id,
        int32_t slot_id,
        uint64_t slot_generation);
bool object_engine_configure_role_cache(int32_t cache_id, size_t slot_count);
bool object_engine_invalidate_role_slot(
        int32_t cache_id,
        int32_t slot_id,
        uint64_t slot_generation);
bool object_engine_begin_batch(const object_request *requests, size_t count);
object_transaction_acquire_status object_engine_acquire_transaction(
        const object_request &request,
        object_transaction_ticket &ticket);
bool object_engine_publish_transaction_roles(
        const object_transaction_ticket &ticket,
        uint8_t ready_role_mask);
object_transaction_wait_status object_engine_wait_transaction(
        const object_transaction_ticket &ticket,
        uint8_t required_role_mask,
        uint64_t timeout_us);
bool object_engine_complete_batch(const object_request *requests, size_t count, bool success);
bool object_engine_complete_batch_with_residency(
        const object_request *requests,
        const object_residency_update *updates,
        size_t count,
        bool success);
bool object_engine_get_record(const object_key &key, object_record &record);
bool object_engine_enqueue_batch(
        void *context,
        const object_request *requests,
        size_t count,
        bool speculative,
        size_t capacity);
bool object_engine_enqueue_work(
        void *context,
        uint32_t kind,
        uint8_t priority,
        size_t capacity);
void *object_engine_take_batch();
bool object_engine_has_queued_batch();
size_t object_engine_queued_batches();
bool object_engine_start_worker(const object_worker_provider &provider);
void object_engine_stop_worker();
bool object_engine_worker_active();

} // namespace ggml_moe_common
