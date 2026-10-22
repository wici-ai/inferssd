#pragma once

#include "moe-expert-descriptor.h"
#include "moe-common-object-engine.h"

#include <cstddef>
#include <cstdint>
#include <limits>

struct CUevent_st;
struct CUstream_st;

namespace ggml_moe_common {

// Non-owning view used by model adapters while their cache mutex is held.
// Optional state arrays let legacy caches adopt the shared descriptor and
// lease rules without changing their admission or eviction policy at once.
struct role_cache_view {
    void *pool = nullptr;
    size_t pool_bytes = 0;
    size_t slot_bytes = 0;
    int32_t slot_count = 0;
    uintptr_t *slot_keys = nullptr;
    uint64_t *slot_generations = nullptr;
    size_t *slot_payload_bytes = nullptr;
    role_slot_marker *slot_ready_markers = nullptr;
    role_slot_state *slot_states = nullptr;
    bool *slot_pending = nullptr;
    bool *slot_filling = nullptr;
    uint32_t *slot_lease_counts = nullptr;
    CUevent_st **slot_ready_events = nullptr;
    CUevent_st **slot_lease_events = nullptr;
    bool *slot_lease_pending = nullptr;
};

// Keep these fields inline in each adapter's cache object. Their position is
// performance-sensitive because model policy arrays may sit between identity
// and lifecycle metadata, so sharing a base class would silently change ABI.
#define GGML_MOE_FIXED_ROLE_CACHE_IDENTITY_FIELDS(CAPACITY) \
    void *pool = nullptr;                                      \
    size_t slot_sz = 0;                                        \
    int32_t n_slots = 0;                                       \
    uintptr_t slot_key[CAPACITY] = {};

#define GGML_MOE_FIXED_ROLE_CACHE_READINESS_FIELDS(CAPACITY) \
    size_t slot_payload_bytes[CAPACITY] = {};                    \
    ggml_moe_common::role_slot_marker slot_ready_marker[CAPACITY] = {}; \
    ggml_moe_common::role_slot_state slot_state[CAPACITY] = {};

#define GGML_MOE_FIXED_ROLE_CACHE_LIFECYCLE_FIELDS(CAPACITY) \
    bool slot_pending[CAPACITY] = {};                           \
    bool slot_filling[CAPACITY] = {};                           \
    CUevent_st *slot_ready[CAPACITY] = {};                      \
    uint64_t slot_generation[CAPACITY] = {};                    \
    GGML_MOE_FIXED_ROLE_CACHE_READINESS_FIELDS(CAPACITY)        \
    CUevent_st *slot_lease_ready[CAPACITY] = {};                \
    bool slot_lease_pending[CAPACITY] = {};                     \
    uint32_t slot_lease_count[CAPACITY] = {};

template <typename Storage>
inline role_cache_view fixed_role_cache_view(Storage *storage, bool include_state = true) {
    role_cache_view result;
    if (!storage) return result;
    result.pool = storage->pool;
    result.slot_bytes = storage->slot_sz;
    result.slot_count = storage->n_slots;
    if (storage->n_slots > 0 &&
            storage->slot_sz <= std::numeric_limits<size_t>::max() /
                static_cast<size_t>(storage->n_slots)) {
        result.pool_bytes = storage->slot_sz * static_cast<size_t>(storage->n_slots);
    }
    result.slot_keys = storage->slot_key;
    result.slot_generations = storage->slot_generation;
    result.slot_pending = storage->slot_pending;
    result.slot_filling = storage->slot_filling;
    result.slot_lease_counts = storage->slot_lease_count;
    result.slot_ready_events = storage->slot_ready;
    result.slot_lease_events = storage->slot_lease_ready;
    result.slot_lease_pending = storage->slot_lease_pending;
    if (include_state) {
        result.slot_payload_bytes = storage->slot_payload_bytes;
        result.slot_ready_markers = storage->slot_ready_marker;
        result.slot_states = storage->slot_state;
    }
    return result;
}

struct role_cache_batch_wait_stats {
    uint64_t representative_syncs = 0;
    uint64_t extra_syncs = 0;
};

struct role_cache_h2d_copy {
    role_cache_view view;
    int32_t slot = -1;
    uintptr_t expected_key = 0;
    uint64_t expected_generation = 0;
    size_t expected_payload_bytes = 0;
    void *device_dst = nullptr;
    const void *host_src = nullptr;
    size_t copy_bytes = 0;
};

struct role_cache_h2d_stats {
    uint64_t copies = 0;
    uint64_t bytes = 0;
};

inline bool role_cache_view_valid(const role_cache_view &view) {
    if (!view.pool || view.slot_bytes == 0 || view.slot_count <= 0 ||
            !view.slot_keys || !view.slot_generations || !view.slot_lease_counts) {
        return false;
    }
    const size_t count = static_cast<size_t>(view.slot_count);
    return count <= std::numeric_limits<size_t>::max() / view.slot_bytes &&
        view.pool_bytes >= count * view.slot_bytes;
}

inline bool role_cache_prepare_fill(
        const role_cache_view &view,
        int32_t slot,
        uintptr_t expected_key,
        uint64_t expected_generation,
        size_t expected_payload_bytes) {
    if (!role_cache_view_valid(view) || slot < 0 || slot >= view.slot_count ||
            expected_key == 0 || expected_generation == 0 || expected_payload_bytes == 0 ||
            !view.slot_payload_bytes || !view.slot_ready_markers || !view.slot_states ||
            !view.slot_pending || !view.slot_filling ||
            view.slot_keys[slot] != expected_key ||
            view.slot_generations[slot] != expected_generation ||
            view.slot_lease_counts[slot] != 0 || view.slot_filling[slot]) {
        return false;
    }
    role_slot_state &state = view.slot_states[slot];
    if (state != role_slot_state::free) {
        if (!role_slot_transition_allowed(state, role_slot_state::free)) return false;
        state = role_slot_state::free;
    }
    if (!role_slot_transition_allowed(state, role_slot_state::reserved)) return false;
    view.slot_payload_bytes[slot] = expected_payload_bytes;
    view.slot_ready_markers[slot] = {};
    state = role_slot_state::reserved;
    view.slot_pending[slot] = false;
    view.slot_filling[slot] = true;
    return true;
}

inline bool role_cache_mark_reading(
        const role_cache_view &view,
        int32_t slot,
        uintptr_t expected_key,
        uint64_t expected_generation,
        size_t expected_payload_bytes) {
    if (!role_cache_view_valid(view) || slot < 0 || slot >= view.slot_count ||
            !view.slot_payload_bytes || !view.slot_states || !view.slot_filling ||
            view.slot_keys[slot] != expected_key ||
            view.slot_generations[slot] != expected_generation ||
            view.slot_payload_bytes[slot] != expected_payload_bytes ||
            !view.slot_filling[slot] ||
            !role_slot_transition_allowed(
                view.slot_states[slot], role_slot_state::reading)) {
        return false;
    }
    view.slot_states[slot] = role_slot_state::reading;
    return true;
}

inline bool role_cache_mark_failed(
        const role_cache_view &view,
        int32_t slot,
        uintptr_t expected_key,
        uint64_t expected_generation) {
    if (!role_cache_view_valid(view) || slot < 0 || slot >= view.slot_count ||
            !view.slot_ready_markers || !view.slot_states || !view.slot_pending ||
            !view.slot_filling || view.slot_keys[slot] != expected_key ||
            view.slot_generations[slot] != expected_generation ||
            !role_slot_transition_allowed(
                view.slot_states[slot], role_slot_state::failed)) {
        return false;
    }
    view.slot_ready_markers[slot] = {};
    view.slot_states[slot] = role_slot_state::failed;
    view.slot_pending[slot] = false;
    view.slot_filling[slot] = false;
    return true;
}

inline bool role_cache_clear_readiness(
        const role_cache_view &view,
        int32_t slot) {
    if (!role_cache_view_valid(view) || slot < 0 || slot >= view.slot_count ||
            !view.slot_payload_bytes || !view.slot_ready_markers || !view.slot_states ||
            !view.slot_pending || !view.slot_filling ||
            view.slot_lease_counts[slot] != 0) {
        return false;
    }
    role_slot_state &state = view.slot_states[slot];
    if (state != role_slot_state::free) {
        if (!role_slot_transition_allowed(state, role_slot_state::free)) return false;
        state = role_slot_state::free;
    }
    view.slot_payload_bytes[slot] = 0;
    view.slot_ready_markers[slot] = {};
    view.slot_pending[slot] = false;
    view.slot_filling[slot] = false;
    return true;
}

inline bool role_cache_slot_resident(
        const role_cache_view &view,
        int32_t slot,
        uintptr_t expected_key,
        size_t required_bytes,
        bool allow_pending) {
    if (!role_cache_view_valid(view) || slot < 0 || slot >= view.slot_count ||
            expected_key == 0 || view.slot_keys[slot] != expected_key ||
            view.slot_generations[slot] == 0) {
        return false;
    }
    if (view.slot_filling && view.slot_filling[slot]) return false;
    if (!allow_pending && view.slot_pending && view.slot_pending[slot]) return false;
    if (view.slot_states && view.slot_states[slot] != role_slot_state::h2d_ready) return false;
    if (view.slot_payload_bytes && view.slot_payload_bytes[slot] < required_bytes) return false;
    return true;
}

inline bool role_cache_slot_ready(
        const role_cache_view &view,
        int32_t slot,
        uintptr_t expected_key,
        size_t required_bytes = 1) {
    return role_cache_slot_resident(view, slot, expected_key, required_bytes, false);
}

inline const void *role_cache_slot_pointer(
        const role_cache_view &view,
        int32_t slot,
        size_t role_offset = 0) {
    if (!role_cache_view_valid(view) || slot < 0 || slot >= view.slot_count ||
            role_offset >= view.slot_bytes) {
        return nullptr;
    }
    const size_t offset = static_cast<size_t>(slot) * view.slot_bytes + role_offset;
    if (offset >= view.pool_bytes) return nullptr;
    return static_cast<const char *>(view.pool) + offset;
}

inline bool role_cache_acquire_descriptor_impl(
        const role_cache_view &view,
        int32_t slot,
        uintptr_t expected_key,
        uint64_t object_generation,
        role kind,
        size_t role_offset,
        bool allow_pending,
        ggml_moe_expert_descriptor &descriptor) {
    descriptor = {};
    descriptor.slot_id = -1;
    descriptor.role = static_cast<int32_t>(kind);
    if (static_cast<size_t>(kind) >= static_cast<size_t>(role::count)) {
        return false;
    }
    const size_t required_bytes = role_offset == std::numeric_limits<size_t>::max() ?
        role_offset : role_offset + 1;
    if (required_bytes == std::numeric_limits<size_t>::max() ||
            !role_cache_slot_resident(
                view, slot, expected_key, required_bytes, allow_pending)) {
        return false;
    }
    const void *role_ptr = role_cache_slot_pointer(view, slot, role_offset);
    if (!role_ptr || view.slot_lease_counts[slot] == UINT32_MAX) return false;

    ++view.slot_lease_counts[slot];
    descriptor.role_ptr = role_ptr;
    descriptor.object_generation = object_generation;
    descriptor.slot_generation = view.slot_generations[slot];
    descriptor.slot_id = slot;
    return true;
}

inline bool role_cache_acquire_descriptor(
        const role_cache_view &view,
        int32_t slot,
        uintptr_t expected_key,
        uint64_t object_generation,
        role kind,
        size_t role_offset,
        ggml_moe_expert_descriptor &descriptor) {
    return role_cache_acquire_descriptor_impl(
        view, slot, expected_key, object_generation, kind, role_offset,
        false, descriptor);
}

inline bool role_cache_release_descriptor(
        const role_cache_view &view,
        uintptr_t expected_key,
        const ggml_moe_expert_descriptor &descriptor,
        bool *last_lease = nullptr) {
    if (last_lease) *last_lease = false;
    const int32_t slot = descriptor.slot_id;
    if (!role_cache_view_valid(view) || slot < 0 || slot >= view.slot_count ||
            expected_key == 0 || view.slot_keys[slot] != expected_key ||
            view.slot_generations[slot] != descriptor.slot_generation ||
            view.slot_lease_counts[slot] == 0) {
        return false;
    }
    --view.slot_lease_counts[slot];
    if (last_lease) *last_lease = view.slot_lease_counts[slot] == 0;
    return true;
}

bool role_cache_record_lease_event(
        const role_cache_view &view,
        int32_t slot,
        CUstream_st *stream);
bool role_cache_acquire_descriptor_for_stream(
        const role_cache_view &view,
        int32_t slot,
        uintptr_t expected_key,
        uint64_t object_generation,
        role kind,
        size_t role_offset,
        CUstream_st *stream,
        ggml_moe_expert_descriptor &descriptor);
bool role_cache_stream_wait_lease_event(
        const role_cache_view &view,
        int32_t slot,
        CUstream_st *stream);
bool role_cache_synchronize_lease_event(
        const role_cache_view &view,
        int32_t slot);
bool role_cache_stream_wait_ready_event(
        const role_cache_view &view,
        int32_t slot,
        CUstream_st *stream);
bool role_cache_synchronize_ready_event(
        const role_cache_view &view,
        int32_t slot);
bool role_cache_synchronize_ready_events(
        const role_cache_view &view,
        const int32_t *slots,
        size_t count,
        role_cache_batch_wait_stats *stats = nullptr);
bool role_cache_ensure_ready_event(
        const role_cache_view &view,
        int32_t slot);
bool role_cache_record_ready_event(
        const role_cache_view &view,
        int32_t slot,
        CUstream_st *stream);
bool role_cache_publish_ready(
        const role_cache_view &view,
        int32_t slot,
        uintptr_t expected_key,
        uint64_t expected_generation,
        size_t expected_payload_bytes,
        CUstream_st *stream);
bool role_cache_copy_h2d_batch_async(
        const role_cache_h2d_copy *copies,
        size_t count,
        CUstream_st *stream,
        role_cache_h2d_stats *stats = nullptr);

} // namespace ggml_moe_common
