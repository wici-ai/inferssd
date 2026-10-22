#include "moe-common-object-cache.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <limits>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace ggml_moe_common {

void object_host_slot_release_memory(object_host_slot &slot) {
    if (slot.host_registered && slot.host) {
        (void)cudaHostUnregister(slot.host);
    }
    if (slot.owns_memory && slot.host) {
        if (slot.pageable) {
#if defined(_WIN32)
            _aligned_free(slot.host);
#else
            std::free(slot.host);
#endif
        } else {
            (void)cudaFreeHost(slot.host);
        }
    }
    slot.host = nullptr;
    slot.capacity = 0;
    slot.pageable = false;
    slot.host_registered = false;
    slot.host_register_attempted = false;
    slot.owns_memory = false;
}

bool object_host_slot_allocate_memory(
        object_host_slot &slot,
        size_t bytes,
        bool pageable,
        size_t alignment) {
    if (slot.host || bytes == 0 || alignment < sizeof(void *) ||
            (alignment & (alignment - 1)) != 0) {
        return false;
    }

    void *host = nullptr;
    if (pageable) {
#if defined(_WIN32)
        host = _aligned_malloc(bytes, alignment);
        if (!host) return false;
#else
        if (posix_memalign(&host, alignment, bytes) != 0 || !host) return false;
#endif
    } else if (cudaHostAlloc(&host, bytes, cudaHostAllocDefault) != cudaSuccess) {
        return false;
    }

    slot.host = host;
    slot.capacity = bytes;
    slot.pageable = pageable;
    slot.host_registered = false;
    slot.host_register_attempted = false;
    slot.owns_memory = true;
    return true;
}

bool object_host_slot_register_pageable(object_host_slot &slot) {
    if (!slot.owns_memory || !slot.pageable || !slot.host || slot.capacity == 0 || slot.host_registered ||
            slot.host_register_attempted) {
        return false;
    }
    slot.host_register_attempted = true;
    if (cudaHostRegister(slot.host, slot.capacity, cudaHostRegisterDefault) != cudaSuccess) {
        return false;
    }
    slot.host_registered = true;
    return true;
}

bool object_host_slab_allocate(
        object_host_slab &slab,
        size_t slot_bytes,
        size_t slot_count,
        bool pageable,
        size_t alignment) {
    if (slab.storage.host || slab.slot_bytes != 0 || slab.slot_count != 0 ||
            slot_bytes == 0 || slot_count == 0 ||
            slot_count > std::numeric_limits<size_t>::max() / slot_bytes) {
        return false;
    }
    if (!object_host_slot_allocate_memory(
            slab.storage, slot_bytes * slot_count, pageable, alignment)) {
        return false;
    }
    slab.slot_bytes = slot_bytes;
    slab.slot_count = slot_count;
    return true;
}

void object_host_slab_release(object_host_slab &slab) {
    object_host_slot_release_memory(slab.storage);
    slab.slot_bytes = 0;
    slab.slot_count = 0;
}

void *object_host_slab_slot_pointer(object_host_slab &slab, size_t slot) {
    if (!slab.storage.host || slab.slot_bytes == 0 || slot >= slab.slot_count ||
            slab.slot_bytes > slab.storage.capacity ||
            slot > (slab.storage.capacity - slab.slot_bytes) / slab.slot_bytes) {
        return nullptr;
    }
    return static_cast<char *>(slab.storage.host) + slot * slab.slot_bytes;
}

const void *object_host_slab_slot_pointer(const object_host_slab &slab, size_t slot) {
    if (!slab.storage.host || slab.slot_bytes == 0 || slot >= slab.slot_count ||
            slab.slot_bytes > slab.storage.capacity ||
            slot > (slab.storage.capacity - slab.slot_bytes) / slab.slot_bytes) {
        return nullptr;
    }
    return static_cast<const char *>(slab.storage.host) + slot * slab.slot_bytes;
}

bool object_host_slab_bind_slot(
        object_host_slab &slab,
        size_t slot_index,
        object_host_slot &slot) {
    if (slot.host || slot.owns_memory) return false;
    void *pointer = object_host_slab_slot_pointer(slab, slot_index);
    if (!pointer) return false;
    slot.host = pointer;
    slot.capacity = slab.slot_bytes;
    slot.pageable = slab.storage.pageable;
    slot.host_registered = false;
    slot.host_register_attempted = false;
    slot.owns_memory = false;
    return true;
}

bool object_host_cache_allocate_slab(
        object_host_cache &cache,
        size_t slot_bytes,
        size_t slot_count,
        bool pageable,
        size_t alignment) {
    if (!cache.slots.empty() || cache.slab.storage.host || cache.max_slots != 0 ||
            cache.resident_capacity_bytes != 0) {
        return false;
    }
    if (!object_host_slab_allocate(
            cache.slab, slot_bytes, slot_count, pageable, alignment)) {
        return false;
    }
    cache.slots.resize(slot_count);
    for (size_t i = 0; i < slot_count; ++i) {
        if (!object_host_slab_bind_slot(cache.slab, i, cache.slots[i])) {
            object_host_cache_release_slab(cache);
            return false;
        }
    }
    cache.max_slots = slot_count;
    cache.max_resident_bytes = cache.slab.storage.capacity;
    cache.resident_capacity_bytes = cache.slab.storage.capacity;
    cache.resident_capacity_peak_bytes = cache.slab.storage.capacity;
    return true;
}

void object_host_cache_release_slab(object_host_cache &cache) {
    for (object_host_slot &slot : cache.slots) {
        if (slot.h2d_done) {
            (void)cudaEventDestroy(slot.h2d_done);
            slot.h2d_done = nullptr;
        }
        if (slot.d2h_done) {
            (void)cudaEventDestroy(slot.d2h_done);
            slot.d2h_done = nullptr;
        }
        object_host_slot_release_memory(slot);
    }
    cache.slots.clear();
    object_host_slab_release(cache.slab);
    cache.ownership_cache_id = -1;
    cache.ownership_configured = false;
    cache.ready_index.clear();
    cache.ready_entry_by_slot.clear();
    cache.ready_count = 0;
    cache.ready_tombstones = 0;
    cache.ready_index_failures = 0;
    cache.max_slots = 0;
    cache.max_resident_bytes = 0;
    cache.resident_capacity_bytes = 0;
    cache.resident_capacity_peak_bytes = 0;
    cache.staging.begins.store(0, std::memory_order_relaxed);
    cache.staging.publishes.store(0, std::memory_order_relaxed);
    cache.staging.aborts.store(0, std::memory_order_relaxed);
    cache.staging.failures.store(0, std::memory_order_relaxed);
}

static bool object_host_cache_ownership_result(bool result) {
    return result || !object_engine_enforcing();
}

static uint64_t object_host_key_hash(const object_key &key) {
    const auto mix = [](uint64_t value) {
        value ^= value >> 30;
        value *= UINT64_C(0xbf58476d1ce4e5b9);
        value ^= value >> 27;
        value *= UINT64_C(0x94d049bb133111eb);
        return value ^ (value >> 31);
    };
    uint64_t hash = mix(static_cast<uint32_t>(key.source_idx));
    hash ^= mix(static_cast<uint32_t>(key.layer_idx) + UINT64_C(0x9e3779b97f4a7c15));
    hash ^= mix(static_cast<uint32_t>(key.expert_idx) + UINT64_C(0x243f6a8885a308d3));
    hash ^= mix(key.manifest_generation);
    hash ^= mix(key.offset);
    hash ^= mix(static_cast<uint64_t>(key.bytes));
    hash ^= mix(key.object_generation);
    return mix(hash);
}

static bool object_host_cache_configure_ready_index(object_host_cache &cache) {
    if (cache.slots.empty()) return false;
    size_t capacity = 8;
    if (cache.slots.size() > std::numeric_limits<size_t>::max() / 4) return false;
    const size_t requested = cache.slots.size() * 4;
    while (capacity < requested) {
        if (capacity > std::numeric_limits<size_t>::max() / 2) return false;
        capacity *= 2;
    }
    if (!cache.ready_index.empty()) {
        return cache.ready_index.size() == capacity &&
            cache.ready_entry_by_slot.size() == cache.slots.size();
    }
    cache.ready_index.resize(capacity);
    cache.ready_entry_by_slot.assign(cache.slots.size(), SIZE_MAX);
    return true;
}

static size_t object_host_cache_find_ready_entry(
        const object_host_cache &cache,
        const object_key &key) {
    if (cache.ready_index.empty()) return SIZE_MAX;
    const size_t mask = cache.ready_index.size() - 1;
    size_t entry_index = static_cast<size_t>(object_host_key_hash(key)) & mask;
    for (size_t probe = 0; probe < cache.ready_index.size(); ++probe) {
        const object_host_ready_entry &entry = cache.ready_index[entry_index];
        if (entry.state == object_host_ready_state::empty) return SIZE_MAX;
        if (entry.state == object_host_ready_state::occupied &&
                object_key_matches(entry.key, key)) {
            return entry_index;
        }
        entry_index = (entry_index + 1) & mask;
    }
    return SIZE_MAX;
}

static bool object_host_cache_insert_ready_entry(
        object_host_cache &cache,
        const object_key &key,
        size_t slot_index);

static bool object_host_cache_rebuild_ready_index(object_host_cache &cache) {
    if (cache.ready_index.empty() ||
            cache.ready_entry_by_slot.size() != cache.slots.size()) {
        return false;
    }
    for (object_host_ready_entry &entry : cache.ready_index) entry = {};
    cache.ready_count = 0;
    cache.ready_tombstones = 0;
    for (size_t slot_index = 0; slot_index < cache.slots.size(); ++slot_index) {
        if (cache.ready_entry_by_slot[slot_index] == SIZE_MAX) continue;
        cache.ready_entry_by_slot[slot_index] = SIZE_MAX;
        if (!object_host_cache_insert_ready_entry(
                    cache, cache.slots[slot_index].marker, slot_index)) {
            return false;
        }
    }
    return true;
}

static bool object_host_cache_insert_ready_entry(
        object_host_cache &cache,
        const object_key &key,
        size_t slot_index) {
    if (slot_index >= cache.slots.size() || cache.ready_index.empty() ||
            cache.ready_entry_by_slot.size() != cache.slots.size()) {
        return false;
    }
    const size_t current_entry = cache.ready_entry_by_slot[slot_index];
    if (current_entry != SIZE_MAX) {
        return current_entry < cache.ready_index.size() &&
            cache.ready_index[current_entry].state == object_host_ready_state::occupied &&
            cache.ready_index[current_entry].slot == slot_index &&
            object_key_matches(cache.ready_index[current_entry].key, key);
    }
    const size_t existing = object_host_cache_find_ready_entry(cache, key);
    if (existing != SIZE_MAX) {
        return cache.ready_index[existing].slot == slot_index;
    }
    const size_t mask = cache.ready_index.size() - 1;
    size_t entry_index = static_cast<size_t>(object_host_key_hash(key)) & mask;
    size_t first_tombstone = SIZE_MAX;
    for (size_t probe = 0; probe < cache.ready_index.size(); ++probe) {
        object_host_ready_entry &entry = cache.ready_index[entry_index];
        if (entry.state == object_host_ready_state::tombstone && first_tombstone == SIZE_MAX) {
            first_tombstone = entry_index;
        } else if (entry.state == object_host_ready_state::empty) {
            const size_t selected = first_tombstone != SIZE_MAX ? first_tombstone : entry_index;
            object_host_ready_entry &target = cache.ready_index[selected];
            if (target.state == object_host_ready_state::tombstone) --cache.ready_tombstones;
            target.key = key;
            target.slot = slot_index;
            target.state = object_host_ready_state::occupied;
            cache.ready_entry_by_slot[slot_index] = selected;
            ++cache.ready_count;
            return true;
        }
        entry_index = (entry_index + 1) & mask;
    }
    if (first_tombstone != SIZE_MAX) {
        object_host_ready_entry &target = cache.ready_index[first_tombstone];
        target.key = key;
        target.slot = slot_index;
        target.state = object_host_ready_state::occupied;
        cache.ready_entry_by_slot[slot_index] = first_tombstone;
        --cache.ready_tombstones;
        ++cache.ready_count;
        return true;
    }
    return false;
}

bool object_host_cache_configure_ownership(object_host_cache &cache, int32_t cache_id) {
    if (cache_id < 0 || static_cast<size_t>(cache_id) >= max_host_cache_classes ||
            cache.slots.empty()) {
        return false;
    }
    if (cache.ownership_configured && cache.ownership_cache_id != cache_id) return false;
    if (!object_host_cache_configure_ready_index(cache)) return false;
    const bool configured = object_engine_configure_host_cache(cache_id, cache.slots.size());
    if (!object_host_cache_ownership_result(configured)) return false;
    cache.ownership_cache_id = cache_id;
    cache.ownership_configured = true;
    return true;
}

static bool object_host_cache_slot_valid(const object_host_cache &cache, size_t slot_index) {
    return cache.ownership_configured && cache.ownership_cache_id >= 0 &&
        slot_index < cache.slots.size();
}

bool object_host_cache_find_ready_slot(
        const object_host_cache &cache,
        const object_key &key,
        size_t *slot_index) {
    const size_t entry_index = object_host_cache_find_ready_entry(cache, key);
    if (entry_index == SIZE_MAX) return false;
    const object_host_ready_entry &entry = cache.ready_index[entry_index];
    if (entry.slot >= cache.slots.size()) return false;
    const object_host_slot &slot = cache.slots[entry.slot];
    if (!slot.ready || slot.generation == 0 ||
            !object_key_matches(slot.marker, key)) {
        return false;
    }
    if (slot_index) *slot_index = entry.slot;
    return true;
}

bool object_host_cache_unpublish_ready_slot(object_host_cache &cache, size_t slot_index) {
    if (slot_index >= cache.ready_entry_by_slot.size()) return false;
    const size_t entry_index = cache.ready_entry_by_slot[slot_index];
    if (entry_index == SIZE_MAX) return true;
    if (entry_index >= cache.ready_index.size()) {
        ++cache.ready_index_failures;
        return false;
    }
    object_host_ready_entry &entry = cache.ready_index[entry_index];
    if (entry.state != object_host_ready_state::occupied || entry.slot != slot_index) {
        ++cache.ready_index_failures;
        return false;
    }
    entry = {};
    entry.state = object_host_ready_state::tombstone;
    cache.ready_entry_by_slot[slot_index] = SIZE_MAX;
    --cache.ready_count;
    ++cache.ready_tombstones;
    if (cache.ready_tombstones > cache.ready_index.size() / 2 &&
            !object_host_cache_rebuild_ready_index(cache)) {
        ++cache.ready_index_failures;
        return false;
    }
    return true;
}

size_t object_host_cache_ready_count(const object_host_cache &cache) {
    return cache.ready_count;
}

bool object_host_cache_invalidate_slot(object_host_cache &cache, size_t slot_index) {
    if (!object_host_cache_slot_valid(cache, slot_index)) return !object_engine_enforcing();
    const object_host_slot &slot = cache.slots[slot_index];
    if (slot.pending_h2d) return false;
    const bool invalidated = slot.generation == 0 || object_engine_invalidate_host_slot(
        cache.ownership_cache_id,
        static_cast<int32_t>(slot_index),
        slot.generation);
    if (!invalidated && object_engine_enforcing()) return false;
    const bool unpublished = object_host_cache_unpublish_ready_slot(cache, slot_index);
    return unpublished && object_host_cache_ownership_result(invalidated);
}

bool object_host_cache_prepare_slot(object_host_cache &cache, size_t slot_index) {
    if (!object_host_cache_slot_valid(cache, slot_index) ||
            !object_host_cache_invalidate_slot(cache, slot_index)) {
        return !object_engine_enforcing();
    }
    object_host_slot &slot = cache.slots[slot_index];
    if (++slot.generation == 0) ++slot.generation;
    return true;
}

bool object_host_cache_publish_slot(object_host_cache &cache, size_t slot_index) {
    if (!object_host_cache_slot_valid(cache, slot_index)) return !object_engine_enforcing();
    const object_host_slot &slot = cache.slots[slot_index];
    const bool published = slot.generation != 0 && object_engine_publish_host_slot(
            slot.marker,
            cache.ownership_cache_id,
            static_cast<int32_t>(slot_index),
            slot.generation);
    if (!published && object_engine_enforcing()) return false;
    const bool indexed = slot.generation != 0 &&
        object_host_cache_insert_ready_entry(cache, slot.marker, slot_index);
    if (!indexed) ++cache.ready_index_failures;
    return indexed && object_host_cache_ownership_result(published);
}

static bool object_host_staging_key_valid(const object_key &key) {
    return key.source_idx >= 0 && key.layer_idx >= 0 && key.expert_idx >= 0 &&
        key.bytes != 0 && object_key_effective_generation(key) != 0;
}

static bool object_host_staging_transaction_matches(
        const object_host_cache &cache,
        const object_host_staging_transaction &transaction) {
    if (!transaction.active ||
            !object_host_cache_slot_valid(cache, transaction.slot_index)) {
        return false;
    }
    const object_host_slot &slot = cache.slots[transaction.slot_index];
    return slot.generation == transaction.slot_generation &&
        object_key_matches(slot.marker, transaction.key);
}

bool object_host_cache_begin_staging(
        object_host_cache &cache,
        size_t slot_index,
        const object_key &key,
        object_host_staging_transaction &transaction,
        size_t required_capacity_bytes) {
    if (transaction.active || !object_host_staging_key_valid(key) ||
            !object_host_cache_slot_valid(cache, slot_index)) {
        ++cache.staging.failures;
        return false;
    }
    object_host_slot &slot = cache.slots[slot_index];
    if (slot.loading || slot.in_use || slot.pending_h2d || slot.pending_d2h ||
            !object_host_cache_prepare_slot(cache, slot_index)) {
        ++cache.staging.failures;
        return false;
    }

    slot.marker = key;
    slot.ready = false;
    slot.loading = true;
    slot.in_use = false;
    slot.pending_h2d = false;
    slot.h2d_recorded = false;
    slot.pending_d2h = false;
    slot.delete_after_h2d = false;
    slot.read_us = 0;
    slot.payload_format = object_host_payload_format::raw_object;
    slot.payload_bytes = key.bytes;
    slot.payload_transport_bytes = key.bytes;
    slot.payload_crc32 = 0;
    slot.payload_record_identity = 0;

    transaction.key = key;
    transaction.slot_index = slot_index;
    transaction.required_capacity_bytes = required_capacity_bytes > 0 ?
        required_capacity_bytes : key.bytes;
    transaction.slot_generation = slot.generation;
    transaction.active = true;
    ++cache.staging.begins;
    return true;
}

bool object_host_cache_publish_staging(
        object_host_cache &cache,
        object_host_staging_transaction &transaction,
        uint64_t read_us) {
    if (!object_host_staging_transaction_matches(cache, transaction)) {
        ++cache.staging.failures;
        return false;
    }
    object_host_slot &slot = cache.slots[transaction.slot_index];
    if (!slot.loading || slot.ready || slot.in_use || slot.pending_h2d ||
            slot.pending_d2h || !slot.host ||
            slot.capacity < transaction.required_capacity_bytes ||
            !object_host_cache_publish_slot(cache, transaction.slot_index)) {
        ++cache.staging.failures;
        return false;
    }
    slot.read_us = read_us;
    slot.loading = false;
    slot.ready = true;
    transaction.active = false;
    ++cache.staging.publishes;
    return true;
}

bool object_host_cache_abort_staging(
        object_host_cache &cache,
        object_host_staging_transaction &transaction) {
    if (!object_host_staging_transaction_matches(cache, transaction)) {
        ++cache.staging.failures;
        return false;
    }
    object_host_slot &slot = cache.slots[transaction.slot_index];
    if (slot.in_use || slot.pending_h2d || slot.pending_d2h ||
            !object_host_cache_invalidate_slot(cache, transaction.slot_index)) {
        ++cache.staging.failures;
        return false;
    }
    slot.marker = {};
    slot.ready = false;
    slot.loading = false;
    slot.in_use = false;
    slot.pending_h2d = false;
    slot.h2d_recorded = false;
    slot.pending_d2h = false;
    slot.delete_after_h2d = false;
    slot.read_us = 0;
    slot.payload_format = object_host_payload_format::raw_object;
    slot.payload_bytes = 0;
    slot.payload_transport_bytes = 0;
    slot.payload_crc32 = 0;
    slot.payload_record_identity = 0;
    transaction.active = false;
    ++cache.staging.aborts;
    return true;
}

bool object_host_cache_acquire_slot_lease(object_host_cache &cache, size_t slot_index) {
    if (!object_engine_enabled()) return true;
    if (!object_host_cache_slot_valid(cache, slot_index)) return !object_engine_enforcing();
    const object_host_slot &slot = cache.slots[slot_index];
    const bool acquired = slot.generation != 0 && object_engine_acquire_host_slot(
            slot.marker,
            cache.ownership_cache_id,
            static_cast<int32_t>(slot_index),
            slot.generation);
    return object_host_cache_ownership_result(acquired);
}

bool object_host_cache_release_slot_lease(object_host_cache &cache, size_t slot_index) {
    if (!object_engine_enabled()) return true;
    if (!object_host_cache_slot_valid(cache, slot_index)) return !object_engine_enforcing();
    const object_host_slot &slot = cache.slots[slot_index];
    const bool released = slot.generation != 0 && object_engine_release_host_slot(
            slot.marker,
            cache.ownership_cache_id,
            static_cast<int32_t>(slot_index),
            slot.generation);
    return object_host_cache_ownership_result(released);
}

static bool object_host_cache_finish_h2d_lease(
        object_host_cache &cache,
        size_t slot_index) {
    if (!object_host_cache_slot_valid(cache, slot_index)) return false;
    object_host_slot &slot = cache.slots[slot_index];
    if (!slot.pending_h2d) return true;
    if (!object_host_cache_release_slot_lease(cache, slot_index)) return false;
    slot.pending_h2d = false;
    slot.h2d_recorded = false;
    ++cache.completion.h2d_lease_releases;
    return true;
}

bool object_host_cache_begin_h2d(object_host_cache &cache, size_t slot_index) {
    ++cache.completion.h2d_begins;
    if (!object_host_cache_slot_valid(cache, slot_index)) {
        ++cache.completion.h2d_failures;
        return false;
    }
    object_host_slot &slot = cache.slots[slot_index];
    if (slot.pending_h2d) {
        ++cache.completion.h2d_failures;
        return false;
    }
    if (!object_host_slot_ensure_transfer_event(slot, object_host_transfer::h2d)) {
        ++cache.completion.h2d_failures;
        return false;
    }
    if (!object_host_cache_acquire_slot_lease(cache, slot_index)) {
        ++cache.completion.h2d_failures;
        return false;
    }
    slot.pending_h2d = true;
    slot.h2d_recorded = false;
    ++cache.completion.h2d_lease_acquires;
    return true;
}

bool object_host_cache_record_h2d(
        object_host_cache &cache,
        size_t slot_index,
        cudaStream_t stream) {
    ++cache.completion.h2d_records;
    if (!object_host_cache_slot_valid(cache, slot_index) ||
            !cache.slots[slot_index].pending_h2d ||
            cache.slots[slot_index].h2d_recorded ||
            !cache.slots[slot_index].h2d_done) {
        ++cache.completion.h2d_failures;
        return false;
    }
    if (object_host_slot_record_transfer(
                cache.slots[slot_index], object_host_transfer::h2d, stream)) {
        return true;
    }
    cudaGetLastError();
    if (cudaStreamSynchronize(stream) == cudaSuccess) {
        (void)object_host_cache_finish_h2d_lease(cache, slot_index);
    } else {
        cudaGetLastError();
    }
    ++cache.completion.h2d_failures;
    return false;
}

bool object_host_cache_cancel_h2d(
        object_host_cache &cache,
        size_t slot_index,
        cudaStream_t stream) {
    ++cache.completion.h2d_cancels;
    if (!object_host_cache_slot_valid(cache, slot_index) ||
            !cache.slots[slot_index].pending_h2d) {
        ++cache.completion.h2d_failures;
        return false;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        cudaGetLastError();
        ++cache.completion.h2d_failures;
        return false;
    }
    const bool released = object_host_cache_finish_h2d_lease(cache, slot_index);
    if (!released) ++cache.completion.h2d_failures;
    return released;
}

object_host_completion_state object_host_cache_query_h2d(
        object_host_cache &cache,
        size_t slot_index) {
    if (!object_host_cache_slot_valid(cache, slot_index)) {
        ++cache.completion.h2d_failures;
        return object_host_completion_state::failed;
    }
    object_host_slot &slot = cache.slots[slot_index];
    if (!slot.pending_h2d) return object_host_completion_state::idle;
    if (!slot.h2d_recorded) return object_host_completion_state::pending;
    if (!slot.h2d_done) {
        ++cache.completion.h2d_failures;
        return object_host_completion_state::failed;
    }
    const cudaError_t status = cudaEventQuery(slot.h2d_done);
    if (status == cudaErrorNotReady) return object_host_completion_state::pending;
    if (status != cudaSuccess) {
        cudaGetLastError();
        (void)object_host_cache_finish_h2d_lease(cache, slot_index);
        ++cache.completion.h2d_failures;
        return object_host_completion_state::failed;
    }
    if (!object_host_cache_finish_h2d_lease(cache, slot_index)) {
        ++cache.completion.h2d_failures;
        return object_host_completion_state::failed;
    }
    ++cache.completion.h2d_completions;
    return object_host_completion_state::complete;
}

object_host_completion_state object_host_cache_synchronize_h2d(
        object_host_cache &cache,
        size_t slot_index) {
    if (!object_host_cache_slot_valid(cache, slot_index)) {
        ++cache.completion.h2d_failures;
        return object_host_completion_state::failed;
    }
    object_host_slot &slot = cache.slots[slot_index];
    if (!slot.pending_h2d) return object_host_completion_state::idle;
    if (!slot.h2d_recorded) return object_host_completion_state::pending;
    if (!slot.h2d_done || cudaEventSynchronize(slot.h2d_done) != cudaSuccess) {
        cudaGetLastError();
        ++cache.completion.h2d_failures;
        return object_host_completion_state::failed;
    }
    if (!object_host_cache_finish_h2d_lease(cache, slot_index)) {
        ++cache.completion.h2d_failures;
        return object_host_completion_state::failed;
    }
    ++cache.completion.h2d_completions;
    return object_host_completion_state::complete;
}

bool object_host_cache_stream_wait_h2d(
        object_host_cache &cache,
        size_t slot_index,
        cudaStream_t stream) {
    const object_host_completion_state state = object_host_cache_query_h2d(cache, slot_index);
    if (state == object_host_completion_state::idle ||
            state == object_host_completion_state::complete) {
        return true;
    }
    if (state == object_host_completion_state::failed) return false;
    if (!cache.slots[slot_index].h2d_recorded) return false;
    if (!cache.slots[slot_index].h2d_done ||
            cudaStreamWaitEvent(stream, cache.slots[slot_index].h2d_done, 0) != cudaSuccess) {
        cudaGetLastError();
        ++cache.completion.h2d_failures;
        return false;
    }
    ++cache.completion.h2d_stream_waits;
    return true;
}

bool object_host_cache_select_slot(
        object_host_cache &cache,
        const object_host_slot_selection_policy &policy,
        object_host_slot_classify_fn classify,
        object_host_slot_score_fn score,
        void *context,
        object_host_slot_selection &selection) {
    selection = {};
    if (!classify || ((policy.evaluate_score || policy.select_by_score) && !score)) {
        return false;
    }

    uint64_t lowest_score = UINT64_MAX;
    uint64_t oldest = UINT64_MAX;
    for (size_t slot_index = 0; slot_index < cache.slots.size(); ++slot_index) {
        const object_host_slot_candidate_state state = classify(
            cache, slot_index, context);
        if (state == object_host_slot_candidate_state::excluded) continue;
        if (state == object_host_slot_candidate_state::empty && policy.prefer_empty) {
            selection.slot_index = slot_index;
            selection.score = 0;
            selection.used = cache.slots[slot_index].used;
            selection.empty = true;
            return true;
        }

        const uint64_t candidate_score = (policy.evaluate_score || policy.select_by_score) ?
            score(cache, slot_index, context) : 0;
        const uint64_t candidate_used = cache.slots[slot_index].used;
        if (selection.slot_index == SIZE_MAX ||
                (policy.select_by_score &&
                 (candidate_score < lowest_score ||
                  (candidate_score == lowest_score && candidate_used < oldest))) ||
                (!policy.select_by_score && candidate_used < oldest)) {
            selection.slot_index = slot_index;
            selection.score = candidate_score;
            selection.used = candidate_used;
            selection.empty = state == object_host_slot_candidate_state::empty;
            lowest_score = candidate_score;
            oldest = candidate_used;
        }
    }
    return selection.slot_index != SIZE_MAX;
}

object_host_admission_decision object_host_cache_evaluate_admission(
        const object_host_admission_policy &policy,
        uint64_t frequency,
        uint64_t candidate_score,
        bool has_victim,
        uint64_t victim_score) {
    if (frequency < policy.min_frequency) {
        return object_host_admission_decision::reject_frequency;
    }
    if (candidate_score < policy.min_score) {
        return object_host_admission_decision::reject_score;
    }
    if (!policy.compare_victim || !has_victim) {
        return object_host_admission_decision::admit;
    }
    if (candidate_score < victim_score ||
            (candidate_score == victim_score &&
             (!policy.allow_equal_victim || policy.victim_margin != 0))) {
        return object_host_admission_decision::reject_victim;
    }
    if (candidate_score > victim_score &&
            candidate_score - victim_score <= policy.victim_margin) {
        return object_host_admission_decision::reject_victim;
    }
    return object_host_admission_decision::admit;
}

bool object_host_cache_quota_allows_victim(
        const object_host_layer_quota_policy &policy,
        bool victim_occupied,
        bool victim_ready,
        int32_t victim_layer,
        size_t victim_occupancy,
        size_t victim_quota) {
    if (!policy.enabled) return true;
    if (!victim_occupied) {
        return policy.incoming_occupancy < policy.incoming_quota;
    }
    if (policy.incoming_quota == 0) return false;
    if (policy.incoming_occupancy >= policy.incoming_quota) {
        return victim_ready && victim_layer == policy.incoming_layer;
    }
    if (!policy.protect_victim_quota || victim_quota == 0) return true;
    return victim_occupancy > victim_quota;
}

static cudaEvent_t &object_host_slot_transfer_event(
        object_host_slot &slot,
        object_host_transfer transfer) {
    return transfer == object_host_transfer::h2d ? slot.h2d_done : slot.d2h_done;
}

static cudaEvent_t object_host_slot_transfer_event(
        const object_host_slot &slot,
        object_host_transfer transfer) {
    return transfer == object_host_transfer::h2d ? slot.h2d_done : slot.d2h_done;
}

static bool &object_host_slot_transfer_pending(
        object_host_slot &slot,
        object_host_transfer transfer) {
    return transfer == object_host_transfer::h2d ? slot.pending_h2d : slot.pending_d2h;
}

static bool object_host_slot_transfer_pending(
        const object_host_slot &slot,
        object_host_transfer transfer) {
    return transfer == object_host_transfer::h2d ? slot.pending_h2d : slot.pending_d2h;
}

bool object_host_slot_ensure_transfer_event(
        object_host_slot &slot,
        object_host_transfer transfer) {
    cudaEvent_t &event = object_host_slot_transfer_event(slot, transfer);
    if (event) return true;
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess) return true;
    cudaGetLastError();
    event = nullptr;
    object_host_slot_transfer_pending(slot, transfer) = false;
    return false;
}

bool object_host_slot_record_transfer(
        object_host_slot &slot,
        object_host_transfer transfer,
        CUstream_st *stream) {
    if (!stream || !object_host_slot_ensure_transfer_event(slot, transfer)) return false;
    cudaEvent_t event = object_host_slot_transfer_event(slot, transfer);
    if (cudaEventRecord(event, stream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    object_host_slot_transfer_pending(slot, transfer) = true;
    if (transfer == object_host_transfer::h2d) slot.h2d_recorded = true;
    return true;
}

object_host_transfer_status object_host_slot_query_transfer(
        const object_host_slot &slot,
        object_host_transfer transfer) {
    if (!object_host_slot_transfer_pending(slot, transfer)) {
        return object_host_transfer_status::idle;
    }
    if (transfer == object_host_transfer::h2d && !slot.h2d_recorded) {
        return object_host_transfer_status::pending;
    }
    cudaEvent_t event = object_host_slot_transfer_event(slot, transfer);
    if (!event) return object_host_transfer_status::error;
    const cudaError_t status = cudaEventQuery(event);
    if (status == cudaSuccess) return object_host_transfer_status::complete;
    if (status == cudaErrorNotReady) return object_host_transfer_status::pending;
    cudaGetLastError();
    return object_host_transfer_status::error;
}

object_host_transfer_status object_host_slot_stream_wait_transfer(
        const object_host_slot &slot,
        object_host_transfer transfer,
        CUstream_st *stream) {
    const object_host_transfer_status status =
        object_host_slot_query_transfer(slot, transfer);
    if (status != object_host_transfer_status::pending) return status;
    if (transfer == object_host_transfer::h2d && !slot.h2d_recorded) {
        return object_host_transfer_status::error;
    }
    cudaEvent_t event = object_host_slot_transfer_event(slot, transfer);
    if (!stream || cudaStreamWaitEvent(stream, event, 0) != cudaSuccess) {
        cudaGetLastError();
        return object_host_transfer_status::error;
    }
    return object_host_transfer_status::pending;
}

void object_host_slot_clear_transfer(
        object_host_slot &slot,
        object_host_transfer transfer) {
    object_host_slot_transfer_pending(slot, transfer) = false;
    if (transfer == object_host_transfer::h2d) slot.h2d_recorded = false;
}

bool object_host_cache_budget_enabled(const object_host_cache &cache) {
    return cache.max_resident_bytes > 0;
}

bool object_host_cache_budget_fits(
        const object_host_cache &cache,
        const object_host_slot &slot,
        size_t bytes) {
    if (cache.max_resident_bytes == 0) return true;
    const size_t resident_without_slot = cache.resident_capacity_bytes >= slot.capacity ?
        cache.resident_capacity_bytes - slot.capacity : 0;
    return resident_without_slot <= cache.max_resident_bytes &&
        bytes <= cache.max_resident_bytes - resident_without_slot;
}

void object_host_cache_account_release(
        object_host_cache &cache,
        object_host_slot &slot) {
    if (cache.resident_capacity_bytes >= slot.capacity) {
        cache.resident_capacity_bytes -= slot.capacity;
    } else {
        cache.resident_capacity_bytes = 0;
    }
}

void object_host_cache_account_allocate(
        object_host_cache &cache,
        object_host_slot &slot,
        size_t bytes) {
    slot.capacity = bytes;
    cache.resident_capacity_bytes += bytes;
    if (cache.resident_capacity_peak_bytes < cache.resident_capacity_bytes) {
        cache.resident_capacity_peak_bytes = cache.resident_capacity_bytes;
    }
}

} // namespace ggml_moe_common
