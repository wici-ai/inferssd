#include "moe-role-cache.h"

#include <cuda_runtime.h>

#include <cstdio>

namespace ggml_moe {

static bool role_cache_lease_event_slot_valid(
        const role_cache_view &view,
        int32_t slot) {
    return slot >= 0 && slot < view.slot_count &&
        view.slot_lease_events && view.slot_lease_pending;
}

bool role_cache_record_lease_event(
        const role_cache_view &view,
        int32_t slot,
        CUstream_st *stream) {
    if (!stream || !role_cache_lease_event_slot_valid(view, slot)) return false;
    cudaEvent_t &event = view.slot_lease_events[slot];
    if (!event &&
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        view.slot_lease_pending[slot] = false;
        return false;
    }
    if (cudaEventRecord(event, stream) != cudaSuccess) {
        cudaGetLastError();
        view.slot_lease_pending[slot] = false;
        return false;
    }
    view.slot_lease_pending[slot] = true;
    return true;
}

bool role_cache_acquire_descriptor_for_stream(
        const role_cache_view &view,
        int32_t slot,
        uintptr_t expected_key,
        uint64_t object_generation,
        role kind,
        size_t role_offset,
        CUstream_st *stream,
        ggml_moe_expert_descriptor &descriptor) {
    if (!stream || !role_cache_acquire_descriptor_impl(
                view, slot, expected_key, object_generation, kind, role_offset,
                true, descriptor)) {
        return false;
    }
    if (role_cache_stream_wait_ready_event(view, slot, stream)) return true;

    bool last_lease = false;
    (void) role_cache_release_descriptor(
        view, expected_key, descriptor, &last_lease);
    descriptor = {};
    descriptor.slot_id = -1;
    descriptor.role = static_cast<int32_t>(kind);
    return false;
}

bool role_cache_stream_wait_lease_event(
        const role_cache_view &view,
        int32_t slot,
        CUstream_st *stream) {
    if (slot < 0 || slot >= view.slot_count ||
            !view.slot_lease_pending || !view.slot_lease_pending[slot]) {
        return true;
    }
    if (!view.slot_lease_events) return false;
    cudaEvent_t event = view.slot_lease_events[slot];
    if (!event) {
        view.slot_lease_pending[slot] = false;
        return true;
    }
    const cudaError_t status = cudaEventQuery(event);
    if (status == cudaSuccess) {
        view.slot_lease_pending[slot] = false;
        return true;
    }
    if (status != cudaErrorNotReady) {
        cudaGetLastError();
        return false;
    }
    if (!stream || cudaStreamWaitEvent(stream, event, 0) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

bool role_cache_synchronize_lease_event(
        const role_cache_view &view,
        int32_t slot) {
    if (slot < 0 || slot >= view.slot_count ||
            !view.slot_lease_pending || !view.slot_lease_pending[slot]) {
        return true;
    }
    if (!view.slot_lease_events) return false;
    cudaEvent_t event = view.slot_lease_events[slot];
    if (event && cudaEventSynchronize(event) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    view.slot_lease_pending[slot] = false;
    return true;
}

bool role_cache_stream_wait_ready_event(
        const role_cache_view &view,
        int32_t slot,
        CUstream_st *stream) {
    if (slot < 0 || slot >= view.slot_count ||
            !view.slot_pending || !view.slot_pending[slot]) {
        return true;
    }
    if (!view.slot_ready_events) return false;
    cudaEvent_t event = view.slot_ready_events[slot];
    if (!event) {
        view.slot_pending[slot] = false;
        return true;
    }
    const cudaError_t status = cudaEventQuery(event);
    if (status == cudaSuccess) {
        view.slot_pending[slot] = false;
        return true;
    }
    if (status != cudaErrorNotReady) {
        cudaGetLastError();
        return false;
    }
    if (!stream || cudaStreamWaitEvent(stream, event, 0) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

bool role_cache_synchronize_ready_event(
        const role_cache_view &view,
        int32_t slot) {
    if (slot < 0 || slot >= view.slot_count ||
            !view.slot_pending || !view.slot_pending[slot]) {
        return true;
    }
    if (!view.slot_ready_events) return false;
    cudaEvent_t event = view.slot_ready_events[slot];
    if (event && cudaEventSynchronize(event) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    view.slot_pending[slot] = false;
    return true;
}

bool role_cache_synchronize_ready_events(
        const role_cache_view &view,
        const int32_t *slots,
        size_t count,
        role_cache_batch_wait_stats *stats) {
    if (stats) *stats = {};
    if (count == 0) return true;
    if (!slots || !view.slot_pending || !view.slot_ready_events) return false;

    int32_t representative = -1;
    for (size_t i = 0; i < count; ++i) {
        const int32_t slot = slots[i];
        if (slot < 0 || slot >= view.slot_count) return false;
        if (view.slot_pending[slot] && view.slot_ready_events[slot]) {
            representative = slot;
        }
    }
    if (representative >= 0) {
        if (cudaEventSynchronize(view.slot_ready_events[representative]) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        if (stats) ++stats->representative_syncs;
    }

    for (size_t i = 0; i < count; ++i) {
        const int32_t slot = slots[i];
        if (!view.slot_pending[slot] || slot == representative ||
                !view.slot_ready_events[slot]) {
            continue;
        }
        const cudaError_t status = cudaEventQuery(view.slot_ready_events[slot]);
        if (status == cudaSuccess) continue;
        if (status != cudaErrorNotReady) {
            cudaGetLastError();
            return false;
        }
        if (cudaEventSynchronize(view.slot_ready_events[slot]) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        if (stats) ++stats->extra_syncs;
    }

    for (size_t i = 0; i < count; ++i) {
        view.slot_pending[slots[i]] = false;
    }
    return true;
}

bool role_cache_ensure_ready_event(
        const role_cache_view &view,
        int32_t slot) {
    if (slot < 0 || slot >= view.slot_count ||
            !view.slot_ready_events || !view.slot_pending) {
        return false;
    }
    cudaEvent_t &event = view.slot_ready_events[slot];
    if (event) return true;
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess) {
        return true;
    }
    cudaGetLastError();
    event = nullptr;
    view.slot_pending[slot] = false;
    return false;
}

bool role_cache_record_ready_event(
        const role_cache_view &view,
        int32_t slot,
        CUstream_st *stream) {
    if (!stream || !role_cache_ensure_ready_event(view, slot)) return false;
    if (cudaEventRecord(view.slot_ready_events[slot], stream) != cudaSuccess) {
        cudaGetLastError();
        view.slot_pending[slot] = false;
        return false;
    }
    return true;
}

bool role_cache_publish_ready(
        const role_cache_view &view,
        int32_t slot,
        uintptr_t expected_key,
        uint64_t expected_generation,
        size_t expected_payload_bytes,
        CUstream_st *stream) {
    if (!stream || !role_cache_view_valid(view) || slot < 0 || slot >= view.slot_count ||
            expected_key == 0 || expected_generation == 0 || expected_payload_bytes == 0 ||
            !view.slot_payload_bytes || !view.slot_ready_markers || !view.slot_states ||
            !view.slot_pending || !view.slot_filling ||
            view.slot_keys[slot] != expected_key ||
            view.slot_generations[slot] != expected_generation ||
            view.slot_payload_bytes[slot] != expected_payload_bytes ||
            !view.slot_filling[slot] ||
            (view.slot_states[slot] != role_slot_state::reserved &&
             view.slot_states[slot] != role_slot_state::reading)) {
        return false;
    }
    if (!role_cache_record_ready_event(view, slot, stream)) return false;
    view.slot_ready_markers[slot] = {
        expected_key, expected_generation, expected_payload_bytes,
    };
    view.slot_states[slot] = role_slot_state::h2d_ready;
    view.slot_filling[slot] = false;
    view.slot_pending[slot] = true;
    return true;
}

static bool role_cache_h2d_copy_invalid(
        const role_cache_h2d_copy &copy,
        const char *reason) {
    const role_cache_view &view = copy.view;
    const bool slot_valid = copy.slot >= 0 && copy.slot < view.slot_count;
    std::fprintf(stderr,
            "[moe_role_cache] h2d copy rejected: reason=%s slot=%d slots=%d "
            "key=%lx current_key=%lx generation=%lu current_generation=%lu "
            "payload=%zu current_payload=%zu copy=%zu slot_bytes=%zu "
            "dst=%p slot_ptr=%p state=%d filling=%d pending=%d\n",
            reason, copy.slot, view.slot_count,
            static_cast<unsigned long>(copy.expected_key),
            static_cast<unsigned long>(
                slot_valid && view.slot_keys ? view.slot_keys[copy.slot] : 0),
            static_cast<unsigned long>(copy.expected_generation),
            static_cast<unsigned long>(
                slot_valid && view.slot_generations ? view.slot_generations[copy.slot] : 0),
            copy.expected_payload_bytes,
            slot_valid && view.slot_payload_bytes ? view.slot_payload_bytes[copy.slot] : 0,
            copy.copy_bytes, view.slot_bytes, copy.device_dst,
            slot_valid ? role_cache_slot_pointer(view, copy.slot) : nullptr,
            static_cast<int>(slot_valid && view.slot_states ?
                view.slot_states[copy.slot] : role_slot_state::failed),
            slot_valid && view.slot_filling ? view.slot_filling[copy.slot] : 0,
            slot_valid && view.slot_pending ? view.slot_pending[copy.slot] : 0);
    return false;
}

static bool role_cache_h2d_copy_valid(const role_cache_h2d_copy &copy) {
    const role_cache_view &view = copy.view;
    if (!role_cache_view_valid(view)) return role_cache_h2d_copy_invalid(copy, "view");
    if (copy.slot < 0 || copy.slot >= view.slot_count) {
        return role_cache_h2d_copy_invalid(copy, "slot");
    }
    if (copy.expected_key == 0 || copy.expected_generation == 0 ||
            copy.expected_payload_bytes == 0 || copy.copy_bytes == 0 ||
            copy.copy_bytes != copy.expected_payload_bytes ||
            copy.copy_bytes > view.slot_bytes || !copy.device_dst || !copy.host_src) {
        return role_cache_h2d_copy_invalid(copy, "descriptor");
    }
    if (!view.slot_payload_bytes || !view.slot_states || !view.slot_filling) {
        return role_cache_h2d_copy_invalid(copy, "state_arrays");
    }
    if (view.slot_keys[copy.slot] != copy.expected_key) {
        return role_cache_h2d_copy_invalid(copy, "key");
    }
    if (view.slot_generations[copy.slot] != copy.expected_generation) {
        return role_cache_h2d_copy_invalid(copy, "generation");
    }
    if (view.slot_payload_bytes[copy.slot] != copy.expected_payload_bytes) {
        return role_cache_h2d_copy_invalid(copy, "payload");
    }
    if (!view.slot_filling[copy.slot]) {
        return role_cache_h2d_copy_invalid(copy, "not_filling");
    }
    const void *slot_ptr = role_cache_slot_pointer(view, copy.slot);
    if (slot_ptr != copy.device_dst) return role_cache_h2d_copy_invalid(copy, "address");
    if (view.slot_states[copy.slot] != role_slot_state::reserved &&
            view.slot_states[copy.slot] != role_slot_state::reading) {
        return role_cache_h2d_copy_invalid(copy, "state");
    }
    return true;
}

bool role_cache_copy_h2d_batch_async(
        const role_cache_h2d_copy *copies,
        size_t count,
        CUstream_st *stream,
        role_cache_h2d_stats *stats) {
    if (stats) *stats = {};
    if (!copies || count == 0 || !stream) return false;

    for (size_t i = 0; i < count; ++i) {
        if (!role_cache_h2d_copy_valid(copies[i])) return false;
        if (!role_cache_ensure_ready_event(copies[i].view, copies[i].slot)) {
            return role_cache_h2d_copy_invalid(copies[i], "ready_event");
        }
    }
    for (size_t i = 0; i < count; ++i) {
        const role_cache_h2d_copy &copy = copies[i];
        if (copy.view.slot_states[copy.slot] == role_slot_state::reserved &&
                !role_cache_mark_reading(
                    copy.view, copy.slot, copy.expected_key,
                    copy.expected_generation, copy.expected_payload_bytes)) {
            role_cache_h2d_copy_invalid(copy, "mark_reading");
            return false;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        const role_cache_h2d_copy &copy = copies[i];
        if (cudaMemcpyAsync(
                    copy.device_dst, copy.host_src, copy.copy_bytes,
                    cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            role_cache_h2d_copy_invalid(copy, "memcpy");
            cudaGetLastError();
            return false;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        const role_cache_h2d_copy &copy = copies[i];
        if (!role_cache_publish_ready(
                    copy.view, copy.slot, copy.expected_key,
                    copy.expected_generation, copy.expected_payload_bytes, stream)) {
            role_cache_h2d_copy_invalid(copy, "publish");
            return false;
        }
    }
    if (stats) {
        stats->copies = count;
        for (size_t i = 0; i < count; ++i) stats->bytes += copies[i].copy_bytes;
    }
    return true;
}

} // namespace ggml_moe
