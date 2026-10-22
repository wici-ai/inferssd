#include "../ggml/src/ggml-cuda/moe-object-engine.h"
#include "../ggml/src/ggml-cuda/moe-object-cache.h"
#include "../ggml/src/ggml-cuda/moe-object-transport.h"
#include "../ggml/src/ggml-cuda/moe-device-descriptor.h"
#include "../ggml/src/ggml-cuda/moe-role-cache.h"
#include "../ggml/src/ggml-cuda/moe-object-pipeline.h"
#include "../ggml/src/ggml-cuda/moe-object-pipeline-runtime.h"
#include "../ggml/src/ggml-cuda/moe-transfer-planner.h"
#include "../ggml/src/ggml-cuda/moe-adapter-dispatch.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace {

ggml_moe::object_shape make_shape(size_t gate, size_t up, size_t down) {
    ggml_moe::object_shape shape;
    shape.object_bytes = gate + up + down;
    shape.roles[0] = {ggml_moe::role::gate, 0, gate};
    shape.roles[1] = {ggml_moe::role::up, gate, up};
    shape.roles[2] = {ggml_moe::role::down, gate + up, down};
    return shape;
}

ggml_moe::object_request make_request(
        int layer,
        int expert,
        uint64_t offset,
        const ggml_moe::object_shape &shape,
        uint8_t role_count = 3) {
    ggml_moe::object_request request;
    request.key = {7, layer, expert, 0xabc, offset, shape.object_bytes, 0};
    request.required_role_mask = 0;
    request.role_count = role_count;
    for (uint8_t i = 0; i < role_count; ++i) {
        request.roles[i] = shape.roles[i];
        request.required_role_mask |= ggml_moe::role_bit(shape.roles[i].kind);
    }
    return request;
}

} // namespace

static int run_kimi_iq1_adapter() {
    if (ggml_moe::adapter_is_selected()) return 197;
    ggml_moe::adapter_select_model_shape(896, 2150400, "iq1-adapter-test");
    if (!ggml_moe::adapter_is_selected() || ggml_moe::adapter_uses_common()) return 198;
    return 0;
}

static int run_single_shape() {
    using namespace ggml_moe;

    object_layout layout;
    std::strcpy(layout.model, "single-shape-test");
    layout.source_idx = 8;
    layout.first_layer = 5;
    layout.layer_count = 1;
    layout.first_expert = 0;
    layout.experts_per_layer = 1;
    layout.top_k = 1;
    layout.object_count = 1;
    layout.manifest_generation = 0xdef;
    layout.shape_count = 1;
    layout.shapes[0] = make_shape(10, 20, 30);
    if (!object_engine_configure(layout)) return 20;

    object_request request = make_request(5, 0, 4096, layout.shapes[0]);
    request.key.source_idx = 8;
    request.key.manifest_generation = 0xdef;
    if (!object_engine_begin_batch(&request, 1)) return 21;
    if (!object_engine_complete_batch(&request, 1, true)) return 22;
    return 0;
}

static int run_stage_ring() {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (ggml_moe::object_io_sqpoll_setup_flag() == 0) return 29;
#else
    if (ggml_moe::object_io_sqpoll_setup_flag() != 0) return 29;
#endif
    ggml_moe::object_stage_ring ring;
    ggml_moe::object_stage_ring lazy_ring;
    lazy_ring.slot_sz = 2048;
    lazy_ring.slots.resize(1);
    if (!ggml_moe::object_stage_slot_allocate(lazy_ring.slots[0], lazy_ring.slot_sz, false) ||
            !lazy_ring.slots[0].host || lazy_ring.slots[0].host_capacity != lazy_ring.slot_sz ||
            !lazy_ring.slots[0].done || lazy_ring.slots[0].copy_start || lazy_ring.slots[0].copy_done ||
            ggml_moe::object_stage_slot_allocate(lazy_ring.slots[0], lazy_ring.slot_sz, false)) {
        return 144;
    }
    void *const lazy_host = lazy_ring.slots[0].host;
    if (!ggml_moe::object_stage_slot_ensure_capacity(lazy_ring.slots[0], 1024, false) ||
            lazy_ring.slots[0].host != lazy_host ||
            !ggml_moe::object_stage_slot_ensure_capacity(lazy_ring.slots[0], 4096, true) ||
            !lazy_ring.slots[0].host || lazy_ring.slots[0].host_capacity != 4096 ||
            !lazy_ring.slots[0].done || !lazy_ring.slots[0].copy_start || !lazy_ring.slots[0].copy_done) {
        return 146;
    }
    lazy_ring.slots[0].pending = true;
    if (ggml_moe::object_stage_slot_ensure_capacity(lazy_ring.slots[0], 8192, true)) return 147;
    lazy_ring.slots[0].pending = false;
    ggml_moe::object_stage_ring_release(lazy_ring);
    if (lazy_ring.slots.size() != 0 || lazy_ring.slot_sz != 0) return 145;
    if (!ggml_moe::object_stage_ring_allocate(ring, 4096, 3, true)) return 30;
    if (ring.failed || ring.slot_sz != 4096 || ring.slots.size() != 3) return 31;
    for (size_t i = 0; i < ring.slots.size(); ++i) {
        const ggml_moe::object_stage_slot &slot = ring.slots[i];
        if (!slot.host || slot.host_capacity != ring.slot_sz ||
                !slot.done || !slot.copy_start || !slot.copy_done) return 32;
        for (size_t j = 0; j < i; ++j) {
            if (ring.slots[j].host == slot.host || ring.slots[j].done == slot.done) return 33;
        }
    }
    void *const first_host = ring.slots[0].host;
    if (ggml_moe::object_stage_ring_allocate(ring, 4096, 3, true)) return 34;
    if (ring.slots[0].host != first_host) return 35;
    cudaStream_t stage_stream = nullptr;
    if (cudaStreamCreateWithFlags(&stage_stream, cudaStreamNonBlocking) != cudaSuccess) return 142;
    if (!ggml_moe::object_stage_slot_synchronize(ring, ring.slots[0]) ||
            !ggml_moe::object_stage_slot_record(ring.slots[0], stage_stream) ||
            !ring.slots[0].pending ||
            ggml_moe::object_stage_slot_record(ring.slots[0], stage_stream) ||
            !ggml_moe::object_stage_slot_synchronize(ring, ring.slots[0]) ||
            ring.slots[0].pending || ring.waits != 1 ||
            !ggml_moe::object_stage_slot_record(ring.slots[0], stage_stream)) {
        if (stage_stream) cudaStreamDestroy(stage_stream);
        return 143;
    }
    const cudaError_t stage_query = cudaEventQuery(ring.slots[0].done);
    if (stage_query == cudaSuccess) {
        ggml_moe::object_stage_slot_publish_ready(ring.slots[0]);
    }
    if ((stage_query != cudaSuccess &&
         !ggml_moe::object_stage_slot_synchronize(ring, ring.slots[0])) ||
            ring.slots[0].pending) {
        if (stage_stream) cudaStreamDestroy(stage_stream);
        return 143;
    }
    ring.slots[0].pending = true;
    ggml_moe::object_stage_slot_publish_ready(ring.slots[0]);
    if (ring.slots[0].pending || cudaStreamDestroy(stage_stream) != cudaSuccess) {
        if (stage_stream) cudaStreamDestroy(stage_stream);
        return 143;
    }
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (!ggml_moe::object_io_ring_ensure(ring, 2, 0, nullptr)) return 36;
    io_uring *const first_ring = ring.uring;
    if (!ggml_moe::object_io_ring_ensure(ring, 2, 0, nullptr)) return 37;
    if (ring.uring != first_ring || ring.uring_depth != 2 ||
            ring.iouring_init_attempts != 1 || ring.iouring_init_reuses != 1 ||
            ring.iouring_init_failures != 0 ||
            ring.iouring_sqpoll_affinity_attempts != 0 ||
            ring.iouring_sqpoll_affinity_successes != 0 ||
            ring.iouring_sqpoll_affinity_failures != 0) {
        return 38;
    }
    FILE *file = std::tmpfile();
    if (!file) return 39;
    uint8_t payload[4096];
    for (size_t i = 0; i < sizeof(payload); ++i) payload[i] = (uint8_t)(i % 251);
    if (std::fwrite(payload, 1, sizeof(payload), file) != sizeof(payload) || std::fflush(file) != 0) {
        std::fclose(file);
        return 40;
    }
    uint8_t pread_payload[31] = {};
    if (!ggml_moe::object_pread_all(
                fileno(file), pread_payload, sizeof(pread_payload), 13, 7) ||
            std::memcmp(pread_payload, payload + 13, sizeof(pread_payload)) != 0 ||
            !ggml_moe::object_pread_all(fileno(file), nullptr, 0, 0) ||
            ggml_moe::object_pread_all(-1, pread_payload, sizeof(pread_payload), 0)) {
        std::fclose(file);
        return 140;
    }
    ggml_moe::object_io_runtime_counters io_counters;
    io_counters.batches.fetch_add(2, std::memory_order_relaxed);
    io_counters.reads.fetch_add(3, std::memory_order_relaxed);
    io_counters.bytes.fetch_add(4096, std::memory_order_relaxed);
    io_counters.fallbacks.fetch_add(1, std::memory_order_relaxed);
    io_counters.submit_us.fetch_add(11, std::memory_order_relaxed);
    io_counters.wait_us.fetch_add(17, std::memory_order_relaxed);
    io_counters.submit_calls.fetch_add(4, std::memory_order_relaxed);
    io_counters.wait_calls.fetch_add(5, std::memory_order_relaxed);
    io_counters.cqes.fetch_add(6, std::memory_order_relaxed);
    io_counters.inflight_sum.fetch_add(21, std::memory_order_relaxed);
    io_counters.inflight_samples.fetch_add(7, std::memory_order_relaxed);
    io_counters.inflight_max.store(8, std::memory_order_relaxed);
    const ggml_moe::object_io_runtime_snapshot io_snapshot =
        ggml_moe::object_io_runtime_snapshot_load(io_counters);
    if (io_snapshot.batches != 2 || io_snapshot.reads != 3 || io_snapshot.bytes != 4096 ||
            io_snapshot.fallbacks != 1 || io_snapshot.submit_us != 11 || io_snapshot.wait_us != 17 ||
            io_snapshot.submit_calls != 4 || io_snapshot.wait_calls != 5 || io_snapshot.cqes != 6 ||
            io_snapshot.inflight_sum != 21 || io_snapshot.inflight_samples != 7 ||
            io_snapshot.inflight_max != 8) {
        std::fclose(file);
        return 141;
    }
    std::memset(ring.slots[0].host, 0, sizeof(payload));
    ggml_moe::object_io_window_slot io_window[3];
    const ggml_moe::object_io_window_request io_request{
        fileno(file), ring.slots[0].host, sizeof(payload), 0, 17, false,
    };
    if (!ggml_moe::object_io_window_prepare_read(
                ring, io_window, 3, 0, io_request) ||
            ggml_moe::object_io_ring_submit_count(ring) != 1) {
        std::fclose(file);
        return 41;
    }
    ggml_moe::object_io_completion completion;
    if (!ggml_moe::object_io_ring_wait(ring, completion)) {
        std::fclose(file);
        return 42;
    }
    ggml_moe::object_io_window_completion io_resolved;
    if (ggml_moe::object_io_window_resolve(
                ring, io_window, 3, completion, io_resolved) !=
                ggml_moe::object_io_completion_status::complete ||
            io_resolved.window_slot != 0 || io_resolved.request_id != 17 ||
            io_resolved.result != static_cast<int>(sizeof(payload)) || io_window[0].active ||
            ring.window_prepares != 1 || ring.window_resolves != 1 ||
            std::memcmp(ring.slots[0].host, payload, sizeof(payload)) != 0) {
        std::fclose(file);
        return 42;
    }
    std::memset(ring.slots[1].host, 0, sizeof(payload));
    if (!ggml_moe::object_io_ring_prepare_read(
                ring, fileno(file), ring.slots[1].host, sizeof(payload), 0, 0x5678) ||
            !ggml_moe::object_io_ring_submit(ring) ||
            !ggml_moe::object_io_ring_wait_for(ring, completion, 5000)) {
        std::fclose(file);
        return 43;
    }
    if (ggml_moe::object_io_completion_validate(
                ring, completion, completion.user_data == 0x5678, sizeof(payload)) !=
                ggml_moe::object_io_completion_status::complete ||
            std::memcmp(ring.slots[1].host, payload, sizeof(payload)) != 0) {
        std::fclose(file);
        return 43;
    }
    if (ggml_moe::object_io_ring_peek(ring, completion) != ggml_moe::object_io_poll_result::empty ||
            ggml_moe::object_io_ring_prepare_read(ring, -1, ring.slots[0].host, sizeof(payload), 0, 1)) {
        std::fclose(file);
        return 44;
    }
    std::memset(ring.slots[2].host, 0, sizeof(payload));
    if (!ggml_moe::object_io_ring_register_file(ring, fileno(file)) ||
            !ggml_moe::object_io_ring_prepare_read(
                ring, fileno(file), ring.slots[2].host, sizeof(payload), 0, 0x9abc, true) ||
            ggml_moe::object_io_ring_submit_count(ring) != 1 ||
            !ggml_moe::object_io_ring_wait(ring, completion)) {
        std::fclose(file);
        return 45;
    }
    if (ggml_moe::object_io_completion_validate(
                ring, completion, completion.user_data == 0x9abc, sizeof(payload)) !=
                ggml_moe::object_io_completion_status::complete ||
            std::memcmp(ring.slots[2].host, payload, sizeof(payload)) != 0 ||
            !ggml_moe::object_io_ring_unregister_files(ring) ||
            ring.uring_fixed_file_registered || ring.uring_fixed_fd != -1 ||
            ring.iouring_submit_calls != 3 || ring.iouring_wait_calls != 3 || ring.iouring_cqes != 3) {
        std::fclose(file);
        return 45;
    }
    std::fclose(file);
#endif
    ggml_moe::object_io_completion synthetic = {7, 4096};
    if (ggml_moe::object_io_completion_validate(ring, synthetic, false, 4096) !=
            ggml_moe::object_io_completion_status::invalid_user_data) return 48;
    if (ggml_moe::object_io_completion_validate(ring, synthetic, true, 0) !=
            ggml_moe::object_io_completion_status::invalid_expected_bytes) return 48;
    synthetic.result = -5;
    if (ggml_moe::object_io_completion_validate(ring, synthetic, true, 4096) !=
            ggml_moe::object_io_completion_status::io_error) return 48;
    synthetic.result = 4095;
    if (ggml_moe::object_io_completion_validate(ring, synthetic, true, 4096) !=
            ggml_moe::object_io_completion_status::short_read) return 48;
    synthetic.result = 4097;
    if (ggml_moe::object_io_completion_validate(ring, synthetic, true, 4096) !=
            ggml_moe::object_io_completion_status::oversized_read) return 48;
    const ggml_moe::object_io_completion_counters &counters = ring.completion_counters;
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    const uint64_t expected_validations = 8;
    const uint64_t expected_completed = 3;
#else
    const uint64_t expected_validations = 5;
    const uint64_t expected_completed = 0;
#endif
    if (counters.validations != expected_validations || counters.completed != expected_completed ||
            counters.invalid_user_data != 1 || counters.invalid_expected_bytes != 1 ||
            counters.io_errors != 1 || counters.short_reads != 1 || counters.oversized_reads != 1) {
        return 49;
    }
    ggml_moe::object_stage_ring_release(ring);
    if (!ring.slots.empty() || ring.slot_sz != 0 || ring.next != 0) return 46;
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (ring.uring || ring.uring_depth != 0 || ring.uring_flags != 0 ||
            ring.uring_sq_thread_idle != 0 || ring.uring_fixed_file_registered ||
            ring.uring_fixed_fd != -1) {
        return 47;
    }
#endif
    return 0;
}

static int run_host_slot() {
    ggml_moe::object_host_slot pinned;
    if (!ggml_moe::object_host_slot_allocate_memory(pinned, 8192, false, 4096)) return 50;
    if (!pinned.host || pinned.capacity != 8192 || pinned.pageable ||
            pinned.host_registered || pinned.host_register_attempted || !pinned.owns_memory) {
        return 51;
    }
    if (ggml_moe::object_host_slot_allocate_memory(pinned, 4096, false, 4096)) return 52;
    ggml_moe::object_host_slot_release_memory(pinned);
    if (pinned.host || pinned.capacity != 0 || pinned.pageable ||
            pinned.host_registered || pinned.host_register_attempted || pinned.owns_memory) {
        return 53;
    }

    ggml_moe::object_host_slot pageable;
    if (!ggml_moe::object_host_slot_allocate_memory(pageable, 4096, true, 4096)) return 54;
    if (!pageable.host || pageable.capacity != 4096 || !pageable.pageable || !pageable.owns_memory ||
            (reinterpret_cast<uintptr_t>(pageable.host) % 4096) != 0) {
        return 55;
    }
    const bool registered = ggml_moe::object_host_slot_register_pageable(pageable);
    if (!pageable.host_register_attempted || registered != pageable.host_registered) return 56;
    ggml_moe::object_host_slot_release_memory(pageable);
    if (pageable.host || pageable.capacity != 0 || pageable.pageable ||
            pageable.host_registered || pageable.host_register_attempted || pageable.owns_memory) {
        return 57;
    }

    ggml_moe::object_host_slot invalid;
    if (ggml_moe::object_host_slot_allocate_memory(invalid, 4096, true, 3)) return 58;

    ggml_moe::object_host_cache cache;
    cache.max_resident_bytes = 8192;
    cache.slots.resize(2);
    cache.slots[0].capacity = 4096;
    cache.resident_capacity_bytes = 4096;
    if (!ggml_moe::object_host_cache_budget_enabled(cache) ||
            !ggml_moe::object_host_cache_budget_fits(cache, cache.slots[0], 8192)) {
        return 59;
    }
    ggml_moe::object_host_cache_account_release(cache, cache.slots[0]);
    ggml_moe::object_host_cache_account_allocate(cache, cache.slots[0], 8192);
    if (cache.resident_capacity_bytes != 8192 || cache.resident_capacity_peak_bytes != 8192 ||
            ggml_moe::object_host_cache_budget_fits(cache, cache.slots[1], 1)) {
        return 60;
    }
    ggml_moe::object_host_cache_account_release(cache, cache.slots[0]);
    if (cache.resident_capacity_bytes != 0) return 61;
    cache.max_resident_bytes = 0;
    if (ggml_moe::object_host_cache_budget_enabled(cache) ||
            !ggml_moe::object_host_cache_budget_fits(cache, cache.slots[1], SIZE_MAX)) {
        return 62;
    }
    if (!ggml_moe::object_host_cache_configure_ownership(cache, 7)) return 101;
    ggml_moe::object_key ready_key;
    ready_key.source_idx = 0;
    ready_key.layer_idx = 1;
    ready_key.expert_idx = 2;
    ready_key.manifest_generation = 3;
    ready_key.offset = 4096;
    ready_key.bytes = 256;
    if (!ggml_moe::object_host_cache_prepare_slot(cache, 0)) return 102;
    cache.slots[0].marker = ready_key;
    cache.slots[0].ready = true;
    if (!ggml_moe::object_host_cache_publish_slot(cache, 0)) return 103;
    size_t ready_slot = SIZE_MAX;
    if (!ggml_moe::object_host_cache_find_ready_slot(cache, ready_key, &ready_slot) ||
            ready_slot != 0 || ggml_moe::object_host_cache_ready_count(cache) != 1) {
        return 104;
    }
    if (!ggml_moe::object_host_cache_prepare_slot(cache, 1)) return 105;
    cache.slots[1].marker = ready_key;
    cache.slots[1].ready = true;
    if (ggml_moe::object_host_cache_publish_slot(cache, 1) ||
            cache.ready_index_failures != 1 ||
            !ggml_moe::object_host_cache_find_ready_slot(cache, ready_key, &ready_slot) ||
            ready_slot != 0) {
        return 106;
    }
    if (!ggml_moe::object_host_cache_invalidate_slot(cache, 1)) return 107;
    for (uint64_t generation = 4; generation < 40; ++generation) {
        if (!ggml_moe::object_host_cache_prepare_slot(cache, 0)) return 108;
        ready_key.manifest_generation = generation;
        ready_key.offset += 4096;
        cache.slots[0].marker = ready_key;
        cache.slots[0].ready = true;
        if (!ggml_moe::object_host_cache_publish_slot(cache, 0) ||
                !ggml_moe::object_host_cache_find_ready_slot(cache, ready_key, &ready_slot) ||
                ready_slot != 0 || ggml_moe::object_host_cache_ready_count(cache) != 1) {
            return 109;
        }
    }
    if (!ggml_moe::object_host_cache_unpublish_ready_slot(cache, 0) ||
            ggml_moe::object_host_cache_find_ready_slot(cache, ready_key, nullptr) ||
            ggml_moe::object_host_cache_ready_count(cache) != 0) {
        return 110;
    }

    struct selection_context {
        ggml_moe::object_host_slot_candidate_state states[4];
        uint64_t scores[4];
    } selection_ctx = {
        {
            ggml_moe::object_host_slot_candidate_state::resident,
            ggml_moe::object_host_slot_candidate_state::empty,
            ggml_moe::object_host_slot_candidate_state::excluded,
            ggml_moe::object_host_slot_candidate_state::resident,
        },
        {100, 0, 1, 50},
    };
    auto classify = [](ggml_moe::object_host_cache &, size_t slot_index, void *context) {
        return static_cast<selection_context *>(context)->states[slot_index];
    };
    auto score = [](const ggml_moe::object_host_cache &, size_t slot_index, void *context) {
        return static_cast<selection_context *>(context)->scores[slot_index];
    };
    ggml_moe::object_host_cache selection_cache;
    selection_cache.slots.resize(4);
    selection_cache.slots[0].used = 40;
    selection_cache.slots[1].used = 30;
    selection_cache.slots[2].used = 20;
    selection_cache.slots[3].used = 10;
    ggml_moe::object_host_slot_selection selected;
    ggml_moe::object_host_slot_selection_policy score_policy;
    score_policy.evaluate_score = true;
    score_policy.select_by_score = true;
    if (!ggml_moe::object_host_cache_select_slot(
                selection_cache, score_policy, classify, score, &selection_ctx, selected) ||
            selected.slot_index != 1 || !selected.empty || selected.score != 0) {
        return 111;
    }
    selection_ctx.states[1] = ggml_moe::object_host_slot_candidate_state::excluded;
    if (!ggml_moe::object_host_cache_select_slot(
                selection_cache, score_policy, classify, score, &selection_ctx, selected) ||
            selected.slot_index != 3 || selected.empty || selected.score != 50) {
        return 112;
    }
    selection_ctx.scores[3] = 100;
    if (!ggml_moe::object_host_cache_select_slot(
                selection_cache, score_policy, classify, score, &selection_ctx, selected) ||
            selected.slot_index != 3 || selected.used != 10) {
        return 113;
    }
    selection_ctx.states[1] = ggml_moe::object_host_slot_candidate_state::empty;
    selection_ctx.scores[1] = 200;
    ggml_moe::object_host_slot_selection_policy profile_lru_policy;
    profile_lru_policy.prefer_empty = false;
    profile_lru_policy.evaluate_score = true;
    if (!ggml_moe::object_host_cache_select_slot(
                selection_cache, profile_lru_policy, classify, score, &selection_ctx, selected) ||
            selected.slot_index != 3 || selected.empty || selected.score != 100) {
        return 117;
    }
    selection_ctx.states[1] = ggml_moe::object_host_slot_candidate_state::excluded;
    selection_cache.slots[0].used = 5;
    ggml_moe::object_host_slot_selection_policy lru_policy;
    if (!ggml_moe::object_host_cache_select_slot(
                selection_cache, lru_policy, classify, nullptr, &selection_ctx, selected) ||
            selected.slot_index != 0 || selected.score != 0 || selected.used != 5) {
        return 114;
    }
    selection_ctx.states[0] = ggml_moe::object_host_slot_candidate_state::excluded;
    selection_ctx.states[3] = ggml_moe::object_host_slot_candidate_state::excluded;
    if (ggml_moe::object_host_cache_select_slot(
                selection_cache, lru_policy, classify, nullptr, &selection_ctx, selected) ||
            selected.slot_index != SIZE_MAX) {
        return 115;
    }
    if (ggml_moe::object_host_cache_select_slot(
                selection_cache, lru_policy, nullptr, nullptr, &selection_ctx, selected) ||
            ggml_moe::object_host_cache_select_slot(
                selection_cache, score_policy, classify, nullptr, &selection_ctx, selected)) {
        return 116;
    }

    ggml_moe::object_host_admission_policy admission;
    using admission_decision = ggml_moe::object_host_admission_decision;
    if (ggml_moe::object_host_cache_evaluate_admission(admission, 0, 0, false, 0) !=
            admission_decision::admit) {
        return 118;
    }
    admission.min_frequency = 3;
    if (ggml_moe::object_host_cache_evaluate_admission(admission, 2, 100, false, 0) !=
            admission_decision::reject_frequency ||
            ggml_moe::object_host_cache_evaluate_admission(admission, 3, 100, false, 0) !=
            admission_decision::admit) {
        return 119;
    }
    admission.min_score = 101;
    if (ggml_moe::object_host_cache_evaluate_admission(admission, 3, 100, false, 0) !=
            admission_decision::reject_score ||
            ggml_moe::object_host_cache_evaluate_admission(admission, 3, 101, false, 0) !=
            admission_decision::admit) {
        return 120;
    }
    admission = {};
    admission.compare_victim = true;
    if (ggml_moe::object_host_cache_evaluate_admission(admission, 0, 99, true, 100) !=
            admission_decision::reject_victim ||
            ggml_moe::object_host_cache_evaluate_admission(admission, 0, 100, true, 100) !=
            admission_decision::admit ||
            ggml_moe::object_host_cache_evaluate_admission(admission, 0, 99, false, 100) !=
            admission_decision::admit) {
        return 121;
    }
    admission.allow_equal_victim = false;
    if (ggml_moe::object_host_cache_evaluate_admission(admission, 0, 100, true, 100) !=
            admission_decision::reject_victim ||
            ggml_moe::object_host_cache_evaluate_admission(admission, 0, 101, true, 100) !=
            admission_decision::admit) {
        return 122;
    }
    admission.allow_equal_victim = true;
    admission.victim_margin = 5;
    if (ggml_moe::object_host_cache_evaluate_admission(admission, 0, 105, true, 100) !=
            admission_decision::reject_victim ||
            ggml_moe::object_host_cache_evaluate_admission(admission, 0, 106, true, 100) !=
            admission_decision::admit ||
            ggml_moe::object_host_cache_evaluate_admission(
                admission, 0, UINT64_MAX, true, UINT64_MAX - 5) !=
            admission_decision::reject_victim ||
            ggml_moe::object_host_cache_evaluate_admission(
                admission, 0, UINT64_MAX, true, UINT64_MAX - 6) !=
            admission_decision::admit) {
        return 123;
    }
    ggml_moe::object_host_layer_quota_policy quota;
    if (!ggml_moe::object_host_cache_quota_allows_victim(
                quota, false, false, -1, 0, 0)) {
        return 124;
    }
    quota.enabled = true;
    quota.incoming_layer = 7;
    quota.incoming_quota = 2;
    if (!ggml_moe::object_host_cache_quota_allows_victim(
                quota, false, false, -1, 0, 0) ||
            !ggml_moe::object_host_cache_quota_allows_victim(
                quota, true, true, 9, 0, 0)) {
        return 125;
    }
    quota.incoming_occupancy = 2;
    if (ggml_moe::object_host_cache_quota_allows_victim(
                quota, false, false, -1, 0, 0) ||
            ggml_moe::object_host_cache_quota_allows_victim(
                quota, true, true, 9, 0, 0) ||
            ggml_moe::object_host_cache_quota_allows_victim(
                quota, true, false, 7, 0, 0) ||
            !ggml_moe::object_host_cache_quota_allows_victim(
                quota, true, true, 7, 0, 0)) {
        return 126;
    }
    quota.incoming_occupancy = 1;
    quota.protect_victim_quota = true;
    if (ggml_moe::object_host_cache_quota_allows_victim(
                quota, true, true, 9, 3, 3) ||
            !ggml_moe::object_host_cache_quota_allows_victim(
                quota, true, true, 9, 4, 3) ||
            !ggml_moe::object_host_cache_quota_allows_victim(
                quota, true, true, 9, 0, 0)) {
        return 127;
    }
    ggml_moe::object_host_slot event_slot;
    cudaStream_t event_stream = nullptr;
    if (cudaStreamCreateWithFlags(&event_stream, cudaStreamNonBlocking) != cudaSuccess) return 128;
    if (ggml_moe::object_host_slot_query_transfer(
                event_slot, ggml_moe::object_host_transfer::d2h) !=
            ggml_moe::object_host_transfer_status::idle ||
            !ggml_moe::object_host_slot_record_transfer(
                event_slot, ggml_moe::object_host_transfer::d2h, event_stream)) {
        cudaStreamDestroy(event_stream);
        return 129;
    }
    const ggml_moe::object_host_transfer_status event_status =
        ggml_moe::object_host_slot_stream_wait_transfer(
            event_slot, ggml_moe::object_host_transfer::d2h, event_stream);
    if ((event_status != ggml_moe::object_host_transfer_status::pending &&
         event_status != ggml_moe::object_host_transfer_status::complete) ||
            cudaStreamSynchronize(event_stream) != cudaSuccess ||
            ggml_moe::object_host_slot_query_transfer(
                event_slot, ggml_moe::object_host_transfer::d2h) !=
                ggml_moe::object_host_transfer_status::complete) {
        cudaStreamDestroy(event_stream);
        return 130;
    }
    ggml_moe::object_host_slot_clear_transfer(
        event_slot, ggml_moe::object_host_transfer::d2h);
    if (ggml_moe::object_host_slot_query_transfer(
                event_slot, ggml_moe::object_host_transfer::d2h) !=
            ggml_moe::object_host_transfer_status::idle) {
        cudaStreamDestroy(event_stream);
        return 131;
    }
    cudaStreamDestroy(event_stream);
    if (event_slot.d2h_done) cudaEventDestroy(event_slot.d2h_done);

    cudaStream_t producer = nullptr;
    cudaStream_t consumer = nullptr;
    if (cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking) != cudaSuccess ||
            cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking) != cudaSuccess) {
        if (producer) cudaStreamDestroy(producer);
        if (consumer) cudaStreamDestroy(consumer);
        return 132;
    }
    if (!ggml_moe::object_host_cache_begin_h2d(cache, 0) ||
            !cache.slots[0].pending_h2d || cache.slots[0].h2d_recorded ||
            ggml_moe::object_host_cache_query_h2d(cache, 0) !=
                ggml_moe::object_host_completion_state::pending ||
            ggml_moe::object_host_cache_stream_wait_h2d(cache, 0, consumer) ||
            ggml_moe::object_host_cache_invalidate_slot(cache, 0) ||
            ggml_moe::object_host_cache_begin_h2d(cache, 0)) {
        cudaStreamDestroy(consumer);
        cudaStreamDestroy(producer);
        return 133;
    }
    if (!ggml_moe::object_host_cache_record_h2d(cache, 0, producer) ||
            !cache.slots[0].h2d_recorded ||
            !ggml_moe::object_host_cache_stream_wait_h2d(cache, 0, consumer) ||
            cudaStreamSynchronize(consumer) != cudaSuccess) {
        cudaStreamDestroy(consumer);
        cudaStreamDestroy(producer);
        return 134;
    }
    const ggml_moe::object_host_completion_state synchronized =
        ggml_moe::object_host_cache_synchronize_h2d(cache, 0);
    if ((synchronized != ggml_moe::object_host_completion_state::complete &&
                synchronized != ggml_moe::object_host_completion_state::idle) ||
            cache.slots[0].pending_h2d || cache.slots[0].h2d_recorded ||
            cache.completion.h2d_begins.load() != 2 ||
            cache.completion.h2d_records.load() != 1 ||
            cache.completion.h2d_completions.load() != 1 ||
            cache.completion.h2d_failures.load() != 1 ||
            cache.completion.h2d_lease_acquires.load() != 1 ||
            cache.completion.h2d_lease_releases.load() != 1 ||
            cache.completion.h2d_stream_waits.load() > 1) {
        cudaStreamDestroy(consumer);
        cudaStreamDestroy(producer);
        return 135;
    }
    if (!ggml_moe::object_host_cache_begin_h2d(cache, 0) ||
            !ggml_moe::object_host_cache_cancel_h2d(cache, 0, producer) ||
            cache.slots[0].pending_h2d || cache.slots[0].h2d_recorded ||
            cache.completion.h2d_begins.load() != 3 ||
            cache.completion.h2d_cancels.load() != 1 ||
            cache.completion.h2d_lease_acquires.load() != 2 ||
            cache.completion.h2d_lease_releases.load() != 2) {
        cudaStreamDestroy(consumer);
        cudaStreamDestroy(producer);
        return 136;
    }
    if (ggml_moe::object_host_cache_begin_h2d(cache, cache.slots.size()) ||
            cache.completion.h2d_begins.load() != 4 ||
            cache.completion.h2d_failures.load() != 2) {
        cudaStreamDestroy(consumer);
        cudaStreamDestroy(producer);
        return 137;
    }
    if (cache.slots[0].h2d_done) {
        if (cudaEventDestroy(cache.slots[0].h2d_done) != cudaSuccess) {
            cudaStreamDestroy(consumer);
            cudaStreamDestroy(producer);
            return 138;
        }
        cache.slots[0].h2d_done = nullptr;
    }
    if (cudaStreamDestroy(consumer) != cudaSuccess ||
            cudaStreamDestroy(producer) != cudaSuccess) {
        return 139;
    }

    ggml_moe::object_host_slab slab;
    if (!ggml_moe::object_host_slab_allocate(slab, 4096, 3, false, 4096)) return 63;
    if (!slab.storage.host || slab.storage.capacity != 12288 || slab.slot_bytes != 4096 ||
            slab.slot_count != 3 || slab.storage.pageable || !slab.storage.owns_memory) {
        return 64;
    }
    void *slot0 = ggml_moe::object_host_slab_slot_pointer(slab, 0);
    void *slot1 = ggml_moe::object_host_slab_slot_pointer(slab, 1);
    void *slot2 = ggml_moe::object_host_slab_slot_pointer(slab, 2);
    if (!slot0 || !slot1 || !slot2 ||
            static_cast<char *>(slot1) - static_cast<char *>(slot0) != 4096 ||
            static_cast<char *>(slot2) - static_cast<char *>(slot1) != 4096 ||
            ggml_moe::object_host_slab_slot_pointer(slab, 3)) {
        return 65;
    }
    if (ggml_moe::object_host_slab_allocate(slab, 4096, 3, false, 4096)) return 66;
    const ggml_moe::object_host_slab &const_slab = slab;
    if (ggml_moe::object_host_slab_slot_pointer(const_slab, 2) != slot2) return 67;
    ggml_moe::object_host_slab_release(slab);
    if (slab.storage.host || slab.storage.capacity != 0 || slab.slot_bytes != 0 || slab.slot_count != 0) {
        return 68;
    }
    if (ggml_moe::object_host_slab_allocate(slab, SIZE_MAX, 2, false, 4096)) return 69;

    ggml_moe::object_host_cache slab_cache;
    if (!ggml_moe::object_host_cache_allocate_slab(slab_cache, 4096, 3, false, 4096)) return 70;
    if (!slab_cache.slab.storage.host || !slab_cache.slab.storage.owns_memory ||
            slab_cache.slots.size() != 3 ||
            slab_cache.max_slots != 3 || slab_cache.max_resident_bytes != 12288 ||
            slab_cache.resident_capacity_bytes != 12288 ||
            slab_cache.resident_capacity_peak_bytes != 12288) {
        return 71;
    }
    for (size_t i = 0; i < slab_cache.slots.size(); ++i) {
        const ggml_moe::object_host_slot &slot = slab_cache.slots[i];
        if (!slot.host || slot.capacity != 4096 || slot.owns_memory ||
                slot.host != ggml_moe::object_host_slab_slot_pointer(slab_cache.slab, i)) {
            return 72;
        }
    }
    if (!ggml_moe::object_host_cache_configure_ownership(slab_cache, 6)) return 140;
    ggml_moe::object_key stage_key;
    stage_key.source_idx = 0;
    stage_key.layer_idx = 7;
    stage_key.expert_idx = 9;
    stage_key.manifest_generation = 11;
    stage_key.offset = 16384;
    stage_key.bytes = 4096;
    ggml_moe::object_host_staging_transaction stage;
    if (!ggml_moe::object_host_cache_begin_staging(slab_cache, 1, stage_key, stage) ||
            !stage.active || stage.slot_index != 1 || stage.slot_generation == 0 ||
            !slab_cache.slots[1].loading || slab_cache.slots[1].ready) {
        return 141;
    }
    std::memset(slab_cache.slots[1].host, 0x5a, stage_key.bytes);
    if (!ggml_moe::object_host_cache_publish_staging(slab_cache, stage, 1234) ||
            stage.active || !slab_cache.slots[1].ready || slab_cache.slots[1].loading ||
            slab_cache.slots[1].read_us != 1234 ||
            !ggml_moe::object_host_cache_find_ready_slot(slab_cache, stage_key, &ready_slot) ||
            ready_slot != 1) {
        return 142;
    }
    stage_key.expert_idx = 10;
    stage_key.offset += 4096;
    if (!ggml_moe::object_host_cache_begin_staging(slab_cache, 1, stage_key, stage) ||
            !stage.active ||
            !ggml_moe::object_host_cache_abort_staging(slab_cache, stage) || stage.active ||
            ggml_moe::object_host_cache_find_ready_slot(slab_cache, stage_key, nullptr) ||
            ggml_moe::object_host_cache_publish_staging(slab_cache, stage, 0)) {
        return 143;
    }
    if (slab_cache.staging.begins.load() != 2 ||
            slab_cache.staging.publishes.load() != 1 ||
            slab_cache.staging.aborts.load() != 1 ||
            slab_cache.staging.failures.load() != 1) {
        return 144;
    }
    if (!ggml_moe::object_host_cache_budget_fits(slab_cache, slab_cache.slots[1], 4096) ||
            ggml_moe::object_host_cache_budget_fits(slab_cache, slab_cache.slots[1], 4097) ||
            ggml_moe::object_host_cache_allocate_slab(slab_cache, 4096, 3, false, 4096)) {
        return 73;
    }
    void *const slab_base = slab_cache.slab.storage.host;
    ggml_moe::object_host_slot_release_memory(slab_cache.slots[0]);
    if (slab_cache.slots[0].host || !slab_cache.slab.storage.host ||
            slab_cache.slots[1].host != static_cast<char *>(slab_base) + 4096) {
        return 74;
    }
    ggml_moe::object_host_cache_release_slab(slab_cache);
    if (slab_cache.slab.storage.host || !slab_cache.slots.empty() ||
            slab_cache.slab.slot_bytes != 0 || slab_cache.slab.slot_count != 0 ||
            slab_cache.max_slots != 0 || slab_cache.max_resident_bytes != 0 ||
            slab_cache.resident_capacity_bytes != 0 ||
            slab_cache.resident_capacity_peak_bytes != 0 ||
            !slab_cache.ready_index.empty() || !slab_cache.ready_entry_by_slot.empty() ||
            slab_cache.ready_count != 0 || slab_cache.ready_tombstones != 0) {
        return 75;
    }
    if (!ggml_moe::object_host_cache_allocate_slab(slab_cache, 4096, 1, true, 4096)) return 76;
    if (slab_cache.slots[0].owns_memory ||
            ggml_moe::object_host_slot_register_pageable(slab_cache.slots[0]) ||
            slab_cache.slots[0].host_register_attempted) {
        return 77;
    }
    ggml_moe::object_host_cache_release_slab(slab_cache);
    return 0;
}

static int run_role_cache() {
    using namespace ggml_moe;

    alignas(64) uint8_t pool[128] = {};
    uintptr_t keys[2] = {0x101, 0x202};
    uint64_t generations[2] = {3, 7};
    size_t payload_bytes[2] = {32, 48};
    role_slot_marker ready_markers[2] = {};
    role_slot_state states[2] = {role_slot_state::h2d_ready, role_slot_state::h2d_ready};
    bool pending[2] = {};
    bool filling[2] = {};
    uint32_t leases[2] = {};
    role_cache_view view{
        pool, sizeof(pool), 64, 2, keys, generations, payload_bytes,
        ready_markers, states, pending, filling, leases,
    };
    if (!role_cache_view_valid(view) || !role_cache_slot_ready(view, 1, keys[1], 48)) return 70;
    if (role_cache_slot_pointer(view, 1, 8) != pool + 72) return 71;

    struct test_role_cache_storage {
        GGML_MOE_FIXED_ROLE_CACHE_IDENTITY_FIELDS(2)
        GGML_MOE_FIXED_ROLE_CACHE_LIFECYCLE_FIELDS(2)
    } storage;
    storage.pool = pool;
    storage.slot_sz = 64;
    storage.n_slots = 2;
    storage.slot_key[1] = keys[1];
    storage.slot_generation[1] = generations[1];
    storage.slot_payload_bytes[1] = payload_bytes[1];
    storage.slot_state[1] = role_slot_state::h2d_ready;
    role_cache_view storage_view = fixed_role_cache_view(&storage);
    if (!role_cache_view_valid(storage_view) ||
            !role_cache_slot_ready(storage_view, 1, keys[1], payload_bytes[1]) ||
            role_cache_slot_pointer(storage_view, 1, 8) != pool + 72 ||
            storage_view.slot_ready_markers != storage.slot_ready_marker ||
            storage_view.slot_lease_events != storage.slot_lease_ready) {
        return 148;
    }

    storage.slot_key[0] = keys[0];
    storage.slot_generation[0] = generations[0];
    if (!role_cache_prepare_fill(
                storage_view, 0, keys[0], generations[0], payload_bytes[0]) ||
            storage.slot_state[0] != role_slot_state::reserved ||
            !storage.slot_filling[0] || storage.slot_pending[0] ||
            storage.slot_payload_bytes[0] != payload_bytes[0] ||
            !role_cache_mark_reading(
                storage_view, 0, keys[0], generations[0], payload_bytes[0]) ||
            storage.slot_state[0] != role_slot_state::reading ||
            role_cache_clear_readiness(storage_view, 0) ||
            !role_cache_mark_failed(storage_view, 0, keys[0], generations[0]) ||
            storage.slot_state[0] != role_slot_state::failed || storage.slot_filling[0] ||
            !role_cache_clear_readiness(storage_view, 0) ||
            storage.slot_state[0] != role_slot_state::free ||
            storage.slot_payload_bytes[0] != 0) {
        return 149;
    }

    ggml_moe_expert_descriptor first;
    if (!role_cache_acquire_descriptor(
            view, 1, keys[1], 0xabc, role::up, 8, first)) return 72;
    if (first.role_ptr != pool + 72 || first.object_generation != 0xabc ||
            first.slot_generation != 7 || first.slot_id != 1 ||
            first.role != static_cast<int32_t>(role::up) || leases[1] != 1) {
        return 73;
    }

    ggml_moe_expert_descriptor second;
    if (!role_cache_acquire_descriptor(
            view, 1, keys[1], 0xabc, role::down, 0, second) || leases[1] != 2) {
        return 74;
    }
    bool last = true;
    if (role_cache_release_descriptor(view, keys[0], first, &last) || last || leases[1] != 2) {
        return 75;
    }
    if (!role_cache_release_descriptor(view, keys[1], first, &last) || last || leases[1] != 1) {
        return 76;
    }
    if (!role_cache_release_descriptor(view, keys[1], second, &last) || !last || leases[1] != 0) {
        return 77;
    }

    pending[1] = true;
    if (role_cache_acquire_descriptor(view, 1, keys[1], 1, role::gate, 0, first)) return 78;
    pending[1] = false;
    states[1] = role_slot_state::reading;
    if (role_cache_acquire_descriptor(view, 1, keys[1], 1, role::gate, 0, first)) return 79;
    states[1] = role_slot_state::h2d_ready;
    if (role_cache_acquire_descriptor(view, 1, keys[1], 1, role::gate, payload_bytes[1], first)) return 80;
    if (role_cache_acquire_descriptor(
            view, 1, keys[1], 1, static_cast<role>(static_cast<uint8_t>(role::count)), 0, first)) {
        return 81;
    }

    role_cache_view invalid = view;
    invalid.pool_bytes = 64;
    if (role_cache_view_valid(invalid)) return 82;
    cudaEvent_t lease_events[2] = {};
    bool lease_pending[2] = {};
    view.slot_lease_events = lease_events;
    view.slot_lease_pending = lease_pending;
    cudaStream_t lease_stream = nullptr;
    if (cudaStreamCreateWithFlags(&lease_stream, cudaStreamNonBlocking) != cudaSuccess) return 83;
    if (!role_cache_record_lease_event(view, 1, lease_stream) ||
            !lease_events[1] || !lease_pending[1] ||
            !role_cache_stream_wait_lease_event(view, 1, lease_stream) ||
            cudaStreamSynchronize(lease_stream) != cudaSuccess ||
            !role_cache_stream_wait_lease_event(view, 1, lease_stream) ||
            lease_pending[1]) {
        cudaStreamDestroy(lease_stream);
        return 84;
    }
    if (!role_cache_record_lease_event(view, 1, lease_stream) ||
            !lease_pending[1] ||
            !role_cache_synchronize_lease_event(view, 1) ||
            lease_pending[1]) {
        cudaStreamDestroy(lease_stream);
        return 90;
    }
    cudaStreamDestroy(lease_stream);
    cudaEventDestroy(lease_events[1]);
    cudaEvent_t ready_events[2] = {};
    view.slot_ready_events = ready_events;
    if (cudaStreamCreateWithFlags(&lease_stream, cudaStreamNonBlocking) != cudaSuccess) {
        return 85;
    }
    pending[1] = true;
    if (!role_cache_ensure_ready_event(view, 1) ||
            !role_cache_record_ready_event(view, 1, lease_stream) ||
            !role_cache_acquire_descriptor_for_stream(
                view, 1, keys[1], 0xdef, role::gate, 0,
                lease_stream, first) ||
            first.role_ptr != pool + 64 || first.object_generation != 0xdef ||
            first.slot_generation != generations[1] || first.slot_id != 1 ||
            first.role != static_cast<int32_t>(role::gate) || leases[1] != 1 ||
            !role_cache_release_descriptor(view, keys[1], first, &last) ||
            !last || leases[1] != 0) {
        cudaStreamDestroy(lease_stream);
        if (ready_events[1]) cudaEventDestroy(ready_events[1]);
        return 145;
    }
    pending[1] = true;
    cudaEvent_t *const saved_ready_events = view.slot_ready_events;
    view.slot_ready_events = nullptr;
    if (role_cache_acquire_descriptor_for_stream(
            view, 1, keys[1], 0xdef, role::gate, 0,
            lease_stream, first) || leases[1] != 0 || first.slot_id != -1) {
        view.slot_ready_events = saved_ready_events;
        cudaStreamDestroy(lease_stream);
        cudaEventDestroy(ready_events[1]);
        return 146;
    }
    view.slot_ready_events = saved_ready_events;
    pending[1] = false;
    states[0] = role_slot_state::reserved;
    filling[0] = true;
    if (role_cache_publish_ready(view, 0, keys[0], generations[0] + 1,
                payload_bytes[0], lease_stream) ||
            role_cache_publish_ready(view, 0, keys[0], generations[0],
                payload_bytes[0] + 1, lease_stream) ||
            !role_cache_publish_ready(view, 0, keys[0], generations[0],
                payload_bytes[0], lease_stream) ||
            ready_markers[0].key != keys[0] ||
            ready_markers[0].generation != generations[0] ||
            ready_markers[0].bytes != payload_bytes[0] ||
            states[0] != role_slot_state::h2d_ready || filling[0] || !pending[0] ||
            !role_cache_synchronize_ready_event(view, 0) || pending[0]) {
        cudaStreamDestroy(lease_stream);
        if (ready_events[0]) cudaEventDestroy(ready_events[0]);
        return 144;
    }
    if (!role_cache_ensure_ready_event(view, 1) ||
            !role_cache_record_ready_event(view, 1, lease_stream)) {
        cudaStreamDestroy(lease_stream);
        cudaEventDestroy(ready_events[0]);
        return 85;
    }
    if (!ready_events[1] || pending[1]) {
        cudaStreamDestroy(lease_stream);
        cudaEventDestroy(ready_events[0]);
        if (ready_events[1]) cudaEventDestroy(ready_events[1]);
        return 87;
    }
    const cudaEvent_t fixed_ready_event = ready_events[1];
    pending[1] = true;
    if (!role_cache_stream_wait_ready_event(view, 1, lease_stream) ||
            cudaStreamSynchronize(lease_stream) != cudaSuccess ||
            !role_cache_synchronize_ready_event(view, 1) ||
            pending[1]) {
        cudaStreamDestroy(lease_stream);
        cudaEventDestroy(ready_events[1]);
        return 86;
    }
    const int32_t ready_slots[2] = {0, 1};
    role_cache_batch_wait_stats wait_stats;
    if (!role_cache_ensure_ready_event(view, 0) ||
            !role_cache_record_ready_event(view, 0, lease_stream) ||
            !role_cache_record_ready_event(view, 1, lease_stream) ||
            ready_events[1] != fixed_ready_event) {
        cudaStreamDestroy(lease_stream);
        if (ready_events[0]) cudaEventDestroy(ready_events[0]);
        cudaEventDestroy(ready_events[1]);
        return 88;
    }
    pending[0] = true;
    pending[1] = true;
    if (!role_cache_synchronize_ready_events(view, ready_slots, 2, &wait_stats) ||
            pending[0] || pending[1] ||
            wait_stats.representative_syncs != 1 ||
            wait_stats.extra_syncs != 0) {
        cudaStreamDestroy(lease_stream);
        cudaEventDestroy(ready_events[0]);
        cudaEventDestroy(ready_events[1]);
        return 89;
    }
    if (!role_cache_clear_readiness(view, 0)) {
        cudaStreamDestroy(lease_stream);
        cudaEventDestroy(ready_events[0]);
        cudaEventDestroy(ready_events[1]);
        return 150;
    }
    void *device_pool = nullptr;
    if (cudaMalloc(&device_pool, sizeof(pool)) != cudaSuccess) {
        cudaStreamDestroy(lease_stream);
        cudaEventDestroy(ready_events[0]);
        cudaEventDestroy(ready_events[1]);
        return 151;
    }
    role_cache_view h2d_view = view;
    h2d_view.pool = device_pool;
    uint8_t host_payload[32];
    std::memset(host_payload, 0x5a, sizeof(host_payload));
    if (!role_cache_prepare_fill(
                h2d_view, 0, keys[0], generations[0], sizeof(host_payload))) {
        cudaFree(device_pool);
        cudaStreamDestroy(lease_stream);
        cudaEventDestroy(ready_events[0]);
        cudaEventDestroy(ready_events[1]);
        return 152;
    }
    const role_cache_h2d_copy h2d_copy{
        h2d_view, 0, keys[0], generations[0], sizeof(host_payload),
        device_pool, host_payload, sizeof(host_payload),
    };
    role_cache_h2d_stats h2d_stats;
    if (!role_cache_copy_h2d_batch_async(
                &h2d_copy, 1, lease_stream, &h2d_stats) ||
            h2d_stats.copies != 1 || h2d_stats.bytes != sizeof(host_payload) ||
            states[0] != role_slot_state::h2d_ready || filling[0] || !pending[0]) {
        cudaFree(device_pool);
        cudaStreamDestroy(lease_stream);
        cudaEventDestroy(ready_events[0]);
        cudaEventDestroy(ready_events[1]);
        return 153;
    }
    uint8_t copied_payload[32] = {};
    if (!role_cache_synchronize_ready_event(h2d_view, 0) ||
            cudaMemcpy(copied_payload, device_pool, sizeof(copied_payload),
                cudaMemcpyDeviceToHost) != cudaSuccess ||
            std::memcmp(copied_payload, host_payload, sizeof(host_payload)) != 0) {
        cudaFree(device_pool);
        cudaStreamDestroy(lease_stream);
        cudaEventDestroy(ready_events[0]);
        cudaEventDestroy(ready_events[1]);
        return 154;
    }
    cudaFree(device_pool);
    cudaStreamDestroy(lease_stream);
    cudaEventDestroy(ready_events[0]);
    cudaEventDestroy(ready_events[1]);
    return 0;
}

static int run_device_descriptor_buffer() {
    using namespace ggml_moe;

    device_descriptor_buffer buffer;
    if (device_descriptor_buffer_allocate_fixed(buffer, 0) ||
            buffer.allocation_attempts != 1 || buffer.allocation_failures != 1) {
        return 91;
    }
    if (!device_descriptor_buffer_allocate_fixed(buffer, 8) || !buffer.device ||
            buffer.capacity != 8 || buffer.bytes != 8 * sizeof(ggml_moe_expert_descriptor) ||
            buffer.allocations != 1) {
        return 92;
    }
    ggml_moe_expert_descriptor *const fixed_address = buffer.device;
    if (!device_descriptor_buffer_allocate_fixed(buffer, 4) ||
            buffer.device != fixed_address || buffer.allocation_reuses != 1) {
        return 93;
    }
    if (device_descriptor_buffer_allocate_fixed(buffer, 9) ||
            device_descriptor_buffer_allocate_fixed(buffer, SIZE_MAX) ||
            buffer.device != fixed_address || buffer.capacity != 8 ||
            buffer.allocation_failures != 3) {
        return 94;
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return 95;
    ggml_moe_expert_descriptor input[2] = {};
    input[0] = {reinterpret_cast<const void *>(uintptr_t{0x1000}), 11, 21, 1, 0};
    input[1] = {reinterpret_cast<const void *>(uintptr_t{0x2000}), 12, 22, 2, 1};
    if (!device_descriptor_buffer_copy_async(buffer, nullptr, 0, stream) ||
            !device_descriptor_buffer_copy_async(buffer, input, 2, stream) ||
            device_descriptor_buffer_copy_async(buffer, input, 9, stream)) {
        cudaStreamDestroy(stream);
        return 96;
    }
    ggml_moe_expert_descriptor output[2] = {};
    if (cudaMemcpyAsync(output, buffer.device, sizeof(output),
                cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
        cudaStreamDestroy(stream);
        return 97;
    }
    if (output[0].role_ptr != input[0].role_ptr ||
            output[0].object_generation != input[0].object_generation ||
            output[0].slot_generation != input[0].slot_generation ||
            output[0].slot_id != input[0].slot_id || output[0].role != input[0].role ||
            output[1].role_ptr != input[1].role_ptr ||
            output[1].object_generation != input[1].object_generation ||
            output[1].slot_generation != input[1].slot_generation ||
            output[1].slot_id != input[1].slot_id || output[1].role != input[1].role ||
            buffer.copy_calls != 1 || buffer.copy_failures != 1 ||
            buffer.copied_entries != 2 ||
            buffer.copied_bytes != 2 * sizeof(ggml_moe_expert_descriptor)) {
        cudaStreamDestroy(stream);
        return 98;
    }

    device_descriptor_buffer_release(buffer);
    if (buffer.device || buffer.capacity != 0 || buffer.bytes != 0 ||
            device_descriptor_buffer_copy_async(buffer, input, 1, stream) ||
            buffer.copy_failures != 2) {
        cudaStreamDestroy(stream);
        return 99;
    }
    if (cudaStreamDestroy(stream) != cudaSuccess) return 100;
    return 0;
}

static int run_dependency_transfer_plan() {
    using namespace ggml_moe;

    const object_shape shape = make_shape(10, 20, 30);
    const object_request request = make_request(3, 10, 4096, shape);
    const uint8_t full_mask =
        role_bit(role::gate) | role_bit(role::up) | role_bit(role::down);
    const uint8_t critical_mask = role_bit(role::gate) | role_bit(role::up);
    dependency_transfer_plan plan;

    if (build_dependency_transfer_plan(request, 0, critical_mask, plan) !=
            dependency_transfer_status::planned ||
            plan.required_role_mask != full_mask || plan.resident_role_mask != 0 ||
            plan.missing_role_mask != full_mask ||
            plan.critical_role_mask != critical_mask ||
            plan.deferred_role_mask != role_bit(role::down) ||
            plan.critical_count != 2 || plan.deferred_count != 1 ||
            plan.critical_bytes != 30 || plan.deferred_bytes != 30 ||
            !plan.all_required_roles_missing || !plan.overlap_available ||
            plan.critical[0].kind != role::gate || plan.critical[0].host_offset != 0 ||
            plan.critical[0].bytes != 10 || plan.critical[1].kind != role::up ||
            plan.critical[1].host_offset != 10 || plan.critical[1].bytes != 20 ||
            plan.deferred[0].kind != role::down || plan.deferred[0].host_offset != 30 ||
            plan.deferred[0].bytes != 30) {
        return 160;
    }

    const uint8_t gate_down_resident = role_bit(role::gate) | role_bit(role::down);
    if (build_dependency_transfer_plan(
                request, gate_down_resident, critical_mask, plan) !=
            dependency_transfer_status::planned ||
            plan.missing_role_mask != role_bit(role::up) ||
            plan.critical_count != 1 || plan.critical[0].kind != role::up ||
            plan.critical_bytes != 20 || plan.deferred_count != 0 ||
            plan.deferred_bytes != 0 || plan.all_required_roles_missing ||
            plan.overlap_available) {
        return 161;
    }

    if (build_dependency_transfer_plan(request, full_mask, critical_mask, plan) !=
            dependency_transfer_status::ready ||
            plan.missing_role_mask != 0 || plan.critical_count != 0 ||
            plan.deferred_count != 0 || plan.critical_bytes != 0 ||
            plan.deferred_bytes != 0 || plan.overlap_available) {
        return 162;
    }

    const uint8_t gate_up_resident = role_bit(role::gate) | role_bit(role::up);
    if (build_dependency_transfer_plan(
                request, gate_up_resident, critical_mask, plan) !=
            dependency_transfer_status::planned ||
            plan.critical_count != 0 || plan.deferred_count != 1 ||
            plan.deferred[0].kind != role::down || plan.deferred_bytes != 30 ||
            plan.overlap_available) {
        return 163;
    }

    object_request two_roles = make_request(3, 11, 8192, shape, 2);
    if (build_dependency_transfer_plan(two_roles, 0, critical_mask, plan) !=
            dependency_transfer_status::planned ||
            plan.required_role_mask != critical_mask || plan.critical_count != 2 ||
            plan.deferred_count != 0 || plan.critical_bytes != 30) {
        return 164;
    }

    object_request invalid = request;
    invalid.roles[1].relative_offset = 5;
    if (build_dependency_transfer_plan(invalid, 0, critical_mask, plan) !=
            dependency_transfer_status::invalid) {
        return 165;
    }
    invalid = request;
    invalid.required_role_mask &= ~role_bit(role::down);
    if (build_dependency_transfer_plan(invalid, 0, critical_mask, plan) !=
            dependency_transfer_status::invalid) {
        return 166;
    }
    if (build_dependency_transfer_plan(
                two_roles, role_bit(role::down), critical_mask, plan) !=
            dependency_transfer_status::invalid) {
        return 167;
    }
    return 0;
}

static int run_object_pipeline_policy() {
    using namespace ggml_moe;

    const object_pipeline_policy decode =
        object_pipeline_default_policy(object_pipeline_phase::decode);
    const object_pipeline_policy prompt =
        object_pipeline_default_policy(object_pipeline_phase::prompt);
    if (!object_pipeline_policy_valid(decode) || !object_pipeline_policy_valid(prompt) ||
            decode.io_depth != 16 || decode.initial_batch != 8 ||
            decode.refill_batch != 4 || decode.min_flush_batch != 2 ||
            decode.flush_timeout_us != 40 || decode.max_inflight_h2d != 8 ||
            prompt.initial_batch != 16 || prompt.refill_batch != 8 ||
            prompt.min_flush_batch != 8 || prompt.flush_timeout_us != 150) {
        return 170;
    }

    object_pipeline_queue_state state;
    object_pipeline_dispatch_plan plan = object_pipeline_plan_dispatch(decode, state);
    if (plan.reason != object_pipeline_dispatch_reason::idle || plan.submit_count != 0) {
        return 171;
    }

    state.queued = 1;
    state.oldest_wait_us = 10;
    plan = object_pipeline_plan_dispatch(decode, state);
    if (plan.reason != object_pipeline_dispatch_reason::wait_timeout ||
            plan.wait_remaining_us != 30 || plan.submit_count != 0) {
        return 172;
    }

    state.oldest_wait_us = 40;
    plan = object_pipeline_plan_dispatch(decode, state);
    if (plan.reason != object_pipeline_dispatch_reason::timeout ||
            plan.submit_count != 1 || plan.batch_limit != 8 || plan.io_capacity != 16) {
        return 173;
    }

    state = {};
    state.queued = 20;
    plan = object_pipeline_plan_dispatch(decode, state);
    if (plan.reason != object_pipeline_dispatch_reason::batch_ready ||
            plan.submit_count != 8 || plan.batch_limit != 8) {
        return 174;
    }

    state.inflight_io = 6;
    plan = object_pipeline_plan_dispatch(decode, state);
    if (plan.reason != object_pipeline_dispatch_reason::batch_ready ||
            plan.submit_count != 4 || plan.batch_limit != 4 || plan.io_capacity != 10) {
        return 175;
    }

    state = {};
    state.queued = 1;
    state.force = true;
    plan = object_pipeline_plan_dispatch(decode, state);
    if (plan.reason != object_pipeline_dispatch_reason::forced || plan.submit_count != 1) {
        return 176;
    }

    state = {};
    state.queued = 4;
    state.inflight_io = decode.io_depth;
    plan = object_pipeline_plan_dispatch(decode, state);
    if (plan.reason != object_pipeline_dispatch_reason::io_backpressure ||
            plan.submit_count != 0) {
        return 177;
    }

    state.inflight_io = 0;
    state.inflight_h2d = decode.max_inflight_h2d;
    plan = object_pipeline_plan_dispatch(decode, state);
    if (plan.reason != object_pipeline_dispatch_reason::h2d_backpressure ||
            plan.submit_count != 0) {
        return 178;
    }

    object_pipeline_policy invalid = decode;
    invalid.refill_batch = invalid.io_depth + 1;
    if (object_pipeline_policy_valid(invalid) ||
            object_pipeline_plan_dispatch(invalid, state).reason !=
                object_pipeline_dispatch_reason::invalid) {
        return 179;
    }
    invalid = decode;
    invalid.flush_timeout_us = object_pipeline_max_flush_timeout_us + 1;
    if (object_pipeline_policy_valid(invalid)) return 226;

    object_pipeline_refill_cursor cursor;
    if (!cursor.reset(10) || cursor.total() != 10 || cursor.next() != 0 ||
            cursor.queued() != 10 || cursor.inflight_io() != 0 ||
            cursor.completed_io() != 0 || cursor.done() ||
            cursor.ticket().ticket_id == 0 || cursor.ticket().generation == 0 ||
            !object_pipeline_ticket_conserved(cursor.ticket())) {
        return 197;
    }
    plan = cursor.plan(decode, 0, 0, true);
    if (plan.reason != object_pipeline_dispatch_reason::forced ||
            plan.submit_count != 8 || !cursor.record_submit(plan.submit_count) ||
            cursor.next() != 8 || cursor.inflight_io() != 8 ||
            cursor.ticket().submitted != 8 ||
            cursor.record_submit(3)) {
        return 198;
    }
    plan = cursor.plan(decode, 0, 0, true);
    if (plan.reason != object_pipeline_dispatch_reason::forced ||
            plan.submit_count != 2 || !cursor.record_submit(plan.submit_count) ||
            !cursor.record_io_completion(3) || cursor.next() != 10 ||
            cursor.inflight_io() != 7 || cursor.completed_io() != 3) {
        return 199;
    }
    plan = cursor.plan(decode, 0, 0, true);
    if (plan.reason != object_pipeline_dispatch_reason::idle ||
            plan.submit_count != 0 || !cursor.record_io_completion(7) ||
            !cursor.done() ||
            !object_pipeline_ticket_terminal(cursor.ticket()) ||
            cursor.record_io_completion() || cursor.record_submit(1)) {
        return 200;
    }

    const object_pipeline_policy rolling{8, 4, 2, 1, 1, 8};
    object_pipeline_refill_cursor runtime_cursor;
    if (!object_pipeline_refill_reset(runtime_cursor, 6)) return 201;
    plan = object_pipeline_refill_plan(runtime_cursor, rolling, 0, 0, true);
    if (plan.submit_count != 4 ||
            !object_pipeline_refill_record_submit(runtime_cursor, plan.submit_count) ||
            !object_pipeline_refill_record_io_completion(runtime_cursor)) {
        return 202;
    }
    plan = object_pipeline_refill_plan(runtime_cursor, rolling, 0, 0, true);
    if (plan.submit_count != 2 ||
            !object_pipeline_refill_record_submit(runtime_cursor, plan.submit_count) ||
            runtime_cursor.inflight_io() != 5 || runtime_cursor.queued() != 0) {
        return 203;
    }
    if (!object_pipeline_refill_record_io_completion(runtime_cursor, 5) ||
            !object_pipeline_refill_done(runtime_cursor) ||
            !object_pipeline_ticket_terminal(runtime_cursor.ticket())) {
        return 204;
    }

    object_pipeline_refill_cursor timeout_cursor;
    const object_pipeline_policy timeout_policy{4, 4, 2, 2, 10, 4};
    if (!object_pipeline_refill_reset(timeout_cursor, 1)) return 223;
    const object_pipeline_dispatch_runtime_snapshot timeout_before =
        object_pipeline_dispatch_runtime_snapshot_load();
    plan = object_pipeline_refill_plan_ready(
        timeout_cursor, timeout_policy, 0, 0, true);
    const object_pipeline_dispatch_runtime_snapshot timeout_after =
        object_pipeline_dispatch_runtime_snapshot_load();
    if (plan.reason != object_pipeline_dispatch_reason::timeout ||
            plan.submit_count != 1 ||
            timeout_after.timeout_flush_rounds != timeout_before.timeout_flush_rounds + 1 ||
            timeout_after.timeout_wait_us != timeout_before.timeout_wait_us + 10) {
        return 224;
    }
    plan = object_pipeline_refill_plan_ready(
        timeout_cursor, timeout_policy, 0, 0, false);
    if (plan.reason != object_pipeline_dispatch_reason::forced ||
            plan.submit_count != 1) {
        return 225;
    }

    object_pipeline_ticket ticket;
    if (!object_pipeline_ticket_reset(ticket, 77, 9, 7) ||
            !object_pipeline_ticket_record_submit(ticket, 4) ||
            !object_pipeline_ticket_record_completion(ticket, 3) ||
            !object_pipeline_ticket_record_critical_ready(ticket, 2) ||
            !object_pipeline_ticket_record_failure(ticket) ||
            !object_pipeline_ticket_record_cancellation(ticket, 3) ||
            !object_pipeline_ticket_terminal(ticket) ||
            ticket.submitted != 4 || ticket.completed != 3 ||
            ticket.failed != 1 || ticket.cancelled != 3 ||
            ticket.critical_ready != 2 ||
            object_pipeline_ticket_record_completion(ticket) ||
            object_pipeline_ticket_record_submit(ticket, 1)) {
        return 205;
    }

    const uint64_t previous_ticket = runtime_cursor.ticket().ticket_id;
    const uint64_t previous_generation = runtime_cursor.ticket().generation;
    if (!object_pipeline_refill_reset(runtime_cursor, 1) ||
            runtime_cursor.ticket().ticket_id == previous_ticket ||
            runtime_cursor.ticket().generation == previous_generation ||
            !object_pipeline_ticket_conserved(runtime_cursor.ticket())) {
        return 206;
    }

    object_pipeline_mode parsed_mode = object_pipeline_mode::persistent_demand;
    if (!object_pipeline_parse_mode("sync", parsed_mode) ||
            parsed_mode != object_pipeline_mode::sync_qd1 ||
            !object_pipeline_parse_mode("persistent_demand", parsed_mode) ||
            parsed_mode != object_pipeline_mode::persistent_demand ||
            !object_pipeline_parse_mode("rolling_refill", parsed_mode) ||
            parsed_mode != object_pipeline_mode::rolling_refill ||
            object_pipeline_parse_mode("invalid", parsed_mode)) {
        return 207;
    }

    const char *pipeline_envs[] = {
        "GGML_MOE_PIPELINE_MODE",
        "GGML_MOE_PIPELINE_IO_DEPTH",
        "GGML_MOE_PIPELINE_INITIAL_BATCH",
        "GGML_MOE_PIPELINE_REFILL_BATCH",
        "GGML_MOE_PIPELINE_MIN_FLUSH_BATCH",
        "GGML_MOE_PIPELINE_FLUSH_TIMEOUT_US",
    };
    for (const char *name : pipeline_envs) unsetenv(name);

    object_pipeline_mode_config mode_config = object_pipeline_resolve_mode(
        object_pipeline_mode::persistent_demand, false);
    if (!mode_config.valid ||
            mode_config.mode != object_pipeline_mode::persistent_demand ||
            mode_config.source != object_pipeline_config_source::model_default) {
        return 208;
    }
    mode_config = object_pipeline_resolve_mode(
        object_pipeline_mode::persistent_demand, true);
    if (!mode_config.valid || mode_config.mode != object_pipeline_mode::rolling_refill ||
            mode_config.source != object_pipeline_config_source::legacy) {
        return 209;
    }
    if (setenv("GGML_MOE_PIPELINE_MODE", "sync_qd1", 1) != 0) return 210;
    mode_config = object_pipeline_resolve_mode(
        object_pipeline_mode::rolling_refill, false);
    if (!mode_config.valid || mode_config.mode != object_pipeline_mode::sync_qd1 ||
            mode_config.source != object_pipeline_config_source::common) {
        return 211;
    }
    if (setenv("GGML_MOE_PIPELINE_MODE", "bad-mode", 1) != 0) return 212;
    mode_config = object_pipeline_resolve_mode(
        object_pipeline_mode::rolling_refill, false);
    if (mode_config.valid || mode_config.mode != object_pipeline_mode::rolling_refill ||
            mode_config.source != object_pipeline_config_source::common) {
        return 213;
    }
    unsetenv("GGML_MOE_PIPELINE_MODE");

    const object_pipeline_policy fallback_policy{16, 8, 4, 2, 40, 8};
    if (setenv("GGML_MOE_PIPELINE_IO_DEPTH", "8", 1) != 0 ||
            setenv("GGML_MOE_PIPELINE_INITIAL_BATCH", "6", 1) != 0 ||
            setenv("GGML_MOE_PIPELINE_REFILL_BATCH", "3", 1) != 0 ||
            setenv("GGML_MOE_PIPELINE_MIN_FLUSH_BATCH", "2", 1) != 0 ||
            setenv("GGML_MOE_PIPELINE_FLUSH_TIMEOUT_US", "50", 1) != 0) {
        return 214;
    }
    object_pipeline_policy_config policy_config =
        object_pipeline_resolve_policy(fallback_policy);
    if (!policy_config.valid || policy_config.override_mask != 0x1f ||
            policy_config.invalid_mask != 0 || policy_config.policy.io_depth != 8 ||
            policy_config.policy.initial_batch != 6 || policy_config.policy.refill_batch != 3 ||
            policy_config.policy.min_flush_batch != 2 ||
            policy_config.policy.flush_timeout_us != 50) {
        return 215;
    }
    if (setenv("GGML_MOE_PIPELINE_REFILL_BATCH", "9", 1) != 0) return 216;
    policy_config = object_pipeline_resolve_policy(fallback_policy);
    if (policy_config.valid || policy_config.policy.io_depth != fallback_policy.io_depth ||
            policy_config.policy.refill_batch != fallback_policy.refill_batch) {
        return 217;
    }
    if (setenv("GGML_MOE_PIPELINE_REFILL_BATCH", "3", 1) != 0 ||
            setenv("GGML_MOE_PIPELINE_FLUSH_TIMEOUT_US", "12us", 1) != 0) {
        return 218;
    }
    policy_config = object_pipeline_resolve_policy(fallback_policy);
    if (policy_config.valid ||
            (policy_config.invalid_mask & object_pipeline_override_flush_timeout_us) == 0) {
        return 219;
    }
    for (const char *name : pipeline_envs) unsetenv(name);

    object_pipeline_metrics metrics;
    metrics.timing_mask = object_pipeline_timing_foreground_wait |
        object_pipeline_timing_ssd_read;
    metrics.submitted = 10;
    metrics.completed = 7;
    metrics.failed = 1;
    metrics.cancelled = 2;
    metrics.inflight_sum = 12;
    metrics.inflight_samples = 4;
    metrics.inflight_max = 5;
    metrics.foreground_wait_us = 80;
    metrics.ssd_reads = 6;
    metrics.ssd_bytes = 600;
    metrics.ssd_read_us = 30;
    if (!object_pipeline_metrics_valid(metrics)) return 220;
    metrics.failed = 2;
    if (object_pipeline_metrics_valid(metrics)) return 221;
    metrics.failed = 1;
    metrics.ssd_bytes = 0;
    if (object_pipeline_metrics_valid(metrics)) return 222;
    return 0;
}

static int run_object_transaction_rendezvous() {
    using namespace ggml_moe;

    object_layout layout;
    std::strcpy(layout.model, "transaction-test");
    layout.source_idx = 9;
    layout.first_layer = 2;
    layout.layer_count = 1;
    layout.first_expert = 4;
    layout.experts_per_layer = 1;
    layout.top_k = 1;
    layout.object_count = 1;
    layout.manifest_generation = 0x1234;
    layout.shape_count = 1;
    layout.shapes[0] = make_shape(10, 20, 30);
    if (!object_engine_configure(layout)) return 180;

    object_request request = make_request(2, 4, 0, layout.shapes[0]);
    request.key.source_idx = 9;
    request.key.manifest_generation = 0x1234;
    object_transaction_ticket owner;
    if (object_engine_acquire_transaction(request, owner) !=
            object_transaction_acquire_status::owner ||
            !owner.owner || owner.transaction == 0) {
        return 181;
    }
    object_transaction_ticket joined;
    if (object_engine_acquire_transaction(request, joined) !=
            object_transaction_acquire_status::joined ||
            joined.owner || joined.transaction != owner.transaction) {
        return 182;
    }
    const uint8_t critical = role_bit(role::gate) | role_bit(role::up);
    if (object_engine_wait_transaction(joined, critical, 0) !=
            object_transaction_wait_status::timeout ||
            !object_engine_publish_transaction_roles(owner, critical) ||
            object_engine_wait_transaction(joined, critical, 0) !=
                object_transaction_wait_status::ready) {
        return 183;
    }
    if (object_engine_publish_transaction_roles(joined, role_bit(role::down)) ||
            object_engine_publish_transaction_roles(owner, uint8_t{0x8})) {
        return 184;
    }
    if (!object_engine_complete_batch(&request, 1, true) ||
            object_engine_wait_transaction(joined, role_bit(role::down), 0) !=
                object_transaction_wait_status::completed ||
            object_engine_publish_transaction_roles(owner, critical)) {
        return 185;
    }

    object_transaction_ticket failed_owner;
    object_transaction_ticket failed_join;
    if (object_engine_acquire_transaction(request, failed_owner) !=
            object_transaction_acquire_status::owner ||
            failed_owner.transaction == owner.transaction ||
            object_engine_acquire_transaction(request, failed_join) !=
                object_transaction_acquire_status::joined ||
            !object_engine_complete_batch(&request, 1, false) ||
            object_engine_wait_transaction(failed_join, critical, 0) !=
                object_transaction_wait_status::failed) {
        return 186;
    }
    object_transaction_ticket invalid;
    object_request stale = request;
    stale.key.object_generation = 7;
    if (object_engine_acquire_transaction(stale, invalid) !=
            object_transaction_acquire_status::invalid ||
            object_engine_wait_transaction({}, critical, 0) !=
                object_transaction_wait_status::invalid) {
        return 187;
    }
    return 0;
}

static int run_object_pipeline_lifetime() {
    using namespace ggml_moe;

    object_pipeline_transaction_pool pool;
    if (!pool.configure(2) || !pool.configure(2) || pool.configure(3) ||
            pool.capacity() != 2 || pool.in_use() != 0) {
        return 190;
    }
    object_pipeline_transaction_handle first;
    object_pipeline_transaction_handle second;
    object_pipeline_transaction_handle blocked;
    if (!pool.acquire(first) || !pool.acquire(second) || pool.acquire(blocked) ||
            pool.in_use() != 2 || first.slot == second.slot ||
            first.generation == 0 || second.generation == 0) {
        return 191;
    }
    if (pool.wait_critical(first, 0) !=
            object_pipeline_transaction_wait_status::timeout ||
            !pool.publish_critical(first, true) ||
            pool.publish_critical(first, true) ||
            pool.wait_critical(first, 0) !=
                object_pipeline_transaction_wait_status::critical_ready) {
        return 192;
    }
    if (!pool.release_caller(first) || pool.release_caller(first) ||
            pool.acquire(blocked) || !pool.complete_worker(first, true) ||
            pool.in_use() != 1 || !pool.acquire(blocked) ||
            blocked.slot != first.slot || blocked.generation == first.generation) {
        return 193;
    }
    if (pool.publish_critical(first, true) || pool.complete_worker(first, true) ||
            pool.release_caller(first)) {
        return 194;
    }
    if (!pool.complete_worker(blocked, false) ||
            pool.wait_critical(blocked, 0) !=
                object_pipeline_transaction_wait_status::failed ||
            !pool.release_caller(blocked)) {
        return 195;
    }
    if (!pool.complete_worker(second, true) ||
            pool.wait_critical(second, 0) !=
                object_pipeline_transaction_wait_status::completed ||
            !pool.release_caller(second) || pool.in_use() != 0) {
        return 196;
    }
    return 0;
}

int main(int argc, char **argv) {
    using namespace ggml_moe;

    if (argc == 2 && std::strcmp(argv[1], "single") == 0) return run_single_shape();
    if (argc == 2 && std::strcmp(argv[1], "stage") == 0) return run_stage_ring();
    if (argc == 2 && std::strcmp(argv[1], "host") == 0) return run_host_slot();
    if (argc == 2 && std::strcmp(argv[1], "role") == 0) return run_role_cache();
    if (argc == 2 && std::strcmp(argv[1], "descriptor") == 0) return run_device_descriptor_buffer();
    if (argc == 2 && std::strcmp(argv[1], "transfer") == 0) return run_dependency_transfer_plan();
    if (argc == 2 && std::strcmp(argv[1], "pipeline") == 0) return run_object_pipeline_policy();
    if (argc == 2 && std::strcmp(argv[1], "transaction") == 0) return run_object_transaction_rendezvous();
    if (argc == 2 && std::strcmp(argv[1], "pipeline-lifetime") == 0) return run_object_pipeline_lifetime();
    if (argc == 2 && std::strcmp(argv[1], "kimi-iq1-adapter") == 0) return run_kimi_iq1_adapter();

    object_layout layout;
    std::strcpy(layout.model, "shape-test");
    layout.source_idx = 7;
    layout.first_layer = 3;
    layout.layer_count = 2;
    layout.first_expert = 10;
    layout.experts_per_layer = 2;
    layout.top_k = 2;
    layout.object_count = 4;
    layout.manifest_generation = 0xabc;
    layout.shape_count = 2;
    layout.shapes[0] = make_shape(10, 20, 30);
    layout.shapes[1] = make_shape(15, 15, 30);
    const uint8_t shape_indices[] = {0, 1, 0, 1};
    layout.object_shape_indices = shape_indices;
    layout.object_shape_index_count = 4;

    // Shape classes may have the same object size but different role extents.
    if (layout.shapes[0].object_bytes != layout.shapes[1].object_bytes) return 1;
    if (!object_engine_configure(layout)) return 2;
    if (!object_engine_configure_host_cache(0, 8) ||
            !object_engine_configure_host_cache(0, 8) ||
            object_engine_configure_host_cache(0, 9)) {
        return 23;
    }
    if (!object_engine_configure_role_cache(0, 16) ||
            !object_engine_configure_role_cache(1, 16) ||
            !object_engine_configure_role_cache(0, 16) ||
            object_engine_configure_role_cache(0, 17)) {
        return 21;
    }

    object_request full = make_request(3, 10, 4096, layout.shapes[0]);
    if (!object_engine_publish_host_slot(full.key, 0, 2, 40)) return 24;
    object_record full_record;
    if (!object_engine_get_record(full.key, full_record) ||
            full_record.host_cache_id != 0 || full_record.host_slot != 2 ||
            full_record.host_slot_generation != 40 || full_record.host_lease_count != 0) {
        return 25;
    }
    if (!object_engine_acquire_host_slot(full.key, 0, 2, 40) ||
            object_engine_invalidate_host_slot(0, 2, 40) ||
            object_engine_release_host_slot(full.key, 0, 2, 41) ||
            !object_engine_release_host_slot(full.key, 0, 2, 40) ||
            !object_engine_invalidate_host_slot(0, 2, 40) ||
            !object_engine_publish_host_slot(full.key, 0, 2, 41)) {
        return 26;
    }
    object_request host_conflict = make_request(3, 11, 8192, layout.shapes[1], 2);
    object_record host_conflict_record;
    if (object_engine_publish_host_slot(host_conflict.key, 0, 2, 41) ||
            object_engine_publish_host_slot(full.key, 0, 3, 41) ||
            object_engine_get_record(host_conflict.key, host_conflict_record)) {
        return 27;
    }
    object_host_cache shared_host_cache;
    shared_host_cache.slots.resize(4);
    if (!object_host_cache_configure_ownership(shared_host_cache, 1) ||
            !object_host_cache_configure_ownership(shared_host_cache, 1) ||
            object_host_cache_configure_ownership(shared_host_cache, 2) ||
            !object_host_cache_prepare_slot(shared_host_cache, 1)) {
        return 28;
    }
    shared_host_cache.slots[1].marker = host_conflict.key;
    shared_host_cache.slots[1].ready = true;
    size_t shared_ready_slot = SIZE_MAX;
    if (shared_host_cache.slots[1].generation != 1 ||
            !object_host_cache_publish_slot(shared_host_cache, 1) ||
            !object_host_cache_find_ready_slot(
                shared_host_cache, host_conflict.key, &shared_ready_slot) ||
            shared_ready_slot != 1 || object_host_cache_ready_count(shared_host_cache) != 1 ||
            !object_host_cache_acquire_slot_lease(shared_host_cache, 1) ||
            object_host_cache_invalidate_slot(shared_host_cache, 1) ||
            !object_host_cache_release_slot_lease(shared_host_cache, 1) ||
            !object_host_cache_invalidate_slot(shared_host_cache, 1) ||
            object_host_cache_find_ready_slot(shared_host_cache, host_conflict.key, nullptr) ||
            object_host_cache_ready_count(shared_host_cache) != 0 ||
            !object_engine_get_record(host_conflict.key, host_conflict_record) ||
            host_conflict_record.host_cache_id != -1 || host_conflict_record.host_slot != -1 ||
            host_conflict_record.host_slot_generation != 0 || host_conflict_record.host_lease_count != 0) {
        return 29;
    }
    if (!object_engine_begin_batch(&full, 1)) return 3;
    object_residency_update full_update;
    full_update.valid_role_mask = full.required_role_mask;
    full_update.resident_role_mask = full.required_role_mask;
    for (size_t i = 0; i < 3; ++i) {
        full_update.role_cache_id[i] = static_cast<int32_t>(i % 2);
        full_update.role_slot[i] = static_cast<int32_t>(4 + i);
        full_update.role_slot_generation[i] = 100 + i;
        full_update.role_lease_count[i] = static_cast<uint32_t>(i);
    }
    if (!object_engine_complete_batch_with_residency(&full, &full_update, 1, true)) return 4;
    if (!object_engine_get_record(full.key, full_record) ||
            full_record.generation != full.key.manifest_generation ||
            full_record.pack_offset != full.key.offset || full_record.inflight_transaction != 0 ||
            full_record.host_cache_id != 0 || full_record.host_slot != 2 ||
            full_record.host_slot_generation != 41 || full_record.host_lease_count != 0 ||
            full_record.resident_mask != full.required_role_mask ||
            full_record.pending_mask != 0) {
        return 16;
    }
    for (size_t i = 0; i < 3; ++i) {
        if (full_record.role_cache_id[i] != full_update.role_cache_id[i] ||
                full_record.role_slot[i] != full_update.role_slot[i] ||
                full_record.role_slot_generation[i] != full_update.role_slot_generation[i] ||
                full_record.role_lease_count[i] != full_update.role_lease_count[i]) {
            return 17;
        }
    }
    if (!object_engine_invalidate_role_slot(
                full_update.role_cache_id[0], full_update.role_slot[0],
                full_update.role_slot_generation[0]) ||
            !object_engine_get_record(full.key, full_record) ||
            (full_record.resident_mask & role_bit(role::gate)) != 0 ||
            full_record.role_cache_id[0] != -1 || full_record.role_slot[0] != -1 ||
            full_record.role_slot_generation[0] != 0 ||
            object_engine_invalidate_role_slot(
                full_update.role_cache_id[1], full_update.role_slot[1],
                full_update.role_slot_generation[1] + 1)) {
        return 22;
    }
    object_key stale_full_key = full.key;
    ++stale_full_key.offset;
    if (object_engine_get_record(stale_full_key, full_record) ||
            object_engine_publish_host_slot(stale_full_key, 0, 3, 1)) {
        return 18;
    }

    object_request partial = host_conflict;
    if (!object_engine_begin_batch(&partial, 1)) return 5;
    object_residency_update invalid_update;
    invalid_update.valid_role_mask = partial.required_role_mask;
    invalid_update.resident_role_mask = partial.required_role_mask;
    invalid_update.role_cache_id[0] = 0;
    invalid_update.role_cache_id[1] = 1;
    invalid_update.role_slot_generation[0] = 1;
    invalid_update.role_slot_generation[1] = 1;
    if (object_engine_complete_batch_with_residency(&partial, &invalid_update, 1, true)) return 19;
    object_record partial_record;
    if (!object_engine_get_record(partial.key, partial_record) ||
            partial_record.inflight_transaction == 0 ||
            partial_record.pending_mask != partial.required_role_mask) {
        return 20;
    }
    if (!object_engine_complete_batch(&partial, 1, true)) return 6;

    object_request wrong_object_shape = make_request(4, 10, 12288, layout.shapes[1]);
    if (object_engine_begin_batch(&wrong_object_shape, 1)) return 7;

    object_request invalid_extent = partial;
    ++invalid_extent.roles[1].bytes;
    if (object_engine_begin_batch(&invalid_extent, 1)) return 8;

    object_request invalid_size = partial;
    ++invalid_size.key.bytes;
    if (object_engine_begin_batch(&invalid_size, 1)) return 9;

    object_request duplicate = make_request(4, 10, 16384, layout.shapes[0]);
    if (!object_engine_begin_batch(&duplicate, 1)) return 10;
    if (object_engine_begin_batch(&duplicate, 1)) return 11;
    if (!object_engine_complete_batch(&duplicate, 1, true)) return 12;

    object_request duplicate_batch[] = {
        make_request(4, 11, 20480, layout.shapes[1]),
        make_request(4, 11, 20480, layout.shapes[1]),
    };
    if (object_engine_begin_batch(duplicate_batch, 2)) return 13;
    // A rejected enforcing batch must not leave a partial transaction behind.
    if (!object_engine_begin_batch(&duplicate_batch[0], 1)) return 14;
    if (!object_engine_complete_batch(&duplicate_batch[0], 1, true)) return 15;
    return 0;
}
