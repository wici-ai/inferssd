#include "moe-common-object-engine.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace ggml_moe_common {
namespace {

struct shadow_record {
    object_record record;
    object_key inflight_key;
};

struct role_slot_owner {
    size_t object_index = SIZE_MAX;
    uint64_t slot_generation = 0;
    uint8_t role_index = 0;
};

struct host_slot_owner {
    size_t object_index = SIZE_MAX;
    uint64_t slot_generation = 0;
};

struct queued_batch {
    void *context = nullptr;
    bool speculative = false;
    uint32_t kind = 0;
    uint8_t priority = 0;
};

struct shadow_state {
    std::mutex mutex;
    std::condition_variable transaction_cv;
    object_layout layout;
    std::vector<uint8_t> object_shape_indices;
    std::vector<shadow_record> records;
    std::vector<host_slot_owner> host_slot_owners[max_host_cache_classes];
    std::vector<role_slot_owner> role_slot_owners[max_role_cache_classes];
    bool configured = false;
    bool report_registered = false;
    uint64_t next_transaction = 1;
    uint64_t submitted_batches = 0;
    uint64_t submitted_objects = 0;
    uint64_t submitted_roles = 0;
    uint64_t submitted_bytes = 0;
    uint64_t speculative_objects = 0;
    uint64_t completed_objects = 0;
    uint64_t failed_objects = 0;
    uint64_t duplicate_inflight = 0;
    uint64_t joined_inflight = 0;
    uint64_t transaction_role_publishes = 0;
    uint64_t transaction_waits = 0;
    uint64_t transaction_wait_timeouts = 0;
    uint64_t unmatched_completions = 0;
    uint64_t stale_completions = 0;
    uint64_t invalid_requests = 0;
    uint64_t host_publishes = 0;
    uint64_t host_invalidations = 0;
    uint64_t host_acquires = 0;
    uint64_t host_releases = 0;
    uint64_t host_ownership_failures = 0;
    uint64_t role_mask_counts[8] = {};
    uint64_t request_fingerprint = UINT64_C(1469598103934665603);
    std::deque<queued_batch> queue;
    std::condition_variable queue_cv;
    std::thread worker;
    object_worker_provider provider;
    bool worker_active = false;
    bool worker_started = false;
    bool worker_stop = false;
    bool worker_failed = false;
    uint64_t enqueued_batches = 0;
    uint64_t dequeued_batches = 0;
    uint64_t demand_dequeues = 0;
    uint64_t speculative_dequeues = 0;
    size_t max_queue_depth = 0;
    uint64_t generic_enqueued = 0;
    uint64_t worker_processed = 0;
    uint64_t worker_process_failed = 0;
    uint64_t worker_affinity_attempts = 0;
    uint64_t worker_affinity_successes = 0;
    uint64_t worker_affinity_failures = 0;
};

shadow_state g_shadow;

enum class engine_mode : uint8_t {
    off,
    shadow,
    kimi,
    deepseek,
    glm52,
    glm52_enforce,
};

static engine_mode mode() {
    static const engine_mode value = [] {
        const char *env = std::getenv("GGML_MOE_OBJECT_ENGINE");
        if (!env || !env[0] || std::strcmp(env, "off") == 0) return engine_mode::off;
        if (std::strcmp(env, "shadow") == 0) return engine_mode::shadow;
        if (std::strcmp(env, "kimi") == 0) return engine_mode::kimi;
        if (std::strcmp(env, "deepseek") == 0 || std::strcmp(env, "ds") == 0) {
            return engine_mode::deepseek;
        }
        if (std::strcmp(env, "glm52") == 0 || std::strcmp(env, "glm") == 0) {
            return engine_mode::glm52;
        }
        if (std::strcmp(env, "glm52-enforce") == 0 || std::strcmp(env, "glm-enforce") == 0) {
            return engine_mode::glm52_enforce;
        }
        return engine_mode::off;
    }();
    return value;
}

static const char *mode_name() {
    switch (mode()) {
        case engine_mode::off: return "off";
        case engine_mode::shadow: return "shadow";
        case engine_mode::kimi: return "kimi";
        case engine_mode::deepseek: return "deepseek";
        case engine_mode::glm52: return "glm52";
        case engine_mode::glm52_enforce: return "glm52-enforce";
    }
    return "off";
}

enum class worker_affinity_status : uint8_t {
    not_requested,
    success,
    failure,
};

static worker_affinity_status apply_worker_affinity(
        const object_worker_provider &provider) {
    const char *env_name = provider.cpu_affinity_env;
    const char *spec = env_name ? std::getenv(env_name) : nullptr;
    if ((!spec || !spec[0]) && provider.legacy_cpu_affinity_env) {
        env_name = provider.legacy_cpu_affinity_env;
        spec = std::getenv(env_name);
    }
    if (!spec || !spec[0]) return worker_affinity_status::not_requested;

#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    bool any = false;
    const char *cursor = spec;
    while (*cursor) {
        while (*cursor == ' ' || *cursor == ',') ++cursor;
        if (!*cursor) break;
        char *end = nullptr;
        const long first = std::strtol(cursor, &end, 10);
        if (end == cursor || first < 0 || first >= CPU_SETSIZE) {
            any = false;
            break;
        }
        long last = first;
        cursor = end;
        if (*cursor == '-') {
            ++cursor;
            last = std::strtol(cursor, &end, 10);
            if (end == cursor || last < first || last >= CPU_SETSIZE) {
                any = false;
                break;
            }
            cursor = end;
        }
        for (long cpu = first; cpu <= last; ++cpu) CPU_SET((int)cpu, &cpuset);
        any = true;
        while (*cursor == ' ') ++cursor;
        if (*cursor && *cursor != ',') {
            any = false;
            break;
        }
    }
    const int rc = any ? pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) : -1;
    std::fprintf(stderr,
            "[moe_object_engine] worker affinity: role=%s env=%s cpus=%s status=%s rc=%d\n",
            provider.worker_role ? provider.worker_role : "object-worker",
            env_name ? env_name : "none", spec,
            any && rc == 0 ? "ok" : "failed", rc);
    return any && rc == 0 ? worker_affinity_status::success : worker_affinity_status::failure;
#else
    std::fprintf(stderr,
            "[moe_object_engine] worker affinity: role=%s env=%s cpus=%s status=unsupported\n",
            provider.worker_role ? provider.worker_role : "object-worker",
            env_name ? env_name : "none", spec);
    return worker_affinity_status::failure;
#endif
}

static uint64_t fingerprint_mix(uint64_t hash, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        hash ^= value & 0xff;
        hash *= UINT64_C(1099511628211);
        value >>= 8;
    }
    return hash;
}

static bool direct_index_locked(const object_key &key, size_t *index_out) {
    if (!g_shadow.configured || key.source_idx != g_shadow.layout.source_idx ||
            key.manifest_generation != g_shadow.layout.manifest_generation ||
            key.layer_idx < g_shadow.layout.first_layer ||
            key.expert_idx < g_shadow.layout.first_expert) {
        return false;
    }
    const uint64_t layer = static_cast<uint64_t>(key.layer_idx - g_shadow.layout.first_layer);
    const uint64_t expert = static_cast<uint64_t>(key.expert_idx - g_shadow.layout.first_expert);
    if (layer >= g_shadow.layout.layer_count || expert >= g_shadow.layout.experts_per_layer) {
        return false;
    }
    const uint64_t index = layer * g_shadow.layout.experts_per_layer + expert;
    if (index >= g_shadow.records.size()) return false;
    if (index_out) *index_out = static_cast<size_t>(index);
    return true;
}

static bool extent_matches_shape(const role_extent &extent, const object_shape &shape) {
    const size_t role_index = static_cast<size_t>(extent.kind);
    if (role_index >= static_cast<size_t>(role::count)) return false;
    const role_extent &expected = shape.roles[role_index];
    return expected.kind == extent.kind &&
        expected.relative_offset == extent.relative_offset &&
        expected.bytes == extent.bytes;
}

static bool request_matches_shape(const object_request &request, const object_shape &shape) {
    if (request.key.bytes != shape.object_bytes) return false;
    for (uint8_t i = 0; i < request.role_count; ++i) {
        if (!extent_matches_shape(request.roles[i], shape)) return false;
    }
    return true;
}

static bool request_valid_locked(const object_request &request, size_t *index_out) {
    size_t index = 0;
    if (request.key.bytes == 0 || request.role_count == 0 || request.role_count > 3 ||
            request.required_role_mask == 0 || (request.required_role_mask & ~uint8_t{0x7}) != 0 ||
            !direct_index_locked(request.key, &index)) {
        return false;
    }
    uint8_t observed_mask = 0;
    for (uint8_t i = 0; i < request.role_count; ++i) {
        const role_extent &extent = request.roles[i];
        const size_t role_index = static_cast<size_t>(extent.kind);
        if (role_index >= static_cast<size_t>(role::count)) return false;
        const uint8_t bit = role_bit(extent.kind);
        if (extent.bytes == 0 || observed_mask & bit ||
                extent.relative_offset > request.key.bytes ||
                extent.bytes > request.key.bytes - extent.relative_offset) {
            return false;
        }
        observed_mask |= bit;
    }
    if (observed_mask != request.required_role_mask) return false;
    const uint8_t shape_index = g_shadow.object_shape_indices.empty() ?
        uint8_t{0} : g_shadow.object_shape_indices[index];
    if (shape_index >= g_shadow.layout.shape_count ||
            !request_matches_shape(request, g_shadow.layout.shapes[shape_index])) {
        return false;
    }
    if (index_out) *index_out = index;
    return true;
}

static bool object_key_valid_locked(const object_key &key, size_t *index_out) {
    size_t index = 0;
    if (key.bytes == 0 || !direct_index_locked(key, &index)) return false;
    const uint8_t shape_index = g_shadow.object_shape_indices.empty() ?
        uint8_t{0} : g_shadow.object_shape_indices[index];
    if (shape_index >= g_shadow.layout.shape_count ||
            key.bytes != g_shadow.layout.shapes[shape_index].object_bytes) {
        return false;
    }
    if (index_out) *index_out = index;
    return true;
}

static bool record_identity_matches_locked(const object_record &record, const object_key &key) {
    return record.generation == object_key_effective_generation(key) &&
        record.pack_offset == key.offset;
}

static bool record_identity_bind_locked(object_record &record, const object_key &key) {
    if (record.generation != 0) return record_identity_matches_locked(record, key);
    record.generation = object_key_effective_generation(key);
    record.pack_offset = key.offset;
    return true;
}

static void report() {
    if (!object_engine_enabled()) return;
    std::lock_guard<std::mutex> lock(g_shadow.mutex);
    if (!g_shadow.configured) return;
    uint64_t inflight = 0;
    for (const shadow_record &record : g_shadow.records) {
        if (record.record.inflight_transaction != 0) ++inflight;
    }
    std::fprintf(stderr,
            "[moe_object_engine] mode=%s model=%s configured=1 objects=%zu "
            "batches=%lu requests=%lu roles=%lu bytes=%lu speculative=%lu "
            "completed=%lu failed=%lu inflight=%lu queued=%zu enqueued=%lu dequeued=%lu "
            "demand_dequeues=%lu speculative_dequeues=%lu max_queue_depth=%zu generic_enqueued=%lu "
            "worker_active=%d worker_init_failed=%d worker_processed=%lu worker_process_failed=%lu "
            "worker_affinity_attempts=%lu worker_affinity_successes=%lu worker_affinity_failures=%lu "
            "duplicate_inflight=%lu joined_inflight=%lu role_publishes=%lu "
            "transaction_waits=%lu transaction_wait_timeouts=%lu "
            "unmatched_completion=%lu stale_completion=%lu invalid=%lu "
            "host_publishes=%lu host_invalidations=%lu host_acquires=%lu host_releases=%lu "
            "host_ownership_failures=%lu "
            "mask_gate=%lu mask_up=%lu mask_down=%lu mask_gate_up=%lu "
            "mask_gate_down=%lu mask_up_down=%lu mask_gate_up_down=%lu fingerprint=%016lx\n",
            mode_name(), g_shadow.layout.model,
            g_shadow.records.size(),
            (unsigned long)g_shadow.submitted_batches,
            (unsigned long)g_shadow.submitted_objects,
            (unsigned long)g_shadow.submitted_roles,
            (unsigned long)g_shadow.submitted_bytes,
            (unsigned long)g_shadow.speculative_objects,
            (unsigned long)g_shadow.completed_objects,
            (unsigned long)g_shadow.failed_objects,
            (unsigned long)inflight,
            g_shadow.queue.size(),
            (unsigned long)g_shadow.enqueued_batches,
            (unsigned long)g_shadow.dequeued_batches,
            (unsigned long)g_shadow.demand_dequeues,
            (unsigned long)g_shadow.speculative_dequeues,
            g_shadow.max_queue_depth,
            (unsigned long)g_shadow.generic_enqueued,
            g_shadow.worker_active ? 1 : 0,
            g_shadow.worker_failed ? 1 : 0,
            (unsigned long)g_shadow.worker_processed,
            (unsigned long)g_shadow.worker_process_failed,
            (unsigned long)g_shadow.worker_affinity_attempts,
            (unsigned long)g_shadow.worker_affinity_successes,
            (unsigned long)g_shadow.worker_affinity_failures,
            (unsigned long)g_shadow.duplicate_inflight,
            (unsigned long)g_shadow.joined_inflight,
            (unsigned long)g_shadow.transaction_role_publishes,
            (unsigned long)g_shadow.transaction_waits,
            (unsigned long)g_shadow.transaction_wait_timeouts,
            (unsigned long)g_shadow.unmatched_completions,
            (unsigned long)g_shadow.stale_completions,
            (unsigned long)g_shadow.invalid_requests,
            (unsigned long)g_shadow.host_publishes,
            (unsigned long)g_shadow.host_invalidations,
            (unsigned long)g_shadow.host_acquires,
            (unsigned long)g_shadow.host_releases,
            (unsigned long)g_shadow.host_ownership_failures,
            (unsigned long)g_shadow.role_mask_counts[role_bit(role::gate)],
            (unsigned long)g_shadow.role_mask_counts[role_bit(role::up)],
            (unsigned long)g_shadow.role_mask_counts[role_bit(role::down)],
            (unsigned long)g_shadow.role_mask_counts[role_bit(role::gate) | role_bit(role::up)],
            (unsigned long)g_shadow.role_mask_counts[role_bit(role::gate) | role_bit(role::down)],
            (unsigned long)g_shadow.role_mask_counts[role_bit(role::up) | role_bit(role::down)],
            (unsigned long)g_shadow.role_mask_counts[role_bit(role::gate) | role_bit(role::up) | role_bit(role::down)],
            (unsigned long)g_shadow.request_fingerprint);
}

} // namespace

bool object_engine_enabled() {
    return mode() != engine_mode::off;
}

bool object_engine_enforcing() {
    return mode() == engine_mode::kimi || mode() == engine_mode::deepseek ||
        mode() == engine_mode::glm52_enforce;
}

static bool shape_valid(const object_shape &shape) {
    if (shape.object_bytes == 0) return false;
    size_t covered_bytes = 0;
    for (size_t role_index = 0; role_index < static_cast<size_t>(role::count); ++role_index) {
        const role_extent &extent = shape.roles[role_index];
        if (extent.kind != static_cast<role>(role_index) || extent.bytes == 0 ||
                extent.relative_offset != covered_bytes ||
                extent.relative_offset > shape.object_bytes ||
                extent.bytes > shape.object_bytes - extent.relative_offset) {
            return false;
        }
        covered_bytes += extent.bytes;
    }
    return covered_bytes == shape.object_bytes;
}

static bool layouts_match_locked(const object_layout &left, const object_layout &right) {
    if (std::strcmp(left.model, right.model) != 0 ||
            left.source_idx != right.source_idx ||
            left.first_layer != right.first_layer ||
            left.layer_count != right.layer_count ||
            left.first_expert != right.first_expert ||
            left.experts_per_layer != right.experts_per_layer ||
            left.top_k != right.top_k ||
            left.object_count != right.object_count ||
            left.manifest_generation != right.manifest_generation ||
            left.shape_count != right.shape_count) {
        return false;
    }
    for (uint8_t shape_index = 0; shape_index < left.shape_count; ++shape_index) {
        const object_shape &left_shape = left.shapes[shape_index];
        const object_shape &right_shape = right.shapes[shape_index];
        if (left_shape.object_bytes != right_shape.object_bytes) return false;
        for (size_t role_index = 0; role_index < static_cast<size_t>(role::count); ++role_index) {
            const role_extent &left_extent = left_shape.roles[role_index];
            const role_extent &right_extent = right_shape.roles[role_index];
            if (left_extent.kind != right_extent.kind ||
                    left_extent.relative_offset != right_extent.relative_offset ||
                    left_extent.bytes != right_extent.bytes) {
                return false;
            }
        }
    }
    if (right.object_shape_index_count != 0) {
        if (!right.object_shape_indices || right.object_shape_index_count != right.object_count ||
                g_shadow.object_shape_indices.size() != right.object_shape_index_count) {
            return false;
        }
        for (size_t i = 0; i < right.object_shape_index_count; ++i) {
            if (g_shadow.object_shape_indices[i] != right.object_shape_indices[i]) return false;
        }
    } else if (!g_shadow.object_shape_indices.empty()) {
        return false;
    }
    return true;
}

bool object_engine_configure(const object_layout &layout) {
    if (!object_engine_enabled()) return true;
    if (layout.model[0] == '\0' || layout.source_idx < 0 || layout.layer_count == 0 ||
            layout.experts_per_layer == 0 || layout.top_k == 0 ||
            layout.top_k > layout.experts_per_layer || layout.object_count == 0 ||
            layout.manifest_generation == 0 ||
            layout.object_count != static_cast<uint64_t>(layout.layer_count) * layout.experts_per_layer ||
            layout.shape_count == 0 || layout.shape_count > max_object_shape_classes) {
        return false;
    }
    for (uint8_t i = 0; i < layout.shape_count; ++i) {
        if (!shape_valid(layout.shapes[i])) return false;
    }
    if ((layout.object_shape_index_count != 0 &&
                (!layout.object_shape_indices || layout.object_shape_index_count != layout.object_count)) ||
            (layout.shape_count > 1 && layout.object_shape_index_count != layout.object_count)) {
        return false;
    }
    for (size_t i = 0; i < layout.object_shape_index_count; ++i) {
        if (layout.object_shape_indices[i] >= layout.shape_count) return false;
    }
    std::lock_guard<std::mutex> lock(g_shadow.mutex);
    if (g_shadow.configured) {
        return layouts_match_locked(g_shadow.layout, layout);
    }
    g_shadow.layout = layout;
    if (layout.object_shape_index_count != 0) {
        g_shadow.object_shape_indices.assign(
                layout.object_shape_indices,
                layout.object_shape_indices + layout.object_shape_index_count);
    }
    g_shadow.layout.object_shape_indices = nullptr;
    g_shadow.layout.object_shape_index_count = g_shadow.object_shape_indices.size();
    g_shadow.records.resize(static_cast<size_t>(layout.object_count));
    g_shadow.configured = true;
    if (!g_shadow.report_registered) {
        std::atexit(report);
        g_shadow.report_registered = true;
    }
    std::fprintf(stderr,
            "[moe_object_engine] configured mode=%s model=%s source=%d layers=%u first_layer=%d "
            "experts=%u first_expert=%d top_k=%u objects=%lu generation=%016lx shapes=%u object_bytes=%zu\n",
            mode_name(), layout.model, layout.source_idx, layout.layer_count, layout.first_layer,
            layout.experts_per_layer, layout.first_expert, layout.top_k,
            (unsigned long)layout.object_count,
            (unsigned long)layout.manifest_generation, (unsigned)layout.shape_count,
            layout.shapes[0].object_bytes);
    for (uint8_t i = 0; i < layout.shape_count; ++i) {
        const object_shape &shape = layout.shapes[i];
        std::fprintf(stderr,
                "[moe_object_engine] shape=%u object_bytes=%zu gate=%zu up=%zu down=%zu\n",
                (unsigned)i, shape.object_bytes,
                shape.roles[static_cast<size_t>(role::gate)].bytes,
                shape.roles[static_cast<size_t>(role::up)].bytes,
                shape.roles[static_cast<size_t>(role::down)].bytes);
    }
    return true;
}

bool object_engine_configure_host_cache(int32_t cache_id, size_t slot_count) {
    if (!object_engine_enabled()) return true;
    if (cache_id < 0 || static_cast<size_t>(cache_id) >= max_host_cache_classes ||
            slot_count == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_shadow.mutex);
    std::vector<host_slot_owner> &owners = g_shadow.host_slot_owners[cache_id];
    if (!owners.empty()) return owners.size() == slot_count;
    owners.resize(slot_count);
    return true;
}

static bool host_slot_identity_valid_locked(
        int32_t cache_id,
        int32_t slot_id,
        uint64_t slot_generation) {
    return cache_id >= 0 && static_cast<size_t>(cache_id) < max_host_cache_classes &&
        slot_id >= 0 && slot_generation != 0 &&
        static_cast<size_t>(slot_id) < g_shadow.host_slot_owners[cache_id].size();
}

bool object_engine_publish_host_slot(
        const object_key &key,
        int32_t cache_id,
        int32_t slot_id,
        uint64_t slot_generation) {
    if (!object_engine_enabled()) return true;
    std::lock_guard<std::mutex> lock(g_shadow.mutex);
    size_t index = 0;
    if (!host_slot_identity_valid_locked(cache_id, slot_id, slot_generation) ||
            !object_key_valid_locked(key, &index)) {
        ++g_shadow.host_ownership_failures;
        return false;
    }
    object_record &record = g_shadow.records[index].record;
    host_slot_owner &owner = g_shadow.host_slot_owners[cache_id][slot_id];
    if ((record.generation != 0 && !record_identity_matches_locked(record, key)) ||
            (record.host_slot >= 0 &&
             (record.host_cache_id != cache_id || record.host_slot != slot_id ||
              record.host_slot_generation != slot_generation)) ||
            (owner.object_index != SIZE_MAX &&
             (owner.object_index != index || owner.slot_generation != slot_generation))) {
        ++g_shadow.host_ownership_failures;
        return false;
    }
    if (!record_identity_bind_locked(record, key)) {
        ++g_shadow.host_ownership_failures;
        return false;
    }
    record.host_cache_id = cache_id;
    record.host_slot = slot_id;
    record.host_slot_generation = slot_generation;
    owner.object_index = index;
    owner.slot_generation = slot_generation;
    ++g_shadow.host_publishes;
    return true;
}

static bool object_engine_change_host_lease_locked(
        const object_key &key,
        int32_t cache_id,
        int32_t slot_id,
        uint64_t slot_generation,
        bool acquire) {
    size_t index = 0;
    if (!host_slot_identity_valid_locked(cache_id, slot_id, slot_generation) ||
            !object_key_valid_locked(key, &index)) {
        return false;
    }
    object_record &record = g_shadow.records[index].record;
    const host_slot_owner &owner = g_shadow.host_slot_owners[cache_id][slot_id];
    if (!record_identity_matches_locked(record, key) || owner.object_index != index ||
            owner.slot_generation != slot_generation || record.host_cache_id != cache_id ||
            record.host_slot != slot_id || record.host_slot_generation != slot_generation) {
        return false;
    }
    if (acquire) {
        if (record.host_lease_count == UINT32_MAX) return false;
        ++record.host_lease_count;
        ++g_shadow.host_acquires;
    } else {
        if (record.host_lease_count == 0) return false;
        --record.host_lease_count;
        ++g_shadow.host_releases;
    }
    return true;
}

bool object_engine_acquire_host_slot(
        const object_key &key,
        int32_t cache_id,
        int32_t slot_id,
        uint64_t slot_generation) {
    if (!object_engine_enabled()) return true;
    std::lock_guard<std::mutex> lock(g_shadow.mutex);
    const bool ok = object_engine_change_host_lease_locked(
            key, cache_id, slot_id, slot_generation, true);
    if (!ok) ++g_shadow.host_ownership_failures;
    return ok;
}

bool object_engine_release_host_slot(
        const object_key &key,
        int32_t cache_id,
        int32_t slot_id,
        uint64_t slot_generation) {
    if (!object_engine_enabled()) return true;
    std::lock_guard<std::mutex> lock(g_shadow.mutex);
    const bool ok = object_engine_change_host_lease_locked(
            key, cache_id, slot_id, slot_generation, false);
    if (!ok) ++g_shadow.host_ownership_failures;
    return ok;
}

bool object_engine_invalidate_host_slot(
        int32_t cache_id,
        int32_t slot_id,
        uint64_t slot_generation) {
    if (!object_engine_enabled()) return true;
    std::lock_guard<std::mutex> lock(g_shadow.mutex);
    if (!host_slot_identity_valid_locked(cache_id, slot_id, slot_generation)) {
        ++g_shadow.host_ownership_failures;
        return false;
    }
    host_slot_owner &owner = g_shadow.host_slot_owners[cache_id][slot_id];
    if (owner.object_index == SIZE_MAX) return true;
    if (owner.slot_generation != slot_generation || owner.object_index >= g_shadow.records.size()) {
        ++g_shadow.host_ownership_failures;
        return false;
    }
    object_record &record = g_shadow.records[owner.object_index].record;
    if (record.host_cache_id != cache_id || record.host_slot != slot_id ||
            record.host_slot_generation != slot_generation || record.host_lease_count != 0) {
        ++g_shadow.host_ownership_failures;
        return false;
    }
    record.host_cache_id = -1;
    record.host_slot = -1;
    record.host_slot_generation = 0;
    owner = {};
    ++g_shadow.host_invalidations;
    return true;
}

bool object_engine_configure_role_cache(int32_t cache_id, size_t slot_count) {
    if (!object_engine_enabled()) return true;
    if (cache_id < 0 || static_cast<size_t>(cache_id) >= max_role_cache_classes ||
            slot_count == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_shadow.mutex);
    std::vector<role_slot_owner> &owners = g_shadow.role_slot_owners[cache_id];
    if (!owners.empty()) return owners.size() == slot_count;
    owners.resize(slot_count);
    return true;
}

static bool clear_role_slot_owner_locked(
        int32_t cache_id,
        int32_t slot_id,
        uint64_t slot_generation,
        bool require_generation) {
    if (cache_id < 0 || static_cast<size_t>(cache_id) >= max_role_cache_classes || slot_id < 0) {
        return false;
    }
    std::vector<role_slot_owner> &owners = g_shadow.role_slot_owners[cache_id];
    if (static_cast<size_t>(slot_id) >= owners.size()) return false;
    role_slot_owner &owner = owners[slot_id];
    if (owner.object_index == SIZE_MAX) return true;
    if (require_generation && owner.slot_generation != slot_generation) return false;
    if (owner.object_index >= g_shadow.records.size() ||
            owner.role_index >= static_cast<uint8_t>(role::count)) {
        return false;
    }
    object_record &record = g_shadow.records[owner.object_index].record;
    const size_t role_index = owner.role_index;
    if (record.role_cache_id[role_index] != cache_id ||
            record.role_slot[role_index] != slot_id ||
            record.role_slot_generation[role_index] != owner.slot_generation ||
            record.role_lease_count[role_index] != 0) {
        return false;
    }
    record.resident_mask &= ~uint32_t{role_bit(static_cast<role>(role_index))};
    record.role_cache_id[role_index] = -1;
    record.role_slot[role_index] = -1;
    record.role_slot_generation[role_index] = 0;
    record.role_lease_count[role_index] = 0;
    owner = {};
    return true;
}

bool object_engine_invalidate_role_slot(
        int32_t cache_id,
        int32_t slot_id,
        uint64_t slot_generation) {
    if (!object_engine_enabled()) return true;
    if (slot_generation == 0) return false;
    std::lock_guard<std::mutex> lock(g_shadow.mutex);
    return clear_role_slot_owner_locked(cache_id, slot_id, slot_generation, true);
}

static bool object_engine_begin_batch_locked(const object_request *requests, size_t count) {
    ++g_shadow.submitted_batches;
    std::vector<size_t> indices(count, SIZE_MAX);
    bool valid_batch = true;
    for (size_t i = 0; i < count; ++i) {
        const object_request &request = requests[i];
        size_t index = 0;
        if (!request_valid_locked(request, &index)) {
            ++g_shadow.invalid_requests;
            valid_batch = false;
            continue;
        }
        bool duplicate_in_batch = false;
        for (size_t previous = 0; previous < i; ++previous) {
            if (indices[previous] == index) {
                duplicate_in_batch = true;
                break;
            }
        }
        if (duplicate_in_batch) {
            ++g_shadow.duplicate_inflight;
            valid_batch = false;
            continue;
        }
        indices[i] = index;
        ++g_shadow.submitted_objects;
        g_shadow.submitted_roles += request.role_count;
        g_shadow.submitted_bytes += request.key.bytes;
        if (request.speculative) ++g_shadow.speculative_objects;
        ++g_shadow.role_mask_counts[request.required_role_mask];
        g_shadow.request_fingerprint = fingerprint_mix(g_shadow.request_fingerprint, index);
        g_shadow.request_fingerprint = fingerprint_mix(g_shadow.request_fingerprint, request.key.offset);
        g_shadow.request_fingerprint = fingerprint_mix(g_shadow.request_fingerprint, request.key.bytes);
        g_shadow.request_fingerprint = fingerprint_mix(
                g_shadow.request_fingerprint, object_key_effective_generation(request.key));
        g_shadow.request_fingerprint = fingerprint_mix(g_shadow.request_fingerprint, request.required_role_mask);

        shadow_record &record = g_shadow.records[index];
        if (record.record.generation != 0 &&
                !record_identity_matches_locked(record.record, request.key)) {
            ++g_shadow.invalid_requests;
            valid_batch = false;
            continue;
        }
        if (record.record.inflight_transaction != 0) {
            ++g_shadow.duplicate_inflight;
            valid_batch = false;
        }
    }
    if (object_engine_enforcing() && !valid_batch) return false;
    for (size_t i = 0; i < count; ++i) {
        if (indices[i] == SIZE_MAX) continue;
        const object_request &request = requests[i];
        shadow_record &record = g_shadow.records[indices[i]];
        if (record.record.inflight_transaction != 0) continue;
        if (!record_identity_bind_locked(record.record, request.key)) continue;
        record.record.pending_mask = request.required_role_mask;
        record.record.inflight_transaction = g_shadow.next_transaction++;
        record.inflight_key = request.key;
    }
    return true;
}

bool object_engine_begin_batch(const object_request *requests, size_t count) {
    if (!object_engine_enabled()) return true;
    if (!requests || count == 0) return false;
    std::lock_guard<std::mutex> lock(g_shadow.mutex);
    return object_engine_begin_batch_locked(requests, count);
}

object_transaction_acquire_status object_engine_acquire_transaction(
        const object_request &request,
        object_transaction_ticket &ticket) {
    ticket = {};
    if (!object_engine_enabled()) return object_transaction_acquire_status::invalid;
    std::lock_guard<std::mutex> lock(g_shadow.mutex);
    size_t index = 0;
    if (!request_valid_locked(request, &index)) {
        ++g_shadow.invalid_requests;
        return object_transaction_acquire_status::invalid;
    }

    shadow_record &record = g_shadow.records[index];
    if (record.record.generation != 0 &&
            !record_identity_matches_locked(record.record, request.key)) {
        ++g_shadow.invalid_requests;
        return object_transaction_acquire_status::invalid;
    }

    ticket.key = request.key;
    ticket.requested_role_mask = request.required_role_mask;
    if (record.record.inflight_transaction != 0) {
        if (!object_key_matches(record.inflight_key, request.key)) {
            ++g_shadow.invalid_requests;
            ticket = {};
            return object_transaction_acquire_status::invalid;
        }
        record.record.pending_mask |= request.required_role_mask;
        ticket.transaction = record.record.inflight_transaction;
        ++g_shadow.joined_inflight;
        return object_transaction_acquire_status::joined;
    }

    if (!record_identity_bind_locked(record.record, request.key)) {
        ++g_shadow.invalid_requests;
        ticket = {};
        return object_transaction_acquire_status::invalid;
    }
    record.record.pending_mask = request.required_role_mask;
    record.record.ready_mask = 0;
    record.record.inflight_transaction = g_shadow.next_transaction++;
    record.inflight_key = request.key;
    ticket.transaction = record.record.inflight_transaction;
    ticket.owner = true;
    ++g_shadow.submitted_batches;
    ++g_shadow.submitted_objects;
    g_shadow.submitted_roles += request.role_count;
    g_shadow.submitted_bytes += request.key.bytes;
    if (request.speculative) ++g_shadow.speculative_objects;
    ++g_shadow.role_mask_counts[request.required_role_mask];
    return object_transaction_acquire_status::owner;
}

bool object_engine_publish_transaction_roles(
        const object_transaction_ticket &ticket,
        uint8_t ready_role_mask) {
    if (!object_engine_enabled() || !ticket.owner || ticket.transaction == 0 ||
            ready_role_mask == 0 || (ready_role_mask & ~uint8_t{0x7}) != 0) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(g_shadow.mutex);
        size_t index = 0;
        if (!object_key_valid_locked(ticket.key, &index)) return false;
        shadow_record &record = g_shadow.records[index];
        if (record.record.inflight_transaction != ticket.transaction ||
                !object_key_matches(record.inflight_key, ticket.key) ||
                (ready_role_mask & ~record.record.pending_mask) != 0) {
            return false;
        }
        record.record.ready_mask |= ready_role_mask;
        ++g_shadow.transaction_role_publishes;
    }
    g_shadow.transaction_cv.notify_all();
    return true;
}

object_transaction_wait_status object_engine_wait_transaction(
        const object_transaction_ticket &ticket,
        uint8_t required_role_mask,
        uint64_t timeout_us) {
    if (!object_engine_enabled() || ticket.transaction == 0 ||
            required_role_mask == 0 ||
            (required_role_mask & ~ticket.requested_role_mask) != 0) {
        return object_transaction_wait_status::invalid;
    }
    std::unique_lock<std::mutex> lock(g_shadow.mutex);
    size_t index = 0;
    if (!object_key_valid_locked(ticket.key, &index)) {
        return object_transaction_wait_status::invalid;
    }
    ++g_shadow.transaction_waits;
    const auto status = [&]() {
        const shadow_record &record = g_shadow.records[index];
        if (record.record.inflight_transaction == ticket.transaction &&
                object_key_matches(record.inflight_key, ticket.key) &&
                (record.record.ready_mask & required_role_mask) == required_role_mask) {
            return object_transaction_wait_status::ready;
        }
        if (record.record.last_transaction == ticket.transaction) {
            return record.record.last_transaction_succeeded ?
                object_transaction_wait_status::completed :
                object_transaction_wait_status::failed;
        }
        if (record.record.inflight_transaction != ticket.transaction ||
                !object_key_matches(record.inflight_key, ticket.key)) {
            return object_transaction_wait_status::invalid;
        }
        return object_transaction_wait_status::timeout;
    };
    object_transaction_wait_status result = status();
    if (result != object_transaction_wait_status::timeout || timeout_us == 0) {
        if (result == object_transaction_wait_status::timeout) {
            ++g_shadow.transaction_wait_timeouts;
        }
        return result;
    }
    const bool notified = g_shadow.transaction_cv.wait_for(
        lock, std::chrono::microseconds(timeout_us), [&] {
            return status() != object_transaction_wait_status::timeout;
        });
    if (!notified) {
        ++g_shadow.transaction_wait_timeouts;
        return object_transaction_wait_status::timeout;
    }
    return status();
}

static bool residency_update_valid_locked(
        const object_request &request,
        const object_residency_update &update) {
    if ((update.valid_role_mask & ~uint8_t{0x7}) != 0 ||
            (update.resident_role_mask & ~update.valid_role_mask) != 0 ||
            (update.valid_role_mask & ~request.required_role_mask) != 0) {
        return false;
    }
    for (size_t role_index = 0; role_index < static_cast<size_t>(role::count); ++role_index) {
        const uint8_t bit = role_bit(static_cast<role>(role_index));
        if ((update.valid_role_mask & bit) == 0) continue;
        if (update.resident_role_mask & bit) {
            if (update.role_cache_id[role_index] < 0 ||
                    static_cast<size_t>(update.role_cache_id[role_index]) >= max_role_cache_classes ||
                    update.role_slot[role_index] < 0 ||
                    update.role_slot_generation[role_index] == 0 ||
                    static_cast<size_t>(update.role_slot[role_index]) >=
                        g_shadow.role_slot_owners[update.role_cache_id[role_index]].size()) {
                return false;
            }
        } else if (update.role_cache_id[role_index] != -1 ||
                update.role_slot[role_index] != -1 ||
                update.role_slot_generation[role_index] != 0 ||
                update.role_lease_count[role_index] != 0) {
            return false;
        }
    }
    return true;
}

bool object_engine_complete_batch_with_residency(
        const object_request *requests,
        const object_residency_update *updates,
        size_t count,
        bool success) {
    if (!object_engine_enabled()) return true;
    if (!requests || count == 0) return false;
    std::lock_guard<std::mutex> lock(g_shadow.mutex);
    std::vector<size_t> indices(count, SIZE_MAX);
    std::vector<uint64_t> completion_slots;
    completion_slots.reserve(count * static_cast<size_t>(role::count));
    bool valid_batch = true;
    for (size_t i = 0; i < count; ++i) {
        const object_request &request = requests[i];
        size_t index = 0;
        if (!request_valid_locked(request, &index)) {
            valid_batch = false;
            continue;
        }
        indices[i] = index;
        shadow_record &record = g_shadow.records[index];
        if (record.record.inflight_transaction == 0) {
            ++g_shadow.unmatched_completions;
            valid_batch = false;
            continue;
        }
        if (!object_key_matches(record.inflight_key, request.key)) {
            ++g_shadow.stale_completions;
            valid_batch = false;
            continue;
        }
        if (success && updates) {
            const object_residency_update &update = updates[i];
            if (!residency_update_valid_locked(request, update)) {
                ++g_shadow.invalid_requests;
                valid_batch = false;
                continue;
            }
            for (size_t role_index = 0; role_index < static_cast<size_t>(role::count); ++role_index) {
                const uint8_t bit = role_bit(static_cast<role>(role_index));
                if ((update.valid_role_mask & bit) == 0) continue;
                if ((update.resident_role_mask & bit) == 0) {
                    if ((record.record.resident_mask & bit) != 0 &&
                            record.record.role_lease_count[role_index] != 0) {
                        valid_batch = false;
                    }
                    continue;
                }
                const uint64_t identity =
                    (static_cast<uint64_t>(static_cast<uint32_t>(update.role_cache_id[role_index])) << 32) |
                    static_cast<uint32_t>(update.role_slot[role_index]);
                if (std::find(completion_slots.begin(), completion_slots.end(), identity) !=
                        completion_slots.end()) {
                    valid_batch = false;
                    continue;
                }
                completion_slots.push_back(identity);
                const role_slot_owner &owner =
                    g_shadow.role_slot_owners[update.role_cache_id[role_index]][update.role_slot[role_index]];
                if (owner.object_index != SIZE_MAX &&
                        (owner.object_index != index || owner.role_index != role_index ||
                         owner.slot_generation != update.role_slot_generation[role_index])) {
                    if (owner.object_index >= g_shadow.records.size() ||
                            owner.role_index >= static_cast<uint8_t>(role::count) ||
                            g_shadow.records[owner.object_index].record.role_lease_count[owner.role_index] != 0) {
                        valid_batch = false;
                    }
                }
            }
        }
    }
    if (!valid_batch) {
        if (!object_engine_enforcing()) {
            for (size_t i = 0; i < count; ++i) {
                if (indices[i] == SIZE_MAX) continue;
                shadow_record &record = g_shadow.records[indices[i]];
                if (record.record.inflight_transaction != 0 &&
                        object_key_matches(record.inflight_key, requests[i].key)) {
                    record.record.inflight_transaction = 0;
                    record.record.pending_mask = 0;
                    ++g_shadow.failed_objects;
                }
            }
        }
        return !object_engine_enforcing();
    }

    for (size_t i = 0; i < count; ++i) {
        shadow_record &record = g_shadow.records[indices[i]];
        if (success && updates) {
            const object_residency_update &update = updates[i];
            for (size_t role_index = 0; role_index < static_cast<size_t>(role::count); ++role_index) {
                const uint8_t bit = role_bit(static_cast<role>(role_index));
                if ((update.valid_role_mask & bit) == 0) continue;
                if (update.resident_role_mask & bit) {
                    role_slot_owner &owner =
                        g_shadow.role_slot_owners[update.role_cache_id[role_index]][update.role_slot[role_index]];
                    if (owner.object_index != SIZE_MAX &&
                            (owner.object_index != indices[i] || owner.role_index != role_index ||
                             owner.slot_generation != update.role_slot_generation[role_index])) {
                        (void)clear_role_slot_owner_locked(
                                update.role_cache_id[role_index], update.role_slot[role_index], 0, false);
                    }
                    record.record.resident_mask |= bit;
                    record.record.role_cache_id[role_index] = update.role_cache_id[role_index];
                    record.record.role_slot[role_index] = update.role_slot[role_index];
                    record.record.role_slot_generation[role_index] =
                        update.role_slot_generation[role_index];
                    record.record.role_lease_count[role_index] = update.role_lease_count[role_index];
                    owner.object_index = indices[i];
                    owner.slot_generation = update.role_slot_generation[role_index];
                    owner.role_index = static_cast<uint8_t>(role_index);
                } else {
                    if (record.record.resident_mask & bit) {
                        (void)clear_role_slot_owner_locked(
                                record.record.role_cache_id[role_index],
                                record.record.role_slot[role_index],
                                record.record.role_slot_generation[role_index],
                                true);
                    }
                    record.record.resident_mask &= ~uint32_t{bit};
                    record.record.role_cache_id[role_index] = -1;
                    record.record.role_slot[role_index] = -1;
                    record.record.role_slot_generation[role_index] = 0;
                    record.record.role_lease_count[role_index] = 0;
                }
            }
        }
        record.record.last_transaction = record.record.inflight_transaction;
        record.record.last_transaction_succeeded = success;
        record.record.inflight_transaction = 0;
        record.record.pending_mask = 0;
        record.record.ready_mask = 0;
        if (success) {
            ++g_shadow.completed_objects;
        } else {
            ++g_shadow.failed_objects;
        }
    }
    g_shadow.transaction_cv.notify_all();
    return true;
}

bool object_engine_complete_batch(const object_request *requests, size_t count, bool success) {
    return object_engine_complete_batch_with_residency(requests, nullptr, count, success);
}

bool object_engine_get_record(const object_key &key, object_record &record) {
    if (!object_engine_enabled()) return false;
    std::lock_guard<std::mutex> lock(g_shadow.mutex);
    size_t index = 0;
    if (!direct_index_locked(key, &index)) return false;
    const object_record &current = g_shadow.records[index].record;
    if (current.generation != object_key_effective_generation(key) ||
            current.pack_offset != key.offset) {
        return false;
    }
    record = current;
    return true;
}

bool object_engine_enqueue_batch(
        void *context,
        const object_request *requests,
        size_t count,
        bool speculative,
        size_t capacity) {
    if (!object_engine_enforcing() || !context || capacity == 0) return false;
    std::lock_guard<std::mutex> lock(g_shadow.mutex);
    if (g_shadow.worker_stop || g_shadow.queue.size() >= capacity || !requests || count == 0 ||
            !object_engine_begin_batch_locked(requests, count)) {
        return false;
    }
    g_shadow.queue.push_back({context, speculative, 0, speculative ? uint8_t{1} : uint8_t{0}});
    ++g_shadow.enqueued_batches;
    if (g_shadow.max_queue_depth < g_shadow.queue.size()) {
        g_shadow.max_queue_depth = g_shadow.queue.size();
    }
    g_shadow.queue_cv.notify_one();
    return true;
}

bool object_engine_enqueue_work(
        void *context,
        uint32_t kind,
        uint8_t priority,
        size_t capacity) {
    if (!object_engine_enforcing() || !context || kind == 0 || capacity == 0) return false;
    std::lock_guard<std::mutex> lock(g_shadow.mutex);
    if (g_shadow.queue.size() >= capacity || g_shadow.worker_stop) return false;
    g_shadow.queue.push_back({context, false, kind, priority});
    ++g_shadow.generic_enqueued;
    if (g_shadow.max_queue_depth < g_shadow.queue.size()) {
        g_shadow.max_queue_depth = g_shadow.queue.size();
    }
    g_shadow.queue_cv.notify_one();
    return true;
}

void *object_engine_take_batch() {
    if (!object_engine_enforcing()) return nullptr;
    std::lock_guard<std::mutex> lock(g_shadow.mutex);
    if (g_shadow.queue.empty()) return nullptr;
    auto selected = g_shadow.queue.begin();
    for (auto it = g_shadow.queue.begin(); it != g_shadow.queue.end(); ++it) {
        if (it->priority < selected->priority) {
            selected = it;
        }
    }
    const bool speculative = selected->speculative;
    void *context = selected->context;
    g_shadow.queue.erase(selected);
    ++g_shadow.dequeued_batches;
    if (speculative) {
        ++g_shadow.speculative_dequeues;
    } else {
        ++g_shadow.demand_dequeues;
    }
    return context;
}

bool object_engine_has_queued_batch() {
    if (!object_engine_enforcing()) return false;
    std::lock_guard<std::mutex> lock(g_shadow.mutex);
    return !g_shadow.queue.empty();
}

size_t object_engine_queued_batches() {
    if (!object_engine_enforcing()) return 0;
    std::lock_guard<std::mutex> lock(g_shadow.mutex);
    return g_shadow.queue.size();
}

bool object_engine_start_worker(const object_worker_provider &provider) {
    if (!object_engine_enforcing() || !provider.init || !provider.process) return false;
    {
        std::lock_guard<std::mutex> lock(g_shadow.mutex);
        if (g_shadow.worker_active) return true;
        if (g_shadow.worker.joinable()) return false;
        g_shadow.provider = provider;
        g_shadow.worker_started = false;
        g_shadow.worker_stop = false;
        g_shadow.worker_failed = false;
        g_shadow.worker = std::thread([] {
            const worker_affinity_status affinity = apply_worker_affinity(g_shadow.provider);
            const bool initialized = g_shadow.provider.init(g_shadow.provider.user_data);
            {
                std::lock_guard<std::mutex> lock(g_shadow.mutex);
                if (affinity != worker_affinity_status::not_requested) {
                    ++g_shadow.worker_affinity_attempts;
                    if (affinity == worker_affinity_status::success) {
                        ++g_shadow.worker_affinity_successes;
                    } else {
                        ++g_shadow.worker_affinity_failures;
                    }
                }
                g_shadow.worker_failed = !initialized;
                g_shadow.worker_active = initialized;
                g_shadow.worker_started = true;
            }
            g_shadow.queue_cv.notify_all();
            if (!initialized) return;
            for (;;) {
                queued_batch work;
                {
                    std::unique_lock<std::mutex> lock(g_shadow.mutex);
                    g_shadow.queue_cv.wait(lock, [] {
                        return g_shadow.worker_stop || !g_shadow.queue.empty();
                    });
                    if (g_shadow.worker_stop && g_shadow.queue.empty()) break;
                    auto selected = g_shadow.queue.begin();
                    for (auto it = g_shadow.queue.begin(); it != g_shadow.queue.end(); ++it) {
                        if (it->priority < selected->priority) selected = it;
                    }
                    work = *selected;
                    g_shadow.queue.erase(selected);
                    ++g_shadow.dequeued_batches;
                    if (work.kind == 0) {
                        if (work.speculative) {
                            ++g_shadow.speculative_dequeues;
                        } else {
                            ++g_shadow.demand_dequeues;
                        }
                    }
                }
                const bool success = g_shadow.provider.process(
                        g_shadow.provider.user_data, work.context, work.kind);
                std::lock_guard<std::mutex> lock(g_shadow.mutex);
                ++g_shadow.worker_processed;
                if (!success) ++g_shadow.worker_process_failed;
            }
            std::lock_guard<std::mutex> lock(g_shadow.mutex);
            g_shadow.worker_active = false;
        });
    }
    std::unique_lock<std::mutex> lock(g_shadow.mutex);
    g_shadow.queue_cv.wait(lock, [] { return g_shadow.worker_started; });
    const bool active = g_shadow.worker_active;
    lock.unlock();
    if (!active && g_shadow.worker.joinable()) g_shadow.worker.join();
    return active;
}

void object_engine_stop_worker() {
    if (!object_engine_enforcing()) return;
    {
        std::lock_guard<std::mutex> lock(g_shadow.mutex);
        g_shadow.worker_stop = true;
    }
    g_shadow.queue_cv.notify_all();
    if (g_shadow.worker.joinable()) g_shadow.worker.join();
}

bool object_engine_worker_active() {
    if (!object_engine_enforcing()) return false;
    std::lock_guard<std::mutex> lock(g_shadow.mutex);
    return g_shadow.worker_active;
}

} // namespace ggml_moe_common
