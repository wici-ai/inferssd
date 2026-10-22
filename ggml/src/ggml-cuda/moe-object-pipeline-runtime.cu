#include "moe-object-pipeline-runtime.h"

#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>

namespace ggml_moe {

namespace {

std::atomic<uint64_t> g_next_pipeline_ticket{1};
std::atomic<uint64_t> g_timeout_flush_rounds{0};
std::atomic<uint64_t> g_timeout_wait_us{0};

enum class env_value_status : uint8_t {
    absent,
    valid,
    invalid,
};

static env_value_status parse_positive_env(const char *name, uint64_t &value) {
    const char *text = std::getenv(name);
    if (!text || !text[0]) return env_value_status::absent;

    uint64_t parsed = 0;
    for (const char *cursor = text; *cursor; ++cursor) {
        if (*cursor < '0' || *cursor > '9') return env_value_status::invalid;
        const uint64_t digit = static_cast<uint64_t>(*cursor - '0');
        if (parsed > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
            return env_value_status::invalid;
        }
        parsed = parsed * 10 + digit;
    }
    if (parsed == 0) return env_value_status::invalid;
    value = parsed;
    return env_value_status::valid;
}

template<typename T>
void apply_policy_env(
        const char *name,
        uint32_t mask,
        T &target,
        object_pipeline_policy_config &config) {
    uint64_t value = 0;
    const env_value_status status = parse_positive_env(name, value);
    if (status == env_value_status::absent) return;
    config.override_mask |= mask;
    if (status == env_value_status::invalid ||
            value > static_cast<uint64_t>(std::numeric_limits<T>::max())) {
        config.invalid_mask |= mask;
        return;
    }
    target = static_cast<T>(value);
}

} // namespace

bool object_pipeline_parse_mode(
        const char *value,
        object_pipeline_mode &mode) {
    if (!value || !value[0]) return false;
    if (std::strcmp(value, "sync") == 0 || std::strcmp(value, "sync_qd1") == 0) {
        mode = object_pipeline_mode::sync_qd1;
        return true;
    }
    if (std::strcmp(value, "persistent") == 0 ||
            std::strcmp(value, "persistent_demand") == 0) {
        mode = object_pipeline_mode::persistent_demand;
        return true;
    }
    if (std::strcmp(value, "rolling") == 0 ||
            std::strcmp(value, "rolling_refill") == 0) {
        mode = object_pipeline_mode::rolling_refill;
        return true;
    }
    return false;
}

const char *object_pipeline_mode_name(object_pipeline_mode mode) {
    switch (mode) {
        case object_pipeline_mode::sync_qd1: return "sync_qd1";
        case object_pipeline_mode::persistent_demand: return "persistent_demand";
        case object_pipeline_mode::rolling_refill: return "rolling_refill";
    }
    return "invalid";
}

const char *object_pipeline_config_source_name(
        object_pipeline_config_source source) {
    switch (source) {
        case object_pipeline_config_source::model_default: return "model_default";
        case object_pipeline_config_source::legacy: return "legacy";
        case object_pipeline_config_source::common: return "common";
    }
    return "invalid";
}

object_pipeline_mode_config object_pipeline_resolve_mode(
        object_pipeline_mode model_default,
        bool legacy_rolling_requested) {
    object_pipeline_mode_config config;
    config.mode = model_default;
    const char *value = std::getenv("GGML_MOE_PIPELINE_MODE");
    if (value && value[0]) {
        config.source = object_pipeline_config_source::common;
        config.valid = object_pipeline_parse_mode(value, config.mode);
        if (!config.valid) config.mode = model_default;
        return config;
    }
    if (legacy_rolling_requested) {
        config.mode = object_pipeline_mode::rolling_refill;
        config.source = object_pipeline_config_source::legacy;
    }
    return config;
}

object_pipeline_policy_config object_pipeline_resolve_policy(
        const object_pipeline_policy &fallback) {
    object_pipeline_policy_config config;
    config.policy = fallback;
    if (!object_pipeline_policy_valid(fallback)) {
        config.valid = false;
        return config;
    }

    apply_policy_env("GGML_MOE_PIPELINE_IO_DEPTH",
        object_pipeline_override_io_depth, config.policy.io_depth, config);
    apply_policy_env("GGML_MOE_PIPELINE_INITIAL_BATCH",
        object_pipeline_override_initial_batch, config.policy.initial_batch, config);
    apply_policy_env("GGML_MOE_PIPELINE_REFILL_BATCH",
        object_pipeline_override_refill_batch, config.policy.refill_batch, config);
    apply_policy_env("GGML_MOE_PIPELINE_MIN_FLUSH_BATCH",
        object_pipeline_override_min_flush_batch, config.policy.min_flush_batch, config);
    apply_policy_env("GGML_MOE_PIPELINE_FLUSH_TIMEOUT_US",
        object_pipeline_override_flush_timeout_us, config.policy.flush_timeout_us, config);

    if ((config.override_mask & object_pipeline_override_io_depth) != 0 &&
            (config.invalid_mask & object_pipeline_override_io_depth) == 0) {
        if ((config.override_mask & object_pipeline_override_initial_batch) == 0) {
            config.policy.initial_batch = std::min(
                config.policy.initial_batch, config.policy.io_depth);
        }
        if ((config.override_mask & object_pipeline_override_refill_batch) == 0) {
            config.policy.refill_batch = std::min(
                config.policy.refill_batch, config.policy.io_depth);
        }
        if ((config.override_mask & object_pipeline_override_min_flush_batch) == 0) {
            config.policy.min_flush_batch = std::min(
                config.policy.min_flush_batch, config.policy.io_depth);
        }
    }

    if (config.invalid_mask != 0 || !object_pipeline_policy_valid(config.policy)) {
        config.policy = fallback;
        config.valid = false;
    }
    return config;
}

bool object_pipeline_metrics_valid(const object_pipeline_metrics &metrics) {
    if (metrics.completed > metrics.submitted ||
            metrics.failed > metrics.submitted - metrics.completed ||
            metrics.cancelled > metrics.submitted - metrics.completed - metrics.failed ||
            (metrics.inflight_samples == 0 && metrics.inflight_sum != 0) ||
            (metrics.inflight_samples == 0 && metrics.inflight_max != 0) ||
            ((metrics.ssd_reads == 0) != (metrics.ssd_bytes == 0))) {
        return false;
    }
    return true;
}

void object_pipeline_metrics_report(
        const char *model,
        const object_pipeline_metrics &metrics) {
    const bool valid = object_pipeline_metrics_valid(metrics);
    const double average_inflight = metrics.inflight_samples != 0 ?
        static_cast<double>(metrics.inflight_sum) /
            static_cast<double>(metrics.inflight_samples) : 0.0;
    const double idle_ratio =
        (metrics.timing_mask & object_pipeline_timing_worker_idle) != 0 &&
            metrics.pipeline_lifetime_us != 0 ?
        100.0 * static_cast<double>(metrics.worker_idle_us) /
            static_cast<double>(metrics.pipeline_lifetime_us) : 0.0;
    const double ssd_gib_s =
        (metrics.timing_mask & object_pipeline_timing_ssd_read) != 0 &&
            metrics.ssd_read_us != 0 ?
        static_cast<double>(metrics.ssd_bytes) * 1000000.0 /
            (static_cast<double>(metrics.ssd_read_us) * 1024.0 * 1024.0 * 1024.0) : 0.0;
    std::fprintf(stderr,
        "[moe_pipeline] common metrics: model=%s valid=%d timing=0x%x "
        "queue_formation_us=%lu pipeline_lifetime_us=%lu worker_idle_us=%lu "
        "idle_ratio=%.3f%% submitted=%lu completed=%lu failed=%lu cancelled=%lu "
        "initial_rounds=%lu refill_rounds=%lu timeout_flush_rounds=%lu "
        "timeout_wait_us=%lu inflight_avg=%.3f inflight_max=%lu "
        "foreground_wait_us=%lu ssd_reads=%lu ssd_bytes=%lu ssd_read_us=%lu "
        "ssd_gib_s=%.3f h2d_bytes=%lu h2d_us=%lu fallbacks=%lu,%lu,%lu,%lu,%lu "
        "duplicate_inflight=%lu stale_completions=%lu\n",
        model && model[0] ? model : "unknown",
        valid ? 1 : 0,
        metrics.timing_mask,
        (unsigned long) metrics.queue_formation_us,
        (unsigned long) metrics.pipeline_lifetime_us,
        (unsigned long) metrics.worker_idle_us,
        idle_ratio,
        (unsigned long) metrics.submitted,
        (unsigned long) metrics.completed,
        (unsigned long) metrics.failed,
        (unsigned long) metrics.cancelled,
        (unsigned long) metrics.initial_submit_rounds,
        (unsigned long) metrics.refill_submit_rounds,
        (unsigned long) metrics.timeout_flush_rounds,
        (unsigned long) metrics.timeout_wait_us,
        average_inflight,
        (unsigned long) metrics.inflight_max,
        (unsigned long) metrics.foreground_wait_us,
        (unsigned long) metrics.ssd_reads,
        (unsigned long) metrics.ssd_bytes,
        (unsigned long) metrics.ssd_read_us,
        ssd_gib_s,
        (unsigned long) metrics.h2d_bytes,
        (unsigned long) metrics.h2d_us,
        (unsigned long) metrics.queue_full_fallbacks,
        (unsigned long) metrics.timeout_fallbacks,
        (unsigned long) metrics.generation_fallbacks,
        (unsigned long) metrics.io_fallbacks,
        (unsigned long) metrics.h2d_fallbacks,
        (unsigned long) metrics.duplicate_inflight,
        (unsigned long) metrics.stale_completions);
}

object_pipeline_dispatch_runtime_snapshot
object_pipeline_dispatch_runtime_snapshot_load() {
    object_pipeline_dispatch_runtime_snapshot snapshot;
    snapshot.timeout_flush_rounds =
        g_timeout_flush_rounds.load(std::memory_order_relaxed);
    snapshot.timeout_wait_us = g_timeout_wait_us.load(std::memory_order_relaxed);
    return snapshot;
}

bool object_pipeline_refill_reset(
        object_pipeline_refill_cursor &cursor,
        size_t total) {
    uint64_t ticket_id = g_next_pipeline_ticket.fetch_add(1, std::memory_order_relaxed);
    if (ticket_id == 0) {
        ticket_id = g_next_pipeline_ticket.fetch_add(1, std::memory_order_relaxed);
    }
    return cursor.reset(total, ticket_id);
}

object_pipeline_dispatch_plan object_pipeline_refill_plan(
        const object_pipeline_refill_cursor &cursor,
        const object_pipeline_policy &policy,
        size_t inflight_h2d,
        uint64_t oldest_wait_us,
        bool force) {
    return cursor.plan(policy, inflight_h2d, oldest_wait_us, force);
}

object_pipeline_dispatch_plan object_pipeline_refill_plan_ready(
        const object_pipeline_refill_cursor &cursor,
        const object_pipeline_policy &policy,
        size_t inflight_h2d,
        uint64_t oldest_wait_us,
        bool wait_for_timeout) {
    object_pipeline_dispatch_plan plan =
        cursor.plan(policy, inflight_h2d, oldest_wait_us, !wait_for_timeout);
    if (!wait_for_timeout ||
            plan.reason != object_pipeline_dispatch_reason::wait_timeout) {
        return plan;
    }

    const uint64_t wait_us = plan.wait_remaining_us;
    std::this_thread::sleep_for(std::chrono::microseconds(wait_us));
    plan = cursor.plan(policy, inflight_h2d, oldest_wait_us + wait_us, false);
    if (plan.reason == object_pipeline_dispatch_reason::timeout) {
        g_timeout_flush_rounds.fetch_add(1, std::memory_order_relaxed);
        g_timeout_wait_us.fetch_add(wait_us, std::memory_order_relaxed);
    }
    return plan;
}

bool object_pipeline_refill_record_submit(
        object_pipeline_refill_cursor &cursor,
        size_t count) {
    return cursor.record_submit(count);
}

bool object_pipeline_refill_record_io_completion(
        object_pipeline_refill_cursor &cursor,
        size_t count) {
    return cursor.record_io_completion(count);
}

bool object_pipeline_refill_record_io_failure(
        object_pipeline_refill_cursor &cursor,
        size_t count) {
    return cursor.record_io_failure(count);
}

bool object_pipeline_refill_cancel_queued(
        object_pipeline_refill_cursor &cursor,
        size_t count) {
    return cursor.cancel_queued(count);
}

bool object_pipeline_refill_record_critical_ready(
        object_pipeline_refill_cursor &cursor,
        size_t count) {
    return cursor.record_critical_ready(count);
}

bool object_pipeline_refill_done(
        const object_pipeline_refill_cursor &cursor) {
    return cursor.done();
}

} // namespace ggml_moe
