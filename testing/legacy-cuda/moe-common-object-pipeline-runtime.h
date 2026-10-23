#pragma once

#include "moe-common-object-pipeline.h"

namespace ggml_moe_common {

bool object_pipeline_parse_mode(
    const char *value,
    object_pipeline_mode &mode);

const char *object_pipeline_mode_name(object_pipeline_mode mode);

const char *object_pipeline_config_source_name(
    object_pipeline_config_source source);

object_pipeline_mode_config object_pipeline_resolve_mode(
    object_pipeline_mode model_default,
    bool legacy_rolling_requested);

object_pipeline_policy_config object_pipeline_resolve_policy(
    const object_pipeline_policy &fallback);

enum object_pipeline_metric_timing : uint32_t {
    object_pipeline_timing_queue_formation = 1U << 0,
    object_pipeline_timing_lifetime = 1U << 1,
    object_pipeline_timing_worker_idle = 1U << 2,
    object_pipeline_timing_foreground_wait = 1U << 3,
    object_pipeline_timing_ssd_read = 1U << 4,
    object_pipeline_timing_h2d = 1U << 5,
};

struct object_pipeline_metrics {
    uint32_t timing_mask = 0;
    uint64_t queue_formation_us = 0;
    uint64_t pipeline_lifetime_us = 0;
    uint64_t worker_idle_us = 0;
    uint64_t submitted = 0;
    uint64_t completed = 0;
    uint64_t failed = 0;
    uint64_t cancelled = 0;
    uint64_t initial_submit_rounds = 0;
    uint64_t refill_submit_rounds = 0;
    uint64_t timeout_flush_rounds = 0;
    uint64_t timeout_wait_us = 0;
    uint64_t inflight_sum = 0;
    uint64_t inflight_samples = 0;
    uint64_t inflight_max = 0;
    uint64_t foreground_wait_us = 0;
    uint64_t ssd_reads = 0;
    uint64_t ssd_bytes = 0;
    uint64_t ssd_read_us = 0;
    uint64_t h2d_bytes = 0;
    uint64_t h2d_us = 0;
    uint64_t queue_full_fallbacks = 0;
    uint64_t timeout_fallbacks = 0;
    uint64_t generation_fallbacks = 0;
    uint64_t io_fallbacks = 0;
    uint64_t h2d_fallbacks = 0;
    uint64_t duplicate_inflight = 0;
    uint64_t stale_completions = 0;
};

bool object_pipeline_metrics_valid(const object_pipeline_metrics &metrics);

void object_pipeline_metrics_report(
    const char *model,
    const object_pipeline_metrics &metrics);

struct object_pipeline_dispatch_runtime_snapshot {
    uint64_t timeout_flush_rounds = 0;
    uint64_t timeout_wait_us = 0;
};

object_pipeline_dispatch_runtime_snapshot
object_pipeline_dispatch_runtime_snapshot_load();

bool object_pipeline_refill_reset(
    object_pipeline_refill_cursor &cursor,
    size_t total);

object_pipeline_dispatch_plan object_pipeline_refill_plan(
    const object_pipeline_refill_cursor &cursor,
    const object_pipeline_policy &policy,
    size_t inflight_h2d,
    uint64_t oldest_wait_us,
    bool force);

object_pipeline_dispatch_plan object_pipeline_refill_plan_ready(
    const object_pipeline_refill_cursor &cursor,
    const object_pipeline_policy &policy,
    size_t inflight_h2d,
    uint64_t oldest_wait_us,
    bool wait_for_timeout);

bool object_pipeline_refill_record_submit(
    object_pipeline_refill_cursor &cursor,
    size_t count);

bool object_pipeline_refill_record_io_completion(
    object_pipeline_refill_cursor &cursor,
    size_t count = 1);

bool object_pipeline_refill_record_io_failure(
    object_pipeline_refill_cursor &cursor,
    size_t count = 1);

bool object_pipeline_refill_cancel_queued(
    object_pipeline_refill_cursor &cursor,
    size_t count = 1);

bool object_pipeline_refill_record_critical_ready(
    object_pipeline_refill_cursor &cursor,
    size_t count = 1);

bool object_pipeline_refill_done(
    const object_pipeline_refill_cursor &cursor);

} // namespace ggml_moe_common
