#pragma once
#include "ggml-cpu.h"
#ifdef __cplusplus
extern "C" {
#endif

enum kimi_cache_compute_mode {
    KIMI_CACHE_COMPUTE_OFF = 0,
    KIMI_CACHE_COMPUTE_BONUS = 1,
    KIMI_CACHE_COMPUTE_HYBRID = 2,
    KIMI_CACHE_COMPUTE_SHRINK = 3,
};

// Version 1, fixed-size C layout. No pointers or wall-clock reads in the planner.
struct kimi_cache_compute_config {
    uint32_t version;
    int32_t mode;
    int32_t bonus_max;
    int32_t min_s;
    int32_t max_s;
    float s_fraction;
    uint64_t deadline_us;
    float protect_mass;
    int32_t protect_min;
    int32_t apply_guard;
    int32_t trace;
};

struct kimi_cache_compute_plan {
    struct kimi_cache_compute_config config;
    int32_t native_count;
    int32_t original_count;
    int32_t guarded;
    int32_t deadline_applied;
    int32_t native_ids[16];
    float native_weights[16];
    // Indexed by native slot; original_order retains the original S slot order.
    float original_weights[16];
    int32_t original_order[16];
    uint32_t original_mask;
    uint32_t cached_mask;
    uint32_t base_mask;
    uint32_t protected_mask;
    uint32_t pin_mask;
    uint32_t bonus_mask;
    uint32_t replacement_mask;
    uint32_t dropped_mask;
    uint32_t final_mask;
};

GGML_BACKEND_API bool kimi_cache_compute_config_from_env(
        struct kimi_cache_compute_config * config, char * error, size_t error_size);
GGML_BACKEND_API bool kimi_cache_compute_plan_init(
        const struct kimi_cache_compute_config * config,
        const int32_t * native_ids, const float * native_weights, int native_count,
        const int32_t * original_ids, const float * original_weights, int original_slots,
        uint32_t cached_mask, bool guarded, struct kimi_cache_compute_plan * plan,
        char * error, size_t error_size);
// ready_by_deadline contains only completions already observed by the caller
// with observed-ready timestamp <= route_visible + deadline. Never pass future
// CQEs. Once deadline_applied, later completions cannot change the final set.
GGML_BACKEND_API void kimi_cache_compute_plan_update(
        struct kimi_cache_compute_plan * plan, uint32_t ready_by_deadline, uint64_t elapsed_us);
#ifdef __cplusplus
}
#endif
