// SPDX-License-Identifier: MIT
#pragma once

#include "kimi-cpu-prefill.h"
#include <cstddef>
#include <cstdint>

// A backend owns this immutable table and every callback it contains. Consumers
// borrow it only while that backend stays loaded, just as for its other procs.
namespace ssd { namespace cpu {
struct api_v1 {
    uint32_t version;
    size_t size;
    decltype(&kimi_cpu_sink_source_size) source_size;
    decltype(&kimi_cpu_sink_create) create;
    decltype(&kimi_cpu_sink_run) run;
    decltype(&kimi_cpu_sink_run_activation) run_activation;
    decltype(&kimi_cpu_sink_destroy) destroy;
    decltype(&kimi_cpu_sink_cache_configure) cache_configure;
    decltype(&kimi_cpu_sink_cache_begin) cache_begin;
    decltype(&kimi_cpu_sink_vram_phase_complete) vram_phase_complete;
    decltype(&kimi_cpu_sink_cache_get_stats) cache_get_stats;
    decltype(&kimi_cpu_sink_cache_make_room) cache_make_room;
    decltype(&kimi_cpu_sink_cache_prefill_room) cache_prefill_room;
    decltype(&kimi_cpu_sink_prefetch_hint) prefetch_hint;
    decltype(&kimi_cpu_sink_vram_reserve) vram_reserve;
    decltype(&kimi_cpu_sink_run_cache_compute) run_cache_compute;
    decltype(&kimi_cache_compute_config_from_env) cache_compute_config;
    decltype(&kimi_cpu_prefill_create) prefill_create;
    decltype(&kimi_cpu_prefill_run) prefill_run;
    decltype(&kimi_cpu_prefill_run_activation) prefill_run_activation;
    decltype(&kimi_cpu_prefill_destroy) prefill_destroy;
};
using get_api_v1 = const api_v1 * (*)();

// Check the prefix before accessing members from a possibly different build.
inline bool valid(const api_v1 * api) noexcept {
    return api && api->version == 1 && api->size == sizeof(api_v1) &&
        api->source_size &&
        api->create &&
        api->run &&
        api->run_activation &&
        api->destroy &&
        api->cache_configure &&
        api->cache_begin &&
        api->vram_phase_complete &&
        api->cache_get_stats &&
        api->cache_make_room &&
        api->cache_prefill_room &&
        api->prefetch_hint &&
        api->vram_reserve &&
        api->run_cache_compute &&
        api->cache_compute_config &&
        api->prefill_create &&
        api->prefill_run &&
        api->prefill_run_activation &&
        api->prefill_destroy;
}
} } // namespace ssd::cpu
