// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cpu-prefill.h"
#include "runtime.h"
#include "../../ggml-ssd/host/arbiter.h"
#include <cstddef>
#include <cstdint>

// A backend owns this immutable table and every callback it contains. Consumers
// borrow it only while that backend stays loaded, just as for its other procs.
namespace ssd { namespace cpu {
struct api_v2 {
    uint32_t version;   // 9: the pass's GPU working memory with the prompt graph (vram_reserve_pass)
    size_t size;
    decltype(&create_configured) create;
    decltype(&ssd_cpu_sink_run) run;
    decltype(&ssd_cpu_sink_run_activation) run_activation;
    decltype(&ssd_cpu_sink_destroy) destroy;
    decltype(&ssd_cpu_sink_call_open) call_open;
    decltype(&ssd_cpu_sink_call_close) call_close;
    decltype(&ssd_cpu_sink_call_join) call_join;
    decltype(&ssd_cpu_sink_decode_threads) decode_threads;
    decltype(&ssd_cpu_sink_cache_configure) cache_configure;
    decltype(&ssd_cpu_sink_cache_begin) cache_begin;
    decltype(&ssd_cpu_sink_request_begin) request_begin;
    decltype(&ssd_cpu_sink_vram_phase_complete) vram_phase_complete;
    decltype(&ssd_cpu_sink_cache_get_stats) cache_get_stats;
    decltype(&cache_make_room_configured) cache_make_room;
    decltype(&cache_prefill_room_configured) cache_prefill_room;
    decltype(&ssd_cpu_sink_prefetch_hint) prefetch_hint;
    decltype(&ssd_cpu_sink_prefill_hint) prefill_hint;
    decltype(&ssd_cpu_sink_vram_reserve) vram_reserve;
    decltype(&ssd_cpu_sink_preload) preload;
    decltype(&ssd_cpu_prefill_create) prefill_create;
    decltype(&ssd_cpu_prefill_run) prefill_run;
    decltype(&ssd_cpu_prefill_run_activation) prefill_run_activation;
    decltype(&ssd_cpu_prefill_destroy) prefill_destroy;
    ssd::result<ssd::host::allocation_ticket> (*take_room)(
        ssd_cpu_sink *, const ssd::settings::snapshot *, const ssd::runtime_services *,
        uint64_t, uint64_t, uint64_t) noexcept;
    bool (*attach_services)(ggml_backend_t, const ssd::runtime_services *) noexcept;
    size_t runtime_services_size;
    decltype(&ssd_cpu_sink_vram_reserve_pass) vram_reserve_pass;
};
using get_api_v2 = const api_v2 * (*)();

// Check the prefix before accessing members from a possibly different build.
inline bool valid(const api_v2 * api) noexcept {
    return api && api->version == 9 && api->size == sizeof(api_v2) &&
        api->runtime_services_size == sizeof(runtime_services) &&
        api->create &&
        api->run &&
        api->run_activation &&
        api->destroy &&
        api->call_open && api->call_close && api->call_join && api->decode_threads &&
        api->cache_configure &&
        api->cache_begin && api->request_begin &&
        api->vram_phase_complete &&
        api->cache_get_stats &&
        api->cache_make_room &&
        api->cache_prefill_room &&
        api->prefetch_hint &&
        api->prefill_hint &&
        api->vram_reserve &&
        api->preload &&
        api->prefill_create &&
        api->prefill_run &&
        api->prefill_run_activation &&
        api->prefill_destroy &&
        api->take_room && api->attach_services && api->vram_reserve_pass;
}
} } // namespace ssd::cpu
