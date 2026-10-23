#pragma once

#include "ssd/api.h"

// Companion to the decode runtime. It borrows the same sink, so prefill adds
// neither a second read ring nor another persistent thread pool.
struct llama_kimi_cpu_prefill_runtime {
    kimi_cpu_prefill * context = nullptr;
    decltype(&kimi_cpu_prefill_run) run = nullptr;
    decltype(&kimi_cpu_prefill_run_activation) run_activation = nullptr;
    decltype(&kimi_cpu_prefill_destroy) destroy = nullptr;

    void reset() {
        if (context) {
            destroy(context);
            context = nullptr;
        }
    }

    void initialize(ggml_backend_reg_t registry, kimi_cpu_sink * sink) {
        if (context) {
            return;
        }
        const auto get = reinterpret_cast<ssd::cpu::get_api_v1>(
            ggml_backend_reg_get_proc_address(registry, "ggml_backend_cpu_ssd_api_v1"));
        const auto * api = get ? get() : nullptr;
        if (!ssd::cpu::valid(api)) {
            GGML_ABORT("CPU prefill executor ABI mismatch; rebuild the CPU backend and model together");
        }
        const auto create = api->prefill_create;
        run = api->prefill_run;
        run_activation = api->prefill_run_activation;
        destroy = api->prefill_destroy;
        char error[512] = {};
        context = create(sink, error, sizeof(error));
        if (!context) {
            GGML_ABORT("CPU prefill initialization failed: %s; no fallback", error);
        }
    }
};
