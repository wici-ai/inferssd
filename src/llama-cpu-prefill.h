#pragma once

#include "kimi-cpu-prefill.h"

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
        auto create = reinterpret_cast<decltype(&kimi_cpu_prefill_create)>(
            ggml_backend_reg_get_proc_address(registry, "kimi_cpu_prefill_create"));
        run = reinterpret_cast<decltype(run)>(
            ggml_backend_reg_get_proc_address(registry, "kimi_cpu_prefill_run"));
        run_activation = reinterpret_cast<decltype(run_activation)>(
            ggml_backend_reg_get_proc_address(registry, "kimi_cpu_prefill_run_activation"));
        destroy = reinterpret_cast<decltype(destroy)>(
            ggml_backend_reg_get_proc_address(registry, "kimi_cpu_prefill_destroy"));
        if (!create || !run || !run_activation || !destroy) {
            GGML_ABORT("CPU prefill executor is unavailable; no fallback");
        }
        char error[512] = {};
        context = create(sink, error, sizeof(error));
        if (!context) {
            GGML_ABORT("CPU prefill initialization failed: %s; no fallback", error);
        }
    }
};
