// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "llama-arch.h"
#include <cstdio>

// Snapshot before each prefill and at decode steps 0, 16, 32, ... of each request.
struct llama_selection_bias_schedule {
    unsigned step = 0;
    bool refresh(bool decode) {
        if (!decode) { step = 0; return true; }
        const bool due = step == 0;
        step = (step + 1) % 16;
        return due;
    }
};

// Context-owned experiment. No process settings or command line overrides.
struct llama_selection_bias_options {
    bool enabled = false;
    static llama_selection_bias_options activate(llm_arch arch, bool cuda_sink, FILE * receipt = stderr) {
        const bool enabled = arch == LLM_ARCH_DEEPSEEK41 && cuda_sink;
        const char * model = arch == LLM_ARCH_DEEPSEEK41 ? "\"deepseek\"" :
            arch == LLM_ARCH_MUSE_SPARK ? "\"muse\"" : "null";
        std::fprintf(receipt, "rsi.design.active {\"consumer\":\"deepseek-cache-residency-selection-bias\","
            "\"model\":%s,\"enabled\":%s,\"once_per_context\":true}\n", model, enabled ? "true" : "false");
        return { enabled };
    }
    static float bonus(bool ram, bool gpu) {
        // One payload resident in both tiers earns only the larger bonus.
        return gpu ? 0.3f : ram ? 0.2f : 0.0f;
    }
};
