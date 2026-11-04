// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "llama-arch.h"
#include <cstdio>

// Context-local experimental policy. No process-global backend state: graphs from
// different model contexts may share a CUDA device without sharing this option.
inline bool llama_configure_thin_mmvf(llm_arch arch) {
    const bool enabled = arch == LLM_ARCH_MUSE_SPARK;
    const char * model = arch == LLM_ARCH_MUSE_SPARK ? "muse" :
                        arch == LLM_ARCH_DEEPSEEK41 ? "deepseek" : "unknown";
    std::fprintf(stderr, "rsi.design.active {\"consumer\":\"cuda-thin-float-mmvf\","
        "\"model\":\"%s\",\"enabled\":%s,\"once_per_context\":true}\n",
        model, enabled ? "true" : "false");
    return enabled;
}

inline void llama_mark_thin_mmvf(ggml_tensor * node, bool enabled) {
    // MUL_MAT_ID and the streaming MMVF bridge are outside this experiment.
    if (node->op == GGML_OP_MUL_MAT) {
        node->flags &= ~GGML_TENSOR_FLAG_THIN_MMVF;
        if (enabled) {
            node->flags |= GGML_TENSOR_FLAG_THIN_MMVF;
        }
    }
}
