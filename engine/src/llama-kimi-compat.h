// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "llama-arch.h"
#include "llama-hparams.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

static inline bool llama_string_contains_case_insensitive(const std::string & haystack, const char * needle) {
    if (needle == nullptr || needle[0] == '\0') {
        return true;
    }

    const char * needle_end = needle + std::strlen(needle);
    const auto it = std::search(
            haystack.begin(), haystack.end(),
            needle, needle_end,
            [](char a, char b) {
                return std::tolower((unsigned char) a) == std::tolower((unsigned char) b);
            });

    return it != haystack.end();
}

static inline bool llama_model_name_is_kimi(const std::string & name) {
    return llama_string_contains_case_insensitive(name, "kimi");
}

static inline bool llama_is_kimi_k2_deepseek2_layout(llm_arch arch, const llama_hparams & hparams) {
    return arch == LLM_ARCH_DEEPSEEK2 &&
        hparams.n_expert == 384 &&
        hparams.n_expert_groups == 1;
}

// With --defer-experts (mmap), keep routed-expert tensors in mmap-backed CPU
// buffers even when layers are offloaded, so they are never uploaded to VRAM.
// DeepSeek-V4.1 streams them from an expert pack through the CPU sink.
static inline bool llama_should_defer_kimi_experts_on_gpu(llm_arch arch, const std::string & name) {
    return arch == LLM_ARCH_KIMI_LINEAR ||
        arch == LLM_ARCH_DEEPSEEK41 ||
        arch == LLM_ARCH_MUSE_SPARK ||
        (arch == LLM_ARCH_DEEPSEEK2 && llama_model_name_is_kimi(name));
}

static inline bool llama_moe_should_use_plain_top_k(llm_arch arch, const llama_hparams & hparams) {
    return arch == LLM_ARCH_MISTRAL4 ||
        arch == LLM_ARCH_KIMI_LINEAR ||
        llama_is_kimi_k2_deepseek2_layout(arch, hparams);
}

static inline float llama_deepseek2_yarn_log_mul_for_graph(llm_arch arch, const llama_hparams & hparams) {
    return llama_is_kimi_k2_deepseek2_layout(arch, hparams) ?
        hparams.rope_yarn_log_mul : 0.1f * hparams.rope_yarn_log_mul;
}
