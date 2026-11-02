// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ggml-ssd/settings/settings.h"
#include "llama-arch.h"

namespace ssd { namespace settings {
inline bool resident_rotation_requested(const snapshot * settings, llm_arch arch) {
    // Architecture is a hard consumer gate, including with an explicit option=1.
    if (arch != LLM_ARCH_DEEPSEEK41) { return false; }
    const auto * option = settings ? settings->integer("model.kv_rotation_resident") : nullptr;
    return !option || *option == 1;
}
inline std::string resident_rotation_receipt(llm_arch arch, bool enabled) {
    const char * model = nullptr;
    switch (arch) {
        case LLM_ARCH_DEEPSEEK41: model = "deepseek"; break;
        case LLM_ARCH_MUSE_SPARK: model = "muse"; enabled = false; break;
        default: return {}; // Unknown architectures are never labeled as Muse.
    }
    return std::string("{\"consumer\":\"backend-resident-kv-hadamard-rotation-constants\",\"model\":\"") +
        model + "\",\"enabled\":" + (enabled ? "true" : "false") +
        ",\"once_per_context\":true}";
}
}} // namespace ssd::settings
