// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "llama-arch.h"

struct llama_cuda_recapture_option {
    const char * model;
    bool enabled;
};

// Narrowed experimental scope; qualification requires final-source device measurements.
inline llama_cuda_recapture_option llama_cuda_recapture_default(llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_DEEPSEEK41: return {"deepseek", true};
        case LLM_ARCH_MUSE_SPARK: return {"muse", false};
        default: return {"unknown", false};
    }
}
