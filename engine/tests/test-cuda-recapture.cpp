// SPDX-License-Identifier: Apache-2.0
#include "llama-cuda-recapture.h"
#include "graph-recapture.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstring>
#include <cstdio>

int main() {
    // Every known architecture plus an invalid value: no implicit Muse fallback.
    for (int i = 0; i <= LLM_ARCH_UNKNOWN; ++i) {
        auto option = llama_cuda_recapture_default(static_cast<llm_arch>(i));
        assert(option.enabled == (i == LLM_ARCH_DEEPSEEK41));
        assert(std::strcmp(option.model, i == LLM_ARCH_DEEPSEEK41 ? "deepseek" :
            i == LLM_ARCH_MUSE_SPARK ? "muse" : "unknown") == 0);
    }
    assert(!llama_cuda_recapture_default(static_cast<llm_arch>(-1)).enabled);
    for (unsigned mask = 0; mask < 32; ++mask) {
        bool enabled = mask & 1, vmm = mask & 2, instance = mask & 4;
        bool replayed = mask & 8, changed = mask & 16;
        assert(ggml_cuda_can_recapture(enabled, vmm, instance, replayed, changed) == (mask == 15));
    }
    for (int op = 0; op < GGML_OP_COUNT; ++op) {
        auto operation = static_cast<ggml_op>(op);
        assert(!ggml_cuda_recapture_node_changed(operation, false, operation, false));
        bool noop = operation == GGML_OP_NONE || operation == GGML_OP_VIEW ||
            operation == GGML_OP_RESHAPE || operation == GGML_OP_TRANSPOSE || operation == GGML_OP_PERMUTE;
        assert(ggml_cuda_recapture_node_changed(operation, false, operation, true) == !noop);
        assert(ggml_cuda_recapture_node_changed(operation, true, operation, false) == !noop);
        for (int other = 0; other < GGML_OP_COUNT; ++other) {
            assert(ggml_cuda_recapture_node_changed(operation, false,
                static_cast<ggml_op>(other), false) == (op != other));
        }
    }
    std::puts("scope, unknown architecture, 32 guard combinations and all operation/activity transitions passed");
}
