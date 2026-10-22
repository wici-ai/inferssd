#pragma once

#include "llama.h"
#include "llama-kimi-compat.h"

#include <cstdint>

// Internal K3 launch configuration. The context owns this immutable snapshot;
// the pointer and all values remain stable until the context is destroyed.
struct llama_k3_runtime_config {
    llama_k3_decode_expert_policy primary;
    llama_k3_decode_expert_policy guard;
    llama_k3_decode_expert_table table;
    uint32_t guard_tokens = 0;
    uint32_t guard_start_tokens = 0;
    uint32_t guard_control_tokens = 0;
};

// No configuration is available for non-K3 or graphless tokenizer-only models
// without routing metadata. These are internal C++ entry points, not public C API.
LLAMA_API const llama_k3_runtime_config * llama_k3_runtime_config_get(const llama_context * ctx);

// Only the context owner may switch between forward calls, with no concurrent
// context use. True selects the frozen guard plus an empty layer table; false
// restores the frozen primary plus its table. True is rejected if all guard
// budgets are zero. Synchronizes prior work and invalidates graph reuse when changed.
// A within-response schedule intentionally retains its prior KV; the server
// must clear mixed-policy KV/recurrent/prompt/checkpoint state before reuse.
LLAMA_API bool llama_k3_runtime_guard_set(llama_context * ctx, bool active);
