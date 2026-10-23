#pragma once

#include <cstddef>
#include <cstdint>

namespace ggml_moe {

// Observe the first valid routed-expert shape without initializing transport.
// The common implementation is the sole supported adapter.
void adapter_select_model_shape(
    int64_t n_experts,
    size_t expert_bytes,
    const char * source);

bool adapter_uses_common();
bool adapter_is_selected();

} // namespace ggml_moe

extern "C" void ggml_cuda_moe_adapter_select_model_shape(
    int64_t n_experts,
    size_t expert_bytes);
