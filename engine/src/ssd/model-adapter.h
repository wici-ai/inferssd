// SPDX-License-Identifier: MIT
#pragma once

#include "llama-arch.h"

struct llama_hparams;

namespace llama_ssd {

// Model facts remain in L3. Neither this description nor its validator knows
// an executor's private cache, kernels, worker threads, or I/O implementation.
enum class expert_quantization { unsupported, all_mxfp4, matching_up_gate };
enum class expert_activation { unsupported, swiglu_clamp };

struct expert_requirements {
    expert_quantization quantization = expert_quantization::unsupported;
    expert_activation activation = expert_activation::unsupported;
    float clamp = 0.0f;
    bool finite_clamp = false;
    bool require_post_norm = false;
    bool single_token_or_prefill = false;
};

[[nodiscard]] expert_requirements describe_experts(
        llm_arch architecture, const llama_hparams & hparams, int layer) noexcept;

struct expert_inputs {
    bool backend_available;
    bool runtime_available;
    const ggml_tensor * up;
    const ggml_tensor * gate;
    const ggml_tensor * down;
    const ggml_tensor * gate_up;
    const ggml_tensor * gate_up_bias;
    const ggml_tensor * gate_input_bias;
    const ggml_tensor * up_bias;
    const ggml_tensor * gate_bias;
    const ggml_tensor * down_bias;
    const ggml_tensor * up_scale;
    const ggml_tensor * gate_scale;
    const ggml_tensor * down_scale;
    int64_t n_embd;
    int64_t n_tokens;
    bool silu;
    bool cache_compute;
    bool prefill;
    bool weight_before_ffn;
    bool has_lora;
    const ggml_tensor * post_norm;
    float post_norm_eps;
    float routed_scale;
};

// The shared graph bridge translates this validated, model-neutral description
// to the existing executor node. Kernel/type support remains checked by the
// selected backend, at its existing initialization boundary.
struct expert_execution {
    expert_activation activation = expert_activation::unsupported;
    float clamp = 0.0f;
    bool post_norm = false;
    float post_norm_eps = 0.0f;
    float routed_scale = 1.0f;
};

struct expert_validation {
    bool accepted = false;
    expert_execution execution;
    const char * error = nullptr;
};

[[nodiscard]] expert_validation validate_experts(
        const expert_requirements & requirements, const expert_inputs & inputs) noexcept;

} // namespace llama_ssd
