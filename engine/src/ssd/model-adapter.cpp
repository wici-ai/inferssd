// SPDX-License-Identifier: MIT
#include "model-adapter.h"

#include "llama-hparams.h"

#include <cmath>

namespace llama_ssd {

expert_requirements describe_experts(
        llm_arch architecture, const llama_hparams & hparams, int layer) noexcept {
    switch (architecture) {
        case LLM_ARCH_DEEPSEEK41:
            return {expert_quantization::all_mxfp4, expert_activation::swiglu_clamp,
                layer >= 0 ? hparams.swiglu_clamp_exp[layer] : 0.0f, true, false, true};
        case LLM_ARCH_MUSE_SPARK:
            return {expert_quantization::matching_up_gate, expert_activation::swiglu_clamp,
                INFINITY, false, true, false};
        default:
            return {};
    }
}

expert_validation validate_experts(const expert_requirements & requirements, const expert_inputs & in) noexcept {
    const bool layout = in.backend_available && in.runtime_available && in.up && in.gate && in.down &&
        ggml_is_contiguous(in.up) &&
        ggml_is_contiguous(in.gate) && ggml_is_contiguous(in.down) &&
        ggml_are_same_shape(in.up, in.gate) && in.up->ne[3] == 1 &&
        in.down->ne[3] == 1 && in.down->ne[0] == in.up->ne[1] &&
        in.down->ne[1] == in.n_embd && in.up->ne[0] == in.n_embd &&
        in.down->ne[2] == in.up->ne[2] &&
        !in.gate_up && !in.gate_up_bias && !in.gate_input_bias &&
        !in.up_bias && !in.gate_bias && !in.down_bias &&
        !in.up_scale && !in.gate_scale && !in.down_scale &&
        !in.weight_before_ffn && !in.has_lora;
    bool quantization = false;
    if (layout) {
        switch (requirements.quantization) {
            case expert_quantization::all_mxfp4:
                quantization = in.up->type == GGML_TYPE_MXFP4 &&
                    in.gate->type == GGML_TYPE_MXFP4 && in.down->type == GGML_TYPE_MXFP4;
                break;
            case expert_quantization::matching_up_gate:
                // Do not invent a model-specific type list: the selected
                // executor validates its actual vec-dot/quantization traits.
                quantization = in.gate->type == in.up->type;
                break;
            case expert_quantization::unsupported:
                break;
        }
    }
    const bool accepted = layout && quantization &&
        requirements.activation == expert_activation::swiglu_clamp && in.silu && !in.cache_compute &&
        (!requirements.single_token_or_prefill || in.n_tokens == 1 || in.prefill) &&
        (!requirements.finite_clamp || (std::isfinite(requirements.clamp) && requirements.clamp > 1e-6f)) &&
        (!requirements.require_post_norm || in.post_norm);
    if (!accepted) {
        return {false, {}, " requires contiguous DeepSeek-V4.1 clamped SwiGLU experts in all-MXFP4, or "
            "Muse Spark SwiGLU with per-expert post norm, without LoRA/bias/scales; no fallback"};
    }
    return {true, {requirements.activation, requirements.clamp, in.post_norm != nullptr,
        in.post_norm_eps, in.routed_scale == 0.0f ? 1.0f : in.routed_scale}, nullptr};
}

} // namespace llama_ssd
