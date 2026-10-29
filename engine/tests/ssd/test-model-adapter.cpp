// SPDX-License-Identifier: Apache-2.0
// Real tensor metadata/ggml layout predicates; no payload or device execution.
#include "ssd/model-adapter.h"
#include "llama-hparams.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {
size_t comparisons = 0;
#define CHECK(value) do { if (!(value)) { \
    std::fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #value); std::abort(); \
} } while (0)

ggml_tensor tensor(ggml_type type, int64_t columns, int64_t rows, int64_t experts) {
    ggml_tensor value{};
    value.type = type;
    value.ne[0] = columns; value.ne[1] = rows; value.ne[2] = experts; value.ne[3] = 1;
    value.nb[0] = ggml_type_size(type);
    value.nb[1] = ggml_row_size(type, columns);
    value.nb[2] = value.nb[1] * rows;
    value.nb[3] = value.nb[2] * experts;
    return value;
}

struct fixture {
    ggml_tensor up = tensor(GGML_TYPE_MXFP4, 256, 512, 4);
    ggml_tensor gate = up;
    ggml_tensor down = tensor(GGML_TYPE_MXFP4, 512, 256, 4);
    ggml_tensor norm = tensor(GGML_TYPE_F32, 256, 4, 1);

    llama_ssd::expert_inputs inputs() const {
        return {true, true, &up, &gate, &down, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            256, 1, true, false, false, false, false, &norm, 1e-5f, 0.0f};
    }
};

// Frozen acceptance expressions from main b3904ade6's shared graph callsite,
// and qwen4exp's: SiLU-gated experts, up and gate of one type, no post norm
// required. Keep this independent of the extracted requirement flags and validator.
bool baseline_accepts(llm_arch arch, float clamp, const llama_ssd::expert_inputs & in) {
    const bool layout = in.backend_available && in.runtime_available && in.up && in.gate && in.down &&
        ggml_is_contiguous(in.up) && ggml_is_contiguous(in.gate) && ggml_is_contiguous(in.down) &&
        ggml_are_same_shape(in.up, in.gate) && in.up->ne[3] == 1 &&
        in.down->ne[3] == 1 && in.down->ne[0] == in.up->ne[1] &&
        in.down->ne[1] == in.n_embd && in.up->ne[0] == in.n_embd &&
        in.down->ne[2] == in.up->ne[2] &&
        !in.gate_up && !in.gate_up_bias && !in.gate_input_bias &&
        !in.up_bias && !in.gate_bias && !in.down_bias &&
        !in.up_scale && !in.gate_scale && !in.down_scale &&
        !in.weight_before_ffn && !in.has_lora;
    const bool mxfp4 = layout && in.up->type == GGML_TYPE_MXFP4 &&
        in.gate->type == GGML_TYPE_MXFP4 && in.down->type == GGML_TYPE_MXFP4;
    const bool deepseek = layout && arch == LLM_ARCH_DEEPSEEK41 &&
        in.silu && !in.cache_compute && (in.n_tokens == 1 || in.prefill) && mxfp4 &&
        std::isfinite(clamp) && clamp > 1e-6f;
    const bool muse = layout && arch == LLM_ARCH_MUSE_SPARK && in.silu && !in.cache_compute &&
        in.post_norm && in.gate->type == in.up->type;
    const bool qwen = layout && arch == LLM_ARCH_QWEN4EXP && in.silu && !in.cache_compute &&
        in.gate->type == in.up->type;
    return deepseek || muse || qwen;
}

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }

bool compare(llm_arch arch, float clamp, const llama_ssd::expert_inputs & in, int layer = 0) {
    llama_hparams hparams{};
    hparams.swiglu_clamp_exp[0] = clamp;
    const auto requirements = llama_ssd::describe_experts(arch, hparams, layer);
    const auto actual = llama_ssd::validate_experts(requirements, in);
    const float old_clamp = arch == LLM_ARCH_DEEPSEEK41 && layer >= 0 ? clamp : 0.0f;
    CHECK(actual.accepted == baseline_accepts(arch, old_clamp, in));
    ++comparisons;
    if (actual.accepted) {
        const auto & out = actual.execution;
        CHECK(out.activation == (arch == LLM_ARCH_DEEPSEEK41 ? llama_ssd::expert_activation::swiglu_clamp :
            llama_ssd::expert_activation::swiglu));
        CHECK(same_bits(out.clamp, old_clamp));
        CHECK(out.post_norm == (in.post_norm != nullptr));
        CHECK(same_bits(out.post_norm_eps, in.post_norm_eps));
        CHECK(same_bits(out.routed_scale, in.routed_scale == 0.0f ? 1.0f : in.routed_scale));
        CHECK(actual.error == nullptr);
    } else {
        CHECK(std::strcmp(actual.error, " requires the model's routed experts as contiguous, separate up/gate/down "
            "SiLU-gated tensors of a supported format, without LoRA, biases or scales; no fallback") == 0);
    }
    return actual.accepted;
}

void model_and_numeric_boundaries() {
    fixture f;
    auto in = f.inputs();
    CHECK(compare(LLM_ARCH_DEEPSEEK41, 7.0f, in));
    CHECK(compare(LLM_ARCH_MUSE_SPARK, 0.0f, in));
    CHECK(compare(LLM_ARCH_QWEN4EXP, 0.0f, in));
    CHECK(!compare(LLM_ARCH_LLAMA, 7.0f, in));
    const float limits[] = {-INFINITY, -1.0f, -0.0f, 0.0f, 1e-6f,
        std::nextafter(1e-6f, INFINITY), 1.0f, INFINITY, std::numeric_limits<float>::quiet_NaN()};
    for (float limit : limits) {
        compare(LLM_ARCH_DEEPSEEK41, limit, in);
        CHECK(compare(LLM_ARCH_MUSE_SPARK, limit, in));
        CHECK(compare(LLM_ARCH_QWEN4EXP, limit, in)); // no clamp: the layer limit is not read
    }
    CHECK(!compare(LLM_ARCH_DEEPSEEK41, 7.0f, in, -1));
    CHECK(compare(LLM_ARCH_MUSE_SPARK, 7.0f, in, -1));
    CHECK(compare(LLM_ARCH_QWEN4EXP, 7.0f, in, -1));
    llama_hparams layers{};
    layers.swiglu_clamp_exp[0] = 3.0f;
    layers.swiglu_clamp_exp[1] = 9.0f;
    CHECK(llama_ssd::describe_experts(LLM_ARCH_DEEPSEEK41, layers, 0).clamp == 3.0f);
    CHECK(llama_ssd::describe_experts(LLM_ARCH_DEEPSEEK41, layers, 1).clamp == 9.0f);
    for (float scale : limits) {
        in.routed_scale = scale;
        in.post_norm_eps = scale; // Preserve the backend's existing validation boundary.
        CHECK(compare(LLM_ARCH_DEEPSEEK41, 7.0f, in));
        CHECK(compare(LLM_ARCH_MUSE_SPARK, 7.0f, in));
        CHECK(compare(LLM_ARCH_QWEN4EXP, 7.0f, in));
    }
    in = f.inputs(); in.post_norm = nullptr;
    CHECK(compare(LLM_ARCH_DEEPSEEK41, 7.0f, in)); // optional for DeepSeek
    CHECK(!compare(LLM_ARCH_MUSE_SPARK, 7.0f, in));
    CHECK(compare(LLM_ARCH_QWEN4EXP, 7.0f, in));
    in = f.inputs(); in.cache_compute = true;
    CHECK(!compare(LLM_ARCH_DEEPSEEK41, 7.0f, in));
    CHECK(!compare(LLM_ARCH_MUSE_SPARK, 7.0f, in));
    CHECK(!compare(LLM_ARCH_QWEN4EXP, 7.0f, in));
    for (int tokens : {1, 2, 32}) {
        for (bool prefill : {false, true}) {
            in = f.inputs(); in.n_tokens = tokens; in.prefill = prefill;
            compare(LLM_ARCH_DEEPSEEK41, 7.0f, in);
            CHECK(compare(LLM_ARCH_MUSE_SPARK, 7.0f, in));
            CHECK(compare(LLM_ARCH_QWEN4EXP, 7.0f, in));
        }
    }
}

// qwen4exp's routed experts as the model loads them: separate Q8_0 up/gate
// [2560, 640, 512] and down [640, 2560, 512], no post norm. Its fused gate_up
// form, per-expert scales and LoRA stay on the graph.
void qwen4exp_experts() {
    fixture f;
    f.up = f.gate = tensor(GGML_TYPE_Q8_0, 2560, 640, 512);
    f.down = tensor(GGML_TYPE_Q8_0, 640, 2560, 512);
    auto in = f.inputs();
    in.n_embd = 2560; in.post_norm = nullptr; in.post_norm_eps = 0.0f; in.routed_scale = 1.0f;
    CHECK(compare(LLM_ARCH_QWEN4EXP, 0.0f, in));
    const auto accepted = llama_ssd::validate_experts(
        llama_ssd::describe_experts(LLM_ARCH_QWEN4EXP, llama_hparams{}, 0), in);
    CHECK(accepted.accepted && accepted.execution.activation == llama_ssd::expert_activation::swiglu &&
        !accepted.execution.post_norm && accepted.execution.routed_scale == 1.0f);
    for (int prefill = 0; prefill < 2; ++prefill) {
        auto batch = in; batch.n_tokens = 512; batch.prefill = prefill;
        CHECK(compare(LLM_ARCH_QWEN4EXP, 0.0f, batch));
    }
    const ggml_tensor gate_up = tensor(GGML_TYPE_Q8_0, 2560, 1280, 512);
    const ggml_tensor scale = tensor(GGML_TYPE_F32, 512, 1, 1);
    auto fused = in; fused.up = nullptr; fused.gate = nullptr; fused.gate_up = &gate_up;
    CHECK(!compare(LLM_ARCH_QWEN4EXP, 0.0f, fused));
    fused = in; fused.gate_up = &gate_up;
    CHECK(!compare(LLM_ARCH_QWEN4EXP, 0.0f, fused));
    using member = const ggml_tensor * llama_ssd::expert_inputs::*;
    for (member field : {&llama_ssd::expert_inputs::up_scale, &llama_ssd::expert_inputs::gate_scale,
            &llama_ssd::expert_inputs::down_scale}) {
        auto scaled = in; scaled.*field = &scale;
        CHECK(!compare(LLM_ARCH_QWEN4EXP, 0.0f, scaled));
    }
    auto lora = in; lora.has_lora = true;
    CHECK(!compare(LLM_ARCH_QWEN4EXP, 0.0f, lora));
    auto gelu = in; gelu.silu = false;
    CHECK(!compare(LLM_ARCH_QWEN4EXP, 0.0f, gelu));
    const ggml_tensor q4_k = tensor(GGML_TYPE_Q4_K, 2560, 640, 512);
    auto mixed = in; mixed.gate = &q4_k;
    CHECK(!compare(LLM_ARCH_QWEN4EXP, 0.0f, mixed));
}

void layout_refusals() {
    fixture f;
    using member = const ggml_tensor * llama_ssd::expert_inputs::*;
    for (member field : {&llama_ssd::expert_inputs::up, &llama_ssd::expert_inputs::gate,
            &llama_ssd::expert_inputs::down}) {
        auto in = f.inputs(); in.*field = nullptr;
        CHECK(!compare(LLM_ARCH_DEEPSEEK41, 7.0f, in));
        CHECK(!compare(LLM_ARCH_MUSE_SPARK, 7.0f, in));
        CHECK(!compare(LLM_ARCH_QWEN4EXP, 7.0f, in));
    }
    for (member field : {&llama_ssd::expert_inputs::gate_up, &llama_ssd::expert_inputs::gate_up_bias,
            &llama_ssd::expert_inputs::gate_input_bias, &llama_ssd::expert_inputs::up_bias,
            &llama_ssd::expert_inputs::gate_bias, &llama_ssd::expert_inputs::down_bias,
            &llama_ssd::expert_inputs::up_scale, &llama_ssd::expert_inputs::gate_scale,
            &llama_ssd::expert_inputs::down_scale}) {
        auto in = f.inputs(); in.*field = &f.up;
        CHECK(!compare(LLM_ARCH_DEEPSEEK41, 7.0f, in));
        CHECK(!compare(LLM_ARCH_MUSE_SPARK, 7.0f, in));
        CHECK(!compare(LLM_ARCH_QWEN4EXP, 7.0f, in));
    }
    using flag = bool llama_ssd::expert_inputs::*;
    for (flag field : {&llama_ssd::expert_inputs::backend_available,
            &llama_ssd::expert_inputs::runtime_available, &llama_ssd::expert_inputs::silu,
            &llama_ssd::expert_inputs::weight_before_ffn, &llama_ssd::expert_inputs::has_lora}) {
        auto in = f.inputs(); in.*field = !(in.*field);
        CHECK(!compare(LLM_ARCH_DEEPSEEK41, 7.0f, in));
        CHECK(!compare(LLM_ARCH_MUSE_SPARK, 7.0f, in));
        CHECK(!compare(LLM_ARCH_QWEN4EXP, 7.0f, in));
    }
    for (int defect = 0; defect < 8; ++defect) {
        fixture bad;
        switch (defect) {
            case 0: bad.up.nb[1] *= 2; break;
            case 1: bad.gate.nb[1] *= 2; break;
            case 2: bad.down.nb[1] *= 2; break;
            case 3: bad.gate = tensor(GGML_TYPE_MXFP4, 256, 256, 4); break;
            case 4: bad.down = tensor(GGML_TYPE_MXFP4, 256, 256, 4); break;
            case 5: bad.down = tensor(GGML_TYPE_MXFP4, 512, 512, 4); break;
            case 6: bad.down = tensor(GGML_TYPE_MXFP4, 512, 256, 3); break;
            case 7: bad.up.ne[3] = bad.gate.ne[3] = 2; break;
        }
        CHECK(!compare(LLM_ARCH_DEEPSEEK41, 7.0f, bad.inputs()));
        CHECK(!compare(LLM_ARCH_MUSE_SPARK, 7.0f, bad.inputs()));
        CHECK(!compare(LLM_ARCH_QWEN4EXP, 7.0f, bad.inputs()));
    }
}

void quantization_contract() {
    const ggml_type types[] = {GGML_TYPE_MXFP4, GGML_TYPE_Q8_0, GGML_TYPE_BF16,
        GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_F16, GGML_TYPE_F32};
    for (ggml_type up : types) {
        for (ggml_type gate : types) {
            for (ggml_type down : types) {
                fixture f;
                f.up = tensor(up, 256, 512, 4);
                f.gate = tensor(gate, 256, 512, 4);
                f.down = tensor(down, 512, 256, 4);
                compare(LLM_ARCH_DEEPSEEK41, 7.0f, f.inputs());
                compare(LLM_ARCH_MUSE_SPARK, 7.0f, f.inputs());
                compare(LLM_ARCH_QWEN4EXP, 7.0f, f.inputs());
            }
        }
    }
    // No process/layer-wide format restriction is added by model admission.
    // CUDA residency's separate, inherited cross-layer restriction is unchanged.
    fixture first, next;
    first.up = first.gate = tensor(GGML_TYPE_Q4_K, 256, 512, 4);
    first.down = tensor(GGML_TYPE_Q5_K, 512, 256, 4);
    next.up = next.gate = tensor(GGML_TYPE_Q8_0, 256, 512, 4);
    next.down = tensor(GGML_TYPE_Q4_K, 512, 256, 4);
    CHECK(compare(LLM_ARCH_MUSE_SPARK, 0.0f, first.inputs()));
    CHECK(compare(LLM_ARCH_MUSE_SPARK, 0.0f, next.inputs()));
    CHECK(compare(LLM_ARCH_MUSE_SPARK, 0.0f, first.inputs()));
}
} // namespace

int main() {
    model_and_numeric_boundaries();
    qwen4exp_experts();
    layout_refusals();
    quantization_contract();
    std::printf("model adapter: %zu frozen acceptance/descriptor comparisons passed\n", comparisons);
}
