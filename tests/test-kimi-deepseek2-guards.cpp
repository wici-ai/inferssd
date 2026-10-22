#include "llama-kimi-compat.h"
#include "dense-parity.cuh"

#include <cstdio>

static int expect_true(bool value, const char * msg) {
    if (!value) {
        std::fprintf(stderr, "expected true: %s\n", msg);
        return 1;
    }
    return 0;
}

static int expect_false(bool value, const char * msg) {
    if (value) {
        std::fprintf(stderr, "expected false: %s\n", msg);
        return 1;
    }
    return 0;
}

static int expect_float_eq(float actual, float expected, const char * msg) {
    if (actual != expected) {
        std::fprintf(stderr, "expected %s = %.6f, got %.6f\n", msg, expected, actual);
        return 1;
    }
    return 0;
}

static int expect_i64_eq(int64_t actual, int64_t expected, const char * msg) {
    if (actual != expected) {
        std::fprintf(stderr, "expected %s = %lld, got %lld\n", msg,
            (long long) expected, (long long) actual);
        return 1;
    }
    return 0;
}

static int expect_dense_parity_layout(int ngl, int host_last) {
    int rc = 0;
    for (int layer = 0; layer <= 92; ++layer) {
        char name[128];
        if (layer <= host_last) {
            std::snprintf(name, sizeof(name),
                "blk.%d.ffn_up_shexp.weight#dense-prefetch", layer);
        } else {
            std::snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", layer);
        }
        const bool selected = ggml_cuda_dense_parity_q4k_selected_for_spec("0-29", name);
        if (selected != (layer <= 29)) {
            std::fprintf(stderr,
                "unexpected dense parity mapping: ngl=%d layer=%d host_last=%d selected=%d\n",
                ngl, layer, host_last, selected ? 1 : 0);
            rc = 1;
        }
    }
    return rc;
}

int main() {
    llama_hparams kimi = {};
    kimi.n_expert = 384;
    kimi.n_expert_groups = 1;
    kimi.rope_yarn_log_mul = 0.1f;

    llama_hparams deepseek = {};
    deepseek.n_expert = 256;
    deepseek.n_expert_groups = 8;
    deepseek.rope_yarn_log_mul = 1.0f;

    llama_hparams grouped_kimi_size = kimi;
    grouped_kimi_size.n_expert_groups = 8;

    int rc = 0;
    rc |= expect_true(llama_model_name_is_kimi("Moonshot Kimi K2"), "case-insensitive Kimi model-name guard");
    rc |= expect_false(llama_model_name_is_kimi("DeepSeek-V3"), "DeepSeek model names are not Kimi");

    rc |= expect_true(
            llama_should_defer_kimi_experts_on_gpu(LLM_ARCH_KIMI_LINEAR, "any-name"),
            "native Kimi arch defers expert tensors under GPU offload");
    rc |= expect_true(
            llama_should_defer_kimi_experts_on_gpu(LLM_ARCH_DEEPSEEK2, "Kimi-K2-Instruct"),
            "Kimi K2 reports deepseek2 but still defers expert tensors");
    rc |= expect_false(
            llama_should_defer_kimi_experts_on_gpu(LLM_ARCH_DEEPSEEK2, "DeepSeek-V3"),
            "ordinary DeepSeek2 keeps merged DeepSeek tensor placement");

    rc |= expect_true(
            llama_is_kimi_k2_deepseek2_layout(LLM_ARCH_DEEPSEEK2, kimi),
            "Kimi K2 deepseek2 layout is recognized");
    rc |= expect_false(
            llama_is_kimi_k2_deepseek2_layout(LLM_ARCH_DEEPSEEK2, grouped_kimi_size),
            "grouped DeepSeek2 routing is not treated as Kimi K2");
    rc |= expect_false(
            llama_is_kimi_k2_deepseek2_layout(static_cast<llm_arch>(69), kimi),
            "retired architecture is not treated as Kimi K2");

    rc |= expect_true(
            llama_moe_should_use_plain_top_k(LLM_ARCH_DEEPSEEK2, kimi),
            "Kimi K2 deepseek2 layout uses plain top-k");
    rc |= expect_false(
            llama_moe_should_use_plain_top_k(LLM_ARCH_DEEPSEEK2, deepseek),
            "ordinary grouped DeepSeek2 keeps argsort top-k path");

    rc |= expect_float_eq(
            llama_deepseek2_yarn_log_mul_for_graph(LLM_ARCH_DEEPSEEK2, kimi),
            0.1f,
            "Kimi graph YaRN multiplier");
    rc |= expect_float_eq(
            llama_deepseek2_yarn_log_mul_for_graph(LLM_ARCH_DEEPSEEK2, deepseek),
            0.1f,
            "DeepSeek graph YaRN multiplier after convert-factor correction");

    const auto off = llama_k3_decode_expert_policy_parse(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 16, 95);
    rc |= expect_true(off.valid(), "unset K3 decode expert policy is valid");
    rc |= expect_false(off.enabled(), "unset K3 decode expert policy is off");
    rc |= expect_i64_eq(off.top_k, 16, "disabled policy preserves model top-k");
    rc |= expect_i64_eq(off.layer_end, 94, "disabled policy spans the model by default");

    const auto hard = llama_k3_decode_expert_policy_parse(
        "hard", "12", nullptr, nullptr, nullptr, "20", "94", 16, 95);
    rc |= expect_true(hard.valid(), "hard K3 decode expert policy parses");
    rc |= expect_true(hard.enabled(), "hard K3 decode expert policy is enabled");
    rc |= expect_i64_eq(hard.top_k, 12, "hard policy top-k");
    rc |= expect_true(hard.applies(1, 20), "hard policy applies to decode in its layer range");
    rc |= expect_false(hard.applies(2, 20), "hard policy leaves prefill untouched");
    rc |= expect_false(hard.applies(1, 19), "hard policy protects shallow layers outside its range");

    auto hard_small_batch = hard;
    hard_small_batch.small_batch = true;
    rc |= expect_true(hard_small_batch.applies(2, 20), "opt-in K3 policy applies to a two-token decode batch");
    rc |= expect_true(hard_small_batch.applies(8, 20), "opt-in K3 policy includes the eight-token limit");
    rc |= expect_false(hard_small_batch.applies(9, 20), "opt-in K3 policy stops above eight tokens");
    rc |= expect_false(hard_small_batch.applies(0, 20), "opt-in K3 policy rejects an empty batch");
    rc |= expect_false(
        llama_k3_decode_expert_small_batch_enabled(nullptr),
        "unset K3 small-batch policy is disabled");
    rc |= expect_false(
        llama_k3_decode_expert_small_batch_enabled("0"),
        "zero K3 small-batch policy is disabled");
    rc |= expect_true(
        llama_k3_decode_expert_small_batch_enabled("1"),
        "nonzero K3 small-batch policy is enabled");

    const auto mass = llama_k3_decode_expert_policy_parse(
        "mass", nullptr, "0.95", nullptr, nullptr, nullptr, nullptr, 16, 95);
    rc |= expect_true(mass.valid(), "mass K3 decode expert policy parses");
    rc |= expect_float_eq(mass.mass, 0.95f, "mass policy threshold");
    rc |= expect_i64_eq(mass.min_k, 4, "mass policy default minimum k");
    rc |= expect_i64_eq(mass.max_k, 0, "mass policy default maximum k is disabled");

    const auto capped_mass = llama_k3_decode_expert_policy_parse(
        "mass", nullptr, "0.9", "4", "8", nullptr, nullptr, 16, 95);
    rc |= expect_true(capped_mass.valid(), "capped mass K3 decode expert policy parses");
    rc |= expect_i64_eq(capped_mass.max_k, 8, "mass policy maximum k");

    const auto bad_hard = llama_k3_decode_expert_policy_parse(
        "hard", "12junk", nullptr, nullptr, nullptr, nullptr, nullptr, 16, 95);
    rc |= expect_false(bad_hard.valid(), "hard policy rejects a non-integer top-k");
    rc |= expect_true(
        bad_hard.error == llama_k3_decode_expert_policy_error::invalid_top_k,
        "hard policy reports its top-k error");

    const auto bad_mass = llama_k3_decode_expert_policy_parse(
        "mass", nullptr, "1.01", "4", nullptr, nullptr, nullptr, 16, 95);
    rc |= expect_false(bad_mass.valid(), "mass policy rejects thresholds above one");
    rc |= expect_true(
        bad_mass.error == llama_k3_decode_expert_policy_error::invalid_mass,
        "mass policy reports its threshold error");

    const auto bad_min = llama_k3_decode_expert_policy_parse(
        "mass", nullptr, "0.95", "0", nullptr, nullptr, nullptr, 16, 93);
    rc |= expect_false(bad_min.valid(), "mass policy rejects a zero minimum k");
    rc |= expect_true(
        bad_min.error == llama_k3_decode_expert_policy_error::invalid_min_k,
        "mass policy reports its minimum-k error");

    const auto bad_max = llama_k3_decode_expert_policy_parse(
        "mass", nullptr, "0.95", "4", "17", nullptr, nullptr, 16, 93);
    rc |= expect_false(bad_max.valid(), "mass policy rejects a maximum k above model top-k");
    rc |= expect_true(
        bad_max.error == llama_k3_decode_expert_policy_error::invalid_max_k,
        "mass policy reports its maximum-k error");

    const auto min_above_max = llama_k3_decode_expert_policy_parse(
        "mass", nullptr, "0.95", "9", "8", nullptr, nullptr, 16, 93);
    rc |= expect_false(min_above_max.valid(), "mass policy rejects minimum k above maximum k");
    rc |= expect_true(
        min_above_max.error == llama_k3_decode_expert_policy_error::invalid_min_k,
        "mass policy reports a minimum k above maximum k");

    const auto bad_layers = llama_k3_decode_expert_policy_parse(
        "hard", "12", nullptr, nullptr, nullptr, "30", "93", 16, 93);
    rc |= expect_false(bad_layers.valid(), "policy rejects a layer past the model end");
    rc |= expect_true(
        bad_layers.error == llama_k3_decode_expert_policy_error::invalid_layer_range,
        "policy reports its layer-range error");

    const auto bad_mode = llama_k3_decode_expert_policy_parse(
        "adaptive", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 16, 93);
    rc |= expect_false(bad_mode.valid(), "policy rejects an unknown mode");
    rc |= expect_true(
        bad_mode.error == llama_k3_decode_expert_policy_error::invalid_mode,
        "policy reports its mode error");

    const float sorted_weights[] = { 0.50f, 0.25f, 0.15f, 0.10f };
    rc |= expect_i64_eq(
        (int64_t) llama_k3_decode_expert_prefix_count(sorted_weights, 4, 0.75f, 1),
        2,
        "mass prefix includes the route that reaches the threshold");
    rc |= expect_i64_eq(
        (int64_t) llama_k3_decode_expert_prefix_count(sorted_weights, 4, 0.60f, 3),
        3,
        "mass prefix enforces the minimum k");
    rc |= expect_i64_eq(
        (int64_t) llama_k3_decode_expert_prefix_count(sorted_weights, 4, 0.99f, 1),
        4,
        "mass prefix falls back to all routes when needed");

    rc |= expect_true(
        ggml_cuda_dense_parity_q4k_selected_for_spec(nullptr,
            "blk.22.ffn_up_shexp.weight#dense-prefetch"),
        "staged Q4_K parity remains active when resident parity is disabled");
    rc |= expect_false(
        ggml_cuda_dense_parity_q4k_selected_for_spec(nullptr,
            "blk.22.ffn_up_shexp.weight"),
        "resident parity is disabled by default");
    rc |= expect_true(
        ggml_cuda_dense_parity_q4k_selected_for_spec("0-29",
            "blk.0.ffn_up_shexp.weight"),
        "resident parity includes original stream layer zero");
    rc |= expect_true(
        ggml_cuda_dense_parity_q4k_selected_for_spec("0-29",
            "blk.29.ffn_up_shexp.weight"),
        "resident parity includes original stream layer 29");
    rc |= expect_false(
        ggml_cuda_dense_parity_q4k_selected_for_spec("0-29",
            "blk.30.ffn_up_shexp.weight"),
        "resident parity leaves original resident layer 30 unchanged");
    rc |= expect_false(
        ggml_cuda_dense_parity_q4k_selected_for_spec("0-29",
            "blk.22.ffn_up_exps.weight"),
        "resident parity never selects expert matrices");
    rc |= expect_false(
        ggml_cuda_dense_parity_q4k_selected_for_spec("0-29,broken",
            "blk.22.ffn_up_shexp.weight"),
        "resident parity rejects a partially valid layer list");
    rc |= expect_true(
        ggml_cuda_dense_parity_f16_selected_for_spec("0-29", "blk.22.ssm_f_b.weight"),
        "resident parity selects the F16 state-space matvec");
    rc |= expect_false(
        ggml_cuda_dense_parity_f16_selected_for_spec("0-29", "blk.30.ssm_f_b.weight"),
        "resident F16 parity stops at layer 29");
    rc |= expect_true(
        ggml_cuda_dense_parity_f32_selected_for_spec("0-29", "blk.22.ffn_gate_inp.weight"),
        "resident parity selects the F32 router matvec");
    rc |= expect_false(
        ggml_cuda_dense_parity_f32_selected_for_spec("0-29", "blk.30.ffn_gate_inp.weight"),
        "resident F32 parity stops at layer 29");

    rc |= expect_dense_parity_layout(64, 29);
    rc |= expect_dense_parity_layout(72, 21);
    rc |= expect_dense_parity_layout(78, 15);
    rc |= expect_dense_parity_layout(84, 9);

    return rc;
}
