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
