// SPDX-License-Identifier: Apache-2.0
#include "llama-graph.h"
#include "ggml-cpu.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>

static void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

static void close_to(float actual, double expected, const char * message) {
    require(std::isfinite(actual), "router produced a non-finite weight");
    require(std::abs(double(actual) - expected) <= std::abs(expected) * 2e-6 + 1e-30, message);
}

static void test_arch(llm_arch arch) {
    constexpr int selected = 6, tokens = 4;
    constexpr float scale = 1.5f;
    ggml_context * ctx = ggml_init({1024 * 1024, nullptr, false});
    require(ctx != nullptr, "router test context allocation failed");

    // Four tokens after top-six selection: scores below the former F16 floor,
    // all zero, ordinary scores, and scores whose sum is below the DSV41 epsilon.
    const std::array<float, tokens> units = {0x1p-30f, 0.0f, 1.0f, 1e-23f};
    ggml_tensor * scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, selected, tokens);
    auto * input = static_cast<float *>(scores->data);
    for (int t = 0; t < tokens; ++t) {
        for (int i = 0; i < selected; ++i) { input[t * selected + i] = (i + 1) * units[t]; }
    }

    ggml_tensor * sum = ggml_sum_rows(ctx, scores);
    // This is the same production helper called by build_moe_ffn. A regression
    // to the generic F16 clamp must fail on the first column, even if the SSD
    // and ordinary execution paths would otherwise agree with each other.
    ggml_tensor * denominator = llm_moe_weights_denominator(ctx, sum, arch);
    ggml_tensor * weights = ggml_scale(ctx, ggml_div(ctx, scores, denominator), scale);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, weights);
    require(ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS,
            "router graph execution failed");

    const auto * actual = static_cast<const float *>(weights->data);
    const auto * actual_denominator = static_cast<const float *>(denominator->data);
    const bool deepseek = arch == LLM_ARCH_DEEPSEEK41;
    for (int i = 0; i < selected; ++i) {
        const double rank = i + 1;
        // Independent expected proportions for the known score vectors. The
        // epsilon-dominated case has total unnormalized mass 2.1e-22.
        const double tiny = deepseek ? rank / 14.0 : std::ldexp(1.5 * rank, -16);
        const double epsilon = deepseek ? rank * (1.5e-23 / 1.021e-20) : rank * 2.4576e-19;
        close_to(actual[i], tiny, "tiny router scores were attenuated or legacy behavior changed");
        close_to(actual[selected + i], 0.0, "zero router scores must remain zero");
        close_to(actual[2 * selected + i], rank / 14.0, "ordinary router normalization changed");
        close_to(actual[3 * selected + i], epsilon, "epsilon-dominated router normalization changed");
    }
    if (deepseek) {
        require(actual_denominator[0] < 0x1p-14f, "DSV41 denominator was clamped to F16 minimum");
        close_to(actual_denominator[1], 1e-20, "DSV41 zero guard changed");
        double total = 0.0;
        for (int i = 0; i < selected; ++i) { total += actual[i]; }
        require(std::abs(total - scale) < 2e-6, "tiny DSV41 scores lost routed output scale");
    } else {
        require(actual_denominator[0] == 0x1p-14f && actual_denominator[1] == 0x1p-14f &&
                actual_denominator[3] == 0x1p-14f, "legacy F16 denominator floor changed");
    }
    ggml_free(ctx);
}

int main() {
    try {
        test_arch(LLM_ARCH_DEEPSEEK41);
        test_arch(LLM_ARCH_DEEPSEEK2);
        test_arch(LLM_ARCH_QWEN3MOE);
        std::puts("MoE router normalization: tiny, zero, normal, epsilon and legacy cases passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
