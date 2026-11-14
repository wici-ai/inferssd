// SPDX-License-Identifier: Apache-2.0
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T, void (*release)(T *)>
using owner = std::unique_ptr<T, decltype(release)>;

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void test_top_k(const std::string & scenario, uint32_t seed) {
    const bool chained = scenario == "chained" || scenario == "chained-dist";
    const bool dist = scenario == "dist" || scenario == "chained-dist";
    const bool mapped = scenario == "mapped";
    std::vector<float> logits = scenario == "single" ? std::vector<float>{0.7f}
                                                    : std::vector<float>{-2.0f, 0.7f, -0.1f, 0.2f};
    int32_t k = 40;
    if (scenario == "maximum") {
        k = std::numeric_limits<int32_t>::max();
    } else if (scenario == "equal") {
        k = logits.size();
    } else if (scenario == "smaller") {
        k = 2;
    } else if (scenario == "disabled") {
        k = 0;
    } else if (scenario == "negative") {
        k = -1;
    } else if (chained) {
        k = 3;
    }

    owner<llama_sampler, llama_sampler_free> cpu(
            llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    owner<llama_sampler, llama_sampler_free> device(
            llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    for (auto * sampler : {cpu.get(), device.get()}) {
        if (chained) {
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(2));
        }
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(k));
        if (dist) {
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
        }
    }

    std::vector<llama_token> input_ids = mapped ? std::vector<llama_token>{7, 2, 10, 4}
                                               : std::vector<llama_token>{0, 1, 2, 3};
    std::vector<llama_token_data> reference;
    for (size_t i = 0; i < logits.size(); ++i) {
        reference.push_back({input_ids[i], logits[i], 0.0f});
    }
    llama_token_data_array expected = {reference.data(), reference.size(), -1, false};
    llama_sampler_apply(cpu.get(), &expected);

    owner<ggml_backend, ggml_backend_free> backend(ggml_backend_cpu_init(), ggml_backend_free);
    require(bool(backend), "CPU backend initialization failed");
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    require(device->iface->backend_init(device.get(), ggml_backend_cpu_buffer_type()),
            "CPU backend does not support the sampler chain");
    owner<ggml_context, ggml_free> ctx(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
    require(bool(ctx), "graph context allocation failed");
    auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, logits.size());
    ggml_set_input(input);
    auto * input_candidates = mapped ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, logits.size()) : nullptr;
    llama_sampler_data data = {input, nullptr, nullptr, input_candidates};
    auto * graph = ggml_new_graph(ctx.get());
    device->iface->backend_apply(device.get(), ctx.get(), graph, &data);
    for (auto * output : {data.logits, data.probs, data.sampled, data.candidates}) {
        if (output) {
            ggml_build_forward_expand(graph, output);
        }
    }
    owner<ggml_backend_buffer, ggml_backend_buffer_free> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    require(bool(buffer), "graph buffer allocation failed");
    ggml_backend_tensor_set(input, logits.data(), 0, logits.size() * sizeof(float));
    if (input_candidates) {
        ggml_backend_tensor_set(input_candidates, input_ids.data(), 0, logits.size() * sizeof(llama_token));
    }
    device->iface->backend_set_input(device.get());
    require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "graph execution failed");

    const size_t count = ggml_nelements(data.logits);
    require(count == expected.size, "backend retained the wrong number of candidates");
    std::vector<float> actual(count);
    std::vector<llama_token> ids(count);
    ggml_backend_tensor_get(data.logits, actual.data(), 0, count * sizeof(float));
    if (data.candidates) {
        ggml_backend_tensor_get(data.candidates, ids.data(), 0, count * sizeof(llama_token));
    } else {
        ids.assign(input_ids.begin(), input_ids.begin() + count);
    }
    // GGML TOP_K deliberately makes no ordering guarantee. Compare the full
    // token/logit mapping, and use its actual order when checking sampling.
    std::vector<llama_token> actual_ids = ids;
    std::vector<llama_token> expected_ids;
    std::vector<llama_token_data> ordered;
    for (size_t i = 0; i < count; ++i) {
        expected_ids.push_back(expected.data[i].id);
        ordered.push_back({ids[i], actual[i], 0.0f});
        auto * match = std::find_if(expected.data, expected.data + expected.size,
                                   [&](const llama_token_data & token) { return token.id == ids[i]; });
        require(match != expected.data + expected.size, "backend retained an unexpected vocabulary token");
        require(actual[i] == match->logit, "backend changed a retained token's logit");
    }
    std::sort(actual_ids.begin(), actual_ids.end());
    std::sort(expected_ids.begin(), expected_ids.end());
    require(actual_ids == expected_ids, "backend changed the complete retained vocabulary support");
    if (dist) {
        owner<llama_sampler, llama_sampler_free> ordered_dist(llama_sampler_init_dist(seed), llama_sampler_free);
        llama_token_data_array ordered_expected = {ordered.data(), ordered.size(), -1, false};
        llama_sampler_apply(ordered_dist.get(), &ordered_expected);
        std::vector<float> probabilities(count);
        llama_token selected = -1;
        ggml_backend_tensor_get(data.probs, probabilities.data(), 0, count * sizeof(float));
        ggml_backend_tensor_get(data.sampled, &selected, 0, sizeof(selected));
        require(selected == ordered[ordered_expected.selected].id, "backend sampled the wrong vocabulary token");
        for (size_t i = 0; i < count; ++i) {
            require(std::abs(probabilities[i] - ordered[i].p) < 1e-6f,
                    "backend changed the final probability distribution");
        }
    }
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "expected one test scenario");
        const std::string scenario = argv[1];
        for (uint32_t seed = 0; seed < (scenario == "dist" || scenario == "chained-dist" ? 32u : 1u); ++seed) {
            test_top_k(scenario, seed);
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "backend top-k bounds: %s\n", error.what());
        return 1;
    }
}
