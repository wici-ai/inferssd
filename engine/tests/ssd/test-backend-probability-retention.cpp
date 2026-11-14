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

static void test_retention(const std::string & scenario, uint32_t seed) {
    const bool top_p = scenario.rfind("top-p-", 0) == 0;
    const std::string mode = scenario.substr(6);
    const bool dist = mode == "dist";
    const bool mapped = mode == "mapped";
    std::vector<float> logits = {-2.0f, 1.0f, 0.2f, -0.4f, 0.0f};
    size_t min_keep = 3;
    float p = top_p ? 0.1f : 0.8f;
    if (mode == "default") {
        min_keep = 0;
    } else if (mode == "maximum") {
        min_keep = std::numeric_limits<size_t>::max();
    } else if (mode == "single") {
        logits = {0.7f};
    } else if (mode == "more-than-minimum") {
        min_keep = 2;
        p = top_p ? 0.99f : 0.01f;
    } else if (mode == "disabled") {
        p = top_p ? 1.0f : 0.0f;
    } else if (mode == "zero") {
        p = 0.0f;
    } else if (mode == "unit") {
        p = 1.0f;
        min_keep = 0;
    } else if (mode == "above-unit") {
        p = 1.05f;
        min_keep = 0;
    } else if (mode == "ties") {
        logits = {1.0f, -1.0f, 1.0f, 1.0f};
        p = 1.0f;
        min_keep = 0;
    } else if (mode == "boundary") {
        logits = {std::log(0.5f), 0.0f, std::log(0.25f), std::log(0.5f)};
        p = 0.5f;
        min_keep = 0;
    }

    owner<llama_sampler, llama_sampler_free> cpu(
            llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    owner<llama_sampler, llama_sampler_free> device(
            llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    for (auto * sampler : {cpu.get(), device.get()}) {
        llama_sampler_chain_add(sampler, top_p ? llama_sampler_init_top_p(p, min_keep)
                                               : llama_sampler_init_min_p(p, min_keep));
        if (mode == "chained") {
            llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.9f, 2));
        }
        if (dist) {
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
        }
    }

    std::vector<llama_token> input_ids;
    std::vector<llama_token_data> reference;
    for (size_t i = 0; i < logits.size(); ++i) {
        input_ids.push_back(mapped ? 11 + 3 * i : i);
        reference.push_back({input_ids.back(), logits[i], 0.0f});
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
        ggml_backend_tensor_set(input_candidates, input_ids.data(), 0, input_ids.size() * sizeof(llama_token));
    }
    device->iface->backend_set_input(device.get());
    require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "graph execution failed");

    const size_t count = ggml_nelements(data.logits);
    require(count == logits.size(), "probability mask changed the candidate tensor width");
    std::vector<float> actual(count);
    std::vector<llama_token> ids = input_ids;
    ggml_backend_tensor_get(data.logits, actual.data(), 0, count * sizeof(float));
    if (data.candidates) {
        ggml_backend_tensor_get(data.candidates, ids.data(), 0, count * sizeof(llama_token));
    }
    std::vector<llama_token> support;
    std::vector<llama_token_data> ordered;
    for (size_t i = 0; i < count; ++i) {
        const auto * match = std::find_if(expected.data, expected.data + expected.size,
                                         [&](const llama_token_data & token) { return token.id == ids[i]; });
        if (match != expected.data + expected.size) {
            require(actual[i] == match->logit, "backend removed or changed a required candidate");
            support.push_back(ids[i]);
            ordered.push_back({ids[i], match->logit, 0.0f});
        } else {
            require(std::isinf(actual[i]) && actual[i] < 0.0f, "backend retained a filtered candidate");
            ordered.push_back({ids[i], -INFINITY, 0.0f});
        }
    }
    std::vector<llama_token> expected_support;
    for (size_t i = 0; i < expected.size; ++i) {
        expected_support.push_back(expected.data[i].id);
    }
    std::sort(support.begin(), support.end());
    std::sort(expected_support.begin(), expected_support.end());
    require(support == expected_support, "backend changed the complete retained vocabulary support");
    std::sort(ids.begin(), ids.end());
    std::sort(input_ids.begin(), input_ids.end());
    require(ids == input_ids, "backend changed the complete candidate ID mapping");
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
        for (uint32_t seed = 0; seed < (scenario.find("dist") != std::string::npos ? 32u : 1u); ++seed) {
            test_retention(scenario, seed);
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "backend probability retention: %s\n", error.what());
        return 1;
    }
}
