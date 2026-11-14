// SPDX-License-Identifier: Apache-2.0
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

static void add_filter(llama_sampler * chain, const std::string & scenario) {
    if (scenario.find("top-k") == 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(2));
    } else if (scenario == "top-p") {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.9f, 0));
    } else if (scenario == "min-p") {
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(0.2f, 0));
    } else if (scenario == "temperature") {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(0.0f));
    }
}

static void test_backend(const std::string & scenario, uint32_t seed) {
    const bool mapped = scenario == "mapped" || scenario == "identity-mapped" || scenario == "single";
    const bool dist = scenario.find("dist") != std::string::npos;
    std::vector<float> logits = {-2.0f, 0.3f, -1.0f, 0.0f, 4.0f, -3.0f, -4.0f, 2.0f};
    std::vector<llama_token> input_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<llama_logit_bias> biases = {{0, 20.0f}, {4, -6.0f}, {7, 1.5f}};
    if (scenario == "mapped") {
        logits.resize(5);
        input_ids = {7, 2, 5, 1, 6};
    } else if (scenario == "single") {
        logits = {0.2f};
        input_ids = {7};
    } else if (scenario == "prefix") {
        logits.resize(3);
        input_ids.resize(3);
    } else if (scenario == "empty") {
        biases.clear();
    }

    owner<llama_sampler, llama_sampler_free> filter(
            llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    owner<llama_sampler, llama_sampler_free> device(
            llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    add_filter(filter.get(), scenario);
    add_filter(device.get(), scenario);
    llama_sampler_chain_add(device.get(), llama_sampler_init_logit_bias(8, biases.size(), biases.data()));
    if (dist) {
        llama_sampler_chain_add(device.get(), llama_sampler_init_dist(seed));
    }

    std::vector<llama_token_data> reference;
    for (size_t i = 0; i < logits.size(); ++i) {
        reference.push_back({input_ids[i], logits[i], 0.0f});
    }
    llama_token_data_array expected = {reference.data(), reference.size(), -1, false};
    llama_sampler_apply(filter.get(), &expected);
    // Compute the expected bias by vocabulary ID, independently of either
    // sampler implementation and of the filter's candidate order.
    for (size_t i = 0; i < expected.size; ++i) {
        for (const auto & bias : biases) {
            if (expected.data[i].id == bias.token) {
                expected.data[i].logit += bias.bias;
            }
        }
    }

    owner<ggml_backend, ggml_backend_free> backend(ggml_backend_cpu_init(), ggml_backend_free);
    require(bool(backend), "CPU backend initialization failed");
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    require(device->iface->backend_init(device.get(), ggml_backend_cpu_buffer_type()), "sampler backend initialization failed");
    owner<ggml_context, ggml_free> ctx(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
    require(bool(ctx), "graph context allocation failed");
    auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, logits.size());
    ggml_set_input(input);
    auto * candidates = mapped ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, input_ids.size()) : nullptr;
    llama_sampler_data data = {input, nullptr, nullptr, candidates};
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
    if (candidates) {
        ggml_backend_tensor_set(candidates, input_ids.data(), 0, input_ids.size() * sizeof(llama_token));
    }
    device->iface->backend_set_input(device.get());
    require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "graph execution failed");

    const size_t count = ggml_nelements(data.logits);
    std::vector<float> actual(count);
    std::vector<llama_token> ids(input_ids.begin(), input_ids.begin() + count);
    ggml_backend_tensor_get(data.logits, actual.data(), 0, count * sizeof(float));
    if (data.candidates) {
        require(size_t(ggml_nelements(data.candidates)) == count, "candidate mapping width changed");
        ggml_backend_tensor_get(data.candidates, ids.data(), 0, count * sizeof(llama_token));
    }
    std::vector<llama_token_data> ordered;
    std::vector<llama_token> support, expected_support;
    for (size_t i = 0; i < count; ++i) {
        const auto * match = std::find_if(expected.data, expected.data + expected.size,
                                         [&](const llama_token_data & value) { return value.id == ids[i]; });
        const float logit = match == expected.data + expected.size ? -INFINITY : match->logit;
        require(actual[i] == logit, "logit bias was applied to the wrong candidate");
        ordered.push_back({ids[i], logit, 0.0f});
        if (std::isfinite(logit)) {
            support.push_back(ids[i]);
        }
    }
    for (size_t i = 0; i < expected.size; ++i) {
        if (std::isfinite(expected.data[i].logit)) {
            expected_support.push_back(expected.data[i].id);
        }
    }
    std::sort(support.begin(), support.end());
    std::sort(expected_support.begin(), expected_support.end());
    require(support == expected_support, "bias changed retained vocabulary support");
    if (dist) {
        owner<llama_sampler, llama_sampler_free> oracle(llama_sampler_init_dist(seed), llama_sampler_free);
        llama_token_data_array ordered_expected = {ordered.data(), ordered.size(), -1, false};
        llama_sampler_apply(oracle.get(), &ordered_expected);
        std::vector<float> probabilities(count);
        llama_token selected = -1;
        ggml_backend_tensor_get(data.probs, probabilities.data(), 0, count * sizeof(float));
        ggml_backend_tensor_get(data.sampled, &selected, 0, sizeof(selected));
        require(selected == ordered[ordered_expected.selected].id, "biased distribution selected the wrong token");
        for (size_t i = 0; i < count; ++i) {
            require(std::abs(probabilities[i] - ordered[i].p) < 1e-6f, "biased distribution probabilities differ");
        }
    }
}

static void test_cpu_order(const std::string & scenario) {
    const bool dist = scenario.find("dist") != std::string::npos;
    const bool direct = scenario.find("direct") != std::string::npos;
    const bool absent = scenario == "cpu-absent";
    std::vector<llama_token_data> actual = {{0, -2.0f, 0.0f}, {1, 0.0f, 0.0f}, {2, 3.0f, 0.0f}, {3, 1.0f, 0.0f}};
    if (direct) {
        actual = {{0, 3.0f, 0.0f}, {1, 1.0f, 0.0f}, {2, 0.0f, 0.0f}, {3, -2.0f, 0.0f}};
    }
    llama_token_data_array result = {actual.data(), actual.size(), -1, false};
    owner<llama_sampler, llama_sampler_free> sorting(llama_sampler_init_top_k(actual.size()), llama_sampler_free);
    llama_sampler_apply(sorting.get(), &result);
    std::vector<llama_token_data> reference(result.data, result.data + result.size);
    const llama_logit_bias entry = {absent ? 7 : direct ? 3 : 0, dist ? 1000.0f : 20.0f};
    for (auto & token : reference) {
        if (token.id == entry.token) {
            token.logit += entry.bias;
        }
    }
    owner<llama_sampler, llama_sampler_free> bias(llama_sampler_init_logit_bias(8, 1, &entry), llama_sampler_free);
    llama_sampler_apply(bias.get(), &result);
    llama_token_data_array expected = {reference.data(), reference.size(), -1, false};
    owner<llama_sampler, llama_sampler_free> after(dist ? llama_sampler_init_dist(17) : llama_sampler_init_top_p(0.8f, 0), llama_sampler_free);
    owner<llama_sampler, llama_sampler_free> oracle(dist ? llama_sampler_init_dist(17) : llama_sampler_init_top_p(0.8f, 0), llama_sampler_free);
    llama_sampler_apply(after.get(), &result);
    llama_sampler_apply(oracle.get(), &expected);
    require(result.size == expected.size, "top-p used the pre-bias sorting order");
    for (size_t i = 0; i < result.size; ++i) {
        require(result.data[i].id == expected.data[i].id, "bias left candidates in the wrong order");
        require(std::isfinite(result.data[i].p) && std::abs(result.data[i].p - expected.data[i].p) < 1e-6f,
                "bias left an invalid softmax maximum");
    }
    if (dist) {
        require(result.data[result.selected].id == expected.data[expected.selected].id, "CPU biased sample differs");
    }
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "expected one test scenario");
        const std::string scenario = argv[1];
        if (scenario.find("cpu-") == 0) {
            test_cpu_order(scenario);
        } else {
            for (uint32_t seed = 0; seed < (scenario.find("dist") != std::string::npos ? 32u : 1u); ++seed) {
                test_backend(scenario, seed);
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "logit bias candidates: %s\n", error.what());
        return 1;
    }
}
