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

static void test_temperature(const std::string & scenario) {
    const bool top_k = scenario == "top-k" || scenario == "top-k-dist";
    const bool dist = scenario == "dist" || scenario == "top-k-dist";
    const bool multi = scenario == "multiple";
    const float delta = scenario == "disabled" ? 0.0f : 0.5f;
    const float exponent = scenario == "zero-exponent" ? 0.0f : 1.5f;
    const float temp = 0.8f;
    std::vector<float> logits = top_k || multi ? std::vector<float>{-2.0f, 0.7f, -0.1f, 0.2f}
                                             : std::vector<float>{scenario == "zero" ? 0.0f : -2.0f};

    owner<llama_sampler, llama_sampler_free> cpu(
            llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    owner<llama_sampler, llama_sampler_free> device(
            llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    for (auto * sampler : {cpu.get(), device.get()}) {
        if (top_k) {
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(1));
        }
        llama_sampler_chain_add(sampler, llama_sampler_init_temp_ext(temp, delta, exponent));
        if (dist) {
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(123));
        }
    }

    std::vector<llama_token_data> reference;
    for (size_t i = 0; i < logits.size(); ++i) {
        reference.push_back({llama_token(i), logits[i], 0.0f});
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
    llama_sampler_data data = {input, nullptr, nullptr, nullptr};
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
    device->iface->backend_set_input(device.get());
    require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "graph execution failed");

    const size_t count = ggml_nelements(data.logits);
    require(count == expected.size, "backend changed the number of candidates");
    std::vector<float> actual(count);
    std::vector<llama_token> ids(count);
    ggml_backend_tensor_get(data.logits, actual.data(), 0, actual.size() * sizeof(float));
    if (data.candidates) {
        ggml_backend_tensor_get(data.candidates, ids.data(), 0, ids.size() * sizeof(llama_token));
    } else {
        for (size_t i = 0; i < count; ++i) {
            ids[i] = llama_token(i);
        }
    }
    for (size_t i = 0; i < count; ++i) {
        auto * match = std::find_if(expected.data, expected.data + expected.size,
                                   [&](const llama_token_data & token) { return token.id == ids[i]; });
        require(match != expected.data + expected.size, "backend changed the candidate identity");
        require(std::isfinite(actual[i]), "dynamic temperature produced a non-finite logit");
        require(std::abs(actual[i] - match->logit) < 1e-5f, "backend temperature differs from CPU sampling");
    }
    if (dist) {
        float probability = 0.0f;
        llama_token selected = -1;
        ggml_backend_tensor_get(data.probs, &probability, 0, sizeof(probability));
        ggml_backend_tensor_get(data.sampled, &selected, 0, sizeof(selected));
        require(probability == 1.0f, "the only candidate did not receive probability one");
        require(selected == expected.data[expected.selected].id, "dist did not select the remaining token");
    }
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "expected one test scenario");
        test_temperature(argv[1]);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "single-candidate temperature: %s\n", error.what());
        return 1;
    }
}
