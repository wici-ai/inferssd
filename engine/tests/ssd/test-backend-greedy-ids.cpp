// SPDX-License-Identifier: Apache-2.0
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"

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

static void test_greedy(const std::string & scenario) {
    const bool mapped = scenario.find("mapped") != std::string::npos || scenario == "single" || scenario == "ties";
    const bool top_k = scenario.find("top-k") == 0 || scenario == "chained";
    const bool top_p = scenario.find("top-p") == 0 || scenario == "chained";
    std::vector<float> logits = {-2.0f, 0.3f, -1.0f, 0.0f, 4.0f};
    if (scenario == "single") {
        logits = {-0.2f};
    } else if (scenario == "ties") {
        logits = {4.0f, -1.0f, 0.0f, 4.0f};
    }

    owner<llama_sampler, llama_sampler_free> cpu(
            llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    owner<llama_sampler, llama_sampler_free> device(
            llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    for (auto * sampler : {cpu.get(), device.get()}) {
        if (top_k) {
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(2));
        }
        if (top_p) {
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.5f, 0));
        }
        if (scenario == "min-p") {
            llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.5f, 0));
        }
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    }

    std::vector<llama_token> ids;
    std::vector<llama_token_data> reference;
    for (size_t i = 0; i < logits.size(); ++i) {
        ids.push_back(mapped ? 42 + 3 * i : i);
        reference.push_back({ids.back(), logits[i], 0.0f});
    }
    llama_token_data_array expected = {reference.data(), reference.size(), -1, false};
    llama_sampler_apply(cpu.get(), &expected);
    llama_token expected_token = expected.data[expected.selected].id;

    owner<ggml_backend, ggml_backend_free> backend(ggml_backend_cpu_init(), ggml_backend_free);
    require(bool(backend), "CPU backend initialization failed");
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    require(device->iface->backend_init(device.get(), ggml_backend_cpu_buffer_type()),
            "CPU backend does not support the sampler chain");
    owner<ggml_context, ggml_free> ctx(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
    require(bool(ctx), "graph context allocation failed");
    auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, logits.size());
    ggml_set_input(input);
    // Preserve the backend's existing tie choice instead of imposing the CPU
    // sampler's first-maximum order on GGML's argmax implementation.
    auto * tie_choice = scenario == "ties" ? ggml_argmax(ctx.get(), input) : nullptr;
    auto * candidates = mapped ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, ids.size()) : nullptr;
    llama_sampler_data data = {input, nullptr, nullptr, candidates};
    auto * graph = ggml_new_graph(ctx.get());
    device->iface->backend_apply(device.get(), ctx.get(), graph, &data);
    for (auto * output : {data.logits, data.sampled, data.candidates, tie_choice}) {
        if (output) {
            ggml_build_forward_expand(graph, output);
        }
    }
    owner<ggml_backend_buffer, ggml_backend_buffer_free> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    require(bool(buffer), "graph buffer allocation failed");
    ggml_backend_tensor_set(input, logits.data(), 0, logits.size() * sizeof(float));
    if (candidates) {
        ggml_backend_tensor_set(candidates, ids.data(), 0, ids.size() * sizeof(llama_token));
    }
    device->iface->backend_set_input(device.get());
    require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "graph execution failed");
    if (tie_choice) {
        int32_t index = -1;
        ggml_backend_tensor_get(tie_choice, &index, 0, sizeof(index));
        require(index >= 0 && size_t(index) < ids.size(), "argmax returned an invalid candidate offset");
        expected_token = ids[index];
    }
    require(data.sampled && ggml_nelements(data.sampled) == 1, "greedy did not return one token");
    llama_token selected = -1;
    ggml_backend_tensor_get(data.sampled, &selected, 0, sizeof(selected));
    if (selected != expected_token) {
        std::fprintf(stderr, "expected vocabulary token %d, received %d\n", expected_token, selected);
        throw std::runtime_error("backend greedy returned a candidate offset instead of its vocabulary token");
    }
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "expected one test scenario");
        test_greedy(argv[1]);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "backend greedy token IDs: %s\n", error.what());
        return 1;
    }
}
