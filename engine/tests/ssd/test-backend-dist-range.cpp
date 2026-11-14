// SPDX-License-Identifier: Apache-2.0
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T, void (*release)(T *)>
using owner = std::unique_ptr<T, decltype(release)>;

static void require(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

static void test_range(const std::string & scenario) {
    const bool mapped = scenario.find("mapped") != std::string::npos;
    const bool rounded = scenario.find("rounded") != std::string::npos || scenario == "single";
    const bool large = scenario.find("large") != std::string::npos || scenario == "tail" || scenario == "prefix";
    const uint32_t seed = rounded ? 2180022 : large ? 14104 : 42;
    std::vector<float> logits(large ? 32000 : scenario == "above" ? 10000 : 8, 0.0f);
    if (scenario == "single") {
        logits = {0.0f};
    } else if (scenario == "rounded-tail") {
        logits = {0.0f, -INFINITY, -INFINITY};
    } else if (scenario == "tail") {
        logits.insert(logits.end(), 12, -INFINITY);
    } else if (scenario == "prefix") {
        logits.insert(logits.begin(), 12, -INFINITY);
    } else if (scenario == "ordinary" || scenario == "ordinary-mapped") {
        logits = {-2.0f, 0.3f, -1.0f, 0.0f, 4.0f};
    }
    std::vector<llama_token> ids;
    for (size_t i = 0; i < logits.size(); ++i) {
        ids.push_back(mapped ? 17 + 3 * i : i);
    }

    owner<ggml_backend, ggml_backend_free> backend(ggml_backend_cpu_init(), ggml_backend_free);
    require(bool(backend), "CPU backend initialization failed");
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    owner<llama_sampler, llama_sampler_free> sampler(llama_sampler_init_dist(seed), llama_sampler_free);
    require(sampler->iface->backend_init(sampler.get(), ggml_backend_cpu_buffer_type()), "dist backend unsupported");
    owner<ggml_context, ggml_free> ctx(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
    require(bool(ctx), "graph context allocation failed");
    auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, logits.size());
    ggml_set_input(input);
    auto * candidates = mapped ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, ids.size()) : nullptr;
    llama_sampler_data data = {input, nullptr, nullptr, candidates};
    auto * graph = ggml_new_graph(ctx.get());
    sampler->iface->backend_apply(sampler.get(), ctx.get(), graph, &data);
    ggml_build_forward_expand(graph, data.sampled);
    owner<ggml_backend_buffer, ggml_backend_buffer_free> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    require(bool(buffer), "graph buffer allocation failed");
    ggml_backend_tensor_set(input, logits.data(), 0, logits.size() * sizeof(float));
    if (candidates) {
        ggml_backend_tensor_set(candidates, ids.data(), 0, ids.size() * sizeof(llama_token));
    }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const int draws = scenario == "ordinary" || scenario == "ordinary-mapped" ? 32 : 1;
    owner<llama_sampler, llama_sampler_free> cpu(llama_sampler_init_dist(seed), llama_sampler_free);
    for (int draw = 0; draw < draws; ++draw) {
        sampler->iface->backend_set_input(sampler.get());
        require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "graph execution failed");
        llama_token selected = -1;
        ggml_backend_tensor_get(data.sampled, &selected, 0, sizeof(selected));
        std::vector<float> probabilities(logits.size());
        ggml_backend_tensor_get(data.probs, probabilities.data(), 0, probabilities.size() * sizeof(float));
        std::vector<float> cdf;
        float sum = 0.0f;
        for (float p : probabilities) {
            require(std::isfinite(p) && p >= 0.0f, "invalid softmax probability");
            sum += p;
            cdf.push_back(sum);
        }
        require(sum > 0.0f && std::isfinite(sum), "invalid CDF total");
        const float random = std::min(static_cast<float>(uniform(rng)), std::nextafter(1.0f, 0.0f));
        const float threshold = random * sum;
        size_t expected = 0;
        while (expected < cdf.size() && cdf[expected] <= threshold) {
            ++expected;
        }
        require(expected < cdf.size(), "reference quantile exceeded the CDF range");
        if (selected != ids[expected]) {
            std::fprintf(stderr, "width=%zu CDF=%.9g random=%.9g expected=%d received=%d\n",
                    ids.size(), sum, random, ids[expected], selected);
            throw std::runtime_error("backend dist selected outside the expected CDF interval");
        }
        require(probabilities[expected] > 0.0f && std::isfinite(logits[expected]), "selected a masked token");
        if (draws > 1) {
            std::vector<llama_token_data> values;
            for (size_t i = 0; i < logits.size(); ++i) {
                values.push_back({ids[i], logits[i], 0.0f});
            }
            llama_token_data_array reference = {values.data(), values.size(), -1, false};
            llama_sampler_apply(cpu.get(), &reference);
            require(selected == reference.data[reference.selected].id, "ordinary draw differs from CPU sampling");
            for (size_t i = 0; i < values.size(); ++i) {
                require(std::abs(probabilities[i] - values[i].p) < 1e-6f, "softmax probabilities changed");
            }
        }
    }
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "expected a test scenario");
        test_range(argv[1]);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "backend dist range: %s\n", error.what());
        return 1;
    }
}
