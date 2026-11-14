// SPDX-License-Identifier: Apache-2.0
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

using sampler_ptr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;

static std::vector<llama_token_data> power_law(float slope, size_t count = 8) {
    std::vector<llama_token_data> result;
    for (size_t i = 0; i < count; ++i) {
        result.push_back({llama_token(count - 1 - i), -slope * std::log(float(i + 1)), 0.0f});
    }
    return result;
}

static std::vector<llama_token_data> expected_candidates(std::vector<llama_token_data> values, size_t keep) {
    std::stable_sort(values.begin(), values.end(), [](const auto & a, const auto & b) {
        return a.logit > b.logit;
    });
    values.resize(keep);
    return values;
}

static void compare(const llama_token_data_array & actual, const llama_token_data_array & expected) {
    require(actual.size == expected.size, "Mirostat retained the wrong number of candidates");
    require(actual.selected >= 0 && size_t(actual.selected) < actual.size, "selected index is outside candidates");
    double total = 0.0;
    for (size_t i = 0; i < actual.size; ++i) {
        require(actual.data[i].id == expected.data[i].id, "candidate vocabulary ID changed");
        require(actual.data[i].logit == expected.data[i].logit, "candidate logit changed");
        require(std::isfinite(actual.data[i].p), "candidate probability is not finite");
        require(std::abs(actual.data[i].p - expected.data[i].p) < 2e-6f, "candidate probability differs");
        total += actual.data[i].p;
    }
    require(std::abs(total - 1.0) < 2e-6, "probabilities are not normalized");
    require(actual.data[actual.selected].id == expected.data[expected.selected].id, "seeded token differs");
}

static void check_once(
        const std::vector<llama_token_data> & input, int32_t n_vocab, float tau, int32_t m, size_t keep) {
    for (uint32_t seed = 0; seed < 32; ++seed) {
        auto values = input;
        auto expected = expected_candidates(input, keep);
        llama_token_data_array actual = {values.data(), values.size(), -1, false};
        llama_token_data_array oracle = {expected.data(), expected.size(), -1, true};
        sampler_ptr sampler(llama_sampler_init_mirostat(n_vocab, seed, tau, 0.1f, m), llama_sampler_free);
        sampler_ptr dist(llama_sampler_init_dist(seed), llama_sampler_free);
        llama_sampler_apply(dist.get(), &oracle);
        llama_sampler_apply(sampler.get(), &actual);
        if (std::all_of(input.begin(), input.end(), [&](const auto & token) { return token.logit == input[0].logit; })) {
            // The existing sort is free to choose any token when all logits tie.
            require(actual.size == 1 && actual.selected == 0 && actual.data[0].p == 1.0f,
                    "flat distribution did not retain its single-token fallback");
            require(std::any_of(input.begin(), input.end(), [&](const auto & token) {
                return token.id == actual.data[0].id;
            }), "flat fallback selected an unknown token");
        } else {
            compare(actual, oracle);
        }
    }
}

static void check_feedback_and_reset() {
    // For a Zipf exponent of 2, the paper's cutoff is sqrt(2^mu / (1 - 1/N)).
    // Use that closed form, not the production estimator, for each feedback step.
    const auto input = power_law(2.0f);
    constexpr float tau = 2.1f;
    constexpr float eta = 0.1f;
    for (uint32_t seed = 0; seed < 8; ++seed) {
        sampler_ptr sampler(llama_sampler_init_mirostat(8, seed, tau, eta, 100), llama_sampler_free);
        sampler_ptr dist(llama_sampler_init_dist(seed), llama_sampler_free);
        for (int run = 0; run < 2; ++run) {
            float mu = 2.0f * tau;
            for (int step = 0; step < 24; ++step) {
                const double cutoff = std::sqrt(std::exp2(double(mu)) / (1.0 - 1.0 / 8.0));
                const size_t keep = size_t(std::clamp(cutoff, 1.0, 8.0));
                auto values = input;
                auto expected = expected_candidates(input, keep);
                llama_token_data_array actual = {values.data(), values.size(), -1, false};
                llama_token_data_array oracle = {expected.data(), expected.size(), -1, true};
                llama_sampler_apply(dist.get(), &oracle);
                llama_sampler_apply(sampler.get(), &actual);
                compare(actual, oracle);
                mu -= eta * (-std::log2(oracle.data[oracle.selected].p) - tau);
                llama_sampler_accept(sampler.get(), actual.data[actual.selected].id);
            }
            llama_sampler_reset(sampler.get());
            llama_sampler_reset(dist.get());
        }
    }
}

int main(int argc, char ** argv) {
    require(argc == 2, "expected one scenario");
    const std::string scenario = argv[1];
    auto input = power_law(2.0f);
    int32_t n_vocab = 8;
    float tau = 2.0f;
    int32_t m = 100;
    size_t keep = 4;

    if (scenario == "unit-slope" || scenario == "unit-slope-all") {
        input = power_law(1.0f);
        tau = 1.5f;
        m = scenario == "unit-slope" ? 2 : 100;
        keep = 3; // lim(s -> 1) k = 2^mu / log(N) = 8 / log(8).
    } else if (scenario == "near-unit-lower" || scenario == "near-unit-upper") {
        input = power_law(scenario == "near-unit-lower" ? 0.999999f : 1.000001f);
        tau = 1.5f;
        m = 2;
        keep = 3;
    } else if (scenario == "masked-tail" || scenario == "masked-shuffled") {
        input = power_law(1.0f);
        for (size_t i = 4; i < input.size(); ++i) {
            input[i].logit = -std::numeric_limits<float>::infinity();
        }
        if (scenario == "masked-shuffled") {
            std::reverse(input.begin(), input.end());
        }
        tau = 1.5f;
        keep = 3;
    } else if (scenario == "high-target") {
        tau = 100.0f;
        keep = 8; // sqrt(2^200 / (1 - 1/8)) exceeds the available candidates.
    } else if (scenario == "underflow-high-target") {
        input = {{2, 0.0f, 0.0f}, {0, -800.0f, 0.0f}, {1, -801.0f, 0.0f}};
        n_vocab = 3;
        tau = 1000.0f;
        keep = 3; // s ~= 860; the finite log-domain cutoff is greater than 3.
    } else if (scenario == "underflow") {
        input = {{2, 0.0f, 0.0f}, {0, -800.0f, 0.0f}, {1, -801.0f, 0.0f}};
        n_vocab = 3;
        tau = 5.0f;
        keep = 1;
    } else if (scenario == "single") {
        input = {{0, 2.0f, 0.0f}};
        n_vocab = 1;
        keep = 1;
    } else if (scenario == "one-unmasked") {
        input = {{0, -INFINITY, 0.0f}, {2, 2.0f, 0.0f}, {1, -INFINITY, 0.0f}};
        n_vocab = 3;
        keep = 1;
    } else if (scenario == "zero-temperature") {
        llama_token_data_array values = {input.data(), input.size(), -1, false};
        sampler_ptr temperature(llama_sampler_init_temp(0.0f), llama_sampler_free);
        llama_sampler_apply(temperature.get(), &values);
        keep = 1;
    } else if (scenario == "m-one") {
        m = 1;
        keep = 1;
    } else if (scenario == "flat") {
        input = power_law(0.0f);
        keep = 1; // Preserve the existing greedy fallback when no slope can be estimated.
    } else if (scenario == "partial-vocabulary") {
        n_vocab = 32;
    } else if (scenario == "shifted") {
        for (auto & token : input) {
            token.logit += 1000.0f;
        }
    } else if (scenario == "feedback-reset") {
        check_feedback_and_reset();
        return 0;
    } else {
        require(scenario == "ordinary", "unknown scenario");
    }
    check_once(input, n_vocab, tau, m, keep);
    return 0;
}
