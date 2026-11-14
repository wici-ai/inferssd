// SPDX-License-Identifier: Apache-2.0
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using sampler_ptr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static std::vector<llama_token> ids(const llama_token_data_array & data) {
    std::vector<llama_token> result;
    for (size_t i = 0; i < data.size; ++i) {
        result.push_back(data.data[i].id);
    }
    std::sort(result.begin(), result.end());
    return result;
}

static void test_typical(const std::string & scenario) {
    const bool chain = scenario == "chain" || scenario == "underflow-chain";
    const bool bias = scenario == "bias" || scenario == "chain";
    for (uint32_t seed = 0; seed < (chain ? 32u : 1u); ++seed) {
        // Entropy is closer to -log(.2) than -log(.4). With mass .5,
        // typical sampling therefore retains tokens 1, 2, 3, not token 0.
        std::vector<llama_token_data> candidates = {
            {0, std::log(0.4f), 0.0f}, {1, std::log(0.2f), 0.0f},
            {2, std::log(0.2f), 0.0f}, {3, std::log(0.2f), 0.0f},
        };
        float masked_logit = -std::numeric_limits<float>::infinity();
        if (scenario == "underflow" || scenario == "underflow-chain") {
            masked_logit = -1000.0f;
        } else if (scenario == "finite-min") {
            masked_logit = std::numeric_limits<float>::lowest();
        }
        if (scenario != "unmasked") {
            candidates.push_back({4, bias ? 0.0f : masked_logit, 0.0f});
        }
        if (scenario == "many-masked" || scenario == "scrambled") {
            for (llama_token token = 5; token < 68; ++token) {
                candidates.push_back({token, masked_logit, 0.0f});
            }
        }
        if (scenario == "scrambled") {
            std::reverse(candidates.begin(), candidates.end());
        } else if (scenario == "shifted") {
            for (auto & candidate : candidates) {
                candidate.logit += 1000.0f;
            }
        } else if (scenario == "allowed-only") {
            for (auto & candidate : candidates) {
                candidate.logit = candidate.id == 2 ? 0.0f : masked_logit;
            }
        }

        sampler_ptr sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
        if (bias) {
            const llama_logit_bias blocked = {4, -std::numeric_limits<float>::infinity()};
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_logit_bias(5, 1, &blocked));
        }
        const bool disabled = scenario == "disabled";
        const size_t min_keep = scenario == "minimum-kept" ? 4 : 1;
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_typical(disabled ? 1.0f : 0.5f, min_keep));
        if (chain) {
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(seed));
        }
        llama_token_data_array data = {candidates.data(), candidates.size(), -1, false};
        llama_sampler_apply(sampler.get(), &data);

        std::vector<llama_token> expected = {1, 2, 3};
        if (disabled) {
            expected = {0, 1, 2, 3, 4};
        } else if (scenario == "minimum-kept") {
            expected = {0, 1, 2, 3};
        } else if (scenario == "allowed-only") {
            expected = {2};
        }
        require(ids(data) == expected, "zero-probability candidates changed the retained typical support");
        if (!disabled) {
            for (size_t i = 0; i < data.size; ++i) {
                require(std::isfinite(data.data[i].p) && data.data[i].p > 0.0f,
                        "retained a masked or non-finite probability");
            }
        }
        if (chain) {
            require(data.selected >= 0 && size_t(data.selected) < data.size, "invalid sampled index");
            require(data.data[data.selected].id != 0 && data.data[data.selected].id != 4,
                    "sampled a token outside the typical support");
            for (size_t i = 0; i < data.size; ++i) {
                require(std::abs(data.data[i].p - 1.0f / 3.0f) < 1e-6f,
                        "final sampling probabilities are not normalized over typical support");
            }
        }
    }
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "expected typical sampling scenario");
        const std::string scenario = argv[1];
        const std::vector<std::string> scenarios = {
            "masked", "many-masked", "underflow", "finite-min", "scrambled", "shifted",
            "minimum-kept", "allowed-only", "unmasked", "disabled", "bias", "chain", "underflow-chain",
        };
        require(std::find(scenarios.begin(), scenarios.end(), scenario) != scenarios.end(), "unknown scenario");
        test_typical(scenario);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
