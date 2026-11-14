// SPDX-License-Identifier: Apache-2.0
#include "llama.h"

#include <cmath>
#include <cstdio>
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

static sampler_ptr make_sampler(const std::string & kind, uint32_t seed) {
    llama_sampler * sampler = nullptr;
    if (kind == "dist") {
        sampler = llama_sampler_init_dist(seed);
    } else if (kind == "mirostat") {
        sampler = llama_sampler_init_mirostat(64, seed, 3.0f, 0.25f, 20);
    } else if (kind == "mirostat-v2") {
        sampler = llama_sampler_init_mirostat_v2(seed, 3.0f, 0.25f);
    } else if (kind == "adaptive-p") {
        sampler = llama_sampler_init_adaptive_p(0.15f, 0.8f, seed);
    } else if (kind == "penalties-chain") {
        sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(sampler, llama_sampler_init_penalties(8, 1.1f, 0.25f, 0.4f));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
    } else if (kind == "xtc-chain") {
        sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(sampler, llama_sampler_init_xtc(0.5f, 0.04f, 1, seed));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
    } else {
        throw std::runtime_error("unknown sampler");
    }
    return sampler_ptr(sampler, llama_sampler_free);
}

struct sample_result {
    std::vector<llama_token_data> candidates;
    llama_token selected;
};

static sample_result sample(llama_sampler * sampler, int step, bool accept = true) {
    std::vector<llama_token_data> candidates;
    for (llama_token token = 0; token < 64; ++token) {
        const int rank = (token + step * 7) % 64;
        const float logit = -1.2f * std::log(float(rank + 1)) + 0.07f * float((token + step) % 3);
        candidates.push_back({token, logit, 0.0f});
    }
    llama_token_data_array data = {candidates.data(), candidates.size(), -1, false};
    llama_sampler_apply(sampler, &data);
    require(data.selected >= 0 && size_t(data.selected) < data.size, "invalid selected candidate");
    const llama_token selected = data.data[data.selected].id;
    sample_result result = {std::vector<llama_token_data>(data.data, data.data + data.size), selected};
    if (accept) {
        llama_sampler_accept(sampler, selected);
    }
    return result;
}

static void compare(const sample_result & left, const sample_result & right) {
    require(left.selected == right.selected, "cloned sampler selects a different token");
    require(left.candidates.size() == right.candidates.size(), "cloned sampler retains different candidates");
    for (size_t i = 0; i < left.candidates.size(); ++i) {
        require(left.candidates[i].id == right.candidates[i].id, "cloned sampler reorders candidates");
        require(left.candidates[i].logit == right.candidates[i].logit, "cloned sampler changes logits");
        require(left.candidates[i].p == right.candidates[i].p, "cloned sampler changes probabilities");
    }
}

static void advance(llama_sampler * sampler, int begin, int end) {
    for (int i = begin; i < end; ++i) {
        sample(sampler, i);
    }
}

static void continue_together(llama_sampler * left, llama_sampler * right, int begin) {
    for (int i = begin; i < begin + 40; ++i) {
        compare(sample(left, i), sample(right, i));
    }
}

static void test_clone(const std::string & kind, const std::string & scenario) {
    const bool random_seed = scenario == "seed";
    auto original = make_sampler(kind, random_seed ? LLAMA_DEFAULT_SEED : 42);
    const uint32_t actual_seed = llama_sampler_get_seed(original.get());
    const int warmup = scenario == "fresh" ? 0 : 23;
    advance(original.get(), 0, warmup);
    llama_token pending = LLAMA_TOKEN_NULL;
    if (scenario == "pending") {
        pending = sample(original.get(), warmup, false).selected;
    }
    sampler_ptr clone(llama_sampler_clone(original.get()), llama_sampler_free);
    require(clone != nullptr && clone.get() != original.get(), "clone must own a separate sampler");

    if (scenario == "seed") {
        require(llama_sampler_get_seed(clone.get()) == actual_seed, "clone reports a different effective seed");
        auto replay = make_sampler(kind, actual_seed);
        advance(replay.get(), 0, warmup);
        continue_together(clone.get(), replay.get(), warmup);
    } else if (scenario == "pending") {
        // Adaptive-p must retain the probabilities needed by this delayed accept.
        llama_sampler_accept(original.get(), pending);
        llama_sampler_accept(clone.get(), pending);
        continue_together(original.get(), clone.get(), warmup + 1);
    } else if (scenario == "independent") {
        auto replay = make_sampler(kind, 42);
        advance(replay.get(), 0, warmup);
        // Mutating and releasing the original cannot change the checkpoint.
        advance(original.get(), warmup, warmup + 11);
        llama_sampler_reset(original.get());
        original.reset();
        continue_together(clone.get(), replay.get(), warmup);
    } else if (scenario == "reset") {
        // Resetting the clone must not reset its original, and restores its fixed seed.
        llama_sampler_reset(clone.get());
        auto fresh = make_sampler(kind, 42);
        continue_together(clone.get(), fresh.get(), 0);
        auto replay = make_sampler(kind, 42);
        advance(replay.get(), 0, warmup);
        continue_together(original.get(), replay.get(), warmup);
    } else {
        require(scenario == "fresh" || scenario == "warmed", "unknown scenario");
        continue_together(original.get(), clone.get(), warmup);
    }
}

int main(int argc, char ** argv) {
    try {
        require(argc == 3, "expected sampler and clone scenario");
        test_clone(argv[1], argv[2]);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
