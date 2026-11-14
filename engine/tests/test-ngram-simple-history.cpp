// SPDX-License-Identifier: Apache-2.0
#include "speculative.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

static void require(bool condition, const char * message) {
    if (!condition) { fprintf(stderr, "%s\n", message); std::exit(1); }
}

struct fixture {
    common_params_speculative params;
    common_speculative_ptr state;

    fixture(uint16_t n, uint16_t m) {
        params.type = COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE;
        params.ngram_simple.size_n = n;
        params.ngram_simple.size_m = m;
        state.reset(common_speculative_init(params, nullptr));
        require(bool(state), "could not create the model-free simple n-gram drafter");
    }

    void check(const llama_tokens & history, llama_token sampled, const llama_tokens & expected) {
        common_speculative_begin(state.get(), history);
        const auto actual = common_speculative_draft(state.get(), params, history, sampled);
        if (actual != expected) {
            fprintf(stderr, "history=%zu, N=%u, M=%u: expected %zu draft tokens, got %zu\n",
                    history.size(), params.ngram_simple.size_n, params.ngram_simple.size_m,
                    expected.size(), actual.size());
            std::exit(1);
        }
        if (!actual.empty()) { common_speculative_accept(state.get(), actual.size()); }
    }
};

static llama_tokens reference(const llama_tokens & history, llama_token sampled, size_t n, size_t m) {
    llama_tokens complete = history;
    complete.push_back(sampled);
    if (complete.size() < n) { return {}; }
    const llama_tokens pattern(complete.end() - n, complete.end());
    llama_tokens result;
    // Enumerate forward and retain the latest complete candidate, independently
    // of the production reverse search and its index arithmetic.
    for (size_t start = 0; start < history.size(); ++start) {
        if (history.size() - start < n + m) { continue; }
        const auto begin = history.begin() + start;
        if (std::equal(pattern.begin(), pattern.end(), begin)) {
            result.assign(begin + n, begin + n + m);
        }
    }
    return result;
}

int main(int argc, char ** argv) {
    require(argc == 2, "expected a scenario name");
    const std::string scenario = argv[1];
    if (scenario == "prefix") {
        fixture(3, 3).check({1, 2, 3, 9, 8, 7, 100, 101, 102, 1, 2}, 3, {9, 8, 7});
    } else if (scenario == "exact-history") {
        fixture(3, 3).check({1, 2, 3, 9, 1, 2}, 3, {9, 1, 2});
    } else if (scenario == "short-draft") {
        fixture(3, 2).check({1, 2, 3, 1, 2}, 3, {1, 2});
    } else if (scenario == "recent-short") {
        fixture(2, 3).check({4, 5, 7, 8, 9, 11, 12, 4, 5, 99, 4}, 5, {7, 8, 9});
    } else if (scenario == "unigram") {
        fixture(1, 1).check({7, 7}, 7, {7});
    } else if (scenario == "default-prefix") {
        llama_tokens history, expected;
        for (llama_token i = 1; i <= 12; ++i) { history.push_back(i); }
        for (llama_token i = 100; i < 148; ++i) { history.push_back(i); expected.push_back(i); }
        for (llama_token i = 200; i < 240; ++i) { history.push_back(i); }
        for (llama_token i = 1; i < 12; ++i) { history.push_back(i); }
        fixture(12, 48).check(history, 12, expected);
    } else if (scenario == "latest-match") {
        fixture(2, 3).check({100, 4, 5, 7, 8, 9, 44, 4, 5, 9, 8, 7, 66, 4}, 5, {9, 8, 7});
    } else if (scenario == "no-match") {
        fixture(2, 3).check({1, 2, 3, 4, 5, 6, 7, 8}, 9, {});
    } else if (scenario == "too-short") {
        fixture f(3, 3);
        for (size_t size = 0; size < 6; ++size) { f.check(llama_tokens(size, 1), 1, {}); }
    } else if (scenario == "sampled-token") {
        fixture(2, 3).check({100, 1, 2, 9, 8, 7, 44, 1}, 3, {});
    } else if (scenario == "reference") {
        std::mt19937 random(1234);
        for (uint16_t n = 1; n <= 4; ++n) {
            for (uint16_t m = 1; m <= 5; ++m) {
                fixture f(n, m);
                for (size_t trial = 0; trial < 100; ++trial) {
                    llama_tokens history(trial % 21);
                    for (auto & token : history) { token = random() % 3; }
                    const llama_token sampled = random() % 3;
                    f.check(history, sampled, reference(history, sampled, n, m));
                }
            }
        }
    } else { require(false, "unknown scenario"); }
}
