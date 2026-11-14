// SPDX-License-Identifier: Apache-2.0
#include "ngram-cache.h"
#include "speculative.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

static void require(bool condition, const char * message) {
    if (!condition) { fprintf(stderr, "%s\n", message); std::exit(1); }
}

struct fixture {
    common_params_speculative params;
    common_speculative_ptr state;

    fixture() {
        params.type = COMMON_SPECULATIVE_TYPE_NGRAM_CACHE;
        state.reset(common_speculative_init(params, nullptr));
        require(bool(state), "could not create the model-free n-gram drafter");
    }

    void begin(const llama_tokens & prompt) { common_speculative_begin(state.get(), prompt); }

    llama_tokens draft(const llama_tokens & history) {
        require(!history.empty(), "empty fixture history");
        return common_speculative_draft(state.get(), params,
                llama_tokens(history.begin(), history.end() - 1), history.back());
    }

    size_t compare(const llama_tokens & history, bool populated = false) {
        // Rebuild the empirical distribution independently from the complete
        // history, rather than replaying the stateful implementation's deltas.
        llama_tokens input = history;
        common_ngram_cache context, dynamic, fixed;
        common_ngram_cache_update(context, 1, 4, input, input.size(), false);
        llama_tokens expected{history.back()};
        common_ngram_cache_draft(input, expected, common_speculative_n_max(state.get(), params),
                1, 4, context, dynamic, fixed);
        expected.erase(expected.begin());
        if (populated) { require(!expected.empty(), "fixture must have a learned continuation"); }
        const auto actual = draft(history);
        require(actual == expected, "stateful drafts disagree with the full-history counts");
        return actual.size();
    }
};

static llama_tokens repeated(llama_token second, size_t repetitions) {
    llama_tokens result;
    for (size_t i = 0; i < repetitions; ++i) { result.insert(result.end(), {1, second}); }
    return result;
}

static void incremental(size_t step) {
    fixture f;
    llama_tokens tokens{1, 2};
    f.begin(tokens);
    f.compare(tokens);
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < step; ++j) { tokens.push_back(tokens.size() % 2 + 1); }
        const size_t drafted = f.compare(tokens, tokens.size() >= 6);
        if (drafted) { common_speculative_accept(f.state.get(), std::min(step - 1, drafted)); }
    }
}

static void next_generation(size_t new_size) {
    fixture f;
    const auto first = repeated(2, 16);
    f.begin(first);
    f.compare(first, true);
    const auto second = repeated(3, new_size);
    f.begin(second);
    f.compare(second, true);
}

static void shifted() {
    fixture f;
    const auto first = repeated(2, 16);
    f.begin(first);
    f.compare(first, true);
    const auto second = repeated(3, 6);
    f.compare(second, true);
}

int main(int argc, char ** argv) {
    require(argc == 2, "expected a scenario name");
    const std::string scenario = argv[1];
    if (scenario == "single") { incremental(1); }
    else if (scenario == "pairs") { incremental(2); }
    else if (scenario == "batch") { incremental(5); }
    else if (scenario == "same-size") { next_generation(16); }
    else if (scenario == "shorter") { next_generation(6); }
    else if (scenario == "longer") { next_generation(22); }
    else if (scenario == "shifted") { shifted(); }
    else if (scenario == "initial") {
        fixture f; const auto tokens = repeated(2, 10); f.begin(tokens); f.compare(tokens, true);
    } else if (scenario == "unchanged") {
        fixture f; const auto tokens = repeated(2, 10); f.begin(tokens);
        for (size_t i = 0; i < 5; ++i) { f.compare(tokens, true); }
    } else if (scenario == "unseen") {
        fixture f; f.begin({1, 2, 3}); f.compare({1, 2, 3, 4});
    } else { require(false, "unknown scenario"); }
}
