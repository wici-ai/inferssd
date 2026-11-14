// SPDX-License-Identifier: Apache-2.0
#include "ngram-cache.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

static void require(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

static void check_distribution(bool repeated) {
    common_ngram_cache cache;
    for (llama_token i = 1; i <= 2048; ++i) {
        const llama_token tokens[] = {i, i, i + 1, i + 1};
        const common_ngram key(tokens, repeated ? 2 : 4);
        cache[key][i + 2] = 1;
    }
    require(cache.size() == 2048, "distinct n-grams were lost");
    size_t largest = 0;
    for (size_t i = 0; i < cache.bucket_count(); ++i) {
        largest = std::max(largest, cache.bucket_size(i));
    }
    printf("keys=%zu buckets=%zu largest=%zu\n", cache.size(), cache.bucket_count(), largest);
    require(largest < 64, "repeated token pairs concentrate the cache into long linear chains");
    for (llama_token i = 1; i <= 2048; ++i) {
        const llama_token tokens[] = {i, i, i + 1, i + 1};
        require(cache.at(common_ngram(tokens, repeated ? 2 : 4)).at(i + 2) == 1,
                "an inserted n-gram no longer returns its own continuation");
    }
}

static void check_order() {
    std::array<llama_token, 4> tokens{{1, 2, 3, 4}};
    std::map<size_t, size_t> counts;
    do {
        ++counts[common_ngram_hash_function{}(common_ngram(tokens.data(), tokens.size()))];
    } while (std::next_permutation(tokens.begin(), tokens.end()));
    require(counts.size() >= 20, "token permutations still share the same hash");
}

static void check_update() {
    std::vector<llama_token> tokens{10, 20, 10, 20, 30, 10, 20, 10, 20, 30};
    common_ngram_cache actual;
    common_ngram_cache_update(actual, 1, 4, tokens, tokens.size(), false);
    std::map<std::array<llama_token, 4>, std::map<llama_token, int32_t>> expected;
    for (size_t end = 1; end < tokens.size(); ++end) {
        for (size_t size = 1; size <= std::min(size_t(4), end); ++size) {
            std::array<llama_token, 4> key{{-1, -1, -1, -1}};
            std::copy(tokens.begin() + end - size, tokens.begin() + end, key.begin());
            ++expected[key][tokens[end]];
        }
    }
    require(actual.size() == expected.size(), "cache update changed the n-gram set");
    for (const auto & entry : expected) {
        common_ngram key;
        std::copy(entry.first.begin(), entry.first.end(), key.tokens);
        const auto & counts = actual.at(key);
        require(counts.size() == entry.second.size(), "cache update changed continuations");
        for (const auto & continuation : entry.second) {
            require(counts.at(continuation.first) == continuation.second, "cache counts changed");
        }
    }
}

static void check_draft() {
    common_ngram_cache context, dynamic, fixed;
    std::vector<llama_token> tokens{10, 20, 30, 10, 20, 30, 10, 20};
    common_ngram_cache_update(context, 1, 4, tokens, tokens.size(), false);
    std::vector<llama_token> draft{20};
    common_ngram_cache_draft(tokens, draft, 1, 1, 4, context, dynamic, fixed);
    require(draft == std::vector<llama_token>({20, 30}), "draft continuation changed");
}

int main(int argc, char ** argv) {
    require(argc == 2, "expected a scenario name");
    const std::string scenario = argv[1];
    if (scenario == "repeated") { check_distribution(true); }
    else if (scenario == "paired") { check_distribution(false); }
    else if (scenario == "order") { check_order(); }
    else if (scenario == "update") { check_update(); }
    else if (scenario == "draft") { check_draft(); }
    else { require(false, "unknown scenario"); }
    return 0;
}
