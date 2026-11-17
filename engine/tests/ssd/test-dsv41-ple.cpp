// SPDX-License-Identifier: Apache-2.0
#include "models/deepseek41-ple.h"

#include <cstdio>
#include <functional>
#include <string>

namespace {
struct metadata {
    uint32_t n_layer = 40, n_vocab = 5, ngram = 4, heads = 2, key_len = 256, pad = 2;
    std::vector<uint32_t> layers{1, 14};
    std::vector<uint64_t> multipliers{3, 5, 7, 11, 13, 17, 19, 23};
    std::vector<uint64_t> offsets{0, 2, 5, 10, 17, 28, 41, 58, 77, 100, 129, 160};
    std::vector<uint64_t> primes{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    std::vector<int32_t> token_map{4, -1, 0, 3, 1};

    uint64_t validate() const {
        return llama_dsv41_validate_ple(n_layer, n_vocab, layers, ngram, heads, key_len,
                multipliers, offsets, primes, token_map, pad);
    }
};

void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

void rejects(const std::function<void(metadata &)> & mutate, const char * diagnostic) {
    metadata m;
    mutate(m);
    try { m.validate(); }
    catch (const std::runtime_error & error) {
        require(std::string(error.what()).find(diagnostic) != std::string::npos,
                "wrong PLE rejection diagnostic");
        return;
    }
    throw std::runtime_error("malformed PLE metadata was admitted");
}
} // namespace

int main() {
    require(metadata{}.validate() == 197, "incorrect shared row extent");
    rejects([](metadata & m) { m.layers.clear(); }, "layer count");
    rejects([](metadata & m) { m.n_layer = 1; }, "layer count");
    rejects([](metadata & m) { m.layers[1] = 40; }, "layer ID");
    rejects([](metadata & m) { m.layers[1] = 1; }, "unique");
    rejects([](metadata & m) { m.ngram = 1; }, "dimensions");
    rejects([](metadata & m) { m.ngram = 0; }, "dimensions");
    rejects([](metadata & m) { m.heads = 0; }, "dimensions");
    rejects([](metadata & m) { m.key_len = 0; }, "dimensions");
    rejects([](metadata & m) { m.ngram = m.heads = m.key_len = UINT32_MAX; }, "overflow");
    rejects([](metadata & m) { m.multipliers.pop_back(); }, "multiplier count");
    rejects([](metadata & m) { m.offsets.pop_back(); }, "bucket counts");
    rejects([](metadata & m) { m.primes.push_back(41); }, "bucket counts");
    rejects([](metadata & m) { m.primes[0] = 0; }, "positive");
    rejects([](metadata & m) { m.primes[0] = UINT64_MAX; }, "int32");
    rejects([](metadata & m) { m.primes.back() = INT32_MAX; }, "int32");
    rejects([](metadata & m) { m.offsets[0] = UINT64_MAX; }, "int32");
    rejects([](metadata & m) { m.offsets[0] = uint64_t(INT32_MAX) + 1; }, "int32");
    rejects([](metadata & m) { m.offsets[0] = uint64_t(INT32_MAX); }, "int32");
    rejects([](metadata & m) { m.offsets[0] = UINT64_MAX - 1; m.primes[0] = 3; }, "int32");
    rejects([](metadata & m) { m.token_map.pop_back(); }, "vocabulary");
    rejects([](metadata & m) { m.n_vocab = 0; }, "vocabulary");
    rejects([](metadata & m) { m.pad = 5; }, "padding token");
    rejects([](metadata & m) { m.pad = 1; }, "padding token");
    rejects([](metadata & m) { m.token_map[0] = -2; }, "negative ID");

    metadata boundary;
    boundary.primes.back() = uint64_t(INT32_MAX) - boundary.offsets.back();
    require(boundary.validate() == INT32_MAX, "int32 row-count boundary was rejected");
    boundary = metadata{};
    boundary.offsets[0] = uint64_t(INT32_MAX) - boundary.primes[0];
    require(boundary.validate() == INT32_MAX, "maximum bucket extent must include earlier buckets");

    boundary = metadata{};
    boundary.offsets[0] = uint64_t(INT32_MAX);
    boundary.primes[0] = 1;
    require(boundary.validate() == uint64_t(INT32_MAX) + 1,
            "a one-row bucket at the largest I32 index was rejected");

    metadata padded;
    for (uint64_t & offset : padded.offsets) { offset += 11; }
    padded.offsets.back() += 23;
    require(padded.validate() == 231, "nonzero start and gaps must preserve shared row extent");
    metadata overlapping;
    overlapping.offsets[6] = overlapping.offsets[5];
    require(overlapping.validate() == 197, "overlapping bucket ranges were rejected");
    // Layer order is explicit metadata, and -1 is a supported dead-token sentinel.
    metadata reordered;
    reordered.layers = {14, 1};
    require(reordered.validate() == 197, "explicit layer order was rejected");
    std::puts("DeepSeek PLE metadata validation passed");
}
