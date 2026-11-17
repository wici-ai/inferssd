// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// Validate the shared-table PLE layout before constructing tensors or hashing
// token IDs. Offsets address the one table globally, not separate layer tables.
inline uint64_t llama_dsv41_validate_ple(
        uint32_t n_layer, uint32_t n_vocab, const std::vector<uint32_t> & layers,
        uint32_t ngram, uint32_t heads, uint32_t key_len,
        const std::vector<uint64_t> & multipliers,
        const std::vector<uint64_t> & offsets, const std::vector<uint64_t> & primes,
        const std::vector<int32_t> & token_map, uint32_t pad_token_id) {
    const auto require = [](bool condition, const char * message) {
        if (!condition) { throw std::runtime_error(message); }
    };
    require(!layers.empty() && layers.size() <= n_layer,
            "DeepSeek-V4.1 PLE layer count is invalid");
    for (size_t i = 0; i < layers.size(); ++i) {
        require(layers[i] < n_layer, "DeepSeek-V4.1 PLE layer ID is out of range");
        require(std::find(layers.begin(), layers.begin() + i, layers[i]) == layers.begin() + i,
                "DeepSeek-V4.1 PLE layer IDs must be unique");
    }
    require(ngram >= 2 && heads > 0 && key_len > 0,
            "DeepSeek-V4.1 PLE dimensions must have ngram >= 2 and positive heads/key length");
    const uint64_t buckets = uint64_t(ngram - 1) * heads;
    require(buckets <= uint64_t(std::numeric_limits<int64_t>::max()) / key_len,
            "DeepSeek-V4.1 PLE embedding dimensions overflow int64");
    require(layers.size() <= std::numeric_limits<size_t>::max() / buckets &&
            layers.size() <= std::numeric_limits<size_t>::max() / ngram,
            "DeepSeek-V4.1 PLE metadata counts overflow size_t");
    const size_t n_buckets = size_t(buckets) * layers.size();
    require(multipliers.size() == size_t(ngram) * layers.size(),
            "DeepSeek-V4.1 PLE multiplier count does not match layers * ngram");
    require(offsets.size() == n_buckets && primes.size() == n_buckets,
            "DeepSeek-V4.1 PLE bucket counts do not match layers * heads * (ngram - 1)");

    uint64_t rows = 0;
    for (size_t i = 0; i < n_buckets; ++i) {
        require(primes[i] >= 1, "DeepSeek-V4.1 PLE bucket size must be positive");
        // All hashes feed GGML I32 get_rows indices. Check before addition so
        // corrupt UINT64 metadata cannot wrap the sum and pass admission. The
        // shared table may contain gaps, padding or overlapping bucket ranges.
        require(offsets[i] <= uint64_t(INT32_MAX) &&
                primes[i] <= uint64_t(INT32_MAX) + 1 - offsets[i],
                "DeepSeek-V4.1 PLE shared row count exceeds int32 indices");
        rows = std::max(rows, offsets[i] + primes[i]);
    }
    require(token_map.size() >= n_vocab && n_vocab > 0,
            "DeepSeek-V4.1 PLE token map is shorter than the vocabulary");
    require(pad_token_id < token_map.size() && token_map[pad_token_id] >= 0,
            "DeepSeek-V4.1 PLE padding token must have a valid mapped ID");
    for (int32_t token : token_map) {
        require(token >= -1, "DeepSeek-V4.1 PLE token map contains an invalid negative ID");
    }
    return rows;
}
