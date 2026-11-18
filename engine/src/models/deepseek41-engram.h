// SPDX-License-Identifier: MIT
#pragma once

#include "ggml.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

// Independently implemented against ggml-org/llama.cpp's DeepSeek41 Engram
// semantics at 9581154a2250cc9e6948e4c2a2fc4b4e19d1a57e; see NOTICE entry 3.
// PLE stores pad_token_id in the tokenizer vocabulary, and -1 in token_map
// marks a history barrier. Offsets address one table shared by all modules.
// history is newest first; metadata validation is the loader's responsibility.
static inline void llama_dsv41_ple_hash(
        const int32_t * history, size_t ngram, size_t heads,
        const int32_t * token_map, size_t vocabulary, uint32_t pad_token_id,
        const uint64_t * multipliers, const uint64_t * primes, const uint64_t * offsets,
        int32_t * rows) {
    const uint64_t padding = (uint64_t) token_map[pad_token_id];
    uint64_t rolling = 0;
    bool barrier = false;
    for (size_t shift = 0; shift < ngram; ++shift) {
        const int32_t token = history[shift];
        const int32_t mapped = token >= 0 && (size_t) token < vocabulary ? token_map[token] : -1;
        barrier = barrier || mapped < 0;
        rolling ^= (barrier ? padding : (uint64_t) mapped) * multipliers[shift];
        if (shift == 0) {
            continue;
        }
        for (size_t head = 0; head < heads; ++head) {
            const size_t bucket = (shift - 1) * heads + head;
            rows[bucket] = (int32_t) (offsets[bucket] + rolling % primes[bucket]);
        }
    }
}

static inline ggml_tensor * llama_dsv41_ple_gate(ggml_context * ctx, ggml_tensor * dot) {
    ggml_tensor * magnitude = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_abs(ctx, dot), 1e-6f, INFINITY));
    // STEP is strict > 0: the zero dot uses the positive branch, as copysign.
    ggml_tensor * sign = ggml_scale_bias(ctx, ggml_step(ctx, ggml_neg(ctx, dot)), -2.0f, 1.0f);
    return ggml_sigmoid(ctx, ggml_mul(ctx, magnitude, sign));
}
