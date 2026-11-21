// SPDX-License-Identifier: MIT
#pragma once

#include "ggml.h"
#include "llama.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

// The n-gram hash of one DeepSeek-V4.1 Engram layer, one path for every layout a GGUF stores (engram.*, ple.* and
// e4m3_e8m0_32_row264): each layout's reader (llama_model::load_hparams) maps its metadata onto these, the pad as a
// compressed token ID.
struct llama_dsv41_engram_hash_params {
    int64_t          n_gram;    // longest n-gram
    int64_t          n_head;    // heads per n-gram
    const uint64_t * mult;      // [n_gram]
    const uint64_t * primes;    // [(n_gram - 1) * n_head]: each bucket's row count
    const uint64_t * offsets;   // as primes: the table row each bucket starts at
    const int32_t  * token_map; // token ID -> compressed ID; -1 marks a token no n-gram reaches across
    size_t           n_map;
    uint64_t         pad;       // the compressed ID in place of a token past the start of the sequence or a -1
};

// Ported from upstream llama.cpp 9581154a (llama_model_deepseek41::engram_hash, after engram.py
// NgramHashState.forward). Each of the N_TOKENS tokens (TOKEN null for embeddings, POS its positions, PREV its
// n_gram - 1 predecessors oldest first) gathers (n_gram - 1) * n_head rows into OUT:
//   rolling_s = t[0]*mult[0] ^ ... ^ t[s]*mult[s];  row = rolling_s % primes[s][h] + offsets[s][h]
// A shift is blocked, and so is every longer one, at an embedding, at the start of the sequence, at a token outside
// the map or one the map marks -1; a blocked shift hashes the pad.
static inline void llama_dsv41_engram_hash(const llama_dsv41_engram_hash_params & p, int64_t n_tokens,
        const llama_token * token, const llama_pos * pos, const llama_token * prev, int32_t * out) {
    constexpr int64_t DEAD = -1;
    const int64_t n_prev = p.n_gram - 1;
    const int64_t n_cols = n_prev*p.n_head;

    const auto cid = [&](llama_token t) -> int64_t {
        return t < 0 || (size_t) t >= p.n_map ? DEAD : p.token_map[t];
    };

    for (int64_t i = 0; i < n_tokens; ++i) {
        // shift 0 is the token itself; once a shift is blocked (sequence start or dead token) every longer one is too
        int64_t c = !token ? DEAD : cid(token[i]);
        bool blocked = c == DEAD;
        uint64_t rolling = (blocked ? p.pad : (uint64_t) c)*p.mult[0];
        for (int64_t s = 1; s < p.n_gram; ++s) {
            c = pos[i] - s < 0 ? DEAD : cid(prev[i*n_prev + (n_prev - s)]);
            blocked = blocked || c == DEAD;
            rolling ^= (blocked ? p.pad : (uint64_t) c)*p.mult[s];
            for (int64_t h = 0; h < p.n_head; ++h) {
                const int64_t col = (s - 1)*p.n_head + h;
                out[i*n_cols + col] = (int32_t) (rolling % p.primes[col] + p.offsets[col]);
            }
        }
    }
}

// The gate of an Engram layer from its dot products DOT: sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)), where
// sign(0) = +1 as in copysign (upstream llama.cpp 9581154a, llama_model_deepseek41::graph::build_engram).
static inline ggml_tensor * llama_dsv41_engram_gate(ggml_context * ctx, ggml_tensor * dot) {
    ggml_tensor * mag  = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_abs(ctx, dot), 1e-6f, INFINITY));
    ggml_tensor * sign = ggml_scale_bias(ctx, ggml_step(ctx, ggml_neg(ctx, dot)), -2.0f, 1.0f);
    return ggml_sigmoid(ctx, ggml_mul(ctx, mag, sign));
}
