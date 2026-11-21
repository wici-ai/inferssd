// SPDX-License-Identifier: MIT
#include "llama-engram.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"
#include "llama-model.h"
#include "models/deepseek41-engram.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static uint32_t bits_of(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Direct transcription of antirez/ds4 ds4_engram.c at commit
// 8db1d1d155cb0400a86a86b9c62d0defb3a6148b (source SHA-256
// 02785ce48599fa4f514a530697efd0f4483d405c42367332748f8facb2c106cc).
// Copyright (c) 2026 The ds4.c authors
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2023 DeepSeek
// MIT License; see NOTICE, entry 4.
// Keep this reference separate from llama_dsv41_engram_decode_value so changes
// to either side cannot silently redefine the test expectation.
static bool antirez_reference(uint8_t code, uint8_t scale, float & out) {
    if ((code & 127) == 127 || scale == 255) {
        return false;
    }
    const int exponent = (code >> 3) & 15;
    const int mantissa = code & 7;
    float value = exponent ? std::ldexp(float(8 + mantissa), exponent - 10) :
                             std::ldexp(float(mantissa), -9);
    if (code & 128) {
        value = -value;
    }
    value = std::ldexp(value, int(scale) - 127);
    uint32_t bits = bits_of(value);
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(value));
    if (!std::isfinite(value)) {
        return false;
    }
    out = value;
    return true;
}

static void exhaustive_value_test() {
    size_t valid = 0;
    size_t invalid = 0;
    for (unsigned code = 0; code < 256; ++code) {
        for (unsigned scale = 0; scale < 256; ++scale) {
            float expected = 0.0f;
            float actual = 0.0f;
            const bool reference_ok = antirez_reference(uint8_t(code), uint8_t(scale), expected);
            const auto status = llama_dsv41_engram_decode_value(uint8_t(code), uint8_t(scale), actual);
            const bool actual_ok = status == llama_dsv41_engram_decode_status::ok;
            require(reference_ok == actual_ok, "Engram decoder validity differs from antirez reference");
            if (reference_ok) {
                require(bits_of(expected) == bits_of(actual), "Engram decoder bits differ from antirez reference");
                ++valid;
            } else {
                ++invalid;
            }
        }
    }
    require(valid > 0 && invalid > 0, "Engram exhaustive vector did not cover both outcomes");
    std::printf("exhaustive_pairs=65536 valid=%zu invalid=%zu bitwise=1\n", valid, invalid);
}

static void bf16_ties_to_even_test() {
    const std::array<uint32_t, 8> input = {
        0x3f807fffu, // below an even tie
        0x3f808000u, // even tie rounds down
        0x3f808001u, // above the tie rounds up
        0x3f818000u, // odd tie rounds up to even
        0xbf807fffu,
        0xbf808000u,
        0xbf808001u,
        0xbf818000u,
    };
    const std::array<uint32_t, 8> expected = {
        0x3f800000u,
        0x3f800000u,
        0x3f810000u,
        0x3f820000u,
        0xbf800000u,
        0xbf800000u,
        0xbf810000u,
        0xbf820000u,
    };
    for (size_t i = 0; i < input.size(); ++i) {
        float value;
        std::memcpy(&value, &input[i], sizeof(value));
        require(bits_of(llama_dsv41_engram_bf16_round(value)) == expected[i],
                "BF16 ties-to-even boundary mismatch");
    }
}

static void antirez_golden_boundaries_test() {
    struct golden_case {
        uint8_t code;
        uint8_t scale;
        uint32_t bits;
        llama_dsv41_engram_decode_status status;
    };
    const std::array<golden_case, 14> cases = {{
        { 0x00,   0, 0x00000000u, llama_dsv41_engram_decode_status::ok },
        { 0x80, 254, 0x80000000u, llama_dsv41_engram_decode_status::ok },
        { 0x01,   0, 0x00000000u, llama_dsv41_engram_decode_status::ok },
        { 0x04,   0, 0x00000000u, llama_dsv41_engram_decode_status::ok },
        { 0x05,   0, 0x00010000u, llama_dsv41_engram_decode_status::ok },
        { 0x01, 254, 0x7a800000u, llama_dsv41_engram_decode_status::ok },
        { 0x7e,   0, 0x04600000u, llama_dsv41_engram_decode_status::ok },
        { 0x7e, 246, 0x7f600000u, llama_dsv41_engram_decode_status::ok },
        { 0x7e, 247, 0x00000000u, llama_dsv41_engram_decode_status::nonfinite },
        { 0x7e, 254, 0x00000000u, llama_dsv41_engram_decode_status::nonfinite },
        { 0x7f, 127, 0x00000000u, llama_dsv41_engram_decode_status::invalid_e4m3 },
        { 0xff, 127, 0x00000000u, llama_dsv41_engram_decode_status::invalid_e4m3 },
        { 0x01, 255, 0x00000000u, llama_dsv41_engram_decode_status::invalid_e8m0 },
        { 0x81, 255, 0x00000000u, llama_dsv41_engram_decode_status::invalid_e8m0 },
    }};

    for (const golden_case & test : cases) {
        float actual = 0.0f;
        const auto status = llama_dsv41_engram_decode_value(test.code, test.scale, actual);
        require(status == test.status, "Engram golden boundary status mismatch");
        if (status == llama_dsv41_engram_decode_status::ok) {
            require(bits_of(actual) == test.bits, "Engram golden boundary bits mismatch");
        }
    }
}

static void row_and_special_test() {
    std::array<uint8_t, LLAMA_DSV41_ENGRAM_ROW_BYTES> raw = {};
    std::array<float, LLAMA_DSV41_ENGRAM_DIM> actual = {};
    for (size_t i = 0; i < LLAMA_DSV41_ENGRAM_DIM; ++i) {
        // The last group uses the largest valid E8M0 scale. Keep its E4M3
        // magnitudes small enough to stay finite; overflow is tested below.
        raw[i] = i / LLAMA_DSV41_ENGRAM_SCALE_GROUP == 7 ?
            uint8_t((i & 0x80u) | (i % 15u)) :
            uint8_t(i == 127 ? 126 : i);
    }
    for (size_t i = 0; i < LLAMA_DSV41_ENGRAM_SCALE_COUNT; ++i) {
        raw[LLAMA_DSV41_ENGRAM_DIM + i] = uint8_t(i == 0 ? 0 : i == 7 ? 254 : 120 + i);
    }
    require(llama_dsv41_engram_decode_row(raw.data(), actual.data()) ==
            llama_dsv41_engram_decode_status::ok, "valid Engram row was rejected");
    for (size_t i = 0; i < LLAMA_DSV41_ENGRAM_DIM; ++i) {
        float expected;
        require(antirez_reference(raw[i], raw[LLAMA_DSV41_ENGRAM_DIM + i / 32], expected),
                "reference rejected valid Engram row");
        require(bits_of(expected) == bits_of(actual[i]), "Engram row decode differs bitwise");
    }

    raw[17] = 0x7f;
    require(llama_dsv41_engram_decode_row(raw.data(), actual.data()) ==
            llama_dsv41_engram_decode_status::invalid_e4m3, "E4M3 NaN sentinel was accepted");
    raw[17] = 1;
    raw[LLAMA_DSV41_ENGRAM_DIM + 3] = 255;
    require(llama_dsv41_engram_decode_row(raw.data(), actual.data()) ==
            llama_dsv41_engram_decode_status::invalid_e8m0, "E8M0 sentinel was accepted");

    float overflow = 0.0f;
    require(llama_dsv41_engram_decode_value(0x7e, 254, overflow) ==
            llama_dsv41_engram_decode_status::nonfinite, "scaled E4M3 infinity was accepted");
    float negative_sentinel = 0.0f;
    require(llama_dsv41_engram_decode_value(0xff, 127, negative_sentinel) ==
            llama_dsv41_engram_decode_status::invalid_e4m3,
            "negative E4M3 all-ones sentinel was accepted");
    float subnormal = 0.0f;
    require(llama_dsv41_engram_decode_value(0x01, 127, subnormal) ==
            llama_dsv41_engram_decode_status::ok && bits_of(subnormal) == 0x3b000000u,
            "E4M3 subnormal boundary mismatch");
    float positive_zero = 1.0f, negative_zero = 1.0f;
    require(llama_dsv41_engram_decode_value(0x00, 127, positive_zero) ==
            llama_dsv41_engram_decode_status::ok && bits_of(positive_zero) == 0x00000000u,
            "positive E4M3 zero mismatch");
    require(llama_dsv41_engram_decode_value(0x80, 127, negative_zero) ==
            llama_dsv41_engram_decode_status::ok && bits_of(negative_zero) == 0x80000000u,
            "negative E4M3 zero mismatch");
}

// unsloth's DeepSeek-V4.1 GGUFs store the Engram as MXFP8 blocks: an E8M0 scale, then 32 E4M3 codes.
// ggml decodes each value exactly, e4m3 * 2^(scale - 127): the reference above wherever its BF16
// rounding leaves the value as it is (every normal F32 result), NaN for the NaN code or the NaN scale,
// and the exact value (or infinity) otherwise.
static void mxfp8_block_test() {
    const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_MXFP8);
    require(traits->blck_size == 32 && traits->type_size == 33 && traits->to_float != nullptr,
            "MXFP8 block geometry mismatch");
    for (int scale = 0; scale < 256; ++scale) {
        for (int first = 0; first < 256; first += 32) {
            uint8_t block[33];
            block[0] = uint8_t(scale);
            for (int j = 0; j < 32; ++j) {
                block[1 + j] = uint8_t(first + j);
            }
            float decoded[32];
            traits->to_float(block, decoded, 32);
            for (int j = 0; j < 32; ++j) {
                const int code = first + j;
                if ((code & 127) == 127 || scale == 255) {
                    require(std::isnan(decoded[j]), "MXFP8 NaN code or scale was decoded as a number");
                    continue;
                }
                const int exponent = (code >> 3) & 15, mantissa = code & 7;
                const double magnitude = exponent ? std::ldexp(8.0 + mantissa, exponent - 10) : std::ldexp(mantissa, -9);
                const float exact = float(std::ldexp((code & 128) ? -magnitude : magnitude, scale - 127));
                require(bits_of(decoded[j]) == bits_of(exact), "MXFP8 value is not the exact E4M3 value");
                float reference = 0.0f;
                if (std::fpclassify(exact) == FP_NORMAL) {
                    require(llama_dsv41_engram_decode_value(uint8_t(code), uint8_t(scale), reference) ==
                            llama_dsv41_engram_decode_status::ok && bits_of(reference) == bits_of(exact),
                            "MXFP8 value differs from the Engram reference decode");
                }
            }
        }
    }
}

// Upstream llama.cpp 9581154a, src/models/deepseek41.cpp, llama_model_deepseek41::engram_hash, transcribed over its
// data as upstream holds it (ple.* metadata: the bucket arrays as uint32, the pad as a token id the map looks up) and a
// minimal ubatch; this snapshot has no mixed ubatches. Keep it apart from llama_dsv41_engram_hash, so that a change to
// either cannot redefine what the other is held to.
struct upstream_ubatch {
    uint32_t            n_tokens;
    const llama_token * token;
    const llama_pos   * pos;
};
struct upstream_hparams {
    uint32_t              ple_ngram_size;
    uint32_t              ple_heads_per_ngram;
    uint32_t              ple_n_heads;
    uint32_t              engram_pad_id; // engram.pad_token_id
    std::vector<uint64_t> ple_layer_multipliers;
    std::vector<uint32_t> ple_head_vocab_sizes;
    std::vector<uint32_t> ple_head_offsets;
};
static void upstream_engram_hash(const upstream_hparams & hparams, const std::vector<int32_t> & engram_token_map,
        int e, const upstream_ubatch & ubatch, const std::vector<llama_token> & prev, std::vector<int32_t> & out) {
    static constexpr int64_t DSV41_ENGRAM_DEAD = -1;
    const int64_t n_gram = hparams.ple_ngram_size;
    const int64_t n_prev = n_gram - 1;
    const int64_t n_head = hparams.ple_heads_per_ngram;
    const int64_t n_cols = hparams.ple_n_heads;

    const uint64_t * mult   = hparams.ple_layer_multipliers.data() + (size_t) e*n_gram;
    const uint32_t * primes = hparams.ple_head_vocab_sizes.data()  + (size_t) e*n_cols;
    const int64_t    pad    = engram_token_map[hparams.engram_pad_id];
    const uint32_t * offs   = hparams.ple_head_offsets.data()      + (size_t) e*n_cols;

    const auto cid = [&](llama_token t) -> int64_t {
        return t < 0 || t >= (llama_token) engram_token_map.size() ? DSV41_ENGRAM_DEAD : engram_token_map[t];
    };

    out.resize(ubatch.n_tokens*n_cols);
    std::vector<uint64_t> toks(n_gram);
    for (int64_t i = 0; i < ubatch.n_tokens; ++i) {
        // shift 0 is the token itself; once a shift is blocked (sequence start or dead token) every longer one is too
        const bool is_embd = !ubatch.token;
        int64_t c = is_embd ? DSV41_ENGRAM_DEAD : cid(ubatch.token[i]);
        bool blocked = c == DSV41_ENGRAM_DEAD;
        toks[0] = blocked ? pad : c;
        for (int64_t s = 1; s < n_gram; ++s) {
            c = ubatch.pos[i] - s < 0 ? DSV41_ENGRAM_DEAD : cid(prev[i*n_prev + (n_prev - s)]);
            blocked = blocked || c == DSV41_ENGRAM_DEAD;
            toks[s] = blocked ? pad : c;
        }

        uint64_t rolling = toks[0]*mult[0];
        for (int64_t o = 1; o < n_gram; ++o) {
            rolling ^= toks[o]*mult[o];
            for (int64_t h = 0; h < n_head; ++h) {
                const int64_t col = (o - 1)*n_head + h;
                out[i*n_cols + col] = (int32_t) (rolling % primes[col] + offs[col]);
            }
        }
    }
}

// The engine's rows of Engram layer E for a ubatch, from metadata as UPSTREAM holds it: the pad its compressed id, as
// the ple.* reader leaves it.
static std::vector<int32_t> engine_rows(const upstream_hparams & upstream, const std::vector<int32_t> & token_map,
        int e, const upstream_ubatch & ubatch, const std::vector<llama_token> & prev) {
    const size_t n_cols = upstream.ple_n_heads;
    const std::vector<uint64_t> primes(upstream.ple_head_vocab_sizes.begin(), upstream.ple_head_vocab_sizes.end());
    const std::vector<uint64_t> offsets(upstream.ple_head_offsets.begin(), upstream.ple_head_offsets.end());
    const llama_dsv41_engram_hash_params params = {
        upstream.ple_ngram_size, upstream.ple_heads_per_ngram,
        upstream.ple_layer_multipliers.data() + (size_t) e*upstream.ple_ngram_size,
        primes.data() + (size_t) e*n_cols, offsets.data() + (size_t) e*n_cols,
        token_map.data(), token_map.size(), (uint64_t) token_map[upstream.engram_pad_id],
    };
    std::vector<int32_t> rows(ubatch.n_tokens*n_cols);
    llama_dsv41_engram_hash(params, ubatch.n_tokens, ubatch.token, ubatch.pos, prev.data(), rows.data());
    return rows;
}

// Hand-worked rows: one layer of 4-grams with 2 heads, the pad token 0 (compressed id 3, not 0). A token the map marks
// -1, one outside it and one before the start of the sequence stop every longer shift alike.
static void engram_hash_golden_test() {
    const std::vector<int32_t> token_map = {3, -1, 7, 11};
    upstream_hparams hp = { 4, 2, 6, 0, {3, 5, 7, 11}, {5, 7, 11, 13, 17, 19}, {0, 5, 12, 23, 36, 53} };
    // newest first: the token, then its predecessors
    const int32_t history[][4] = {{2, 3, 0, 2}, {2, -1, 3, 2}, {2, 1, 3, 2},
                                  {-1, 3, 2, 0}, {99, 3, 2, 0}, {2, 3, -1, 0}};
    const int32_t expected[][6] = {{4, 11, 12, 26, 39, 61}, {1, 10, 16, 25, 48, 61},
                                   {1, 10, 16, 25, 48, 61}, {1, 11, 20, 29, 52, 65},
                                   {1, 11, 20, 29, 52, 65}, {4, 11, 12, 26, 41, 56}};
    for (size_t i = 0; i < 6; ++i) {
        const llama_token token = history[i][0];
        const llama_pos pos = 3;
        const std::vector<llama_token> prev = {history[i][3], history[i][2], history[i][1]};
        const upstream_ubatch ubatch = {1, &token, &pos};
        std::vector<int32_t> reference;
        upstream_engram_hash(hp, token_map, 0, ubatch, prev, reference);
        require(std::equal(reference.begin(), reference.end(), expected[i]), "upstream Engram hash golden mismatch");
        require(engine_rows(hp, token_map, 0, ubatch, prev) == reference, "Engram hash golden mismatch");
    }
    // position 1: the predecessors two and three back are before the start, whatever the cells hold
    const llama_token token = 2;
    const llama_pos pos = 1;
    const std::vector<llama_token> prev = {2, 0, 3};
    const upstream_ubatch ubatch = {1, &token, &pos};
    std::vector<int32_t> reference;
    upstream_engram_hash(hp, token_map, 0, ubatch, prev, reference);
    require(reference == std::vector<int32_t>({4, 11, 12, 26, 41, 56}), "upstream Engram hash at the start of a sequence");
    require(engine_rows(hp, token_map, 0, ubatch, prev) == reference, "Engram hash at the start of a sequence");
}

// Random metadata, maps with -1 entries, tokens outside the vocabulary, sequences starting inside a ubatch and
// embeddings: the engine's rows are upstream's.
static void engram_hash_upstream_test() {
    std::mt19937_64 rng(20261121);
    size_t compared = 0;
    for (int trial = 0; trial < 200; ++trial) {
        const uint32_t n_gram = 2 + rng() % 4, n_head = 1 + rng() % 8, n_layer = 1 + rng() % 3;
        const uint32_t n_cols = (n_gram - 1)*n_head, n_vocab = 16 + rng() % 300;
        upstream_hparams hp;
        hp.ple_ngram_size = n_gram;
        hp.ple_heads_per_ngram = n_head;
        hp.ple_n_heads = n_cols;
        std::vector<int32_t> token_map(n_vocab);
        for (auto & id : token_map) { id = rng() % 8 == 0 ? -1 : int32_t(rng() % 131072); }
        do { hp.engram_pad_id = rng() % n_vocab; } while (token_map[hp.engram_pad_id] < 0);
        uint64_t base = 0;
        for (uint32_t c = 0; c < n_layer*n_cols; ++c) {
            const uint32_t prime = 2 + rng() % 1000003;
            hp.ple_head_vocab_sizes.push_back(prime);
            hp.ple_head_offsets.push_back((uint32_t) base);
            base += prime;
        }
        for (uint32_t i = 0; i < n_layer*n_gram; ++i) { hp.ple_layer_multipliers.push_back(rng() | 1); }

        const uint32_t n_tokens = 1 + rng() % 40;
        std::vector<llama_token> tokens(n_tokens);
        std::vector<llama_pos> pos(n_tokens);
        llama_pos p = rng() % 5;
        for (uint32_t i = 0; i < n_tokens; ++i) {
            tokens[i] = int32_t(rng() % (n_vocab + 8)) - 2; // a few outside the map on both sides
            if (rng() % 16 == 0) { p = 0; }                  // another sequence starts
            pos[i] = p++;
        }
        std::vector<llama_token> prev(n_tokens*(n_gram - 1));
        for (auto & t : prev) { t = rng() % 10 == 0 ? LLAMA_TOKEN_NULL : int32_t(rng() % n_vocab); }
        for (const bool embd : {false, true}) {
            const upstream_ubatch ubatch = {n_tokens, embd ? nullptr : tokens.data(), pos.data()};
            for (uint32_t e = 0; e < n_layer; ++e) {
                std::vector<int32_t> reference;
                upstream_engram_hash(hp, token_map, e, ubatch, prev, reference);
                require(engine_rows(hp, token_map, e, ubatch, prev) == reference, "Engram hash differs from upstream");
                compared += reference.size();
            }
        }
    }
    std::printf("engram hash: %zu rows equal upstream's\n", compared);
}

// The gate is upstream's copysign: a zero dot gates sigmoid(0.001), and every other dot gates exactly as
// sigmoid(sgn(dot)*mag), the gate this engine had, did.
static void engram_gate_test() {
    const std::array<float, 11> dots = {-4.0f, -1.0f, -1e-6f, -1e-8f, -0.0f, 0.0f, 1e-8f, 1e-6f, 1.0f, 4.0f, 0.37f};
    ggml_context * ctx = ggml_init({1024 * 1024, nullptr, false});
    require(ctx != nullptr, "gate test context allocation failed");
    ggml_tensor * dot = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dots.size());
    std::memcpy(dot->data, dots.data(), sizeof(dots));
    ggml_tensor * gate = llama_dsv41_engram_gate(ctx, dot);
    ggml_tensor * mag = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_abs(ctx, dot), 1e-6f, INFINITY));
    ggml_tensor * sgn_gate = ggml_sigmoid(ctx, ggml_mul(ctx, ggml_sgn(ctx, dot), mag));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, gate);
    ggml_build_forward_expand(graph, sgn_gate);
    require(ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS, "gate graph execution failed");
    const auto * actual = static_cast<const float *>(gate->data);
    const auto * before = static_cast<const float *>(sgn_gate->data);
    for (size_t i = 0; i < dots.size(); ++i) {
        const double root = dots[i] == 0.0f ? std::sqrt(1e-6) :
            std::copysign(std::sqrt(std::max(std::abs(double(dots[i])), 1e-6)), dots[i]);
        const double expected = 1.0 / (1.0 + std::exp(-root));
        require(std::abs(actual[i] - expected) < 2e-6, "signed-root gate mismatch");
        if (dots[i] != 0.0f) {
            require(bits_of(actual[i]) == bits_of(before[i]), "a nonzero dot gates differently than sgn * mag");
        } else {
            require(actual[i] > 0.5f && before[i] == 0.5f, "a zero dot gates with the positive sign");
        }
    }
    ggml_free(ctx);
}

// With GGUF paths (the model, then its --shard/--ple files): the reader's parameters for that file's Engram, and its
// rows for a prompt with the -1 tokens of its map, against upstream's hash over the file's metadata as it stands.
static std::vector<uint64_t> u64_array(const gguf_context * g, const std::string & key) {
    const int64_t id = gguf_find_key(g, key.c_str());
    require(id >= 0, ("no " + key).c_str());
    const size_t n = gguf_get_arr_n(g, id);
    std::vector<uint64_t> out(n);
    const gguf_type type = gguf_get_arr_type(g, id);
    for (size_t i = 0; i < n; ++i) {
        out[i] = type == GGUF_TYPE_UINT64 ? static_cast<const uint64_t *>(gguf_get_arr_data(g, id))[i]
               : type == GGUF_TYPE_UINT32 ? static_cast<const uint32_t *>(gguf_get_arr_data(g, id))[i]
               : (uint64_t) static_cast<const int32_t *>(gguf_get_arr_data(g, id))[i];
    }
    return out;
}

static uint32_t u32_value(const gguf_context * g, const std::string & key) {
    const int64_t id = gguf_find_key(g, key.c_str());
    require(id >= 0, ("no " + key).c_str());
    return gguf_get_val_u32(g, id);
}

static int real_metadata(const std::vector<std::string> & paths) {
    llama_backend_init();
    // the model as a load lays it out, no weight read (as the memory fit does)
    llama_model_params mparams = llama_model_default_params();
    mparams.no_alloc = true;
    mparams.use_mmap = false;
    std::vector<const char *> c_paths;
    for (const auto & path : paths) { c_paths.push_back(path.c_str()); }
    llama_model * model = c_paths.size() == 1 ? llama_model_load_from_file(c_paths[0], mparams)
                                              : llama_model_load_from_splits(c_paths.data(), c_paths.size(), mparams);
    require(model != nullptr, "cannot load the model's metadata");

    gguf_init_params gparams = { true, nullptr };
    gguf_context * g = gguf_init_from_file(paths[0].c_str(), gparams);
    require(g != nullptr, "cannot read the first file's header");
    const std::string arch = "deepseek41.";
    const bool ple = gguf_find_key(g, (arch + "ple.layers").c_str()) >= 0;
    const bool antirez = gguf_find_key(g, (arch + "engram.encoding").c_str()) >= 0;
    require(!antirez, "an e4m3_e8m0_32_row264 file has no upstream reader to compare with");
    const std::string ns = arch + (ple ? "ple." : "engram.");
    upstream_hparams hp;
    hp.ple_ngram_size      = u32_value(g, ns + (ple ? "ngram_size" : "max_ngram_size"));
    hp.ple_heads_per_ngram = u32_value(g, ns + (ple ? "heads_per_ngram" : "head_count"));
    hp.ple_n_heads         = (hp.ple_ngram_size - 1)*hp.ple_heads_per_ngram;
    hp.ple_layer_multipliers = u64_array(g, ns + (ple ? "layer_multipliers" : "multipliers"));
    for (uint64_t v : u64_array(g, ns + (ple ? "head_vocab_sizes" : "primes"))) { hp.ple_head_vocab_sizes.push_back((uint32_t) v); }
    for (uint64_t v : u64_array(g, ns + (ple ? "head_offsets" : "offsets"))) { hp.ple_head_offsets.push_back((uint32_t) v); }
    const std::vector<uint64_t> layers = u64_array(g, ns + (ple ? "layers" : "layer_ids"));
    std::vector<int32_t> token_map;
    for (uint64_t v : u64_array(g, arch + "engram.token_map")) { token_map.push_back((int32_t) v); }
    // upstream reads the pad token's id (ple.*); an engram.* file gives the compressed id, which the reference finds
    // through one more map entry, past every token
    uint32_t pad_compressed = 0;
    if (ple) {
        hp.engram_pad_id = u32_value(g, arch + "engram.pad_token_id");
        pad_compressed = (uint32_t) token_map.at(hp.engram_pad_id);
    } else {
        pad_compressed = u32_value(g, arch + "engram.pad_id");
    }
    std::vector<int32_t> reference_map = token_map;
    if (!ple) {
        hp.engram_pad_id = (uint32_t) reference_map.size();
        reference_map.push_back((int32_t) pad_compressed);
    }
    size_t dead = 0;
    std::vector<llama_token> dead_tokens;
    for (size_t t = 0; t < token_map.size(); ++t) {
        if (token_map[t] < 0) {
            ++dead;
            if (dead_tokens.size() < 3) { dead_tokens.push_back((llama_token) t); }
        }
    }

    // the reader's parameters
    require(model->engram_n_layer == layers.size(), "engram layer count");
    require(model->engram_pad_id == pad_compressed, "the pad is not the compressed id");
    require(model->engram_token_map == token_map, "token map");
    require(model->engram_multipliers == hp.ple_layer_multipliers, "multipliers");
    require(std::equal(model->engram_primes.begin(), model->engram_primes.end(), hp.ple_head_vocab_sizes.begin(),
            hp.ple_head_vocab_sizes.end()), "bucket sizes");
    require(std::equal(model->engram_offsets.begin(), model->engram_offsets.end(), hp.ple_head_offsets.begin(),
            hp.ple_head_offsets.end()), "bucket offsets");

    // a prompt, with the map's first -1 tokens in it
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string text = "The capital of France is Paris. In 2026, 2 + 3 = 5; a mixture of experts routes each token.";
    std::vector<llama_token> tokens(text.size() + 8);
    const int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(),
                                     true, false);
    require(n > 0, "tokenize");
    tokens.resize(n);
    tokens.insert(tokens.begin() + std::min<size_t>(5, tokens.size()), dead_tokens.begin(), dead_tokens.end());
    const uint32_t n_tokens = (uint32_t) tokens.size();
    const int64_t n_prev = hp.ple_ngram_size - 1;
    std::vector<llama_pos> pos(n_tokens);
    std::vector<llama_token> prev(n_tokens*n_prev);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        pos[i] = (llama_pos) i;
        for (int64_t s = 1; s <= n_prev; ++s) {
            prev[i*n_prev + (n_prev - s)] = (int64_t) i - s >= 0 ? tokens[i - s] : LLAMA_TOKEN_NULL;
        }
    }
    const upstream_ubatch ubatch = {n_tokens, tokens.data(), pos.data()};
    size_t rows = 0;
    for (uint32_t e = 0; e < model->engram_n_layer; ++e) {
        const llama_dsv41_engram_hash_params params = {
            (int64_t) model->hparams.engram_max_ngram_size, (int64_t) model->hparams.engram_n_head,
            model->engram_multipliers.data() + (size_t) e*hp.ple_ngram_size,
            model->engram_primes.data() + (size_t) e*hp.ple_n_heads,
            model->engram_offsets.data() + (size_t) e*hp.ple_n_heads,
            model->engram_token_map.data(), model->engram_token_map.size(), model->engram_pad_id,
        };
        std::vector<int32_t> engine(n_tokens*hp.ple_n_heads), reference;
        llama_dsv41_engram_hash(params, n_tokens, tokens.data(), pos.data(), prev.data(), engine.data());
        upstream_engram_hash(hp, reference_map, (int) e, ubatch, prev, reference);
        require(engine == reference, "the model's rows differ from upstream's hash");
        rows += engine.size();
    }
    std::printf("%s: %s layout, Engram layers", paths[0].c_str(), ple ? "ple.*" : "engram.*");
    for (uint64_t layer : layers) { std::printf(" %u", (unsigned) layer); }
    std::printf("; token map %zu entries, %zu of them -1; pad %u (compressed %u); %u tokens (%zu -1 tokens), %zu rows "
                "equal upstream's\n", token_map.size(), dead, ple ? hp.engram_pad_id : pad_compressed, pad_compressed,
                n_tokens, dead_tokens.size(), rows);
    gguf_free(g);
    llama_model_free(model);
    return 0;
}

int main(int argc, char ** argv) {
    try {
        if (argc > 1) {
            // test-dsv41-engram MODEL [SHARD...]: a real file's Engram metadata
            return real_metadata(std::vector<std::string>(argv + 1, argv + argc));
        }
        exhaustive_value_test();
        bf16_ties_to_even_test();
        antirez_golden_boundaries_test();
        row_and_special_test();
        mxfp8_block_test();
        engram_hash_golden_test();
        engram_hash_upstream_test();
        engram_gate_test();
        std::printf("DSV41_ENGRAM_REFERENCE_OK\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "DSV41_ENGRAM_REFERENCE_FAILED: %s\n", error.what());
        return 1;
    }
}
