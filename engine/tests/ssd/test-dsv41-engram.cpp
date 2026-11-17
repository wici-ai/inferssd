// SPDX-License-Identifier: MIT
#include "llama-engram.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "models/deepseek41-engram.h"

#include <algorithm>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

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

static void ple_hash_test() {
    const int32_t token_map[] = {3, -1, 7, 11};
    const uint64_t multipliers[] = {3, 5, 7, 11};
    const uint64_t primes[] = {5, 7, 11, 13, 17, 19};
    const uint64_t offsets[] = {0, 5, 12, 23, 36, 53};
    // Padding token 0 maps to compressed ID 3, not 0. A mapped -1 and an
    // absent/out-of-vocabulary token must stop all older history alike.
    const int32_t history[][4] = {{2, 3, 0, 2}, {2, -1, 3, 2}, {2, 1, 3, 2},
                                  {-1, 3, 2, 0}, {99, 3, 2, 0}, {2, 3, -1, 0}};
    const int32_t expected[][6] = {{4, 11, 12, 26, 39, 61}, {1, 10, 16, 25, 48, 61},
                                   {1, 10, 16, 25, 48, 61}, {1, 11, 20, 29, 52, 65},
                                   {1, 11, 20, 29, 52, 65}, {4, 11, 12, 26, 41, 56}};
    for (size_t i = 0; i < 6; ++i) {
        int32_t rows[6];
        llama_dsv41_ple_hash(history[i], 4, 2, token_map, 4, 0,
                            multipliers, primes, offsets, rows);
        require(std::memcmp(rows, expected[i], sizeof(rows)) == 0, "PLE hash golden mismatch");
        uint64_t shared_offsets[6];
        for (size_t j = 0; j < 6; ++j) { shared_offsets[j] = offsets[j] + 1000; }
        llama_dsv41_ple_hash(history[i], 4, 2, token_map, 4, 0,
                            multipliers, primes, shared_offsets, rows);
        for (size_t j = 0; j < 6; ++j) {
            require(rows[j] == expected[i][j] + 1000, "PLE module offset was lost");
        }
    }
}

static void ple_fused_gate_test() {
    constexpr int width = 8, copies = 2, tokens = 3;
    constexpr float eps = 1e-6f;
    ggml_context * ctx = ggml_init({2 * 1024 * 1024, nullptr, false});
    require(ctx != nullptr, "PLE test context allocation failed");
    ggml_tensor * query = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, copies, tokens);
    ggml_tensor * key = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, copies, tokens);
    ggml_tensor * scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, copies);
    auto * q = static_cast<float *>(query->data);
    auto * k = static_cast<float *>(key->data);
    auto * w = static_cast<float *>(scale->data);
    float q_scale[width * copies], k_scale[width * copies];
    for (int j = 0; j < width * copies; ++j) {
        q_scale[j] = float(j % 5 - 2) / 3.0f;
        k_scale[j] = float(j % 7 + 1) / 4.0f;
        w[j] = q_scale[j] * k_scale[j];
    }
    for (int j = 0; j < width * copies * tokens; ++j) {
        q[j] = j < width ? 0.0f : float(j % 11 - 5) / 4.0f;
        k[j] = float(j % 13 - 6) / 5.0f;
    }
    ggml_tensor * dot = llama_dsv41_ple_dot(ctx, query, key, scale, eps);
    ggml_tensor * gate = llama_dsv41_ple_gate(ctx, dot);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, gate);
    require(ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS, "PLE graph execution failed");
    const auto * actual_dot = static_cast<const float *>(dot->data);
    const auto * actual_gate = static_cast<const float *>(gate->data);
    for (int stream = 0; stream < copies * tokens; ++stream) {
        double qq = 0, kk = 0, weighted = 0;
        for (int j = 0; j < width; ++j) {
            const int at = stream * width + j;
            const int sw = (stream % copies) * width + j;
            qq += double(q[at]) * q[at];
            kk += double(k[at]) * k[at];
            weighted += double(q[at]) * q_scale[sw] * k[at] * k_scale[sw];
        }
        const double expected_dot = weighted / std::sqrt(qq / width + eps) /
                                    std::sqrt(kk / width + eps) / std::sqrt(double(width));
        const double signed_root = (expected_dot < 0 ? -1.0 : 1.0) *
                                   std::sqrt(std::max(std::abs(expected_dot), 1e-6));
        const double expected_gate = 1.0 / (1.0 + std::exp(-signed_root));
        require(std::abs(actual_dot[stream] - expected_dot) < 2e-6, "PLE fused norm scales mismatch");
        require(std::abs(actual_gate[stream] - expected_gate) < 2e-6, "PLE signed-root gate mismatch");
    }
    require(actual_gate[0] > 0.5f, "PLE zero-dot gate must use positive sign");
    ggml_free(ctx);
}

int main() {
    try {
        exhaustive_value_test();
        bf16_ties_to_even_test();
        antirez_golden_boundaries_test();
        row_and_special_test();
        mxfp8_block_test();
        ple_hash_test();
        ple_fused_gate_test();
        std::printf("DSV41_ENGRAM_REFERENCE_OK\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "DSV41_ENGRAM_REFERENCE_FAILED: %s\n", error.what());
        return 1;
    }
}
