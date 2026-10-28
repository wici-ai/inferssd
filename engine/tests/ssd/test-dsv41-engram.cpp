// SPDX-License-Identifier: MIT
#include "llama-engram.h"

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

int main() {
    try {
        exhaustive_value_test();
        bf16_ties_to_even_test();
        antirez_golden_boundaries_test();
        row_and_special_test();
        std::printf("DSV41_ENGRAM_REFERENCE_OK\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "DSV41_ENGRAM_REFERENCE_FAILED: %s\n", error.what());
        return 1;
    }
}
