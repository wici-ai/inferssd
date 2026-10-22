#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

enum class llama_dsv41_engram_decode_status : uint8_t {
    ok,
    invalid_e4m3,
    invalid_e8m0,
    nonfinite,
};

static constexpr size_t LLAMA_DSV41_ENGRAM_DIM = 256;
static constexpr size_t LLAMA_DSV41_ENGRAM_SCALE_GROUP = 32;
static constexpr size_t LLAMA_DSV41_ENGRAM_SCALE_COUNT =
    LLAMA_DSV41_ENGRAM_DIM / LLAMA_DSV41_ENGRAM_SCALE_GROUP;
static constexpr size_t LLAMA_DSV41_ENGRAM_ROW_BYTES =
    LLAMA_DSV41_ENGRAM_DIM + LLAMA_DSV41_ENGRAM_SCALE_COUNT;

// Round an F32 value to BF16 using round-to-nearest, ties-to-even, while
// retaining F32 storage. This is the operation used by antirez's Engram reader.
static inline float llama_dsv41_engram_bf16_round(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static inline llama_dsv41_engram_decode_status llama_dsv41_engram_decode_value(
        uint8_t code,
        uint8_t scale,
        float & value) {
    // In this encoding magnitude 0x7f and E8M0 exponent 255 are sentinels.
    // There is no separate finite/Inf interpretation for those bit patterns.
    if ((code & 127u) == 127u) {
        return llama_dsv41_engram_decode_status::invalid_e4m3;
    }
    if (scale == 255u) {
        return llama_dsv41_engram_decode_status::invalid_e8m0;
    }

    const int exponent = (code >> 3) & 15;
    const int mantissa = code & 7;
    const float magnitude = exponent ?
        std::ldexp(float(8 + mantissa), exponent - 10) :
        std::ldexp(float(mantissa), -9);
    value = std::ldexp((code & 128u) ? -magnitude : magnitude, int(scale) - 127);
    value = llama_dsv41_engram_bf16_round(value);
    uint32_t rounded_bits;
    std::memcpy(&rounded_bits, &value, sizeof(rounded_bits));
    return (rounded_bits & 0x7f800000u) != 0x7f800000u ?
        llama_dsv41_engram_decode_status::ok : llama_dsv41_engram_decode_status::nonfinite;
}

static inline llama_dsv41_engram_decode_status llama_dsv41_engram_decode_row(
        const uint8_t raw[LLAMA_DSV41_ENGRAM_ROW_BYTES],
        float out[LLAMA_DSV41_ENGRAM_DIM]) {
    for (size_t i = 0; i < LLAMA_DSV41_ENGRAM_DIM; ++i) {
        const auto status = llama_dsv41_engram_decode_value(
            raw[i], raw[LLAMA_DSV41_ENGRAM_DIM + i / LLAMA_DSV41_ENGRAM_SCALE_GROUP], out[i]);
        if (status != llama_dsv41_engram_decode_status::ok) {
            return status;
        }
    }
    return llama_dsv41_engram_decode_status::ok;
}
