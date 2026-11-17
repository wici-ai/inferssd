// SPDX-License-Identifier: Apache-2.0
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

// Literal wire bytes, independent of the encoder: four consecutive 2-bit codes per byte.
static void decode_and_dot(int columns) {
    const auto * traits = ggml_get_type_traits(GGML_TYPE_Q2_0);
    const auto * cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q2_0);
    const int blocks = columns / 64;
    std::vector<block_q2_0> weights(blocks * 3);
    std::vector<block_q8_0> input(blocks * 2);
    const float scales[] = { 0.0f, 0.5f, -2.0f, 0.00006103515625f };
    for (size_t i = 0; i < input.size(); ++i) {
        input[i].d = ggml_fp32_to_fp16(i % 2 ? 0.25f : 2.0f);
        for (int j = 0; j < 32; ++j) { input[i].qs[j] = int8_t((i * 37 + j * 17) % 256 - 128); }
    }
    for (int pattern = 0; pattern < 256; ++pattern) {
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i].d = ggml_fp32_to_fp16(scales[(i + pattern) % 4]);
            for (int j = 0; j < 16; ++j) { weights[i].qs[j] = uint8_t(pattern + i * 19 + j * 7); }
        }
        std::vector<float> decoded(columns * 3);
        traits->to_float(weights.data(), decoded.data(), decoded.size());
        for (int row = 0; row < 3; ++row) {
            double expected_dot = 0.0;
            for (int col = 0; col < columns; ++col) {
                const auto & block = weights[row * blocks + col / 64];
                const int code = (block.qs[(col % 64) / 4] >> (2 * (col % 4))) & 3;
                const float expected = (code - 1) * scales[(row * blocks + col / 64 + pattern) % 4];
                require(decoded[row * columns + col] == expected, "Q2_0 wire decoding differs");
                const auto & y = input[col / 32];
                expected_dot += double(expected) * y.qs[col % 32] * (col / 32 % 2 ? 0.25 : 2.0);
            }
            float dot = NAN;
            cpu->vec_dot(columns, &dot, 0, weights.data() + row * blocks, 0, input.data(), 0, 1);
            require(std::isfinite(dot) && std::abs(dot - expected_dot) <= 0.002 + 2e-6 * std::abs(expected_dot),
                    "Q2_0/Q8_0 dot differs from independent decoded reference");
        }
    }
}

static void quantize_and_validate() {
    constexpr int columns = 640, rows = 3;
    std::vector<float> values(columns * rows);
    std::vector<block_q2_0> packed(columns * rows / 64);
    for (int row = 0; row < rows; ++row) {
        const float scale = float(row + 1) * 0.5f;
        const float pattern[] = { -scale, 0.0f, scale, 0.0f };
        for (int col = 0; col < columns; ++col) { values[row * columns + col] = pattern[col % 4]; }
    }
    const size_t bytes = ggml_quantize_chunk(GGML_TYPE_Q2_0, values.data(), packed.data(), 0, rows, columns, nullptr);
    require(bytes == 540, "Q2_0 multirow byte count");
    for (int i = 0; i < int(packed.size()); ++i) {
        require(ggml_fp16_to_fp32(packed[i].d) == float(i / 10 + 1) * 0.5f, "Q2_0 row scale");
        for (uint8_t b : packed[i].qs) { require(b == 0x64, "Q2_0 encoder packing"); }
    }
    require(ggml_validate_row_data(GGML_TYPE_Q2_0, packed.data(), bytes), "valid Q2_0 rejected");
    require(!ggml_validate_row_data(GGML_TYPE_Q2_0, packed.data(), bytes - 1), "partial block accepted");
    for (uint16_t invalid : {uint16_t(0x7c00), uint16_t(0xfc00), uint16_t(0x7e00)}) {
        packed[0].d = invalid;
        require(!ggml_validate_row_data(GGML_TYPE_Q2_0, packed.data(), bytes), "nonfinite scale accepted");
    }
    require(!ggml_validate_row_data(GGML_TYPE_COUNT, nullptr, 0), "unknown type accepted");
}

int main() {
    try {
        ggml_cpu_init();
        require(int(GGML_TYPE_Q2_0) == 42, "Q2_0 wire type");
        require(sizeof(block_q2_0) == 18 && ggml_type_size(GGML_TYPE_Q2_0) == 18, "Q2_0 block bytes");
        require(ggml_blck_size(GGML_TYPE_Q2_0) == 64, "Q2_0 block width");
        require(ggml_row_size(GGML_TYPE_Q2_0, 640) == 180, "Q2_0 640-column row bytes");
        const auto * cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q2_0);
        require(cpu->vec_dot && cpu->from_float && cpu->vec_dot_type == GGML_TYPE_Q8_0 && cpu->nrows == 1,
                "Q2_0 CPU dispatch");
        for (int columns : {64, 128, 192, 640}) { decode_and_dot(columns); }
        quantize_and_validate();
        std::puts("Q2_0 wire decoding, mixed-scale dot, multirow quantization and validation: PASS");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
