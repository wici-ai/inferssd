// SPDX-License-Identifier: Apache-2.0
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static float rounded(float x, ggml_type type) {
    if (type == GGML_TYPE_F16)  { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)); }
    if (type == GGML_TYPE_BF16) { return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x)); }
    return x;
}

static char * row(ggml_tensor * t, int r, int b) {
    return static_cast<char *>(t->data) + r*t->nb[1] + b*t->nb[2];
}

static void test(ggml_type type, bool strided, bool inplace, int threads) {
    constexpr int width = 33, rows = 7, batches = 2;
    const float inf = std::numeric_limits<float>::infinity();
    const float max = std::numeric_limits<float>::max();
    const std::vector<float> cases = {
        -inf, -max, -1000, -80, -30, -20, -17, -16, -10, -5, -1,
        -0.001f, -0.0f, 0.0f, 0.001f, 1, 5, 10,
        std::nextafter(20.0f, -inf), 20, std::nextafter(20.0f, inf),
        21, 80, 1000, max, inf, std::numeric_limits<float>::quiet_NaN(),
    };
    auto * ctx = ggml_init({1024*1024, nullptr, false});
    require(ctx != nullptr, "context allocation failed");
    auto * base = ggml_new_tensor_3d(ctx, type, width, rows + (strided ? 3 : 0), batches);
    auto * input = strided ? ggml_view_3d(ctx, base, width, rows, batches, base->nb[1], base->nb[2], 0) : base;
    std::vector<float> stored(width*rows*batches);
    for (int b = 0; b < batches; ++b) {
        for (int r = 0; r < rows; ++r) {
            float * values = stored.data() + (b*rows + r)*width;
            for (int i = 0; i < width; ++i) {
                values[i] = rounded(cases[(i + r + b)%cases.size()], type);
            }
            if (type == GGML_TYPE_F32) {
                std::memcpy(row(input, r, b), values, width*sizeof(float));
            } else {
                ggml_get_type_traits(type)->from_float_ref(values, row(input, r, b), width);
            }
        }
    }
    auto * output = inplace ? ggml_softplus_inplace(ctx, input) : ggml_softplus(ctx, input);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    require(ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS, "compute failed");
    for (int b = 0; b < batches; ++b) {
        for (int r = 0; r < rows; ++r) {
            float actual[width];
            if (type == GGML_TYPE_F32) {
                std::memcpy(actual, row(output, r, b), sizeof(actual));
            } else {
                ggml_get_type_traits(type)->to_float(row(output, r, b), actual, width);
            }
            for (int i = 0; i < width; ++i) {
                const float x = stored[(b*rows + r)*width + i];
                if (std::isnan(x)) {
                    require(std::isnan(actual[i]), "NaN propagation changed");
                    continue;
                }
                // Double-precision reference avoids overflow for large positive
                // inputs and retains small positive results for negative inputs.
                const double reference = x > 0 ? double(x) + std::log1p(std::exp(-double(x))) :
                                                            std::log1p(std::exp(double(x)));
                const float expected = rounded(float(reference), type);
                if (expected == 0 || std::isinf(expected) || x > 20) {
                    require(actual[i] == expected, "zero, infinity or positive fast path changed");
                    if (expected == 0) { require(!std::signbit(actual[i]), "softplus returned negative zero"); }
                } else {
                    const double tolerance = type == GGML_TYPE_F32 ? 4e-7 : type == GGML_TYPE_F16 ? 1e-3 : 8e-3;
                    if (!(actual[i] > 0 && std::abs(double(actual[i]) - expected) <= tolerance*expected)) {
                        std::fprintf(stderr, "%s strided=%d inplace=%d threads=%d x=%g actual=%.9g expected=%.9g\n",
                                     ggml_type_name(type), strided, inplace, threads, x, actual[i], expected);
                        throw std::runtime_error("softplus lost accuracy against the double-precision reference");
                    }
                }
            }
        }
    }
    ggml_free(ctx);
}

int main() {
    try {
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
            for (bool strided : {false, true}) {
                for (bool inplace : {false, true}) {
                    for (int threads : {1, 4, 16}) { test(type, strided, inplace, threads); }
                }
            }
        }
        std::puts("CPU softplus: 36 cases passed (tails, threshold, special values, types, strides and threads)");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
