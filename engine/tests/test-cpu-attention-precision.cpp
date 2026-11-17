// SPDX-License-Identifier: Apache-2.0
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

struct shape {
    int dk, dv, nq, nk, heads, kv_heads, batches;
    bool padded, masked, sinks;
    float bias, cap;
};

static char * row(ggml_tensor * t, int n, int h, int b) {
    return static_cast<char *>(t->data) + n*t->nb[1] + h*t->nb[2] + b*t->nb[3];
}

static ggml_tensor * input(ggml_context * ctx, ggml_type type, int d, int n, int h, int b,
                          bool padded, int seed, std::vector<float> & stored) {
    auto * base = ggml_new_tensor_4d(ctx, type, d, n + (padded ? 3 : 0), h, b);
    auto * t = padded ? ggml_view_4d(ctx, base, d, n, h, b, base->nb[1], base->nb[2], base->nb[3], 0) : base;
    stored.resize(size_t(d)*n*h*b);
    std::vector<float> values(d);
    uint32_t state = seed;
    for (int ib = 0; ib < b; ++ib) {
        for (int ih = 0; ih < h; ++ih) {
            for (int in = 0; in < n; ++in) {
                for (float & x : values) {
                    state = state*1664525u + 1013904223u;
                    x = (float(state >> 8)/8388608.0f - 1.0f)*1.7f;
                }
                float * out = stored.data() + size_t(((ib*h + ih)*n + in)*d);
                if (type == GGML_TYPE_F32) {
                    std::memcpy(row(t, in, ih, ib), values.data(), d*sizeof(float));
                    std::copy(values.begin(), values.end(), out);
                } else {
                    const auto * traits = ggml_get_type_traits(type);
                    traits->from_float_ref(values.data(), row(t, in, ih, ib), d);
                    traits->to_float(row(t, in, ih, ib), out, d);
                }
            }
        }
    }
    return t;
}

// Independent, two-pass softmax and double-precision dot products over the
// values actually stored in K/V. This does not reuse the attention kernels.
static std::vector<double> reference(const shape & s, const std::vector<float> & q,
                                    const std::vector<float> & k, const std::vector<float> & v,
                                    ggml_tensor * mask, ggml_tensor * sinks, float scale) {
    std::vector<double> out(size_t(s.dv)*s.heads*s.nq*s.batches);
    const int head_pow2 = 1 << int(std::floor(std::log2(s.heads)));
    for (int b = 0; b < s.batches; ++b) {
        for (int h = 0; h < s.heads; ++h) {
            const int kh = h/(s.heads/s.kv_heads);
            const double slope = s.bias == 0 ? 1.0 : std::pow(2.0, h < head_pow2 ?
                -double(s.bias)*(h + 1)/head_pow2 : -double(s.bias)*(2*(h - head_pow2) + 1)/(2*head_pow2));
            for (int n = 0; n < s.nq; ++n) {
                std::vector<double> scores(s.nk);
                double maximum = sinks ? static_cast<float *>(sinks->data)[h] : -INFINITY;
                for (int j = 0; j < s.nk; ++j) {
                    double dot = 0;
                    for (int d = 0; d < s.dk; ++d) {
                        dot += double(q[((b*s.heads + h)*s.nq + n)*s.dk + d]) *
                                      k[((b*s.kv_heads + kh)*s.nk + j)*s.dk + d];
                    }
                    double score = dot*scale;
                    if (s.cap) { score = s.cap*std::tanh(score/s.cap); }
                    if (mask) { score += slope*ggml_fp16_to_fp32(static_cast<ggml_fp16_t *>(mask->data)[n*s.nk + j]); }
                    scores[j] = score;
                    maximum = std::max(maximum, score);
                }
                if (maximum == -INFINITY) { continue; }
                double sum = sinks ? std::exp(static_cast<float *>(sinks->data)[h] - maximum) : 0;
                for (double & score : scores) { score = std::exp(score - maximum); sum += score; }
                for (int d = 0; d < s.dv; ++d) {
                    double value = 0;
                    for (int j = 0; j < s.nk; ++j) {
                        value += scores[j]*v[((b*s.kv_heads + kh)*s.nk + j)*s.dv + d];
                    }
                    out[((b*s.nq + n)*s.heads + h)*s.dv + d] = value/sum;
                }
            }
        }
    }
    return out;
}

static void test(const shape & s, ggml_type kt, ggml_type vt, int threads, int id, const char * record) {
    auto * ctx = ggml_init({64*1024*1024, nullptr, false});
    require(ctx != nullptr, "context allocation failed");
    std::vector<float> qv, kv, vv;
    auto * q = input(ctx, GGML_TYPE_F32, s.dk, s.nq, s.heads, s.batches, s.padded, 13, qv);
    auto * k = input(ctx, kt, s.dk, s.nk, s.kv_heads, s.batches, s.padded, 29, kv);
    auto * v = input(ctx, vt, s.dv, s.nk, s.kv_heads, s.batches, s.padded, 71, vv);
    auto * mask = s.masked ? ggml_new_tensor_2d(ctx, GGML_TYPE_F16, s.nk, s.nq) : nullptr;
    if (mask) {
        for (int n = 0; n < s.nq; ++n) {
            for (int j = 0; j < s.nk; ++j) {
                // Include a fully masked query as well as finite bias and a causal tail.
                const float x = n == s.nq - 1 || j > s.nk - s.nq + n ? -INFINITY : -0.125f*(j%7);
                static_cast<ggml_fp16_t *>(mask->data)[n*s.nk + j] = ggml_fp32_to_fp16(x);
            }
        }
    }
    auto * sinks = s.sinks ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.heads) : nullptr;
    if (sinks) {
        for (int h = 0; h < s.heads; ++h) { static_cast<float *>(sinks->data)[h] = 0.1f*h - 1; }
    }
    const float scale = 1.0f/std::sqrt(float(s.dk));
    std::vector<float> defaults;
    for (const auto precision : {GGML_PREC_DEFAULT, GGML_PREC_F32}) {
        auto * out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, s.bias, s.cap);
        ggml_flash_attn_ext_set_prec(out, precision);
        if (sinks) { ggml_flash_attn_ext_add_sinks(out, sinks); }
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        require(ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS, "compute failed");
        const auto * actual = static_cast<float *>(out->data);
        for (int64_t i = 0; i < ggml_nelements(out); ++i) { require(std::isfinite(actual[i]), "non-finite attention output"); }
        if (record) {
            const std::string path = std::string(record) + "/" + std::to_string(id) +
                (precision == GGML_PREC_DEFAULT ? "-default.bin" : "-f32.bin");
            std::ofstream file(path, std::ios::binary);
            file.write(static_cast<const char *>(out->data), ggml_nbytes(out));
            require(bool(file), "cannot write comparison artifact");
        }
        if (precision == GGML_PREC_DEFAULT) {
            defaults.assign(actual, actual + ggml_nelements(out));
        } else if (kt != GGML_TYPE_F16 && vt != GGML_TYPE_F16) {
            require(std::memcmp(defaults.data(), actual, ggml_nbytes(out)) == 0, "unaffected KV types changed with precision");
        }
        if (!record && precision == GGML_PREC_F32 && (kt == GGML_TYPE_F16 || kt == GGML_TYPE_F32) &&
                                                    (vt == GGML_TYPE_F16 || vt == GGML_TYPE_F32)) {
            const auto expected = reference(s, qv, kv, vv, mask, sinks, scale);
            double max_error = 0;
            for (size_t i = 0; i < expected.size(); ++i) { max_error = std::max(max_error, std::abs(actual[i] - expected[i])); }
            if (max_error > 1e-5) {
                std::fprintf(stderr, "case %d K=%s V=%s threads=%d max_error=%.9g\n", id,
                             ggml_type_name(kt), ggml_type_name(vt), threads, max_error);
            }
            require(max_error <= 1e-5, "FP32 attention disagrees with independent reference");
        }
    }
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    try {
        require(argc == 1 || (argc == 3 && std::strcmp(argv[1], "--record") == 0), "usage: test-cpu-attention-precision [--record existing-directory]");
        const char * record = argc == 3 ? argv[2] : nullptr;
        // Vector decode, batched/GQA strided vectors, split-KV decode, tiled
        // prefill, and unequal K/V widths. No model or GPU is required.
        const shape shapes[] = {
            {64, 64, 1, 17, 8, 2, 1, false, false, false, 0, 0},
            {128, 64, 3, 33, 6, 2, 2, true, true, true, 4, 2},
            {512, 512, 1, 513, 8, 1, 1, false, false, true, 0, 0},
            {64, 128, 16, 67, 8, 2, 1, true, true, false, 0, 0},
            {64, 64, 16, 67, 8, 1, 2, false, true, true, 3, 4},
        };
        int id = 0;
        for (const auto & s : shapes) {
            for (const auto kt : {GGML_TYPE_F16, GGML_TYPE_F32}) {
                for (const auto vt : {GGML_TYPE_F16, GGML_TYPE_F32}) {
                    for (int threads : {1, 4, 16}) { test(s, kt, vt, threads, id++, record); }
                }
            }
        }
        for (const auto type : {GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
            for (int threads : {1, 4}) { test(shapes[1], type, type, threads, id++, record); }
        }
        // The CPU tiled path starts at 64 queries. Include a partial final
        // tile, non-power-of-two head count, and a fully masked query.
        const shape tiled[] = {
            {64, 128, 65, 67, 6, 2, 1, true, true, true, 3, 4},
            {128, 64, 64, 71, 8, 2, 2, false, true, false, 0, 0},
        };
        for (const auto & s : tiled) {
            for (const auto kt : {GGML_TYPE_F16, GGML_TYPE_F32}) {
                for (const auto vt : {GGML_TYPE_F16, GGML_TYPE_F32}) {
                    for (int threads : {1, 4, 16}) { test(s, kt, vt, threads, id++, record); }
                }
            }
        }
        std::printf("CPU attention: %d cases, DEFAULT and F32, %s\n", id, record ? "outputs recorded" : "passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
