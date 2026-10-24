// The optional MXFP4 GPU hidden chain (kimi-mxfp4-chain.cu) against the host bridge the
// VRAM tier runs by default: stage-1 matvec on the GPU with the parity kernel, the CPU
// sink's clamped SwiGLU and the CPU Q8_0 quantizer on the host, stage 2 on the GPU. The
// chain is not bitwise, so this test is statistical: synthetic MXFP4 experts at the
// DeepSeek-V4.1 shapes (5120 x 2304 up/gate, 2304 x 5120 down), two weight families,
// four input regimes, SwiGLU limits 0.75 and 10, 16 hits per layer. Per chain kernel it
// reports the relative L2 and L-inf errors of the down rows and the fraction of equal
// hidden Q8_0 bytes, and fails on non-finite rows or on errors far above float rounding
// (relative L2 above 1e-3 for the Q8_0 kernels; the float-hidden kernel skips Q8_0 and
// is only bounded by 5e-2). It then times one layer of 16 hits per path with the
// component's timing events and host clocks (no H2D pacing in this process).
// usage: test-kimi-mxfp4-chain [timing_iterations]
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-kimi-mxfp4-chain.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int latent = 5120, hidden = 2304, hits = 16, distinct = 8;
using clock_type = std::chrono::steady_clock;
double us(clock_type::time_point a, clock_type::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }
void need(bool ok, const char * why) { if (!ok) { throw std::runtime_error(why ? why : "unknown error"); } }
void need(bool ok, const std::string & why) { if (!ok) { throw std::runtime_error(why); } }

std::vector<uint8_t> q8_0(const float * x, int n) {
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q8_0, n));
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(x, q.data(), n);
    return q;
}

std::vector<uint8_t> mxfp4_matrix(std::mt19937 & rng, int columns, int rows, int family) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> w(size_t(columns) * size_t(rows));
    for (int r = 0; r < rows; ++r) {
        const float scale = family == 0 ? 0.02f : std::ldexp(1.0f, int(r % 13) - 9);
        for (int c = 0; c < columns; ++c) { w[size_t(r) * size_t(columns) + size_t(c)] = normal(rng) * scale; }
    }
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_MXFP4, columns) * size_t(rows));
    need(ggml_quantize_chunk(GGML_TYPE_MXFP4, w.data(), q.data(), 0, rows, columns, nullptr) == q.size(), "MXFP4 quantization bytes");
    return q;
}

// Layer inputs: normal, large, sparse (zero blocks) and heavy-tailed (1% outliers x30).
std::vector<float> make_input(std::mt19937 & rng, int regime) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::vector<float> x(static_cast<size_t>(latent), 0.0f);
    for (int i = 0; i < latent; ++i) {
        const float n = normal(rng);
        x[size_t(i)] = regime == 0 ? n : regime == 1 ? 8.0f * n : regime == 2 ? (i / 32 % 3 == 0 ? 0.0f : n) :
            n * (unit(rng) < 0.01f ? 30.0f : 1.0f);
    }
    return x;
}

struct component {
    const kimi_mxfp4_gpu_api_v1 * api = nullptr;
    const kimi_mxfp4_gpu_chain_api_v1 * chain_api = nullptr;
    kimi_mxfp4_gpu * handle = nullptr;
    std::vector<void *> held;
    component() {
        api = kimi_mxfp4_gpu_get_api_v1();
        chain_api = kimi_mxfp4_gpu_get_chain_api_v1();
        need(api->version == 1 && api->job_size == sizeof(kimi_mxfp4_gpu_job), "base ABI");
        need(chain_api->version == 1 && chain_api->size == sizeof(kimi_mxfp4_gpu_chain_api_v1), "chain ABI");
        char error[256] = {};
        const kimi_mxfp4_gpu_config config{0, 2 * hits, latent, latent};
        handle = api->create(&config, error, sizeof(error));
        need(handle != nullptr, error);
    }
    ~component() {
        if (!handle) { return; }
        api->wait(handle);
        for (void * p : held) { api->release_weights(handle, p); }
        api->destroy(handle);
    }
    const void * upload(const std::vector<uint8_t> & w) {
        void * p = nullptr;
        need(api->upload(handle, w.data(), w.size(), &p), api->error(handle));
        held.push_back(p);
        return p;
    }
    const float * result(int job, int rows) {
        int n = 0;
        const float * p = api->result(handle, job, &n);
        need(p && n == rows, "result shape");
        return p;
    }
};

struct host_times { double enqueue = 0, wait = 0, activation = 0, quantize = 0, copy = 0, total = 0; };
struct bridge_out { std::vector<float> down, act; std::vector<uint8_t> hidden_q8; };
struct chain_out { std::vector<float> down; std::vector<uint8_t> hidden; };

// The tier's default MXFP4 path (mxfp4-compute-adapter.cpp): GPU parity matvec, host
// SwiGLU with the CPU sink's expression, CPU Q8_0, GPU parity matvec.
bridge_out run_bridge(component & g, const std::vector<kimi_mxfp4_chain_expert> & experts, const std::vector<uint8_t> & q8,
                      float limit, host_times * t) {
    const size_t up_stride = ggml_row_size(GGML_TYPE_MXFP4, latent), down_stride = ggml_row_size(GGML_TYPE_MXFP4, hidden);
    const size_t hidden_bytes = ggml_row_size(GGML_TYPE_Q8_0, hidden);
    std::vector<kimi_mxfp4_gpu_job> jobs;
    for (int i = 0; i < hits; ++i) {
        jobs.push_back({experts[i].up, up_stride * hidden, up_stride, 0, latent, hidden});
        jobs.push_back({experts[i].gate, up_stride * hidden, up_stride, 0, latent, hidden});
    }
    if (t) { g.chain_api->set_timing_stage(g.handle, 1); }
    const auto t0 = clock_type::now();
    need(g.api->submit(g.handle, jobs.data(), int(jobs.size()), q8.data(), q8.size()), g.api->error(g.handle));
    const auto t1 = clock_type::now();
    need(g.api->wait(g.handle), g.api->error(g.handle));
    const auto t2 = clock_type::now();
    bridge_out out;
    out.act.resize(size_t(hits) * hidden);
    out.hidden_q8.resize(size_t(hits) * hidden_bytes);
    double activation_us = 0, quantize_us = 0;
    for (int i = 0; i < hits; ++i) {
        const float * up = g.result(2 * i, hidden), * gate = g.result(2 * i + 1, hidden);
        const auto a0 = clock_type::now();
        for (int row = 0; row < hidden; ++row) {
            const float gate_value = std::min(gate[row], limit);
            const float up_value = std::clamp(up[row], -limit, limit);
            out.act[size_t(i) * hidden + size_t(row)] = gate_value / (1.f + expf(-gate_value)) * up_value;
        }
        const auto a1 = clock_type::now();
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(out.act.data() + size_t(i) * hidden,
            out.hidden_q8.data() + size_t(i) * hidden_bytes, hidden);
        const auto a2 = clock_type::now();
        activation_us += us(a0, a1);
        quantize_us += us(a1, a2);
    }
    jobs.clear();
    for (int i = 0; i < hits; ++i) {
        jobs.push_back({experts[i].down, down_stride * latent, down_stride, size_t(i) * hidden_bytes, hidden, latent});
    }
    if (t) { g.chain_api->set_timing_stage(g.handle, 2); }
    const auto t3 = clock_type::now();
    need(g.api->submit(g.handle, jobs.data(), int(jobs.size()), out.hidden_q8.data(), out.hidden_q8.size()), g.api->error(g.handle));
    const auto t4 = clock_type::now();
    need(g.api->wait(g.handle), g.api->error(g.handle));
    const auto t5 = clock_type::now();
    out.down.resize(size_t(hits) * latent);
    for (int i = 0; i < hits; ++i) { std::memcpy(out.down.data() + size_t(i) * latent, g.result(i, latent), latent * sizeof(float)); }
    const auto t6 = clock_type::now();
    if (t) {
        t->enqueue += us(t0, t1) + us(t3, t4);
        t->wait += us(t1, t2) + us(t4, t5);
        t->activation += activation_us;
        t->quantize += quantize_us;
        t->copy += us(t5, t6);
        t->total += us(t0, t6);
    }
    return out;
}

chain_out run_chain(component & g, kimi_mxfp4_gpu_chain * chain, const std::vector<kimi_mxfp4_chain_expert> & experts,
                    const std::vector<uint8_t> & q8, float limit, host_times * t) {
    const auto t0 = clock_type::now();
    need(g.chain_api->submit(chain, experts.data(), hits, q8.data(), q8.size(), limit), g.api->error(g.handle));
    const auto t1 = clock_type::now();
    need(g.api->wait(g.handle), g.api->error(g.handle));
    const auto t2 = clock_type::now();
    chain_out out;
    out.down.resize(size_t(hits) * latent);
    for (int i = 0; i < hits; ++i) { std::memcpy(out.down.data() + size_t(i) * latent, g.result(i, latent), latent * sizeof(float)); }
    const auto t3 = clock_type::now();
    size_t bytes = 0;
    if (const void * h = g.chain_api->hidden(chain, &bytes)) {
        out.hidden.assign(static_cast<const uint8_t *>(h), static_cast<const uint8_t *>(h) + bytes);
    }
    if (t) {
        t->enqueue += us(t0, t1);
        t->wait += us(t1, t2);
        t->copy += us(t2, t3);
        t->total += us(t0, t3);
    }
    return out;
}

struct error_stats {
    size_t vectors = 0, nonfinite = 0, q8_bytes = 0, q8_equal_bytes = 0, q8_blocks = 0, q8_equal_blocks = 0;
    double rel_max = 0, rel_sum = 0, linf_max = 0, hidden_rel_max = 0;
    static double rel_l2(const float * c, const float * r, size_t n, double * linf_rel, bool * finite) {
        double num = 0, den = 0, linf = 0, rmax = 0;
        *finite = true;
        for (size_t k = 0; k < n; ++k) {
            if (!std::isfinite(c[k])) { *finite = false; }
            const double d = double(c[k]) - double(r[k]);
            num += d * d;
            den += double(r[k]) * double(r[k]);
            linf = std::max(linf, std::abs(d));
            rmax = std::max(rmax, std::abs(double(r[k])));
        }
        *linf_rel = rmax > 0 ? linf / rmax : (linf > 0 ? INFINITY : 0.0);
        return den > 0 ? std::sqrt(num / den) : (num > 0 ? INFINITY : 0.0);
    }
    void add_down(const float * c, const float * r, size_t n) {
        double linf = 0;
        bool finite = true;
        const double rel = rel_l2(c, r, n, &linf, &finite);
        ++vectors;
        if (!finite) { ++nonfinite; return; }
        rel_max = std::max(rel_max, rel);
        rel_sum += rel;
        linf_max = std::max(linf_max, linf);
    }
    void add_q8(const uint8_t * c, const uint8_t * r, size_t blocks) {
        for (size_t b = 0; b < blocks; ++b) {
            bool same = true;
            for (size_t k = 0; k < 34; ++k) {
                ++q8_bytes;
                if (c[b * 34 + k] == r[b * 34 + k]) { ++q8_equal_bytes; } else { same = false; }
            }
            ++q8_blocks;
            q8_equal_blocks += same;
        }
    }
    void add_hidden(const float * c, const float * r, size_t n) {
        double linf = 0;
        bool finite = true;
        hidden_rel_max = std::max(hidden_rel_max, rel_l2(c, r, n, &linf, &finite));
        if (!finite) { ++nonfinite; }
    }
};

const char * kernel_name(int kernel) {
    return kernel == KIMI_MXFP4_CHAIN_PARITY_Q8 ? "parity_q8" : kernel == KIMI_MXFP4_CHAIN_ROWS_Q8 ? "rows_q8" : "rows_float";
}

void print_timing(const char * path, int iterations, const kimi_mxfp4_gpu_timing & a, const kimi_mxfp4_gpu_timing & b, const host_times & h) {
    const double n = iterations;
    std::printf("{\"timing\":\"%s\",\"iterations\":%d,\"hits\":%d,\"gpu_submits_per_layer\":%.1f,\"ms_per_layer\":{"
        "\"gpu_h2d\":%.4f,\"gpu_stage1\":%.4f,\"gpu_activation\":%.4f,\"gpu_quantize\":%.4f,\"gpu_stage2\":%.4f,\"gpu_d2h\":%.4f,"
        "\"host_h2d_call\":%.4f,\"host_enqueue\":%.4f,\"host_wait\":%.4f,\"host_activation\":%.4f,\"host_quantize\":%.4f,"
        "\"host_copy\":%.4f,\"host_total\":%.4f}}\n",
        path, iterations, hits, double(b.submits - a.submits) / n,
        (b.h2d_ms - a.h2d_ms) / n, (b.stage1_ms - a.stage1_ms) / n, (b.activation_ms - a.activation_ms) / n,
        (b.quantize_ms - a.quantize_ms) / n, (b.stage2_ms - a.stage2_ms) / n, (b.d2h_ms - a.d2h_ms) / n,
        (b.h2d_call_us - a.h2d_call_us) / 1e3 / n, h.enqueue / 1e3 / n, h.wait / 1e3 / n, h.activation / 1e3 / n,
        h.quantize / 1e3 / n, h.copy / 1e3 / n, h.total / 1e3 / n);
    std::fflush(stdout);
}
}

int main(int argc, char ** argv) {
    try {
        need(argc <= 2, "usage: test-kimi-mxfp4-chain [timing_iterations]");
        const int iterations = argc == 2 ? std::max(1, std::atoi(argv[1])) : 20;
        ggml_cpu_init();
        component g;
        std::mt19937 rng(20260915);
        std::vector<std::vector<kimi_mxfp4_chain_expert>> family(2);
        for (int fam = 0; fam < 2; ++fam) {
            for (int e = 0; e < distinct; ++e) {
                const void * up = g.upload(mxfp4_matrix(rng, latent, hidden, fam));
                const void * gate = g.upload(mxfp4_matrix(rng, latent, hidden, fam));
                const void * down = g.upload(mxfp4_matrix(rng, hidden, latent, fam));
                family[size_t(fam)].push_back({up, gate, down});
            }
        }
        const auto layer = [&](int fam, int shift) {
            std::vector<kimi_mxfp4_chain_expert> v;
            for (int i = 0; i < hits; ++i) { v.push_back(family[size_t(fam)][size_t((i * 3 + shift) % distinct)]); }
            return v;
        };

        // Reference rows once per case, shared by every kernel.
        struct case_def { int fam, regime; float limit; std::vector<uint8_t> q8; bridge_out ref; };
        std::vector<case_def> cases;
        for (int fam = 0; fam < 2; ++fam) {
            for (int regime = 0; regime < 4; ++regime) {
                for (float limit : {0.75f, 10.0f}) {
                    const auto x = make_input(rng, regime);
                    case_def c{fam, regime, limit, q8_0(x.data(), latent), {}};
                    c.ref = run_bridge(g, layer(fam, int(cases.size())), c.q8, limit, nullptr);
                    cases.push_back(std::move(c));
                }
            }
        }

        int failures = 0;
        for (int kernel = KIMI_MXFP4_CHAIN_PARITY_Q8; kernel <= KIMI_MXFP4_CHAIN_ROWS_FLOAT; ++kernel) {
            kimi_mxfp4_gpu_chain * chain = g.chain_api->create(g.handle, latent, hidden, hits, kernel, 1);
            need(chain != nullptr, g.api->error(g.handle));
            error_stats s;
            for (size_t ci = 0; ci < cases.size(); ++ci) {
                const auto & c = cases[ci];
                const auto out = run_chain(g, chain, layer(c.fam, int(ci)), c.q8, c.limit, nullptr);
                for (int i = 0; i < hits; ++i) {
                    s.add_down(out.down.data() + size_t(i) * latent, c.ref.down.data() + size_t(i) * latent, latent);
                }
                if (kernel == KIMI_MXFP4_CHAIN_ROWS_FLOAT) {
                    need(out.hidden.size() == c.ref.act.size() * sizeof(float), "kept float hidden size");
                    std::vector<float> h(c.ref.act.size());
                    std::memcpy(h.data(), out.hidden.data(), out.hidden.size());
                    s.add_hidden(h.data(), c.ref.act.data(), h.size());
                } else {
                    need(out.hidden.size() == c.ref.hidden_q8.size(), "kept hidden Q8_0 size");
                    s.add_q8(out.hidden.data(), c.ref.hidden_q8.data(), out.hidden.size() / 34);
                }
            }
            need(g.chain_api->destroy(chain) == 1, "chain destroy");
            const double bound = kernel == KIMI_MXFP4_CHAIN_ROWS_FLOAT ? 5e-2 : 1e-3;
            const bool ok = s.nonfinite == 0 && s.rel_max <= bound;
            failures += !ok;
            std::printf("{\"kernel\":\"%s\",\"cases\":%zu,\"down_vectors\":%zu,\"nonfinite\":%zu,\"down_rel_l2_max\":%.3e,"
                "\"down_rel_l2_mean\":%.3e,\"down_linf_rel_max\":%.3e,\"hidden_q8_equal_bytes\":%.6f,\"hidden_q8_equal_blocks\":%.6f,"
                "\"hidden_float_rel_l2_max\":%.3e,\"bound\":%.0e,\"pass\":%s}\n",
                kernel_name(kernel), cases.size(), s.vectors, s.nonfinite, s.rel_max, s.vectors ? s.rel_sum / double(s.vectors) : 0.0,
                s.linf_max, s.q8_bytes ? double(s.q8_equal_bytes) / double(s.q8_bytes) : -1.0,
                s.q8_blocks ? double(s.q8_equal_blocks) / double(s.q8_blocks) : -1.0, s.hidden_rel_max, bound, ok ? "true" : "false");
            std::fflush(stdout);
        }

        // One layer of 16 hits per path, timing events on the base handle.
        need(g.chain_api->set_timing(g.handle, 1) == 1, g.api->error(g.handle));
        const auto timed = layer(0, 1);
        const auto tx = make_input(rng, 3);
        const auto tq = q8_0(tx.data(), latent);
        {
            for (int w = 0; w < 3; ++w) { run_bridge(g, timed, tq, 10.0f, nullptr); }
            kimi_mxfp4_gpu_timing a{}, b{};
            host_times h;
            g.chain_api->timing(g.handle, &a);
            for (int it = 0; it < iterations; ++it) { run_bridge(g, timed, tq, 10.0f, &h); }
            g.chain_api->timing(g.handle, &b);
            print_timing("bridge_parity", iterations, a, b, h);
        }
        for (int kernel = KIMI_MXFP4_CHAIN_PARITY_Q8; kernel <= KIMI_MXFP4_CHAIN_ROWS_FLOAT; ++kernel) {
            kimi_mxfp4_gpu_chain * chain = g.chain_api->create(g.handle, latent, hidden, hits, kernel, 0);
            need(chain != nullptr, g.api->error(g.handle));
            for (int w = 0; w < 3; ++w) { run_chain(g, chain, timed, tq, 10.0f, nullptr); }
            kimi_mxfp4_gpu_timing a{}, b{};
            host_times h;
            g.chain_api->timing(g.handle, &a);
            for (int it = 0; it < iterations; ++it) { run_chain(g, chain, timed, tq, 10.0f, &h); }
            g.chain_api->timing(g.handle, &b);
            print_timing((std::string("chain_") + kernel_name(kernel)).c_str(), iterations, a, b, h);
            need(g.chain_api->destroy(chain) == 1, "chain destroy");
        }
        std::printf("{\"final\":true,\"failures\":%d}\n", failures);
        return failures ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "MXFP4 chain test: %s\n", e.what());
        return 2;
    }
}
