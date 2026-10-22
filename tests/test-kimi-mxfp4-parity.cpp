// Bitwise parity of the MXFP4 x Q8_0 CUDA component with the linked CPU
// backend's ggml_vec_dot_mxfp4_q8_0, plus the host clamped-SwiGLU/Q8_0 bridge
// the VRAM tier runs between its two matvec stages. Synthetic MXFP4 weights at
// the DeepSeek-V4.1 expert shapes; no model or VRAM tier is involved.
// usage: test-kimi-mxfp4-parity path/to/libkimi-mxfp4-parity.so
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-kimi-mxfp4-parity.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void need(bool ok, const char * why) { if (!ok) { throw std::runtime_error(why); } }
static size_t checks = 0, floats = 0, bad = 0;

static void equal(const char * phase, const float * a, const float * b, size_t n) {
    size_t unequal = 0, first = n;
    double max_abs = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::memcmp(a + i, b + i, 4)) { ++unequal; first = std::min(first, i); }
        need(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite test output");
        max_abs = std::max(max_abs, std::abs(double(a[i]) - double(b[i])));
    }
    ++checks; floats += n; bad += unequal;
    std::printf("{\"phase\":\"%s\",\"elements\":%zu,\"unequal_bits\":%zu,\"max_abs\":%.12g,\"first\":%zu}\n",
        phase, n, unequal, max_abs, first);
    if (unequal) { std::printf("DIFF %s %zu cpu=%a gpu=%a\n", phase, first, a[first], b[first]); }
    std::fflush(stdout);
}

static std::vector<uint8_t> quant_q8_0(const float * x, size_t n) {
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q8_0, int64_t(n)));
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(x, q.data(), int64_t(n));
    return q;
}

static std::vector<uint8_t> mxfp4_weights(std::mt19937 & rng, int columns, int rows, int variant) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> w(size_t(columns) * rows);
    for (int r = 0; r < rows; ++r) {
        const float scale = variant == 0 ? 0.02f : std::ldexp(1.0f, int(r % 17) - 12);
        for (int c = 0; c < columns; ++c) { w[size_t(r) * columns + c] = normal(rng) * scale; }
    }
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_MXFP4, columns) * size_t(rows));
    need(ggml_quantize_chunk(GGML_TYPE_MXFP4, w.data(), q.data(), 0, rows, columns, nullptr) == q.size(), "MXFP4 quantization bytes");
    return q;
}

static std::vector<float> cpu_dot(const std::vector<uint8_t> & w, int columns, int rows, const uint8_t * q) {
    const auto fn = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4)->vec_dot;
    const size_t stride = ggml_row_size(GGML_TYPE_MXFP4, columns);
    std::vector<float> y(static_cast<size_t>(rows), 0.0f);
    for (int r = 0; r < rows; ++r) { fn(columns, &y[r], 0, w.data() + size_t(r) * stride, 0, q, 0, 1); }
    return y;
}

// kimi-cpu-sink.cpp compute_rows, SWIGLU_CLAMP branch (also in the tier bridge).
static std::vector<float> swiglu_clamp(const std::vector<float> & up, const std::vector<float> & gate, float limit) {
    std::vector<float> out(up.size());
    for (size_t row = 0; row < up.size(); ++row) {
        const float u = up[row], g = gate[row];
        const float gate_value = std::min(g, limit);
        const float up_value = std::clamp(u, -limit, limit);
        out[row] = gate_value / (1.f + expf(-gate_value)) * up_value;
    }
    return out;
}

struct gpu {
    const kimi_mxfp4_gpu_api_v1 * api = nullptr;
    kimi_mxfp4_gpu * handle = nullptr;
    std::vector<void *> held;
    explicit gpu(const char * path) {
        void * so = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        need(so, dlerror());
        auto get = reinterpret_cast<const kimi_mxfp4_gpu_api_v1 * (*)()>(dlsym(so, "kimi_mxfp4_gpu_get_api_v1"));
        need(get, "API export");
        api = get();
        need(api->version == 1 && api->job_size == sizeof(kimi_mxfp4_gpu_job), "ABI");
        char err[256] = {};
        const kimi_mxfp4_gpu_config odd_columns{0, 32, 5000, 5120};
        need(!api->create(&odd_columns, err, sizeof(err)) && err[0], "reject non-64-column capacity");
        const kimi_mxfp4_gpu_config cfg{0, 32, 5120, 5120};
        handle = api->create(&cfg, err, sizeof(err));
        need(handle, err);
        std::printf("{\"host_scratch_bytes\":%zu,\"device_scratch_bytes\":%zu}\n", api->host_bytes(handle), api->device_bytes(handle));
    }
    ~gpu() {
        if (!handle) { return; }
        for (void * p : held) { api->release_weights(handle, p); }
        api->destroy(handle);
    }
    void clear() {
        need(api->wait(handle), api->error(handle));
        for (void * p : held) { need(api->release_weights(handle, p), api->error(handle)); }
        held.clear();
    }
    void * upload(const std::vector<uint8_t> & w) {
        void * p = nullptr;
        need(api->upload(handle, w.data(), w.size(), &p), api->error(handle));
        held.push_back(p);
        return p;
    }
    std::vector<std::vector<float>> run(const std::vector<kimi_mxfp4_gpu_job> & jobs, const std::vector<uint8_t> & q) {
        need(api->submit(handle, jobs.data(), int(jobs.size()), q.data(), q.size()), api->error(handle));
        need(!api->submit(handle, jobs.data(), int(jobs.size()), q.data(), q.size()), "reject in-flight reuse");
        need(api->wait(handle), api->error(handle));
        std::vector<std::vector<float>> out;
        for (size_t i = 0; i < jobs.size(); ++i) {
            int n = 0;
            const float * p = api->result(handle, int(i), &n);
            need(p && n == jobs[i].rows, "output shape");
            out.emplace_back(p, p + n);
        }
        return out;
    }
};

static kimi_mxfp4_gpu_job job(void * w, size_t bytes, int columns, int rows, size_t offset = 0) {
    return {w, bytes, ggml_row_size(GGML_TYPE_MXFP4, columns), offset, columns, rows};
}

int main(int argc, char ** argv) {
    try {
        need(argc == 2, "usage: test-kimi-mxfp4-parity path/to/libkimi-mxfp4-parity.so");
        ggml_cpu_init();
        gpu g(argv[1]);
        std::mt19937 rng(20260914);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        const int latent = 5120, hidden = 2304;

        // Expert shapes, two weight-scale families, several activation regimes,
        // then the host bridge and the down stage on each hidden vector.
        for (int variant = 0; variant < 2; ++variant) {
            g.clear();
            auto wu = mxfp4_weights(rng, latent, hidden, variant);
            auto wg = mxfp4_weights(rng, latent, hidden, variant);
            auto wd = mxfp4_weights(rng, hidden, latent, variant);
            void * du = g.upload(wu), * dg = g.upload(wg), * dd = g.upload(wd);
            for (int input_case = 0; input_case < 6; ++input_case) {
                std::vector<float> x(latent);
                for (int i = 0; i < latent; ++i) {
                    x[i] = input_case == 0 ? 0.0f : input_case == 1 ? normal(rng) : input_case == 2 ? normal(rng) * 64 :
                        input_case == 3 ? (i % 2 ? 1.0f : -1.0f) : input_case == 4 ? normal(rng) * 1e-3f :
                        (i / 32 % 3 == 0 ? 0.0f : normal(rng));
                }
                auto q = quant_q8_0(x.data(), x.size());
                auto u = cpu_dot(wu, latent, hidden, q.data());
                auto v = cpu_dot(wg, latent, hidden, q.data());
                auto uv = g.run({job(du, wu.size(), latent, hidden), job(dg, wg.size(), latent, hidden)}, q);
                equal("up", u.data(), uv[0].data(), u.size());
                equal("gate", v.data(), uv[1].data(), v.size());
                for (float limit : {10.0f, 0.75f}) {
                    auto h = swiglu_clamp(u, v, limit), gh = swiglu_clamp(uv[0], uv[1], limit);
                    equal("host_swiglu_clamp", h.data(), gh.data(), h.size());
                    auto qh = quant_q8_0(h.data(), h.size()), gqh = quant_q8_0(gh.data(), gh.size());
                    need(qh == gqh, "hidden Q8_0 bytes differ");
                    auto down = cpu_dot(wd, hidden, latent, qh.data());
                    auto gd = g.run({job(dd, wd.size(), hidden, latent)}, gqh);
                    equal("down", down.data(), gd[0].data(), down.size());
                }
            }
        }

        // E8M0 scales at the edges (denormal 2^-128 and 2^-127, around 1, large)
        // with a large-magnitude Q8_0 input.
        {
            g.clear();
            const int columns = 256, rows = 8;
            auto w = mxfp4_weights(rng, columns, rows, 0);
            const size_t stride = ggml_row_size(GGML_TYPE_MXFP4, columns);
            const uint8_t exponents[] = {0, 1, 2, 60, 127, 128, 140, 170};
            for (int r = 0; r < rows; ++r) {
                for (int b = 0; b < columns / 32; ++b) { w[size_t(r) * stride + size_t(b) * 17] = exponents[(r + b) % 8]; }
            }
            void * dw = g.upload(w);
            for (float magnitude : {1.0f, 1e3f, 1e6f}) {
                std::vector<float> x(columns);
                for (auto & value : x) { value = normal(rng) * magnitude; }
                auto q = quant_q8_0(x.data(), x.size());
                auto expected = cpu_dot(w, columns, rows, q.data());
                auto got = g.run({job(dw, w.size(), columns, rows)}, q);
                equal("e8m0_edges", expected.data(), got[0].data(), expected.size());
            }
        }

        // Batched down jobs with per-hit Q8_0 offsets, rows that end inside a
        // CUDA block, job counts up to the cap, and rejected layouts.
        {
            g.clear();
            auto wd = mxfp4_weights(rng, hidden, latent, 0);
            void * dd = g.upload(wd);
            const int hits = 6;
            const size_t hidden_bytes = ggml_row_size(GGML_TYPE_Q8_0, hidden);
            std::vector<uint8_t> qs(size_t(hits) * hidden_bytes);
            std::vector<std::vector<float>> expected;
            std::vector<kimi_mxfp4_gpu_job> jobs;
            for (int i = 0; i < hits; ++i) {
                std::vector<float> a(hidden);
                for (auto & value : a) { value = normal(rng) * float(i + 1); }
                auto qi = quant_q8_0(a.data(), a.size());
                std::memcpy(qs.data() + size_t(i) * hidden_bytes, qi.data(), hidden_bytes);
                const int rows = 1 + (i * 853) % latent;
                expected.push_back(cpu_dot(wd, hidden, rows, qi.data()));
                jobs.push_back(job(dd, wd.size(), hidden, rows, size_t(i) * hidden_bytes));
            }
            auto got = g.run(jobs, qs);
            for (int i = 0; i < hits; ++i) { equal("down_batch", expected[i].data(), got[i].data(), expected[i].size()); }
            const auto full = cpu_dot(wd, hidden, latent, qs.data());
            for (int n : {1, 2, 3, 5, 8, 16, 32}) {
                std::vector<kimi_mxfp4_gpu_job> many;
                for (int i = 0; i < n; ++i) { many.push_back(job(dd, wd.size(), hidden, 1 + (i * 17) % latent)); }
                auto ys = g.run(many, qs);
                for (int i = 0; i < n; ++i) { equal("job_shapes", full.data(), ys[i].data(), ys[i].size()); }
            }
            auto invalid = job(dd, wd.size(), hidden + 1, 1);
            need(!g.api->submit(g.handle, &invalid, 1, qs.data(), qs.size()), "reject columns");
            invalid = job(dd, wd.size(), 96, 1);
            need(!g.api->submit(g.handle, &invalid, 1, qs.data(), qs.size()), "reject an odd block count");
            invalid = job(dd, 1, hidden, 1);
            need(!g.api->submit(g.handle, &invalid, 1, qs.data(), qs.size()), "reject weight range");
            invalid = job(dd, wd.size(), hidden, 1, 1);
            need(!g.api->submit(g.handle, &invalid, 1, qs.data(), qs.size()), "reject Q8_0 alignment");
            invalid = job(dd, wd.size(), hidden, 1);
            need(!g.api->submit(g.handle, &invalid, 33, qs.data(), qs.size()), "reject job cap");
            need(g.api->query(g.handle) == 1, "idle after rejected requests");
        }

        std::printf("{\"final\":true,\"checks\":%zu,\"floats\":%zu,\"unequal\":%zu}\n", checks, floats, bad);
        return bad ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "MXFP4 parity test: %s\n", e.what());
        return 2;
    }
}
