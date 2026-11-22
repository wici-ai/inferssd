// SPDX-License-Identifier: Apache-2.0
// An op that reads a cache a GPU holds (a KV or recurrent state the context keeps in that GPU's buffer) runs on that
// GPU, wherever it falls in the graph. Placed by its neighbours instead, a row read of the cache between two CPU nodes
// (the CPU-run experts of a MoE layer, then a row lookup in a host table) went to the CPU, where no decode graph runs
// it, and a prompt graph that did so was refused at every ubatch size. A cache in host memory is placed as before.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace {

constexpr int channels = 4;
constexpr int rows = 3; // the cache's rows, one a sequence

// A second backend, as the scheduler sees a GPU: a CPU behind its own GPU-typed device and a buffer type only it can use.
ggml_backend_buffer_type gpu_buffer_type;

ggml_backend_buffer_type_t gpu_get_buffer_type(ggml_backend_dev_t) { return &gpu_buffer_type; }
bool gpu_is_host(ggml_backend_buffer_type_t) { return false; }
bool gpu_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) { return buft == &gpu_buffer_type; }
bool shared_gpu_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return buft == &gpu_buffer_type || buft == ggml_backend_cpu_buffer_type();
}
enum ggml_backend_dev_type gpu_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }

struct fake_gpu {
    ggml_backend_device device;
    ggml_backend backend;

    explicit fake_gpu(ggml_backend_t cpu, bool shared = false) {
        device = *ggml_backend_get_device(cpu);
        device.iface.get_buffer_type = gpu_get_buffer_type;
        device.iface.supports_buft = shared ? shared_gpu_supports_buft : gpu_supports_buft;
        device.iface.get_type = gpu_type;
        backend = *cpu;
        backend.device = &device;
    }
};

// x -> a = 2x (on the CPU) -> r = the cache's row ids[0] -> t = 3x (on the CPU) -> out = a + r + t
struct cache_read {
    ggml_context * ctx = nullptr;
    ggml_tensor * x = nullptr, * ids = nullptr, * a = nullptr, * r = nullptr, * t = nullptr;
    ggml_cgraph * graph = nullptr;

    explicit cache_read(ggml_tensor * cache) {
        ggml_init_params params{ ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false), nullptr, true };
        ctx = ggml_init(params);
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels, 1);
        ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_input(x);
        ggml_set_input(ids);
        a = ggml_scale(ctx, x, 2.0f);
        r = ggml_get_rows(ctx, cache, ids);
        t = ggml_scale(ctx, x, 3.0f);
        graph = ggml_new_graph_custom(ctx, 16, false);
        // the graph's order: a, then the cache's row, then t
        ggml_build_forward_expand(graph, a);
        ggml_build_forward_expand(graph, r);
        ggml_build_forward_expand(graph, t);
        ggml_build_forward_expand(graph, ggml_add(ctx, ggml_add(ctx, a, r), t));
    }
    ~cache_read() { ggml_free(ctx); }
};

bool require(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); }
    return ok;
}

// Places the graph with a and t on the CPU, runs it and checks the output: the backend the cache's row read ran on.
ggml_backend_t place_and_run(ggml_backend_sched_t sched, ggml_backend_t cpu, ggml_tensor * cache, bool & ok) {
    cache_read g(cache);
    ggml_backend_sched_reset(sched);
    ggml_backend_sched_set_tensor_backend(sched, g.a, cpu);
    ggml_backend_sched_set_tensor_backend(sched, g.t, cpu);
    if (!require(ggml_backend_sched_alloc_graph(sched, g.graph), "the graph was placed")) {
        ok = false;
        return nullptr;
    }
    ggml_backend_t placed = ggml_backend_sched_get_tensor_backend(sched, g.r);
    const std::vector<float> x = { 1.0f, 2.0f, 3.0f, 4.0f };
    const int32_t row = 2;
    ggml_backend_tensor_set(g.x, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(g.ids, &row, 0, sizeof(row));
    ok = require(ggml_backend_sched_graph_compute(sched, g.graph) == GGML_STATUS_SUCCESS, "the graph ran") && ok;
    std::vector<float> out(channels);
    ggml_backend_tensor_get(ggml_graph_node(g.graph, -1), out.data(), 0, out.size() * sizeof(float));
    for (int c = 0; c < channels; c++) {
        const float want = 5.0f * x[c] + float(100 * row + c);
        if (out[c] != want) {
            std::fprintf(stderr, "FAIL: out[%d] is %g, expected %g\n", c, out[c], want);
            ok = false;
        }
    }
    return placed;
}

// a cache of `rows` rows, row i holding 100 i + c in channel c, in a buffer of type buft
ggml_tensor * make_cache(ggml_context * ctx, ggml_backend_buffer_type_t buft, ggml_backend_buffer_t & buffer) {
    ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels, rows);
    buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    std::vector<float> values(channels * rows);
    for (int i = 0; i < rows; i++) {
        for (int c = 0; c < channels; c++) { values[i * channels + c] = float(100 * i + c); }
    }
    if (buffer) { ggml_backend_tensor_set(cache, values.data(), 0, values.size() * sizeof(float)); }
    return cache;
}

// A shared-memory GPU that can directly use regular host data must still stage the mapped KV.
bool test_spilled_reads(ggml_backend_t cpu, ggml_tensor * resident, ggml_tensor * mapped) {
    constexpr int view_channels = channels - 1; // leave a gap between rows
    fake_gpu gpu(cpu, true);
    ggml_backend_t backends[2] = { &gpu.backend, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 64, false, false);
    ggml_context * ctx = ggml_init({
        ggml_tensor_overhead() * 24 + ggml_graph_overhead_custom(16, false), nullptr, true,
    });
    auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(ids);
    auto * a = ggml_view_2d(ctx, mapped, view_channels, rows - 1, mapped->nb[1], 0);
    auto * b = ggml_view_2d(ctx, mapped, view_channels, rows - 1, mapped->nb[1], mapped->nb[1]);
    auto * resident_view = ggml_view_2d(ctx, resident, view_channels, rows, resident->nb[1], 0);
    auto * read_resident = ggml_get_rows(ctx, resident_view, ids);
    auto * read_a = ggml_get_rows(ctx, a, ids);
    auto * read_b = ggml_get_rows(ctx, b, ids);
    auto * first = ggml_add(ctx, read_resident, read_a);
    auto * out = ggml_add(ctx, first, read_b);
    auto * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, out);
    for (auto * node : {read_resident, read_a, read_b, first, out}) {
        ggml_backend_sched_set_tensor_backend(sched, node, &gpu.backend);
    }

    bool ok = require(ggml_backend_sched_alloc_graph(sched, graph), "spill graph was placed");
    if (ok) {
        ok = require(ggml_backend_sched_get_n_splits(sched) == 2, "new spilled view starts its own GPU split") && ok;
        ok = require(read_resident->src[0] == resident_view, "resident read stays on its original GPU buffer") && ok;
        for (const auto & [read, view] : {std::pair{read_a, a}, std::pair{read_b, b}}) {
            const ggml_tensor * stage = read->src[0];
            ok = require(stage != view && ggml_backend_buffer_get_type(stage->buffer) == &gpu_buffer_type,
                "spilled view was staged into a GPU buffer") && ok;
            ok = require(stage->type == view->type && ggml_are_same_shape(stage, view) &&
                std::memcmp(stage->nb, view->nb, sizeof(stage->nb)) == 0, "staging preserves the view's layout") && ok;
        }
        const int32_t row = 1;
        ggml_backend_tensor_set(ids, &row, 0, sizeof(row));
        ok = require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "spill graph ran") && ok;
        float values[view_channels];
        ggml_backend_tensor_get(out, values, 0, sizeof(values));
        for (int c = 0; c < view_channels; ++c) {
            ok = require(values[c] == float(400 + 3*c), "staged row values match the resident calculation") && ok;
        }
    }
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return ok;
}

} // namespace

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    gpu_buffer_type = *ggml_backend_cpu_buffer_type();
    gpu_buffer_type.iface.is_host = gpu_is_host;
    fake_gpu gpu(cpu);

    ggml_init_params cache_params{ ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context * gpu_ctx = ggml_init(cache_params);
    ggml_context * host_ctx = ggml_init(cache_params);
    ggml_backend_buffer_t gpu_buffer = nullptr, host_buffer = nullptr;
    ggml_tensor * gpu_cache = make_cache(gpu_ctx, &gpu_buffer_type, gpu_buffer);
    ggml_tensor * host_cache = make_cache(host_ctx, ggml_backend_cpu_buffer_type(), host_buffer);

    ggml_backend_t backends[2] = { &gpu.backend, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 64, false, false);

    bool ok = require(gpu_buffer != nullptr && host_buffer != nullptr, "the caches were allocated");
    if (ok) {
        // between two CPU nodes, the row read of the GPU's cache runs on the GPU
        ok = require(place_and_run(sched, cpu, gpu_cache, ok) == &gpu.backend,
            "the row read of a GPU's cache runs on that GPU") && ok;
        // a cache in host memory is left to the neighbours: here the CPU's
        ok = require(place_and_run(sched, cpu, host_cache, ok) == cpu,
            "the row read of a host cache stays with its neighbours") && ok;
        ggml_backend_buffer_set_usage(host_buffer, GGML_BACKEND_BUFFER_USAGE_SSD_KV);
        ok = test_spilled_reads(cpu, gpu_cache, host_cache) && ok;
    }

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(gpu_buffer);
    ggml_backend_buffer_free(host_buffer);
    ggml_free(gpu_ctx);
    ggml_free(host_ctx);
    ggml_backend_free(cpu);
    if (ok) { std::printf("sched_cache_placement: ok\n"); }
    return ok ? 0 : 1;
}
