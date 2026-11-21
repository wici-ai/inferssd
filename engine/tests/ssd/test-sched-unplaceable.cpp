// SPDX-License-Identifier: Apache-2.0
// An op that writes into a GPU's own buffer cannot move to another backend. When that GPU refuses the op (its source
// outgrows what the GPU's kernels address, as a long prompt's view of a whole ubatch does), the scheduler cannot place
// the graph: it refuses it, and the same graph over fewer tokens runs. It used to abort the process.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int channels = 4;
constexpr int kept = 3; // the last columns a state cache keeps
// the largest source the GPU copies, in bytes: the view of 16 positions fits, the view of 100 does not
constexpr size_t gpu_copy_limit = sizeof(float) * kept * 32;

// A second backend, as the scheduler sees a GPU: a CPU behind its own device and a buffer type only it can use, so a
// tensor in that buffer cannot run on the CPU backend.
ggml_backend_buffer_type gpu_buffer_type;

ggml_backend_buffer_type_t gpu_get_buffer_type(ggml_backend_dev_t) { return &gpu_buffer_type; }
bool gpu_is_host(ggml_backend_buffer_type_t) { return false; }
bool gpu_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) { return buft == &gpu_buffer_type; }
bool gpu_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    return op->op != GGML_OP_CPY || ggml_nbytes(op->src[0]) <= gpu_copy_limit;
}

struct fake_gpu {
    ggml_backend_device device;
    ggml_backend backend;

    explicit fake_gpu(ggml_backend_t cpu) {
        device = *ggml_backend_get_device(cpu);
        device.iface.get_buffer_type = gpu_get_buffer_type;
        device.iface.supports_buft = gpu_supports_buft;
        device.iface.supports_op = gpu_supports_op;
        backend = *cpu;
        backend.device = &device;
    }
};

// x [positions, channels] -> y = 2x -> the last `kept` positions of every channel (a strided view across all of y)
// copied into a state tensor held in the GPU's buffer: the shape of a recurrent layer's conv-state update.
struct state_update {
    ggml_context * ctx = nullptr;
    ggml_tensor * x = nullptr;
    ggml_cgraph * graph = nullptr;

    state_update(ggml_tensor * state, int positions) {
        ggml_init_params params{ ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false), nullptr, true };
        ctx = ggml_init(params);
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, positions, channels);
        ggml_set_input(x);
        ggml_tensor * y = ggml_scale(ctx, x, 2.0f);
        ggml_tensor * last = ggml_view_2d(ctx, y, kept, channels, y->nb[1], (positions - kept) * sizeof(float));
        ggml_tensor * update = ggml_cpy(ctx, last, state);
        graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, update);
    }
    ~state_update() { ggml_free(ctx); }
};

bool require(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); }
    return ok;
}

} // namespace

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    gpu_buffer_type = *ggml_backend_cpu_buffer_type();
    gpu_buffer_type.iface.is_host = gpu_is_host;
    fake_gpu gpu(cpu);

    // the state cache, allocated in the GPU's buffer before any graph
    ggml_init_params state_params{ ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context * state_ctx = ggml_init(state_params);
    ggml_tensor * state = ggml_new_tensor_2d(state_ctx, GGML_TYPE_F32, kept, channels);
    ggml_backend_buffer_t state_buffer = ggml_backend_alloc_ctx_tensors_from_buft(state_ctx, &gpu_buffer_type);

    ggml_backend_t backends[2] = { &gpu.backend, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 64, false, false);

    bool ok = require(state_buffer != nullptr, "the state buffer was allocated");
    {
        // 100 positions: the view the copy reads spans all of y, past what the GPU copies
        state_update big(state, 100);
        size_t sizes[2] = { 0, 0 };
        ok = require(!ggml_backend_sched_prepare_graph(sched, big.graph, sizes), "prepare refused the graph") && ok;
        ggml_backend_sched_reset(sched);
        ok = require(!ggml_backend_sched_reserve(sched, big.graph), "reserve refused the graph") && ok;
        ggml_backend_sched_reset(sched);
        ok = require(!ggml_backend_sched_alloc_graph(sched, big.graph), "allocation refused the graph") && ok;
        ggml_backend_sched_reset(sched);
    }
    {
        // 16 positions: the same graph runs, on the same scheduler, and writes the state on the GPU
        constexpr int positions = 16;
        state_update small(state, positions);
        ok = require(ggml_backend_sched_alloc_graph(sched, small.graph), "the smaller graph was placed") && ok;
        if (ok) {
            const ggml_tensor * update = ggml_graph_node(small.graph, -1);
            ok = require(ggml_backend_sched_get_tensor_backend(sched, const_cast<ggml_tensor *>(update)) == &gpu.backend,
                "the copy runs on the GPU that holds the state") && ok;
            std::vector<float> values(positions * channels);
            for (size_t i = 0; i < values.size(); i++) { values[i] = float(i); }
            ggml_backend_tensor_set(small.x, values.data(), 0, values.size() * sizeof(float));
            ok = require(ggml_backend_sched_graph_compute(sched, small.graph) == GGML_STATUS_SUCCESS, "the graph ran") && ok;
            std::vector<float> got(kept * channels);
            ggml_backend_tensor_get(state, got.data(), 0, got.size() * sizeof(float));
            for (int c = 0; c < channels; c++) {
                for (int k = 0; k < kept; k++) {
                    const float want = 2.0f * values[c * positions + positions - kept + k];
                    if (got[c * kept + k] != want) {
                        std::fprintf(stderr, "FAIL: state[%d][%d] is %g, expected %g\n", c, k, got[c * kept + k], want);
                        ok = false;
                    }
                }
            }
        }
    }

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(state_buffer);
    ggml_free(state_ctx);
    ggml_backend_free(cpu);
    if (ok) { std::printf("sched_unplaceable: ok\n"); }
    return ok ? 0 : 1;
}
