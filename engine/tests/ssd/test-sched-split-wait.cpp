// SPDX-License-Identifier: Apache-2.0
// A split with no inputs waits for the previous split's backend before it runs: the allocator may have given it memory
// that the previous split still reads. A GPU backend copies a split's inputs from host memory asynchronously, so the
// copy runs on the GPU after the scheduler has moved on to the next split. A CPU split with no inputs after it then must
// not write that memory before the copy has read it. The fake GPU here queues its copies and graphs and runs them at
// synchronize, as a GPU does.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

namespace {

// A second backend, as the scheduler sees a GPU: a CPU behind its own device and a buffer type of its own that the CPU
// does not read in place, so tensors cross between the two by copies.
ggml_backend_buffer_type gpu_buffer_type;

ggml_backend_buffer_type_t gpu_get_buffer_type(ggml_backend_dev_t) { return &gpu_buffer_type; }

const char * gpu_buffer_type_name(ggml_backend_buffer_type_t) { return "fake GPU"; }

bool gpu_buffer_type_is_host(ggml_backend_buffer_type_t) { return false; }

bool gpu_supports_every_op(ggml_backend_dev_t, const ggml_tensor *) { return true; }

bool gpu_supports_own_buffers(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) { return buft == &gpu_buffer_type; }

ggml_backend_t cpu;                          // computes the fake GPU's graphs when they run
std::vector<std::function<void()>> queued;   // the fake GPU's work, in the order it was submitted

void gpu_synchronize(ggml_backend_t) {
    for (auto & work : queued) {
        work();
    }
    queued.clear();
}

enum ggml_status gpu_graph_compute(ggml_backend_t, ggml_cgraph * graph) {
    queued.push_back([graph] { ggml_backend_graph_compute(cpu, graph); });
    return GGML_STATUS_SUCCESS;
}

bool gpu_cpy_tensor_async(ggml_backend_t, ggml_backend_t, const ggml_tensor * src, ggml_tensor * dst) {
    if (!src->buffer || !ggml_backend_buffer_is_host(src->buffer)) {
        return false;
    }
    queued.push_back([src, dst] { std::memcpy(dst->data, src->data, ggml_nbytes(src)); });
    return true;
}

bool overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * pa = (const char *) a->data, * pb = (const char *) b->data;
    return pa < pb + ggml_nbytes(b) && pb < pa + ggml_nbytes(a);
}

} // namespace

int main() {
    cpu = ggml_backend_cpu_init();
    gpu_buffer_type = *ggml_backend_cpu_buffer_type();
    gpu_buffer_type.iface.get_name = gpu_buffer_type_name;
    gpu_buffer_type.iface.is_host = gpu_buffer_type_is_host;
    ggml_backend_device gpu_device = *ggml_backend_get_device(cpu);
    gpu_device.iface.get_buffer_type = gpu_get_buffer_type;
    gpu_device.iface.supports_op = gpu_supports_every_op;
    gpu_device.iface.supports_buft = gpu_supports_own_buffers;
    ggml_backend gpu = *cpu;
    gpu.device = &gpu_device;
    gpu.iface.graph_compute = gpu_graph_compute;
    gpu.iface.synchronize = gpu_synchronize;
    gpu.iface.cpy_tensor_async = gpu_cpy_tensor_async;

    ggml_backend_t backends[2] = { &gpu, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 64, false, false);

    const int n = 1024;
    ggml_init_params params{ ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_input(a);
    ggml_set_input(b);
    ggml_tensor * x = ggml_scale(ctx, a, 2.0f);   // CPU
    ggml_tensor * y = ggml_scale(ctx, x, 3.0f);   // GPU: x is copied to it asynchronously
    ggml_tensor * c = ggml_scale(ctx, b, 5.0f);   // CPU: a split with no inputs, in the memory x had (b stays in use)
    ggml_tensor * z = ggml_scale(ctx, c, 7.0f);   // GPU
    ggml_tensor * out = ggml_add(ctx, ggml_add(ctx, y, z), b); // CPU
    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_sched_set_tensor_backend(sched, x, cpu);
    ggml_backend_sched_set_tensor_backend(sched, y, &gpu);
    ggml_backend_sched_set_tensor_backend(sched, c, cpu);
    ggml_backend_sched_set_tensor_backend(sched, z, &gpu);
    ggml_backend_sched_set_tensor_backend(sched, out, cpu);
    ggml_backend_sched_set_tensor_backend(sched, out->src[0], cpu);

    bool ok = ggml_backend_sched_alloc_graph(sched, graph);
    if (!ok) {
        std::fprintf(stderr, "allocation failed\n");
    }
    // The case only tests something while the allocator gives c memory that x had.
    if (ok && !overlap(x, c)) {
        std::fprintf(stderr, "c does not reuse the memory of x: the graph no longer exercises the wait\n");
        ok = false;
    }
    if (ok) {
        std::vector<float> ones(n, 1.0f), tens(n, 10.0f), got(n);
        ggml_backend_tensor_set(a, ones.data(), 0, n * sizeof(float));
        ggml_backend_tensor_set(b, tens.data(), 0, n * sizeof(float));
        ok = ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
        ggml_backend_tensor_get(out, got.data(), 0, n * sizeof(float));
        // y = 3 * 2 * 1 = 6, z = 7 * 5 * 10 = 350, b = 10
        for (int i = 0; ok && i < n; i++) {
            if (got[i] != 366.0f) {
                std::fprintf(stderr, "out[%d] = %g, expected 366: the GPU's copy of x read memory a later split had "
                        "written\n", i, got[i]);
                ok = false;
            }
        }
    }
    ggml_free(ctx);
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
