// SPDX-License-Identifier: Apache-2.0
// The scheduler offloads an op on host weights only while op offload is on, and that can change between graphs: a pass
// short enough to be faster on the CPU stays there, and a long one goes to the GPU, with one scheduler.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <cstdio>

namespace {

// A second backend, as the scheduler sees a GPU that wants to take every op: a CPU behind its own device and buffer type.
ggml_backend_buffer_type gpu_buffer_type;

ggml_backend_buffer_type_t gpu_get_buffer_type(ggml_backend_dev_t) { return &gpu_buffer_type; }
bool gpu_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) { return buft == &gpu_buffer_type; }
bool gpu_supports_op(ggml_backend_dev_t, const ggml_tensor *) { return true; }
bool gpu_offloads_op(ggml_backend_dev_t, const ggml_tensor *) { return true; }

} // namespace

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    gpu_buffer_type = *ggml_backend_cpu_buffer_type();
    ggml_backend_device gpu_device = *ggml_backend_get_device(cpu);
    gpu_device.iface.get_buffer_type = gpu_get_buffer_type;
    gpu_device.iface.supports_buft = gpu_supports_buft;
    gpu_device.iface.supports_op = gpu_supports_op;
    gpu_device.iface.offload_op = gpu_offloads_op;
    ggml_backend gpu_backend = *cpu;
    gpu_backend.device = &gpu_device;
    ggml_backend_t gpu = &gpu_backend;

    // A host weight, as a model mapped on the CPU holds it.
    ggml_init_params weight_params{ ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context * weight_ctx = ggml_init(weight_params);
    ggml_tensor * weight = ggml_new_tensor_2d(weight_ctx, GGML_TYPE_F32, 4, 2);
    ggml_backend_buffer_t weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_backend_t backends[2] = { gpu, cpu };
    // Created with offload off: the setter alone has to turn it on.
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 16, false, false);

    bool ok = true;
    const auto check = [&](const char * name, bool offload, ggml_backend_t expected) {
        ggml_init_params params{ ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false), nullptr, true };
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8);
        ggml_set_input(rows);
        ggml_tensor * product = ggml_mul_mat(ctx, weight, rows);
        ggml_set_output(product);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, product);

        ggml_backend_sched_reset(sched);
        ggml_backend_sched_set_op_offload(sched, offload);
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            std::fprintf(stderr, "%s: allocation failed\n", name);
            ok = false;
        } else if (ggml_backend_sched_get_tensor_backend(sched, product) != expected) {
            std::fprintf(stderr, "%s: the op is on the wrong backend\n", name);
            ok = false;
        }
        ggml_backend_sched_reset(sched);
        ggml_free(ctx);
    };

    check("offload on", true, gpu);
    check("offload off", false, cpu);
    check("offload on again", true, gpu);

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(weight_ctx);
    ggml_backend_free(cpu);
    return ok ? 0 : 1;
}
