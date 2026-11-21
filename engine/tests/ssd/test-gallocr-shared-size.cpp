// SPDX-License-Identifier: Apache-2.0
// Backends of one buffer type share one compute buffer (the CPU and BLAS backends). The size probe reports that
// buffer once, at the first of its ids, as the buffer's held size is reported: a caller that adds the sizes up to ask
// for memory asked for it twice, and a prompt graph on a Mac was refused memory it did not need.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <cstdio>
#include <vector>

static bool require(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); }
    return ok;
}

int main() {
    ggml_backend_buffer_type_t cpu = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type_t bufts[2] = { cpu, cpu };

    ggml_init_params params{ ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    ggml_set_input(x);
    ggml_tensor * a = ggml_scale(ctx, x, 2.0f);   // on buffer id 0
    ggml_tensor * b = ggml_scale(ctx, a, 3.0f);   // on buffer id 1, the same buffer type
    ggml_tensor * c = ggml_add(ctx, a, b);        // on buffer id 1
    ggml_set_output(c);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, c);

    std::vector<int> node_ids(ggml_graph_n_nodes(graph), 1), leaf_ids(graph->n_leafs, 0);
    node_ids[0] = 0;

    ggml_gallocr_t galloc = ggml_gallocr_new_n(bufts, 2);
    size_t sizes[2] = { 0, 0 };
    ggml_gallocr_reserve_n_size(galloc, graph, node_ids.data(), leaf_ids.data(), sizes);
    bool ok = require(sizes[0] > 0, "the shared buffer has a size");
    ok = require(sizes[1] == 0, "the shared buffer is reported once, at its first id") && ok;
    ok = require(ggml_gallocr_reserve_n(galloc, graph, node_ids.data(), leaf_ids.data()), "the graph reserves") && ok;
    ok = require(ggml_gallocr_get_buffer_size(galloc, 0) == sizes[0] && ggml_gallocr_get_buffer_size(galloc, 1) == 0,
        "the probe's sizes are the buffer's held sizes") && ok;

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    if (ok) { std::printf("gallocr_shared_size: ok\n"); }
    return ok ? 0 : 1;
}
