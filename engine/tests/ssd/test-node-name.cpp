// SPDX-License-Identifier: Apache-2.0
// A prompt graph is held to run on the GPU the ops the decode graph runs there, matched by name. ggml names a node the
// model left unnamed by its position, so two graphs of one model give one such name to two different ops: a prompt's
// last, short ubatch of DeepSeek-V4.1 had its engram row gather, which runs on the CPU, at the position of a node the
// decode graph ran on the GPU, and the prompt failed. Only the names the model gives count.

#include "ggml.h"
#include "llama-node-name.h"

#include <cstdio>
#include <cstring>

static bool require(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); }
    return ok;
}

int main() {
    ggml_init_params params{ ggml_tensor_overhead() * 32 + ggml_graph_overhead() * 2, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);

    // two graphs whose first unnamed node is a different op
    ggml_tensor * sum = ggml_add(ctx, x, w);
    ggml_cgraph * decode = ggml_new_graph(ctx);
    ggml_build_forward_expand(decode, sum);
    ggml_tensor * product = ggml_mul(ctx, x, w);
    ggml_cgraph * prompt = ggml_new_graph(ctx);
    ggml_build_forward_expand(prompt, product);

    bool ok = require(std::strcmp(sum->name, product->name) == 0 && sum->op != product->op,
        "ggml gives two graphs' unnamed nodes one positional name");
    ok = require(!llama_node_name_is_given(sum->name) && !llama_node_name_is_given(product->name),
        "a positional node name is not the model's") && ok;
    ok = require(std::strncmp(x->name, "leaf_", 5) == 0 && !llama_node_name_is_given(x->name),
        "a positional leaf name is not the model's") && ok;
    ok = require(!llama_node_name_is_given("leaf_17") && !llama_node_name_is_given("node_2036 (view)"),
        "a positional name stays positional with a suffix") && ok;

    // a node the model names, and ggml's names derived from it or from an unnamed tensor
    ggml_tensor * named = ggml_scale(ctx, x, 2.0f);
    ggml_set_name(named, "attn_norm-3");
    ggml_tensor * view = ggml_view_1d(ctx, named, 4, 0);
    ggml_tensor * unnamed_view = ggml_view_1d(ctx, ggml_scale(ctx, w, 3.0f), 4, 0);
    ok = require(llama_node_name_is_given(named->name) && llama_node_name_is_given(view->name),
        "a name the model gives, and its view's, are the model's") && ok;
    ok = require(!llama_node_name_is_given(unnamed_view->name), "a view of an unnamed tensor has no name of its own") && ok;
    ok = require(llama_node_name_is_given("node_norm-1") && llama_node_name_is_given("leaf_count"),
        "a model name that starts like ggml's is the model's") && ok;
    ok = require(!llama_node_name_is_given("") && !llama_node_name_is_given(nullptr), "no name is not the model's") && ok;

    ggml_free(ctx);
    if (ok) { std::printf("node_name: ok\n"); }
    return ok ? 0 : 1;
}
