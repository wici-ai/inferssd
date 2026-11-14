// SPDX-License-Identifier: Apache-2.0
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void write_model(const std::string & path, const std::string & arch, const std::string & mode) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({1u << 20, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(gguf_init_empty(), gguf_free);
    require(ctx && gguf, "could not create Qwen fixture");
    gguf_set_val_str(gguf.get(), "general.architecture", arch.c_str());
    gguf_set_val_str(gguf.get(), "tokenizer.ggml.model", "none");
    const auto integer = [&](const char * key, uint32_t value) {
        gguf_set_val_u32(gguf.get(), (arch + "." + key).c_str(), value);
    };
    integer("vocab_size", 32);
    integer("context_length", 32);
    integer("embedding_length", 8);
    integer("block_count", 2);
    integer("feed_forward_length", 16);
    integer("attention.head_count", 1);
    integer("full_attention_interval", 1);
    integer("ssm.conv_kernel", 4);
    integer("ssm.inner_size", 8);
    integer("ssm.state_size", 8);
    integer("ssm.time_step_rank", 1);
    integer("ssm.group_count", 1);
    gguf_set_val_f32(gguf.get(), (arch + ".attention.layer_norm_rms_epsilon").c_str(), 1e-5f);
    const uint32_t sections[] = {2, 2, 2, 2};
    gguf_set_arr_data(gguf.get(), (arch + ".rope.dimension_sections").c_str(), GGUF_TYPE_UINT32, sections, 4);
    if (arch == "qwen35moe") {
        integer("expert_count", 2);
        integer("expert_used_count", 1);
        integer("expert_feed_forward_length", 16);
        integer("expert_shared_feed_forward_length", 16);
    }
    if (mode == "wrong-type") {
        gguf_set_val_str(gguf.get(), (arch + ".nextn_predict_layers").c_str(), "0");
    } else if (mode != "absent") {
        integer("nextn_predict_layers", mode == "zero" ? 0 :
            mode == "all-head" ? 2 : mode == "maximum" ? std::numeric_limits<uint32_t>::max() : 1);
    }

    const auto add = [&](const std::string & name, std::vector<int64_t> shape) {
        auto * tensor = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, (int) shape.size(), shape.data());
        ggml_set_name(tensor, name.c_str());
        auto * values = static_cast<float *>(tensor->data);
        std::fill(values, values + ggml_nelements(tensor), 0.25f);
        gguf_add_tensor(gguf.get(), tensor);
    };
    add("token_embd.weight", {8, 32});
    add("output_norm.weight", {8});
    add("output.weight", {8, 32});
    // A complete two-block trunk for the absent/zero and type-error controls;
    // a trunk plus a NextN projection for files declaring prediction layers.
    const int trunk_layers = mode == "absent" || mode == "zero" || mode == "wrong-type" ? 2 : 1;
    for (int i = 0; i < trunk_layers; ++i) {
        const std::string block = "blk." + std::to_string(i) + ".";
        add(block + "attn_norm.weight", {8});
        add(block + "post_attention_norm.weight", {8});
        add(block + "attn_q.weight", {8, 16});
        add(block + "attn_k.weight", {8, 8});
        add(block + "attn_v.weight", {8, 8});
        add(block + "attn_output.weight", {8, 8});
        add(block + "attn_q_norm.weight", {8});
        add(block + "attn_k_norm.weight", {8});
        if (arch == "qwen35moe") {
            add(block + "ffn_gate_inp.weight", {8, 2});
            add(block + "ffn_down_exps.weight", {16, 8, 2});
            add(block + "ffn_gate_exps.weight", {8, 16, 2});
            add(block + "ffn_up_exps.weight", {8, 16, 2});
            add(block + "ffn_gate_inp_shexp.weight", {8});
            add(block + "ffn_gate_shexp.weight", {8, 16});
            add(block + "ffn_up_shexp.weight", {8, 16});
            add(block + "ffn_down_shexp.weight", {16, 8});
        } else {
            add(block + "ffn_gate.weight", {8, 16});
            add(block + "ffn_down.weight", {16, 8});
            add(block + "ffn_up.weight", {8, 16});
        }
    }
    if (trunk_layers == 1) {
        add("blk.1.nextn.eh_proj.weight", {16, 8});
    }
    require(gguf_write_to_file(gguf.get(), path.c_str(), false), "could not write Qwen fixture");
}

int main(int argc, char ** argv) {
    require(argc == 4, "expected architecture, mode and fixture path");
    const std::string arch = argv[1], mode = argv[2], path = argv[3];
    require(arch == "qwen35" || arch == "qwen35moe", "unknown architecture");
    struct remove_fixture {
        std::string path;
        ~remove_fixture() { std::remove(path.c_str()); }
    } cleanup{path};
    write_model(path, arch, mode);
    ggml_backend_load_all();
    llama_backend_init();
    std::string diagnostics;
    llama_log_set([](ggml_log_level, const char * text, void * data) {
        static_cast<std::string *>(data)->append(text);
    }, &diagnostics);
    ggml_backend_dev_t no_devices[] = {nullptr};
    auto params = llama_model_default_params();
    params.devices = no_devices;
    params.n_gpu_layers = 0;
    params.use_mmap = false;
    params.use_mlock = false;
    params.vocab_only = mode == "vocab-only";
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(path.c_str(), params), llama_model_free);
    llama_log_set(nullptr, nullptr);
    const bool should_load = mode == "absent" || mode == "zero" || mode == "vocab-only";
    if ((model != nullptr) != should_load) {
        std::fprintf(stderr, "%s", diagnostics.c_str());
        require(false, "unexpected Qwen model-load result");
    }
    if (should_load) {
        if (!params.vocab_only) {
            require(llama_model_n_layer(model.get()) == 2, "ordinary trunk layer count changed");
        }
        require(llama_vocab_n_tokens(llama_model_get_vocab(model.get())) == 32, "vocabulary changed");
    } else {
        const std::string key = arch + ".nextn_predict_layers";
        const bool explains = diagnostics.find(key) != std::string::npos &&
            (mode == "wrong-type" ? diagnostics.find("expected type u32") != std::string::npos :
                diagnostics.find("NextN/MTP") != std::string::npos &&
                diagnostics.find("absent or zero") != std::string::npos);
        if (!explains) {
            std::fprintf(stderr, "%s", diagnostics.c_str());
            require(false, "missing actionable NextN metadata diagnostic");
        }
        require(diagnostics.find("missing tensor") == std::string::npos, "misreported missing tensor");
    }
    model.reset();
    llama_backend_free();
    std::printf("%s %s: passed\n", arch.c_str(), mode.c_str());
    return 0;
}
