// SPDX-License-Identifier: Apache-2.0
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

struct fixture {
    const char * name;
    uint32_t experts, used, groups, groups_used;
    const char * invalid_key;
    bool omit = false, vocab_only = false;
    const char * wrong_type = nullptr;
};

static const fixture cases[] = {
    {"too-many-experts", 1025, 1, 0, 0, "expert_count"},
    {"too-many-active", 4, 5, 0, 0, "expert_used_count"},
    {"no-active", 4, 0, 0, 0, "expert_used_count"},
    {"too-many-groups", 4, 1, 4, 1, "expert_group_count"},
    {"uneven-groups", 4, 1, 3, 1, "expert_group_count"},
    {"no-active-groups", 4, 1, 2, 0, "expert_group_used_count"},
    {"all-groups-active", 4, 1, 2, 2, "expert_group_used_count"},
    {"excess-active-groups", 4, 1, 2, 3, "expert_group_used_count"},
    {"dense-with-groups", 0, 0, 1, 0, "expert_group_count"},
    {"dense-with-active", 0, 1, 0, 0, "expert_used_count"},
    {"dense-absent", 0, 0, 0, 0, nullptr, true},
    {"dense-zero", 0, 0, 0, 0, nullptr},
    {"routed", 4, 2, 0, 0, nullptr},
    {"single-expert", 1, 1, 0, 0, nullptr},
    {"all-experts-active", 4, 4, 0, 0, nullptr},
    {"single-group", 4, 2, 1, 0, nullptr},
    {"grouped", 4, 2, 2, 1, nullptr},
    {"maximum", 1024, 1, 0, 0, nullptr},
    {"inactive-group-setting", 4, 1, 0, 99, nullptr},
    {"single-group-setting", 4, 1, 1, 99, nullptr},
    {"vocab-only", 1025, 1026, 1027, 1028, nullptr, false, true},
    {"count-type", 4, 1, 0, 0, "expert_count", false, false, "expert_count"},
    {"used-type", 4, 1, 0, 0, "expert_used_count", false, false, "expert_used_count"},
    {"groups-type", 4, 1, 2, 1, "expert_group_count", false, false, "expert_group_count"},
    {"groups-used-type", 4, 1, 2, 1, "expert_group_used_count", false, false, "expert_group_used_count"},
};

static void write_model(const std::string & path, const fixture & entry) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({4u << 20, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(gguf_init_empty(), gguf_free);
    require(ctx && gguf, "could not create expert metadata fixture");
    gguf_set_val_str(gguf.get(), "general.architecture", "llama");
    gguf_set_val_str(gguf.get(), "tokenizer.ggml.model", "none");
    const auto integer = [&](const char * key, uint32_t value) {
        const std::string name = std::string("llama.") + key;
        if (entry.wrong_type && std::string(entry.wrong_type) == key) {
            gguf_set_val_str(gguf.get(), name.c_str(), "1");
        } else {
            gguf_set_val_u32(gguf.get(), name.c_str(), value);
        }
    };
    integer("vocab_size", 32);
    integer("context_length", 32);
    integer("embedding_length", 8);
    integer("block_count", 1);
    integer("feed_forward_length", 16);
    integer("attention.head_count", 1);
    if (!entry.omit) {
        integer("expert_count", entry.experts);
        integer("expert_used_count", entry.used);
        integer("expert_group_count", entry.groups);
        integer("expert_group_used_count", entry.groups_used);
    }
    gguf_set_val_f32(gguf.get(), "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    const auto add = [&](const char * name, std::vector<int64_t> shape) {
        auto * tensor = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, (int) shape.size(), shape.data());
        ggml_set_name(tensor, name);
        auto * values = static_cast<float *>(tensor->data);
        std::fill(values, values + ggml_nelements(tensor), 0.25f);
        gguf_add_tensor(gguf.get(), tensor);
    };
    add("token_embd.weight", {8, 32});
    add("output_norm.weight", {8});
    add("output.weight", {8, 32});
    add("blk.0.attn_norm.weight", {8});
    add("blk.0.attn_q.weight", {8, 8});
    add("blk.0.attn_k.weight", {8, 8});
    add("blk.0.attn_v.weight", {8, 8});
    add("blk.0.attn_output.weight", {8, 8});
    add("blk.0.ffn_norm.weight", {8});
    if (entry.experts) {
        add("blk.0.ffn_gate_inp.weight", {8, entry.experts});
        add("blk.0.ffn_gate_exps.weight", {8, 16, entry.experts});
        add("blk.0.ffn_up_exps.weight", {8, 16, entry.experts});
        add("blk.0.ffn_down_exps.weight", {16, 8, entry.experts});
    } else {
        add("blk.0.ffn_gate.weight", {8, 16});
        add("blk.0.ffn_up.weight", {8, 16});
        add("blk.0.ffn_down.weight", {16, 8});
    }
    require(gguf_write_to_file(gguf.get(), path.c_str(), false), "could not write expert metadata fixture");
}

int main(int argc, char ** argv) {
    require(argc == 3, "expected fixture name and path");
    const auto found = std::find_if(std::begin(cases), std::end(cases), [&](const fixture & entry) {
        return std::string(argv[1]) == entry.name;
    });
    require(found != std::end(cases), "unknown fixture");
    const auto & entry = *found;
    const std::string path = argv[2];
    struct remove_fixture {
        std::string path;
        ~remove_fixture() { std::remove(path.c_str()); }
    } cleanup{path};
    write_model(path, entry);
    ggml_backend_load_all();
    llama_backend_init();
    ggml_backend_dev_t no_devices[] = {nullptr};
    auto params = llama_model_default_params();
    params.devices = no_devices;
    params.n_gpu_layers = 0;
    params.use_mmap = false;
    params.use_mlock = false;
    params.vocab_only = entry.vocab_only;
    std::string diagnostics;
    llama_log_set([](ggml_log_level, const char * text, void * data) {
        static_cast<std::string *>(data)->append(text);
    }, &diagnostics);
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(path.c_str(), params), llama_model_free);
    const bool should_load = entry.invalid_key == nullptr;
    if ((model != nullptr) != should_load) {
        std::fprintf(stderr, "%s", diagnostics.c_str());
        require(false, "unexpected expert metadata load result");
    }
    if (!should_load) {
        require(diagnostics.find(entry.invalid_key) != std::string::npos, "missing invalid metadata key");
        require(diagnostics.find(entry.wrong_type ? "expected type u32" : "invalid expert metadata") !=
                    std::string::npos, "missing metadata error explanation");
        require(diagnostics.find("missing tensor") == std::string::npos, "misreported missing tensor");
        // A rejected model must leave the host usable for its next load.
        write_model(path, {"recovery", 0, 0, 0, 0, nullptr});
        model.reset(llama_model_load_from_file(path.c_str(), params));
        require(model != nullptr, "could not load a valid model after rejection");
    }
    llama_log_set(nullptr, nullptr);
    require(llama_vocab_n_tokens(llama_model_get_vocab(model.get())) == 32, "vocabulary changed");
    if (!entry.vocab_only) { require(llama_model_n_layer(model.get()) == 1, "layer count changed"); }
    model.reset();
    llama_backend_free();
    std::printf("%s: passed\n", entry.name);
    return 0;
}
