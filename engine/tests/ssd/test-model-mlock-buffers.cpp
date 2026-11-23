// SPDX-License-Identifier: Apache-2.0
#include "llama.h"
#include "llama-model.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
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

// Like a pinned host backend, this type can split a context across allocations.
struct host_type {
    size_t max_size;
    std::vector<ggml_backend_buffer_t> buffers;
    ggml_backend_buffer_type type;

    explicit host_type(size_t maximum) : max_size(maximum), type(*ggml_backend_cpu_buffer_type()) {
        type.context = this;
        type.iface.get_name = [](ggml_backend_buffer_type_t) { return "mlock_host_fixture"; };
        type.iface.get_max_size = [](ggml_backend_buffer_type_t buft) {
            return static_cast<host_type *>(buft->context)->max_size;
        };
        type.iface.alloc_buffer = [](ggml_backend_buffer_type_t buft, size_t size) {
            auto * buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
            if (buffer) {
                buffer->buft = buft;
                static_cast<host_type *>(buft->context)->buffers.push_back(buffer);
            }
            return buffer;
        };
    }
};

static void write_model(const std::string & path) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({1u << 20, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(gguf_init_empty(), gguf_free);
    require(ctx && gguf, "could not create model fixture");
    gguf_set_val_str(gguf.get(), "general.architecture", "llama");
    gguf_set_val_str(gguf.get(), "tokenizer.ggml.model", "none");
    gguf_set_val_u32(gguf.get(), "llama.vocab_size", 32);
    gguf_set_val_u32(gguf.get(), "llama.context_length", 32);
    gguf_set_val_u32(gguf.get(), "llama.embedding_length", 32);
    gguf_set_val_u32(gguf.get(), "llama.block_count", 1);
    gguf_set_val_u32(gguf.get(), "llama.feed_forward_length", 32);
    gguf_set_val_u32(gguf.get(), "llama.attention.head_count", 1);
    gguf_set_val_f32(gguf.get(), "llama.attention.layer_norm_rms_epsilon", 1e-5f);

    const auto add = [&](const char * name, bool matrix) {
        auto * tensor = matrix ? ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 32) :
                                 ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 32);
        ggml_set_name(tensor, name);
        auto * values = static_cast<float *>(tensor->data);
        std::fill(values, values + ggml_nelements(tensor), 0.25f);
        gguf_add_tensor(gguf.get(), tensor);
    };
    add("token_embd.weight", true);
    add("output_norm.weight", false);
    add("output.weight", true);
    add("blk.0.attn_norm.weight", false);
    add("blk.0.attn_q.weight", true);
    add("blk.0.attn_k.weight", true);
    add("blk.0.attn_v.weight", true);
    add("blk.0.attn_output.weight", true);
    add("blk.0.ffn_norm.weight", false);
    add("blk.0.ffn_gate.weight", true);
    add("blk.0.ffn_down.weight", true);
    add("blk.0.ffn_up.weight", true);
    require(gguf_write_to_file(gguf.get(), path.c_str(), false), "could not write model fixture");
}

int main(int argc, char ** argv) {
    require(argc == 3, "expected a test case name and fixture path");
    const std::string mode = argv[1];
    require(mode == "single" || mode == "split" || mode == "unlocked" || mode == "no-alloc", "unknown test case");
    const bool split = mode != "single";
    const bool no_alloc = mode == "no-alloc";
    const bool use_mlock = mode != "unlocked";
    const std::string path = argv[2];
    struct remove_fixture {
        std::string path;
        ~remove_fixture() { std::remove(path.c_str()); }
    } cleanup{path};
    write_model(path);

    ggml_backend_load_all();
    llama_backend_init();
    host_type host(split ? 4096 : std::numeric_limits<size_t>::max());
    llama_model_tensor_buft_override overrides[] = {{".*", &host.type}, {nullptr, nullptr}};
    ggml_backend_dev_t no_devices[] = {nullptr};
    auto params = llama_model_default_params();
    params.devices = no_devices;
    params.n_gpu_layers = 0;
    params.use_mmap = false;
    params.use_mlock = use_mlock;
    params.no_alloc = no_alloc;
    params.use_extra_bufts = false;
    params.tensor_buft_overrides = overrides;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(path.c_str(), params), llama_model_free);
    require(model != nullptr, "model loading failed");
    if (no_alloc) {
        const auto * tensor = model->tok_embd;
        require(host.buffers.empty() && tensor && tensor->buffer &&
                    ggml_backend_buffer_get_size(tensor->buffer) == 0 && tensor->data == nullptr,
                "fixture did not allocate a zero-sized placeholder");
    } else {
        require(split ? host.buffers.size() > 1 : host.buffers.size() == 1,
                "fixture did not exercise the requested allocation layout");
        for (const auto * tensor : {model->tok_embd, model->output, model->layers[0].ffn_up}) {
            require(tensor && tensor->buffer && tensor->data, "loaded tensor has no physical allocation");
            std::vector<float> values(ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
            require(std::all_of(values.begin(), values.end(), [](float value) { return value == 0.25f; }),
                    "tensor data changed during model loading");
        }
    }
    std::printf("%s: loaded successfully with %zu physical buffers\n", mode.c_str(), host.buffers.size());
    model.reset();
    llama_backend_free();
    return 0;
}
