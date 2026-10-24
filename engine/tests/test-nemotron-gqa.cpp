// SPDX-License-Identifier: MIT
// Synthetic GGUFs: standard/non-standard GQA, split/merged QKV and optional biases.
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static constexpr int embd = 64, head_dim = 16, kv_heads = 2, vocab = 32;

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static std::vector<float> weights(int count, unsigned seed) {
    std::vector<float> result(count);
    for (auto & v : result) {
        seed = seed * 1664525u + 1013904223u;
        v = (int(seed >> 16) - 32768) / 32768.0f * 0.05f;
    }
    return result;
}

static llama_model * make_model(int heads, bool merged, bool bias, bool invalid, bool gpu) {
    auto * meta = gguf_init_empty();
    auto * data = ggml_init({2 * 1024 * 1024, nullptr, false});
    check(meta && data, "fixture allocation failed");
    gguf_set_val_str(meta, "general.architecture", "nemotron");
    gguf_set_val_str(meta, "tokenizer.ggml.model", "none");
    for (const auto & item : std::vector<std::pair<const char *, uint32_t>>{
            {"vocab_size", vocab}, {"context_length", 32}, {"embedding_length", embd},
            {"block_count", 1}, {"feed_forward_length", 96},
            {"attention.head_count", uint32_t(heads)}, {"attention.head_count_kv", kv_heads},
            {"attention.key_length", head_dim}, {"attention.value_length", head_dim},
            {"rope.dimension_count", 8}}) {
        gguf_set_val_u32(meta, ("nemotron." + std::string(item.first)).c_str(), item.second);
    }
    gguf_set_val_f32(meta, "nemotron.attention.layer_norm_epsilon", 1e-5f);
    auto tensor = [&](const char * name, int x, int y, const std::vector<float> & values) {
        check(int(values.size()) == x * y, "wrong fixture size");
        auto * t = y == 1 ? ggml_new_tensor_1d(data, GGML_TYPE_F32, x) :
                           ggml_new_tensor_2d(data, GGML_TYPE_F32, x, y);
        ggml_set_name(t, name);
        std::memcpy(t->data, values.data(), values.size() * sizeof(float));
        gguf_add_tensor(meta, t);
    };
    tensor("token_embd.weight", embd, vocab, weights(embd * vocab, 1));
    tensor("output.weight", embd, vocab, weights(embd * vocab, 2));
    tensor("output_norm.weight", embd, 1, std::vector<float>(embd, 1.0f));
    tensor("output_norm.bias", embd, 1, std::vector<float>(embd, 0.0f));
    tensor("blk.0.attn_norm.weight", embd, 1, std::vector<float>(embd, 1.0f));
    tensor("blk.0.attn_norm.bias", embd, 1, std::vector<float>(embd, 0.0f));
    const int q_dim = invalid ? embd : heads * head_dim, kv_dim = kv_heads * head_dim;
    const auto q = weights(embd * q_dim, 3), k = weights(embd * kv_dim, 4), v = weights(embd * kv_dim, 5);
    const auto qb = weights(q_dim, 6), kb = weights(kv_dim, 7), vb = weights(kv_dim, 8);
    if (merged) {
        auto qkv = q;
        qkv.insert(qkv.end(), k.begin(), k.end());
        qkv.insert(qkv.end(), v.begin(), v.end());
        tensor("blk.0.attn_qkv.weight", embd, q_dim + 2 * kv_dim, qkv);
        if (bias) {
            auto qkv_b = qb;
            qkv_b.insert(qkv_b.end(), kb.begin(), kb.end());
            qkv_b.insert(qkv_b.end(), vb.begin(), vb.end());
            tensor("blk.0.attn_qkv.bias", q_dim + 2 * kv_dim, 1, qkv_b);
        }
    } else {
        tensor("blk.0.attn_q.weight", embd, q_dim, q);
        tensor("blk.0.attn_k.weight", embd, kv_dim, k);
        tensor("blk.0.attn_v.weight", embd, kv_dim, v);
        if (bias) {
            tensor("blk.0.attn_q.bias", q_dim, 1, qb);
            tensor("blk.0.attn_k.bias", kv_dim, 1, kb);
            tensor("blk.0.attn_v.bias", kv_dim, 1, vb);
        }
    }
    tensor("blk.0.attn_output.weight", heads * head_dim, embd, weights(heads * head_dim * embd, 9));
    if (bias) { tensor("blk.0.attn_output.bias", embd, 1, weights(embd, 10)); }
    tensor("blk.0.ffn_norm.weight", embd, 1, std::vector<float>(embd, 1.0f));
    tensor("blk.0.ffn_norm.bias", embd, 1, std::vector<float>(embd, 0.0f));
    tensor("blk.0.ffn_up.weight", embd, 96, weights(embd * 96, 11));
    tensor("blk.0.ffn_down.weight", 96, embd, weights(96 * embd, 12));

    auto * file = std::tmpfile();
    check(file != nullptr, "tmpfile failed");
    check(gguf_write_to_file_ptr(meta, file, false), "writing synthetic GGUF failed");
    std::rewind(file);
    auto params = llama_model_default_params();
    params.n_gpu_layers = gpu ? 99 : 0;
    params.use_mmap = false;
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_type(gpu ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU), nullptr};
    check(devices[0] != nullptr, "requested backend unavailable");
    params.devices = devices;
    auto * model = llama_model_load_from_file_ptr(file, params);
    std::fclose(file);
    gguf_free(meta);
    ggml_free(data);
    return model;
}

static std::vector<float> infer(llama_model * model) {
    check(model != nullptr, "valid synthetic Nemotron GGUF failed to load");
    auto params = llama_context_default_params();
    params.n_ctx = 32;
    params.n_batch = params.n_ubatch = 8;
    params.n_threads = params.n_threads_batch = 2;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto * ctx = llama_init_from_model(model, params);
    check(ctx != nullptr, "context creation failed");
    llama_token prompt[] = {1, 2, 3, 4};
    check(llama_decode(ctx, llama_batch_get_one(prompt, 4)) == 0, "prefill failed");
    auto * logits = llama_get_logits(ctx);
    std::vector<float> result(logits, logits + vocab);
    llama_token token = 5;
    check(llama_decode(ctx, llama_batch_get_one(&token, 1)) == 0, "decode failed");
    logits = llama_get_logits(ctx);
    result.insert(result.end(), logits, logits + vocab);
    for (const auto v : result) { check(std::isfinite(v), "non-finite logits"); }
    llama_free(ctx);
    llama_model_free(model);
    return result;
}

int main(int argc, char ** argv) {
    try {
        const bool gpu = argc == 2 && std::string(argv[1]) == "--gpu";
        check(argc == 1 || gpu, "usage: test-nemotron-gqa [--gpu]");
        ggml_backend_load_all();
        llama_backend_init();
        for (int heads : {4, 6}) {
            for (bool bias : {false, true}) {
                const auto split = infer(make_model(heads, false, bias, false, gpu));
                const auto merged = infer(make_model(heads, true, bias, false, gpu));
                double max_diff = 0;
                for (size_t i = 0; i < split.size(); ++i) {
                    const double diff = std::abs(split[i] - merged[i]);
                    if (diff > max_diff) { max_diff = diff; }
                    check(diff <= 1e-5 + 1e-4 * std::abs(split[i]), "split/merged logits differ");
                }
                std::printf("heads=%d bias=%d split/merged max_diff=%g OK\n", heads, bias, max_diff);
            }
        }
        for (bool merged : {false, true}) {
            auto * invalid = make_model(6, merged, false, true, gpu);
            const bool rejected = invalid == nullptr;
            llama_model_free(invalid);
            check(rejected, "invalid Q projection shape was accepted");
        }
        llama_backend_free();
        std::puts("Nemotron GQA regression: OK (8 valid fixtures, 2 invalid fixtures)");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
