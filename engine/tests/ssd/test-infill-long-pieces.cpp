// SPDX-License-Identifier: Apache-2.0
#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

static std::vector<std::string> words() {
    std::string unicode;
    for (int i = 0; i < 180; ++i) { unicode += "\xe7\x95\x8c"; }
    return {
        "<unk>", "<s>", "</s>", "test", "tested", std::string(512, 'a'),
        std::string(513, 'a'), std::string(513, 'a') + "b", "other",
        "test" + std::string(600, 'a'), unicode, unicode + "x",
        std::string(2048, 'a'), std::string(2048, 'a') + "b",
    };
}

static void write_vocabulary(const char * path) {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
    require(bool(file), "could not allocate vocabulary");
    gguf_set_val_str(file.get(), "general.architecture", "llama");
    gguf_set_val_str(file.get(), "general.name", "infill-pieces");
    gguf_set_val_str(file.get(), "tokenizer.ggml.model", "llama");
    auto tokens = words();
    std::vector<int32_t> types(tokens.size(), LLAMA_TOKEN_TYPE_NORMAL);
    types[0] = LLAMA_TOKEN_TYPE_UNKNOWN;
    types[1] = types[2] = LLAMA_TOKEN_TYPE_CONTROL;
    for (unsigned byte = 0; byte < 256; ++byte) {
        char spelling[7];
        std::snprintf(spelling, sizeof(spelling), "<0x%02X>", byte);
        tokens.emplace_back(spelling);
        types.push_back(LLAMA_TOKEN_TYPE_BYTE);
    }
    std::vector<const char *> pointers;
    for (const auto & token : tokens) { pointers.push_back(token.c_str()); }
    std::vector<float> scores(tokens.size(), 0.0f);
    gguf_set_arr_str(file.get(), "tokenizer.ggml.tokens", pointers.data(), pointers.size());
    gguf_set_arr_data(file.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_arr_data(file.get(), "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_val_u32(file.get(), "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(file.get(), "tokenizer.ggml.eos_token_id", 2);
    gguf_set_val_u32(file.get(), "tokenizer.ggml.unknown_token_id", 0);
    gguf_set_val_bool(file.get(), "tokenizer.ggml.add_space_prefix", false);
    require(gguf_write_to_file(file.get(), path, false), "could not write vocabulary");
}

static void check(llama_sampler * sampler, llama_token first, llama_token second, bool merge) {
    // Distinct logits make the sort order independent of tie ordering.
    std::vector<llama_token_data> data = {{first, 0.2f, 0.0f}, {second, 0.0f, 0.0f}};
    llama_token_data_array candidates = {data.data(), data.size(), -1, false};
    llama_sampler_apply(sampler, &candidates);
    require(candidates.size == (merge ? 1u : 2u), "unexpected number of surviving candidates");
    require(candidates.data[0].id == first, "higher-probability token changed");
    const float expected = merge ? 1.0f : 1.0f / (1.0f + std::exp(-0.2f));
    require(std::fabs(candidates.data[0].p - expected) < 1e-6f, "first probability changed");
    if (!merge) {
        require(candidates.data[1].id == second, "unrelated token was lost");
        require(std::fabs(candidates.data[1].p - (1.0f - expected)) < 1e-6f, "second probability changed");
    }
}

int main(int argc, char ** argv) {
    try {
        require(argc == 3, "expected scenario and fixture path");
        const std::string scenario = argv[1];
        struct remove_file {
            const char * path;
            ~remove_file() { std::remove(path); }
        } cleanup{argv[2]};
        write_vocabulary(argv[2]);
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
        llama_backend_init();
        ggml_backend_dev_t no_devices[] = {nullptr};
        auto params = llama_model_default_params();
        params.devices = no_devices;
        params.n_gpu_layers = 0;
        params.vocab_only = true;
        params.use_mmap = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[2], params), llama_model_free);
        require(bool(model), "could not load vocabulary");
        const auto * vocab = llama_model_get_vocab(model.get());
        const auto tokens = words();
        for (size_t token = 3; token < tokens.size(); ++token) {
            char small[512];
            const int length = llama_token_to_piece(vocab, llama_token(token), small, sizeof(small), 0, false);
            require(length == (tokens[token].size() <= sizeof(small) ? int(tokens[token].size()) : -int(tokens[token].size())),
                    "unexpected public token-to-piece buffer requirement");
            std::vector<char> piece(tokens[token].size());
            require(llama_token_to_piece(vocab, llama_token(token), piece.data(), int(piece.size()), 0, false) == int(piece.size()),
                    "could not decode complete token piece");
            require(std::string(piece.begin(), piece.end()) == tokens[token], "token bytes changed");
        }
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
            llama_sampler_init_infill(vocab), llama_sampler_free);
        if (scenario == "short") {
            check(sampler.get(), 3, 4, true);
        } else if (scenario == "boundary") {
            check(sampler.get(), 5, 8, false);
        } else if (scenario == "first") {
            check(sampler.get(), 6, 8, false);
        } else if (scenario == "second") {
            check(sampler.get(), 3, 9, true);
        } else if (scenario == "both") {
            check(sampler.get(), 6, 7, true);
        } else if (scenario == "higher-long") {
            check(sampler.get(), 7, 6, true);
        } else if (scenario == "unicode") {
            check(sampler.get(), 10, 11, true);
        } else if (scenario == "reuse") {
            check(sampler.get(), 3, 4, true);
            check(sampler.get(), 6, 8, false);
            check(sampler.get(), 12, 13, true);
            check(sampler.get(), 3, 9, true);
            check(sampler.get(), 3, 4, true);
            std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> clone(
                llama_sampler_clone(sampler.get()), llama_sampler_free);
            check(clone.get(), 12, 13, true);
        } else if (scenario == "eog") {
            std::vector<llama_token_data> data = {{12, 0.0f, 0.0f}, {2, 10.0f, 0.0f}, {13, -1.0f, 0.0f}};
            llama_token_data_array candidates = {data.data(), data.size(), -1, false};
            llama_sampler_apply(sampler.get(), &candidates);
            require(candidates.size == 1 && candidates.data[0].id == 2 && candidates.data[0].p == 1.0f,
                    "end-of-generation preference changed");
        } else {
            throw std::runtime_error("unknown scenario");
        }
        sampler.reset();
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
