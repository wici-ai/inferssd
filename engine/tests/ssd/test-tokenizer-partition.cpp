// SPDX-License-Identifier: Apache-2.0
#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

static constexpr llama_token byte_base = 11;

static void append_bytes(std::vector<llama_token> & tokens, const std::string & text) {
    for (unsigned char byte : text) { tokens.push_back(byte == ' ' ? 10 : byte_base + byte); }
}

static void write_vocabulary(const char * path, const std::string & name) {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
    require(bool(file), "could not allocate vocabulary");
    gguf_set_val_str(file.get(), "general.architecture", "llama");
    gguf_set_val_str(file.get(), "general.name", name.c_str());
    gguf_set_val_str(file.get(), "tokenizer.ggml.model", "llama");
    std::vector<std::string> words = {
        "<unk>", "<s>", "</s>", "<|endoftext|>", "<AA>", "<A>", "<B>", "[MASK]", "<U>", "<A><B>", "\xe2\x96\x81",
    };
    std::vector<int32_t> types = {2, 3, 3, 3, 3, 3, 3, 3, 4, 3, 1};
    for (unsigned byte = 0; byte < 256; ++byte) {
        char spelling[7];
        std::snprintf(spelling, sizeof(spelling), "<0x%02X>", byte);
        words.emplace_back(spelling);
        types.push_back(LLAMA_TOKEN_TYPE_BYTE);
    }
    std::vector<const char *> pointers;
    for (const auto & word : words) { pointers.push_back(word.c_str()); }
    std::vector<float> scores(words.size(), 0.0f);
    gguf_set_arr_str(file.get(), "tokenizer.ggml.tokens", pointers.data(), pointers.size());
    gguf_set_arr_data(file.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_arr_data(file.get(), "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_val_u32(file.get(), "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(file.get(), "tokenizer.ggml.eos_token_id", 2);
    gguf_set_val_u32(file.get(), "tokenizer.ggml.unknown_token_id", 0);
    gguf_set_val_bool(file.get(), "tokenizer.ggml.add_bos_token", true);
    gguf_set_val_bool(file.get(), "tokenizer.ggml.add_eos_token", true);
    gguf_set_val_bool(file.get(), "tokenizer.ggml.add_space_prefix", false);
    require(gguf_write_to_file(file.get(), path, false), "could not write vocabulary");
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text,
        bool parse_special = true, bool add_special = false,
        const std::vector<llama_tokenize_input_span> * spans = nullptr) {
    std::vector<llama_token> tokens(text.size() + 2);
    const int32_t count = spans
        ? llama_tokenize_with_input_spans(vocab, text.data(), int32_t(text.size()), tokens.data(), int32_t(tokens.size()),
              add_special, spans->data(), int32_t(spans->size()))
        : llama_tokenize(vocab, text.data(), int32_t(text.size()), tokens.data(), int32_t(tokens.size()),
              add_special, parse_special);
    require(count >= 0, "tokenization failed");
    tokens.resize(size_t(count));
    return tokens;
}

static void check(const llama_vocab * vocab, const std::string & text, const std::vector<llama_token> & expected,
        bool parse_special = true, bool add_special = false,
        const std::vector<llama_tokenize_input_span> * spans = nullptr) {
    const auto actual = tokenize(vocab, text, parse_special, add_special, spans);
    if (actual != expected) {
        std::cerr << "unexpected IDs for " << text << "\nactual:";
        for (const auto token : actual) { std::cerr << ' ' << token; }
        std::cerr << "\nexpected:";
        for (const auto token : expected) { std::cerr << ' ' << token; }
        std::cerr << '\n';
        throw std::runtime_error("token IDs changed");
    }
}

static void repeat_case(const llama_vocab * vocab, size_t repeats, size_t iterations, bool controls = true) {
    std::string text;
    std::vector<llama_token> expected;
    for (size_t i = 0; i < repeats; ++i) {
        if (controls) {
            text += "x<A>y<B>";
            expected.insert(expected.end(), {byte_base + 'x', 5, byte_base + 'y', 6});
        } else {
            text += "plain x y\n";
            append_bytes(expected, "plain x y\n");
        }
    }
    check(vocab, text, expected);
    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        require(tokenize(vocab, text) == expected, "repeated prompt token IDs changed");
    }
    const double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cout << "{\"repeats\":" << repeats << ",\"iterations\":" << iterations
              << ",\"tokens\":" << expected.size() << ",\"milliseconds\":" << milliseconds << "}\n";
}

int main(int argc, char ** argv) {
    try {
        require(argc >= 3, "expected scenario and fixture path");
        const std::string scenario = argv[1];
        struct remove_file {
            const char * path;
            ~remove_file() { std::remove(path); }
        } cleanup{argv[2]};
        write_vocabulary(argv[2], scenario == "lstrip" ? "modern-bert" : scenario == "rstrip" ? "phi-3" : "partition");
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
        if (scenario == "boundaries") {
            check(vocab, "", {});
            check(vocab, "<A>", {5});
            check(vocab, "<A><A><A>", {5, 5, 5});
            check(vocab, "a<A>b", {byte_base + 'a', 5, byte_base + 'b'});
            check(vocab, "<AA>x<A>y<B>z<AA>", {4, byte_base + 'x', 5, byte_base + 'y', 6, byte_base + 'z', 4});
            check(vocab, "<A><B><A>", {9, 5});
            check(vocab, "x<A>", {1, byte_base + 'x', 5, 2}, true, true);
        } else if (scenario == "ordinary") {
            std::vector<llama_token> expected;
            append_bytes(expected, "plain \xce\xbc \xe6\xbc\xa2\xe5\xad\x97 text");
            check(vocab, "plain \xce\xbc \xe6\xbc\xa2\xe5\xad\x97 text", expected);
            expected.clear();
            append_bytes(expected, "<A>");
            expected.push_back(8);
            append_bytes(expected, "<B>");
            check(vocab, "<A><U><B>", expected, false);
        } else if (scenario == "protected") {
            std::vector<llama_token> expected = {5, byte_base + 'x'};
            append_bytes(expected, "<A>y");
            expected.push_back(5);
            const std::vector<llama_tokenize_input_span> spans = {{4, 3}};
            check(vocab, "<A>x<A>y<A>", expected, true, false, &spans);
            std::vector<llama_token> literal;
            append_bytes(literal, "<A>");
            const std::vector<llama_tokenize_input_span> boundary = {{1, 1}};
            check(vocab, "<A>", literal, true, false, &boundary);
            const std::vector<llama_tokenize_input_span> empty = {{0, 0}, {3, 0}};
            check(vocab, "<A><A>", {5, 5}, true, false, &empty);
        } else if (scenario == "lstrip") {
            require(llama_vocab_get_attr(vocab, 7) & LLAMA_TOKEN_ATTR_LSTRIP, "LSTRIP fixture not active");
            check(vocab, "x \t [MASK]y", {byte_base + 'x', 7, byte_base + 'y'});
            check(vocab, "  [MASK][MASK]z", {7, 7, byte_base + 'z'});
            check(vocab, "[MASK] [MASK]", {7, 7});
        } else if (scenario == "rstrip") {
            require(llama_vocab_get_attr(vocab, 5) & LLAMA_TOKEN_ATTR_RSTRIP, "RSTRIP fixture not active");
            check(vocab, "x<A> \t\n<B>   ", {byte_base + 'x', 5, 6});
            check(vocab, "<A>   ", {5});
            check(vocab, "<A> \t <A>z", {5, 5, byte_base + 'z'});
        } else if (scenario == "repeated") {
            repeat_case(vocab, 1000, 1);
        } else if (scenario == "benchmark" || scenario == "benchmark-ordinary") {
            require(argc == 5, "expected repetition and iteration counts");
            repeat_case(vocab, std::stoul(argv[3]), std::stoul(argv[4]), scenario == "benchmark");
        } else {
            throw std::runtime_error("unknown scenario");
        }
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
