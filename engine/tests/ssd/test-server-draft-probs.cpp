// SPDX-License-Identifier: Apache-2.0
// A completion that drafts reports every token's probabilities as one that does not: each token a verified draft
// keeps has those of its own row, whether the draft token was accepted, rejected or followed the draft.
#include "server-context.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

static constexpr int vocab_size = 3 + 256;
static constexpr int width = 8;
static constexpr llama_token tok_a = 3 + 'A';
static constexpr llama_token tok_b = 3 + 'B';
static constexpr llama_token tok_c = 3 + 'C';

static void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// A bigram model: the attention and the feed-forward block add nothing, so a token's logits depend on that token
// alone. A is followed by B, B by C and C by A, each with two runners-up of its own.
static void write_model(const fs::path & path) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> weights(ggml_init({1u << 20, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> model(gguf_init_empty(), gguf_free);
    require(weights && model, "could not create the model fixture");
    gguf_set_val_str(model.get(), "general.architecture", "llama");
    gguf_set_val_str(model.get(), "tokenizer.ggml.model", "llama");
    gguf_set_val_u32(model.get(), "llama.context_length", 256);
    gguf_set_val_u32(model.get(), "llama.embedding_length", width);
    gguf_set_val_u32(model.get(), "llama.block_count", 1);
    gguf_set_val_u32(model.get(), "llama.feed_forward_length", width);
    gguf_set_val_u32(model.get(), "llama.attention.head_count", 1);
    gguf_set_val_f32(model.get(), "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(model.get(), "tokenizer.ggml.unknown_token_id", 0);
    gguf_set_val_u32(model.get(), "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(model.get(), "tokenizer.ggml.eos_token_id", 2);

    std::vector<std::string> words = { "<unk>", "<s>", "</s>" };
    for (unsigned i = 0; i < 256; ++i) {
        char word[7];
        std::snprintf(word, sizeof(word), "<0x%02X>", i);
        words.emplace_back(word);
    }
    std::vector<const char *> tokens;
    for (const auto & word : words) {
        tokens.push_back(word.c_str());
    }
    std::vector<int32_t> types(vocab_size, 6);
    types[0] = 2;
    types[1] = types[2] = 3;
    const std::vector<float> scores(vocab_size, 0.0f);
    gguf_set_arr_str(model.get(), "tokenizer.ggml.tokens", tokens.data(), tokens.size());
    gguf_set_arr_data(model.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_arr_data(model.get(), "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());

    const auto add = [&](const char * name, int rows = 0, float value = 0.0f) {
        auto * tensor = rows ? ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, width, rows) :
                               ggml_new_tensor_1d(weights.get(), GGML_TYPE_F32, width);
        ggml_set_name(tensor, name);
        auto * values = static_cast<float *>(tensor->data);
        std::fill(values, values + ggml_nelements(tensor), value);
        gguf_add_tensor(model.get(), tensor);
        return values;
    };
    // A, B and C each have a component of their own; every other token shares the last one
    auto * embeddings = add("token_embd.weight", vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        embeddings[i * width + (i == tok_a ? 0 : i == tok_b ? 1 : i == tok_c ? 2 : width - 1)] = 1.0f;
    }
    add("output_norm.weight", 0, 1.0f);
    auto * output = add("output.weight", vocab_size);
    const auto next = [&](int component, llama_token first, llama_token second, llama_token third, float a, float b,
            float c) {
        output[first * width + component] = a;
        output[second * width + component] = b;
        output[third * width + component] = c;
    };
    next(0, tok_b, tok_c, tok_a, 3.0f, 1.5f, 0.5f);
    next(1, tok_c, tok_a, tok_b, 3.0f, 1.2f, 0.3f);
    next(2, tok_a, tok_b, tok_c, 3.0f, 1.0f, 0.7f);
    add("blk.0.attn_norm.weight", 0, 1.0f);
    add("blk.0.attn_q.weight", width);
    add("blk.0.attn_k.weight", width);
    add("blk.0.attn_v.weight", width);
    add("blk.0.attn_output.weight", width);
    add("blk.0.ffn_norm.weight", 0, 1.0f);
    add("blk.0.ffn_gate.weight", width);
    add("blk.0.ffn_down.weight", width);
    add("blk.0.ffn_up.weight", width);
    require(gguf_write_to_file(model.get(), path.string().c_str(), false), "could not write the model fixture");
}

struct server_run {
    server_context server;
    std::thread worker;

    explicit server_run(common_params params) {
        require(server.load_model(params), "could not load the model fixture");
        worker = std::thread([this] { server.start_loop(); });
    }

    ~server_run() {
        if (worker.joinable()) {
            server.terminate();
            worker.join();
        }
    }
};

struct generated {
    llama_tokens tokens;
    std::vector<completion_token_output> probs;
    int drafted = 0;
    int accepted = 0;
};

// A greedy completion of C A B A A C A. Its tokens run B C A B ...; the first draft follows the prompt, where A B is
// followed by A A, and is rejected; the later ones follow the tokens the model wrote, and are accepted.
static generated complete(const common_params & params, bool post_sampling) {
    server_run run(params);
    auto reader = run.server.get_response_reader();
    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.id = reader.get_new_id();
    task.tokens = server_tokens({ tok_c, tok_a, tok_b, tok_a, tok_a, tok_c, tok_a }, false);
    task.params.n_predict = 24;
    task.params.stream = false;
    task.params.return_tokens = true;
    task.params.sampling.temp = 0;
    task.params.sampling.n_probs = 3;
    task.params.post_sampling_probs = post_sampling;
    reader.post_task(std::move(task));
    const auto deadline = std::chrono::steady_clock::now() + 30s;
    const auto result = reader.next([&] { return std::chrono::steady_clock::now() >= deadline; });
    require(result && !result->is_error(), "the completion failed");
    const auto * done = dynamic_cast<const server_task_result_cmpl_final *>(result.get());
    require(done != nullptr, "the completion did not finish");
    return { done->tokens, done->probs_output, done->timings.draft_n, done->timings.draft_n_accepted };
}

static void compare(const common_params & params, bool post_sampling) {
    const char * mode = post_sampling ? "post-sampling" : "pre-sampling";
    common_params plain = params;
    plain.speculative.type = COMMON_SPECULATIVE_TYPE_NONE;
    const auto base = complete(plain, post_sampling);
    const auto drafts = complete(params, post_sampling);

    require(drafts.drafted > drafts.accepted && drafts.accepted > 0,
            std::string(mode) + ": the fixture must both reject and accept draft tokens");
    // a completion that drafts can stop a few tokens short of n_predict: it counts a block's tokens before it sends them
    const size_t n = drafts.tokens.size();
    require(n >= 16 && n <= base.tokens.size() && std::equal(drafts.tokens.begin(), drafts.tokens.end(), base.tokens.begin()),
            std::string(mode) + ": drafts changed the greedy tokens");
    require(base.probs.size() == base.tokens.size() && drafts.probs.size() == n,
            std::string(mode) + ": a token is missing from the probabilities");
    for (size_t i = 0; i < n; ++i) {
        const auto & want = base.probs[i];
        const auto & got = drafts.probs[i];
        const std::string at = std::string(mode) + ", token " + std::to_string(i) + ": ";
        require(!want.probs.empty(), at + "the fixture must report probabilities without drafts");
        require(got.tok == want.tok, at + "another token");
        require(got.probs.size() == want.probs.size(), at + "a drafted token has other or no probabilities");
        require(std::fabs(got.prob - want.prob) <= 1e-5f, at + "another probability of the token");
        for (size_t k = 0; k < want.probs.size(); ++k) {
            require(got.probs[k].tok == want.probs[k].tok && std::fabs(got.probs[k].prob - want.probs[k].prob) <= 1e-5f,
                    at + "another probability of a candidate");
        }
    }
}

int main() {
    const auto work = fs::temp_directory_path() / ("inferssd-draft-probs-" + std::to_string(std::random_device{}()));
    try {
        require(fs::create_directory(work), "could not create the test directory");
        struct cleanup {
            const fs::path & path;
            ~cleanup() { std::error_code error; fs::remove_all(path, error); }
        } remove_fixture{work};
        ggml_backend_load_all();
        llama_backend_init();

        const fs::path model = work / "model.gguf";
        write_model(model);
        common_params params;
        params.model.path = model.string();
        params.n_ctx = 256;
        params.n_batch = params.n_ubatch = 64;
        params.n_parallel = 1;
        params.n_gpu_layers = 0;
        params.devices = { nullptr };
        params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 1;
        params.n_ctx_checkpoints = 0;
        params.warmup = false;
        params.fit_params = false;
        // drafts of two tokens: the two that followed the last two tokens where they last occurred
        params.speculative.type = COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE;
        params.speculative.ngram_simple.size_n = 2;
        params.speculative.ngram_simple.size_m = 2;

        compare(params, false);
        compare(params, true);
        std::printf("ok\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
