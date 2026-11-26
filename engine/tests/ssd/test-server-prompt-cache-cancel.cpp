// SPDX-License-Identifier: Apache-2.0
#include "server-context.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

static constexpr int prefix_tokens = 96;
static constexpr int vocab_size = 259;

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// One small dense layer and byte tokens. The regression needs real context state,
// but the generated text is immaterial and the fixture needs no downloaded model.
static void write_model(const fs::path & path) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> weights(ggml_init({1u << 20, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> model(gguf_init_empty(), gguf_free);
    require(weights && model, "could not create prompt-cache model fixture");
    gguf_set_val_str(model.get(), "general.architecture", "llama");
    gguf_set_val_str(model.get(), "tokenizer.ggml.model", "llama");
    gguf_set_val_u32(model.get(), "llama.context_length", 512);
    gguf_set_val_u32(model.get(), "llama.embedding_length", 32);
    gguf_set_val_u32(model.get(), "llama.block_count", 1);
    gguf_set_val_u32(model.get(), "llama.feed_forward_length", 32);
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

    const auto add = [&](const char * name, int rows = 0) {
        auto * tensor = rows ? ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, 32, rows) :
                               ggml_new_tensor_1d(weights.get(), GGML_TYPE_F32, 32);
        ggml_set_name(tensor, name);
        auto * values = static_cast<float *>(tensor->data);
        std::fill(values, values + ggml_nelements(tensor), 0.25f);
        gguf_add_tensor(model.get(), tensor);
    };
    add("token_embd.weight", vocab_size);
    add("output_norm.weight");
    add("output.weight", vocab_size);
    add("blk.0.attn_norm.weight");
    add("blk.0.attn_q.weight", 32);
    add("blk.0.attn_k.weight", 32);
    add("blk.0.attn_v.weight", 32);
    add("blk.0.attn_output.weight", 32);
    add("blk.0.ffn_norm.weight");
    add("blk.0.ffn_gate.weight", 32);
    add("blk.0.ffn_down.weight", 32);
    add("blk.0.ffn_up.weight", 32);
    require(gguf_write_to_file(model.get(), path.string().c_str(), false), "could not write prompt-cache model fixture");
}

static bool has_prompt_file(const fs::path & cache) {
    for (const auto & item : fs::directory_iterator(cache)) {
        if (item.path().extension() == ".prompt") {
            return true;
        }
    }
    return false;
}

// If the server got ahead of the reader, pause before computing the first batch
// after the file appears. The reader can then queue cancellation before the
// server gets through the rest of the prompt or generates an answer.
struct suffix_gate {
    fs::path cache;
    std::mutex mutex;
    std::condition_variable condition;
    bool released = false;

    explicit suffix_gate(fs::path cache) : cache(std::move(cache)) {}

    static bool observe(ggml_tensor *, bool ask, void * user_data) {
        if (ask) {
            auto & gate = *static_cast<suffix_gate *>(user_data);
            std::unique_lock<std::mutex> lock(gate.mutex);
            if (!gate.released && has_prompt_file(gate.cache)) {
                gate.condition.wait(lock, [&] { return gate.released; });
            }
        }
        return false;
    }

    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        condition.notify_all();
    }
};

struct server_run {
    server_context server;
    suffix_gate * gate;
    std::thread worker;

    explicit server_run(common_params & params, suffix_gate * gate = nullptr) : gate(gate) {
        if (gate) {
            params.cb_eval = suffix_gate::observe;
            params.cb_eval_user_data = gate;
        }
        require(server.load_model(params), "could not load prompt-cache model fixture");
        worker = std::thread([this] { server.start_loop(); });
    }

    ~server_run() {
        if (gate) {
            gate->release();
        }
        server.terminate();
        worker.join();
    }
};

static server_task_result_ptr receive(server_response_reader & reader) {
    const auto deadline = std::chrono::steady_clock::now() + 15s;
    auto result = reader.next([&] { return std::chrono::steady_clock::now() >= deadline; });
    require(result && !result->is_error(), "prompt-cache request failed");
    return result;
}

static void post_prompt(server_response_reader & reader, llama_token suffix) {
    llama_tokens prompt(384, suffix);
    std::fill_n(prompt.begin(), prefix_tokens, 3 + 'A');
    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.id = reader.get_new_id();
    task.tokens = server_tokens(prompt, false);
    task.n_shared_prefix = prefix_tokens;
    task.params.n_predict = 1;
    task.params.stream = true;
    task.params.return_progress = true;
    reader.post_task(std::move(task));
}

static void test_restart(const fs::path & work) {
    const fs::path model = work / "model.gguf";
    const fs::path cache = work / "cache";
    write_model(model);
    require(fs::create_directory(cache), "could not create prompt-cache directory");
    common_params params;
    params.model.path = model.string();
    params.n_ctx = 512;
    params.n_batch = params.n_ubatch = 128;
    params.n_parallel = 1;
    params.n_gpu_layers = 0;
    params.devices = { nullptr };
    params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 1;
    params.n_ctx_checkpoints = 0;
    params.cache_disk = cache.string();
    params.cache_disk_mib = 4;
    params.warmup = false;
    params.fit_params = false;
    {
        suffix_gate gate(cache);
        server_run first(params, &gate);
        auto reader = first.server.get_response_reader();
        post_prompt(reader, 3 + 'x');
        // The server first acknowledges the request, before decoding any tokens.
        receive(reader);
        const auto result = receive(reader);
        const auto * progress = dynamic_cast<const server_task_result_cmpl_partial *>(result.get());
        require(progress && progress->is_progress && progress->progress.cache == 0 &&
                progress->progress.processed == prefix_tokens, "first prefill batch did not end at the shared prefix");
        require(has_prompt_file(cache), "first successful prefix batch was not saved");

        reader.stop();
        gate.release();
        // This task runs after cancellation and before we destroy the server.
        auto metrics = first.server.get_response_reader();
        server_task request(SERVER_TASK_TYPE_METRICS);
        request.id = metrics.get_new_id();
        metrics.post_task(std::move(request));
        const auto receipt = receive(metrics);
        const auto * state = dynamic_cast<const server_task_result_metrics *>(receipt.get());
        require(state && state->n_idle_slots == 1 && state->n_tokens_predicted_total == 0,
                "the first request was not canceled during prefill");
    }
    params.cb_eval = nullptr;
    params.cb_eval_user_data = nullptr;
    {
        server_run restarted(params);
        auto reader = restarted.server.get_response_reader();
        post_prompt(reader, 3 + 'y');
        const auto result = receive(reader);
        const auto * progress = dynamic_cast<const server_task_result_cmpl_partial *>(result.get());
        require(progress && progress->is_progress && progress->progress.cache == prefix_tokens,
                "restart did not restore the exact shared prefix for a different suffix");
        reader.stop();
    }
}

int main() {
    const auto work = fs::temp_directory_path() / ("inferssd-prefix-restart-" + std::to_string(std::random_device{}()));
    try {
        require(fs::create_directory(work), "could not create prompt-cache test directory");
        struct cleanup {
            const fs::path & path;
            ~cleanup() { std::error_code error; fs::remove_all(path, error); }
        } remove_fixture{work};
        ggml_backend_load_all();
        llama_backend_init();
        test_restart(work);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
