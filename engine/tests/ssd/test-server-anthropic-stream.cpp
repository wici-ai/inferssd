// SPDX-License-Identifier: Apache-2.0

#include "server-context.h"
#include "server-http.h"
#include "gguf.h"

#include <cpp-httplib/httplib.h>
#include <algorithm>
#include <atomic>
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

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

// The real route needs a vocabulary and context. A single CPU layer supplies
// both without a download; the queue supplies the scripted results.
static void write_model(const fs::path & path) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> weights(ggml_init({1u << 20, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> model(gguf_init_empty(), gguf_free);
    require(weights && model, "could not create stream fixture");
    gguf_set_val_str(model.get(), "general.architecture", "llama");
    gguf_set_val_str(model.get(), "tokenizer.ggml.model", "llama");
    gguf_set_val_u32(model.get(), "llama.context_length", 128);
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
        char text[7];
        std::snprintf(text, sizeof(text), "<0x%02X>", i);
        words.emplace_back(text);
    }
    std::vector<const char *> tokens;
    for (const auto & word : words) {
        tokens.push_back(word.c_str());
    }
    std::vector<int32_t> types(words.size(), 6);
    types[0] = 2;
    types[1] = types[2] = 3;
    const std::vector<float> scores(words.size(), 0.0f);
    gguf_set_arr_str(model.get(), "tokenizer.ggml.tokens", tokens.data(), tokens.size());
    gguf_set_arr_data(model.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_arr_data(model.get(), "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());

    const auto add = [&](const char * name, int rows) {
        auto * tensor = rows ? ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, 32, rows) :
                               ggml_new_tensor_1d(weights.get(), GGML_TYPE_F32, 32);
        ggml_set_name(tensor, name);
        auto * values = static_cast<float *>(tensor->data);
        std::fill(values, values + ggml_nelements(tensor), 0.25f);
        gguf_add_tensor(model.get(), tensor);
    };
    add("token_embd.weight", 259);
    add("output_norm.weight", 0);
    add("output.weight", 259);
    add("blk.0.attn_norm.weight", 0);
    add("blk.0.attn_q.weight", 32);
    add("blk.0.attn_k.weight", 32);
    add("blk.0.attn_v.weight", 32);
    add("blk.0.attn_output.weight", 32);
    add("blk.0.ffn_norm.weight", 0);
    add("blk.0.ffn_gate.weight", 32);
    add("blk.0.ffn_down.weight", 32);
    add("blk.0.ffn_up.weight", 32);
    require(gguf_write_to_file(model.get(), path.string().c_str(), false), "could not write stream fixture");
}

enum class outcome { normal, early_error, late_error };

struct wire_probe {
    std::mutex mutex;
    std::condition_variable changed;
    size_t pings = 0;
    std::atomic<outcome> selected{outcome::normal};

    void send_result(server_response & queue, const server_task & request) {
        const outcome scenario = selected.load();
        const auto fail = [&] {
            auto result = std::make_unique<server_task_result_error>();
            result->id = request.id;
            result->index = request.index;
            result->err_type = ERROR_TYPE_INVALID_REQUEST;
            result->err_msg = "scripted request error";
            queue.send(std::move(result));
        };
        if (scenario == outcome::early_error) {
            fail();
            return;
        }

        auto begin = std::make_unique<server_task_result_cmpl_partial>();
        begin->id = request.id;
        begin->index = request.index;
        begin->is_begin = true;
        begin->res_type = TASK_RESPONSE_TYPE_ANTHROPIC;
        queue.send(std::move(begin));

        // Cap the wait even if the stream stopped sending heartbeat events.
        {
            std::unique_lock<std::mutex> lock(mutex);
            const size_t count = scenario == outcome::normal ? 2 : 1;
            changed.wait_for(lock, 6s, [&] { return pings >= count; });
        }
        if (scenario == outcome::late_error) {
            fail();
            return;
        }

        auto partial = std::make_unique<server_task_result_cmpl_partial>();
        partial->id = request.id;
        partial->index = request.index;
        partial->res_type = TASK_RESPONSE_TYPE_ANTHROPIC;
        partial->content = "Ready.";
        partial->n_decoded = 1;
        partial->oaicompat_model = "test-model";
        partial->oaicompat_cmpl_id = "wire-test";
        queue.send(std::move(partial));

        auto final = std::make_unique<server_task_result_cmpl_final>();
        final->id = request.id;
        final->index = request.index;
        final->stream = true;
        final->res_type = TASK_RESPONSE_TYPE_ANTHROPIC;
        final->stop = STOP_TYPE_EOS;
        final->n_decoded = 1;
        final->oaicompat_model = "test-model";
        final->oaicompat_cmpl_id = "wire-test";
        queue.send(std::move(final));
    }

    void check(httplib::Client & client, outcome scenario) {
        selected.store(scenario);
        {
            std::lock_guard<std::mutex> lock(mutex);
            pings = 0;
        }
        std::string received;
        constexpr char ping[] = "event: ping\ndata: {\"type\":\"ping\"}\n\n";
        size_t scan = 0;
        const auto response = client.Post("/v1/messages", httplib::Headers{},
            R"({"model":"test-model","max_tokens":8,"stream":true,"messages":[{"role":"user","content":"Hello"}]})",
            "application/json", [&](const char * data, size_t size) {
                received.append(data, size);
                std::lock_guard<std::mutex> lock(mutex);
                while (true) {
                    const auto at = received.find(ping, scan);
                    if (at == std::string::npos) { break; }
                    ++pings;
                    scan = at + sizeof(ping) - 1;
                    changed.notify_all();
                }
                return true;
            });
        require(bool(response), "HTTP request did not complete");
        if (scenario == outcome::early_error) {
            require(response->status == 400, "error before slot start must remain HTTP 400");
            require(received.find("scripted request error") != std::string::npos && pings == 0,
                    "early error did not arrive as JSON without a ping");
            return;
        }

        require(response->status == 200 &&
                response->get_header_value("Content-Type").find("text/event-stream") == 0,
                "slot-start did not open an SSE response");
        if (scenario == outcome::late_error) {
            const auto error = received.find("event: error\n");
            require(pings >= 1 && error != std::string::npos && received.find(ping) < error,
                    "late error must follow the ping inside the HTTP 200 stream");
            require(received.find("\"type\":\"error\",\"error\":") != std::string::npos &&
                    received.find("scripted request error") != std::string::npos,
                    "late error did not use the Anthropic envelope");
            return;
        }

        const auto first = received.find(ping);
        const auto second = first == std::string::npos ? first : received.find(ping, first + sizeof(ping) - 1);
        const auto start = received.find("event: message_start\n");
        const auto stop = received.find("event: message_stop\n");
        require(pings >= 2 && second < start && start < stop && stop != std::string::npos,
                "expected two pings before a normal message start and stop");
        require(received.find("Ready.") != std::string::npos, "completion text did not reach the client");
    }
};

static void test_wire(const fs::path & model) {
    common_params params;
    params.model.path = model.string();
    params.devices = { nullptr };
    params.n_gpu_layers = 0;
    params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 1;
    params.n_ctx = 128;
    params.n_parallel = 1;
    params.fit_params = params.warmup = false;
    params.hostname = "127.0.0.1";
    params.port = 0;
    params.sse_ping_interval = 1;

    server_context context;
    require(context.load_model(params), "could not load stream fixture");
    auto channel = context.get_response_reader();
    wire_probe probe;
    channel.queue_tasks.on_new_task([&](server_task && task) {
        if (task.type == SERVER_TASK_TYPE_COMPLETION) {
            probe.send_result(channel.queue_results, task);
        }
    });
    channel.queue_tasks.on_update_slots([] {});

    server_routes routes(params, context);
    routes.init_routes();
    routes.update_meta(context);
    server_http_context http;
    require(http.init(params), "could not initialize HTTP server");
    http.post("/v1/messages", routes.post_anthropic_messages);
    http.is_ready.store(true);
    require(http.start(), "could not start HTTP server");

    std::thread worker([&] { context.start_loop(); });
    struct cleanup {
        server_context & context;
        server_http_context & http;
        std::thread & worker;
        ~cleanup() {
            http.stop();
            http.thread.join();
            context.terminate();
            worker.join();
        }
    } stop{context, http, worker};
    httplib::Client client("127.0.0.1", http.port);
    client.set_read_timeout(9s);
    probe.check(client, outcome::early_error);
    probe.check(client, outcome::normal);
    probe.check(client, outcome::late_error);
}

int main() {
    const auto model = fs::temp_directory_path() / ("inferssd-anthropic-stream-" +
        std::to_string(std::random_device{}()) + ".gguf");
    struct cleanup {
        const fs::path & path;
        ~cleanup() { std::error_code ec; fs::remove(path, ec); }
    } remove_model{model};
    try {
        ggml_backend_load_all();
        llama_backend_init();
        write_model(model);
        test_wire(model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
