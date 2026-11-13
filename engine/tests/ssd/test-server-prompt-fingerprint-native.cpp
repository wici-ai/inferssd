// SPDX-License-Identifier: Apache-2.0
// Actual server disk-cache compatibility with a fully resident tiny DeepSeek41 model.
#include "server-context.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <thread>

static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
struct output { std::map<int, float> logits; int reused; };
static output request(server_context & server, const llama_tokens & tokens) {
    auto reader = server.get_response_reader();
    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.id = reader.get_new_id();
    task.tokens = server_tokens(tokens, false);
    task.params.stream = false;
    task.params.return_tokens = true;
    task.params.n_predict = 1;
    task.params.sampling.seed = 42;
    task.params.sampling.temp = 1;
    task.params.sampling.n_probs = 259;
    task.params.sampling.top_k = 0;
    task.params.sampling.top_p = 1;
    task.params.sampling.min_p = 0;
    task.params.sampling.ignore_eos = true;
    reader.post_task(std::move(task));
    auto result = reader.next([] { return false; });
    require(bool(result), "request returned no result");
    if (result->is_error()) throw std::runtime_error(result->to_json().dump());
    auto * complete = dynamic_cast<server_task_result_cmpl_final *>(result.get());
    require(complete, "missing completion result");
    output out{};
    out.reused = complete->n_prompt_tokens_cache;
    return out;
}
static output run(const char * model, const std::filesystem::path & dir, float rope,
                  const llama_tokens & tokens, bool seed_cache) {
    common_params params;
    params.model.path = model;
    params.n_gpu_layers = 0;
    params.n_ctx = 128;
    params.n_batch = params.n_ubatch = 32;
    params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 2;
    params.n_parallel = 1;
    params.fit_params = false;
    params.warmup = false;
    params.swa_full = true;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.rope_freq_base = rope;
    params.cache_disk = dir.string();
    params.chat_template = "chatml";
    server_context server;
    require(server.load_model(params), "server failed to load model");
    std::thread loop([&] { server.start_loop(); });
    try {
        auto result = request(server, tokens);
        if (seed_cache) (void)request(server, {200, 201, 202, 203, 204, 205, 206, 207});
        server.terminate();
        loop.join();
        auto * ctx = server.get_llama_context();
        const auto * values = llama_get_logits_ith(ctx, -1);
        require(values, "missing actual completion logits");
        const int count = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        for (int i = 0; i < count; ++i) {
            require(std::isfinite(values[i]), "nonfinite actual completion logit");
            result.logits[i] = values[i];
        }
        return result;
    } catch (...) {
        server.terminate();
        if (loop.joinable()) loop.join();
        throw;
    }
}
static double difference(const output & left, const output & right) {
    require(left.logits.size() == right.logits.size(), "logit counts differ");
    double out = 0;
    for (const auto & p : left.logits) out = std::max(out, std::abs(double(p.second) - right.logits.at(p.first)));
    return out;
}
int main(int argc, char ** argv) {
    try {
        require(argc == 3, "MODEL OUTPUT_DIR required");
        llama_backend_init();
        const std::filesystem::path root(argv[2]);
        require(std::filesystem::create_directory(root), "output directory must be new");
        llama_tokens prefix;
        for (int i = 0; i < 24; ++i) prefix.push_back(3 + (i * 37) % 180);
        const auto seeded = run(argv[1], root / "reused", 10000, prefix, true);
        auto longer = prefix; longer.push_back(88); longer.push_back(89);
        const auto same = run(argv[1], root / "reused", 10000, longer, false);
        const auto cold_same = run(argv[1], root / "cold-same", 10000, longer, false);
        const auto changed = run(argv[1], root / "reused", 100, longer, false);
        const auto cold_changed = run(argv[1], root / "cold-changed", 100, longer, false);
        const double same_diff = difference(same, cold_same);
        const double changed_diff = difference(changed, cold_changed);
        const double cold_diff = difference(cold_same, cold_changed);
        std::cout << "PROMPT_POSITION_RESULT same_reused=" << same.reused << " same_diff=" << same_diff
                  << " changed_reused=" << changed.reused << " changed_diff=" << changed_diff
                  << " cold_diff=" << cold_diff << " logits=" << seeded.logits.size() << '\n';
        constexpr double tolerance = 1e-5;
        require(same.reused == int(prefix.size()) && same_diff <= tolerance, "same-position cache control differs");
        require(cold_diff > 1e-5, "model fixture does not exercise the changed RoPE setting");
        require(changed.reused == 0 && changed_diff <= tolerance, "changed position reused incompatible cached state");
        std::cout << "PROMPT_POSITION_PASS\n";
        llama_backend_free();
    } catch (const std::exception & e) { std::cerr << "PROMPT_POSITION_FAIL " << e.what() << '\n'; return 1; }
}
