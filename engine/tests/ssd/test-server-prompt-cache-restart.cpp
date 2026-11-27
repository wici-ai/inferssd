// SPDX-License-Identifier: Apache-2.0
#include "server-context.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

static constexpr int prefix_tokens = 96;
static constexpr int prompt_tokens = 384;
static constexpr llama_token merged_token = 259;
static constexpr int vocab_size = merged_token + 1;
static constexpr int width = 32;
static constexpr llama_token base_token = 3 + 'A';
static constexpr llama_token adapted_token = 3 + 'B';

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Uniform attention makes the first output depend on values stored during prefill.
// The logits choose A for the base model and B with the active value adapter.
static void write_model(const fs::path & path) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> weights(ggml_init({1u << 20, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> model(gguf_init_empty(), gguf_free);
    require(weights && model, "could not create prompt-cache model fixture");
    gguf_set_val_str(model.get(), "general.architecture", "llama");
    gguf_set_val_str(model.get(), "tokenizer.ggml.model", "llama");
    gguf_set_val_u32(model.get(), "llama.context_length", 512);
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
    words.emplace_back("yy");
    std::vector<const char *> tokens;
    for (const auto & word : words) {
        tokens.push_back(word.c_str());
    }
    std::vector<int32_t> types(vocab_size, 6);
    types[0] = 2;
    types[1] = types[2] = 3;
    types[merged_token] = 1;
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
    auto * embeddings = add("token_embd.weight", vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        embeddings[i * width] = 1.0f;
    }
    add("output_norm.weight", 0, 1.0f);
    auto * output = add("output.weight", vocab_size);
    output[base_token * width] = 1.0f;
    output[adapted_token * width + 1] = 1.0f;
    add("blk.0.attn_norm.weight", 0, 1.0f);
    add("blk.0.attn_q.weight", width);
    add("blk.0.attn_k.weight", width);
    add("blk.0.attn_v.weight", width);
    add("blk.0.attn_output.weight", width)[width + 1] = 1.0f;
    add("blk.0.ffn_norm.weight", 0, 1.0f);
    add("blk.0.ffn_gate.weight", width);
    add("blk.0.ffn_down.weight", width);
    add("blk.0.ffn_up.weight", width);
    require(gguf_write_to_file(model.get(), path.string().c_str(), false), "could not write prompt-cache model fixture");
}

// A rank-one update sends the first input component to the second value component.
static void write_adapter(const fs::path & path, float value, std::optional<uint32_t> invocation = {}) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> weights(ggml_init({1u << 16, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> adapter(gguf_init_empty(), gguf_free);
    require(weights && adapter, "could not create LoRA fixture");
    gguf_set_val_str(adapter.get(), "general.type", "adapter");
    gguf_set_val_str(adapter.get(), "general.architecture", "llama");
    gguf_set_val_str(adapter.get(), "adapter.type", "lora");
    gguf_set_val_f32(adapter.get(), "adapter.lora.alpha", 1.0f);
    if (invocation) {
        gguf_set_arr_data(adapter.get(), "adapter.alora.invocation_tokens", GGUF_TYPE_UINT32, &*invocation, 1);
    }

    const auto add = [&](const char * name, int columns, int rows, int component, float value) {
        auto * tensor = ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, columns, rows);
        ggml_set_name(tensor, name);
        auto * values = static_cast<float *>(tensor->data);
        std::fill(values, values + ggml_nelements(tensor), 0.0f);
        values[component] = value;
        gguf_add_tensor(adapter.get(), tensor);
    };
    add("blk.0.attn_v.weight.lora_a", width, 1, 0, 1.0f);
    add("blk.0.attn_v.weight.lora_b", 1, width, 1, value);
    require(gguf_write_to_file(adapter.get(), path.string().c_str(), false), "could not write LoRA fixture");
}

static bool has_prompt_file(const fs::path & cache) {
    for (const auto & item : fs::directory_iterator(cache)) {
        if (item.path().extension() == ".prompt") {
            return true;
        }
    }
    return false;
}

struct server_run {
    server_context server;
    std::thread worker;

    explicit server_run(common_params params) {
        require(server.load_model(params), "could not load prompt-cache model fixture");
        worker = std::thread([this] { server.start_loop(); });
    }

    void stop() {
        if (worker.joinable()) {
            server.terminate();
            worker.join();
        }
    }

    ~server_run() {
        stop();
    }
};

static server_task_result_ptr receive(server_response_reader & reader) {
    const auto deadline = std::chrono::steady_clock::now() + 15s;
    auto result = reader.next([&] { return std::chrono::steady_clock::now() >= deadline; });
    require(result && !result->is_error(), "prompt-cache request failed");
    return result;
}

static llama_tokens make_prompt(llama_token suffix, int count = prompt_tokens) {
    llama_tokens prompt(count, suffix);
    std::fill_n(prompt.begin(), prefix_tokens, base_token);
    return prompt;
}

struct completion {
    int cached;
    llama_token token;
};

static void require_completion(completion actual, int cached, llama_token token, const char * message) {
    if (actual.cached != cached || actual.token != token) {
        std::fprintf(stderr, "Cached tokens: %d (expected %d), first token: %d (expected %d)\n",
                     actual.cached, cached, actual.token, token);
        throw std::runtime_error(message);
    }
}

static completion first_token(server_context & server, const llama_tokens & prompt, std::optional<float> scale,
        int id_slot, int id_adapter = 0, int n_shared = prefix_tokens, int n_cache_reuse = 0) {
    auto reader = server.get_response_reader();
    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.id = reader.get_new_id();
    task.id_slot = id_slot;
    task.tokens = server_tokens(prompt, false);
    task.points.n_shared = n_shared;
    task.params.n_predict = 1;
    task.params.n_cache_reuse = n_cache_reuse;
    task.params.stream = false;
    task.params.return_tokens = true;
    task.params.sampling.temp = 0;
    if (scale) {
        task.params.lora[id_adapter] = *scale;
    }
    reader.post_task(std::move(task));
    const auto result = receive(reader);
    const auto * done = dynamic_cast<const server_task_result_cmpl_final *>(result.get());
    require(done && done->tokens.size() == 1, "the request did not complete with exactly one token");
    return {done->n_prompt_tokens_cache, done->tokens.front()};
}

static completion first_token(server_context & server, std::optional<float> scale = {}, int id_slot = -1) {
    return first_token(server, make_prompt(3 + 'y'), scale, id_slot);
}

static completion first_token(const common_params & params, std::optional<float> scale = {}) {
    server_run restarted(params);
    return first_token(restarted.server, scale);
}

static void save_completed_prefix(const common_params & params, std::optional<float> scale = {}) {
    const fs::path cache = params.cache_disk;
    require(fs::create_directory(cache), "could not create prompt-cache directory");
    server_run original(params);
    first_token(original.server, llama_tokens(prefix_tokens, base_token), scale, -1, 0, 0);
    // A disjoint request reuses the one idle slot and saves its completed state to disk.
    first_token(original.server, llama_tokens(prefix_tokens, 3 + 'x'), scale, -1, 0, 0);
    require(has_prompt_file(cache), "the completed prefix was not saved on slot reuse");
}

static void test_import_replacement(common_params params, const fs::path & work) {
    const auto source = fs::directory_iterator(params.cache_disk)->path();
    require(source.extension() == ".prompt", "expected the saved prefix file");
    const auto dir = work / "imported-prefixes";
    require(fs::create_directory(dir), "could not create the prefix import directory");
    const auto imported = dir / "prefix.prompt";
    const auto replacement = work / "replacement.prompt";
    fs::copy_file(source, imported);
    const auto directory_time = fs::last_write_time(dir);

    params.prefix_cache = dir.string();
    params.cache_disk.clear();
    params.cache_ram_mib = 4;
    server_run original(params);
    const auto request = [&] {
        // Leave the shared prefix unspecified so a cold request does not save
        // another copy of it in the server's own cache.
        return first_token(original.server, make_prompt(3 + 'y'), {}, -1, 0, 0);
    };

    fs::copy_file(source, replacement);
    {
        // Keep the offsets and real KV payload intact, but replace the proven
        // base-model marker with an unrecognized value of the same length.
        std::fstream file(replacement, std::ios::binary | std::ios::in | std::ios::out);
        uint32_t fingerprint_size = 0;
        file.seekg(2 * sizeof(uint32_t));
        file.read(reinterpret_cast<char *>(&fingerprint_size), sizeof(fingerprint_size));
        require(bool(file) && fingerprint_size > 0, "could not read the prefix fingerprint length");
        file.seekp(3 * sizeof(uint32_t) + fingerprint_size - 1);
        file.put('0');
        file.close();
        require(bool(file), "could not write the invalid prefix marker");
    }
    fs::rename(replacement, imported);
    fs::last_write_time(dir, directory_time);
    require_completion(request(), 0, base_token, "a replacement must not inherit the earlier file's validation");

    auto reader = original.server.get_response_reader();
    server_task erase(SERVER_TASK_TYPE_SLOT_ERASE);
    erase.id = reader.get_new_id();
    erase.slot_action.id_slot = 0;
    reader.post_task(std::move(erase));
    const auto receipt = receive(reader);
    require(dynamic_cast<const server_task_result_slot_erase *>(receipt.get()), "could not erase the live slot");

    fs::copy_file(source, replacement);
    fs::rename(replacement, imported);
    fs::last_write_time(dir, directory_time + 2s);
    require_completion(request(), prefix_tokens, base_token, "the live server must load a valid same-path replacement");
}

static void slot_file(server_context & server, const fs::path & path, bool save) {
    auto reader = server.get_response_reader();
    server_task request(save ? SERVER_TASK_TYPE_SLOT_SAVE : SERVER_TASK_TYPE_SLOT_RESTORE);
    request.id = reader.get_new_id();
    request.slot_action = {0, path.filename().string(), path.string()};
    reader.post_task(std::move(request));
    const auto result = receive(reader);
    const auto * receipt = dynamic_cast<const server_task_result_slot_save_load *>(result.get());
    require(receipt && receipt->is_save == save && receipt->n_tokens == prompt_tokens && receipt->n_bytes > 0,
            "explicit slot operation did not contain the entire prompt");
}

static void test_explicit_slots(common_params params, const fs::path & work) {
    params.cache_disk.clear();
    params.cache_ram_mib = 4;
    params.slot_save_path = work.string();
    const auto base_path    = work / "base.slot";
    const auto adapted_path = work / "adapted.slot";
    const auto legacy_path  = work / "legacy.slot";

    auto with_base_only = params;
    with_base_only.lora_adapters.clear();
    {
        server_run original(with_base_only);
        require_completion(first_token(original.server), 0, base_token, "cold base request failed");
        slot_file(original.server, base_path, true);
    }
    {
        auto writer_params = params;
        writer_params.cache_ram_mib = 0;
        server_run original(writer_params);
        require_completion(first_token(original.server, 1.0f), 0, adapted_token, "cold adapted request failed");
        slot_file(original.server, adapted_path, true);
        original.stop();

        // The old explicit save contains native state and tokens with no adapter identity.
        const auto input = make_prompt(3 + 'y');
        require(llama_state_seq_save_file(original.server.get_llama_context(), legacy_path.string().c_str(), 0,
                    input.data(), input.size()) > 0, "could not write legacy slot fixture");
    }

    const auto alora = work / "alora.gguf";
    constexpr llama_token invocation = 3 + 'z';
    write_adapter(alora, 4.0f, invocation);
    auto with_alora = params;
    with_alora.lora_adapters.push_back({alora.string(), 0.0f, "", "", nullptr});
    const auto prompt = make_prompt(3 + 'y');
    auto invoked_prompt = prompt;
    invoked_prompt[prefix_tokens] = invocation;
    auto later_invocation = invoked_prompt;
    later_invocation[prompt_tokens - 4] = invocation;
    for (const int id_slot : {-1, 0}) {
        {
            server_run restarted(with_base_only);
            slot_file(restarted.server, base_path, false);
            require_completion(first_token(restarted.server, {}, id_slot), prompt_tokens - 1, base_token,
                    "a compatible base slot must retain its prefix when selected automatically or explicitly");
        }
        {
            server_run restarted(params);
            slot_file(restarted.server, adapted_path, false);
            require_completion(first_token(restarted.server, 1.0f, id_slot), prompt_tokens - 1, adapted_token,
                    "a slot saved with matching adapter weights and scale must reuse its prompt");
        }
        {
            server_run restarted(params);
            slot_file(restarted.server, adapted_path, false);
            require_completion(first_token(restarted.server, {}, id_slot), 0, base_token,
                    "a base request must recompute an adapted slot when selected automatically or explicitly");
        }
        {
            server_run restarted(with_alora);
            slot_file(restarted.server, adapted_path, false);
            require_completion(first_token(restarted.server, prompt, 1.0f, id_slot, 1), 0, base_token,
                    "a request without the aLoRA invocation must recompute the restored ordinary LoRA state");
            require_completion(first_token(restarted.server, invoked_prompt, 1.0f, id_slot, 1), prefix_tokens - 1, adapted_token,
                    "an invoked aLoRA must reuse the prefix computed without an adapter");
            // The last invocation moves near the end while leaving the raw prompt prefix almost intact.
            require_completion(first_token(restarted.server, later_invocation, 1.0f, id_slot, 1, prefix_tokens, 32),
                    prefix_tokens - 1, base_token, "moving the invocation must discard cached tokens after the old boundary");
        }
    }

    // An old adapted file remains unsafe even when the new server loads no adapter.
    server_run restarted(with_base_only);
    {
        auto reader = restarted.server.get_response_reader();
        server_task request(SERVER_TASK_TYPE_SLOT_RESTORE);
        request.id = reader.get_new_id();
        request.slot_action = {0, legacy_path.filename().string(), legacy_path.string()};
        reader.post_task(std::move(request));
        const auto deadline = std::chrono::steady_clock::now() + 15s;
        const auto result = reader.next([&] { return std::chrono::steady_clock::now() >= deadline; });
        require(result && result->is_error(), "legacy slot state without adapter provenance must be rejected");
    }
    require_completion(first_token(restarted.server, {}, 0), 0, base_token,
            "a rejected legacy restore must not populate the pinned slot");
}

static void test_live_slot_text_reuse(common_params params) {
    params.cache_disk.clear();
    params.cache_ram_mib = 0;
    params.slot_save_path.clear();

    // Use the same rendered prompt with two different token boundaries. The last
    // token is evaluated again for sampling after the 384 matching cached tokens.
    const auto split = make_prompt(3 + 'y', prompt_tokens + 1);
    auto merged = split;
    merged[prefix_tokens] = merged_token;
    merged.erase(merged.begin() + prefix_tokens + 1);
    for (const int id_slot : {-1, 0}) {
        server_run server(params);
        const auto * ctx = server.server.get_llama_context();
        require(common_detokenize(ctx, split, false) == common_detokenize(ctx, merged, false),
                "the two tokenizations must render the same prompt");
        require_completion(first_token(server.server, split, 1.0f, id_slot), 0, adapted_token, "cold adapted request failed");
        require_completion(first_token(server.server, merged, 1.0f, id_slot), prompt_tokens, adapted_token,
                "a matching live LoRA slot must reuse text when shared caching is disabled");
    }
}

static void test_restart(const fs::path & work) {
    const fs::path model = work / "model.gguf";
    const fs::path cache = work / "cache";
    write_model(model);
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

    save_completed_prefix(params);
    require_completion(first_token(params), prefix_tokens, base_token,
            "restart did not restore the exact shared prefix for a different suffix");
    test_import_replacement(params, work);

    const fs::path adapter = work / "adapter.gguf";
    write_adapter(adapter, 4.0f);
    params.lora_adapters.push_back({adapter.string(), 0.0f, "", "", nullptr});
    test_explicit_slots(params, work);
    test_live_slot_text_reuse(params);
    params.cache_disk.clear();
    const auto cold_base = first_token(params);
    const auto cold_adapted = first_token(params, 1.0f);
    const auto cold_low_scale = first_token(params, 0.03125f);
    require(cold_base.token == base_token && cold_adapted.token == adapted_token && cold_low_scale.token == base_token,
            "the fixture must distinguish the adapter and its scale in the first output token");

    params.cache_disk = cache.string();
    require_completion(first_token(params), prefix_tokens, cold_base.token,
            "loading a disabled adapter must keep base-model reuse");

    const fs::path adapted_cache = work / "adapted-cache";
    params.cache_disk = adapted_cache.string();
    save_completed_prefix(params, 1.0f);

    require_completion(first_token(params), 0, cold_base.token,
            "a default-scale request must match a cold base run after an adapted prefix was saved");
    require_completion(first_token(params, 0.03125f), 0, cold_low_scale.token,
            "a different nonzero scale must match a cold run");
    require_completion(first_token(params, 1.0f), prefix_tokens, cold_adapted.token,
            "matching adapters and scales must restore the full prefix and match a cold run");

    // Keep the path, metadata, size, and timestamp while replacing the adapter weights.
    const auto original_mtime = fs::last_write_time(adapter);
    write_adapter(adapter, 0.125f);
    fs::last_write_time(adapter, original_mtime);
    params.cache_disk.clear();
    const auto cold_replacement = first_token(params, 1.0f);
    require(cold_replacement.token == base_token, "the replacement adapter must produce a different token");
    params.cache_disk = adapted_cache.string();
    require_completion(first_token(params, 1.0f), 0, cold_replacement.token,
            "different adapter weights with the same scale must match a cold run");
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
