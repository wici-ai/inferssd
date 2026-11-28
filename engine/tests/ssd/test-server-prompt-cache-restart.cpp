// SPDX-License-Identifier: Apache-2.0
#include "server-context.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
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
// With long_rope, a llama model with long and short RoPE factors (LongRoPE) whose queries and keys see the tokens:
// its attention, and the first output with it, depends on the factors the context selects. `width` is the embedding
// and head size.
static void write_model(const fs::path & path, bool sliding_window = false, bool long_rope = false,
                        int width = ::width) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> weights(ggml_init({1u << 20, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> model(gguf_init_empty(), gguf_free);
    require(weights && model, "could not create prompt-cache model fixture");
    const std::string arch = sliding_window ? "gemma2" : "llama";
    const auto key = [&](const char * name) { return arch + "." + name; };
    gguf_set_val_str(model.get(), "general.architecture", arch.c_str());
    gguf_set_val_str(model.get(), "tokenizer.ggml.model", "llama");
    gguf_set_val_u32(model.get(), key("context_length").c_str(), long_rope ? 1024 : 512);
    gguf_set_val_u32(model.get(), key("embedding_length").c_str(), width);
    gguf_set_val_u32(model.get(), key("block_count").c_str(), sliding_window ? 2 : 1);
    gguf_set_val_u32(model.get(), key("feed_forward_length").c_str(), width);
    gguf_set_val_u32(model.get(), key("attention.head_count").c_str(), 1);
    gguf_set_val_f32(model.get(), key("attention.layer_norm_rms_epsilon").c_str(), 1e-5f);
    if (sliding_window) {
        gguf_set_val_u32(model.get(), "gemma2.attention.sliding_window", 16);
    }
    if (long_rope) {
        // the long factors past a sequence context of 256
        gguf_set_val_str(model.get(), "llama.rope.scaling.type", "longrope");
        gguf_set_val_u32(model.get(), "llama.rope.scaling.original_context_length", 256);
    }
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

    const auto add = [&](const std::string & name, int rows = 0, float value = 0.0f) {
        auto * tensor = rows ? ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, width, rows) :
                               ggml_new_tensor_1d(weights.get(), GGML_TYPE_F32, width);
        ggml_set_name(tensor, name.c_str());
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
    if (sliding_window) {
        // Gemma 2 ties its output weights to the input embeddings.
        embeddings[base_token * width] = 2.0f;
        // Layer 0 copies the suffix marker into channel 1. Layer 1 reads that channel
        // from cached rows and activates a distinct output token when the suffix leaked.
        embeddings[adapted_token * width + 1] = 1.0f;
        embeddings[merged_token * width + 2] = 10.0f;
    } else {
        auto * output = add("output.weight", vocab_size);
        output[base_token * width] = 1.0f;
        output[adapted_token * width + 1] = 1.0f;
    }
    if (long_rope) {
        // tokens and logits of no particular pattern
        for (int i = 0; i < vocab_size * width; ++i) {
            embeddings[i] = std::sin(0.7f * (i / width) + 1.3f * (i % width));
        }
        auto * output = static_cast<float *>(ggml_get_tensor(weights.get(), "output.weight")->data);
        for (int i = 0; i < vocab_size * width; ++i) {
            // logits for the printable ASCII bytes only: another token ends the reply or leaves it incomplete text
            const int token = i / width;
            const bool printable = token >= 3 + ' ' && token <= 3 + '~';
            output[i] = printable ? std::cos(0.9f * token + 0.5f * (i % width)) : 0.0f;
        }
        for (const auto & [name, factor] : { std::pair<const char *, float>{ "rope_factors_long.weight", 32.0f },
                                             std::pair<const char *, float>{ "rope_factors_short.weight", 1.0f } }) {
            auto * factors = ggml_new_tensor_1d(weights.get(), GGML_TYPE_F32, width / 2);
            ggml_set_name(factors, name);
            std::fill_n(static_cast<float *>(factors->data), width / 2, factor);
            gguf_add_tensor(model.get(), factors);
        }
    }
    for (int layer = 0; layer < (sliding_window ? 2 : 1); ++layer) {
        const auto block = "blk." + std::to_string(layer) + ".";
        add(block + "attn_norm.weight", 0, 1.0f);
        auto * queries = add(block + "attn_q.weight", width);
        auto * keys = add(block + "attn_k.weight", width);
        auto * values = add(block + "attn_v.weight", width);
        auto * output = add(block + "attn_output.weight", width);
        output[(layer + 1) * width + 1] = 1.0f;
        if (long_rope) {
            for (int i = 0; i < width; ++i) {
                queries[i * width + i] = keys[i * width + i] = values[i * width + i] = output[i * width + i] = 1.0f;
            }
        }
        if (sliding_window) {
            values[width + 1] = 1.0f;
            add(block + "post_attention_norm.weight", 0, 1.0f);
            add(block + "post_ffw_norm.weight", 0, 1.0f);
        }
        add(block + "ffn_norm.weight", 0, 1.0f);
        add(block + "ffn_gate.weight", width);
        add(block + "ffn_down.weight", width);
        add(block + "ffn_up.weight", width);
    }
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
    float probability;
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
    task.params.sampling.n_probs = 1;
    if (scale) {
        task.params.lora[id_adapter] = *scale;
    }
    reader.post_task(std::move(task));
    const auto result = receive(reader);
    const auto * done = dynamic_cast<const server_task_result_cmpl_final *>(result.get());
    require(done && done->tokens.size() == 1, "the request did not complete with exactly one token");
    require(done->probs_output.size() == 1, "missing greedy token probability");
    return {done->n_prompt_tokens_cache, done->tokens.front(), done->probs_output.front().prob};
}

// Saves slot 0 to `path` or restores it from there; the number of tokens of the state.
static size_t slot_action(server_context & server, server_task_type type, const fs::path & path) {
    auto reader = server.get_response_reader();
    server_task task(type);
    task.id = reader.get_new_id();
    task.slot_action = {0, path.filename().string(), path.string()};
    reader.post_task(std::move(task));
    const auto result = receive(reader);
    const auto * receipt = dynamic_cast<const server_task_result_slot_save_load *>(result.get());
    require(receipt != nullptr, "the slot was not saved or restored");
    return receipt->n_tokens;
}

// The tokens a request generates, which may shift the context to make room: n_discard tokens after the first go.
static llama_tokens generate(server_context & server, const llama_tokens & prompt, int n_predict, int n_discard) {
    auto reader = server.get_response_reader();
    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.id = reader.get_new_id();
    task.id_slot = 0;
    task.tokens = server_tokens(prompt, false);
    task.params.n_predict = n_predict;
    task.params.n_discard = n_discard;
    task.params.stream = false;
    task.params.return_tokens = true;
    task.params.sampling.temp = 0;
    reader.post_task(std::move(task));
    const auto result = receive(reader);
    const auto * done = dynamic_cast<const server_task_result_cmpl_final *>(result.get());
    require(done && (int) done->tokens.size() == n_predict, "the request did not generate every token");
    return done->tokens;
}

static void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

static completion first_token(server_context & server, std::optional<float> scale = {}, int id_slot = -1) {
    return first_token(server, make_prompt(3 + 'y'), scale, id_slot);
}

static completion first_token(const common_params & params, std::optional<float> scale = {}) {
    server_run restarted(params);
    return first_token(restarted.server, scale);
}

// Processes the prompt, with a shared prefix of n_shared tokens, and then a disjoint prompt on the server's one slot.
static void write_completed_prefix(const common_params & params, std::optional<float> scale,
        const llama_tokens & prompt, int n_shared = 0) {
    require(fs::create_directory(params.cache_disk), "could not create prompt-cache directory");
    server_run original(params);
    first_token(original.server, prompt, scale, -1, 0, n_shared);
    // A disjoint request reuses the one idle slot and saves its completed state to disk.
    first_token(original.server, llama_tokens(prefix_tokens, 3 + 'x'), scale, -1, 0, 0);
}

static void save_completed_prefix(const common_params & params, std::optional<float> scale = {},
        const llama_tokens & prompt = llama_tokens(prefix_tokens, base_token), bool saved = true) {
    write_completed_prefix(params, scale, prompt);
    require(has_prompt_file(params.cache_disk) == saved,
            saved ? "the completed prefix was not saved on slot reuse" : "a server without causal attention saved a state");
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

// A control vector that adds `value` to the first channel of layer 1's output.
static void write_control_vector(const fs::path & path, float value) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> weights(ggml_init({1u << 16, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> vector(gguf_init_empty(), gguf_free);
    require(weights && vector, "could not create control vector fixture");
    auto * direction = ggml_new_tensor_1d(weights.get(), GGML_TYPE_F32, width);
    ggml_set_name(direction, "direction.1");
    auto * values = static_cast<float *>(direction->data);
    std::fill(values, values + width, 0.0f);
    values[0] = value;
    gguf_add_tensor(vector.get(), direction);
    require(gguf_write_to_file(vector.get(), path.string().c_str(), false), "could not write control vector fixture");
}

// A state of 90 tokens whose last 16, a window, are the suffix marker, and a prompt that shares its first 74 tokens and
// then parts from it: rows that depend on the suffix choose another first token.
static llama_tokens window_saved_prompt() {
    llama_tokens saved_prompt(90, base_token);
    std::fill(saved_prompt.begin() + 74, saved_prompt.end(), adapted_token);
    return saved_prompt;
}

static llama_tokens window_parted_prompt() {
    llama_tokens prompt(75, base_token);
    prompt.back() = 3 + 'y';
    return prompt;
}

static void require_cold_probability(completion actual, completion cold) {
    require(std::abs(std::log(actual.probability) - std::log(cold.probability)) < 1e-5f,
            "cached greedy log probability differs from cold inference");
}

static void make_non_causal(common_params & params) {
    require(string_parse_kv_override("gemma2.attention.causal=bool:false", params.kv_overrides),
            "invalid attention override");
    params.kv_overrides.emplace_back();
}

// A slot with non-causal attention serves no later prompt: its rows depend on the tokens after them in their batch,
// which a cold prompt that parts earlier never saw and one that goes on sees with the tokens it adds. It is not cut
// back where it still holds every row of the window, and not extended either.
static void test_sliding_window_non_causal_slot(common_params params) {
    params.cache_disk.clear();
    params.cache_ram_mib = 0;
    make_non_causal(params);
    const auto saved_prompt = window_saved_prompt();
    auto extended = saved_prompt;
    extended.push_back(3 + 'y');
    for (const auto & prompt : {window_parted_prompt(), extended}) {
        const auto cold = [&] {
            server_run server(params);
            return first_token(server.server, prompt, {}, 0, 0, 0);
        }();
        server_run live(params);
        first_token(live.server, saved_prompt, {}, 0, 0, 0);
        const auto actual = first_token(live.server, prompt, {}, 0, 0, 0);
        require_completion(actual, 0, cold.token, "a non-causal slot served a later prompt");
        require_cold_probability(actual, cold);
    }
}

// Without causal attention a row also depends on the tokens after it in its batch, so a state saved at a checkpoint or
// at the end of a shared prefix inside the batch holds rows that saw the rest of it. A server with such attention saves
// no cached state, and a reader with the same settings, from its prefix caches or its own cache on disk, evaluates the
// prompt as a cold server does.
static void test_sliding_window_non_causal_cache(common_params params, const fs::path & work) {
    make_non_causal(params);
    params.cache_disk.clear();
    params.cache_ram_mib = 4;
    llama_tokens written(86, base_token);
    written.resize(90, adapted_token);
    llama_tokens prompt(86, base_token);
    prompt.push_back(3 + 'y');
    const auto request = [&](const common_params & config) {
        server_run server(config);
        return first_token(server.server, prompt, {}, -1, 0, 0);
    };
    const auto cold = request(params);
    for (bool checkpoint : {true, false}) {
        // the state after 86 tokens: a checkpoint 4 tokens before the end, or the end of the shared prefix
        auto writer = params;
        writer.n_ctx_checkpoints = checkpoint ? 4 : 0;
        const auto dir = work / (checkpoint ? "non-causal-checkpoint" : "non-causal-shared-prefix");
        writer.cache_disk = dir.string();
        write_completed_prefix(writer, {}, written, checkpoint ? 0 : 86);
        for (bool imported : {true, false}) {
            auto reader = params;
            reader.prefix_cache = imported ? dir.string() : "";
            reader.cache_disk = imported ? "" : dir.string();
            const auto actual = request(reader);
            require_completion(actual, 0, cold.token, "a non-causal reader took a cached state");
            require_cold_probability(actual, cold);
        }
        require(!has_prompt_file(dir), "a server without causal attention saved a cached prompt");
    }
}

struct prompt_file_header {
    std::string fingerprint;
    llama_tokens tokens;
};

static prompt_file_header read_prompt_header(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    uint32_t head[3] = {};
    in.read(reinterpret_cast<char *>(head), sizeof(head));
    prompt_file_header header;
    header.fingerprint.resize(head[2] <= 4096 ? head[2] : 0);
    in.read(header.fingerprint.data(), header.fingerprint.size());
    uint32_t n_tokens = 0;
    in.read(reinterpret_cast<char *>(&n_tokens), sizeof(n_tokens));
    header.tokens.resize(n_tokens <= 4096 ? n_tokens : 0);
    in.read(reinterpret_cast<char *>(header.tokens.data()), header.tokens.size() * sizeof(llama_token));
    require(bool(in) && head[0] == 0x4643504c && header.fingerprint.size() == head[2] &&
            header.tokens.size() == n_tokens, "could not read a cached prompt header");
    return header;
}

static std::vector<fs::path> prompt_files(const fs::path & dir) {
    std::vector<fs::path> files;
    for (const auto & item : fs::directory_iterator(dir)) {
        if (item.path().extension() == ".prompt") {
            files.push_back(item.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

// A context shift moves the rows after the tokens it discards, and those rows still depend on them: the state then
// serves only its own generation. It is not saved, from the shift on, and a later prompt on the slot reuses none of
// it past the tokens the shift kept, though it starts with the tokens the state now lists. The same tokens computed
// without a shift serve such a prompt as before.
static void test_sliding_window_context_shift(common_params params, const fs::path & work) {
    params.n_ctx = 256;
    params.cache_disk.clear();
    params.cache_ram_mib = 4;
    // the suffix marker in the 16 tokens before the moved rows, which the window of the first of them holds
    llama_tokens written(140, base_token);
    written.resize(161, adapted_token);
    written.resize(250, base_token);
    // a text that another prompt spells with one token, in the rows the shift moves
    written[200] = written[201] = 3 + 'y';
    const int n_discard = 160;
    auto shift_writer = params;
    shift_writer.ctx_shift = true;
    // what the slot lists after the shift: the tokens it keeps, the rest of the prompt and the generated tokens
    llama_tokens shifted;
    const auto shift = [&](server_context & server) {
        const auto generated = generate(server, written, 12, n_discard);
        // the server keeps the first token, which the tokenizer adds
        const int n_keep = llama_vocab_get_add_bos(llama_model_get_vocab(llama_get_model(server.get_llama_context())));
        llama_tokens tokens(written.begin(), written.begin() + n_keep);
        tokens.insert(tokens.end(), written.begin() + n_keep + n_discard, written.end());
        tokens.insert(tokens.end(), generated.begin(), generated.end());
        require(shifted.empty() || shifted == tokens, "the same prompt must generate the same tokens");
        shifted = tokens;
        return n_keep;
    };
    {
        // the slot itself, and the slot saved explicitly and restored
        auto config = shift_writer;
        config.cache_ram_mib = 0;
        config.slot_save_path = work.string();
        const auto saved_slot = work / "shifted.slot";
        server_run live(config);
        const int n_keep = shift(live.server);
        require(slot_action(live.server, SERVER_TASK_TYPE_SLOT_SAVE, saved_slot) >= shifted.size() - 1,
                "the shifted slot was not saved");
        llama_tokens parted(shifted.begin(), shifted.begin() + 32);
        parted.push_back(3 + 'y');
        const auto cold = [&] {
            server_run server(config);
            return first_token(server.server, parted, {}, 0, 0, 0);
        }();
        const auto actual = first_token(live.server, parted, {}, 0, 0, 0);
        require(actual.cached <= n_keep, "a slot reused rows that a context shift moved");
        require_completion(actual, actual.cached, cold.token, "a slot reused rows that a context shift moved");
        require_cold_probability(actual, cold);
        live.stop();
        server_run restored(config);
        slot_action(restored.server, SERVER_TASK_TYPE_SLOT_RESTORE, saved_slot);
        require_completion(first_token(restored.server, parted, {}, 0, 0, 0), 0, cold.token,
                           "a slot restored from a save after a context shift reused its rows");
    }
    // A prompt that ends by spelling the text of moved rows otherwise keeps its own ids, as the slot reuses none of
    // those rows: its last token is the one it sent.
    for (const int id_slot : { -1, 0 }) {
        auto config = shift_writer;
        config.cache_ram_mib = 0;
        const size_t at = std::find(shifted.begin(), shifted.end(), 3 + 'y') - shifted.begin();
        require(at + 1 < shifted.size() && shifted[at + 1] == 3 + 'y', "expected the two-token text after the shift");
        llama_tokens respelled(shifted.begin(), shifted.begin() + at);
        respelled.push_back(merged_token);
        const auto cold = [&] {
            server_run server(config);
            return first_token(server.server, respelled, {}, id_slot, 0, 0);
        }();
        server_run live(config);
        shift(live.server);
        const auto actual = first_token(live.server, respelled, {}, id_slot, 0, 0);
        require_completion(actual, actual.cached, cold.token, "a slot rewrote a prompt's ids to rows a shift moved");
        require_cold_probability(actual, cold);
    }
    shift_writer.cache_disk = (work / "shifted-writer").string();
    require(fs::create_directory(shift_writer.cache_disk), "could not create prompt-cache directory");
    {
        server_run writer(shift_writer);
        shift(writer.server);
        // a disjoint prompt takes the slot, whose state goes to the prompt cache if it can serve another prompt
        first_token(writer.server, llama_tokens(prefix_tokens, 3 + 'x'), {}, -1, 0, 0);
    }

    auto extended = shifted;
    extended.push_back(3 + 'y');
    const auto request = [&](const common_params & config, const llama_tokens & prompt) {
        server_run server(config);
        return first_token(server.server, prompt, {}, -1, 0, 0);
    };
    const auto cold_extended = request(params, extended);
    require(cold_extended.cached == 0, "a cold request must evaluate every token");

    // the same tokens without a shift: the extended prompt reuses the whole state
    auto writer = params;
    writer.cache_disk = (work / "unshifted-writer").string();
    save_completed_prefix(writer, {}, shifted);
    for (const auto & config : { shift_writer, writer }) {
        const bool moved = config.ctx_shift;
        for (bool imported : { true, false }) {
            // each reader starts from the writer's files alone
            int n_readers = 0;
            const auto reader = [&] {
                const auto copy = config.cache_disk + "-reader-" + std::to_string(imported) +
                                  std::to_string(n_readers++);
                fs::copy(config.cache_disk, copy);
                auto out = params;
                out.prefix_cache = imported ? copy : "";
                out.cache_disk = imported ? "" : copy;
                return out;
            };
            const auto actual_extended = request(reader(), extended);
            require_completion(actual_extended, moved ? 0 : (int) shifted.size(), cold_extended.token,
                               moved ? "a state saved after a context shift was reused" : "incorrect exact reuse");
            require_cold_probability(actual_extended, cold_extended);
        }
    }
    for (const auto & path : prompt_files(shift_writer.cache_disk)) {
        const auto tokens = read_prompt_header(path).tokens;
        require(tokens.empty() || tokens.front() != base_token, "a state was saved after a context shift");
    }
}

// Without causal attention a slot keeps the ids a prompt spells its text with as well: the same text in other ids is
// other rows. A prompt that ends in "yy" as one token after one that ended in two "y" tokens is evaluated as on a
// fresh slot, whether it picks the slot or names it.
static void test_sliding_window_non_causal_text(common_params params) {
    params.cache_disk.clear();
    params.cache_ram_mib = 0;
    make_non_causal(params);
    llama_tokens split(prefix_tokens, base_token);
    split.push_back(3 + 'y');
    split.push_back(3 + 'y');
    auto merged = split;
    merged.resize(prefix_tokens);
    merged.push_back(merged_token);
    for (const int id_slot : { -1, 0 }) {
        const auto cold = [&] {
            server_run server(params);
            return first_token(server.server, merged, {}, id_slot, 0, 0);
        }();
        server_run server(params);
        const auto * ctx = server.server.get_llama_context();
        require(common_detokenize(ctx, split, false) == common_detokenize(ctx, merged, false),
                "the two tokenizations must render the same prompt");
        first_token(server.server, split, {}, id_slot, 0, 0);
        const auto actual = first_token(server.server, merged, {}, id_slot, 0, 0);
        require_completion(actual, 0, cold.token, "a non-causal slot rewrote the prompt's ids");
        require_cold_probability(actual, cold);
    }
}

// A quantized KV cache stores its rows Hadamard-rotated, or with LLAMA_ATTN_ROT_DISABLE as they are: the same layout
// in another basis. A writer that stored them the other way serves no reader, one that stored them the same way does.
static void test_sliding_window_kv_rotation(common_params params, const fs::path & work) {
    // heads of 64, which the rotation needs
    const auto model = work / "sliding-window-64.gguf";
    write_model(model, true, false, 64);
    params.model.path = model.string();
    params.cache_type_k = params.cache_type_v = GGML_TYPE_Q8_0;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.cache_disk.clear();
    params.cache_ram_mib = 4;
    // rows that are not zero, whose basis tells
    auto saved_prompt = window_saved_prompt();
    std::fill(saved_prompt.begin() + 10, saved_prompt.begin() + 15, adapted_token);
    auto prompt = saved_prompt;
    prompt.push_back(3 + 'y');
    const auto request = [&](const common_params & config) {
        server_run server(config);
        return first_token(server.server, prompt, {}, -1, 0, 0);
    };
    const auto cold = request(params);
    for (bool rotated : { true, false }) {
        auto writer = params;
        writer.cache_disk = (work / (rotated ? "kv-rotated" : "kv-unrotated")).string();
        set_env("LLAMA_ATTN_ROT_DISABLE", rotated ? nullptr : "1");
        save_completed_prefix(writer, {}, saved_prompt);
        set_env("LLAMA_ATTN_ROT_DISABLE", nullptr);
        for (bool imported : { true, false }) {
            auto reader = params;
            reader.prefix_cache = imported ? writer.cache_disk : "";
            reader.cache_disk = imported ? "" : writer.cache_disk;
            const auto actual = request(reader);
            require_completion(actual, rotated ? (int) saved_prompt.size() : 0, cold.token,
                               "rows stored in another basis were reused, or in the same one not");
            require_cold_probability(actual, cold);
        }
    }
}

// Rows computed with other settings (an overridden window, another RoPE base, a control vector)
// serve only a reader with the same settings, after a restart and as a prefix cache alike.
static void test_sliding_window_writer_settings(common_params params, const fs::path & work) {
    const auto saved_prompt = window_saved_prompt();
    // a prompt that the saved state is an exact prefix of
    auto prompt = saved_prompt;
    prompt.push_back(3 + 'y');
    const int n_saved = saved_prompt.size();
    const auto request = [&](const common_params & config) {
        server_run reader(config);
        return first_token(reader.server, prompt, {}, -1, 0, 0);
    };
    params.cache_disk.clear();
    params.cache_ram_mib = 4;
    const auto cold = request(params);
    require(cold.cached == 0, "a cold request must evaluate every token");

    // the prefix cache first: a writable cache removes the files of other fingerprints
    const auto check = [&](common_params reader, const fs::path & dir, int expected, completion baseline) {
        for (bool imported : {true, false}) {
            reader.prefix_cache = imported ? dir.string() : "";
            reader.cache_disk = imported ? "" : dir.string();
            auto actual = request(reader);
            require_completion(actual, expected, baseline.token, "incorrect prefix after restart");
            require_cold_probability(actual, baseline);
        }
    };
    const auto vector = work / "control-vector.gguf";
    write_control_vector(vector, 4.0f);
    // (a writer without causal attention saves no state: test_sliding_window_non_causal_cache)
    for (const std::string kind : {"default", "short-window", "rope", "control-vector"}) {
        auto writer = params;
        if (kind == "short-window") {
            require(string_parse_kv_override("gemma2.attention.sliding_window=int:8", writer.kv_overrides),
                    "invalid attention override");
            writer.kv_overrides.emplace_back();
        } else if (kind == "rope") {
            writer.rope_freq_base = 32;
        } else if (kind == "control-vector") {
            writer.control_vectors.push_back({1.0f, vector.string()});
        }
        const auto dir = work / ("writer-" + kind);
        writer.cache_disk = dir.string();
        save_completed_prefix(writer, {}, saved_prompt);
        for (const auto & path : prompt_files(dir)) {
            const auto fingerprint = read_prompt_header(path).fingerprint;
            require((fingerprint.rfind("prefix-cache 1|settings=", 0) == 0) == (kind != "default"),
                    "a state names the settings that change its values, and only those");
        }
        if (kind != "default") {
            auto same = writer;
            same.cache_disk.clear();
            check(same, dir, n_saved, request(same));
        }
        check(params, dir, kind == "default" ? n_saved : 0, cold);
    }
}

static void test_sliding_window(common_params params, const fs::path & work) {
    const auto model = work / "sliding-window.gguf";
    write_model(model, true);
    params.model.path = model.string();
    test_sliding_window_writer_settings(params, work);
    test_sliding_window_kv_rotation(params, work);
    test_sliding_window_non_causal_slot(params);
    test_sliding_window_non_causal_cache(params, work);
    test_sliding_window_non_causal_text(params);
    test_sliding_window_context_shift(params, work);

    // The attention type of a request: only a causal slot is cut back to the tokens a prompt shares with it, and only
    // a causal server keeps a state.
    for (bool causal : {true, false}) {
        params.attention_type = causal ? LLAMA_ATTENTION_TYPE_CAUSAL : LLAMA_ATTENTION_TYPE_NON_CAUSAL;
        params.cache_disk = (work / (causal ? "causal-cache" : "non-causal-cache")).string();
        save_completed_prefix(params, {}, llama_tokens(prefix_tokens, base_token), causal);
        server_run restarted(params);
        // past the window, where the slot still holds every row the shared tokens attend to
        llama_tokens prompt(41, base_token);
        prompt.back() = adapted_token;
        first_token(restarted.server, prompt, {}, 0, 0, 0);
        prompt.pop_back();
        prompt.back() = adapted_token;
        const auto live = first_token(restarted.server, prompt, {}, 0, 0, 0);
        require(live.cached == (causal ? 39 : 0), "live reuse ignored the attention mode");
    }
}

// A model with long and short RoPE factors (LongRoPE) rotates its keys by the long ones when a sequence's context is
// past the model's original one and by the short ones otherwise: a writer whose context selects the other factors
// serves no reader, one whose context selects the same ones serves it, from its prefix caches and its own cache alike.
static void test_long_rope(common_params params, const fs::path & work) {
    const auto model = work / "long-rope.gguf";
    write_model(model, false, true);
    params.model.path = model.string();
    params.cache_disk.clear();
    params.cache_ram_mib = 4;
    params.n_ctx = 512;
    // F32 rows and no flash attention: the attention sees the tokens, and a prompt's last token then reads reused rows
    // as a cold pass reads its own, without the F16 rounding or tiling that differ between them
    params.cache_type_k = params.cache_type_v = GGML_TYPE_F32;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_tokens saved_prompt(prefix_tokens);
    for (int i = 0; i < prefix_tokens; ++i) {
        saved_prompt[i] = base_token + i % 22;
    }
    auto prompt = saved_prompt;
    prompt.push_back(4);
    const auto request = [&](const common_params & config) {
        server_run server(config);
        return first_token(server.server, prompt, {}, -1, 0, 0);
    };
    const auto cold = request(params);
    auto short_context = params;
    short_context.n_ctx = 256;
    const auto cold_short = request(short_context);
    require(cold.cached == 0 && cold_short.cached == 0 &&
            std::abs(std::log(cold.probability) - std::log(cold_short.probability)) > 1e-3f,
            "the fixture's first output must depend on the RoPE factors");
    for (const int n_ctx : { 768, 256 }) {
        auto writer = params;
        writer.n_ctx = n_ctx;
        writer.cache_disk = (work / ("long-rope-" + std::to_string(n_ctx))).string();
        save_completed_prefix(writer, {}, saved_prompt);
        for (bool imported : { true, false }) {
            auto reader = params;
            reader.prefix_cache = imported ? writer.cache_disk : "";
            reader.cache_disk = imported ? "" : writer.cache_disk;
            const auto actual = request(reader);
            require_completion(actual, n_ctx > 256 ? prefix_tokens : 0, cold.token,
                               "rows of other RoPE factors were reused, or of the same ones not");
            require_cold_probability(actual, cold);
        }
    }
}

// A slot saved explicitly after a context shift moved its rows is restored to go on with its own generation only: no
// prompt reuses the moved rows, on a model without a sliding window too, whose slot is cut back to any shared prefix.
static void test_shifted_slot_file(common_params params, const fs::path & work) {
    params.n_ctx = 256;
    params.ctx_shift = true;
    params.cache_disk.clear();
    params.cache_ram_mib = 0;
    params.slot_save_path = work.string();
    const auto path = work / "shifted-llama.slot";
    const llama_tokens written(250, base_token);
    {
        server_run server(params);
        const auto generated = generate(server.server, written, 12, 160);
        require(std::all_of(generated.begin(), generated.end(), [](llama_token t) { return t == base_token; }),
                "the fixture generates base tokens after base tokens");
        const auto n_tokens = slot_action(server.server, SERVER_TASK_TYPE_SLOT_SAVE, path);
        require(n_tokens > 32 && n_tokens < written.size(), "expected a shifted slot");
    }
    server_run restored(params);
    slot_action(restored.server, SERVER_TASK_TYPE_SLOT_RESTORE, path);
    // the restored slot lists only base tokens: this prompt shares its first 32
    llama_tokens prompt(32, base_token);
    prompt.push_back(3 + 'y');
    require_completion(first_token(restored.server, prompt, {}, 0, 0, 0), 0, base_token,
                       "a slot restored from a save after a context shift reused its rows");
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

    test_sliding_window(params, work);
    test_long_rope(params, work);
    test_shifted_slot_file(params, work);

    // The rows of a server with other settings are not this server's on any model: not even the exact shared prefix
    // of a writer with another RoPE base is reused, after a restart or as a prefix cache.
    {
        auto writer = params;
        writer.rope_freq_base = 32;
        writer.cache_disk = (work / "rope-cache").string();
        save_completed_prefix(writer);
        require_completion(first_token(writer), prefix_tokens, base_token, "the same settings must reuse the prefix");
        auto reader = params;
        reader.cache_disk.clear();
        reader.cache_ram_mib = 4;
        reader.prefix_cache = writer.cache_disk;
        require_completion(first_token(reader), 0, base_token, "a prefix cache computed with other settings was reused");
        reader.prefix_cache.clear();
        reader.cache_disk = writer.cache_disk;
        require_completion(first_token(reader), 0, base_token, "a cached prompt computed with other settings was reused");
    }

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
