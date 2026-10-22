#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr const char * DEFAULT_PROMPT =
    "The key invariant for reliable speculative decoding is";

static constexpr const char * DEFAULT_CONTINUATION =
    " that rejected recurrent states must be discarded and the accepted prefix replayed exactly.";

struct rollback_params {
    int32_t block_size = 8;
    std::string prompt = DEFAULT_PROMPT;
    std::string continuation = DEFAULT_CONTINUATION;
    std::string json_out;
};

struct rollback_result {
    int32_t rejection_index = 0;
    int32_t replay_rows = 0;
    bool equal = false;
    size_t first_diff = 0;
    uint64_t rollback_hash = 0;
    uint64_t clean_hash = 0;
};

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -c 128 -b 8 -ub 8 -ngl 99 --rollback-json-out result.json\n", argv[0]);
    LOG("\nKimi rollback-check specific params:\n");
    LOG("  --rollback-block-size N       verifier rows including the committed anchor (default: 8)\n");
    LOG("  --rollback-prompt TEXT        prompt used to establish the single checkpoint\n");
    LOG("  --rollback-continuation TEXT  tokens used for dirty verification and replay\n");
    LOG("  --rollback-json-out FILE      write machine-readable gate evidence\n\n");
}

static int32_t parse_i32(const std::string & value, const char * name) {
    size_t used = 0;
    const long v = std::stol(value, &used);
    if (used != value.size() || v < 2 || v > INT32_MAX) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<int32_t>(v);
}

static std::vector<std::string> parse_rollback_args(
        int argc, char ** argv, rollback_params & rparams) {
    std::vector<std::string> filtered;
    filtered.reserve(argc);
    filtered.emplace_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--rollback-block-size") {
            rparams.block_size = parse_i32(
                require_value("--rollback-block-size"), "--rollback-block-size");
        } else if (arg == "--rollback-prompt") {
            rparams.prompt = require_value("--rollback-prompt");
        } else if (arg == "--rollback-continuation") {
            rparams.continuation = require_value("--rollback-continuation");
        } else if (arg == "--rollback-json-out") {
            rparams.json_out = require_value("--rollback-json-out");
        } else {
            filtered.emplace_back(arg);
        }
    }
    return filtered;
}

static std::vector<char *> argv_from_storage(std::vector<std::string> & storage) {
    std::vector<char *> result;
    result.reserve(storage.size());
    for (std::string & item : storage) {
        result.push_back(item.data());
    }
    return result;
}

static uint64_t fnv1a64(const std::vector<uint8_t> & data) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (uint8_t byte : data) {
        hash ^= byte;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static std::string hex_u64(uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

static bool decode_tokens(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        llama_pos pos0,
        bool sequential) {
    const int32_t capacity = sequential ? 1 : static_cast<int32_t>(tokens.size());
    llama_batch batch = llama_batch_init(capacity, 0, 1);
    bool ok = true;

    if (sequential) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            common_batch_clear(batch);
            common_batch_add(batch, tokens[i], pos0 + static_cast<llama_pos>(i), { 0 }, true);
            if (llama_decode(ctx, batch) != 0) {
                ok = false;
                break;
            }
        }
    } else {
        for (size_t i = 0; i < tokens.size(); ++i) {
            common_batch_add(batch, tokens[i], pos0 + static_cast<llama_pos>(i), { 0 }, true);
        }
        ok = llama_decode(ctx, batch) == 0;
    }

    llama_synchronize(ctx);
    llama_batch_free(batch);
    return ok;
}

static std::vector<uint8_t> get_partial_state(llama_context * ctx, size_t state_size) {
    std::vector<uint8_t> state(state_size);
    const size_t copied = llama_state_seq_get_data_ext(
        ctx, state.data(), state.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (copied != state.size()) {
        throw std::runtime_error("partial state serialization size mismatch");
    }
    return state;
}

static void restore_checkpoint(
        llama_context * ctx,
        const std::vector<uint8_t> & checkpoint,
        llama_pos replay_pos0) {
    const size_t restored = llama_state_seq_set_data_ext(
        ctx, checkpoint.data(), checkpoint.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (restored != checkpoint.size()) {
        throw std::runtime_error("partial checkpoint restore size mismatch");
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, replay_pos0, -1)) {
        throw std::runtime_error("failed to truncate target memory after checkpoint restore");
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    rollback_params rparams;
    std::vector<std::string> storage;
    try {
        storage = parse_rollback_args(argc, argv, rparams);
    } catch (const std::exception & e) {
        LOG_ERR("%s\n", e.what());
        print_usage(argc, argv);
        return 1;
    }

    std::vector<char *> filtered_argv = argv_from_storage(storage);
    common_params params;
    common_init();
    if (!common_params_parse(
            static_cast<int>(filtered_argv.size()), filtered_argv.data(), params,
            LLAMA_EXAMPLE_COMPLETION, print_usage)) {
        return 1;
    }
    if (params.model.path.empty()) {
        LOG_ERR("%s: --model is required\n", __func__);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model = llama_model_load_from_file(
        params.model.path.c_str(), common_model_params_to_llama(params));
    if (model == nullptr) {
        LOG_ERR("%s: unable to load model '%s'\n", __func__, params.model.path.c_str());
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> prompt = common_tokenize(vocab, rparams.prompt, true, true);
    const std::vector<llama_token> continuation =
        common_tokenize(vocab, rparams.continuation, false, true);
    if (prompt.empty() || continuation.size() < static_cast<size_t>(rparams.block_size)) {
        LOG_ERR("%s: insufficient prompt or continuation tokens\n", __func__);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_context_params cparams = common_context_params_to_llama(params);
    cparams.n_batch = std::max<uint32_t>(cparams.n_batch,
        std::max<uint32_t>(prompt.size(), rparams.block_size));
    cparams.n_ubatch = std::max<uint32_t>(cparams.n_ubatch, rparams.block_size);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        LOG_ERR("%s: failed to create context\n", __func__);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    int result_code = 1;
    try {
        if (!decode_tokens(ctx, prompt, 0, false)) {
            throw std::runtime_error("prompt decode failed");
        }

        const llama_pos replay_pos0 = static_cast<llama_pos>(prompt.size());
        const size_t state_size = llama_state_seq_get_size_ext(
            ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (state_size == 0) {
            throw std::runtime_error("target exposes no partial recurrent checkpoint state");
        }
        const std::vector<uint8_t> checkpoint = get_partial_state(ctx, state_size);
        const std::vector<llama_token> verify_tokens(
            continuation.begin(), continuation.begin() + rparams.block_size);

        std::vector<rollback_result> results;
        bool all_equal = true;
        for (int32_t rejection = 0; rejection < rparams.block_size - 1; ++rejection) {
            restore_checkpoint(ctx, checkpoint, replay_pos0);
            if (!decode_tokens(ctx, verify_tokens, replay_pos0, false)) {
                throw std::runtime_error("dirty verifier block decode failed");
            }

            // A rejection at proposal r commits the verifier anchor plus the
            // r proposals before it. The correction token sampled at the
            // rejected row remains pending and is not part of recurrent state.
            const int32_t replay_rows = rejection + 1;
            const std::vector<llama_token> replay_tokens(
                verify_tokens.begin(), verify_tokens.begin() + replay_rows);

            restore_checkpoint(ctx, checkpoint, replay_pos0);
            if (!decode_tokens(ctx, replay_tokens, replay_pos0, true)) {
                throw std::runtime_error("rollback accepted-prefix replay failed");
            }
            const std::vector<uint8_t> rollback_state = get_partial_state(ctx, state_size);

            restore_checkpoint(ctx, checkpoint, replay_pos0);
            if (!decode_tokens(ctx, replay_tokens, replay_pos0, true)) {
                throw std::runtime_error("token-by-token accepted-prefix replay failed");
            }
            const std::vector<uint8_t> clean_state = get_partial_state(ctx, state_size);

            const auto mismatch = std::mismatch(
                rollback_state.begin(), rollback_state.end(), clean_state.begin());
            rollback_result result;
            result.rejection_index = rejection;
            result.replay_rows = replay_rows;
            result.equal = mismatch.first == rollback_state.end();
            result.first_diff = result.equal ? state_size :
                static_cast<size_t>(mismatch.first - rollback_state.begin());
            result.rollback_hash = fnv1a64(rollback_state);
            result.clean_hash = fnv1a64(clean_state);
            all_equal = all_equal && result.equal;
            results.push_back(result);
        }

        std::ostringstream json;
        json << "{\"tool\":\"llama-kimi-rollback-check\"";
        json << ",\"status\":\"" << (all_equal ? "PASS" : "FAIL") << "\"";
        json << ",\"block_size\":" << rparams.block_size;
        json << ",\"proposal_count\":" << rparams.block_size - 1;
        json << ",\"checkpoint_count\":1";
        json << ",\"checkpoint_bytes\":" << state_size;
        json << ",\"rejection_cases\":[";
        for (size_t i = 0; i < results.size(); ++i) {
            if (i > 0) {
                json << ",";
            }
            const auto & result = results[i];
            json << "{\"rejection_index\":" << result.rejection_index;
            json << ",\"replay_rows\":" << result.replay_rows;
            json << ",\"bitwise_equal\":" << (result.equal ? "true" : "false");
            json << ",\"first_diff\":" << result.first_diff;
            json << ",\"rollback_state_fnv1a64\":\"" << hex_u64(result.rollback_hash) << "\"";
            json << ",\"clean_state_fnv1a64\":\"" << hex_u64(result.clean_hash) << "\"}";
        }
        json << "]}";

        const std::string text = json.str();
        fprintf(stdout, "%s\n", text.c_str());
        if (!rparams.json_out.empty()) {
            std::ofstream out(rparams.json_out);
            if (!out) {
                throw std::runtime_error("failed to open rollback JSON output");
            }
            out << text << "\n";
        }
        result_code = all_equal ? 0 : 2;
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
    }

    llama_perf_context_print(ctx);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return result_code;
}
