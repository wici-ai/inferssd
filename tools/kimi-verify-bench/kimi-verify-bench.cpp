#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
extern "C" void ggml_cuda_moe_stream_batch_phase_report(const char * label) __attribute__((weak));
#else
extern "C" void ggml_cuda_moe_stream_batch_phase_report(const char * label);
#endif

static constexpr const char * DEFAULT_PROMPT =
    "<|im_user|>user<|im_middle|>Please introduce France in a short paragraph.<|im_end|>"
    "<|im_assistant|>assistant<|im_middle|><think></think>";

static constexpr const char * DEFAULT_CONTINUATION =
    "France is a country in Western Europe known for its rich history, culture, "
    "and influence on art, fashion, and cuisine.";

struct verify_params {
    int32_t block_size = 1;
    std::string prompt = DEFAULT_PROMPT;
    std::string continuation = DEFAULT_CONTINUATION;
    std::string json_out;
    std::string block_device;
};

struct verify_io_snapshot {
    uint64_t process_read_bytes = 0;
    uint64_t block_sectors_read = 0;
    uint64_t block_sector_size = 0;
};

static std::string json_escape(const std::string & input) {
    std::ostringstream out;
    for (const unsigned char c : input) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"':  out << "\\\""; break;
            case '\b': out << "\\b";  break;
            case '\f': out << "\\f";  break;
            case '\n': out << "\\n";  break;
            case '\r': out << "\\r";  break;
            case '\t': out << "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out << buf;
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

static std::string read_text_file(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        return "";
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

static std::string current_cgroup_path() {
    std::ifstream in("/proc/self/cgroup");
    std::string line;
    while (std::getline(in, line)) {
        const std::string marker = "0::";
        if (line.rfind(marker, 0) == 0) {
            return "/sys/fs/cgroup" + line.substr(marker.size());
        }
    }
    return "/sys/fs/cgroup";
}

static uint64_t read_named_u64(const std::string & path, const std::string & name) {
    std::ifstream in(path);
    std::string key;
    uint64_t value = 0;
    while (in >> key >> value) {
        if (!key.empty() && key.back() == ':') {
            key.pop_back();
        }
        if (key == name) {
            return value;
        }
    }
    return 0;
}

static verify_io_snapshot read_verify_io_snapshot(const std::string & block_device) {
    verify_io_snapshot result;
    result.process_read_bytes = read_named_u64("/proc/self/io", "read_bytes");
    if (block_device.empty()) {
        return result;
    }

    const std::string root = "/sys/class/block/" + block_device;
    std::ifstream stat(root + "/stat");
    std::vector<uint64_t> values;
    uint64_t value = 0;
    while (stat >> value) {
        values.push_back(value);
    }
    if (values.size() >= 3) {
        result.block_sectors_read = values[2];
    }
    std::ifstream sector_size(root + "/queue/hw_sector_size");
    sector_size >> result.block_sector_size;
    return result;
}

static uint64_t counter_delta(uint64_t after, uint64_t before) {
    return after >= before ? after - before : 0;
}

static void report_moe_phase(const char * label) {
#if defined(__GNUC__) || defined(__clang__)
    if (ggml_cuda_moe_stream_batch_phase_report) {
        ggml_cuda_moe_stream_batch_phase_report(label);
    }
#else
    ggml_cuda_moe_stream_batch_phase_report(label);
#endif
}

static int32_t parse_i32(const std::string & value, const char * name) {
    char * end = nullptr;
    errno = 0;
    const long v = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || v < 0 || v > INT32_MAX) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<int32_t>(v);
}

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -c 512 -b 512 -ub 512 -ngl 99 --verify-block-size 8\n", argv[0]);
    LOG("\nKimi verify-bench specific params:\n");
    LOG("  --verify-block-size N        number of continuation tokens to verify in one target batch\n");
    LOG("  --verify-prompt TEXT         prompt text; defaults to the Kimi France prompt with special tokens\n");
    LOG("  --verify-continuation TEXT   continuation text used to form candidate verification tokens\n");
    LOG("  --verify-json-out FILE       write clean JSON metrics to FILE in addition to stdout\n");
    LOG("  --verify-block-device DEV    record block-device reads inside the verify window\n");
    LOG("\n");
}

static std::vector<char *> argv_from_storage(std::vector<std::string> & storage) {
    std::vector<char *> result;
    result.reserve(storage.size());
    for (std::string & item : storage) {
        result.push_back(item.data());
    }
    return result;
}

static std::vector<std::string> parse_verify_args(int argc, char ** argv, verify_params & vparams) {
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

        if (arg == "--verify-block-size") {
            vparams.block_size = parse_i32(require_value("--verify-block-size"), "--verify-block-size");
        } else if (arg == "--verify-prompt") {
            vparams.prompt = require_value("--verify-prompt");
        } else if (arg == "--verify-continuation") {
            vparams.continuation = require_value("--verify-continuation");
        } else if (arg == "--verify-json-out") {
            vparams.json_out = require_value("--verify-json-out");
        } else if (arg == "--verify-block-device") {
            vparams.block_device = require_value("--verify-block-device");
        } else {
            filtered.emplace_back(arg);
        }
    }

    if (vparams.block_size <= 0) {
        throw std::runtime_error("--verify-block-size must be > 0");
    }

    return filtered;
}

static std::string tokens_json(
        const llama_context * ctx,
        const std::vector<llama_token> & tokens,
        int32_t limit) {
    std::ostringstream out;
    out << "[";
    const int32_t n = std::min<int32_t>(tokens.size(), limit);
    for (int32_t i = 0; i < n; ++i) {
        if (i > 0) {
            out << ",";
        }
        out << "{\"id\":" << tokens[i]
            << ",\"piece\":\"" << json_escape(common_token_to_piece(ctx, tokens[i], true)) << "\"}";
    }
    out << "]";
    return out.str();
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    verify_params vparams;
    std::vector<std::string> filtered_storage;

    try {
        filtered_storage = parse_verify_args(argc, argv, vparams);
    } catch (const std::exception & e) {
        LOG_ERR("%s\n", e.what());
        print_usage(argc, argv);
        return 1;
    }

    std::vector<char *> filtered_argv = argv_from_storage(filtered_storage);
    int filtered_argc = static_cast<int>(filtered_argv.size());

    common_params params;
    common_init();

    if (!common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMPLETION, print_usage)) {
        return 1;
    }

    if (params.model.path.empty()) {
        LOG_ERR("%s: --model is required\n", __func__);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params model_params = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (model == nullptr) {
        LOG_ERR("%s: unable to load model '%s'\n", __func__, params.model.path.c_str());
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> continuation_tokens;

    try {
        prompt_tokens = common_tokenize(vocab, vparams.prompt, true, true);
        continuation_tokens = common_tokenize(vocab, vparams.continuation, false, true);
    } catch (const std::exception & e) {
        LOG_ERR("%s: tokenization failed: %s\n", __func__, e.what());
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    if (prompt_tokens.empty()) {
        LOG_ERR("%s: prompt tokenization produced no tokens\n", __func__);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    if (static_cast<int32_t>(continuation_tokens.size()) < vparams.block_size) {
        LOG_ERR("%s: continuation has only %zu tokens, need %d\n",
                __func__, continuation_tokens.size(), vparams.block_size);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_context_params ctx_params = common_context_params_to_llama(params);
    const uint32_t min_batch = std::max<uint32_t>(
            static_cast<uint32_t>(prompt_tokens.size()),
            static_cast<uint32_t>(vparams.block_size));
    ctx_params.n_batch = std::max<uint32_t>(ctx_params.n_batch, min_batch);
    ctx_params.n_ubatch = std::max<uint32_t>(ctx_params.n_ubatch, std::min<uint32_t>(ctx_params.n_batch, min_batch));

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        LOG_ERR("%s: failed to create llama_context\n", __func__);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const int32_t n_ctx = llama_n_ctx(ctx);
    if (static_cast<int32_t>(prompt_tokens.size()) + vparams.block_size > n_ctx) {
        LOG_ERR("%s: required context %zu + %d exceeds n_ctx %d\n",
                __func__, prompt_tokens.size(), vparams.block_size, n_ctx);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_batch prompt_batch = llama_batch_init(prompt_tokens.size(), 0, 1);
    llama_batch verify_batch = llama_batch_init(vparams.block_size, 0, 1);

    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        common_batch_add(prompt_batch, prompt_tokens[i], static_cast<llama_pos>(i), { 0 }, i + 1 == prompt_tokens.size());
    }

    const int64_t prefill_start_us = ggml_time_us();
    int ret = llama_decode(ctx, prompt_batch);
    llama_synchronize(ctx);
    const int64_t prefill_end_us = ggml_time_us();
    if (ret != 0) {
        LOG_ERR("%s: prompt llama_decode returned %d\n", __func__, ret);
        llama_batch_free(prompt_batch);
        llama_batch_free(verify_batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const llama_pos verify_pos0 = static_cast<llama_pos>(prompt_tokens.size());
    for (int32_t i = 0; i < vparams.block_size; ++i) {
        common_batch_add(verify_batch, continuation_tokens[i], verify_pos0 + i, { 0 }, true);
    }

    report_moe_phase("task82_verify_begin");
    const verify_io_snapshot verify_io_before = read_verify_io_snapshot(vparams.block_device);
    const int64_t verify_start_us = ggml_time_us();
    ret = llama_decode(ctx, verify_batch);
    llama_synchronize(ctx);
    const int64_t verify_end_us = ggml_time_us();
    const verify_io_snapshot verify_io_after = read_verify_io_snapshot(vparams.block_device);
    report_moe_phase("task82_verify_end");
    if (ret != 0) {
        LOG_ERR("%s: verify llama_decode returned %d\n", __func__, ret);
        llama_batch_free(prompt_batch);
        llama_batch_free(verify_batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const double prefill_ms = (prefill_end_us - prefill_start_us) / 1000.0;
    const double verify_ms = (verify_end_us - verify_start_us) / 1000.0;
    const double verify_tps = verify_ms > 0.0 ? (1000.0 * vparams.block_size / verify_ms) : 0.0;
    const uint64_t process_read_bytes = counter_delta(
        verify_io_after.process_read_bytes, verify_io_before.process_read_bytes);
    const uint64_t block_read_sectors = counter_delta(
        verify_io_after.block_sectors_read, verify_io_before.block_sectors_read);
    const uint64_t block_read_bytes = block_read_sectors * verify_io_after.block_sector_size;

    const std::string cg = current_cgroup_path();
    const std::string memory_current = read_text_file(cg + "/memory.current");
    const std::string memory_peak = read_text_file(cg + "/memory.peak");
    const std::string memory_max = read_text_file(cg + "/memory.max");
    const std::string memory_swap_max = read_text_file(cg + "/memory.swap.max");

    std::ostringstream json;
    json << "{";
    json << "\"tool\":\"llama-kimi-verify-bench\"";
    json << ",\"block_size\":" << vparams.block_size;
    json << ",\"prompt_tokens\":" << prompt_tokens.size();
    json << ",\"continuation_tokens_total\":" << continuation_tokens.size();
    json << ",\"n_ctx\":" << n_ctx;
    json << ",\"n_batch\":" << ctx_params.n_batch;
    json << ",\"n_ubatch_requested\":" << ctx_params.n_ubatch;
    json << ",\"n_ubatch_actual\":" << llama_n_ubatch(ctx);
    json << ",\"single_ubatch_capacity\":"
         << (llama_n_ubatch(ctx) >= static_cast<uint32_t>(vparams.block_size) ? "true" : "false");
    json << ",\"prefill_wall_ms\":" << prefill_ms;
    json << ",\"verify_wall_ms\":" << verify_ms;
    json << ",\"verify_tokens_per_s\":" << verify_tps;
    json << ",\"verify_process_read_bytes\":" << process_read_bytes;
    json << ",\"verify_block_device\":\"" << json_escape(vparams.block_device) << "\"";
    json << ",\"verify_block_sector_size\":" << verify_io_after.block_sector_size;
    json << ",\"verify_block_read_sectors\":" << block_read_sectors;
    json << ",\"verify_block_read_bytes\":" << block_read_bytes;
    json << ",\"verify_block_read_GB_s\":" << (verify_ms > 0.0 ? block_read_bytes / verify_ms / 1.0e6 : 0.0);
    json << ",\"verify_pos0\":" << verify_pos0;
    json << ",\"model\":\"" << json_escape(params.model.path) << "\"";
    json << ",\"prompt\":\"" << json_escape(vparams.prompt) << "\"";
    json << ",\"continuation\":\"" << json_escape(vparams.continuation) << "\"";
    json << ",\"verify_tokens\":" << tokens_json(ctx, continuation_tokens, vparams.block_size);
    json << ",\"cgroup\":\"" << json_escape(cg) << "\"";
    json << ",\"memory_current\":\"" << json_escape(memory_current) << "\"";
    json << ",\"memory_peak\":\"" << json_escape(memory_peak) << "\"";
    json << ",\"memory_max\":\"" << json_escape(memory_max) << "\"";
    json << ",\"memory_swap_max\":\"" << json_escape(memory_swap_max) << "\"";
    json << "}";

    const std::string json_text = json.str();
    fprintf(stdout, "%s\n", json_text.c_str());
    fflush(stdout);

    if (!vparams.json_out.empty()) {
        std::ofstream out(vparams.json_out);
        if (!out) {
            LOG_ERR("%s: failed to open --verify-json-out '%s'\n", __func__, vparams.json_out.c_str());
            llama_batch_free(prompt_batch);
            llama_batch_free(verify_batch);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        out << json_text << "\n";
    }

    llama_perf_context_print(ctx);

    llama_batch_free(prompt_batch);
    llama_batch_free(verify_batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
