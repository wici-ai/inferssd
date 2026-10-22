#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include <nlohmann/json.hpp>
extern "C" {
#include "sha256.h"
}

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
extern "C" void ggml_cuda_moe_stream_batch_phase_report(const char * label) __attribute__((weak));
extern "C" void ggml_cuda_moe_ram_tier_request_begin(uint64_t request_id) __attribute__((weak));
#else
extern "C" void ggml_cuda_moe_stream_batch_phase_report(const char * label);
extern "C" void ggml_cuda_moe_ram_tier_request_begin(uint64_t request_id);
#endif

namespace {

constexpr int k_n_predict = 8;
constexpr int k_exam_n_predict = 64;
constexpr int k_n_ctx = 512;
constexpr int k_n_batch = 64;
constexpr int k_n_threads = 3;
constexpr int k_n_gpu_layers = 91;
constexpr int k_m3_n_predict = 64;

using json = nlohmann::ordered_json;

struct exam_question {
    int ordinal = 0;
    int expected_tokens = 0;
    std::string prompt;
};

struct m3_prompt {
    std::string id;
    std::string category;
    std::string text;
    std::string preregister_sha256;
    std::string prompt_sha256;
};

struct m3_io_snapshot {
    uint64_t process_read_bytes = 0;
    uint64_t block_reads = 0;
    uint64_t block_sectors_read = 0;
    uint64_t block_read_ms = 0;
    uint64_t block_sector_size = 0;
};

struct m3_token_timing {
    int output_index = 0;
    llama_token input_token = LLAMA_TOKEN_NULL;
    int64_t decode_start_ns = 0;
    int64_t decode_end_ns = 0;
    int64_t decode_ns = 0;
    int64_t sample_ns = 0;
    int64_t interval_ns = 0;
};

std::string sha256_hex(const void * data, size_t size);

void print_usage(int, char ** argv) {
    std::fprintf(stderr,
            "usage: %s -m MODEL -p hello -n 8 -c 512 -b 64 -ub 64 -t 3 -tb 3 "
            "-ngl 91 --fit off --defer-experts -ot ...\n"
            "M3: %s --engine legacy|pipeline -m MODEL -p PREREGISTERED_PROMPT -n 64 ...\n"
            "required env: GGML_KIMI_PIPELINE_V2_MODE=depth1|pipeline "
            "TASK132_EVIDENCE_DIR=DIR [TASK132_REPEATS=1|2]\n",
            argv[0], argv[0]);
}

bool env_flag(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] && value[0] != '0';
}

bool set_env(const char * name, const std::string & value) {
#if defined(_WIN32)
    return _putenv_s(name, value.c_str()) == 0;
#else
    return setenv(name, value.c_str(), 1) == 0;
#endif
}

bool unset_env(const char * name) {
#if defined(_WIN32)
    return _putenv_s(name, "") == 0;
#else
    return unsetenv(name) == 0;
#endif
}

void report_moe_phase(const char * label) {
#if defined(__GNUC__) || defined(__clang__)
    if (ggml_cuda_moe_stream_batch_phase_report) {
        ggml_cuda_moe_stream_batch_phase_report(label);
    }
#else
    ggml_cuda_moe_stream_batch_phase_report(label);
#endif
}

void begin_moe_request(uint64_t request_id) {
#if defined(__GNUC__) || defined(__clang__)
    if (ggml_cuda_moe_ram_tier_request_begin) {
        ggml_cuda_moe_ram_tier_request_begin(request_id);
    }
#else
    ggml_cuda_moe_ram_tier_request_begin(request_id);
#endif
}

uint64_t read_named_u64(const std::filesystem::path & path, const std::string & name) {
    std::ifstream input(path);
    std::string key;
    uint64_t value = 0;
    while (input >> key >> value) {
        if (!key.empty() && key.back() == ':') {
            key.pop_back();
        }
        if (key == name) {
            return value;
        }
    }
    return 0;
}

m3_io_snapshot read_m3_io_snapshot(const std::string & block_device) {
    m3_io_snapshot result;
    result.process_read_bytes = read_named_u64("/proc/self/io", "read_bytes");
    if (block_device.empty()) {
        return result;
    }
    const std::filesystem::path root = std::filesystem::path("/sys/class/block") / block_device;
    std::ifstream stat(root / "stat");
    std::vector<uint64_t> fields;
    uint64_t value = 0;
    while (stat >> value) {
        fields.push_back(value);
    }
    if (fields.size() >= 4) {
        result.block_reads = fields[0];
        result.block_sectors_read = fields[2];
        result.block_read_ms = fields[3];
    }
    std::ifstream sector_size(root / "queue/hw_sector_size");
    sector_size >> result.block_sector_size;
    return result;
}

uint64_t counter_delta(uint64_t after, uint64_t before) {
    return after >= before ? after - before : 0;
}

int64_t monotonic_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

double percentile_ms(std::vector<int64_t> values, double quantile) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position = quantile * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, values.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return (static_cast<double>(values[lower]) * (1.0 - fraction) +
            static_cast<double>(values[upper]) * fraction) / 1e6;
}

m3_prompt load_m3_prompt(const std::filesystem::path & path, const std::string & prompt_id) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open M3 preregistration");
    }
    std::ostringstream raw;
    raw << input.rdbuf();
    const std::string bytes = raw.str();
    const json preregister = json::parse(bytes);
    const auto & prompts = preregister.at("prompts");
    if (!prompts.is_array() || prompts.size() < 3) {
        throw std::runtime_error("M3 preregistration must contain P0-P2");
    }
    for (size_t index = 0; index < 3; ++index) {
        const std::string expected = "p" + std::to_string(index);
        if (prompts.at(index).at("id").get<std::string>() != expected) {
            throw std::runtime_error("M3 preregistration P0-P2 order drift");
        }
    }
    if (prompt_id != "p0" && prompt_id != "p1" && prompt_id != "p2") {
        throw std::runtime_error("M3 formal prompt must be p0, p1, or p2");
    }
    const size_t index = static_cast<size_t>(prompt_id[1] - '0');
    m3_prompt result;
    result.id = prompt_id;
    result.category = prompts.at(index).at("category").get<std::string>();
    result.text = prompts.at(index).at("text").get<std::string>();
    result.preregister_sha256 = sha256_hex(bytes.data(), bytes.size());
    result.prompt_sha256 = sha256_hex(result.text.data(), result.text.size());
    return result;
}

std::string sha256_hex(const void * data, size_t size) {
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_hash(digest, static_cast<const unsigned char *>(data), size);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned char byte : digest) {
        out << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return out.str();
}

llama_token greedy_token(const llama_context * ctx, const llama_vocab * vocab) {
    const float * logits = llama_get_logits_ith(const_cast<llama_context *>(ctx), -1);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (!logits || n_vocab <= 0) {
        return LLAMA_TOKEN_NULL;
    }
    llama_token best_id = 0;
    float best = -std::numeric_limits<float>::infinity();
    for (llama_token id = 0; id < n_vocab; ++id) {
        const float value = logits[id];
        if (!std::isnan(value) && (value > best || (value == best && id < best_id))) {
            best = value;
            best_id = id;
        }
    }
    return best_id;
}

bool logits_finite(const llama_context * ctx, const llama_vocab * vocab) {
    const float * logits = llama_get_logits_ith(const_cast<llama_context *>(ctx), -1);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (!logits || n_vocab <= 0) {
        return false;
    }
    for (int32_t index = 0; index < n_vocab; ++index) {
        if (!std::isfinite(logits[index])) {
            return false;
        }
    }
    return true;
}

bool decode_prompt(llama_context * ctx, const std::vector<llama_token> & tokens) {
    size_t offset = 0;
    while (offset < tokens.size()) {
        const size_t count = std::min<size_t>(k_n_batch, tokens.size() - offset);
        llama_batch batch = llama_batch_get_one(
                const_cast<llama_token *>(tokens.data() + offset), static_cast<int32_t>(count));
        if (llama_decode(ctx, batch) != 0) {
            return false;
        }
        offset += count;
    }
    llama_synchronize(ctx);
    return true;
}

bool run_top_k_self_test(const std::filesystem::path & evidence_dir) {
    ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu) {
        std::fprintf(stderr, "[kimi_pipeline_m1] no GPU backend for top-k self-test\n");
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(gpu);
    using self_test_fn = bool (*)(char *, size_t);
    auto fn = reinterpret_cast<self_test_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_cuda_kimi_pipeline_top_k_self_test"));
    if (!fn) {
        std::fprintf(stderr, "[kimi_pipeline_m1] CUDA top-k self-test entry is missing\n");
        return false;
    }
    char digest[32] = {};
    const bool ok = fn(digest, sizeof(digest));
    std::ofstream out(evidence_dir / "topk-selftest.txt", std::ios::trunc);
    out << "shape=896 k=16 rows=4 repeats=32 tie=expert_id_ascending ok="
        << (ok ? 1 : 0) << " fnv=" << digest << '\n';
    std::fprintf(stderr,
            "[kimi_pipeline_m1] top-k self-test shape=896 k=16 repeats=32 ok=%d digest=%s\n",
            ok ? 1 : 0, digest);
    return ok && out.good();
}

bool validate_params(const common_params & params, const char * mode, int repeats) {
    bool ok = true;
    auto require = [&](bool condition, const char * message) {
        if (!condition) {
            std::fprintf(stderr, "[kimi_pipeline_m1] guard: %s\n", message);
            ok = false;
        }
    };
    require(mode && (std::strcmp(mode, "depth1") == 0 || std::strcmp(mode, "pipeline") == 0),
            "GGML_KIMI_PIPELINE_V2_MODE must be depth1 or pipeline");
    require(repeats == 1 || repeats == 2, "TASK132_REPEATS must be 1 or 2");
    require(params.model.path.size() > 0, "--model is required");
    require(params.prompt == "hello", "M1 prompt is frozen to literal 'hello'");
    require(params.n_predict == k_n_predict, "M1 requires -n 8");
    require(params.n_ctx == k_n_ctx, "M1 requires -c 512");
    require(params.n_batch == k_n_batch && params.n_ubatch == k_n_batch,
            "M1 requires -b 64 -ub 64");
    require(params.n_parallel == 1, "M1 requires one sequence");
    require(params.cpuparams.n_threads == k_n_threads &&
            params.cpuparams_batch.n_threads == k_n_threads,
            "M1 requires -t 3 -tb 3");
    require(params.n_gpu_layers == k_n_gpu_layers, "M1 requires -ngl 91");
    require(params.defer_experts, "M1 requires --defer-experts");
    require(!params.fit_params, "M1 requires --fit off");
    require(params.sampling.seed == 1, "M1 requires --seed 1");
    require(params.sampling.top_k == 1, "M1 requires --top-k 1");
    require(params.sampling.temp == 0.0f, "M1 requires --temp 0");
    require(env_flag("GGML_CUDA_DISABLE_GRAPHS"), "CUDA graphs must be disabled");
    require(std::getenv("GGML_MOE_EXPERT_PACK") != nullptr, "GGML_MOE_EXPERT_PACK is required");
    return ok;
}

bool validate_exam_params(const common_params & params, const char * mode, const char * arm) {
    bool ok = true;
    auto require = [&](bool condition, const char * message) {
        if (!condition) {
            std::fprintf(stderr, "[kimi_pipeline_m2] guard: %s\n", message);
            ok = false;
        }
    };
    require(mode && std::strcmp(mode, "pipeline") == 0,
            "M2 exam requires GGML_KIMI_PIPELINE_V2_MODE=pipeline");
    require(arm && (std::strcmp(arm, "legacy") == 0 || std::strcmp(arm, "v2") == 0),
            "TASK132_M2_ARM must be legacy or v2");
    require(!params.model.path.empty(), "--model is required");
    require(params.n_predict == k_exam_n_predict, "M2 exam requires -n 64");
    require(params.n_ctx == k_n_ctx, "M2 exam requires -c 512");
    require(params.n_batch == k_n_batch && params.n_ubatch == k_n_batch,
            "M2 exam requires -b 64 -ub 64");
    require(params.n_parallel == 1, "M2 exam requires one sequence");
    require(params.cpuparams.n_threads == k_n_threads &&
            params.cpuparams_batch.n_threads == k_n_threads,
            "M2 exam requires -t 3 -tb 3");
    require(params.n_gpu_layers == k_n_gpu_layers, "M2 exam requires -ngl 91");
    require(params.defer_experts, "M2 exam requires --defer-experts");
    require(!params.fit_params, "M2 exam requires --fit off");
    require(params.sampling.seed == 1, "M2 exam requires --seed 1");
    require(params.sampling.top_k == 1, "M2 exam requires --top-k 1");
    require(params.sampling.temp == 0.0f, "M2 exam requires --temp 0");
    require(env_flag("GGML_CUDA_DISABLE_GRAPHS"), "CUDA graphs must be disabled");
    require(std::getenv("GGML_MOE_EXPERT_PACK") != nullptr, "GGML_MOE_EXPERT_PACK is required");
    return ok;
}

bool validate_m3_params(
        const common_params & params,
        const char * mode,
        const std::string & engine,
        const m3_prompt & prompt,
        bool trace_mode) {
    bool ok = true;
    auto require = [&](bool condition, const char * message) {
        if (!condition) {
            std::fprintf(stderr, "[kimi_pipeline_m3] guard: %s\n", message);
            ok = false;
        }
    };
    require(engine == "legacy" || engine == "pipeline" || (trace_mode && engine == "depth1"),
            "--engine must be legacy|pipeline (depth1 is trace-only)");
    if (engine == "depth1") {
        require(mode && std::strcmp(mode, "depth1") == 0,
                "depth1 engine requires GGML_KIMI_PIPELINE_V2_MODE=depth1");
    } else {
        require(mode && std::strcmp(mode, "pipeline") == 0,
                "legacy/pipeline M3 requires GGML_KIMI_PIPELINE_V2_MODE=pipeline");
    }
    require(!params.model.path.empty(), "--model is required");
    require(params.prompt == prompt.text, "CLI prompt differs from preregistered prompt text");
    if (trace_mode) {
        require(params.n_predict == 16 || params.n_predict == 32,
                "M3 trace requires -n 16 or -n 32");
    } else {
        require(params.n_predict == k_m3_n_predict,
                "M3 formal battery requires -n 64");
    }
    require(params.n_ctx == k_n_ctx, "M3 requires -c 512");
    require(params.n_batch == k_n_batch && params.n_ubatch == k_n_batch,
            "M3 requires -b 64 -ub 64");
    require(params.n_parallel == 1, "M3 requires one sequence");
    require(params.cpuparams.n_threads == k_n_threads &&
            params.cpuparams_batch.n_threads == k_n_threads,
            "M3 requires -t 3 -tb 3");
    require(params.n_gpu_layers == k_n_gpu_layers, "M3 requires -ngl 91");
    require(params.defer_experts, "M3 requires --defer-experts");
    require(!params.fit_params, "M3 requires --fit off");
    require(params.sampling.seed == 1, "M3 requires --seed 1");
    require(params.sampling.top_k == 1, "M3 requires --top-k 1");
    require(params.sampling.temp == 0.0f, "M3 requires --temp 0");
    require(env_flag("GGML_CUDA_DISABLE_GRAPHS"), "CUDA graphs must be disabled");
    require(std::getenv("GGML_MOE_EXPERT_PACK") != nullptr,
            "GGML_MOE_EXPERT_PACK is required");
    const char * parity = std::getenv("GGML_KIMI_PIPELINE_V2_PARITY");
    require(parity && parity[0] == '0', "M3 requires projection parity instrumentation off");
    return ok;
}

std::vector<exam_question> load_exam(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open fixed exam");
    }
    json fixed;
    input >> fixed;
    const auto & rows = fixed.at("questions");
    if (!rows.is_array() || rows.size() != 5) {
        throw std::runtime_error("fixed exam must contain exactly five questions");
    }
    std::vector<exam_question> result;
    result.reserve(5);
    for (size_t index = 0; index < rows.size(); ++index) {
        exam_question question;
        question.ordinal = rows.at(index).at("ordinal").get<int>();
        question.expected_tokens = rows.at(index).at("token_count").get<int>();
        question.prompt = rows.at(index).at("prompt").get<std::string>();
        const std::string expected_sha = rows.at(index).at("prompt_sha256").get<std::string>();
        if (question.ordinal != static_cast<int>(index + 1) ||
                sha256_hex(question.prompt.data(), question.prompt.size()) != expected_sha) {
            throw std::runtime_error("fixed exam order or prompt digest drift");
        }
        result.push_back(std::move(question));
    }
    return result;
}

bool run_exam_request(
        llama_model * model,
        const llama_vocab * vocab,
        const common_params & params,
        const exam_question & question,
        const std::filesystem::path & evidence_dir,
        bool v2,
        int round) {
    const std::string thermal = round == 1 ? "cold" : "hot";
    const std::string label = thermal + "-q" + std::to_string(question.ordinal);
    const std::filesystem::path request_dir = evidence_dir / label;
    std::filesystem::create_directories(request_dir);
    if (!set_env("GGML_KIMI_PIPELINE_M1_RUN", label) ||
            !unset_env("GGML_KIMI_PIPELINE_V2_ACTIVE")) {
        return false;
    }

    const std::vector<llama_token> prompt_tokens = common_tokenize(vocab, question.prompt, true, true);
    if (prompt_tokens.size() != static_cast<size_t>(question.expected_tokens) ||
            prompt_tokens.size() + k_exam_n_predict > k_n_ctx) {
        std::fprintf(stderr,
                "[kimi_pipeline_m2] prompt guard failed round=%d q=%d tokens=%zu expected=%d\n",
                round, question.ordinal, prompt_tokens.size(), question.expected_tokens);
        return false;
    }

    llama_context_params cparams = common_context_params_to_llama(params);
    cparams.n_ctx = k_n_ctx;
    cparams.n_batch = k_n_batch;
    cparams.n_ubatch = k_n_batch;
    cparams.n_seq_max = 1;
    cparams.n_threads = k_n_threads;
    cparams.n_threads_batch = k_n_threads;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "[kimi_pipeline_m2] context create failed round=%d q=%d\n",
                round, question.ordinal);
        return false;
    }

    bool ok = decode_prompt(ctx, prompt_tokens);
    if (ok && v2) {
        ok = set_env("GGML_KIMI_PIPELINE_V2_ACTIVE", "1");
    }

    std::ofstream checkpoints(request_dir / "logits-checkpoints.csv", std::ios::trunc);
    std::ofstream tokens_file(request_dir / "tokens.txt", std::ios::trunc);
    std::ofstream content_file(request_dir / "content.bin", std::ios::binary | std::ios::trunc);
    checkpoints << "step,token_id,logits_sha256\n";
    std::vector<llama_token> generated;
    generated.reserve(k_exam_n_predict);
    bool stopped_eog = false;
    for (int step = 0; ok && step < k_exam_n_predict; ++step) {
        if (!logits_finite(ctx, vocab)) {
            std::fprintf(stderr, "[kimi_pipeline_m2] non-finite logits round=%d q=%d step=%d\n",
                    round, question.ordinal, step);
            ok = false;
            break;
        }
        const llama_token token = greedy_token(ctx, vocab);
        if (token == LLAMA_TOKEN_NULL) {
            ok = false;
            break;
        }
        generated.push_back(token);
        tokens_file << token << '\n';
        const std::string piece = common_token_to_piece(ctx, token, true);
        content_file.write(piece.data(), static_cast<std::streamsize>(piece.size()));

        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(&generated.back()), 1);
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "[kimi_pipeline_m2] decode failed round=%d q=%d step=%d\n",
                    round, question.ordinal, step);
            ok = false;
            break;
        }
        llama_synchronize(ctx);
        const float * logits = llama_get_logits_ith(ctx, -1);
        const size_t logits_bytes = static_cast<size_t>(llama_vocab_n_tokens(vocab)) * sizeof(float);
        if (!logits || !logits_finite(ctx, vocab)) {
            ok = false;
            break;
        }
        checkpoints << step << ',' << token << ',' << sha256_hex(logits, logits_bytes) << '\n';
        if (llama_vocab_is_eog(vocab, token)) {
            stopped_eog = true;
            break;
        }
    }

    checkpoints.flush();
    tokens_file.flush();
    content_file.flush();
    ok = ok && generated.size() == k_exam_n_predict && !stopped_eog &&
        checkpoints.good() && tokens_file.good() && content_file.good();
    std::ofstream manifest(request_dir / "manifest.txt", std::ios::trunc);
    manifest << "arm=" << (v2 ? "v2" : "legacy") << '\n'
             << "round=" << round << '\n'
             << "thermal=" << thermal << '\n'
             << "question=" << question.ordinal << '\n'
             << "prompt_tokens=" << prompt_tokens.size() << '\n'
             << "decode_steps=" << generated.size() << '\n'
             << "stopped_eog=" << (stopped_eog ? 1 : 0) << '\n'
             << "cache_prompt=0\nseed=1\ntemperature=0\nsampler_top_k=1\n"
             << "status=" << (ok ? "ok" : "failed") << '\n';
    ok = ok && manifest.good();

    unset_env("GGML_KIMI_PIPELINE_V2_ACTIVE");
    llama_free(ctx);
    std::fprintf(stderr,
            "[kimi_pipeline_m2] request arm=%s round=%d q=%d prompt=%zu decode=%zu status=%s\n",
            v2 ? "v2" : "legacy", round, question.ordinal, prompt_tokens.size(),
            generated.size(), ok ? "ok" : "failed");
    return ok;
}

bool run_exam(
        llama_model * model,
        const llama_vocab * vocab,
        const common_params & params,
        const std::vector<exam_question> & questions,
        const std::filesystem::path & evidence_dir,
        bool v2) {
    for (int round = 1; round <= 2; ++round) {
        for (const exam_question & question : questions) {
            if (!run_exam_request(model, vocab, params, question, evidence_dir, v2, round)) {
                return false;
            }
        }
    }
    return true;
}

bool run_m3(
        llama_model * model,
        const llama_vocab * vocab,
        const common_params & params,
        const m3_prompt & prompt,
        const std::filesystem::path & evidence_dir,
        const std::string & engine,
        bool trace_mode) {
    const bool v2 = engine == "pipeline" || engine == "depth1";
    const int n_outputs = params.n_predict;
    const std::string run_label = "m3-" + prompt.id + "-" + engine;
    if (!set_env("GGML_KIMI_PIPELINE_M1_RUN", run_label) ||
            !unset_env("GGML_KIMI_PIPELINE_V2_ACTIVE")) {
        return false;
    }

    const std::vector<llama_token> prompt_tokens = common_tokenize(vocab, prompt.text, true, true);
    const bool prompt_count_ok = prompt.id == "p0" ? prompt_tokens.size() == 1 :
        prompt_tokens.size() >= 64 && prompt_tokens.size() <= 256;
    if (!prompt_count_ok || prompt_tokens.size() + static_cast<size_t>(n_outputs) > k_n_ctx) {
        std::fprintf(stderr,
                "[kimi_pipeline_m3] prompt guard failed id=%s tokens=%zu outputs=%d\n",
                prompt.id.c_str(), prompt_tokens.size(), n_outputs);
        return false;
    }

    llama_context_params cparams = common_context_params_to_llama(params);
    cparams.n_ctx = k_n_ctx;
    cparams.n_batch = k_n_batch;
    cparams.n_ubatch = k_n_batch;
    cparams.n_seq_max = 1;
    cparams.n_threads = k_n_threads;
    cparams.n_threads_batch = k_n_threads;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "[kimi_pipeline_m3] context create failed\n");
        return false;
    }

    begin_moe_request(132000 + static_cast<uint64_t>(prompt.id[1] - '0'));
    report_moe_phase("m3_request_begin");
    bool ok = decode_prompt(ctx, prompt_tokens);
    if (ok && v2) {
        ok = set_env("GGML_KIMI_PIPELINE_V2_ACTIVE", "1");
    }
    if (!ok || !logits_finite(ctx, vocab)) {
        std::fprintf(stderr, "[kimi_pipeline_m3] prefill failed id=%s engine=%s\n",
                prompt.id.c_str(), engine.c_str());
        unset_env("GGML_KIMI_PIPELINE_V2_ACTIVE");
        llama_free(ctx);
        return false;
    }

    llama_token token = greedy_token(ctx, vocab);
    if (token == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab, token)) {
        std::fprintf(stderr, "[kimi_pipeline_m3] first output token invalid/eog\n");
        unset_env("GGML_KIMI_PIPELINE_V2_ACTIVE");
        llama_free(ctx);
        return false;
    }

    std::vector<llama_token> generated;
    generated.reserve(static_cast<size_t>(n_outputs));
    generated.push_back(token);
    std::vector<m3_token_timing> timings;
    timings.reserve(static_cast<size_t>(n_outputs - 1));
    const char * block_env = std::getenv("TASK132_M3_BLOCK_DEVICE");
    const std::string block_device = block_env && block_env[0] ? block_env : "nvme0n1";

    report_moe_phase("m3_prefill_done");
    const m3_io_snapshot io_before = read_m3_io_snapshot(block_device);
    const int64_t window_start_ns = monotonic_ns();
    for (int output_index = 1; ok && output_index < n_outputs; ++output_index) {
        m3_token_timing timing;
        timing.output_index = output_index;
        timing.input_token = token;
        const int64_t interval_start_ns = monotonic_ns();
        timing.decode_start_ns = interval_start_ns;
        llama_batch batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr,
                    "[kimi_pipeline_m3] decode failed id=%s engine=%s output=%d\n",
                    prompt.id.c_str(), engine.c_str(), output_index);
            ok = false;
            break;
        }
        llama_synchronize(ctx);
        timing.decode_end_ns = monotonic_ns();
        timing.decode_ns = timing.decode_end_ns - timing.decode_start_ns;
        const int64_t sample_start_ns = timing.decode_end_ns;
        if (!logits_finite(ctx, vocab)) {
            ok = false;
            break;
        }
        token = greedy_token(ctx, vocab);
        const int64_t interval_end_ns = monotonic_ns();
        timing.sample_ns = interval_end_ns - sample_start_ns;
        timing.interval_ns = interval_end_ns - interval_start_ns;
        timings.push_back(timing);
        if (token == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab, token)) {
            std::fprintf(stderr,
                    "[kimi_pipeline_m3] output token invalid/eog id=%s engine=%s output=%d\n",
                    prompt.id.c_str(), engine.c_str(), output_index);
            ok = false;
            break;
        }
        generated.push_back(token);
    }
    const int64_t window_end_ns = monotonic_ns();
    const m3_io_snapshot io_after = read_m3_io_snapshot(block_device);
    report_moe_phase("m3_request_end");

    ok = ok && generated.size() == static_cast<size_t>(n_outputs) &&
        timings.size() == static_cast<size_t>(n_outputs - 1);
    std::vector<int64_t> decode_values;
    std::vector<int64_t> interval_values;
    uint64_t decode_sum_ns = 0;
    uint64_t sample_sum_ns = 0;
    for (const m3_token_timing & timing : timings) {
        decode_values.push_back(timing.decode_ns);
        interval_values.push_back(timing.interval_ns);
        decode_sum_ns += static_cast<uint64_t>(timing.decode_ns);
        sample_sum_ns += static_cast<uint64_t>(timing.sample_ns);
    }

    std::ofstream timing_file(evidence_dir / "token-wall.csv", std::ios::trunc);
    timing_file << "output_index,input_token,decode_start_monotonic_ns,decode_end_monotonic_ns,"
                   "decode_ns,sample_ns,interval_ns\n";
    for (const m3_token_timing & timing : timings) {
        timing_file << timing.output_index << ',' << timing.input_token << ','
                    << timing.decode_start_ns << ',' << timing.decode_end_ns << ','
                    << timing.decode_ns << ',' << timing.sample_ns << ','
                    << timing.interval_ns << '\n';
    }

    std::ofstream tokens_file(evidence_dir / "tokens.txt", std::ios::trunc);
    std::ostringstream token_canonical;
    for (llama_token generated_token : generated) {
        tokens_file << generated_token << '\n';
        token_canonical << generated_token << '\n';
    }
    const std::string token_canonical_bytes = token_canonical.str();
    std::ofstream content_file(
            evidence_dir / "content.bin", std::ios::binary | std::ios::trunc);
    std::string content;
    for (llama_token generated_token : generated) {
        content += common_token_to_piece(ctx, generated_token, true);
    }
    content_file.write(content.data(), static_cast<std::streamsize>(content.size()));

    const uint64_t window_ns = window_end_ns >= window_start_ns ?
        static_cast<uint64_t>(window_end_ns - window_start_ns) : 0;
    const uint64_t block_sector_delta = counter_delta(
            io_after.block_sectors_read, io_before.block_sectors_read);
    const uint64_t block_bytes = block_sector_delta * io_before.block_sector_size;
    ok = ok && io_before.block_sector_size > 0;
    json metrics;
    metrics["schema"] = "task132-m3-direct-n64-v1";
    metrics["status"] = ok ? "ok" : "failed";
    metrics["engine"] = engine;
    metrics["mode"] = v2 ? engine : "legacy";
    metrics["trace_mode"] = trace_mode;
    metrics["prompt_id"] = prompt.id;
    metrics["prompt_category"] = prompt.category;
    metrics["prompt_utf8_sha256"] = prompt.prompt_sha256;
    metrics["preregister_sha256"] = prompt.preregister_sha256;
    metrics["prompt_tokens"] = prompt_tokens.size();
    metrics["outputs"] = generated.size();
    metrics["decode_evals"] = timings.size();
    metrics["normalization_outputs"] = n_outputs;
    metrics["generation_window_ns"] = window_ns;
    metrics["wall_ms_per_output"] = static_cast<double>(window_ns) / 1e6 / n_outputs;
    metrics["tok_per_second"] = window_ns > 0 ?
        static_cast<double>(n_outputs) * 1e9 / static_cast<double>(window_ns) : 0.0;
    metrics["decode_call_sum_ns"] = decode_sum_ns;
    metrics["sample_sum_ns"] = sample_sum_ns;
    metrics["eval_decode_p50_ms"] = percentile_ms(decode_values, 0.50);
    metrics["eval_decode_p90_ms"] = percentile_ms(decode_values, 0.90);
    metrics["eval_decode_p95_ms"] = percentile_ms(decode_values, 0.95);
    metrics["eval_interval_p50_ms"] = percentile_ms(interval_values, 0.50);
    metrics["eval_interval_p90_ms"] = percentile_ms(interval_values, 0.90);
    metrics["process_read_bytes"] = counter_delta(
            io_after.process_read_bytes, io_before.process_read_bytes);
    metrics["block_device"] = block_device;
    metrics["block_reads"] = counter_delta(io_after.block_reads, io_before.block_reads);
    metrics["block_read_bytes"] = block_bytes;
    metrics["block_read_busy_ms"] = counter_delta(
            io_after.block_read_ms, io_before.block_read_ms);
    metrics["tokens_sha256"] = sha256_hex(
            token_canonical_bytes.data(), token_canonical_bytes.size());
    metrics["content_sha256"] = sha256_hex(content.data(), content.size());
    metrics["sampling"] = {
        {"seed", 1}, {"temperature", 0.0}, {"top_k", 1}, {"cache_prompt", false}};
    metrics["projection_parity"] = false;
    metrics["cuda_graphs"] = false;
    metrics["prefill_engine"] = "legacy";

    std::ofstream metrics_file(evidence_dir / "metrics.json", std::ios::trunc);
    metrics_file << std::setw(2) << metrics << '\n';
    timing_file.flush();
    tokens_file.flush();
    content_file.flush();
    metrics_file.flush();
    ok = ok && timing_file.good() && tokens_file.good() && content_file.good() &&
        metrics_file.good();

    unset_env("GGML_KIMI_PIPELINE_V2_ACTIVE");
    llama_free(ctx);
    std::fprintf(stderr,
            "[kimi_pipeline_m3] engine=%s prompt=%s prompt_tokens=%zu outputs=%zu "
            "evals=%zu wall_ms_per_output=%.6f tok_s=%.6f status=%s\n",
            engine.c_str(), prompt.id.c_str(), prompt_tokens.size(), generated.size(),
            timings.size(), static_cast<double>(window_ns) / 1e6 / n_outputs,
            window_ns > 0 ? static_cast<double>(n_outputs) * 1e9 /
                static_cast<double>(window_ns) : 0.0,
            ok ? "ok" : "failed");
    return ok;
}

bool run_once(
        llama_model * model,
        const llama_vocab * vocab,
        const common_params & params,
        const std::vector<llama_token> & prompt_tokens,
        const std::filesystem::path & evidence_dir,
        const char * mode,
        int run_index) {
    const std::string run_name = "run" + std::to_string(run_index);
    const std::filesystem::path run_dir = evidence_dir / run_name;
    std::filesystem::create_directories(run_dir);
    if (!set_env("GGML_KIMI_PIPELINE_M1_RUN", run_name) ||
            !unset_env("GGML_KIMI_PIPELINE_V2_ACTIVE")) {
        return false;
    }

    llama_context_params cparams = common_context_params_to_llama(params);
    cparams.n_ctx = k_n_ctx;
    cparams.n_batch = k_n_batch;
    cparams.n_ubatch = k_n_batch;
    cparams.n_seq_max = 1;
    cparams.n_threads = k_n_threads;
    cparams.n_threads_batch = k_n_threads;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "[kimi_pipeline_m1] failed to create context for %s\n", run_name.c_str());
        return false;
    }

    bool ok = decode_prompt(ctx, prompt_tokens);
    if (!ok) {
        std::fprintf(stderr, "[kimi_pipeline_m1] legacy prefill failed for %s\n", run_name.c_str());
        llama_free(ctx);
        return false;
    }
    if (!set_env("GGML_KIMI_PIPELINE_V2_ACTIVE", "1")) {
        llama_free(ctx);
        return false;
    }

    std::ofstream checkpoints(run_dir / "token-checkpoints.csv", std::ios::trunc);
    std::ofstream tokens_file(run_dir / "tokens.txt", std::ios::trunc);
    std::ofstream content_file(run_dir / "content.bin", std::ios::binary | std::ios::trunc);
    checkpoints << "step,token_id,logits_sha256,state_sha256,state_bytes\n";

    std::vector<llama_token> generated;
    generated.reserve(k_n_predict);
    for (int step = 0; ok && step < k_n_predict; ++step) {
        const llama_token token = greedy_token(ctx, vocab);
        if (token == LLAMA_TOKEN_NULL) {
            ok = false;
            break;
        }
        generated.push_back(token);
        tokens_file << token << '\n';
        const std::string piece = common_token_to_piece(ctx, token, true);
        content_file.write(piece.data(), static_cast<std::streamsize>(piece.size()));

        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(&generated.back()), 1);
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "[kimi_pipeline_m1] v2 decode failed run=%s step=%d\n",
                    run_name.c_str(), step);
            ok = false;
            break;
        }
        llama_synchronize(ctx);

        const float * logits = llama_get_logits_ith(ctx, -1);
        const size_t logits_bytes = (size_t) llama_vocab_n_tokens(vocab) * sizeof(float);
        const size_t state_size = llama_state_seq_get_size_ext(
                ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        std::vector<uint8_t> state(state_size);
        const size_t state_written = state_size > 0 ? llama_state_seq_get_data_ext(
                ctx, state.data(), state.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) : 0;
        if (!logits || state_size == 0 || state_written != state_size) {
            std::fprintf(stderr,
                    "[kimi_pipeline_m1] checkpoint failed run=%s step=%d state=%zu/%zu logits=%p\n",
                    run_name.c_str(), step, state_written, state_size, (const void *) logits);
            ok = false;
            break;
        }
        checkpoints << step << ',' << token << ',' << sha256_hex(logits, logits_bytes) << ','
            << sha256_hex(state.data(), state.size()) << ',' << state.size() << '\n';
    }

    checkpoints.flush();
    tokens_file.flush();
    content_file.flush();
    ok = ok && generated.size() == k_n_predict && checkpoints.good() &&
        tokens_file.good() && content_file.good();

    std::ofstream manifest(run_dir / "manifest.txt", std::ios::trunc);
    manifest << "run=" << run_name << '\n'
             << "mode=" << mode << '\n'
             << "prompt=hello\n"
             << "prefill=legacy\n"
             << "decode_steps=" << generated.size() << '\n'
             << "seed=1\n"
             << "temperature=0\n"
             << "sampler_top_k=1\n"
             << "status=" << (ok ? "ok" : "failed") << '\n';
    ok = ok && manifest.good();

    unset_env("GGML_KIMI_PIPELINE_V2_ACTIVE");
    llama_free(ctx);
    std::fprintf(stderr, "[kimi_pipeline_m1] run=%s mode=%s decode_steps=%zu status=%s\n",
            run_name.c_str(), mode, generated.size(), ok ? "ok" : "failed");
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    std::string engine;
    std::vector<char *> filtered_argv;
    filtered_argv.reserve(static_cast<size_t>(argc));
    filtered_argv.push_back(argv[0]);
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--engine") == 0) {
            if (!engine.empty() || index + 1 >= argc) {
                std::fprintf(stderr, "[kimi_pipeline_m3] invalid or duplicate --engine\n");
                return 2;
            }
            engine = argv[++index];
        } else {
            filtered_argv.push_back(argv[index]);
        }
    }
    common_params params;
    common_init();
    if (!common_params_parse(
            static_cast<int>(filtered_argv.size()), filtered_argv.data(), params,
            LLAMA_EXAMPLE_COMPLETION, print_usage)) {
        return 2;
    }

    const char * mode = std::getenv("GGML_KIMI_PIPELINE_V2_MODE");
    const char * evidence_env = std::getenv("TASK132_EVIDENCE_DIR");
    const char * exam_env = std::getenv("TASK132_M2_EXAM");
    const char * exam_arm = std::getenv("TASK132_M2_ARM");
    const bool exam_mode = exam_env && exam_env[0];
    const char * m3_preregister_env = std::getenv("TASK132_M3_PREREGISTER");
    const char * m3_prompt_id_env = std::getenv("TASK132_M3_PROMPT_ID");
    const bool m3_mode = (m3_preregister_env && m3_preregister_env[0]) ||
        (m3_prompt_id_env && m3_prompt_id_env[0]) || !engine.empty();
    const bool m3_trace_mode = env_flag("TASK132_M3_TRACE");
    const char * repeats_env = std::getenv("TASK132_REPEATS");
    const int repeats = repeats_env ? std::atoi(repeats_env) : 1;
    m3_prompt formal_prompt;
    bool params_ok = false;
    if (m3_mode) {
        if (!m3_preregister_env || !m3_preregister_env[0] ||
                !m3_prompt_id_env || !m3_prompt_id_env[0] || engine.empty() || exam_mode) {
            std::fprintf(stderr,
                    "[kimi_pipeline_m3] --engine, TASK132_M3_PREREGISTER, and "
                    "TASK132_M3_PROMPT_ID are jointly required and exclude M2 mode\n");
        } else {
            try {
                formal_prompt = load_m3_prompt(m3_preregister_env, m3_prompt_id_env);
                params_ok = validate_m3_params(
                        params, mode, engine, formal_prompt, m3_trace_mode);
            } catch (const std::exception & error) {
                std::fprintf(stderr, "[kimi_pipeline_m3] preregistration error: %s\n", error.what());
            }
        }
    } else {
        params_ok = exam_mode ? validate_exam_params(params, mode, exam_arm) :
            validate_params(params, mode, repeats);
    }
    if (!evidence_env || !evidence_env[0] || !params_ok) {
        if (!evidence_env || !evidence_env[0]) {
            std::fprintf(stderr, "[kimi_pipeline_m1] TASK132_EVIDENCE_DIR is required\n");
        }
        return 2;
    }

    std::error_code ec;
    const std::filesystem::path evidence_dir = std::filesystem::absolute(evidence_env, ec);
    if (ec || (!std::filesystem::create_directories(evidence_dir, ec) && ec)) {
        std::fprintf(stderr, "[kimi_pipeline_m1] cannot create evidence directory: %s\n", evidence_env);
        return 2;
    }

    const std::filesystem::path route_path = evidence_dir / "route.csv";
    const std::filesystem::path ledger_path = evidence_dir / "projection-ledger.csv";
    const std::filesystem::path checkpoint_path = evidence_dir / "cpu-checkpoints.csv";
    const bool exam_v2 = exam_mode && std::strcmp(exam_arm, "v2") == 0;
    const bool m3_v2 = m3_mode && (engine == "pipeline" || engine == "depth1");
    if (!m3_mode) {
        std::ofstream(route_path, std::ios::trunc).close();
        std::ofstream(ledger_path, std::ios::trunc).close();
    }
    if (!exam_mode && !m3_mode) {
        std::ofstream(checkpoint_path, std::ios::trunc).close();
    }

    bool env_ok =
        set_env("GGML_KIMI_PIPELINE_V2_TOPK",
                m3_mode ? (m3_v2 ? "1" : "0") :
                    (exam_mode ? (exam_v2 ? "1" : "0") : "1")) &&
        set_env("LLAMA_K3_DECODE_EXPERT_MODE", "hard") &&
        set_env("LLAMA_K3_DECODE_EXPERT_TOP_K", "1") &&
        ((m3_mode ? m3_v2 : (exam_v2 || !exam_mode)) ?
            set_env("GGML_KIMI_PIPELINE_V2_PRESERVE_HARD_IDS", "1") :
            unset_env("GGML_KIMI_PIPELINE_V2_PRESERVE_HARD_IDS")) &&
        (m3_mode ? unset_env("GGML_MOE_ROUTE_SCORE_TRACE_OUT") :
            set_env("GGML_MOE_ROUTE_SCORE_TRACE_OUT", route_path.string())) &&
        (m3_mode ? unset_env("GGML_KIMI_PIPELINE_M1_LEDGER_OUT") :
            ((exam_v2 || !exam_mode) ?
                set_env("GGML_KIMI_PIPELINE_M1_LEDGER_OUT", ledger_path.string()) :
                unset_env("GGML_KIMI_PIPELINE_M1_LEDGER_OUT"))) &&
        ((!exam_mode && !m3_mode) ?
            set_env("GGML_KIMI_PIPELINE_M1_CHECKPOINT_OUT", checkpoint_path.string()) :
            unset_env("GGML_KIMI_PIPELINE_M1_CHECKPOINT_OUT")) &&
        set_env("LLAMA_GRAPH_REUSE_DISABLE", "1") &&
        set_env("GGML_KIMI_DENSE_PREFETCH_LAYERS", "0-2") &&
        set_env("GGML_KIMI_DENSE_PREFETCH_BEFORE_INPUTS", "1") &&
        set_env("GGML_KIMI_DENSE_PREFETCH_PREFIX",
                m3_mode ? (m3_v2 ? "1" : "0") :
                    (exam_mode ? (exam_v2 ? "1" : "0") : "1")) &&
        set_env("GGML_KIMI_DENSE_PREFETCH_PREFIX_STREAM",
                m3_mode ? (engine == "pipeline" ? "1" : "0") :
                    (exam_mode ? (exam_v2 ? "1" : "0") :
                    (std::strcmp(mode, "pipeline") == 0 ? "1" : "0"))) &&
        set_env("GGML_KIMI_DENSE_PUBLICATION_THREAD",
                m3_mode ? (engine == "pipeline" ? "1" : "0") :
                    (exam_mode ? (exam_v2 ? "1" : "0") :
                    (std::strcmp(mode, "pipeline") == 0 ? "1" : "0"))) &&
        unset_env("GGML_KIMI_DENSE_FULL_DOUBLE_BUFFER") &&
        unset_env("GGML_KIMI_PIPELINE_V2_ACTIVE");
    if (!env_ok) {
        std::fprintf(stderr, "[kimi_pipeline_m1] failed to establish private environment: %s\n",
                std::strerror(errno));
        return 2;
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    if ((m3_mode ? m3_v2 : (!exam_mode || exam_v2)) &&
            !run_top_k_self_test(evidence_dir)) {
        llama_backend_free();
        return 3;
    }

    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "[kimi_pipeline_m1] model load failed\n");
        llama_backend_free();
        return 4;
    }

    char architecture[64] = {};
    const bool model_guard =
        llama_model_meta_val_str(model, "general.architecture", architecture, sizeof(architecture)) >= 0 &&
        std::strcmp(architecture, "kimi-k3") == 0 &&
        llama_model_n_layer(model) == 93 &&
        !llama_model_has_encoder(model);
    if (!model_guard) {
        std::fprintf(stderr, "[kimi_pipeline_m1] model guard failed: arch=%s layers=%d encoder=%d\n",
                architecture, llama_model_n_layer(model), llama_model_has_encoder(model) ? 1 : 0);
        llama_model_free(model);
        llama_backend_free();
        return 5;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (m3_mode) {
        const bool ok = run_m3(
                model, vocab, params, formal_prompt, evidence_dir, engine, m3_trace_mode);
        llama_model_free(model);
        llama_backend_free();
        return ok ? 0 : 6;
    }
    if (exam_mode) {
        bool ok = false;
        try {
            const std::vector<exam_question> questions = load_exam(exam_env);
            ok = run_exam(model, vocab, params, questions, evidence_dir, exam_v2);
        } catch (const std::exception & error) {
            std::fprintf(stderr, "[kimi_pipeline_m2] fixed exam error: %s\n", error.what());
        }
        llama_model_free(model);
        llama_backend_free();
        return ok ? 0 : 6;
    }
    const std::vector<llama_token> prompt_tokens = common_tokenize(vocab, params.prompt, true, true);
    if (prompt_tokens.empty() || prompt_tokens.size() + k_n_predict > k_n_ctx) {
        std::fprintf(stderr, "[kimi_pipeline_m1] prompt token guard failed: tokens=%zu\n", prompt_tokens.size());
        llama_model_free(model);
        llama_backend_free();
        return 5;
    }

    bool ok = true;
    for (int run = 0; ok && run < repeats; ++run) {
        ok = run_once(model, vocab, params, prompt_tokens, evidence_dir, mode, run);
    }
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 6;
}
