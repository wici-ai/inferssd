#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

struct results_extra_params {
    std::string top1_report;
    bool        top1_fail_on_mismatch = false;
    bool        sequential_logits = false;
};

static bool parse_results_extra_args(
        int argc, char ** argv, results_extra_params & extra, std::vector<std::string> & args_storage, std::vector<char *> & args_forward) {
    args_storage.clear();
    args_forward.clear();
    args_storage.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--top1-report") {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --top1-report requires a path\n", __func__);
                return false;
            }
            extra.top1_report = argv[++i];
            continue;
        }
        if (arg.rfind("--top1-report=", 0) == 0) {
            extra.top1_report = arg.substr(std::string("--top1-report=").size());
            continue;
        }
        if (arg == "--top1-fail-on-mismatch") {
            extra.top1_fail_on_mismatch = true;
            continue;
        }
        if (arg == "--sequential-logits") {
            extra.sequential_logits = true;
            continue;
        }
        args_storage.push_back(arg);
    }

    args_forward.reserve(args_storage.size());
    for (auto & arg : args_storage) {
        args_forward.push_back(arg.data());
    }
    return true;
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, const bool sequential_logits) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_batch  = llama_n_batch(lctx);
    const uint32_t n_tokens = tokens.size();
    GGML_ASSERT(n_tokens <= n_ctx);

    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    const uint32_t step = sequential_logits ? 1 : n_batch;
    for (uint32_t pos = 0; pos < n_tokens; pos += step) {
        const uint32_t n_cur = std::min(step, n_tokens - pos);
        common_batch_clear(batch);
        for (uint32_t i = 0; i < n_cur; ++i) {
            common_batch_add(batch, tokens[pos + i], pos + i, {0}, true);
        }
        if (llama_decode(lctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to decode batch");
        }

        for (uint32_t i = 0; i < n_cur; i++) {
            const float * logits_ith = llama_get_logits_ith(lctx, i);
            for (uint32_t j = 0; j < n_vocab; j++) {
                ret.push_back(logits_ith[j]);
            }
        }
    }

    llama_batch_free(batch);
    return ret;
}

struct top2_result {
    llama_token top1_id    = -1;
    llama_token top2_id    = -1;
    float       top1_logit = -std::numeric_limits<float>::infinity();
    float       top2_logit = -std::numeric_limits<float>::infinity();
};

struct top1_report_summary {
    uint32_t n_tokens = 0;
    uint32_t n_vocab  = 0;
    uint32_t same_top1 = 0;
    uint32_t next_token_positions = 0;
    uint32_t base_top1_matches_next_token = 0;
    uint32_t calc_top1_matches_next_token = 0;
    int32_t  first_mismatch_pos = -1;
    double   max_abs = 0.0;
    double   mean_abs = 0.0;
};

static top2_result get_top2(const float * logits, const uint32_t n_vocab) {
    top2_result res;
    for (uint32_t i = 0; i < n_vocab; ++i) {
        const float v = logits[i];
        if (v > res.top1_logit) {
            res.top2_id    = res.top1_id;
            res.top2_logit = res.top1_logit;
            res.top1_id    = (llama_token) i;
            res.top1_logit = v;
        } else if (v > res.top2_logit) {
            res.top2_id    = (llama_token) i;
            res.top2_logit = v;
        }
    }
    return res;
}

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    static const char * hex = "0123456789abcdef";
                    out += "\\u00";
                    out += hex[c >> 4];
                    out += hex[c & 0x0f];
                } else {
                    out += (char) c;
                }
                break;
        }
    }
    return out;
}

static void write_token_info(
        std::ostream & out, const llama_context * ctx, const char * name, llama_token id, float logit) {
    out << "\"" << name << "\":{";
    out << "\"id\":" << id << ",";
    out << "\"piece\":\"" << json_escape(common_token_to_piece(ctx, id, true)) << "\",";
    out << "\"logit\":" << logit;
    out << "}";
}

static void write_mismatch(
        std::ostream & out,
        const llama_context * ctx,
        const std::vector<llama_token> & tokens,
        uint32_t pos,
        const top2_result & base,
        const top2_result & calc) {
    out << "{";
    out << "\"position\":" << pos << ",";
    out << "\"predicts_token_position\":";
    if (pos + 1 < tokens.size()) {
        out << (pos + 1) << ",";
        write_token_info(out, ctx, "actual_next_token", tokens[pos + 1], 0.0f);
        out << ",";
    } else {
        out << "null,";
        out << "\"actual_next_token\":null,";
    }
    write_token_info(out, ctx, "base_top1", base.top1_id, base.top1_logit);
    out << ",";
    write_token_info(out, ctx, "base_top2", base.top2_id, base.top2_logit);
    out << ",";
    write_token_info(out, ctx, "calc_top1", calc.top1_id, calc.top1_logit);
    out << ",";
    write_token_info(out, ctx, "calc_top2", calc.top2_id, calc.top2_logit);
    out << ",";
    out << "\"base_margin\":" << (base.top1_logit - base.top2_logit) << ",";
    out << "\"calc_margin\":" << (calc.top1_logit - calc.top2_logit);
    out << "}";
}

static top1_report_summary write_top1_report(
        const std::string & path,
        const llama_context * ctx,
        const uint32_t n_vocab,
        const std::vector<llama_token> & tokens,
        const std::vector<float> & logits_base,
        const std::vector<float> & logits_calc) {
    GGML_ASSERT(logits_base.size() == logits_calc.size());
    GGML_ASSERT(logits_calc.size() == tokens.size() * (size_t) n_vocab);

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open top1 report: " + path);
    }

    top1_report_summary summary;
    summary.n_tokens = tokens.size();
    summary.n_vocab  = n_vocab;
    summary.next_token_positions = tokens.empty() ? 0 : tokens.size() - 1;

    std::vector<std::pair<uint32_t, std::pair<top2_result, top2_result>>> preview;
    preview.reserve(16);

    double sum_abs = 0.0;
    size_t n_values = 0;
    for (uint32_t pos = 0; pos < tokens.size(); ++pos) {
        const float * base = logits_base.data() + (size_t) pos * n_vocab;
        const float * calc = logits_calc.data() + (size_t) pos * n_vocab;
        const top2_result base_top = get_top2(base, n_vocab);
        const top2_result calc_top = get_top2(calc, n_vocab);
        if (base_top.top1_id == calc_top.top1_id) {
            ++summary.same_top1;
        } else {
            if (summary.first_mismatch_pos < 0) {
                summary.first_mismatch_pos = pos;
            }
            if (preview.size() < 16) {
                preview.push_back({pos, {base_top, calc_top}});
            }
        }
        if (pos + 1 < tokens.size()) {
            summary.base_top1_matches_next_token += base_top.top1_id == tokens[pos + 1];
            summary.calc_top1_matches_next_token += calc_top.top1_id == tokens[pos + 1];
        }

        for (uint32_t i = 0; i < n_vocab; ++i) {
            const double diff = std::fabs((double) base[i] - (double) calc[i]);
            summary.max_abs = std::max(summary.max_abs, diff);
            sum_abs += diff;
        }
        n_values += n_vocab;
    }
    summary.mean_abs = n_values ? sum_abs / (double) n_values : 0.0;

    out << "{\n";
    out << "  \"n_tokens\": " << summary.n_tokens << ",\n";
    out << "  \"n_vocab\": " << summary.n_vocab << ",\n";
    out << "  \"same_top1\": " << summary.same_top1 << ",\n";
    out << "  \"same_top1_ratio\": " << (summary.n_tokens ? (double) summary.same_top1 / summary.n_tokens : 0.0) << ",\n";
    out << "  \"first_mismatch_pos\": " << summary.first_mismatch_pos << ",\n";
    out << "  \"next_token_positions\": " << summary.next_token_positions << ",\n";
    out << "  \"base_top1_matches_next_token\": " << summary.base_top1_matches_next_token << ",\n";
    out << "  \"calc_top1_matches_next_token\": " << summary.calc_top1_matches_next_token << ",\n";
    out << "  \"max_abs\": " << summary.max_abs << ",\n";
    out << "  \"mean_abs\": " << summary.mean_abs << ",\n";
    out << "  \"mismatches_preview\": [";
    for (size_t i = 0; i < preview.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << "\n    ";
        write_mismatch(out, ctx, tokens, preview[i].first, preview[i].second.first, preview[i].second.second);
    }
    if (!preview.empty()) {
        out << "\n  ";
    }
    out << "]\n";
    out << "}\n";

    return summary;
}

static void write_single_top1_report(
        const std::string & path,
        const llama_context * ctx,
        const uint32_t n_vocab,
        const std::vector<llama_token> & tokens,
        const std::vector<float> & logits) {
    GGML_ASSERT(logits.size() == tokens.size() * (size_t) n_vocab);

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open top1 report: " + path);
    }

    uint32_t next_matches = 0;
    std::vector<top2_result> tops;
    tops.reserve(tokens.size());
    for (uint32_t pos = 0; pos < tokens.size(); ++pos) {
        const float * row = logits.data() + (size_t) pos * n_vocab;
        tops.push_back(get_top2(row, n_vocab));
        if (pos + 1 < tokens.size()) {
            next_matches += tops.back().top1_id == tokens[pos + 1];
        }
    }

    out << "{\n";
    out << "  \"mode\": \"single\",\n";
    out << "  \"n_tokens\": " << tokens.size() << ",\n";
    out << "  \"n_vocab\": " << n_vocab << ",\n";
    out << "  \"next_token_positions\": " << (tokens.empty() ? 0 : tokens.size() - 1) << ",\n";
    out << "  \"top1_matches_next_token\": " << next_matches << ",\n";
    out << "  \"positions\": [";
    for (uint32_t pos = 0; pos < tokens.size(); ++pos) {
        if (pos > 0) {
            out << ",";
        }
        out << "\n    {";
        out << "\"position\":" << pos << ",";
        out << "\"predicts_token_position\":";
        if (pos + 1 < tokens.size()) {
            out << (pos + 1) << ",";
            write_token_info(out, ctx, "actual_next_token", tokens[pos + 1], 0.0f);
            out << ",";
            out << "\"top1_matches_next_token\":" << (tops[pos].top1_id == tokens[pos + 1] ? "true" : "false") << ",";
        } else {
            out << "null,";
            out << "\"actual_next_token\":null,";
            out << "\"top1_matches_next_token\":null,";
        }
        write_token_info(out, ctx, "top1", tops[pos].top1_id, tops[pos].top1_logit);
        out << ",";
        write_token_info(out, ctx, "top2", tops[pos].top2_id, tops[pos].top2_logit);
        out << ",";
        out << "\"margin\":" << (tops[pos].top1_logit - tops[pos].top2_logit);
        out << "}";
    }
    if (!tokens.empty()) {
        out << "\n  ";
    }
    out << "]\n";
    out << "}\n";
}

int main(int argc, char ** argv) {
    common_params params;
    params.escape = false;
    results_extra_params extra;
    std::vector<std::string> args_storage;
    std::vector<char *> args_forward;

    common_init();

    if (!parse_results_extra_args(argc, argv, extra, args_storage, args_forward)) {
        return 1;
    }
    if (!common_params_parse((int) args_forward.size(), args_forward.data(), params, LLAMA_EXAMPLE_RESULTS)) {
        return 1;
    }
    if (params.out_file.empty() && (params.check || extra.top1_report.empty())) {
        LOG_ERR("%s: an output file must be specified", __func__);
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);
    common_init_result_ptr llama_init = common_init_from_params(params);
    struct llama_model   * model = llama_init->model();
    struct llama_context * lctx  = llama_init->context();
    if (model == nullptr) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    const std::vector<llama_token> tokens_calc = common_tokenize(lctx, params.prompt, true);
    const std::vector<float> logits_calc = get_logits(model, lctx, tokens_calc, extra.sequential_logits);
    GGML_ASSERT(logits_calc.size() == tokens_calc.size()*n_vocab);

    if (params.check) {
        LOG_INF("%s: loading results from %s...\n", __func__, params.out_file.c_str());
        gguf_context_ptr gguf_ctx;
        {
            struct gguf_init_params gguf_params = {
                /*no_alloc =*/ true,
                /*ctx      =*/ nullptr,
            };
            gguf_ctx.reset(gguf_init_from_file(params.out_file.c_str(), gguf_params));
        }
        const std::string path_model_disk = gguf_get_val_str(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), "path_model"));
        GGML_ASSERT(path_model_disk == params.model.path); // TODO better checks

        auto load_tensor_data = [&](const std::string & name, void * dst, const size_t size){
            const int64_t tid    = gguf_find_tensor(gguf_ctx.get(), name.c_str());
            const size_t  offset = gguf_get_data_offset(gguf_ctx.get()) + gguf_get_tensor_offset(gguf_ctx.get(), tid);
            GGML_ASSERT(size == gguf_get_tensor_size(gguf_ctx.get(), tid));

            FILE * file = ggml_fopen(params.out_file.c_str(), "rb");
            if (file == nullptr) {
                throw std::runtime_error("failed to open results file");
            }
            if (fseek(file, offset, SEEK_SET) != 0) {
                throw std::runtime_error("fseek failed");
            }
            const size_t nbytes_read = fread(dst, 1, size, file);
            if (nbytes_read != size) {
                throw std::runtime_error("fread failed");
            }
        };

        std::vector<llama_token> tokens_disk(tokens_calc.size());
        load_tensor_data("tokens", tokens_disk.data(), tokens_disk.size()*sizeof(llama_token));
        GGML_ASSERT(tokens_disk.size() == tokens_calc.size());
        for (size_t i = 0; i < tokens_calc.size(); i++) {
            GGML_ASSERT(tokens_disk[i] == tokens_calc[i]);
        }

        std::vector<float> logits_disk(logits_calc.size());
        load_tensor_data("logits", logits_disk.data(), logits_disk.size()*sizeof(float));
        const double nmse_val = nmse(logits_disk, logits_calc);
        LOG_INF("%s: NMSE=%.3e\n", __func__, nmse_val);

        bool top1_mismatch = false;
        if (!extra.top1_report.empty()) {
            const top1_report_summary top1 = write_top1_report(
                    extra.top1_report, lctx, n_vocab, tokens_calc, logits_disk, logits_calc);
            top1_mismatch = top1.same_top1 != top1.n_tokens;
            LOG_INF("%s: top1 same=%u/%u first_mismatch=%d max_abs=%.6g mean_abs=%.6g report=%s\n",
                    __func__, top1.same_top1, top1.n_tokens, top1.first_mismatch_pos,
                    top1.max_abs, top1.mean_abs, extra.top1_report.c_str());
        }

        if (extra.top1_fail_on_mismatch && top1_mismatch) {
            printf("\033[1;31mFAIL\033[0m\n");
            return 1;
        }

        if (nmse_val > 1e-6) {
            printf("\033[1;31mFAIL\033[0m\n");
            return 1;
        }

        printf("\033[1;32mOK\033[0m\n");
        return 0;
    }

    if (!extra.top1_report.empty()) {
        write_single_top1_report(extra.top1_report, lctx, n_vocab, tokens_calc, logits_calc);
        LOG_INF("%s: wrote top1 report to %s\n", __func__, extra.top1_report.c_str());
        if (params.out_file.empty()) {
            return 0;
        }
    }

    ggml_context_ptr ggml_ctx_calc;
    {
        const size_t size_tokens = tokens_calc.size()*sizeof(llama_token) + ggml_tensor_overhead();
        const size_t size_logits = logits_calc.size()*sizeof(float)  + ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_tokens + size_logits,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        ggml_ctx_calc.reset(ggml_init(params));
    }

    gguf_context_ptr gguf_ctx(gguf_init_empty());
    gguf_set_val_str(gguf_ctx.get(), "path_model", params.model.path.c_str());
    {
        ggml_tensor * t_tokens = ggml_new_tensor_1d(ggml_ctx_calc.get(), GGML_TYPE_I32, tokens_calc.size());
        ggml_set_name(t_tokens, "tokens");
        int32_t * tokens_data = (int32_t *) t_tokens->data;
        for (uint32_t i = 0; i < tokens_calc.size(); i++) {
            tokens_data[i] = tokens_calc[i];
        }
        gguf_add_tensor(gguf_ctx.get(), t_tokens);
    }
    {
        ggml_tensor * t_logits = ggml_new_tensor_2d(ggml_ctx_calc.get(), GGML_TYPE_F32, tokens_calc.size(), n_vocab);
        ggml_set_name(t_logits, "logits");
        float * logits_data = ggml_get_data_f32(t_logits);
        std::copy(logits_calc.begin(), logits_calc.end(), logits_data);
        gguf_add_tensor(gguf_ctx.get(), t_logits);
    }
    LOG_INF("%s: writing results to %s...\n", __func__, params.out_file.c_str());
    gguf_write_to_file(gguf_ctx.get(), params.out_file.c_str(), /*only_meta =*/ false);
    return 0;
}
