#pragma once

#include "llama-arch.h"
#include "llama-hparams.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

static inline bool llama_string_contains_case_insensitive(const std::string & haystack, const char * needle) {
    if (needle == nullptr || needle[0] == '\0') {
        return true;
    }

    const char * needle_end = needle + std::strlen(needle);
    const auto it = std::search(
            haystack.begin(), haystack.end(),
            needle, needle_end,
            [](char a, char b) {
                return std::tolower((unsigned char) a) == std::tolower((unsigned char) b);
            });

    return it != haystack.end();
}

static inline bool llama_model_name_is_kimi(const std::string & name) {
    return llama_string_contains_case_insensitive(name, "kimi");
}

static inline bool llama_is_kimi_k2_deepseek2_layout(llm_arch arch, const llama_hparams & hparams) {
    return arch == LLM_ARCH_DEEPSEEK2 &&
        hparams.n_expert == 384 &&
        hparams.n_expert_groups == 1;
}

// With --defer-experts (mmap), keep routed-expert tensors in mmap-backed CPU
// buffers even when layers are offloaded, so they are never uploaded to VRAM.
// DeepSeek-V4.1 streams them from an expert pack through the CPU sink.
static inline bool llama_should_defer_kimi_experts_on_gpu(llm_arch arch, const std::string & name) {
    return arch == LLM_ARCH_KIMI_LINEAR ||
        arch == LLM_ARCH_KIMI_K3 ||
        arch == LLM_ARCH_DEEPSEEK41 ||
        arch == LLM_ARCH_MUSE_SPARK ||
        (arch == LLM_ARCH_DEEPSEEK2 && llama_model_name_is_kimi(name));
}

static inline bool llama_moe_should_use_plain_top_k(llm_arch arch, const llama_hparams & hparams) {
    return arch == LLM_ARCH_MISTRAL4 ||
        arch == LLM_ARCH_KIMI_LINEAR ||
        arch == LLM_ARCH_KIMI_K3 ||
        llama_is_kimi_k2_deepseek2_layout(arch, hparams);
}

enum class llama_k3_decode_expert_mode {
    disabled,
    hard,
    mass,
};

enum class llama_k3_decode_expert_policy_error {
    none,
    invalid_model_shape,
    invalid_mode,
    invalid_top_k,
    invalid_mass,
    invalid_min_k,
    invalid_max_k,
    invalid_layer_range,
};

struct llama_k3_decode_expert_graph_args {
    int64_t hard_top_k = 0;
    float mass_threshold = 0.0f;
    int64_t mass_min = 0;
};

struct llama_k3_decode_expert_policy {
    llama_k3_decode_expert_mode mode = llama_k3_decode_expert_mode::disabled;
    llama_k3_decode_expert_policy_error error = llama_k3_decode_expert_policy_error::none;
    int64_t top_k = 0;
    float mass = 0.0f;
    int64_t min_k = 4;
    int64_t max_k = 0;
    int64_t layer_start = 0;
    int64_t layer_end = 0;
    bool small_batch = false;

    bool operator==(const llama_k3_decode_expert_policy & other) const {
        // mass is a graph constant too: topology alone is not a sufficient key.
        return mode == other.mode && error == other.error && top_k == other.top_k &&
            mass == other.mass && min_k == other.min_k && max_k == other.max_k &&
            layer_start == other.layer_start && layer_end == other.layer_end &&
            small_batch == other.small_batch;
    }

    bool enabled() const {
        return mode != llama_k3_decode_expert_mode::disabled;
    }

    bool valid() const {
        return error == llama_k3_decode_expert_policy_error::none;
    }

    llama_k3_decode_expert_graph_args graph_args(int64_t n_tokens, int64_t layer) const {
        llama_k3_decode_expert_graph_args args;
        if (applies(n_tokens, layer)) {
            if (mode == llama_k3_decode_expert_mode::hard) {
                args.hard_top_k = top_k;
            } else if (mode == llama_k3_decode_expert_mode::mass) {
                args.hard_top_k = max_k;
                args.mass_threshold = max_k > 0 && mass == 1.0f ? 0.0f : mass;
                args.mass_min = min_k;
            }
        }
        return args;
    }

    bool applies(int64_t n_tokens, int64_t layer) const {
        const int64_t max_tokens = small_batch ? 8 : 1;
        return enabled() && valid() && n_tokens >= 1 && n_tokens <= max_tokens &&
            layer >= layer_start && layer <= layer_end;
    }
};

static inline bool llama_k3_decode_expert_small_batch_enabled(const char * value) {
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static inline const char * llama_k3_decode_expert_mode_name(llama_k3_decode_expert_mode mode) {
    switch (mode) {
        case llama_k3_decode_expert_mode::disabled: return "off";
        case llama_k3_decode_expert_mode::hard:     return "hard";
        case llama_k3_decode_expert_mode::mass:     return "mass";
    }
    return "unknown";
}

static inline const char * llama_k3_decode_expert_policy_error_name(
        llama_k3_decode_expert_policy_error error) {
    switch (error) {
        case llama_k3_decode_expert_policy_error::none:                return "none";
        case llama_k3_decode_expert_policy_error::invalid_model_shape: return "invalid model shape";
        case llama_k3_decode_expert_policy_error::invalid_mode:        return "invalid mode";
        case llama_k3_decode_expert_policy_error::invalid_top_k:       return "invalid top-k";
        case llama_k3_decode_expert_policy_error::invalid_mass:        return "invalid cumulative mass";
        case llama_k3_decode_expert_policy_error::invalid_min_k:       return "invalid minimum k";
        case llama_k3_decode_expert_policy_error::invalid_max_k:       return "invalid maximum k";
        case llama_k3_decode_expert_policy_error::invalid_layer_range: return "invalid layer range";
    }
    return "unknown";
}

static inline bool llama_k3_parse_i64_strict(const char * text, int64_t & value) {
    if (text == nullptr || text[0] == '\0') {
        return false;
    }

    errno = 0;
    char * end = nullptr;
    const long long parsed = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || end == nullptr || end[0] != '\0') {
        return false;
    }

    value = (int64_t) parsed;
    return true;
}

static inline bool llama_k3_parse_f32_strict(const char * text, float & value) {
    if (text == nullptr || text[0] == '\0') {
        return false;
    }

    errno = 0;
    char * end = nullptr;
    const float parsed = std::strtof(text, &end);
    if (errno == ERANGE || end == text || end == nullptr || end[0] != '\0' || !std::isfinite(parsed)) {
        return false;
    }

    value = parsed;
    return true;
}

static inline llama_k3_decode_expert_policy llama_k3_decode_expert_policy_parse(
        const char * mode,
        const char * top_k,
        const char * mass,
        const char * min_k,
        const char * max_k,
        const char * layer_start,
        const char * layer_end,
            int64_t   model_top_k,
            int64_t   n_layer) {
    llama_k3_decode_expert_policy policy;
    policy.top_k = model_top_k;
    policy.layer_end = n_layer - 1;

    if (model_top_k <= 0 || n_layer <= 0) {
        policy.error = llama_k3_decode_expert_policy_error::invalid_model_shape;
        return policy;
    }

    if (mode == nullptr || mode[0] == '\0' || std::strcmp(mode, "0") == 0 ||
            std::strcmp(mode, "off") == 0 || std::strcmp(mode, "disabled") == 0) {
        return policy;
    }

    if (std::strcmp(mode, "hard") == 0) {
        policy.mode = llama_k3_decode_expert_mode::hard;
        if (!llama_k3_parse_i64_strict(top_k, policy.top_k) ||
                policy.top_k < 1 || policy.top_k > model_top_k) {
            policy.error = llama_k3_decode_expert_policy_error::invalid_top_k;
            return policy;
        }
    } else if (std::strcmp(mode, "mass") == 0) {
        policy.mode = llama_k3_decode_expert_mode::mass;
        if (!llama_k3_parse_f32_strict(mass, policy.mass) || policy.mass <= 0.0f || policy.mass > 1.0f) {
            policy.error = llama_k3_decode_expert_policy_error::invalid_mass;
            return policy;
        }
        if (min_k != nullptr && min_k[0] != '\0' && !llama_k3_parse_i64_strict(min_k, policy.min_k)) {
            policy.error = llama_k3_decode_expert_policy_error::invalid_min_k;
            return policy;
        }
        if (max_k != nullptr && max_k[0] != '\0' &&
                (!llama_k3_parse_i64_strict(max_k, policy.max_k) ||
                 policy.max_k < 0 || policy.max_k > model_top_k)) {
            policy.error = llama_k3_decode_expert_policy_error::invalid_max_k;
            return policy;
        }
        if (policy.min_k < 1 || policy.min_k > model_top_k ||
                (policy.max_k > 0 && policy.min_k > policy.max_k)) {
            policy.error = llama_k3_decode_expert_policy_error::invalid_min_k;
            return policy;
        }
    } else {
        policy.error = llama_k3_decode_expert_policy_error::invalid_mode;
        return policy;
    }

    if (layer_start != nullptr && layer_start[0] != '\0' &&
            !llama_k3_parse_i64_strict(layer_start, policy.layer_start)) {
        policy.error = llama_k3_decode_expert_policy_error::invalid_layer_range;
        return policy;
    }
    if (layer_end != nullptr && layer_end[0] != '\0' &&
            !llama_k3_parse_i64_strict(layer_end, policy.layer_end)) {
        policy.error = llama_k3_decode_expert_policy_error::invalid_layer_range;
        return policy;
    }
    if (policy.layer_start < 0 || policy.layer_end < policy.layer_start || policy.layer_end >= n_layer) {
        policy.error = llama_k3_decode_expert_policy_error::invalid_layer_range;
    }

    return policy;
}

static inline llama_k3_decode_expert_policy llama_k3_decode_expert_policy_from_env(
        int64_t model_top_k, int64_t n_layer) {
    llama_k3_decode_expert_policy policy = llama_k3_decode_expert_policy_parse(
        std::getenv("LLAMA_K3_DECODE_EXPERT_MODE"),
        std::getenv("LLAMA_K3_DECODE_EXPERT_TOP_K"),
        std::getenv("LLAMA_K3_DECODE_EXPERT_MASS"),
        std::getenv("LLAMA_K3_DECODE_EXPERT_MIN_K"),
        std::getenv("LLAMA_K3_DECODE_EXPERT_MAX_K"),
        std::getenv("LLAMA_K3_DECODE_EXPERT_LAYER_START"),
        std::getenv("LLAMA_K3_DECODE_EXPERT_LAYER_END"),
        model_top_k,
        n_layer);
    policy.small_batch = llama_k3_decode_expert_small_batch_enabled(
        std::getenv("LLAMA_K3_DECODE_EXPERT_SMALL_BATCH"));
    return policy;
}

// Launch-only per-layer overrides. These internal values never accept a
// request payload and are frozen with the context's runtime configuration.
struct llama_k3_expert_layer {
    int32_t layer;
    float mass;
    int32_t min_k;
    int32_t max_k; // zero leaves the model routing width uncapped
};

using llama_k3_decode_expert_table = std::vector<llama_k3_expert_layer>;

static inline bool llama_k3_expert_tables_equal(
        const llama_k3_decode_expert_table & a, const llama_k3_decode_expert_table & b) {
    if (a.size() != b.size()) { return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].layer != b[i].layer || a[i].mass != b[i].mass ||
                a[i].min_k != b[i].min_k || a[i].max_k != b[i].max_k) { return false; }
    }
    return true;
}

static inline bool llama_k3_expert_table_canonicalize(
        const llama_k3_expert_layer * rows, size_t count, int64_t model_top_k,
        int64_t n_layer, llama_k3_decode_expert_table & output) {
    if (model_top_k <= 0 || n_layer <= 0 || count > size_t(n_layer) || (!rows && count)) { return false; }
    llama_k3_decode_expert_table table;
    if (count) { table.assign(rows, rows + count); }
    std::sort(table.begin(), table.end(), [](const auto & a, const auto & b) { return a.layer < b.layer; });
    int64_t previous = -1;
    for (const auto & row : table) {
        if (row.layer < 0 || row.layer >= n_layer || row.layer == previous ||
                !std::isfinite(row.mass) || row.mass <= 0 || row.mass > 1 ||
                row.min_k < 1 || row.min_k > model_top_k || row.max_k < 0 || row.max_k > model_top_k ||
                (row.max_k && row.min_k > row.max_k)) { return false; }
        previous = row.layer;
    }
    output = std::move(table);
    return true;
}

// Strict records: LAYER[-END] MASS MIN_K MAX_K, delimited by newline or ';'.
// Blank records and '#' comments through the physical newline are accepted.
// Semicolons inside comments do not start records. Overlapping ranges are errors.
static inline llama_k3_decode_expert_table llama_k3_expert_table_parse(
        std::string text, int64_t model_top_k, int64_t n_layer) {
    if (text.size() > 65536) { throw std::invalid_argument("K3 expert layer table exceeds 64 KiB"); }
    if (text.find('\0') != std::string::npos) { throw std::invalid_argument("NUL byte in K3 expert layer table"); }
    bool comment = false;
    for (char & ch : text) {
        if (ch == '\n') { comment = false; }
        else if (comment || ch == '#') { comment = true; ch = ' '; }
        else if (ch == ';') { ch = '\n'; }
    }
    std::istringstream lines(text);
    llama_k3_decode_expert_table table;
    std::string line;
    size_t line_no = 0;
    while (std::getline(lines, line)) {
        ++line_no;
        std::istringstream fields(line);
        std::string layer, mass, min_k, max_k, extra;
        if (!(fields >> layer)) { continue; }
        auto invalid = [&]() {
            return std::invalid_argument("invalid K3 expert layer table record " + std::to_string(line_no));
        };
        if (!(fields >> mass >> min_k >> max_k) || (fields >> extra)) { throw invalid(); }
        int64_t first = -1, last = -1, minimum = 0, maximum = 0;
        float threshold = 0;
        const auto dash = layer.find('-');
        if (dash == std::string::npos) {
            if (!llama_k3_parse_i64_strict(layer.c_str(), first)) { throw invalid(); }
            last = first;
        } else if (!llama_k3_parse_i64_strict(layer.substr(0, dash).c_str(), first) ||
                   !llama_k3_parse_i64_strict(layer.substr(dash + 1).c_str(), last)) { throw invalid(); }
        if (first < 0 || last < first || last >= n_layer ||
                !llama_k3_parse_f32_strict(mass.c_str(), threshold) || threshold <= 0 || threshold > 1 ||
                !llama_k3_parse_i64_strict(min_k.c_str(), minimum) || minimum < 1 || minimum > model_top_k ||
                !llama_k3_parse_i64_strict(max_k.c_str(), maximum) || maximum < 0 || maximum > model_top_k ||
                (maximum && minimum > maximum) || table.size() + size_t(last - first + 1) > size_t(n_layer)) {
            throw invalid();
        }
        for (int64_t il = first; il <= last; ++il) {
            table.push_back({int32_t(il), threshold, int32_t(minimum), int32_t(maximum)});
        }
    }
    llama_k3_decode_expert_table canonical;
    if (!llama_k3_expert_table_canonicalize(table.data(), table.size(), model_top_k, n_layer, canonical)) {
        throw std::invalid_argument("invalid K3 expert layer table (duplicate layer or model shape)");
    }
    return canonical;
}

static inline llama_k3_decode_expert_table llama_k3_expert_table_from_env(int64_t model_top_k, int64_t n_layer) {
    const char * text = std::getenv("LLAMA_K3_DECODE_EXPERT_LAYER_TABLE");
    const char * file = std::getenv("LLAMA_K3_DECODE_EXPERT_LAYER_TABLE_FILE");
    if (text && text[0] && file && file[0]) {
        throw std::invalid_argument("K3 expert layer table env and file are mutually exclusive");
    }
    if (file && file[0]) {
        std::ifstream input(file, std::ios::binary);
        if (!input) { throw std::invalid_argument("cannot open K3 expert layer table file"); }
        std::string contents(65537, '\0');
        input.read(&contents[0], contents.size());
        contents.resize(size_t(input.gcount()));
        if (input.bad() || contents.size() > 65536) {
            throw std::invalid_argument("cannot read K3 expert layer table file (64 KiB maximum)");
        }
        return llama_k3_expert_table_parse(std::move(contents), model_top_k, n_layer);
    }
    return llama_k3_expert_table_parse(text ? text : "", model_top_k, n_layer);
}

static inline llama_k3_decode_expert_graph_args llama_k3_expert_graph_args(
        const llama_k3_decode_expert_policy & policy, const llama_k3_decode_expert_table & table,
        int64_t n_tokens, int64_t layer) {
    // Unlike global mode, the table itself opts its listed layers into mass
    // routing. Preserve the effective parsed token-count/layer-range snapshot.
    // Legacy launch mode off ignores raw layer-range env and yields the full
    // range; the sparse table rows select layers in that case.
    if (policy.valid() && n_tokens >= 1 && n_tokens <= (policy.small_batch ? 8 : 1) &&
            layer >= policy.layer_start && layer <= policy.layer_end) {
        const auto it = std::lower_bound(table.begin(), table.end(), layer,
            [](const auto & row, int64_t il) { return row.layer < il; });
        if (it != table.end() && it->layer == layer) {
            return {it->max_k, it->max_k > 0 && it->mass == 1.0f ? 0.0f : it->mass, it->min_k};
        }
    }
    return policy.graph_args(n_tokens, layer);
}

static inline size_t llama_k3_decode_expert_prefix_count(
        const float * sorted_normalized_weights,
              size_t   n,
               float   mass,
              size_t   min_k) {
    if (sorted_normalized_weights == nullptr || n == 0) {
        return 0;
    }

    min_k = std::max<size_t>(1, std::min(min_k, n));
    float cumulative = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        cumulative += sorted_normalized_weights[i];
        if (i + 1 >= min_k && cumulative >= mass) {
            return i + 1;
        }
    }
    return n;
}

static inline float llama_deepseek2_yarn_log_mul_for_graph(llm_arch arch, const llama_hparams & hparams) {
    return llama_is_kimi_k2_deepseek2_layout(arch, hparams) ?
        hparams.rope_yarn_log_mul : 0.1f * hparams.rope_yarn_log_mul;
}
