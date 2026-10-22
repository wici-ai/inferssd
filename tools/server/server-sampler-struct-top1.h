#pragma once

#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

enum server_struct_class : uint8_t {
    SERVER_STRUCT_CONTROL  = 1 << 0,
    SERVER_STRUCT_EOG      = 1 << 1,
    SERVER_STRUCT_BRACKETS = 1 << 2,
    SERVER_STRUCT_QUOTES  = 1 << 3,
    SERVER_STRUCT_NEWLINE = 1 << 4,
    SERVER_STRUCT_CJK     = 1 << 5,
    SERVER_STRUCT_ALL     = (1 << 6) - 1,
};

struct server_struct_token {
    std::string bytes;
    uint8_t classes = 0;
    bool mixed = false;
    bool special = false;
};

// Examine original bytes. Unknown bytes (including incomplete UTF-8) make a
// piece non-pure; a display decoder must never change its classification.
inline server_struct_token server_struct_classify(std::string bytes, bool control, bool eog) {
    server_struct_token result{std::move(bytes), 0, false, control || eog};
    if (result.special) {
        result.classes = (control ? SERVER_STRUCT_CONTROL : 0) | (eog ? SERVER_STRUCT_EOG : 0);
        return result;
    }
    uint8_t found = 0;
    bool other = false;
    for (size_t i = 0; i < result.bytes.size();) {
        const unsigned char c = result.bytes[i];
        if (c == ' ' || c == '\t' || c == '\v' || c == '\f') { ++i; continue; }
        if (c == '\r' || c == '\n') { found |= SERVER_STRUCT_NEWLINE; ++i; continue; }
        if (std::string("{}[]()<>").find(c) != std::string::npos) { found |= SERVER_STRUCT_BRACKETS; ++i; continue; }
        if (std::string("\"'`").find(c) != std::string::npos) { found |= SERVER_STRUCT_QUOTES; ++i; continue; }
        bool matched = false;
        for (const std::string bracket : {"「", "」", "『", "』", "（", "）", "【", "】", "《", "》", "“", "”", "‘", "’"}) {
            if (result.bytes.compare(i, bracket.size(), bracket) == 0) {
                found |= SERVER_STRUCT_CJK;
                i += bracket.size();
                matched = true;
                break;
            }
        }
        if (matched) { continue; }
        // Explicit Unicode spaces; no locale-dependent or lossy conversion.
        for (const std::string space : {"\xc2\xa0", "\xe3\x80\x80"}) {
            if (result.bytes.compare(i, space.size(), space) == 0) { i += space.size(); matched = true; break; }
        }
        if (!matched) { other = true; ++i; }
    }
    result.classes = other ? 0 : found;
    result.mixed = other && found != 0;
    return result;
}

inline std::string server_struct_bytes_hex(const std::string & bytes) {
    static const char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(2 * bytes.size());
    for (unsigned char c : bytes) { result += digits[c >> 4]; result += digits[c & 15]; }
    return result;
}

struct server_sampler_struct_top1_config {
    bool enabled = false;
    bool protocol_tags = false;
    bool force_response = false;
    uint8_t classes = SERVER_STRUCT_ALL;
    double min_top1_p = 0;

    bool active() const { return enabled || protocol_tags || force_response; }

    static bool flag(const char * value, const char * name) {
        if (value && std::string(value) != "0" && std::string(value) != "1") {
            throw std::invalid_argument(std::string(name) + " must be 0 or 1");
        }
        return value && std::string(value) == "1";
    }

    static server_sampler_struct_top1_config parse(const char * enabled, const char * classes, const char * min_p,
                                                   const char * tags = nullptr, const char * force = nullptr) {
        server_sampler_struct_top1_config result;
        result.enabled = flag(enabled, "LLAMA_K3_SAMPLER_STRUCT_TOP1");
        result.protocol_tags = flag(tags, "LLAMA_K3_PROTOCOL_TAGS");
        result.force_response = flag(force, "LLAMA_K3_PROTOCOL_FORCE_RESPONSE");
        if (!result.enabled) { return result; }
        if (classes) {
            result.classes = 0;
            const std::string list(classes);
            size_t start = 0;
            while (start < list.size()) {
                const auto end = list.find(',', start);
                std::string name = list.substr(start, end == std::string::npos ? end : end - start);
                const auto first = name.find_first_not_of(" \t");
                const auto last = name.find_last_not_of(" \t");
                name = first == std::string::npos ? "" : name.substr(first, last - first + 1);
                if      (name == "control")  { result.classes |= SERVER_STRUCT_CONTROL; }
                else if (name == "eog")      { result.classes |= SERVER_STRUCT_EOG; }
                else if (name == "brackets") { result.classes |= SERVER_STRUCT_BRACKETS; }
                else if (name == "quotes")   { result.classes |= SERVER_STRUCT_QUOTES; }
                else if (name == "newline")  { result.classes |= SERVER_STRUCT_NEWLINE; }
                else if (name == "cjk")      { result.classes |= SERVER_STRUCT_CJK; }
                else { throw std::invalid_argument("unknown LLAMA_K3_SAMPLER_STRUCT_CLASSES entry: " + name); }
                if (end == std::string::npos) { break; }
                start = end + 1;
                if (start == list.size()) { throw std::invalid_argument("empty LLAMA_K3_SAMPLER_STRUCT_CLASSES entry"); }
            }
        }
        if (min_p) {
            char * end = nullptr;
            result.min_top1_p = std::strtod(min_p, &end);
            if (end == min_p || *end != '\0' || !std::isfinite(result.min_top1_p) || result.min_top1_p < 0 || result.min_top1_p > 1) {
                throw std::invalid_argument("LLAMA_K3_SAMPLER_STRUCT_MIN_TOP1P must be a finite probability in [0,1]");
            }
        }
        return result;
    }

    static server_sampler_struct_top1_config from_env() {
        return parse(std::getenv("LLAMA_K3_SAMPLER_STRUCT_TOP1"),
                     std::getenv("LLAMA_K3_SAMPLER_STRUCT_CLASSES"),
                     std::getenv("LLAMA_K3_SAMPLER_STRUCT_MIN_TOP1P"),
                     std::getenv("LLAMA_K3_PROTOCOL_TAGS"),
                     std::getenv("LLAMA_K3_PROTOCOL_FORCE_RESPONSE"));
    }
};

struct server_struct_top1_result {
    llama_token token = LLAMA_TOKEN_NULL;
    llama_token top1 = LLAMA_TOKEN_NULL;
    double sampled_p = 0;
    double top1_p = 0;
    bool replaced = false;
};

struct server_sampler_struct_top1 {
    server_sampler_struct_top1_config config;
    std::vector<server_struct_token> tokens;

    bool is_struct(llama_token token, bool label = false) const {
        return token >= 0 && size_t(token) < tokens.size() &&
               ((tokens[token].classes | (label ? SERVER_STRUCT_CONTROL : 0)) & config.classes) != 0;
    }

    // q is the actual distribution left by the ONE completed sample, after
    // all masks and sampling transforms. Never sort it or consume another RNG.
    server_struct_top1_result apply(llama_token sampled, const llama_token_data_array & q,
                                   bool forced = false, bool label = false) const {
        server_struct_top1_result result;
        result.token = sampled;
        double sum = 0;
        for (size_t i = 0; i < q.size; ++i) {
            const auto & t = q.data[i];
            if (t.id < 0 || size_t(t.id) >= tokens.size() || !std::isfinite(t.p) || t.p < 0) {
                throw std::runtime_error("K3 sampler: invalid effective distribution q");
            }
            sum += t.p;
            if (t.id == sampled) { result.sampled_p = t.p; }
            if (t.p > result.top1_p || (t.p == result.top1_p && (result.top1 < 0 || t.id < result.top1))) {
                result.top1 = t.id;
                result.top1_p = t.p;
            }
        }
        if (std::abs(sum - 1) > 1e-4 || result.sampled_p <= 0) {
            throw std::runtime_error("K3 sampler: missing or unnormalized effective distribution q");
        }
        if (config.enabled && !forced && is_struct(sampled, label) && sampled != result.top1 &&
                (config.min_top1_p == 0 || result.top1_p > config.min_top1_p)) {
            result.token = result.top1;
            result.replaced = true;
        }
        return result;
    }
};
