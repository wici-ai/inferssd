// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "settings.h"
#include "json.h"
#include <set>
#include <sstream>

namespace ssd { namespace settings {

// Facts from loaded GGUF metadata and runtime devices, never a path/profile name.
struct model_identity {
    std::string uuid, architecture, file_type, quantization_version, hardware;
};
struct model_options {
    bool engram_prefetch = false;
    bool hc_fused = true;
    std::string matched_rule;

    std::string json(const model_identity & m) const {
        return "{\"uuid\":" + json::quote(m.uuid) + ",\"architecture\":" + json::quote(m.architecture) +
            ",\"file_type\":" + json::quote(m.file_type) + ",\"quantization_version\":" +
            json::quote(m.quantization_version) + ",\"hardware\":" + json::quote(m.hardware) +
            ",\"matched_rule\":" + json::quote(matched_rule) + ",\"engram_prefetch\":" +
            (engram_prefetch ? "true" : "false") + ",\"hc_fused\":" + (hc_fused ? "true}" : "false}");
    }
};

// Rule: id|off/on|uuid|architecture|file_type|quantization_version|hardware|engram|hc
// '-' retains the existing option. No wildcard or family inference is supported.
inline result<model_options> resolve_model(const snapshot & settings, const model_identity & model) noexcept {
    try {
        model_options out;
        if (auto v = settings.integer("model.engram_prefetch")) { out.engram_prefetch = *v != 0; }
        if (auto v = settings.integer("model.hc_fused")) { out.hc_fused = *v != 0; }
        const auto * entry = settings.find("model.optimizations");
        const auto * rules = entry ? std::get_if<std::vector<std::string>>(&entry->data) : nullptr;
        if (!rules) { return {out, status::ok, {}}; }
        if (rules->size() > 64) { return {{}, status::invalid, "at most 64 model optimization rules"}; }
        std::set<std::string> ids;
        for (const auto & rule : *rules) {
            std::vector<std::string> fields;
            std::istringstream stream(rule);
            std::string field;
            while (std::getline(stream, field, '|')) { fields.push_back(field); }
            if (rule.size() > 4096 || fields.size() != 9 || rule.back() == '|' ||
                !ids.insert(fields[0]).second) {
                return {{}, status::invalid, "invalid or duplicate model optimization rule"};
            }
            for (const auto & f : fields) {
                if (f.empty() || f == "*" || f.find_first_of("\r\n") != std::string::npos) {
                    return {{}, status::invalid, "model rule requires exact nonempty fields"};
                }
            }
            if ((fields[1] != "off" && fields[1] != "on") ||
                (fields[7] != "-" && fields[7] != "0" && fields[7] != "1") ||
                (fields[8] != "-" && fields[8] != "0" && fields[8] != "1")) {
                return {{}, status::invalid, "invalid model optimization state/options"};
            }
            if (fields[1] != "on" || fields[2] != model.uuid || fields[3] != model.architecture ||
                fields[4] != model.file_type || fields[5] != model.quantization_version ||
                fields[6] != model.hardware) { continue; }
            if (!out.matched_rule.empty()) { return {{}, status::invalid, "ambiguous model optimization rules"}; }
            out.matched_rule = fields[0];
            if (fields[7] != "-" && settings.find("model.engram_prefetch")->origin != "cli") { out.engram_prefetch = fields[7] == "1"; }
            if (fields[8] != "-" && settings.find("model.hc_fused")->origin != "cli") { out.hc_fused = fields[8] == "1"; }
        }
        return {std::move(out), status::ok, {}};
    } catch (const std::bad_alloc &) {
        return {{}, status::resource, "oom resolving model options"};
    } catch (const std::exception & e) {
        return {{}, status::invalid, e.what()};
    }
}
}} // namespace ssd::settings
