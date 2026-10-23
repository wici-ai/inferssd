// SPDX-License-Identifier: MIT
#pragma once

#include "../../ggml/src/ggml-cpu/ssd/testing.h"
#include "../../ggml/src/ggml-ssd/telemetry/telemetry.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

// Locally owned fixture construction inputs, passed explicitly to a real sink
// or CUDA factory. This helper never reads or mutates the process environment.
// Its field names are private test data, not supported product settings.
struct fixture_inputs {
    std::map<std::string, std::string> values;
    ssd::telemetry::writer * events = nullptr;
    int set(const char * key, const char * value) {
        if (value) { values[key] = value; } else { values.erase(key); }
        return 0;
    }
    int unset(const char * key) { values.erase(key); return 0; }
    const char * get(const char * key) const {
        const auto found = values.find(key);
        return found == values.end() ? nullptr : found->second.c_str();
    }
    // Explicit data for CTest variants; reject typos rather than silently
    // collapsing two registrations onto the same executor policy.
    void argument(const std::string & value) {
        const auto equal = value.find('=');
        if (equal == std::string::npos || !equal) { throw std::runtime_error("fixture argument must be key=value"); }
        const std::string key = value.substr(0, equal);
        static const char * recognized[] = {
            "split_cv", "ready_first", "early_up_gate", "role_major", "nonblocking_down_sched",
            "dynamic_rows", "read_fixed", "poll_us", "dynamic_chunk_rows", "row_weights", "compute_cpus",
            "io_cpus", "decode_participants", "prefetch_depth", "prefetch_fail_every", "prefetch_throw_drain",
            "prefetch_throw_schedule", "cache_policy", "cache_allocation", "cache_keep", "cache_prefill_read",
            "cache_elastic", "cache_partial_reclaim", "cache_reclaimable_file", "cache_reserve_bytes",
            "cache_prefill_slots", "cache_layer_quotas", "vram_budget", "vram_async", "vram_exclusive",
            "vram_deferred", "vram_prefill", "vram_locations", "place_costs", "trace", "timeline", "profile",
            "mrc", "ledger"
        };
        if (std::none_of(std::begin(recognized), std::end(recognized), [&](const char * item) { return key == item; })) {
            throw std::runtime_error("unknown explicit fixture field " + key);
        }
        set(key.c_str(), value.substr(equal + 1).c_str());
    }
    uint64_t integer(const char * key, uint64_t fallback) const {
        const char * text = get(key);
        if (!text || !*text) { return fallback; }
        char * end = nullptr;
        errno = 0;
        const uint64_t value = std::strtoull(text, &end, 10);
        if (errno || end == text || *end || *text == '-') { throw std::runtime_error(std::string("invalid fixture ") + key); }
        return value;
    }
    double real(const char * key, double fallback) const {
        const char * text = get(key);
        if (!text || !*text) { return fallback; }
        char * end = nullptr;
        errno = 0;
        const double value = std::strtod(text, &end);
        if (errno || end == text || *end || !std::isfinite(value)) { throw std::runtime_error(std::string("invalid fixture ") + key); }
        return value;
    }
    bool boolean(const char * key, bool fallback) const {
        const char * text = get(key);
        if (!text || !*text) { return fallback; }
        if (std::string(text) == "0") { return false; }
        if (std::string(text) == "1") { return true; }
        throw std::runtime_error(std::string("invalid fixture boolean ") + key);
    }
    std::vector<int> cpus(const char * key) const {
        std::vector<int> result;
        const char * text = get(key);
        if (!text || !*text) { return result; }
        while (*text) {
            char * end = nullptr;
            errno = 0;
            const long value = std::strtol(text, &end, 10);
            if (errno || text == end || value < 0 || value > INT32_MAX || (*end && *end != ',')) {
                throw std::runtime_error("invalid fixture CPU list");
            }
            result.push_back(int(value));
            if (!*end) { break; }
            text = end + 1;
            if (!*text) { throw std::runtime_error("trailing fixture CPU separator"); }
        }
        return result;
    }
    ssd::telemetry::debug_channels debug() const {
        ssd::telemetry::debug_channels out;
        out.trace = boolean("trace", false) || boolean("timeline", false);
        out.profile = boolean("profile", false);
        out.mrc = boolean("mrc", false);
        out.ledger = boolean("ledger", false);
        return out;
    }
    ssd::gpu::testing::tier_options tier_policy() const {
        ssd::gpu::testing::tier_options out;
#define FIXTURE_TIER_BOOL(field) out.field = boolean("tier." #field, out.field)
#define FIXTURE_TIER_INT(field) out.field = integer("tier." #field, out.field)
        FIXTURE_TIER_BOOL(async_upload); FIXTURE_TIER_BOOL(memory_trace); FIXTURE_TIER_BOOL(keep_across_requests);
        FIXTURE_TIER_BOOL(timing); FIXTURE_TIER_BOOL(elastic); FIXTURE_TIER_BOOL(demand_reclaim);
        FIXTURE_TIER_BOOL(deferred_upload); FIXTURE_TIER_BOOL(mxfp4_batch); FIXTURE_TIER_BOOL(watermark);
        FIXTURE_TIER_BOOL(global); FIXTURE_TIER_BOOL(adaptive_admission); FIXTURE_TIER_BOOL(mxfp4_gpu_hidden);
        FIXTURE_TIER_INT(safety_mib); FIXTURE_TIER_INT(safety_blocks); FIXTURE_TIER_INT(margin_mib);
        FIXTURE_TIER_INT(max_slots); FIXTURE_TIER_INT(fail_malloc_at); FIXTURE_TIER_INT(fail_free_at);
        FIXTURE_TIER_INT(low_mib); FIXTURE_TIER_INT(high_mib);
#undef FIXTURE_TIER_BOOL
#undef FIXTURE_TIER_INT
        out.async_upload = boolean("vram_async", out.async_upload);
        out.deferred_upload = boolean("vram_deferred", out.deferred_upload);
        out.min_count = real("tier.min_count", out.min_count);
        out.half_life = real("tier.half_life", out.half_life);
        if (const char * policy = get("tier.policy")) {
            if (std::string(policy) != "lru" && std::string(policy) != "count_decay2") {
                throw std::runtime_error("invalid explicit fixture tier policy");
            }
            out.count_policy = std::string(policy) == "count_decay2";
        }
        return out;
    }
    ssd::cpu::testing::execution_policy policy() const {
        ssd::cpu::testing::execution_policy out;
#define FIXTURE_BOOL(field) out.field = boolean(#field, out.field)
#define FIXTURE_INT(field) out.field = integer(#field, out.field)
        FIXTURE_BOOL(split_cv); FIXTURE_BOOL(ready_first); FIXTURE_BOOL(early_up_gate); FIXTURE_BOOL(role_major);
        FIXTURE_BOOL(nonblocking_down_sched); FIXTURE_BOOL(dynamic_rows); FIXTURE_BOOL(read_fixed);
        FIXTURE_BOOL(cache_keep); FIXTURE_BOOL(cache_prefill_read); FIXTURE_BOOL(cache_elastic);
        FIXTURE_BOOL(cache_partial_reclaim); FIXTURE_BOOL(cache_reclaimable_file);
        FIXTURE_BOOL(vram_async); FIXTURE_BOOL(vram_exclusive); FIXTURE_BOOL(vram_deferred);
        FIXTURE_BOOL(vram_prefill); FIXTURE_BOOL(vram_locations);
        FIXTURE_INT(poll_us); FIXTURE_INT(dynamic_chunk_rows); FIXTURE_INT(decode_participants);
        FIXTURE_INT(prefetch_depth); FIXTURE_INT(prefetch_fail_every); FIXTURE_INT(prefetch_throw_drain);
        FIXTURE_INT(prefetch_throw_schedule); FIXTURE_INT(cache_reserve_bytes); FIXTURE_INT(cache_prefill_slots);
        FIXTURE_INT(vram_budget);
#undef FIXTURE_BOOL
#undef FIXTURE_INT
        if (const char * value = get("cache_policy")) { out.cache_policy = value; }
        if (const char * value = get("cache_allocation")) { out.cache_allocation = value; }
        out.compute_cpus = cpus("compute_cpus");
        out.io_cpus = cpus("io_cpus");
        out.tier = tier_policy();
        if (const char * costs = get("place_costs")) {
            std::string words(costs);
            std::replace(words.begin(), words.end(), ',', ' ');
            std::istringstream stream(words);
            std::string word;
            while (stream >> word) {
                const size_t equal = word.find('=');
                if (equal == std::string::npos) { throw std::runtime_error("invalid fixture cost term"); }
                fixture_inputs term;
                term.set("value", word.substr(equal + 1).c_str());
                if (!out.place_costs.emplace(word.substr(0, equal), term.real("value", -1)).second) {
                    throw std::runtime_error("duplicate fixture cost term");
                }
            }
        }
        if (const char * text = get("row_weights")) {
            while (*text) {
                char * end = nullptr;
                errno = 0;
                const double value = std::strtod(text, &end);
                if (errno || end == text || !std::isfinite(value) || value <= 0 || (*end && *end != ',')) {
                    throw std::runtime_error("invalid fixture row weight");
                }
                out.row_weights.push_back(value);
                if (!*end) { break; }
                text = end + 1;
                if (!*text) { throw std::runtime_error("trailing fixture row weight separator"); }
            }
        }
        if (const char * text = get("cache_layer_quotas")) {
            const std::string spec(text);
            size_t at = 0;
            if (spec.empty() || spec.size() > 4096) { throw std::runtime_error("invalid fixture layer quotas"); }
            while (at < spec.size()) {
                const size_t colon = spec.find(':', at), end = spec.find(',', at);
                if (colon == std::string::npos || (end != std::string::npos && colon > end)) {
                    throw std::runtime_error("invalid fixture layer quota pair");
                }
                fixture_inputs pair;
                pair.set("id", spec.substr(at, colon - at).c_str());
                pair.set("slots", spec.substr(colon + 1, end == std::string::npos ? end : end - colon - 1).c_str());
                const auto id = pair.integer("id", UINT64_MAX), slots = pair.integer("slots", UINT64_MAX);
                if (id > 1023 || slots > 896 || out.cache_layer_quotas.size() == 256 ||
                        !out.cache_layer_quotas.emplace(int32_t(id), size_t(slots)).second) {
                    throw std::runtime_error("invalid or duplicate fixture layer quota");
                }
                if (end == std::string::npos) { break; }
                at = end + 1;
                if (at == spec.size()) { throw std::runtime_error("trailing fixture layer quota separator"); }
            }
        }
        if (out.cache_elastic && !get("cache_prefill_slots")) { throw std::runtime_error("fixture elasticity needs slots"); }
        if (!out.cache_elastic && get("cache_prefill_slots")) { throw std::runtime_error("fixture slots need elasticity"); }
        return out;
    }
    kimi_cpu_sink * create(int threads, int depth, kimi_cpu_sink_resolver resolver, void * userdata,
            char * error, size_t size) const noexcept {
        try {
            const auto selection = debug();
            const ssd::runtime_services services{ events, &selection };
            return ssd::cpu::testing::create(threads, depth, policy(), &services, resolver, userdata, error, size);
        } catch (const std::exception & e) {
            if (error && size) { std::snprintf(error, size, "%s", e.what()); }
            return nullptr;
        }
    }
    template<class Ledger> void configure(Ledger & ledger, bool unified) const {
        ledger.configure(unified, uint32_t(integer("host_headroom_mib", 512)), uint32_t(integer("device_reserve_mib", 0)));
        ledger.reserve = integer("cache_reserve_bytes", 2ull << 30);
        if (ledger.reserve < (512ull << 20)) { throw std::runtime_error("fixture reserve below safe floor"); }
        ledger.margin = std::max(ledger.margin, ledger.reserve);
        ledger.reclaimable_file = boolean("cache_reclaimable_file", false);
        ledger.explicit_reserve = Ledger::add(ledger.explicit_reserve, integer("host_reserve_mib", 0) << 20);
    }
};

struct scoped_fixture_input {
    fixture_inputs & inputs;
    std::string key, saved;
    bool had;
    scoped_fixture_input(fixture_inputs & input, const char * name, const char * value) : inputs(input), key(name), had(input.get(name)) {
        if (had) { saved = input.get(name); }
        inputs.set(name, value);
    }
    ~scoped_fixture_input() { if (had) { inputs.set(key.c_str(), saved.c_str()); } else { inputs.unset(key.c_str()); } }
};
