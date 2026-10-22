#include "kimi-cpu-cache-policy.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <array>

static void policy_error(char * error, size_t size, const char * message) {
    if (error && size) { std::snprintf(error, size, "%s", message); }
}
static int bits(uint32_t mask) {
    int n = 0;
    while (mask) { mask &= mask - 1; ++n; }
    return n;
}
static uint32_t take(const kimi_cache_compute_plan & p, uint32_t mask, int count, bool original = false) {
    std::array<int, 16> order{}; int used = 0;
    for (int i = 0; i < p.native_count; ++i) { if (mask & (1u << i)) { order[used++] = i; } }
    std::sort(order.begin(), order.begin() + used, [&](int a, int b) {
        const float wa = original ? p.original_weights[a] : p.native_weights[a];
        const float wb = original ? p.original_weights[b] : p.native_weights[b];
        return wa != wb ? wa > wb : a < b;
    });
    uint32_t result = 0;
    for (int i = 0; i < std::min(count, used); ++i) { result |= 1u << order[i]; }
    return result;
}
static void validate(const kimi_cache_compute_config & c) {
    if (c.version != 1 || c.mode < 0 || c.mode > 3 || c.bonus_max < 0 || c.bonus_max > 16 ||
            c.min_s < 1 || c.min_s > 16 || c.max_s < 0 || c.max_s > 16 ||
            (c.max_s && c.max_s < c.min_s) || !std::isfinite(c.s_fraction) || c.s_fraction <= 0 || c.s_fraction > 1 ||
            c.deadline_us > 10000000 || !std::isfinite(c.protect_mass) || c.protect_mass <= 0 || c.protect_mass > 1 ||
            c.protect_min < 1 || c.protect_min > 16 || (c.apply_guard != 0 && c.apply_guard != 1) ||
            (c.trace != 0 && c.trace != 1)) {
        throw std::runtime_error("invalid cache-compute policy configuration (version 1)");
    }
}
static double number(const char * suffix, double fallback, bool integer) {
    const std::string name = std::string("GGML_KIMI_CPU_SINK_CACHE_COMPUTE_") + suffix;
    const char * s = std::getenv(name.c_str());
    if (!s) { return fallback; }
    char * end = nullptr; errno = 0;
    const double v = std::strtod(s, &end);
    if (errno || end == s || *end || !std::isfinite(v) || (integer && std::floor(v) != v) || v < 0 || v > 10000000) {
        throw std::runtime_error("invalid " + name);
    }
    return v;
}
extern "C" bool kimi_cache_compute_config_from_env(kimi_cache_compute_config * c, char * error, size_t size) {
    policy_error(error, size, "");
    try {
        if (!c) { throw std::runtime_error("null cache-compute config"); }
        *c = {}; c->version = 1;
        const char * mode = std::getenv("GGML_KIMI_CPU_SINK_CACHE_COMPUTE");
        if (!mode || !std::strcmp(mode, "off") || !std::strcmp(mode, "0")) { c->mode = KIMI_CACHE_COMPUTE_OFF; }
        else if (!std::strcmp(mode, "bonus")) { c->mode = KIMI_CACHE_COMPUTE_BONUS; }
        else if (!std::strcmp(mode, "hybrid")) { c->mode = KIMI_CACHE_COMPUTE_HYBRID; }
        else if (!std::strcmp(mode, "shrink")) { c->mode = KIMI_CACHE_COMPUTE_SHRINK; }
        else { throw std::runtime_error("CACHE_COMPUTE must be off, bonus, hybrid, or shrink"); }
        c->bonus_max = int(number("BONUS_MAX", 1, true));
        c->min_s = int(number("MIN_S", 2, true)); c->max_s = int(number("MAX_S", 0, true));
        c->s_fraction = float(number("S_FRACTION", .75, false));
        c->deadline_us = uint64_t(number("DEADLINE_US", 1000, true));
        c->protect_mass = float(number("PROTECT_MASS", .8, false));
        c->protect_min = int(number("PROTECT_MIN", 2, true));
        c->apply_guard = int(number("APPLY_GUARD", 0, true)); c->trace = int(number("TRACE", 0, true));
        validate(*c); return true;
    } catch (const std::exception & e) { policy_error(error, size, e.what()); return false; }
}
extern "C" bool kimi_cache_compute_plan_init(const kimi_cache_compute_config * config,
        const int32_t * native_ids, const float * native_weights, int native_count,
        const int32_t * original_ids, const float * original_weights, int original_slots,
        uint32_t cached_mask, bool guarded, kimi_cache_compute_plan * p, char * error, size_t size) {
    policy_error(error, size, "");
    try {
        if (!config || !p || !native_ids || !native_weights || !original_ids || !original_weights ||
                native_count < 1 || native_count > 16 || original_slots < 0 || original_slots > 16 ||
                cached_mask >> native_count) { throw std::runtime_error("invalid cache-compute route input"); }
        validate(*config); *p = {}; p->config = *config; p->native_count = native_count; p->guarded = guarded;
        if (guarded && !config->apply_guard) { p->config.mode = KIMI_CACHE_COMPUTE_OFF; }
        for (int i = 0; i < native_count; ++i) {
            if (native_ids[i] < 0 || !std::isfinite(native_weights[i]) || native_weights[i] < 0 ||
                    std::find(native_ids, native_ids + i, native_ids[i]) != native_ids + i) {
                throw std::runtime_error("native top16 requires unique IDs and nonnegative finite weights");
            }
            p->native_ids[i] = native_ids[i]; p->native_weights[i] = native_weights[i];
        }
        for (int i = 0; i < original_slots; ++i) {
            if (original_ids[i] < 0) { continue; }
            if (!std::isfinite(original_weights[i]) || original_weights[i] < 0) { throw std::runtime_error("invalid S weight"); }
            if (original_weights[i] == 0) { continue; }
            const auto found = std::find(native_ids, native_ids + native_count, original_ids[i]);
            if (found == native_ids + native_count) { throw std::runtime_error("S ID outside native top16"); }
            const int n = int(found - native_ids);
            if ((p->original_mask & (1u << n)) || native_weights[n] <= 0) { throw std::runtime_error("duplicate S ID or zero native weight"); }
            p->original_order[p->original_count++] = n; p->original_weights[n] = original_weights[i]; p->original_mask |= 1u << n;
        }
        if (!p->original_count) { throw std::runtime_error("cache-compute requires nonempty positive-weight S"); }
        p->cached_mask = cached_mask; p->base_mask = p->original_mask;
        if (p->config.mode == KIMI_CACHE_COMPUTE_SHRINK) {
            int count = std::max(config->min_s, int(std::ceil(p->original_count * double(config->s_fraction))));
            if (config->max_s) { count = std::min(count, config->max_s); }
            p->base_mask = take(*p, p->original_mask, count);
        }
        p->protected_mask = p->base_mask;
        if (p->config.mode == KIMI_CACHE_COMPUTE_HYBRID) {
            double total = 0, covered = 0;
            for (int i = 0; i < native_count; ++i) { total += p->original_weights[i]; }
            p->protected_mask = 0;
            for (int n = 1; n <= p->original_count; ++n) {
                const uint32_t next = take(*p, p->base_mask, n, true);
                const uint32_t add = next & ~p->protected_mask;
                for (int i = 0; i < native_count; ++i) { if (add & (1u << i)) { covered += p->original_weights[i]; } }
                p->protected_mask = next;
                if (n >= config->protect_min && covered >= double(config->protect_mass) * total) { break; }
            }
        }
        uint32_t pool = cached_mask & ~p->base_mask;
        for (int i = 0; i < native_count; ++i) { if (!native_weights[i]) { pool &= ~(1u << i); } }
        const int reserve = p->config.mode == KIMI_CACHE_COMPUTE_HYBRID ?
            bits(p->base_mask & ~p->protected_mask) + config->bonus_max :
            p->config.mode == KIMI_CACHE_COMPUTE_OFF ? 0 : config->bonus_max;
        pool = take(*p, pool, reserve);
        p->pin_mask = (cached_mask & p->base_mask) | pool;
        p->bonus_mask = take(*p, pool, p->config.mode == KIMI_CACHE_COMPUTE_OFF ? 0 : config->bonus_max);
        p->final_mask = p->base_mask | p->bonus_mask;
        return true;
    } catch (const std::exception & e) { policy_error(error, size, e.what()); return false; }
}
extern "C" void kimi_cache_compute_plan_update(kimi_cache_compute_plan * p, uint32_t ready, uint64_t elapsed_us) {
    if (!p || p->config.mode != KIMI_CACHE_COMPUTE_HYBRID || p->deadline_applied || elapsed_us < p->config.deadline_us) { return; }
    p->deadline_applied = 1;
    p->dropped_mask = p->base_mask & ~p->protected_mask & ~(ready | p->cached_mask);
    const uint32_t pool = p->pin_mask & ~p->base_mask;
    p->replacement_mask = take(*p, pool, bits(p->dropped_mask));
    p->bonus_mask = take(*p, pool & ~p->replacement_mask, p->config.bonus_max);
    p->final_mask = (p->base_mask & ~p->dropped_mask) | p->replacement_mask | p->bonus_mask;
}
