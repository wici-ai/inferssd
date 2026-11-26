// SPDX-License-Identifier: Apache-2.0
#pragma once

// Miss-aware batch-union mechanism adapted from woolcoxm's Colibri PR #254,
// commit dd0d60692d338c01ee03e3854284a3eca8601da6 (Apache-2.0).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ssd { namespace cpu {

// Call-owned routing metadata. No expert payloads, cache training or admission.
// Multi-token prefill only: keep every ready hit, then the highest aggregate
// weight misses within max(0, budget - hits). Ties use expert ID order.
struct prefill_union_budget {
    static constexpr int budget = 4;
    std::vector<int32_t> ids, experts, order;
    std::vector<float> weights;
    std::vector<double> mass;
    std::vector<uint8_t> ready, keep;

    uint64_t growth(size_t routes, size_t count) const {
        uint64_t bytes = 0;
        auto add = [&](const auto & v, size_t n) {
            if (n > v.capacity()) { bytes += n * sizeof(typename std::decay_t<decltype(v)>::value_type); }
        };
        add(ids, routes); add(weights, routes);
        add(experts, count); add(order, count); add(mass, count); add(ready, count); add(keep, count);
        return bytes;
    }
    void reserve(size_t routes, size_t count) {
        ids.reserve(routes); weights.reserve(routes);
        experts.reserve(count); order.reserve(count); mass.reserve(count); ready.reserve(count); keep.reserve(count);
        experts.resize(count); ready.resize(count);
        for (size_t e = 0; e < count; ++e) { experts[e] = int32_t(e); }
    }
    uint64_t bytes() const {
        return (ids.capacity() + experts.capacity() + order.capacity()) * sizeof(int32_t) +
            weights.capacity() * sizeof(float) + mass.capacity() * sizeof(double) + ready.capacity() + keep.capacity();
    }
    // reserve and the ready snapshot precede apply. False keeps caller metadata
    // untouched, including its rounding when the union fits or all experts hit.
    bool apply(const int32_t * input_ids, const float * input_weights, int slots, int tokens) {
        if (tokens <= 1 || !slots) { return false; }
        const size_t routes = size_t(slots) * tokens, count = experts.size();
        mass.assign(count, 0.0); keep.assign(count, 0); order.clear();
        for (size_t r = 0; r < routes; ++r) {
            const int32_t e = input_ids[r];
            if (e < 0) { continue; }
            if (size_t(e) >= count || !std::isfinite(input_weights[r]) || input_weights[r] < 0) {
                throw std::runtime_error("invalid CPU prefill expert ID or weight");
            }
            if (!keep[e]) { keep[e] = 1; order.push_back(e); }
            mass[e] += input_weights[r];
        }
        if (order.size() <= size_t(budget)) { return false; }
        int hits = 0;
        for (int32_t e : order) { keep[e] = ready[e] != 0; hits += keep[e]; }
        const int misses = std::max(0, budget - hits);
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            return mass[a] == mass[b] ? a < b : mass[a] > mass[b];
        });
        int selected = 0;
        bool dropped = false;
        for (int32_t e : order) {
            if (keep[e]) { continue; }
            if (selected < misses) { keep[e] = 1; ++selected; } else { dropped = true; }
        }
        if (!dropped) { return false; }
        ids.assign(input_ids, input_ids + routes); weights.assign(input_weights, input_weights + routes);
        for (int token = 0; token < tokens; ++token) {
            double before = 0, after = 0;
            const size_t first = size_t(token) * slots;
            for (int slot = 0; slot < slots; ++slot) {
                const size_t r = first + slot;
                if (ids[r] < 0) { continue; }
                before += weights[r];
                if (keep[ids[r]]) { after += weights[r]; } else { ids[r] = -1; weights[r] = 0; }
            }
            // Preserve the source's zero-survivor behavior: all skipped routes
            // contribute zero. A nonempty token retains its prior routed scale.
            if (after > 0) {
                for (int slot = 0; slot < slots; ++slot) {
                    const size_t r = first + slot;
                    if (ids[r] >= 0) { weights[r] = float(double(weights[r]) * (before / after)); }
                }
            }
        }
        return true;
    }
};

} } // namespace ssd::cpu
