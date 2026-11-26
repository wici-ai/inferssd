// SPDX-License-Identifier: Apache-2.0
#include "ssd/prefill-union-budget.h"
#include <cstdio>
#include <random>
#include <set>
#include <numeric>

static int checks = 0;
static void require(bool ok) { ++checks; if (!ok) { throw std::runtime_error("union budget mismatch"); } }

// Independent reference: enumerate the union with sets, rank by nested sums,
// then compute each token's scaled weighted scalar expert output.
static void check(const std::vector<int32_t> & ids, const std::vector<float> & weights,
        int slots, int tokens, const std::vector<uint8_t> & ready) {
    ssd::cpu::prefill_union_budget policy;
    policy.reserve(ids.size(), ready.size()); policy.ready = ready;
    const bool changed = policy.apply(ids.data(), weights.data(), slots, tokens);
    const auto & actual_ids = changed ? policy.ids : ids;
    const auto & actual_weights = changed ? policy.weights : weights;
    std::set<int> all, retained;
    for (int id : ids) { if (id >= 0) { all.insert(id); } }
    if (tokens <= 1 || all.size() <= 4) { retained = all; }
    else {
        for (int id : all) { if (ready[id]) { retained.insert(id); } }
        std::vector<int> misses;
        for (int id : all) { if (!ready[id]) { misses.push_back(id); } }
        auto mass = [&](int id) { double result = 0; for (size_t r = 0; r < ids.size(); ++r) {
            if (ids[r] == id) { result += weights[r]; }
        } return result; };
        std::stable_sort(misses.begin(), misses.end(), [&](int a, int b) { return mass(a) > mass(b); });
        const size_t n = retained.size() < 4 ? 4 - retained.size() : 0;
        for (size_t i = 0; i < std::min(n, misses.size()); ++i) { retained.insert(misses[i]); }
    }
    require(changed == (retained.size() < all.size()));
    for (int t = 0; t < tokens; ++t) {
        double before = 0, after = 0, reference = 0, actual = 0, scale = 0;
        for (int s = 0; s < slots; ++s) {
            const size_t r = size_t(t) * slots + s;
            if (ids[r] < 0) { continue; }
            before += weights[r];
            if (retained.count(ids[r])) { after += weights[r]; }
        }
        for (int s = 0; s < slots; ++s) {
            const size_t r = size_t(t) * slots + s;
            if (ids[r] < 0) { require(actual_ids[r] == ids[r]); continue; }
            const bool keep = retained.count(ids[r]);
            require(actual_ids[r] == (keep ? ids[r] : -1));
            const double expected = keep ? weights[r] * (after > 0 ? before / after : 1) : 0;
            require(std::abs(actual_weights[r] - expected) <= 1e-5 * (1 + std::abs(expected)));
            reference += expected * (ids[r] + 1.25);
            if (actual_ids[r] >= 0) { actual += actual_weights[r] * (actual_ids[r] + 1.25); scale += actual_weights[r]; }
        }
        require(std::abs(reference - actual) <= 1e-4 * (1 + std::abs(reference)));
        require(std::abs(scale - (after > 0 ? before : 0)) <= 1e-5 * (1 + before));
    }
}
int main() { try {
    check({0,1,2,3,4,5}, {1,1,1,1,1,1}, 1, 6, std::vector<uint8_t>(8)); // ties, empty tokens
    check({0,1,2,3,4,5}, {1,2,3,4,5,6}, 1, 6, std::vector<uint8_t>(8,1)); // all hit
    check({0,1,2,3,4,5}, {1,2,3,4,5,6}, 1, 6, {1,1,1,1,1,0,0,0}); // hits exceed budget
    check({0,0,1,2,3,4}, {2,3,1,1,1,4}, 2, 3, std::vector<uint8_t>(8)); // duplicate aggregation
    check({0,1,2}, {2,3,4}, 1, 3, std::vector<uint8_t>(8)); // union fits
    check({0,1,2,3,4,5}, {1,2,3,4,5,6}, 6, 1, std::vector<uint8_t>(8)); // decode unchanged
    check({}, {}, 0, 4, std::vector<uint8_t>(8));
    std::mt19937 rng(254);
    for (int trial = 0; trial < 1000; ++trial) {
        const int slots = 1 + rng()%16, tokens = 2 + rng()%40;
        std::vector<int32_t> ids(size_t(slots)*tokens);
        std::vector<float> weights(ids.size()); std::vector<uint8_t> ready(32);
        for (auto & hit : ready) { hit = rng()%8 == 0; }
        for (size_t r = 0; r < ids.size(); ++r) { ids[r] = int(rng()%34)-2; weights[r] = float(rng()%100)/37; }
        check(ids, weights, slots, tokens, ready);
    }
    ssd::cpu::prefill_union_budget p; p.reserve(2, 8);
    for (float w : {-1.0f, INFINITY, NAN}) {
        int32_t ids[] = {0,1}; float weights[] = {w,1}; bool rejected = false;
        try { p.apply(ids,weights,1,2); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected);
    }
    std::printf("PREFILL_UNION_BUDGET_PASS checks=%d randomized_batches=1000\n", checks);
    return 0;
} catch (const std::exception & e) { std::fprintf(stderr,"%s\n",e.what()); return 1; } }
