// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ssd { namespace place {
// Independent implementation of the max-rank resident-fill mechanism described
// in JustVugg/colibri PR #199, commit 47b5f075e266384e6a6445089b638babb5f945de.
// Ranking already includes the model's selection bias and group mask. Residency
// is an advisory ready-only snapshot, never a lease or permission to read bytes.
// Keep K, preserve top-J, prefer residents within M, then fill by original rank.
// Return the number of identities substituted relative to true top-K.
inline int bounded_rank_fill(const int32_t * ranked, const float * scores,
        const uint8_t * ready, int m, int k, int j, bool enabled, int32_t * out) {
    std::copy(ranked, ranked + k, out);
    if (!enabled || j >= k || m <= k) { return 0; }
    j = std::max(0, j);
    int chosen = j;
    auto append = [&](int r) {
        const int32_t id = ranked[r];
        if (!std::isfinite(scores[id])) { return; } // excluded DeepSeek groups
        if (std::find(out, out + chosen, id) == out + chosen) { out[chosen++] = id; }
    };
    for (int r = j; r < m && chosen < k; ++r) { if (ready[r]) { append(r); } }
    for (int r = 0; r < m && chosen < k; ++r) { append(r); }
    if (chosen != k) { std::copy(ranked, ranked + k, out); return 0; }
    int swaps = 0;
    for (int i = 0; i < k; ++i) {
        swaps += std::find(ranked, ranked + k, out[i]) == ranked + k;
    }
    // Avoid changing floating-point accumulation order when identities agree.
    if (!swaps) { std::copy(ranked, ranked + k, out); }
    return swaps;
}
} } // namespace ssd::place
