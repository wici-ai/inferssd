#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>

namespace ggml_moe_common {

// Measured x4 runs favor chunks of four for five to eight misses. Keep smaller
// batches together and retain chunks of two outside that measured interval.
inline size_t object_io_adaptive_submit_chunk(size_t jobs) {
    return jobs <= 4 ? jobs : (jobs <= 8 ? 4 : 2);
}

// The caller preserves its existing CHUNK parser and uses max() for an absent
// override. Explicit CHUNK=0/positive wins; adaptive scheduling is opt-in.
inline size_t object_io_select_submit_chunk(size_t jobs, size_t configured_chunk, bool adaptive) {
    if (configured_chunk != std::numeric_limits<size_t>::max()) return configured_chunk;
    return adaptive ? object_io_adaptive_submit_chunk(jobs) : 0;
}

// SQPOLL publishes the new SQ tail before returning. Its return count can
// include older chunks still awaiting the polling thread, or be zero if that
// thread has already consumed the queue; it is not a per-chunk progress count.
inline int object_io_submit_chunk_result(int submitted, size_t remaining, bool sqpoll) {
    return sqpoll && submitted >= 0 ? (int)remaining : submitted;
}

// Keep initial reads on distinct slots even when harvesting CQEs reduces the
// in-flight count: completion of a read does not release its H2D source buffer.
// Submit returns the number of SQEs consumed, which may be a positive short
// submission. Drain only harvests ready CQEs; it must not wait or refill slots.
template <typename Prepare, typename Submit, typename Drain>
bool object_io_submit_initial(
        size_t jobs,
        size_t depth,
        size_t chunk,
        Prepare prepare,
        Submit submit,
        Drain drain) {
    if (jobs == 0) return true;
    const size_t initial_jobs = std::min(jobs, depth);
    if (initial_jobs == 0) return false;
    if (chunk == 0) chunk = initial_jobs;

    size_t next = 0;
    while (next < initial_jobs) {
        const size_t count = std::min(chunk, initial_jobs - next);
        for (size_t i = 0; i < count; ++i) {
            if (!prepare(next + i)) return false;
        }
        size_t remaining = count;
        while (remaining > 0) {
            const int submitted = submit(remaining);
            if (submitted <= 0 || (size_t)submitted > remaining) return false;
            remaining -= (size_t)submitted;
        }
        next += count;
        // Leave the final chunk for the existing wait/refill loop. In
        // particular, it must never wait with zero outstanding requests.
        if (next < initial_jobs && !drain()) return false;
    }
    return true;
}

} // namespace ggml_moe_common
