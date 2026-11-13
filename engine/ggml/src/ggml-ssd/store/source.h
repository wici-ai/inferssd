// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>

namespace ssd { namespace store {
// Borrow the immutable index and its descriptors until all readers have joined.
// The executor keeps its existing physical alignment and owns no source FD.
struct source_view {
    // Explicit opt-in: full payload equality must be established by the caller.
    // Descriptors belong to source_index and outlive every executor using this view.
    void * userdata = nullptr;
    bool (*resolve)(void *, const char *, int, size_t, int64_t,
                    int *, uint64_t *, size_t *) noexcept = nullptr;
    // Exactly contiguous blocks 0..N-1, equal logical bytes, no extra entries.
    size_t (*dense_blocks)(void *, const char *, size_t, size_t *) noexcept = nullptr;
    const int * replica_fds = nullptr;
    size_t replica_count = 0;
    size_t range_max_experts = 0;
};
} }
