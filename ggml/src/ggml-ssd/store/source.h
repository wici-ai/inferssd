// SPDX-License-Identifier: MIT
#pragma once
#include <cstddef>
#include <cstdint>

namespace ssd { namespace store {
// Borrow the immutable index and its descriptors until all readers have joined.
// The executor keeps its existing physical alignment and owns no source FD.
struct source_view {
    void * userdata = nullptr;
    bool (*resolve)(void *, const char *, int, size_t, int64_t,
                    int *, uint64_t *, size_t *) noexcept = nullptr;
};
} }
