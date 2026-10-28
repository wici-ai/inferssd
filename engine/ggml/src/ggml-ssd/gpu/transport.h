// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../runtime-services.h"
#include <cstddef>
#include <cstdint>

namespace ssd { namespace gpu {
using seed_offer_fn = int (*)(void *, const char *, const void *, size_t, int64_t,
    const int32_t *, void *, void **) noexcept;
// Existing transport operations with an explicit context owner. CUDA pointers
// are opaque here; L1 owns no CUDA resources. All source descriptors are borrowed.
struct transport_v1 {
    uint32_t version;
    size_t size;
    size_t runtime_services_size;
    void * (*create)(const runtime_services *, int device, char *, size_t) noexcept;
    void (*quiesce)(void *) noexcept;
    void (*destroy)(void *) noexcept;
    bool (*copy_ranges)(void *, const char *, const int32_t *, const int32_t *, size_t,
        int64_t, size_t, size_t, bool, void *, size_t, void *,
        uint64_t *, uint64_t *, uint64_t *, uint64_t *, uint64_t *) noexcept;
    bool (*dense_has)(void *, const char *, size_t, size_t *) noexcept;
    bool (*dense_prepare)(void *, size_t *) noexcept;
    bool (*dense_read_ahead)(void *, const char *, size_t) noexcept;
    bool (*dense_copy)(void *, const char *, void *, size_t, void *, bool *) noexcept;
    bool (*seed_register)(void *, seed_offer_fn, void *) noexcept;
    bool (*seed_unregister)(void *, seed_offer_fn, void *) noexcept;
    bool (*seed_offer)(void *, const char *, const void *, size_t, int64_t,
        const int32_t *, void *) noexcept;
    bool (*seed_fence)(void *, void *) noexcept;
};
}} // namespace ssd::gpu
