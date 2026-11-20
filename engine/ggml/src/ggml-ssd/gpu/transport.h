// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../runtime-services.h"
#include <cstddef>
#include <cstdint>

namespace ssd { namespace gpu {
// A streamed weight's bytes in host memory, copied into its place in a device ring for each graph. Its place is
// `device`; `start` is where it begins in the ring's virtual order: a graph's items follow each other at increasing
// starts, never across a multiple of the ring's bytes, and the next graph's begin a pass (the pass bytes, a multiple
// of the ring's bytes) later, so an item lands at the same place every graph.
struct stream_item {
    const char * name;
    const void * host;
    size_t bytes;
    void * device;
    uint64_t start;
};

// The context's GPU transport. CUDA pointers are opaque here; L1 owns no CUDA resources. stream_plan: the items of the
// graphs to come, in the order their copies are used, in a ring of ring_bytes whose passes are pass_bytes, after the
// work queued on the stream (no items: the transport finishes its copies and keeps none); a plan equal to the current
// one keeps the copies already made.
// stream_use: the item's bytes are in place before the work queued on the stream next (ahead: whether its copy was
// issued before); its copies follow the plan's order, wrapping into the next graph's. stream_release: the work that
// reads the item's place is queued on the stream.
struct transport_v2 {
    uint32_t version;
    size_t size;
    size_t runtime_services_size;
    void * (*create)(const runtime_services *, int device, char *, size_t) noexcept;
    void (*quiesce)(void *) noexcept;
    void (*destroy)(void *) noexcept;
    bool (*stream_plan)(void *, const stream_item *, size_t, uint64_t ring_bytes, uint64_t pass_bytes,
                        void * stream) noexcept;
    bool (*stream_use)(void *, size_t, void * stream, bool * ahead) noexcept;
    bool (*stream_release)(void *, size_t, void * stream) noexcept;
};
}} // namespace ssd::gpu
