// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ggml-backend.h"
#include "../runtime-services.h"
#include "../gpu/transport.h"
#include <mutex>
#include <vector>

namespace ssd { namespace stream {
// One request/context owner; the scheduler borrows the table and this object. It streams the weights the context
// keeps in host memory to its GPU (ggml_backend_sched_ssd_hooks): it owns the device ring their copies land in, and
// the GPU transport copies them there. No CUDA symbols or mutable process-wide policy live here.
class runtime {
public:
    explicit runtime(const runtime_services & services) : services_(services) {}
    ~runtime();
    runtime(const runtime &) = delete;
    runtime & operator=(const runtime &) = delete;
    void attach(ggml_backend_sched_t scheduler);
    // whether the weights in host memory stream to the GPU (the default) or keep the scheduler's own rule
    void set_streaming(bool streaming) noexcept { streaming_ = streaming; }
    // the device bytes of the ring for streamed weights whose largest has `largest` bytes, at most
    static uint64_t ring_bytes_for(uint64_t largest) { return ring_weights * largest; }
    // the device bytes of the ring now
    uint64_t ring_bytes() const;
    // frees the ring (the graphs split before must not run again): a graph with streamed weights takes a new one
    void release_ring() noexcept;
    // the model files the context maps, with descriptors that read around the page cache (the transport owns them):
    // a streamed weight in a mapping is read from the drive; false when there is no GPU transport (closed then)
    bool stream_files(const gpu::stream_file * files, size_t n) noexcept;
private:
    // The ring holds this many of the largest streamed weights: copies run ahead of their use by about as many
    // weights while the device works on the layers it keeps.
    static constexpr uint64_t ring_weights = 4;
    static const ggml_backend_sched_ssd_hooks hooks_;
    const runtime_services & services_;
    std::mutex mutex_;
    ggml_backend_sched_ssd_copy_stats stats_{};
    int phase_ = 0;
    bool streaming_ = true;
    ggml_backend_buffer_t ring_ = nullptr;
    std::vector<gpu::stream_item> items_;
    bool stream_home(const ggml_tensor *) const noexcept;
    bool stream_place(ggml_backend_t, const ggml_tensor * const *, ggml_tensor * const *, int) noexcept;
    bool stream_copy(ggml_backend_t, int) noexcept;
    void stream_release(ggml_backend_t, int) noexcept;
};
}}
