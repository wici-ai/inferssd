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
private:
    static const ggml_backend_sched_ssd_hooks hooks_;
    const runtime_services & services_;
    std::mutex mutex_;
    ggml_backend_sched_ssd_copy_stats stats_{};
    int phase_ = 0;
    ggml_backend_buffer_t ring_ = nullptr;
    std::vector<gpu::stream_item> items_;
    bool stream_home(const ggml_tensor *) const noexcept;
    bool stream_place(ggml_backend_t, const ggml_tensor * const *, ggml_tensor * const *, int) noexcept;
    bool stream_copy(ggml_backend_t, int) noexcept;
    void stream_release(ggml_backend_t, int) noexcept;
};
}}
