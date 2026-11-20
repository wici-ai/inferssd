// SPDX-License-Identifier: Apache-2.0
#include "runtime.h"
#include "../gpu/provider.h"
#include "../platform/file.h"
#include <algorithm>

namespace ssd { namespace stream {

const ggml_backend_sched_ssd_hooks runtime::hooks_{
    2, sizeof(ggml_backend_sched_ssd_hooks),
    [](void * p, const ggml_tensor * t) { return static_cast<runtime *>(p)->stream_home(t); },
    [](void * p, ggml_backend_t b, const ggml_tensor * const * weights, ggml_tensor * const * copies, int n) {
        return static_cast<runtime *>(p)->stream_place(b, weights, copies, n);
    },
    [](void * p, ggml_backend_t b, int item) { return static_cast<runtime *>(p)->stream_copy(b, item); },
    [](void * p, ggml_backend_t b, int item) { static_cast<runtime *>(p)->stream_release(b, item); },
    [](void * p, int phase) { auto & r = *static_cast<runtime *>(p);
        std::lock_guard<std::mutex> lock(r.mutex_); r.phase_ = phase >= 0 && phase <= 2 ? phase : 0; },
    [](void * p, ggml_backend_sched_ssd_copy_stats * out) { auto & r = *static_cast<runtime *>(p);
        std::lock_guard<std::mutex> lock(r.mutex_); *out = r.stats_; },
    [](void *) {} // The placed copies stay valid across graph replacement; counters survive it.
};

runtime::~runtime() {
    if (ring_) { ggml_backend_buffer_free(ring_); } // freeing device memory waits for the copies into it
}

void runtime::attach(ggml_backend_sched_t scheduler) {
    ggml_backend_sched_set_ssd_runtime(scheduler, &hooks_, this);
}

uint64_t runtime::ring_bytes() const {
    return ring_ ? ggml_backend_buffer_get_size(ring_) : 0;
}

bool runtime::stream_files(const gpu::stream_file * files, size_t n) noexcept {
    if (!services_.gpu || !services_.gpu_transport) {
        for (size_t i = 0; i < n; ++i) { platform::close_fd(files[i].fd); }
        return false;
    }
    return services_.gpu->transport->stream_files(services_.gpu_transport, files, n);
}

void runtime::release_ring() noexcept {
    if (!ring_) { return; }
    if (services_.gpu && services_.gpu_transport) {
        (void) services_.gpu->transport->stream_plan(services_.gpu_transport, nullptr, 0, 0, 0, nullptr);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    items_.clear();
    ggml_backend_buffer_free(ring_); // freeing device memory waits for the work that reads it
    ring_ = nullptr;
}

bool runtime::stream_home(const ggml_tensor * tensor) const noexcept {
    return streaming_ && tensor && services_.gpu && services_.gpu_transport;
}

// The copies in a ring of a few of the largest weights, each at the same place in every graph: a graph's copies follow
// each other, and one that would cross the ring's end starts the ring again. The transport copies each weight into
// its place, as far ahead of its use as the places the uses before it free allow.
bool runtime::stream_place(ggml_backend_t backend, const ggml_tensor * const * weights, ggml_tensor * const * copies,
        int n) noexcept {
    const auto * gpu = services_.gpu;
    if (!gpu || !services_.gpu_transport || n <= 0 || gpu->device_ordinal(backend) != services_.gpu_device) {
        return false;
    }
    try {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        const uint64_t align = ggml_backend_buft_get_alignment(buft);
        std::vector<uint64_t> bytes(static_cast<size_t>(n));
        uint64_t largest = 0, total = 0;
        for (int k = 0; k < n; ++k) {
            const uint64_t size = ggml_backend_buft_get_alloc_size(buft, copies[k]);
            bytes[size_t(k)] = (size + align - 1) / align * align;
            largest = std::max(largest, bytes[size_t(k)]);
            total += bytes[size_t(k)];
        }
        const uint64_t ring = std::min(total, ring_weights * largest);
        void * stream = gpu->stream(backend);
        if (!ring_ || ggml_backend_buffer_get_size(ring_) < ring) {
            // the work queued before may still read the ring and copy into it
            ggml_backend_synchronize(backend);
            if (!gpu->transport->stream_plan(services_.gpu_transport, nullptr, 0, 0, 0, stream)) { return false; }
            items_.clear();
            if (ring_) { ggml_backend_buffer_free(ring_); }
            ring_ = ggml_backend_buft_alloc_buffer(buft, size_t(ring));
            if (!ring_) { return false; }
            // a place is shared by the weights copied there in turn: placing a copy must not write it (a weights
            // buffer zeroes a quantized tensor's pad when it is placed); the backend clears the pad of a quantized
            // matrix in compute memory where its products read it, as for any weight copied in for a split
            ggml_backend_buffer_set_usage(ring_, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
        char * base = static_cast<char *>(ggml_backend_buffer_get_base(ring_));
        std::vector<gpu::stream_item> items(static_cast<size_t>(n));
        uint64_t at = 0;
        for (int k = 0; k < n; ++k) {
            const uint64_t size = bytes[size_t(k)];
            if (at % ring + size > ring) { at = (at / ring + 1) * ring; }
            if (ggml_backend_tensor_alloc(ring_, copies[k], base + at % ring) != GGML_STATUS_SUCCESS) { return false; }
            items[size_t(k)] = {weights[k]->name, weights[k]->data, ggml_nbytes(weights[k]), base + at % ring, at};
            at += size;
        }
        const uint64_t pass = (at + ring - 1) / ring * ring;
        if (!gpu->transport->stream_plan(services_.gpu_transport, items.data(), items.size(), ring, pass, stream)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        items_ = std::move(items);
        return true;
    } catch (...) {
        return false;
    }
}

bool runtime::stream_copy(ggml_backend_t backend, int item) noexcept {
    const auto * gpu = services_.gpu;
    if (!gpu || !services_.gpu_transport || item < 0 || size_t(item) >= items_.size()) { return false; }
    bool ahead = false;
    const int64_t start = ggml_time_us();
    if (!gpu->transport->stream_use(services_.gpu_transport, size_t(item), gpu->stream(backend), &ahead)) {
        return false;
    }
    const uint64_t elapsed = uint64_t(ggml_time_us() - start);
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.dense_tensors;
    stats_.dense_bytes += items_[size_t(item)].bytes;
    stats_.dense_read_us += elapsed;
    stats_.dense_readahead += ahead ? 1 : 0;
    return true;
}

void runtime::stream_release(ggml_backend_t backend, int item) noexcept {
    const auto * gpu = services_.gpu;
    if (!gpu || !services_.gpu_transport || item < 0) { return; }
    // A release that fails leaves the place unreleased: the copy that would overwrite it fails its graph.
    (void) gpu->transport->stream_release(services_.gpu_transport, size_t(item), gpu->stream(backend));
}
}}
