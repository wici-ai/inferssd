// SPDX-License-Identifier: Apache-2.0
#include "ggml-ssd/gpu/transport.h"
#include "ggml-cuda.h"
#include "gpu-runtime.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
[[noreturn]] void fatal(const char * reason) noexcept {
    std::fprintf(stderr, "[ssd.transport] cannot drain owned work: %s\n", reason); std::abort();
}

// The streamed weights of a context's graphs: each graph copies its items into their places in a device ring, in the
// order the graph uses them, on a copy stream of their own. A position counts the uses across graphs (a pass of the
// items a graph); its virtual start is its item's start a pass later each graph. A copy of a position overwrites the
// places of the positions before it whose bytes are less than a ring behind its end: it waits for the release of the
// last of them, recorded on the compute stream once the work that reads that place is queued (releases come in
// position order, a split's work reading only its own items). A use waits on the compute stream for its item's copy.
// Copies run as far ahead as the released places allow: through the rest of the graph and into the next one's first
// items, while the compute stream works on the weights resident in the device.
struct stream_item_state {
    std::string name;
    const void * host = nullptr;
    size_t bytes = 0;
    void * device = nullptr;
    uint64_t start = 0;
    cudaEvent_t ready = nullptr;    // its last copy
    cudaEvent_t released = nullptr; // the last work that reads its place, queued
    int64_t used = -1;              // the position of its last use
};

struct stream_state {
    std::vector<stream_item_state> items;
    uint64_t ring_bytes = 0, pass_bytes = 0;
    cudaStream_t copy = nullptr;
    int64_t used = 0;          // the next position to use
    int64_t issued = 0;        // the next position to copy
    int64_t released = -1;     // every position up to it released
    std::vector<int64_t> released_at; // each item's last released position
};

struct transport_owner {
    int device;
    std::mutex mutex;
    std::atomic<bool> closed{false};
    stream_state stream;
    explicit transport_owner(int ordinal) : device(ordinal) {}
};

uint64_t virtual_start(const stream_state & s, int64_t position) {
    const size_t n = s.items.size();
    return uint64_t(position / int64_t(n)) * s.pass_bytes + s.items[size_t(position % int64_t(n))].start;
}

// the last position whose place the copy of `position` overwrites (its start less than a ring before the copy's end),
// -1 when there is none
int64_t overwritten(const stream_state & s, int64_t position) {
    const auto & item = s.items[size_t(position % int64_t(s.items.size()))];
    const uint64_t end = virtual_start(s, position) + item.bytes;
    if (end <= s.ring_bytes) return -1;
    const uint64_t limit = end - s.ring_bytes;
    for (int64_t r = position - 1; r >= 0; --r) {
        if (virtual_start(s, r) < limit) return r;
    }
    return -1;
}

// finishes every copy and keeps none; the places may still be read by queued work: a later copy waits for that work
// through the release of the positions before it, which the ring's reset leaves to a barrier
bool stream_drop(stream_state & s) noexcept {
    bool ok = true;
    if (s.copy && cudaStreamSynchronize(s.copy) != cudaSuccess) ok = false;
    for (auto & item : s.items) {
        if (item.ready && cudaEventDestroy(item.ready) != cudaSuccess) ok = false;
        if (item.released && cudaEventDestroy(item.released) != cudaSuccess) ok = false;
    }
    s.items.clear();
    s.released_at.clear();
    s.used = s.issued = 0;
    s.released = -1;
    s.ring_bytes = s.pass_bytes = 0;
    return ok;
}

bool stream_issue(stream_state & s) {
    const int64_t position = s.issued;
    const size_t n = s.items.size();
    auto & item = s.items[size_t(position % int64_t(n))];
    const int64_t before = overwritten(s, position);
    if (before > s.released) return false; // the place is still to be read
    if (before >= 0 && cudaStreamWaitEvent(s.copy, s.items[size_t(before % int64_t(n))].released, 0) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpyAsync(item.device, item.host, item.bytes, cudaMemcpyHostToDevice, s.copy) != cudaSuccess ||
            cudaEventRecord(item.ready, s.copy) != cudaSuccess) {
        return false;
    }
    ++s.issued;
    return true;
}

// copies every position whose place is released, no further than a pass ahead of the uses
void stream_ahead(stream_state & s) {
    const int64_t n = int64_t(s.items.size());
    while (s.issued < s.used + n && overwritten(s, s.issued) <= s.released) {
        if (!stream_issue(s)) return;
    }
}

bool stream_plan_impl(transport_owner & owner, const ssd::gpu::stream_item * items, size_t n, uint64_t ring_bytes,
        uint64_t pass_bytes, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(owner.mutex);
    if (owner.closed || cudaSetDevice(owner.device) != cudaSuccess) return false;
    stream_state & s = owner.stream;
    bool same = n == s.items.size() && ring_bytes == s.ring_bytes && pass_bytes == s.pass_bytes;
    for (size_t i = 0; same && i < n; ++i) {
        const auto & a = s.items[i];
        same = a.name == items[i].name && a.host == items[i].host && a.bytes == items[i].bytes &&
            a.device == items[i].device && a.start == items[i].start;
    }
    if (same) return true;
    if (!stream_drop(s)) fatal("stream copies");
    if (n == 0) return true;
    if (!s.copy && cudaStreamCreateWithFlags(&s.copy, cudaStreamNonBlocking) != cudaSuccess) {
        s.copy = nullptr;
        return false;
    }
    // the work queued before the plan may still read the ring: the first copies wait for it
    cudaEvent_t barrier = nullptr;
    if (cudaEventCreateWithFlags(&barrier, cudaEventDisableTiming) != cudaSuccess) return false;
    const bool waited = cudaEventRecord(barrier, stream) == cudaSuccess &&
        cudaStreamWaitEvent(s.copy, barrier, 0) == cudaSuccess;
    if (cudaEventDestroy(barrier) != cudaSuccess || !waited) return false;
    s.items.resize(n);
    s.released_at.assign(n, -1);
    for (size_t i = 0; i < n; ++i) {
        auto & item = s.items[i];
        item.name = items[i].name;
        item.host = items[i].host;
        item.bytes = items[i].bytes;
        item.device = items[i].device;
        item.start = items[i].start;
        if (cudaEventCreateWithFlags(&item.ready, cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreateWithFlags(&item.released, cudaEventDisableTiming) != cudaSuccess) {
            if (!stream_drop(s)) fatal("stream copies");
            return false;
        }
    }
    s.ring_bytes = ring_bytes;
    s.pass_bytes = pass_bytes;
    return true;
}

bool stream_use_impl(transport_owner & owner, size_t index, void * stream_ptr, bool * ahead) {
    std::lock_guard<std::mutex> lock(owner.mutex);
    if (owner.closed || cudaSetDevice(owner.device) != cudaSuccess) return false;
    stream_state & s = owner.stream;
    const int64_t n = int64_t(s.items.size());
    if (index >= s.items.size()) return false;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    // a use out of the plan's order (a graph that stopped early): the copies made ahead are dropped, and the next
    // ones wait for every use queued so far
    if (s.used % n != int64_t(index)) {
        if (cudaStreamSynchronize(s.copy) != cudaSuccess) return false;
        cudaEvent_t & barrier = s.items[index].released;
        if (cudaEventRecord(barrier, stream) != cudaSuccess || cudaStreamWaitEvent(s.copy, barrier, 0) != cudaSuccess) {
            return false;
        }
        const int64_t position = (s.used / n + 1) * n + int64_t(index);
        s.used = s.issued = position;
        s.released = position - 1;
        for (auto & at : s.released_at) at = -1;
    }
    const int64_t position = s.used;
    if (ahead) *ahead = s.issued > position;
    while (s.issued <= position) {
        if (!stream_issue(s)) return false;
    }
    auto & item = s.items[index];
    if (cudaStreamWaitEvent(stream, item.ready, 0) != cudaSuccess) return false;
    item.used = position;
    s.used = position + 1;
    stream_ahead(s);
    return true;
}

bool stream_release_impl(transport_owner & owner, size_t index, void * stream_ptr) {
    std::lock_guard<std::mutex> lock(owner.mutex);
    if (owner.closed || cudaSetDevice(owner.device) != cudaSuccess) return false;
    stream_state & s = owner.stream;
    if (index >= s.items.size()) return false;
    auto & item = s.items[index];
    if (item.used < 0) return true;
    if (cudaEventRecord(item.released, reinterpret_cast<cudaStream_t>(stream_ptr)) != cudaSuccess) return false;
    s.released_at[index] = item.used;
    const int64_t n = int64_t(s.items.size());
    while (s.released + 1 < s.used && s.released_at[size_t((s.released + 1) % n)] == s.released + 1) {
        ++s.released;
    }
    stream_ahead(s);
    return true;
}

void quiesce(void * value) noexcept {
    if (!value) return;
    auto & owner = *static_cast<transport_owner *>(value);
    std::lock_guard<std::mutex> lock(owner.mutex);
    if (owner.closed) return;
    owner.closed = true;
    if (cudaSetDevice(owner.device) != cudaSuccess) fatal("device selection");
    if (!stream_drop(owner.stream)) fatal("stream copies");
    if (owner.stream.copy && cudaStreamDestroy(owner.stream.copy) != cudaSuccess) fatal("stream release");
    owner.stream.copy = nullptr;
}
void destroy(void * value) noexcept {
    if (!value) return;
    quiesce(value);
    delete static_cast<transport_owner *>(value);
}
void * create(const ssd::runtime_services * services, int device, char * error, size_t size) noexcept {
    try {
        if (!services || device < 0) throw std::runtime_error("invalid context transport services");
        return new transport_owner(device);
    } catch (const std::exception & failure) {
        if (error && size) std::snprintf(error, size, "%s", failure.what());
        return nullptr;
    } catch (...) {
        if (error && size) std::snprintf(error, size, "transport construction failed");
        return nullptr;
    }
}
bool stream_plan(void * p, const ssd::gpu::stream_item * items, size_t n, uint64_t ring, uint64_t pass,
        void * stream) noexcept {
    if (!p || (n && (!items || !stream))) return false;
    try {
        return stream_plan_impl(*static_cast<transport_owner *>(p), items, n, ring, pass,
            reinterpret_cast<cudaStream_t>(stream));
    }
    catch (...) { quiesce(p); return false; }
}
bool stream_use(void * p, size_t index, void * stream, bool * ahead) noexcept {
    if (!p || !stream) return false;
    try { return stream_use_impl(*static_cast<transport_owner *>(p), index, stream, ahead); }
    catch (...) { quiesce(p); return false; }
}
bool stream_release(void * p, size_t index, void * stream) noexcept {
    if (!p || !stream) return false;
    try { return stream_release_impl(*static_cast<transport_owner *>(p), index, stream); }
    catch (...) { quiesce(p); return false; }
}
} // namespace
extern "C" const ssd::gpu::transport_v2 * ggml_backend_cuda_ssd_transport_v2() {
    static const ssd::gpu::transport_v2 api{2, sizeof(ssd::gpu::transport_v2), sizeof(ssd::runtime_services),
        create, quiesce, destroy, stream_plan, stream_use, stream_release};
    return &api;
}
