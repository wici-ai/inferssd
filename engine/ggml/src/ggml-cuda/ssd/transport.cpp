// SPDX-License-Identifier: Apache-2.0
#include "ggml-ssd/gpu/transport.h"
#include "ggml-ssd/host/arbiter.h"
#include "ggml-ssd/io/read-ring.h"
#include "ggml-ssd/platform/file.h"
#include "ggml-cuda.h"
#include "gpu-runtime.h"
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
[[noreturn]] void fatal(const char * reason) noexcept {
    std::fprintf(stderr, "[ssd.transport] cannot drain owned work: %s\n", reason); std::abort();
}

// A drive read brings in at most this many bytes of a weight, into one of this many pinned slots: as many reads are
// in flight while the earlier ones cross the bus.
constexpr size_t drive_chunk = size_t(16) << 20;
constexpr size_t drive_slots = 8;

// The streamed weights of a context's graphs: each graph copies its items into their places in a device ring, in the
// order the graph uses them, on a copy stream of their own. A position counts the uses across graphs (a pass of the
// items a graph); its virtual start is its item's start a pass later each graph. A copy of a position overwrites the
// places of the positions before it whose bytes are less than a ring behind its end: it waits for the release of the
// last of them, recorded on the compute stream once the work that reads that place is queued (releases come in
// position order, a split's work reading only its own items). A use waits on the compute stream for its item's copy.
// Copies run as far ahead as the released places allow: through the rest of the graph and into the next one's first
// items, while the compute stream works on the weights resident in the device.
//
// An item whose bytes are in a registered file's mapping (a layer without a pinned copy) is read from the drive
// instead, the page cache bypassed: the drive worker reads its positions in order, in chunks, into pinned slots, and
// copies each chunk into the item's place as it arrives; a use of such a position waits for the worker's copy. A read
// that fails fails the use: the mapping is not read instead.
struct stream_item_state {
    std::string name;
    const void * host = nullptr;
    size_t bytes = 0;
    void * device = nullptr;
    uint64_t start = 0;
    cudaEvent_t ready = nullptr;    // its last copy
    cudaEvent_t released = nullptr; // the last work that reads its place, queued
    int64_t used = -1;              // the position of its last use
    int file = -1;                  // the registered file it is read from, -1 for an item in host memory
    uint64_t file_offset = 0;
    int64_t copied = -1;            // a drive item: the position the worker copied last
};

struct stream_state {
    std::vector<stream_item_state> items;
    uint64_t ring_bytes = 0, pass_bytes = 0;
    cudaStream_t copy = nullptr;
    int64_t used = 0;          // the next position to use
    int64_t issued = 0;        // the next position to copy from host memory
    int64_t released = -1;     // every position up to it released
    std::vector<int64_t> released_at; // each item's last released position
    bool drive = false;        // an item is read from the drive
};

// a model file the context maps, readable around the page cache
struct drive_file {
    const char * base = nullptr;
    uint64_t bytes = 0;
    int fd = -1;
    uint64_t size = 0;
    size_t align = 0;
};

// Pinned slots the drive reads land in, cut from one block at the granule direct reads need, with the read ring that
// fills them. A slot's event records its last copy into the device.
struct drive_stage {
    struct slot { char * host = nullptr; cudaEvent_t done = nullptr; bool copying = false; bool reading = false; };
    void * block = nullptr;
    std::vector<slot> slots;
    size_t slot_bytes = 0;
    std::unique_ptr<ssd::io::read_ring> ring;
    bool failed = false; // could not be had: not asked for again
};

struct transport_owner {
    int device;
    ssd::host::context_client * host;
    std::mutex mutex;
    std::condition_variable changed; // a release, a drive copy done, a new plan, the end
    std::atomic<bool> closed{false};
    stream_state stream;
    std::vector<drive_file> files;
    drive_stage stage;               // the drive worker's alone while it copies
    std::thread worker;
    bool stop = false;
    bool busy = false;               // the drive worker copies a position
    bool failed = false;             // a drive read failed: the plan's uses fail
    int64_t drive_next = 0;          // the next position the drive worker considers
    transport_owner(const ssd::runtime_services & services, int ordinal) : device(ordinal), host(services.host) {}
};

uint64_t align_up(uint64_t value, uint64_t a) { return (value + a - 1) / a * a; }

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

// the slots and their ring, admitted by the host budget and touched inside its reservation
bool stage_allocate(transport_owner & owner, size_t align) {
    drive_stage & st = owner.stage;
    if (st.block) return true;
    if (st.failed || !owner.host) return false;
    const size_t slot_bytes = size_t(align_up(drive_chunk + 2 * align, align));
    const size_t bytes = slot_bytes * drive_slots + align;
    auto ticket = owner.host->take(bytes);
    if (!ticket) { st.failed = true; return false; }
    (void) ggml_cuda_admit_host_alloc(bytes); // unified memory: the reserve first, as for device memory
    if (cudaHostAlloc(&st.block, bytes, cudaHostAllocDefault) != cudaSuccess) {
        st.block = nullptr;
        st.failed = true;
        return false;
    }
    char * first = static_cast<char *>(st.block) + (align - reinterpret_cast<uintptr_t>(st.block) % align) % align;
    st.slots.resize(drive_slots);
    for (size_t i = 0; i < drive_slots; ++i) {
        st.slots[i].host = first + i * slot_bytes;
        if (cudaEventCreateWithFlags(&st.slots[i].done, cudaEventDisableTiming) != cudaSuccess) {
            st.failed = true;
            return false;
        }
    }
    st.ring.reset(new ssd::io::read_ring);
    if (!st.ring->init(unsigned(drive_slots), false)) {
        st.ring.reset();
        st.failed = true;
        return false;
    }
    std::memset(first, 0, slot_bytes * drive_slots);
    st.slot_bytes = slot_bytes;
    ticket.value.commit();
    return true;
}

void stage_release(transport_owner & owner) noexcept {
    drive_stage & st = owner.stage;
    const bool allocated = st.block != nullptr;
    if (st.ring && !st.ring->drain(nullptr, [](void *, const ssd::io::completion &) {})) fatal("drive reads");
    st.ring.reset();
    for (auto & slot : st.slots) {
        if (slot.copying && cudaEventSynchronize(slot.done) != cudaSuccess) fatal("drive copy");
        if (slot.done && cudaEventDestroy(slot.done) != cudaSuccess) fatal("drive stage release");
    }
    st.slots.clear();
    if (st.block && cudaFreeHost(st.block) != cudaSuccess) fatal("drive stage release");
    st.block = nullptr;
    if (allocated && owner.host) (void) owner.host->note_known_memory_change();
}

// One position the drive worker copies: its item's bytes in chunks of at most a slot's payload, read in order of
// position as slots come free and copied into the item's place as each arrives, after the copy stream waits for
// the release of the place it overwrites.
struct drive_job {
    int64_t position = 0;
    size_t index = 0;
    drive_file file;
    uint64_t offset = 0, first = 0, end = 0;
    char * device = nullptr;
    cudaEvent_t wait = nullptr, ready = nullptr;
    size_t n_chunks = 0, issued = 0, done = 0;
};

// Under the lock: the next position the drive worker can copy, skipping the positions in host memory: no more than a
// pass ahead of the uses, its place released; false when there is none yet.
bool drive_due(transport_owner & owner) {
    stream_state & s = owner.stream;
    const int64_t n = int64_t(s.items.size());
    if (!s.drive || owner.failed || n == 0) return false;
    if (owner.drive_next < s.used) owner.drive_next = s.used;
    while (owner.drive_next < s.used + n) {
        if (s.items[size_t(owner.drive_next % n)].file < 0) { ++owner.drive_next; continue; }
        return overwritten(s, owner.drive_next) <= s.released;
    }
    return false;
}


// The drive worker: up to a slot's worth of positions in flight, in order, their chunks read as slots come free so
// that the drive has several reads at once; a position's last chunk copied, its ready event recorded and its use let
// through. A read that fails fails the plan's uses.
void drive_worker(transport_owner & owner) {
    if (cudaSetDevice(owner.device) != cudaSuccess) {
        std::lock_guard<std::mutex> lock(owner.mutex);
        owner.failed = true;
        owner.changed.notify_all();
        return;
    }
    drive_stage & st = owner.stage;
    std::vector<drive_job> jobs;                 // in position order
    std::vector<std::pair<int64_t, size_t>> slot_chunk; // a reading slot's position and chunk
    std::unique_lock<std::mutex> lock(owner.mutex);
    cudaStream_t copy = nullptr;
    for (;;) {
        // positions to start, in order; none while a failure stands (the next plan clears it)
        if (jobs.empty()) {
            owner.changed.wait(lock, [&] { return owner.stop || drive_due(owner); });
        }
        if (owner.stop) return;
        stream_state & s = owner.stream;
        copy = s.copy;
        while (!owner.failed && jobs.size() < st.slots.size() && drive_due(owner)) {
            const int64_t n = int64_t(s.items.size());
            drive_job job;
            job.position = owner.drive_next++;
            job.index = size_t(job.position % n);
            const auto & item = s.items[job.index];
            const int64_t before = overwritten(s, job.position);
            job.wait = before >= 0 ? s.items[size_t(before % n)].released : nullptr;
            job.ready = item.ready;
            job.file = owner.files[size_t(item.file)];
            job.offset = item.file_offset;
            job.first = job.offset / job.file.align * job.file.align;
            job.end = job.offset + item.bytes;
            job.device = static_cast<char *>(item.device);
            const size_t payload = st.slot_bytes - job.file.align;
            job.n_chunks = size_t((job.end - job.first + payload - 1) / payload);
            jobs.push_back(job);
        }
        owner.busy = !jobs.empty();
        if (jobs.empty()) continue;
        lock.unlock();
        if (slot_chunk.size() != st.slots.size()) slot_chunk.assign(st.slots.size(), {-1, 0});
        bool ok = true;
        // the waits of the positions started, in order, before any of their copies
        for (auto & job : jobs) {
            if (job.wait && job.issued == 0 && job.done == 0) {
                if (cudaStreamWaitEvent(copy, job.wait, 0) != cudaSuccess) ok = false;
                job.wait = nullptr;
            }
        }
        // reads into the free slots, the earliest positions first
        unsigned queued = 0;
        for (auto & job : jobs) {
            const size_t payload = st.slot_bytes - job.file.align;
            for (size_t i = 0; ok && i < st.slots.size() && job.issued < job.n_chunks; ++i) {
                auto & slot = st.slots[i];
                if (slot.reading) continue;
                if (slot.copying) {
                    if (cudaEventSynchronize(slot.done) != cudaSuccess) { ok = false; break; }
                    slot.copying = false;
                }
                const uint64_t at = job.first + uint64_t(job.issued) * payload;
                const uint64_t length = align_up(std::min<uint64_t>(at + payload, job.end) - at, job.file.align);
                if (!st.ring->queue({job.file.fd, at, size_t(length)}, slot.host, i, false)) { ok = false; break; }
                slot.reading = true;
                slot_chunk[i] = {job.position, job.issued++};
                ++queued;
            }
        }
        if (ok && queued && !st.ring->submit(queued)) ok = false;
        // one read back: its chunk copied into its place
        bool reading = false;
        for (const auto & slot : st.slots) reading = reading || slot.reading;
        if (ok && reading) {
            ssd::io::completion c{};
            for (;;) {
                const auto popped = st.ring->pop(c);
                if (!popped) { ok = false; break; }
                if (popped.value) break;
                if (!st.ring->wait()) { ok = false; break; }
            }
            const size_t i = size_t(c.user_data);
            drive_job * job = nullptr;
            if (ok && i < st.slots.size()) {
                for (auto & j : jobs) if (j.position == slot_chunk[i].first) job = &j;
            }
            if (ok && job) {
                auto & slot = st.slots[i];
                slot.reading = false;
                const size_t payload = st.slot_bytes - job->file.align;
                const uint64_t at = job->first + uint64_t(slot_chunk[i].second) * payload;
                const uint64_t from = std::max(at, job->offset), to = std::min(at + payload, job->end);
                // a short read is a failure unless the file ends past the bytes the item needs
                if (c.res < 0 || at + uint64_t(c.res) < to ||
                        cudaMemcpyAsync(job->device + (from - job->offset), slot.host + (from - at), size_t(to - from),
                                        cudaMemcpyHostToDevice, copy) != cudaSuccess ||
                        cudaEventRecord(slot.done, copy) != cudaSuccess) {
                    ok = false;
                } else {
                    slot.copying = true;
                    ++job->done;
                }
            } else if (ok) {
                ok = false;
            }
        }
        // the positions whose every chunk was copied: ready
        std::vector<int64_t> finished;
        for (auto & job : jobs) {
            if (ok && job.done == job.n_chunks) {
                if (cudaEventRecord(job.ready, copy) != cudaSuccess) { ok = false; break; }
                finished.push_back(job.position);
            }
        }
        if (!ok) {
            // every read reaped before a slot is reused: none lands later
            if (!st.ring->drain(nullptr, [](void *, const ssd::io::completion &) {})) fatal("drive reads");
            for (auto & slot : st.slots) slot.reading = false;
        }
        lock.lock();
        stream_state & now = owner.stream;
        for (int64_t position : finished) {
            for (size_t k = 0; k < jobs.size(); ++k) {
                if (jobs[k].position != position) continue;
                now.items[jobs[k].index].copied = position;
                jobs.erase(jobs.begin() + long(k));
                break;
            }
        }
        if (!ok) {
            std::fprintf(stderr, "[ssd.transport] a drive read of %zu positions from %lld failed\n", jobs.size(),
                         jobs.empty() ? -1LL : static_cast<long long>(jobs.front().position));
            jobs.clear();
            owner.failed = true;
        }
        owner.busy = !jobs.empty();
        owner.changed.notify_all();
    }
}

// Under the lock: the drive worker copies nothing until the next plan or position (the lock held from here on).
void drive_hold(std::unique_lock<std::mutex> & lock, transport_owner & owner) {
    owner.changed.wait(lock, [&] { return !owner.busy; });
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
    s.drive = false;
    return ok;
}

bool stream_issue(stream_state & s) {
    const int64_t position = s.issued;
    const size_t n = s.items.size();
    auto & item = s.items[size_t(position % int64_t(n))];
    if (item.file >= 0) { // the drive worker's
        ++s.issued;
        return true;
    }
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

// copies every position in host memory whose place is released, no further than a pass ahead of the uses
void stream_ahead(stream_state & s) {
    const int64_t n = int64_t(s.items.size());
    while (s.issued < s.used + n) {
        const bool drive = s.items[size_t(s.issued % n)].file >= 0;
        if (!drive && overwritten(s, s.issued) > s.released) break;
        if (!stream_issue(s)) return;
    }
}

bool stream_plan_impl(transport_owner & owner, const ssd::gpu::stream_item * items, size_t n, uint64_t ring_bytes,
        uint64_t pass_bytes, cudaStream_t stream) {
    std::unique_lock<std::mutex> lock(owner.mutex);
    if (owner.closed || cudaSetDevice(owner.device) != cudaSuccess) return false;
    stream_state & s = owner.stream;
    bool same = n == s.items.size() && ring_bytes == s.ring_bytes && pass_bytes == s.pass_bytes;
    for (size_t i = 0; same && i < n; ++i) {
        const auto & a = s.items[i];
        same = a.name == items[i].name && a.host == items[i].host && a.bytes == items[i].bytes &&
            a.device == items[i].device && a.start == items[i].start;
    }
    if (same) return true;
    drive_hold(lock, owner);
    if (!stream_drop(s)) fatal("stream copies");
    owner.failed = false;
    owner.drive_next = 0;
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
        // bytes in a registered file's mapping, inside the file: read from the drive where its slots can be had
        const char * at = static_cast<const char *>(item.host);
        for (size_t f = 0; f < owner.files.size(); ++f) {
            const auto & file = owner.files[f];
            if (at >= file.base && at + item.bytes <= file.base + file.bytes &&
                    uint64_t(at - file.base) + item.bytes <= file.size && stage_allocate(owner, file.align)) {
                item.file = int(f);
                item.file_offset = uint64_t(at - file.base);
                s.drive = true;
                break;
            }
        }
        if (cudaEventCreateWithFlags(&item.ready, cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreateWithFlags(&item.released, cudaEventDisableTiming) != cudaSuccess) {
            if (!stream_drop(s)) fatal("stream copies");
            return false;
        }
    }
    s.ring_bytes = ring_bytes;
    s.pass_bytes = pass_bytes;
    size_t mapped = 0, drive = 0;
    for (const auto & item : s.items) {
        const char * at = static_cast<const char *>(item.host);
        for (const auto & file : owner.files) {
            if (at >= file.base && at + item.bytes <= file.base + file.bytes) { ++mapped; break; }
        }
        drive += item.file >= 0 ? 1 : 0;
    }
    if (mapped != drive) {
        std::fprintf(stderr, "[ssd.transport] %zu of %zu streamed weights in a mapped file read through the mapping: "
                     "no pinned slots for drive reads\n", mapped - drive, mapped);
    }
    owner.changed.notify_all();
    return true;
}

bool stream_use_impl(transport_owner & owner, size_t index, void * stream_ptr, bool * ahead) {
    std::unique_lock<std::mutex> lock(owner.mutex);
    if (owner.closed || cudaSetDevice(owner.device) != cudaSuccess) return false;
    stream_state & s = owner.stream;
    const int64_t n = int64_t(s.items.size());
    if (index >= s.items.size()) return false;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    // a use out of the plan's order (a graph that stopped early): the copies made ahead are dropped, and the next
    // ones wait for every use queued so far
    if (s.used % n != int64_t(index)) {
        drive_hold(lock, owner);
        if (cudaStreamSynchronize(s.copy) != cudaSuccess) return false;
        cudaEvent_t & barrier = s.items[index].released;
        if (cudaEventRecord(barrier, stream) != cudaSuccess || cudaStreamWaitEvent(s.copy, barrier, 0) != cudaSuccess) {
            return false;
        }
        const int64_t position = (s.used / n + 1) * n + int64_t(index);
        s.used = s.issued = position;
        s.released = position - 1;
        for (auto & at : s.released_at) at = -1;
        owner.drive_next = position;
    }
    const int64_t position = s.used;
    auto & item = s.items[index];
    if (item.file >= 0) {
        if (ahead) *ahead = item.copied == position;
        owner.changed.notify_all();
        owner.changed.wait(lock, [&] { return item.copied == position || owner.failed || owner.closed; });
        if (item.copied != position) return false;
    } else {
        if (ahead) *ahead = s.issued > position;
        while (s.issued <= position) {
            if (!stream_issue(s)) return false;
        }
    }
    if (cudaStreamWaitEvent(stream, item.ready, 0) != cudaSuccess) return false;
    item.used = position;
    s.used = position + 1;
    stream_ahead(s);
    owner.changed.notify_all();
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
    owner.changed.notify_all();
    return true;
}

bool stream_files_impl(transport_owner & owner, const ssd::gpu::stream_file * files, size_t n) {
    std::unique_lock<std::mutex> lock(owner.mutex);
    for (size_t i = 0; i < n; ++i) {
        drive_file file;
        file.base = static_cast<const char *>(files[i].base);
        file.bytes = files[i].bytes;
        file.fd = files[i].fd;
        // a file the drive reads around the page cache, at a granule the slots' payload is a multiple of
        if (file.fd < 0 || !ssd::platform::is_direct(file.fd) || !ssd::platform::file_size(file.fd, file.size) ||
                (file.align = ssd::platform::direct_alignment(file.fd)) == 0 || drive_chunk % file.align != 0) {
            std::fprintf(stderr, "[ssd.transport] file %zu: not read from the drive (fd %d, direct %d, granule %zu)\n",
                         i, file.fd, file.fd >= 0 && ssd::platform::is_direct(file.fd), file.align);
            if (file.fd >= 0) ssd::platform::close_fd(file.fd);
            continue;
        }
        owner.files.push_back(file);
    }
    return true;
}

void quiesce(void * value) noexcept {
    if (!value) return;
    auto & owner = *static_cast<transport_owner *>(value);
    {
        std::lock_guard<std::mutex> lock(owner.mutex);
        if (owner.closed) return;
        owner.closed = true;
        owner.stop = true;
        owner.changed.notify_all();
    }
    if (owner.worker.joinable()) owner.worker.join();
    std::lock_guard<std::mutex> lock(owner.mutex);
    if (cudaSetDevice(owner.device) != cudaSuccess) fatal("device selection");
    if (!stream_drop(owner.stream)) fatal("stream copies");
    stage_release(owner);
    if (owner.stream.copy && cudaStreamDestroy(owner.stream.copy) != cudaSuccess) fatal("stream release");
    owner.stream.copy = nullptr;
    for (auto & file : owner.files) ssd::platform::close_fd(file.fd);
    owner.files.clear();
}
void destroy(void * value) noexcept {
    if (!value) return;
    quiesce(value);
    delete static_cast<transport_owner *>(value);
}
void * create(const ssd::runtime_services * services, int device, char * error, size_t size) noexcept {
    try {
        if (!services || device < 0) throw std::runtime_error("invalid context transport services");
        auto * owner = new transport_owner(*services, device);
        owner->worker = std::thread(drive_worker, std::ref(*owner));
        return owner;
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
bool stream_files(void * p, const ssd::gpu::stream_file * files, size_t n) noexcept {
    if (!p || (n && !files)) {
        for (size_t i = 0; files && i < n; ++i) ssd::platform::close_fd(files[i].fd);
        return false;
    }
    try { return stream_files_impl(*static_cast<transport_owner *>(p), files, n); }
    catch (...) { quiesce(p); return false; }
}
} // namespace
extern "C" const ssd::gpu::transport_v2 * ggml_backend_cuda_ssd_transport_v2() {
    static const ssd::gpu::transport_v2 api{3, sizeof(ssd::gpu::transport_v2), sizeof(ssd::runtime_services),
        create, quiesce, destroy, stream_plan, stream_use, stream_release, stream_files};
    return &api;
}
