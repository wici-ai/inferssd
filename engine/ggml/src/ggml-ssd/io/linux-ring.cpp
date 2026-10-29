// SPDX-License-Identifier: Apache-2.0
#include "linux-ring.h"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <linux/io_uring.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <type_traits>
#include <unistd.h>

namespace ssd { namespace io {
namespace {
static std::runtime_error sys_error(const char * operation) {
    return std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
}
template<class T, class F> result<T> boundary(F operation) noexcept {
    result<T> value;
    try {
        if constexpr (std::is_void<T>::value) { operation(); }
        else { value.value = operation(); }
    } catch (const std::bad_alloc &) { value.code = status::resource; }
    catch (const std::exception & error) {
        value.code = status::io;
        try { value.error = error.what(); } catch (...) { value.code = status::resource; }
    } catch (...) { value.code = status::io; }
    return value;
}
} // namespace
// Single producer/consumer, owned by the persistent I/O thread. Direct syscalls
// avoid adding a liburing dependency to the CPU backend.
struct linux_ring::impl {
    int fd = -1;
    void * sq = MAP_FAILED;
    void * cq = MAP_FAILED;
    io_uring_sqe * sqes = static_cast<io_uring_sqe *>(MAP_FAILED);
    size_t sq_size = 0, cq_size = 0, sqes_size = 0;
    unsigned * sq_head = nullptr, * sq_tail = nullptr, * sq_mask = nullptr, * sq_array = nullptr;
    unsigned * cq_head = nullptr, * cq_tail = nullptr, * cq_mask = nullptr;
    io_uring_cqe * cqes = nullptr;
    unsigned sq_entries = 0;
    uint64_t pending = 0; // published SQEs whose CQEs have not been consumed
    uint64_t completed = 0;
    bool read_fixed = false;
    bool buffers_registered = false;
    std::vector<iovec> fixed_buffers;
    uint64_t register_count = 0, unregister_count = 0;

    ~impl() { close_ring(); }

    void close_ring() {
        // close(io_uring_fd) starts asynchronous cancellation; it does not make
        // freeing user buffers safe. Every runtime error must first drain them.
        if (pending) {
            std::fprintf(stderr, "[kimi_cpu_sink] undrained I/O at ring destruction; aborting\n");
            std::abort();
        }
        try {
            unregister_buffers();
        } catch (const std::exception & e) {
            std::fprintf(stderr, "[kimi_cpu_sink_io] cannot unregister before buffer release: %s; aborting\n", e.what());
            std::abort();
        }
        if (sqes != MAP_FAILED) { munmap(sqes, sqes_size); }
        if (cq != MAP_FAILED && cq != sq) { munmap(cq, cq_size); }
        if (sq != MAP_FAILED) { munmap(sq, sq_size); }
        if (fd >= 0) { close(fd); }
        fd = -1;
        sq = cq = MAP_FAILED;
        sqes = static_cast<io_uring_sqe *>(MAP_FAILED);
    }

    void init(unsigned depth, bool fixed) {
        read_fixed = fixed;
        io_uring_params p = {};
        fd = static_cast<int>(syscall(__NR_io_uring_setup, depth, &p));
        if (fd < 0) { throw sys_error("io_uring_setup (no fallback)"); }
        sq_size = p.sq_off.array + p.sq_entries * sizeof(unsigned);
        cq_size = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
        if (p.features & IORING_FEAT_SINGLE_MMAP) { sq_size = cq_size = std::max(sq_size, cq_size); }
        sq = mmap(nullptr, sq_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, IORING_OFF_SQ_RING);
        if (sq == MAP_FAILED) { throw sys_error("mmap io_uring SQ"); }
        cq = (p.features & IORING_FEAT_SINGLE_MMAP) ? sq :
            mmap(nullptr, cq_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, IORING_OFF_CQ_RING);
        if (cq == MAP_FAILED) { throw sys_error("mmap io_uring CQ"); }
        sqes_size = p.sq_entries * sizeof(io_uring_sqe);
        sqes = static_cast<io_uring_sqe *>(mmap(nullptr, sqes_size, PROT_READ | PROT_WRITE,
                                              MAP_SHARED, fd, IORING_OFF_SQES));
        if (sqes == MAP_FAILED) { throw sys_error("mmap io_uring SQEs"); }
        auto sq_ptr = [this](unsigned off) { return reinterpret_cast<unsigned *>(static_cast<char *>(sq) + off); };
        auto cq_ptr = [this](unsigned off) { return reinterpret_cast<unsigned *>(static_cast<char *>(cq) + off); };
        sq_head = sq_ptr(p.sq_off.head); sq_tail = sq_ptr(p.sq_off.tail);
        sq_mask = sq_ptr(p.sq_off.ring_mask); sq_array = sq_ptr(p.sq_off.array);
        cq_head = cq_ptr(p.cq_off.head); cq_tail = cq_ptr(p.cq_off.tail); cq_mask = cq_ptr(p.cq_off.ring_mask);
        cqes = reinterpret_cast<io_uring_cqe *>(static_cast<char *>(cq) + p.cq_off.cqes);
        sq_entries = p.sq_entries;
    }

    void register_buffers(const std::vector<buffer_span> & buffers) {
        if (!read_fixed || buffers_registered || pending || buffers.empty() || buffers.size() > UINT16_MAX) {
            throw std::runtime_error("invalid CPU sink fixed-buffer registration state");
        }
        // Retain the indexed ranges for submission checks. Allocate before the
        // syscall so a C++ allocation failure cannot lose a live registration.
        fixed_buffers.clear();
        fixed_buffers.reserve(buffers.size());
        for (const auto & buffer : buffers) { fixed_buffers.push_back({buffer.base, buffer.len}); }
        int result;
        do {
            result = static_cast<int>(syscall(__NR_io_uring_register, fd, IORING_REGISTER_BUFFERS,
                                              fixed_buffers.data(), static_cast<unsigned>(fixed_buffers.size())));
        } while (result < 0 && errno == EINTR);
        if (result < 0) {
            const auto error = sys_error("IORING_REGISTER_BUFFERS (check RLIMIT_MEMLOCK; no fallback)");
            fixed_buffers.clear();
            throw error;
        }
        buffers_registered = true;
        ++register_count;
        size_t bytes = 0;
        for (const auto & buffer : fixed_buffers) { bytes += buffer.iov_len; }
        std::fprintf(stderr, "[kimi_cpu_sink_io] read_fixed=1 registered=%zu bytes=%zu register_count=%llu unregister_count=%llu\n",
                     fixed_buffers.size(), bytes, static_cast<unsigned long long>(register_count),
                     static_cast<unsigned long long>(unregister_count));
    }

    void unregister_buffers() {
        if (!buffers_registered) { return; }
        if (pending) { throw std::runtime_error("cannot unregister CPU sink buffers with pending I/O"); }
        int result;
        do {
            result = static_cast<int>(syscall(__NR_io_uring_register, fd, IORING_UNREGISTER_BUFFERS, nullptr, 0));
        } while (result < 0 && errno == EINTR);
        if (result < 0) { throw sys_error("IORING_UNREGISTER_BUFFERS"); }
        buffers_registered = false;
        fixed_buffers.clear();
        ++unregister_count;
        std::fprintf(stderr, "[kimi_cpu_sink_io] read_fixed=1 register_count=%llu unregister_count=%llu\n",
                     static_cast<unsigned long long>(register_count), static_cast<unsigned long long>(unregister_count));
    }

    void queue(const read_range & source, void * data, uint64_t tag, bool fixed) {
        if (!data || reinterpret_cast<uintptr_t>(data) % 4096) {
            throw std::runtime_error("CPU sink read target is not 4096-byte aligned");
        }
        if (fixed && (!read_fixed || !buffers_registered || tag >= fixed_buffers.size() ||
            data != fixed_buffers[tag].iov_base || source.nbytes > fixed_buffers[tag].iov_len)) {
            throw std::runtime_error("READ_FIXED buffer is absent or outside its registered range (no fallback)");
        }
        const unsigned head = __atomic_load_n(sq_head, __ATOMIC_ACQUIRE);
        const unsigned tail = *sq_tail;
        if (tail - head >= sq_entries) { throw std::runtime_error("CPU sink SQ unexpectedly full"); }
        const unsigned index = tail & *sq_mask;
        io_uring_sqe & entry = sqes[index];
        std::memset(&entry, 0, sizeof(entry));
        entry.opcode = fixed ? IORING_OP_READ_FIXED : IORING_OP_READ;
        if (fixed) { entry.buf_index = static_cast<uint16_t>(tag); }
        entry.fd = source.fd_direct;
        entry.off = source.offset;
        entry.addr = reinterpret_cast<uint64_t>(data);
        entry.len = static_cast<unsigned>(source.nbytes);
        entry.user_data = tag;
        sq_array[index] = index;
        __atomic_store_n(sq_tail, tail + 1, __ATOMIC_RELEASE);
        ++pending;
    }

    void submit(unsigned count) {
        while (count) {
            const int result = static_cast<int>(syscall(__NR_io_uring_enter, fd, count, 0, 0, nullptr, 0));
            if (result < 0 && errno == EINTR) { continue; }
            if (result < 0) { throw sys_error("io_uring_enter submit"); }
            if (!result) { throw std::runtime_error("io_uring submission made no progress"); }
            count -= static_cast<unsigned>(result);
        }
    }

    bool pop(completion & result) {
        const unsigned head = *cq_head;
        if (head == __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE)) { return false; }
        const auto & entry = cqes[head & *cq_mask];
        result = {entry.user_data, entry.res, entry.flags};
        __atomic_store_n(cq_head, head + 1, __ATOMIC_RELEASE);
        if (!pending) { throw std::runtime_error("CPU sink received an untracked completion"); }
        --pending;
        ++completed;
        return true;
    }

    void wait() {
        int result;
        do {
            result = static_cast<int>(syscall(__NR_io_uring_enter, fd, 0, 1, IORING_ENTER_GETEVENTS, nullptr, 0));
        } while (result < 0 && errno == EINTR);
        if (result < 0) { throw sys_error("io_uring_enter wait"); }
    }

    void drain(void * userdata, void (*on_completion)(void *, const completion &)) {
        if (!pending) { return; }
        // An interrupted/failed submission may have consumed only part of the
        // published SQEs. Submit the rest before waiting for their completions.
        const unsigned remaining = *sq_tail - __atomic_load_n(sq_head, __ATOMIC_ACQUIRE);
        submit(remaining);
        completion cqe;
        while (pending) {
            while (pop(cqe)) { on_completion(userdata, cqe); }
            if (pending) { wait(); }
        }
    }
};

linux_ring::linux_ring() noexcept = default;
linux_ring::~linux_ring() = default;
result<void> linux_ring::init(unsigned depth, bool fixed) noexcept {
    return boundary<void>([&] {
        if (state_) { throw std::runtime_error("read ring initialized twice"); }
        state_.reset(new impl);
        state_->init(depth, fixed);
    });
}
void linux_ring::close_ring() noexcept { if (state_) { state_->close_ring(); } }
result<void> linux_ring::register_buffers(const std::vector<buffer_span> & buffers) noexcept {
    return boundary<void>([&] { if (!state_) { throw std::runtime_error("ring not initialized"); } state_->register_buffers(buffers); });
}
result<void> linux_ring::unregister_buffers() noexcept {
    return boundary<void>([&] { if (state_) { state_->unregister_buffers(); } });
}
result<void> linux_ring::queue(const read_range & source, void * data, uint64_t tag, bool fixed) noexcept {
    return boundary<void>([&] { if (!state_) { throw std::runtime_error("ring not initialized"); } state_->queue(source, data, tag, fixed); });
}
result<void> linux_ring::submit(unsigned count) noexcept {
    return boundary<void>([&] { if (!state_) { throw std::runtime_error("ring not initialized"); } state_->submit(count); });
}
result<bool> linux_ring::pop(completion & result) noexcept {
    return boundary<bool>([&] { if (!state_) { throw std::runtime_error("ring not initialized"); } return state_->pop(result); });
}
result<void> linux_ring::wait() noexcept {
    return boundary<void>([&] { if (!state_) { throw std::runtime_error("ring not initialized"); } state_->wait(); });
}
result<void> linux_ring::drain(void * userdata, void (*callback)(void *, const completion &)) noexcept {
    return boundary<void>([&] { if (state_) { state_->drain(userdata, callback); } });
}
uint64_t linux_ring::pending() const noexcept { return state_ ? state_->pending : 0; }
uint64_t linux_ring::completed() const noexcept { return state_ ? state_->completed : 0; }
bool linux_ring::read_fixed() const noexcept { return state_ && state_->read_fixed; }
bool linux_ring::buffers_registered() const noexcept { return state_ && state_->buffers_registered; }
}} // namespace ssd::io
