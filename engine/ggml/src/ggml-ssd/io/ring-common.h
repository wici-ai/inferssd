// SPDX-License-Identifier: Apache-2.0
#pragma once
// Internal to the portable read rings (pool_ring, win_ring): linux_ring's
// semantics written once over a small backend interface.
#include "../settings/result.h"
#include "ring-types.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ssd { namespace io { namespace detail {

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

struct staged_read { read_range source; void * data; uint64_t tag; bool fixed; };

// One platform read mechanism. Every read handed to start() yields exactly one
// completion through collect(), including reads that fail before reaching the
// device (for example a bad descriptor), as io_uring reports them.
class ring_backend {
public:
    virtual ~ring_backend() = default;
    virtual const char * name() const noexcept = 0;
    virtual void start(const staged_read & read) = 0;
    // Hands every started read to the device.
    virtual void flush() = 0;
    virtual bool collect(completion & out) = 0;  // never blocks
    virtual void block() = 0;                    // until collect() can succeed
    virtual void register_buffers(const std::vector<buffer_span> & buffers) = 0;
    virtual void unregister_buffers() = 0;
};

// Staging until submit, the pending and completed counters, fixed-buffer
// checks and drain, exactly as linux_ring has them. One owner thread.
class ring_core {
public:
    ring_core() = default;
    ring_core(const ring_core &) = delete;
    ring_core & operator=(const ring_core &) = delete;
    ~ring_core() { close(); }

    void init(std::unique_ptr<ring_backend> backend, unsigned depth, bool fixed) {
        if (!depth) { throw std::runtime_error("read ring depth must be positive"); }
        backend_ = std::move(backend);
        depth_ = depth;
        read_fixed_ = fixed;
    }

    void close() noexcept {
        // As in linux_ring: user buffers are owned by the device until drained.
        if (pending_) {
            std::fprintf(stderr, "[kimi_cpu_sink] undrained I/O at ring destruction; aborting\n");
            std::abort();
        }
        if (!backend_) { return; }
        try {
            unregister_buffers();
        } catch (const std::exception & e) {
            std::fprintf(stderr, "[kimi_cpu_sink_io] cannot unregister before buffer release: %s; aborting\n", e.what());
            std::abort();
        }
        backend_.reset();
    }

    void register_buffers(const std::vector<buffer_span> & buffers) {
        require();
        if (!read_fixed_ || registered_ || pending_ || buffers.empty() || buffers.size() > UINT16_MAX) {
            throw std::runtime_error("invalid CPU sink fixed-buffer registration state");
        }
        fixed_buffers_ = buffers;
        try {
            backend_->register_buffers(fixed_buffers_);
        } catch (...) {
            fixed_buffers_.clear();
            throw;
        }
        registered_ = true;
        ++register_count_;
        size_t bytes = 0;
        for (const auto & buffer : fixed_buffers_) { bytes += buffer.len; }
        std::fprintf(stderr, "[kimi_cpu_sink_io] read_fixed=1 registered=%zu bytes=%zu register_count=%llu unregister_count=%llu\n",
                     fixed_buffers_.size(), bytes, static_cast<unsigned long long>(register_count_),
                     static_cast<unsigned long long>(unregister_count_));
    }

    void unregister_buffers() {
        if (!registered_) { return; }
        if (pending_) { throw std::runtime_error("cannot unregister CPU sink buffers with pending I/O"); }
        backend_->unregister_buffers();
        registered_ = false;
        fixed_buffers_.clear();
        ++unregister_count_;
        std::fprintf(stderr, "[kimi_cpu_sink_io] read_fixed=1 register_count=%llu unregister_count=%llu\n",
                     static_cast<unsigned long long>(register_count_), static_cast<unsigned long long>(unregister_count_));
    }

    void queue(const read_range & source, void * data, uint64_t tag, bool fixed) {
        require();
        if (!data || reinterpret_cast<uintptr_t>(data) % 4096) {
            throw std::runtime_error("CPU sink read target is not 4096-byte aligned");
        }
        if (fixed && (!read_fixed_ || !registered_ || tag >= fixed_buffers_.size() ||
            data != fixed_buffers_[tag].base || source.nbytes > fixed_buffers_[tag].len)) {
            throw std::runtime_error("READ_FIXED buffer is absent or outside its registered range (no fallback)");
        }
        if (source.nbytes > size_t(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("read ring request exceeds the completion byte count");
        }
        if (staged_.size() >= depth_) { throw std::runtime_error("CPU sink SQ unexpectedly full"); }
        // io_uring's CQ holds twice the SQ; the same bound keeps every backend's
        // completion storage finite.
        if (pending_ >= 2 * uint64_t(depth_)) { throw std::runtime_error("read ring completion queue would overflow"); }
        staged_.push_back({source, data, tag, fixed});
        ++pending_;
    }

    void submit(unsigned count) {
        require();
        if (count > staged_.size()) { throw std::runtime_error("read ring submission made no progress"); }
        unsigned started = 0;
        try {
            for (; started < count; ++started) { backend_->start(staged_[started]); }
        } catch (...) {
            staged_.erase(staged_.begin(), staged_.begin() + started);
            throw;
        }
        staged_.erase(staged_.begin(), staged_.begin() + count);
        backend_->flush();
    }

    bool pop(completion & out) {
        require();
        if (!backend_->collect(out)) { return false; }
        if (!pending_) { throw std::runtime_error("CPU sink received an untracked completion"); }
        --pending_;
        ++completed_;
        return true;
    }

    void wait() {
        require();
        if (pending_ == staged_.size()) { throw std::runtime_error("read ring wait without submitted reads"); }
        backend_->block();
    }

    void drain(void * userdata, void (*on_completion)(void *, const completion &)) {
        if (!backend_ || !pending_) { return; }
        // A failed submission may have started only some reads; start the rest.
        submit(unsigned(staged_.size()));
        completion entry;
        while (pending_) {
            while (pop(entry)) { on_completion(userdata, entry); }
            if (pending_) { wait(); }
        }
    }

    uint64_t pending() const noexcept { return pending_; }
    uint64_t completed() const noexcept { return completed_; }
    bool read_fixed() const noexcept { return read_fixed_; }
    bool buffers_registered() const noexcept { return registered_; }
    const char * backend_name() const noexcept { return backend_ ? backend_->name() : "closed"; }

private:
    void require() const {
        if (!backend_) { throw std::runtime_error("ring not initialized"); }
    }

    std::unique_ptr<ring_backend> backend_;
    std::deque<staged_read> staged_;
    std::vector<buffer_span> fixed_buffers_;
    unsigned depth_ = 0;
    uint64_t pending_ = 0;    // queued reads whose completions have not been popped
    uint64_t completed_ = 0;
    uint64_t register_count_ = 0, unregister_count_ = 0;
    bool read_fixed_ = false;
    bool registered_ = false;
};

}}} // namespace ssd::io::detail
