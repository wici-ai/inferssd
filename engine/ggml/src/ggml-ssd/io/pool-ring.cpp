// SPDX-License-Identifier: Apache-2.0
#include "pool-ring.h"
#include "ring-common.h"
#include "../platform/file.h"
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#if defined(__APPLE__)
#include <pthread.h>
#endif

namespace ssd { namespace io {
namespace {
// Each reader owns one read at a time; completions return through a locked queue.
class pool_backend final : public detail::ring_backend {
public:
    explicit pool_backend(unsigned depth) {
        const unsigned readers = std::min(depth, 8u);
        workers_.reserve(readers);
        try {
            for (unsigned i = 0; i < readers; ++i) { workers_.emplace_back([this] { run(); }); }
        } catch (...) {
            stop();
            throw;
        }
    }
    ~pool_backend() override { stop(); }

    const char * name() const noexcept override { return "pool"; }
    void start(const detail::staged_read & read) override { batch_.push_back(read); }
    void flush() override {
        if (batch_.empty()) { return; }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            work_.insert(work_.end(), batch_.begin(), batch_.end());
        }
        const size_t started = batch_.size();
        batch_.clear();
        if (started == 1) { work_cv_.notify_one(); } else { work_cv_.notify_all(); }
    }
    bool collect(completion & out) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (done_.empty()) { return false; }
        out = done_.front();
        done_.pop_front();
        return true;
    }
    void block() override {
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [this] { return !done_.empty(); });
    }
    // Readers take plain addresses, so registration is only the core's bounds checks.
    void register_buffers(const std::vector<buffer_span> &) override {}
    void unregister_buffers() override {}

private:
    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        work_cv_.notify_all();
        for (auto & worker : workers_) { if (worker.joinable()) { worker.join(); } }
        workers_.clear();
    }

    static int32_t read_all(const detail::staged_read & read) {
        auto * target = static_cast<unsigned char *>(read.data);
        size_t done = 0;
        while (done < read.source.nbytes) {
            const int64_t got = platform::pread_some(read.source.fd_direct, target + done, read.source.nbytes - done,
                                                     read.source.offset + done);
            if (got < 0) { return int32_t(got); }
            if (!got) { break; } // end of file: report the bytes that exist, as io_uring does
            done += size_t(got);
        }
        return int32_t(done);
    }

    void run() {
#if defined(__APPLE__)
        // Reads gate the compute that waits on them; keep them off the efficiency cores.
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            work_cv_.wait(lock, [this] { return stopping_ || !work_.empty(); });
            if (stopping_) { return; }
            const detail::staged_read read = work_.front();
            work_.pop_front();
            lock.unlock();
            const completion entry{read.tag, read_all(read), 0};
            lock.lock();
            done_.push_back(entry);
            done_cv_.notify_one();
        }
    }

    std::mutex mutex_;
    std::condition_variable work_cv_, done_cv_;
    std::deque<detail::staged_read> work_;
    std::deque<completion> done_;
    std::vector<detail::staged_read> batch_; // owner thread only
    std::vector<std::thread> workers_;
    bool stopping_ = false;
};
} // namespace

struct pool_ring::impl : detail::ring_core {};

pool_ring::pool_ring() noexcept = default;
pool_ring::~pool_ring() = default;
result<void> pool_ring::init(unsigned depth, bool fixed) noexcept {
    return detail::boundary<void>([&] {
        if (state_) { throw std::runtime_error("read ring initialized twice"); }
        if (!depth) { throw std::runtime_error("read ring depth must be positive"); }
        auto state = std::make_unique<impl>();
        state->init(std::make_unique<pool_backend>(depth), depth, fixed);
        state_ = std::move(state);
    });
}
void pool_ring::close_ring() noexcept { if (state_) { state_->close(); } }
result<void> pool_ring::register_buffers(const std::vector<buffer_span> & buffers) noexcept {
    return detail::boundary<void>([&] { if (!state_) { throw std::runtime_error("ring not initialized"); } state_->register_buffers(buffers); });
}
result<void> pool_ring::unregister_buffers() noexcept {
    return detail::boundary<void>([&] { if (state_) { state_->unregister_buffers(); } });
}
result<void> pool_ring::queue(const read_range & source, void * data, uint64_t tag, bool fixed) noexcept {
    return detail::boundary<void>([&] { if (!state_) { throw std::runtime_error("ring not initialized"); } state_->queue(source, data, tag, fixed); });
}
result<void> pool_ring::submit(unsigned count) noexcept {
    return detail::boundary<void>([&] { if (!state_) { throw std::runtime_error("ring not initialized"); } state_->submit(count); });
}
result<bool> pool_ring::pop(completion & entry) noexcept {
    return detail::boundary<bool>([&] { if (!state_) { throw std::runtime_error("ring not initialized"); } return state_->pop(entry); });
}
result<void> pool_ring::wait() noexcept {
    return detail::boundary<void>([&] { if (!state_) { throw std::runtime_error("ring not initialized"); } state_->wait(); });
}
result<void> pool_ring::drain(void * userdata, void (*callback)(void *, const completion &)) noexcept {
    return detail::boundary<void>([&] { if (state_) { state_->drain(userdata, callback); } });
}
uint64_t pool_ring::pending() const noexcept { return state_ ? state_->pending() : 0; }
uint64_t pool_ring::completed() const noexcept { return state_ ? state_->completed() : 0; }
bool pool_ring::read_fixed() const noexcept { return state_ && state_->read_fixed(); }
bool pool_ring::buffers_registered() const noexcept { return state_ && state_->buffers_registered(); }
const char * pool_ring::backend_name() const noexcept { return state_ ? state_->backend_name() : "none"; }
}} // namespace ssd::io
