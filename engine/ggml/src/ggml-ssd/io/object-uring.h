// SPDX-License-Identifier: MIT
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#if defined(__linux__) && !defined(GGML_MOE_DISABLE_LIBURING) && __has_include(<liburing.h>)
struct io_uring;
#define GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING 1
#endif
namespace ssd { namespace io {
enum class object_io_completion_status : uint8_t {
    complete,
    invalid_user_data,
    invalid_expected_bytes,
    io_error,
    short_read,
    oversized_read,
};

struct object_io_completion_counters {
    uint64_t validations = 0;
    uint64_t completed = 0;
    uint64_t invalid_user_data = 0;
    uint64_t invalid_expected_bytes = 0;
    uint64_t io_errors = 0;
    uint64_t short_reads = 0;
    uint64_t oversized_reads = 0;
};

struct object_io_ring {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    io_uring *uring = nullptr;
    size_t uring_depth = 0;
    unsigned uring_flags = 0;
    unsigned uring_sq_thread_idle = 0;
    bool uring_sqpoll_affinity_requested = false;
    bool uring_sqpoll_affinity_valid = false;
    unsigned uring_sqpoll_cpu = 0;
    bool uring_fixed_file_registered = false;
    int uring_fixed_fd = -1;
    // SQEs prepared on this ring whose CQE has not been reaped yet. A caller
    // that returns early leaves them behind; their user_data (window slot + 1)
    // is reused by the next batch, so they must be drained before reuse.
    uint64_t uring_unreaped = 0;
#endif
    uint64_t iouring_batches = 0;
    uint64_t iouring_jobs = 0;
    uint64_t iouring_init_attempts = 0;
    uint64_t iouring_init_reuses = 0;
    uint64_t iouring_init_failures = 0;
    uint64_t iouring_sqpoll_affinity_attempts = 0;
    uint64_t iouring_sqpoll_affinity_successes = 0;
    uint64_t iouring_sqpoll_affinity_failures = 0;
    uint64_t iouring_submit_calls = 0;
    uint64_t iouring_wait_calls = 0;
    uint64_t iouring_cqes = 0;
    uint64_t iouring_drained_cqes = 0;
    uint64_t iouring_inflight_sum = 0;
    uint64_t iouring_inflight_samples = 0;
    uint64_t iouring_inflight_max = 0;
    uint64_t window_prepares = 0;
    uint64_t window_resolves = 0;
    uint64_t iouring_batch_hist[6] = {};
    uint64_t granularity_calls = 0;
    uint64_t granularity_total_jobs = 0;
    uint64_t granularity_read_jobs = 0;
    uint64_t granularity_depth_sum = 0;
    uint64_t granularity_slots_sum = 0;
    uint64_t granularity_max_jobs = 0;
    uint64_t granularity_max_read_jobs = 0;
    uint64_t granularity_max_depth = 0;
    object_io_completion_counters completion_counters;
};

struct object_io_completion {
    uint64_t user_data = 0;
    int result = 0;
};

struct object_io_window_request {
    int fd = -1;
    void *destination = nullptr;
    size_t bytes = 0;
    uint64_t offset = 0;
    size_t request_id = SIZE_MAX;
    bool fixed_file = false;
    // Exact completion size to accept; 0 means `bytes`. A block-aligned
    // O_DIRECT read that extends past EOF completes with file_size - offset.
    size_t expected_bytes = 0;
};

struct object_io_window_slot {
    size_t request_id = SIZE_MAX;
    size_t expected_bytes = 0;
    bool active = false;
};

struct object_io_window_completion {
    size_t window_slot = SIZE_MAX;
    size_t request_id = SIZE_MAX;
    int result = 0;
};

struct object_io_runtime_snapshot {
    uint64_t batches = 0;
    uint64_t reads = 0;
    uint64_t bytes = 0;
    uint64_t fallbacks = 0;
    uint64_t submit_us = 0;
    uint64_t wait_us = 0;
    uint64_t submit_calls = 0;
    uint64_t wait_calls = 0;
    uint64_t cqes = 0;
    uint64_t inflight_sum = 0;
    uint64_t inflight_samples = 0;
    uint64_t inflight_max = 0;
};

struct object_io_runtime_counters {
    std::atomic<uint64_t> batches{0};
    std::atomic<uint64_t> reads{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> fallbacks{0};
    std::atomic<uint64_t> submit_us{0};
    std::atomic<uint64_t> wait_us{0};
    std::atomic<uint64_t> submit_calls{0};
    std::atomic<uint64_t> wait_calls{0};
    std::atomic<uint64_t> cqes{0};
    std::atomic<uint64_t> inflight_sum{0};
    std::atomic<uint64_t> inflight_samples{0};
    std::atomic<uint64_t> inflight_max{0};
};

enum class object_io_poll_result {
    ready,
    empty,
    error,
};

[[nodiscard]] bool object_pread_all(
        int fd,
        void *dst,
        size_t bytes,
        uint64_t offset,
        size_t max_chunk_bytes = 0) noexcept;
[[nodiscard]] object_io_runtime_snapshot object_io_runtime_snapshot_load(
        const object_io_runtime_counters &counters) noexcept;
[[nodiscard]] unsigned object_io_sqpoll_setup_flag() noexcept;
[[nodiscard]] unsigned object_io_deferred_taskrun_setup_flags() noexcept;
[[nodiscard]] bool object_io_ring_ensure(
        object_io_ring &ring,
        size_t depth,
        unsigned flags,
        const char *sqpoll_cpu_env,
        unsigned sq_thread_idle = 0) noexcept;
[[nodiscard]] bool object_io_ring_register_file(object_io_ring &ring, int fd) noexcept;
[[nodiscard]] bool object_io_ring_unregister_files(object_io_ring &ring) noexcept;
[[nodiscard]] bool object_io_ring_prepare_read(
        object_io_ring &ring,
        int fd,
        void *dst,
        size_t bytes,
        uint64_t offset,
        uint64_t user_data,
        bool fixed_file = false) noexcept;
[[nodiscard]] bool object_io_window_prepare_read(
        object_io_ring &ring,
        object_io_window_slot *window,
        size_t window_size,
        size_t window_slot,
        const object_io_window_request &request) noexcept;
[[nodiscard]] int object_io_ring_submit_count(object_io_ring &ring) noexcept;
// Submits any prepared SQEs and reaps every outstanding CQE of the ring.
// Returns false if some SQE could not be submitted or a wait failed.
[[nodiscard]] bool object_io_ring_drain(object_io_ring &ring) noexcept;
[[nodiscard]] bool object_io_ring_submit(object_io_ring &ring) noexcept;
[[nodiscard]] bool object_io_ring_wait(object_io_ring &ring, object_io_completion &completion) noexcept;
[[nodiscard]] bool object_io_ring_wait_for(
        object_io_ring &ring,
        object_io_completion &completion,
        uint64_t timeout_ms) noexcept;
[[nodiscard]] object_io_poll_result object_io_ring_peek(
        object_io_ring &ring,
        object_io_completion &completion) noexcept;
[[nodiscard]] object_io_completion_status object_io_completion_validate(
        object_io_ring &ring,
        const object_io_completion &completion,
        bool user_data_valid,
        size_t expected_bytes) noexcept;
[[nodiscard]] object_io_completion_status object_io_window_resolve(
        object_io_ring &ring,
        object_io_window_slot *window,
        size_t window_size,
        const object_io_completion &completion,
        object_io_window_completion &resolved) noexcept;
void object_io_window_reset(
        object_io_window_slot *window,
        size_t window_size) noexcept;
[[nodiscard]] bool object_io_ring_release(object_io_ring &ring) noexcept;

}} // namespace ssd::io
