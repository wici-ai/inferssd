// SPDX-License-Identifier: MIT
#include "object-uring.h"
#include <climits>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <new>
#if !defined(_WIN32)
#include <unistd.h>
#endif
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
#include <liburing.h>
#endif
namespace ssd { namespace io {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
namespace {

struct sqpoll_affinity_request {
    const char *env_name = nullptr;
    const char *spec = nullptr;
    unsigned cpu = 0;
    bool requested = false;
    bool valid = false;
};

static sqpoll_affinity_request resolve_sqpoll_affinity(
        unsigned flags,
        const char *legacy_env) {
    sqpoll_affinity_request request;
    if ((flags & IORING_SETUP_SQPOLL) == 0) return request;

    request.env_name = "explicit";
    request.spec = legacy_env;
    if (!request.spec || !request.spec[0]) return request;

    request.requested = true;
    char *end = nullptr;
    const long cpu = std::strtol(request.spec, &end, 10);
    request.valid = end != request.spec && *end == '\0' && cpu >= 0 &&
        static_cast<unsigned long>(cpu) <= UINT_MAX;
    if (request.valid) request.cpu = static_cast<unsigned>(cpu);
    return request;
}

} // namespace
#endif

object_io_runtime_snapshot object_io_runtime_snapshot_load(
        const object_io_runtime_counters &counters) noexcept {
    object_io_runtime_snapshot snapshot;
    snapshot.batches = counters.batches.load(std::memory_order_relaxed);
    snapshot.reads = counters.reads.load(std::memory_order_relaxed);
    snapshot.bytes = counters.bytes.load(std::memory_order_relaxed);
    snapshot.fallbacks = counters.fallbacks.load(std::memory_order_relaxed);
    snapshot.submit_us = counters.submit_us.load(std::memory_order_relaxed);
    snapshot.wait_us = counters.wait_us.load(std::memory_order_relaxed);
    snapshot.submit_calls = counters.submit_calls.load(std::memory_order_relaxed);
    snapshot.wait_calls = counters.wait_calls.load(std::memory_order_relaxed);
    snapshot.cqes = counters.cqes.load(std::memory_order_relaxed);
    snapshot.inflight_sum = counters.inflight_sum.load(std::memory_order_relaxed);
    snapshot.inflight_samples = counters.inflight_samples.load(std::memory_order_relaxed);
    snapshot.inflight_max = counters.inflight_max.load(std::memory_order_relaxed);
    return snapshot;
}

bool object_pread_all(
        int fd,
        void *dst,
        size_t bytes,
        uint64_t offset,
        size_t max_chunk_bytes) noexcept {
#if !defined(_WIN32)
    if (fd < 0 || (!dst && bytes != 0)) return false;
    char *out = static_cast<char *>(dst);
    size_t done = 0;
    while (done < bytes) {
        size_t chunk = bytes - done;
        if (max_chunk_bytes != 0 && chunk > max_chunk_bytes) {
            chunk = max_chunk_bytes;
        }
        const ssize_t got = ::pread(fd, out + done, chunk, static_cast<off_t>(offset + done));
        if (got <= 0 || static_cast<size_t>(got) > chunk) return false;
        done += static_cast<size_t>(got);
    }
    return true;
#else
    (void)fd;
    (void)dst;
    (void)bytes;
    (void)offset;
    (void)max_chunk_bytes;
    return false;
#endif
}

unsigned object_io_sqpoll_setup_flag() noexcept {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    return IORING_SETUP_SQPOLL;
#else
    return 0;
#endif
}

unsigned object_io_deferred_taskrun_setup_flags() noexcept {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING) && \
    defined(IORING_SETUP_SINGLE_ISSUER) && defined(IORING_SETUP_DEFER_TASKRUN)
    return IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN;
#else
    return 0;
#endif
}

bool object_io_ring_release(object_io_ring &ring) noexcept {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (ring.uring) {
        // Do not free staging buffers under reads that are still in flight.
        if (!object_io_ring_drain(ring)) return false;
        io_uring_queue_exit(ring.uring);
        delete ring.uring;
        ring.uring = nullptr;
        ring.uring_depth = 0;
        ring.uring_flags = 0;
        ring.uring_sq_thread_idle = 0;
        ring.uring_sqpoll_affinity_requested = false;
        ring.uring_sqpoll_affinity_valid = false;
        ring.uring_sqpoll_cpu = 0;
        ring.uring_fixed_file_registered = false;
        ring.uring_fixed_fd = -1;
    }
    ring.uring_unreaped = 0;
#else
    (void)ring;
#endif
    return true;
}

bool object_io_ring_ensure(
        object_io_ring &ring,
        size_t depth,
        unsigned flags,
        const char *sqpoll_cpu_env,
        unsigned sq_thread_idle) noexcept {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (depth == 0 || depth > UINT_MAX) return false;
    const sqpoll_affinity_request affinity =
        resolve_sqpoll_affinity(flags, sqpoll_cpu_env);
    if (ring.uring && ring.uring_depth == depth && ring.uring_flags == flags &&
            ring.uring_sq_thread_idle == sq_thread_idle &&
            ring.uring_sqpoll_affinity_requested == affinity.requested &&
            ring.uring_sqpoll_affinity_valid == affinity.valid &&
            (!affinity.valid || ring.uring_sqpoll_cpu == affinity.cpu)) {
        // A batch that returned on an error path may have left reads in
        // flight. Their CQEs carry user_data that the next batch reuses, so
        // reap them first; recreate the ring if that is impossible.
        if (object_io_ring_drain(ring)) {
            ++ring.iouring_init_reuses;
            return true;
        }
    }

    if (!object_io_ring_release(ring)) return false;
    ++ring.iouring_init_attempts;
    ring.uring = new (std::nothrow) io_uring();
    if (!ring.uring) { ++ring.iouring_init_failures; return false; }
    int rc = 0;
    if (affinity.requested) {
        ++ring.iouring_sqpoll_affinity_attempts;
    }
    if (flags == 0 && sq_thread_idle == 0) {
        rc = io_uring_queue_init((unsigned)depth, ring.uring, flags);
    } else {
        io_uring_params params = {};
        params.flags = flags;
        params.sq_thread_idle = sq_thread_idle;
        if (affinity.requested && !affinity.valid) {
            ++ring.iouring_sqpoll_affinity_failures;
            std::fprintf(stderr,
                    "[moe_object_transport] SQPOLL affinity: env=%s cpu=%s status=invalid\n",
                    affinity.env_name, affinity.spec);
        } else if (affinity.valid) {
            params.flags |= IORING_SETUP_SQ_AFF;
            params.sq_thread_cpu = affinity.cpu;
        }
        rc = io_uring_queue_init_params((unsigned)depth, ring.uring, &params);
        if (affinity.valid) {
            if (rc == 0) {
                ++ring.iouring_sqpoll_affinity_successes;
            } else {
                ++ring.iouring_sqpoll_affinity_failures;
            }
            std::fprintf(stderr,
                    "[moe_object_transport] SQPOLL affinity: env=%s cpu=%u status=%s rc=%d\n",
                    affinity.env_name, params.sq_thread_cpu, rc == 0 ? "ok" : "failed", rc);
        }
    }
    if (rc != 0) {
        delete ring.uring;
        ring.uring = nullptr;
        ++ring.iouring_init_failures;
        return false;
    }
    ring.uring_depth = depth;
    ring.uring_flags = flags;
    ring.uring_sq_thread_idle = sq_thread_idle;
    ring.uring_sqpoll_affinity_requested = affinity.requested;
    ring.uring_sqpoll_affinity_valid = affinity.valid;
    ring.uring_sqpoll_cpu = affinity.cpu;
    return true;
#else
    (void)ring;
    (void)depth;
    (void)flags;
    (void)sqpoll_cpu_env;
    (void)sq_thread_idle;
    return false;
#endif
}

bool object_io_ring_register_file(object_io_ring &ring, int fd) noexcept {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (!ring.uring || fd < 0 || ring.uring_fixed_file_registered) return false;
    if (io_uring_register_files(ring.uring, &fd, 1) != 0) return false;
    ring.uring_fixed_file_registered = true;
    ring.uring_fixed_fd = fd;
    return true;
#else
    (void)ring;
    (void)fd;
    return false;
#endif
}

bool object_io_ring_unregister_files(object_io_ring &ring) noexcept {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (!ring.uring || !ring.uring_fixed_file_registered) return false;
    if (io_uring_unregister_files(ring.uring) != 0) return false;
    ring.uring_fixed_file_registered = false;
    ring.uring_fixed_fd = -1;
    return true;
#else
    (void)ring;
    return false;
#endif
}

bool object_io_ring_prepare_read(
        object_io_ring &ring,
        int fd,
        void *dst,
        size_t bytes,
        uint64_t offset,
        uint64_t user_data,
        bool fixed_file) noexcept {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (!ring.uring || fd < 0 || !dst || bytes == 0 || bytes > UINT_MAX || offset > INT64_MAX) {
        return false;
    }
    if (fixed_file && (!ring.uring_fixed_file_registered || ring.uring_fixed_fd != fd)) {
        return false;
    }
    io_uring_sqe *sqe = io_uring_get_sqe(ring.uring);
    if (!sqe) return false;
    io_uring_prep_read(sqe, fixed_file ? 0 : fd, dst, (unsigned)bytes, (off_t)offset);
    if (fixed_file) sqe->flags |= IOSQE_FIXED_FILE;
    io_uring_sqe_set_data(sqe, reinterpret_cast<void *>(static_cast<uintptr_t>(user_data)));
    ++ring.uring_unreaped;
    return true;
#else
    (void)ring;
    (void)fd;
    (void)dst;
    (void)bytes;
    (void)offset;
    (void)user_data;
    (void)fixed_file;
    return false;
#endif
}

bool object_io_window_prepare_read(
        object_io_ring &ring,
        object_io_window_slot *window,
        size_t window_size,
        size_t window_slot,
        const object_io_window_request &request) noexcept {
    const size_t expected_bytes = request.expected_bytes ? request.expected_bytes : request.bytes;
    if (!window || window_slot >= window_size || window_slot == SIZE_MAX ||
            window[window_slot].active || request.fd < 0 || !request.destination ||
            request.bytes == 0 || request.bytes > INT_MAX || request.offset > INT64_MAX ||
            expected_bytes > request.bytes || request.request_id == SIZE_MAX) {
        return false;
    }
    if (!object_io_ring_prepare_read(
                ring, request.fd, request.destination, request.bytes, request.offset,
                static_cast<uint64_t>(window_slot) + 1, request.fixed_file)) {
        return false;
    }
    window[window_slot] = {request.request_id, expected_bytes, true};
    ++ring.window_prepares;
    return true;
}

int object_io_ring_submit_count(object_io_ring &ring) noexcept {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (!ring.uring) return -1;
    const int submitted = io_uring_submit(ring.uring);
    if (submitted < 0) return submitted;
    ++ring.iouring_submit_calls;
    return submitted;
#else
    (void)ring;
    return -1;
#endif
}

bool object_io_ring_submit(object_io_ring &ring) noexcept {
    return object_io_ring_submit_count(ring) >= 0;
}

#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
static void object_io_ring_consume_completion(
        object_io_ring &ring,
        io_uring_cqe *cqe,
        object_io_completion &completion) {
    completion.user_data = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(io_uring_cqe_get_data(cqe)));
    completion.result = cqe->res;
    io_uring_cqe_seen(ring.uring, cqe);
    if (ring.uring_unreaped > 0) --ring.uring_unreaped;
    ++ring.iouring_cqes;
}
#endif

bool object_io_ring_drain(object_io_ring &ring) noexcept {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (!ring.uring) {
        ring.uring_unreaped = 0;
        return true;
    }
    if (ring.uring_unreaped == 0) return true;
    if (io_uring_sq_ready(ring.uring) > 0) (void)io_uring_submit(ring.uring);
    // Without SQPOLL an SQE still in the SQ after a failed submit has no CQE
    // coming, so never wait for it. With SQPOLL, submit only publishes the SQ
    // tail and the kernel thread consumes it asynchronously: every prepared
    // SQE will complete and must be reaped before its buffer is reused.
    const uint64_t unsubmitted = (ring.uring_flags & IORING_SETUP_SQPOLL) ?
        0 : io_uring_sq_ready(ring.uring);
    while (ring.uring_unreaped > unsubmitted) {
        io_uring_cqe *cqe = nullptr;
        const int rc = io_uring_wait_cqe(ring.uring, &cqe);
        if (rc == -EINTR) continue;
        if (rc != 0 || !cqe) return false;
        io_uring_cqe_seen(ring.uring, cqe);
        --ring.uring_unreaped;
        ++ring.iouring_drained_cqes;
    }
    return unsubmitted == 0;
#else
    (void)ring;
    return true;
#endif
}

bool object_io_ring_wait(object_io_ring &ring, object_io_completion &completion) noexcept {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (!ring.uring) return false;
    ++ring.iouring_wait_calls;
    io_uring_cqe *cqe = nullptr;
    if (io_uring_wait_cqe(ring.uring, &cqe) != 0 || !cqe) return false;
    object_io_ring_consume_completion(ring, cqe, completion);
    return true;
#else
    (void)ring;
    (void)completion;
    return false;
#endif
}

bool object_io_ring_wait_for(
        object_io_ring &ring,
        object_io_completion &completion,
        uint64_t timeout_ms) noexcept {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (!ring.uring) return false;
    ++ring.iouring_wait_calls;
    __kernel_timespec timeout = {};
    timeout.tv_sec = (int64_t)(timeout_ms / 1000);
    timeout.tv_nsec = (int64_t)((timeout_ms % 1000) * 1000000);
    io_uring_cqe *cqe = nullptr;
    if (io_uring_wait_cqe_timeout(ring.uring, &cqe, &timeout) != 0 || !cqe) return false;
    object_io_ring_consume_completion(ring, cqe, completion);
    return true;
#else
    (void)ring;
    (void)completion;
    (void)timeout_ms;
    return false;
#endif
}

object_io_poll_result object_io_ring_peek(
        object_io_ring &ring,
        object_io_completion &completion) noexcept {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    if (!ring.uring) return object_io_poll_result::error;
    io_uring_cqe *cqe = nullptr;
    const int rc = io_uring_peek_cqe(ring.uring, &cqe);
    if (rc == -EAGAIN || (rc == 0 && !cqe)) return object_io_poll_result::empty;
    if (rc != 0 || !cqe) return object_io_poll_result::error;
    object_io_ring_consume_completion(ring, cqe, completion);
    return object_io_poll_result::ready;
#else
    (void)ring;
    (void)completion;
    return object_io_poll_result::error;
#endif
}

object_io_completion_status object_io_completion_validate(
        object_io_ring &ring,
        const object_io_completion &completion,
        bool user_data_valid,
        size_t expected_bytes) noexcept {
    object_io_completion_counters &counters = ring.completion_counters;
    ++counters.validations;
    if (!user_data_valid) {
        ++counters.invalid_user_data;
        return object_io_completion_status::invalid_user_data;
    }
    if (expected_bytes == 0 || expected_bytes > INT_MAX) {
        ++counters.invalid_expected_bytes;
        return object_io_completion_status::invalid_expected_bytes;
    }
    if (completion.result < 0) {
        ++counters.io_errors;
        return object_io_completion_status::io_error;
    }
    const size_t actual_bytes = static_cast<size_t>(completion.result);
    if (actual_bytes < expected_bytes) {
        ++counters.short_reads;
        return object_io_completion_status::short_read;
    }
    if (actual_bytes > expected_bytes) {
        ++counters.oversized_reads;
        return object_io_completion_status::oversized_read;
    }
    ++counters.completed;
    return object_io_completion_status::complete;
}

object_io_completion_status object_io_window_resolve(
        object_io_ring &ring,
        object_io_window_slot *window,
        size_t window_size,
        const object_io_completion &completion,
        object_io_window_completion &resolved) noexcept {
    resolved = {};
    const bool encoded_slot_valid = completion.user_data > 0 &&
        completion.user_data <= static_cast<uint64_t>(window_size);
    const size_t window_slot = encoded_slot_valid ?
        static_cast<size_t>(completion.user_data - 1) : SIZE_MAX;
    const bool active_slot = encoded_slot_valid && window && window[window_slot].active;
    const size_t expected_bytes = active_slot ? window[window_slot].expected_bytes : 0;
    const object_io_completion_status status = object_io_completion_validate(
        ring, completion, active_slot, expected_bytes);
    if (active_slot) {
        ++ring.window_resolves;
        resolved.window_slot = window_slot;
        resolved.request_id = window[window_slot].request_id;
        resolved.result = completion.result;
        window[window_slot] = {};
    }
    return status;
}

void object_io_window_reset(
        object_io_window_slot *window,
        size_t window_size) noexcept {
    if (!window) return;
    for (size_t i = 0; i < window_size; ++i) window[i] = {};
}

}} // namespace ssd::io
