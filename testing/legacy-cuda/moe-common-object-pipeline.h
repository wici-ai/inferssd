#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace ggml_moe_common {

enum class object_pipeline_phase : uint8_t {
    prompt,
    decode,
};

enum class object_pipeline_mode : uint8_t {
    sync_qd1,
    persistent_demand,
    rolling_refill,
};

enum class object_pipeline_config_source : uint8_t {
    model_default,
    legacy,
    common,
};

struct object_pipeline_mode_config {
    object_pipeline_mode mode = object_pipeline_mode::persistent_demand;
    object_pipeline_config_source source = object_pipeline_config_source::model_default;
    bool valid = true;
};

enum class object_pipeline_dispatch_reason : uint8_t {
    invalid,
    idle,
    wait_timeout,
    io_backpressure,
    h2d_backpressure,
    forced,
    batch_ready,
    timeout,
};

struct object_pipeline_policy {
    size_t io_depth = 0;
    size_t initial_batch = 0;
    size_t refill_batch = 0;
    size_t min_flush_batch = 0;
    uint64_t flush_timeout_us = 0;
    size_t max_inflight_h2d = 0;
};

inline constexpr uint64_t object_pipeline_max_flush_timeout_us = UINT64_C(1000000);

enum object_pipeline_policy_override : uint32_t {
    object_pipeline_override_io_depth = 1U << 0,
    object_pipeline_override_initial_batch = 1U << 1,
    object_pipeline_override_refill_batch = 1U << 2,
    object_pipeline_override_min_flush_batch = 1U << 3,
    object_pipeline_override_flush_timeout_us = 1U << 4,
};

struct object_pipeline_policy_config {
    object_pipeline_policy policy;
    uint32_t override_mask = 0;
    uint32_t invalid_mask = 0;
    bool valid = true;
};

struct object_pipeline_queue_state {
    size_t queued = 0;
    size_t inflight_io = 0;
    size_t inflight_h2d = 0;
    uint64_t oldest_wait_us = 0;
    bool force = false;
};

struct object_pipeline_dispatch_plan {
    object_pipeline_dispatch_reason reason = object_pipeline_dispatch_reason::invalid;
    size_t submit_count = 0;
    size_t io_capacity = 0;
    size_t batch_limit = 0;
    uint64_t wait_remaining_us = 0;
};

struct object_pipeline_ticket {
    uint64_t ticket_id = 0;
    uint64_t generation = 0;
    size_t requested = 0;
    size_t queued = 0;
    size_t submitted = 0;
    size_t inflight = 0;
    size_t completed = 0;
    size_t failed = 0;
    size_t cancelled = 0;
    size_t critical_ready = 0;
};

inline bool object_pipeline_ticket_conserved(const object_pipeline_ticket &ticket) {
    if (ticket.ticket_id == 0 || ticket.generation == 0 || ticket.requested == 0 ||
            ticket.queued > ticket.requested ||
            ticket.submitted > ticket.requested - ticket.queued ||
            ticket.cancelled != ticket.requested - ticket.queued - ticket.submitted ||
            ticket.inflight > ticket.submitted ||
            ticket.completed > ticket.submitted - ticket.inflight ||
            ticket.failed != ticket.submitted - ticket.inflight - ticket.completed ||
            ticket.critical_ready > ticket.completed) {
        return false;
    }
    return true;
}

inline bool object_pipeline_ticket_reset(
        object_pipeline_ticket &ticket,
        uint64_t ticket_id,
        uint64_t generation,
        size_t requested) {
    ticket = {};
    ticket.ticket_id = ticket_id;
    ticket.generation = generation;
    ticket.requested = requested;
    ticket.queued = requested;
    return object_pipeline_ticket_conserved(ticket);
}

inline bool object_pipeline_ticket_record_submit(
        object_pipeline_ticket &ticket,
        size_t count) {
    if (!object_pipeline_ticket_conserved(ticket) || count == 0 ||
            count > ticket.queued) {
        return false;
    }
    ticket.queued -= count;
    ticket.submitted += count;
    ticket.inflight += count;
    return object_pipeline_ticket_conserved(ticket);
}

inline bool object_pipeline_ticket_record_completion(
        object_pipeline_ticket &ticket,
        size_t count = 1) {
    if (!object_pipeline_ticket_conserved(ticket) || count == 0 ||
            count > ticket.inflight) {
        return false;
    }
    ticket.inflight -= count;
    ticket.completed += count;
    return object_pipeline_ticket_conserved(ticket);
}

inline bool object_pipeline_ticket_record_failure(
        object_pipeline_ticket &ticket,
        size_t count = 1) {
    if (!object_pipeline_ticket_conserved(ticket) || count == 0 ||
            count > ticket.inflight) {
        return false;
    }
    ticket.inflight -= count;
    ticket.failed += count;
    return object_pipeline_ticket_conserved(ticket);
}

inline bool object_pipeline_ticket_record_cancellation(
        object_pipeline_ticket &ticket,
        size_t count = 1) {
    if (!object_pipeline_ticket_conserved(ticket) || count == 0 ||
            count > ticket.queued) {
        return false;
    }
    ticket.queued -= count;
    ticket.cancelled += count;
    return object_pipeline_ticket_conserved(ticket);
}

inline bool object_pipeline_ticket_record_critical_ready(
        object_pipeline_ticket &ticket,
        size_t count = 1) {
    if (!object_pipeline_ticket_conserved(ticket) || count == 0 ||
            count > ticket.completed - ticket.critical_ready) {
        return false;
    }
    ticket.critical_ready += count;
    return object_pipeline_ticket_conserved(ticket);
}

inline bool object_pipeline_ticket_terminal(const object_pipeline_ticket &ticket) {
    return object_pipeline_ticket_conserved(ticket) && ticket.queued == 0 &&
        ticket.inflight == 0;
}

inline object_pipeline_policy object_pipeline_default_policy(object_pipeline_phase phase) {
    object_pipeline_policy policy;
    policy.io_depth = 16;
    policy.max_inflight_h2d = 8;
    if (phase == object_pipeline_phase::decode) {
        policy.initial_batch = 8;
        policy.refill_batch = 4;
        policy.min_flush_batch = 2;
        policy.flush_timeout_us = 40;
    } else {
        policy.initial_batch = 16;
        policy.refill_batch = 8;
        policy.min_flush_batch = 8;
        policy.flush_timeout_us = 150;
    }
    return policy;
}

inline bool object_pipeline_policy_valid(const object_pipeline_policy &policy) {
    return policy.io_depth != 0 && policy.initial_batch != 0 &&
        policy.refill_batch != 0 && policy.min_flush_batch != 0 &&
        policy.flush_timeout_us != 0 && policy.max_inflight_h2d != 0 &&
        policy.flush_timeout_us <= object_pipeline_max_flush_timeout_us &&
        policy.initial_batch <= policy.io_depth &&
        policy.refill_batch <= policy.io_depth &&
        policy.min_flush_batch <= policy.io_depth;
}

inline object_pipeline_dispatch_plan object_pipeline_plan_dispatch(
        const object_pipeline_policy &policy,
        const object_pipeline_queue_state &state) {
    object_pipeline_dispatch_plan plan;
    if (!object_pipeline_policy_valid(policy) ||
            state.inflight_io > policy.io_depth ||
            state.inflight_h2d > policy.max_inflight_h2d) {
        return plan;
    }
    if (state.queued == 0) {
        plan.reason = object_pipeline_dispatch_reason::idle;
        return plan;
    }
    if (state.inflight_h2d >= policy.max_inflight_h2d) {
        plan.reason = object_pipeline_dispatch_reason::h2d_backpressure;
        return plan;
    }
    plan.io_capacity = policy.io_depth - state.inflight_io;
    if (plan.io_capacity == 0) {
        plan.reason = object_pipeline_dispatch_reason::io_backpressure;
        return plan;
    }

    const bool batch_ready = state.queued >= policy.min_flush_batch;
    const bool timed_out = state.oldest_wait_us >= policy.flush_timeout_us;
    if (!state.force && !batch_ready && !timed_out) {
        plan.reason = object_pipeline_dispatch_reason::wait_timeout;
        plan.wait_remaining_us = policy.flush_timeout_us - state.oldest_wait_us;
        return plan;
    }

    plan.batch_limit = state.inflight_io == 0 ?
        policy.initial_batch : policy.refill_batch;
    plan.submit_count = std::min({state.queued, plan.io_capacity, plan.batch_limit});
    if (plan.submit_count == 0) {
        plan.reason = object_pipeline_dispatch_reason::io_backpressure;
        return plan;
    }
    plan.reason = state.force ? object_pipeline_dispatch_reason::forced :
        batch_ready ? object_pipeline_dispatch_reason::batch_ready :
        object_pipeline_dispatch_reason::timeout;
    return plan;
}

struct object_pipeline_refill_cursor {
    bool reset(size_t total, uint64_t ticket_id = 0) {
        if (total == 0 || total > UINT32_MAX) return false;
        if (ticket_id == 0) {
            ticket_id = ticket_id_ + 1;
            if (ticket_id == 0) ticket_id = 1;
        }
        ticket_id_ = ticket_id;
        requested_ = static_cast<uint32_t>(total);
        queued_ = requested_;
        submitted_ = 0;
        inflight_ = 0;
        completed_ = 0;
        critical_ready_ = 0;
        return conserved();
    }

    object_pipeline_dispatch_plan plan(
            const object_pipeline_policy &policy,
            size_t inflight_h2d,
            uint64_t oldest_wait_us,
            bool force) const {
        object_pipeline_queue_state state;
        state.queued = queued_;
        state.inflight_io = inflight_;
        state.inflight_h2d = inflight_h2d;
        state.oldest_wait_us = oldest_wait_us;
        state.force = force;
        return object_pipeline_plan_dispatch(policy, state);
    }

    bool record_submit(size_t count) {
        if (!conserved() || count == 0 || count > queued_) return false;
        const uint32_t delta = static_cast<uint32_t>(count);
        queued_ -= delta;
        submitted_ += delta;
        inflight_ += delta;
        return conserved();
    }

    bool record_io_completion(size_t count = 1) {
        if (!conserved() || count == 0 || count > inflight_) return false;
        const uint32_t delta = static_cast<uint32_t>(count);
        inflight_ -= delta;
        completed_ += delta;
        return conserved();
    }

    bool record_io_failure(size_t count = 1) {
        if (!conserved() || count == 0 || count > inflight_) return false;
        inflight_ -= static_cast<uint32_t>(count);
        return conserved();
    }

    bool cancel_queued(size_t count = 1) {
        if (!conserved() || count == 0 || count > queued_) return false;
        queued_ -= static_cast<uint32_t>(count);
        return conserved();
    }

    bool record_critical_ready(size_t count = 1) {
        if (!conserved() || count == 0 ||
                count > completed_ - critical_ready_) {
            return false;
        }
        critical_ready_ += static_cast<uint32_t>(count);
        return conserved();
    }

    size_t total() const { return requested_; }
    size_t next() const { return submitted_; }
    size_t queued() const { return queued_; }
    size_t inflight_io() const { return inflight_; }
    size_t completed_io() const { return completed_; }

    object_pipeline_ticket ticket() const {
        object_pipeline_ticket result;
        result.ticket_id = ticket_id_;
        result.generation = ticket_id_;
        result.requested = requested_;
        result.queued = queued_;
        result.submitted = submitted_;
        result.inflight = inflight_;
        result.completed = completed_;
        if (inflight_ <= submitted_ && completed_ <= submitted_ - inflight_) {
            result.failed = submitted_ - inflight_ - completed_;
        }
        if (queued_ <= requested_ && submitted_ <= requested_ - queued_) {
            result.cancelled = requested_ - queued_ - submitted_;
        }
        result.critical_ready = critical_ready_;
        return result;
    }

    bool done() const { return conserved() && queued_ == 0 && inflight_ == 0; }

private:
    bool conserved() const {
        return ticket_id_ != 0 && requested_ != 0 &&
            queued_ <= requested_ &&
            submitted_ <= requested_ - queued_ &&
            inflight_ <= submitted_ &&
            completed_ <= submitted_ - inflight_ &&
            critical_ready_ <= completed_;
    }

    uint64_t ticket_id_ = 0;
    uint32_t requested_ = 0;
    uint32_t queued_ = 0;
    uint32_t submitted_ = 0;
    uint32_t inflight_ = 0;
    uint32_t completed_ = 0;
    uint32_t critical_ready_ = 0;
};

static_assert(sizeof(object_pipeline_refill_cursor) == 32,
    "refill cursor must stay out of the Colibri hot-path stack growth");

struct object_pipeline_transaction_handle {
    size_t slot = SIZE_MAX;
    uint64_t generation = 0;
};

enum class object_pipeline_transaction_wait_status : uint8_t {
    invalid,
    critical_ready,
    completed,
    failed,
    timeout,
};

struct object_pipeline_transaction_slot {
    uint64_t generation = 0;
    bool in_use = false;
    bool critical_published = false;
    bool critical_success = false;
    bool worker_done = false;
    bool worker_success = false;
    bool caller_released = false;
};

class object_pipeline_transaction_pool {
public:
    bool configure(size_t capacity) {
        if (capacity == 0) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        if (!slots_.empty()) return slots_.size() == capacity;
        slots_.resize(capacity);
        return true;
    }

    bool acquire(object_pipeline_transaction_handle &handle) {
        handle = {};
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t index = 0; index < slots_.size(); ++index) {
            object_pipeline_transaction_slot &slot = slots_[index];
            if (slot.in_use) continue;
            const uint64_t next_generation = slot.generation + 1;
            slot = {};
            slot.generation = next_generation != 0 ? next_generation : 1;
            slot.in_use = true;
            handle.slot = index;
            handle.generation = slot.generation;
            return true;
        }
        return false;
    }

    bool publish_critical(
            const object_pipeline_transaction_handle &handle,
            bool success) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            object_pipeline_transaction_slot *slot = find_locked(handle);
            if (!slot || slot->critical_published || slot->worker_done) return false;
            slot->critical_published = true;
            slot->critical_success = success;
        }
        cv_.notify_all();
        return true;
    }

    object_pipeline_transaction_wait_status wait_critical(
            const object_pipeline_transaction_handle &handle,
            uint64_t timeout_us) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!find_locked(handle)) {
            return object_pipeline_transaction_wait_status::invalid;
        }
        const auto status = [&]() {
            const object_pipeline_transaction_slot *slot = find_locked(handle);
            if (!slot) return object_pipeline_transaction_wait_status::invalid;
            if (slot->critical_published) {
                return slot->critical_success ?
                    object_pipeline_transaction_wait_status::critical_ready :
                    object_pipeline_transaction_wait_status::failed;
            }
            if (slot->worker_done) {
                return slot->worker_success ?
                    object_pipeline_transaction_wait_status::completed :
                    object_pipeline_transaction_wait_status::failed;
            }
            return object_pipeline_transaction_wait_status::timeout;
        };
        object_pipeline_transaction_wait_status result = status();
        if (result != object_pipeline_transaction_wait_status::timeout || timeout_us == 0) {
            return result;
        }
        const bool notified = cv_.wait_for(
            lock, std::chrono::microseconds(timeout_us), [&] {
                return status() != object_pipeline_transaction_wait_status::timeout;
            });
        return notified ? status() : object_pipeline_transaction_wait_status::timeout;
    }

    bool complete_worker(
            const object_pipeline_transaction_handle &handle,
            bool success) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            object_pipeline_transaction_slot *slot = find_locked(handle);
            if (!slot || slot->worker_done) return false;
            slot->worker_done = true;
            slot->worker_success = success;
            recycle_locked(*slot);
        }
        cv_.notify_all();
        return true;
    }

    bool release_caller(const object_pipeline_transaction_handle &handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        object_pipeline_transaction_slot *slot = find_locked(handle);
        if (!slot || slot->caller_released) return false;
        slot->caller_released = true;
        recycle_locked(*slot);
        return true;
    }

    size_t capacity() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return slots_.size();
    }

    size_t in_use() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<size_t>(std::count_if(
            slots_.begin(), slots_.end(),
            [](const object_pipeline_transaction_slot &slot) {
                return slot.in_use;
            }));
    }

private:
    object_pipeline_transaction_slot *find_locked(
            const object_pipeline_transaction_handle &handle) {
        if (handle.slot >= slots_.size() || handle.generation == 0) return nullptr;
        object_pipeline_transaction_slot &slot = slots_[handle.slot];
        return slot.in_use && slot.generation == handle.generation ? &slot : nullptr;
    }

    const object_pipeline_transaction_slot *find_locked(
            const object_pipeline_transaction_handle &handle) const {
        if (handle.slot >= slots_.size() || handle.generation == 0) return nullptr;
        const object_pipeline_transaction_slot &slot = slots_[handle.slot];
        return slot.in_use && slot.generation == handle.generation ? &slot : nullptr;
    }

    static void recycle_locked(object_pipeline_transaction_slot &slot) {
        if (slot.worker_done && slot.caller_released) {
            slot.in_use = false;
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<object_pipeline_transaction_slot> slots_;
};

} // namespace ggml_moe_common
