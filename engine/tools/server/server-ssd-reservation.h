// SPDX-License-Identifier: MIT
#pragma once
#include "llama-ssd.h"
#include <cstddef>
#include <utility>

// The public opaque handle pins the context budget while an actual allocation
// and its first touch run. A refused reservation never reaches the allocator.
class server_ssd_reservation {
    llama_ssd_host_reservation * handle_ = nullptr;
  public:
    server_ssd_reservation() noexcept = default;
    server_ssd_reservation(llama_context * ctx, size_t bytes) {
        char error[256] = {};
        handle_ = llama_ssd_host_reserve(ctx, bytes, error, sizeof(error));
    }
    ~server_ssd_reservation() { if (handle_) { llama_ssd_host_reservation_finish(handle_, false); } }
    server_ssd_reservation(const server_ssd_reservation &) = delete;
    server_ssd_reservation & operator=(const server_ssd_reservation &) = delete;
    server_ssd_reservation(server_ssd_reservation && other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    server_ssd_reservation & operator=(server_ssd_reservation && other) noexcept {
        if (this != &other) {
            if (handle_) { llama_ssd_host_reservation_finish(handle_, false); }
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }
    explicit operator bool() const noexcept { return handle_ != nullptr; }
    void commit() noexcept {
        if (handle_) { llama_ssd_host_reservation_finish(std::exchange(handle_, nullptr), true); }
    }
};

// Known external frees/replacements invalidate overlapping driver-learning
// intervals, including this actor. Declare before temporary buffers so their
// destruction precedes the final notification.
struct server_ssd_memory_change {
    llama_context * ctx;
    explicit server_ssd_memory_change(llama_context * value) noexcept : ctx(value) {
        llama_ssd_host_note_memory_change(ctx);
    }
    ~server_ssd_memory_change() { llama_ssd_host_note_memory_change(ctx); }
};
