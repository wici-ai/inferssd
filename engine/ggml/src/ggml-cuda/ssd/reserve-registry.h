// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ggml-ssd/settings/result.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <map>
#include <mutex>

namespace ssd { namespace gpu {

// The CUDA translation unit owns the only instance. The allocator reads an
// atomic maximum, while backend attach/free serialize immutable registrations.
template <size_t Devices> class reserve_registry {
  public:
    reserve_registry() noexcept {
        for (auto & maximum : maxima_) { maximum.store(0, std::memory_order_relaxed); }
    }

    [[nodiscard]] status add(const void * owner, int device, size_t bytes) noexcept {
        if (!owner || device < 0 || static_cast<size_t>(device) >= Devices) { return status::invalid; }
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto found = owners_.find(owner);
            if (found != owners_.end()) {
                return found->second.device == device && found->second.bytes == bytes ? status::ok : status::invalid;
            }
            owners_.emplace(owner, entry{device, bytes});
            publish(device);
            return status::ok;
        } catch (...) { return status::resource; }
    }

    [[nodiscard]] status remove(const void * owner) noexcept {
        if (!owner) { return status::invalid; }
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto found = owners_.find(owner);
            if (found != owners_.end()) {
                const int device = found->second.device;
                owners_.erase(found);
                publish(device);
            }
            return status::ok;
        } catch (...) { return status::resource; }
    }

    [[nodiscard]] size_t maximum(int device) const noexcept {
        return device >= 0 && static_cast<size_t>(device) < Devices ?
            maxima_[device].load(std::memory_order_acquire) : 0;
    }

  private:
    struct entry { int device; size_t bytes; };
    void publish(int device) noexcept {
        size_t bytes = 0;
        for (const auto & owner : owners_) {
            if (owner.second.device == device && owner.second.bytes > bytes) { bytes = owner.second.bytes; }
        }
        maxima_[device].store(bytes, std::memory_order_release);
    }
    std::mutex mutex_;
    std::map<const void *, entry> owners_;
    std::array<std::atomic<size_t>, Devices> maxima_{};
};

}} // namespace ssd::gpu
