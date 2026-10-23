#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ggml_moe_common {

// Metadata only: the caller owns the fixed-size pinned payload slots. A fill
// reservation exclusively owns its slot until publish/cancel. A ready-slot
// lease must be held until the last asynchronous reader (including H2D) ends.
class common_dynamic_tier_lru {
public:
    struct lease {
        size_t slot = SIZE_MAX;
        uint64_t generation = 0;

        bool valid() const { return slot != SIZE_MAX && generation != 0; }
    };

    struct reservation {
        size_t slot = SIZE_MAX;
        uint64_t generation = 0;
        bool evicted = false;
        size_t evicted_key = SIZE_MAX;

        bool valid() const { return slot != SIZE_MAX && generation != 0; }
    };

    struct statistics {
        uint64_t admissions = 0;
        uint64_t evictions = 0;
        size_t ready = 0;
        size_t filling = 0;
        size_t leased = 0;
    };

    explicit common_dynamic_tier_lru(size_t slot_count = 0) {
        reset(slot_count);
    }

    // Reconfiguration is allowed only without outstanding payload users.
    bool reset(size_t slot_count) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (filling_ != 0 || leased_ != 0) {
            return false;
        }
        slots_.assign(slot_count, slot_state{});
        index_.clear();
        index_.reserve(slot_count);
        lru_.clear();
        free_.clear();
        free_.reserve(slot_count);
        for (size_t i = slot_count; i != 0; --i) {
            free_.push_back(i - 1);
        }
        admissions_ = evictions_ = 0;
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return slots_.size();
    }

    // A hit refreshes recency even if another reader already holds a lease.
    // The mutex also publishes payload writes completed before publish().
    bool acquire(size_t key, lease & out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (out.valid()) {
            return false;
        }
        const auto it = index_.find(key);
        if (it == index_.end()) {
            return false;
        }
        slot_state & s = slots_[it->second];
        if (!s.ready) {
            return false;
        }
        ++s.leases;
        ++leased_;
        lru_.splice(lru_.end(), lru_, s.recency);
        out.slot = it->second;
        out.generation = s.generation;
        return true;
    }

    // Never waits for a busy slot or duplicate fill. The caller can continue
    // using its SSD staging buffer when admission cannot proceed immediately.
    bool reserve(size_t key, reservation & out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (out.valid() || index_.find(key) != index_.end()) {
            return false;
        }
        size_t slot;
        bool evicted = false;
        size_t evicted_key = SIZE_MAX;
        if (!free_.empty()) {
            slot = free_.back();
            free_.pop_back();
        } else {
            auto victim = lru_.begin();
            while (victim != lru_.end() && slots_[*victim].leases != 0) {
                ++victim;
            }
            if (victim == lru_.end()) {
                return false;
            }
            slot = *victim;
            evicted_key = slots_[slot].key;
            index_.erase(evicted_key);
            lru_.erase(victim);
            evicted = true;
            ++evictions_;
        }
        slot_state & s = slots_[slot];
        s.key = key;
        // Generations survive reset(), so stale handles cannot alias a new fill.
        if (++generation_ == 0) {
            ++generation_;
        }
        s.generation = generation_;
        s.ready = false;
        s.filling = true;
        s.leases = 0;
        index_[key] = slot;
        ++filling_;
        out.slot = slot;
        out.generation = s.generation;
        out.evicted = evicted;
        out.evicted_key = evicted_key;
        return true;
    }

    // Call only after all payload writes have completed. In particular,
    // enqueueing an asynchronous copy is not sufficient to publish its slot.
    bool publish(reservation & fill) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!owns_fill(fill)) {
            return false;
        }
        slot_state & s = slots_[fill.slot];
        lru_.push_back(fill.slot);
        s.recency = --lru_.end();
        s.filling = false;
        s.ready = true;
        --filling_;
        ++admissions_;
        fill = reservation{};
        return true;
    }

    // A cancelled fill is empty; evicted data is never republished, because
    // the failed writer may already have overwritten some or all of it.
    bool cancel(reservation & fill) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!owns_fill(fill)) {
            return false;
        }
        slot_state & s = slots_[fill.slot];
        index_.erase(s.key);
        s.filling = false;
        s.ready = false;
        --filling_;
        free_.push_back(fill.slot);
        fill = reservation{};
        return true;
    }

    // Handles are explicit ownership tokens, not independently releasable
    // copies. Successful release consumes the caller's token.
    bool release(lease & held) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!held.valid() || held.slot >= slots_.size()) {
            return false;
        }
        slot_state & s = slots_[held.slot];
        if (!s.ready || s.generation != held.generation || s.leases == 0) {
            return false;
        }
        --s.leases;
        --leased_;
        held = lease{};
        return true;
    }

    statistics get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        statistics result;
        result.admissions = admissions_;
        result.evictions = evictions_;
        result.ready = lru_.size();
        result.filling = filling_;
        result.leased = leased_;
        return result;
    }

private:
    struct slot_state {
        size_t key = SIZE_MAX;
        uint64_t generation = 0;
        size_t leases = 0;
        bool ready = false;
        bool filling = false;
        std::list<size_t>::iterator recency;
    };

    bool owns_fill(const reservation & fill) const {
        return fill.valid() && fill.slot < slots_.size() &&
            slots_[fill.slot].filling && slots_[fill.slot].generation == fill.generation;
    }

    mutable std::mutex mutex_;
    std::vector<slot_state> slots_;
    std::vector<size_t> free_;
    std::unordered_map<size_t, size_t> index_;
    std::list<size_t> lru_;
    uint64_t generation_ = 0;
    uint64_t admissions_ = 0;
    uint64_t evictions_ = 0;
    size_t filling_ = 0;
    size_t leased_ = 0;
};

} // namespace ggml_moe_common
