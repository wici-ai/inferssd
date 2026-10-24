// SPDX-License-Identifier: MIT
#pragma once

#include "../settings/result.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace ssd { namespace host {

enum room_state { room_ok, room_pressure, room_unreadable };
// Tracked bytes also appear in held_bytes() at interval boundaries. External
// allocations (compute/checkpoints) appear only in the separate grant counter.
enum class allocation_class { external, tracked };
struct room {
    room_state cgroup = room_unreadable;
    uint64_t cgroup_shortage = 0, cgroup_surplus = 0;
    bool system = false;
    uint64_t available = 0, floor = 0, system_shortage = 0, system_surplus = 0, boost_headroom = 0;
    [[nodiscard]] bool fits() const noexcept { return !cgroup_shortage && !system_shortage; }
    [[nodiscard]] uint64_t shortage() const noexcept;
    [[nodiscard]] uint64_t surplus() const noexcept;
};

struct client_config {
    bool unified = false;
    uint64_t reserve_bytes = 512ull << 20;
    uint64_t margin_bytes = 512ull << 20;
    uint64_t explicit_reserve_bytes = 0;
    bool reclaimable_file = true;
};

// Only the requesting context's reclaimers run, synchronously and without the
// arbiter mutex. No callback or raw context pointer is stored in the owner.
struct actions {
    std::function<void()> collect_deferred;
    std::function<bool(uint64_t)> reclaim_tier;
    std::function<bool(uint64_t)> reclaim_cache;
};

struct snapshot {
    bool configured = false, unified = false, reclaimable_file = false;
    uint64_t client_id = 0, reserve = 0, margin = 0, explicit_reserve = 0, floor = 0;
    uint64_t prefill_pages = 0, all_prefill_pages = 0, pending_bytes = 0;
    uint64_t file_reserve = 0, refaults = 0, file_updates = 0;
    uint64_t granted = 0, margin_updates = 0, overlap_skips = 0;
    std::string cgroup;
};

class process_arbiter;
class context_client;
namespace detail { struct client_state; }

// A successful take debits process pending bytes before returning. Keep this
// ticket until allocation AND required first touch complete. Commit releases
// pending accounting to the measured OS usage; cancellation also rolls back
// the unmaterialized grant. Both operations are idempotent and noexcept.
class allocation_ticket {
  public:
    allocation_ticket() noexcept = default;
    ~allocation_ticket();
    allocation_ticket(allocation_ticket &&) noexcept;
    allocation_ticket & operator=(allocation_ticket &&) noexcept;
    allocation_ticket(const allocation_ticket &) = delete;
    allocation_ticket & operator=(const allocation_ticket &) = delete;
    explicit operator bool() const noexcept { return id_ != 0; }
    [[nodiscard]] const room & observation() const noexcept { return observation_; }
    void commit() noexcept;
    void cancel() noexcept;
  private:
    friend class context_client;
    allocation_ticket(std::shared_ptr<process_arbiter>, std::shared_ptr<detail::client_state>,
                      uint64_t id, room observation) noexcept;
    void finish(bool materialized) noexcept;
    std::shared_ptr<process_arbiter> owner_;
    std::shared_ptr<detail::client_state> client_;
    uint64_t id_ = 0;
    room observation_;
};

// One strong owner belongs to L3. The factory obtaining that owner must live in
// one llama translation unit, never in an inline/header singleton or backend.
class process_arbiter : public std::enable_shared_from_this<process_arbiter> {
  public:
    using reader = std::function<bool(const std::string &, std::string &)>;
    explicit process_arbiter(reader injected_reader = {}) noexcept;
    ~process_arbiter();
    process_arbiter(const process_arbiter &) = delete;
    process_arbiter & operator=(const process_arbiter &) = delete;
    [[nodiscard]] result<std::unique_ptr<context_client>> make_client() noexcept;
  private:
    friend class context_client;
    friend class allocation_ticket;
    struct impl;
    std::unique_ptr<impl> impl_;
    status init_status_ = status::ok;
};

class context_client {
  public:
    ~context_client();
    context_client(const context_client &) = delete;
    context_client & operator=(const context_client &) = delete;
    // Configure before every admission path, including null-sink graph reserve.
    // Repeating the identical configuration is harmless; changing it returns
    // invalid. Clients must agree on the owner's physical DRAM/reclaim domain.
    [[nodiscard]] result<void> configure(const client_config &) noexcept;
    [[nodiscard]] result<snapshot> state() const noexcept;
    [[nodiscard]] result<void> set_prefill_pages(uint64_t) noexcept;
    [[nodiscard]] result<void> clear_prefill_pages() noexcept { return set_prefill_pages(0); }
    [[nodiscard]] result<uint64_t> dram_bytes(uint64_t host, uint64_t device) const noexcept;
    [[nodiscard]] result<room> measure(uint64_t extra, bool full_boost = false) const noexcept;
    [[nodiscard]] result<room> project_decode_capacity(uint64_t extra = 0, bool full_boost = false) const noexcept;
    // Observation/reclaim compatibility seam. It does not reserve a future
    // allocation. Production allocations must keep the returned take ticket.
    [[nodiscard]] result<room> make_room(uint64_t extra, const actions & = {}) noexcept;
    [[nodiscard]] result<allocation_ticket> take(uint64_t host, uint64_t device = 0, const actions & = {},
                           bool require_finite_cgroup = false,
                           allocation_class classification = allocation_class::external) noexcept;
    [[nodiscard]] result<bool> update_file_reserve() noexcept;
    [[nodiscard]] result<void> begin_interval(uint64_t held) noexcept;
    [[nodiscard]] result<bool> end_interval(bool consecutive, uint64_t held, uint64_t & untracked) noexcept;
    // Call on known external frees/replacements: invalidate this actor and
    // peers for the current interval; cumulative grants cancel at the next boundary.
    [[nodiscard]] result<void> note_known_memory_change() noexcept;
    // Rejects new operations, then waits for in-flight reclaimers and tickets.
    // Executors must drain before L3 destroys this registration.
    void close() noexcept;
  private:
    friend class process_arbiter;
    context_client(std::shared_ptr<process_arbiter>, std::shared_ptr<detail::client_state>) noexcept;
    std::shared_ptr<process_arbiter> owner_;
    std::shared_ptr<detail::client_state> state_;
};

}} // namespace ssd::host
