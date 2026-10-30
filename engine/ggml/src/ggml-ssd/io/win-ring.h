// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../settings/result.h"
#include "ring-types.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace ssd { namespace io {
// linux_ring's interface on Windows. init() picks IoRing (Windows 11: one
// submission call for a batch, registered buffers) when the system has it,
// else overlapped ReadFile completing on one event per ring. fd_direct comes
// from platform::open_direct, so offsets, lengths and targets are multiples of
// 4096. One owner thread; buffers are borrowed until drain.
class win_ring {
public:
    win_ring() noexcept;
    ~win_ring();
    win_ring(const win_ring &) = delete;
    win_ring & operator=(const win_ring &) = delete;
    [[nodiscard]] result<void> init(unsigned depth, bool fixed) noexcept;
    void close_ring() noexcept;
    [[nodiscard]] result<void> register_buffers(const std::vector<buffer_span> & buffers) noexcept;
    [[nodiscard]] result<void> unregister_buffers() noexcept;
    [[nodiscard]] result<void> queue(const read_range &, void *, uint64_t, bool) noexcept;
    [[nodiscard]] result<void> submit(unsigned count) noexcept;
    [[nodiscard]] result<bool> pop(completion &) noexcept;
    [[nodiscard]] result<void> wait() noexcept;
    [[nodiscard]] result<void> drain(void *, void (*)(void *, const completion &)) noexcept;
    uint64_t pending() const noexcept;
    uint64_t completed() const noexcept;
    bool read_fixed() const noexcept;
    bool buffers_registered() const noexcept;
    // "ioring" or "overlapped" once initialized.
    const char * backend_name() const noexcept;
    // Tests only: later init() calls skip IoRing, to exercise the fallback.
    static void force_fallback_for_testing(bool force) noexcept;
private:
    struct impl;
    std::unique_ptr<impl> state_;
};
}} // namespace ssd::io
