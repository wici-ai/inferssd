// SPDX-License-Identifier: MIT
#pragma once
#include "../settings/result.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sys/uio.h>
#include <vector>

namespace ssd { namespace io {
struct read_range { int fd_direct; uint64_t offset; size_t nbytes; };
struct completion { uint64_t user_data; int32_t res; uint32_t flags; };
// The existing Linux direct-syscall ring, moved without merging its queue or
// scheduling with the graph-stream reader. Buffers are borrowed until drain.
class linux_ring {
public:
    linux_ring() noexcept;
    ~linux_ring();
    linux_ring(const linux_ring &) = delete;
    linux_ring & operator=(const linux_ring &) = delete;
    [[nodiscard]] result<void> init(unsigned depth, bool fixed) noexcept;
    void close_ring() noexcept;
    [[nodiscard]] result<void> register_buffers(const std::vector<iovec> & buffers) noexcept;
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
private:
    struct impl;
    std::unique_ptr<impl> state_;
};
}} // namespace ssd::io
