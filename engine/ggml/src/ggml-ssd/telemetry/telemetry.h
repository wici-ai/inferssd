// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../settings/result.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>

namespace ssd {
namespace settings {
class snapshot;
}
}  // namespace ssd

namespace ssd {
namespace telemetry {

// An owned writer per runtime/context, never a mutable global. Destruction flushes
// and closes after its users/threads have been joined by the context owner.
class writer {
  public:
    [[nodiscard]] static result<std::unique_ptr<writer>> open(const std::string & path) noexcept;
    ~writer();
    writer(const writer &)                               = delete;
    writer &                   operator=(const writer &) = delete;
    [[nodiscard]] result<void> settings(const ssd::settings::snapshot & values) noexcept;
    // payload is one complete JSON object (validated before publication).
    [[nodiscard]] result<void> event(const std::string & name, const std::string & payload) noexcept;
    [[nodiscard]] result<void> flush() noexcept;
  private:
    explicit writer(std::FILE * stream) : stream_(stream) {}

    std::FILE * stream_ = nullptr;
    std::mutex  mutex_;
    uint64_t    sequence_ = 0;
    bool        failed_   = false;
};

struct debug_channels {
    bool        trace = false, profile = false, mrc = false, ledger = false;
    std::string trace_dir;
};

// Sole optional environment reads in the new core. Channels affect observations only.
[[nodiscard]] result<debug_channels> read_debug_environment() noexcept;

}  // namespace telemetry
}  // namespace ssd
