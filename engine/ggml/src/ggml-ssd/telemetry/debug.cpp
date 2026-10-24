// SPDX-License-Identifier: MIT
#include "telemetry.h"

#include <cstdlib>
#include <sstream>

namespace ssd {
namespace telemetry {
result<debug_channels> read_debug_environment() noexcept {
    try {
        debug_channels out;
        const char *   channels  = std::getenv("INFERSSD_DEBUG");
        const char *   directory = std::getenv("INFERSSD_TRACE_DIR");
        if (directory) {
            out.trace_dir = directory;
        }
        if (!channels || !*channels) {
            return { std::move(out), status::ok, {} };
        }
        std::istringstream stream(channels);
        std::string        channel;
        while (std::getline(stream, channel, ',')) {
            if (channel == "trace") {
                out.trace = true;
            } else if (channel == "profile") {
                out.profile = true;
            } else if (channel == "mrc") {
                out.mrc = true;
            } else if (channel == "ledger") {
                out.ledger = true;
            } else {
                return { {}, status::invalid, "unknown INFERSSD_DEBUG channel " + channel };
            }
        }
        return { std::move(out), status::ok, {} };
    } catch (const std::bad_alloc &) {
        return { {}, status::resource, "oom" };
    }
}
}  // namespace telemetry
}  // namespace ssd
