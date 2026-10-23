// SPDX-License-Identifier: MIT
#pragma once

namespace ssd {
namespace telemetry {
class writer;
struct debug_channels;
}

// The runtime owns these services. Executors copy the debug selection and borrow
// the writer only until their workers and publishers have all joined.
struct runtime_services {
    telemetry::writer * events = nullptr;
    const telemetry::debug_channels * debug = nullptr;
};
}  // namespace ssd
