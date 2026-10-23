// SPDX-License-Identifier: MIT
#pragma once
// Private deterministic fixture controls. Production settings never read these.
#include "mac-runtime.h"
#include <cstddef>
#include <string>
#include <vector>
namespace ssd { namespace mac { namespace testing {
struct options {
    size_t budget_bytes = size_t(4096) << 20;
    size_t read_ahead = 2;
    size_t read_alignment = 4096;
    bool lru = false;
    bool prefill = true;
    std::string trace_path;
};
GGML_BACKEND_API ggml_ssd_mac_cache * create(const std::vector<std::string> &, const options &, char *, size_t) noexcept;
} } }
