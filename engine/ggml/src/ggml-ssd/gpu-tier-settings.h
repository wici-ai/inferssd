// SPDX-License-Identifier: MIT
#pragma once

#include "runtime-services.h"
#include <cstddef>
#include <cstdint>

namespace ssd {
namespace settings { class snapshot; }
namespace gpu {
// Same-build private backend extension. The factory copies all configuration
// before returning; it never retains the snapshot/debug pointers.
struct settings_api {
    uint32_t version;
    size_t size;
    size_t runtime_services_size;
    void * (*create)(const settings::snapshot *, const runtime_services *, char *, size_t) noexcept;
};
using get_settings_api = const settings_api * (*)();
}  // namespace gpu
}  // namespace ssd
