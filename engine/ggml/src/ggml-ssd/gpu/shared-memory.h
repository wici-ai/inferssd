// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ggml-backend.h"

namespace ssd { namespace gpu {

// Whether a device's memory is the host's DRAM: an integrated GPU, or a device whose backend says it shares host
// memory (Metal on Apple silicon, CUDA on unified memory). The one test for it: the host ledger prices such a
// device's buffers out of host memory (the GPU lane's provider, device_is_unified), and the context holds them to the
// host's room as well as to the device's own limit.
inline bool shares_host_memory(ggml_backend_dev_t device) {
    if (device == nullptr) {
        return false;
    }
    if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
        return true;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    const auto shares = reg ? reinterpret_cast<bool (*)(ggml_backend_dev_t)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_ssd_dev_shares_host_memory")) : nullptr;
    return shares && shares(device);
}

}} // namespace ssd::gpu
