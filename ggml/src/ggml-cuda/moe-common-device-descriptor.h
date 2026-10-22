#pragma once

#include "moe-expert-descriptor.h"

#include <cuda_runtime_api.h>

namespace ggml_moe_common {

struct device_descriptor_buffer {
    ggml_moe_expert_descriptor *device = nullptr;
    size_t capacity = 0;
    size_t bytes = 0;
    uint64_t allocation_attempts = 0;
    uint64_t allocations = 0;
    uint64_t allocation_reuses = 0;
    uint64_t allocation_failures = 0;
    uint64_t copy_calls = 0;
    uint64_t copy_failures = 0;
    uint64_t copied_entries = 0;
    uint64_t copied_bytes = 0;
};

// A successful first allocation fixes the device address for the buffer's
// lifetime. Later requests may reuse that capacity but never resize it.
bool device_descriptor_buffer_allocate_fixed(
        device_descriptor_buffer &buffer,
        size_t capacity);
bool device_descriptor_buffer_copy_async(
        device_descriptor_buffer &buffer,
        const ggml_moe_expert_descriptor *descriptors,
        size_t count,
        cudaStream_t stream);
void device_descriptor_buffer_release(device_descriptor_buffer &buffer);

} // namespace ggml_moe_common
