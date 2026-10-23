#include "moe-common-device-descriptor.h"

#include <cuda_runtime.h>

#include <limits>

namespace ggml_moe_common {

bool device_descriptor_buffer_allocate_fixed(
        device_descriptor_buffer &buffer,
        size_t capacity) {
    ++buffer.allocation_attempts;
    if (capacity == 0 ||
            capacity > std::numeric_limits<size_t>::max() / sizeof(ggml_moe_expert_descriptor)) {
        ++buffer.allocation_failures;
        return false;
    }
    if (buffer.device) {
        if (buffer.capacity >= capacity) {
            ++buffer.allocation_reuses;
            return true;
        }
        ++buffer.allocation_failures;
        return false;
    }
    if (buffer.capacity != 0 || buffer.bytes != 0) {
        ++buffer.allocation_failures;
        return false;
    }

    const size_t bytes = capacity * sizeof(ggml_moe_expert_descriptor);
    ggml_moe_expert_descriptor *device = nullptr;
    if (cudaMalloc(&device, bytes) != cudaSuccess) {
        cudaGetLastError();
        ++buffer.allocation_failures;
        return false;
    }
    buffer.device = device;
    buffer.capacity = capacity;
    buffer.bytes = bytes;
    ++buffer.allocations;
    return true;
}

bool device_descriptor_buffer_copy_async(
        device_descriptor_buffer &buffer,
        const ggml_moe_expert_descriptor *descriptors,
        size_t count,
        cudaStream_t stream) {
    if (count == 0) return true;
    if (!buffer.device || !descriptors || count > buffer.capacity) {
        ++buffer.copy_failures;
        return false;
    }
    const size_t bytes = count * sizeof(ggml_moe_expert_descriptor);
    if (cudaMemcpyAsync(buffer.device, descriptors, bytes,
                cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        cudaGetLastError();
        ++buffer.copy_failures;
        return false;
    }
    ++buffer.copy_calls;
    buffer.copied_entries += count;
    buffer.copied_bytes += bytes;
    return true;
}

void device_descriptor_buffer_release(device_descriptor_buffer &buffer) {
    if (buffer.device) {
        (void)cudaFree(buffer.device);
    }
    buffer.device = nullptr;
    buffer.capacity = 0;
    buffer.bytes = 0;
}

} // namespace ggml_moe_common
