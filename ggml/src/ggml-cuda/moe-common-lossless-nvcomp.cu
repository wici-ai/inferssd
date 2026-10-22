#include "moe-common-lossless-nvcomp.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <new>
#include <vector>

#if defined(GGML_CUDA_USE_NVCOMP)
#include <nvcomp/ans.hpp>
#endif

namespace ggml_moe_common {

#if defined(GGML_CUDA_USE_NVCOMP)

namespace {

constexpr size_t role_count = 3;
constexpr size_t rank_map_bytes = role_count * 16 * 16;

__constant__ uint8_t g_nvcomp_rank_to_symbol[role_count][16][16];

struct device_descriptor {
    uint8_t *destinations[role_count];
    uint64_t blocks[role_count];
    uint8_t role_mask;
};

__global__ void remap_mxfp4_rank_kernel(
        const uint8_t *transformed,
        size_t transformed_stride,
        const device_descriptor *descriptors,
        size_t objects) {
    const size_t object = blockIdx.y / role_count;
    const size_t role = blockIdx.y % role_count;
    if (object >= objects) return;
    const device_descriptor &descriptor = descriptors[object];
    if ((descriptor.role_mask & (1U << role)) == 0 ||
            !descriptor.destinations[role]) {
        return;
    }
    uint64_t role_offset = 0;
    for (size_t previous = 0; previous < role; ++previous) {
        role_offset += descriptor.blocks[previous] * 17;
    }
    const uint64_t blocks = descriptor.blocks[role];
    constexpr uint32_t quant_bytes_per_block = 16;
    const uint64_t block = static_cast<uint64_t>(blockIdx.x) *
        blockDim.x + threadIdx.x;
    if (block >= blocks) return;
    const uint8_t *source = transformed +
        object * transformed_stride + role_offset;
    const uint8_t *packed = source + block * 16;
    const uint8_t scale = source[blocks * 16 + block];
    const uint8_t context = scale & 15U;
    uint8_t *destination = descriptor.destinations[role] + block * 17;
    destination[0] = scale;
#pragma unroll
    for (uint32_t byte = 0; byte < quant_bytes_per_block; ++byte) {
        const uint8_t ranks = packed[byte];
        const uint8_t low = g_nvcomp_rank_to_symbol[
            role][context][ranks & 15U];
        const uint8_t high = g_nvcomp_rank_to_symbol[
            role][context][ranks >> 4];
        destination[byte + 1] = static_cast<uint8_t>(low | (high << 4));
    }
}

} // namespace

struct lossless_nvcomp_context {
    cudaStream_t stream = nullptr;
    size_t raw_object_bytes = 0;
    size_t transformed_stride = 0;
    size_t max_objects = 0;
    nvcomp::ANSManager *manager = nullptr;
    nvcomp::CompressionConfig compression_config;
    nvcomp::DecompressionConfig decompression_config;
    uint8_t *transformed = nullptr;
    device_descriptor *device_descriptors = nullptr;
    device_descriptor *host_descriptors = nullptr;
    std::vector<const uint8_t *> compressed_ptrs;
    std::vector<uint8_t *> output_ptrs;
    std::vector<std::vector<nvcomp::DecompressionConfig>> configuration_batches;
    std::vector<cudaEvent_t> lane_done;
    std::vector<uint8_t> lane_recorded;
};

lossless_nvcomp_context *lossless_nvcomp_create(
        size_t chunk_bytes,
        size_t raw_object_bytes,
        size_t max_objects,
        const uint8_t *rank_to_symbol,
        size_t rank_to_symbol_bytes,
        cudaStream_t stream) {
    if (chunk_bytes == 0 || raw_object_bytes == 0 || max_objects == 0 ||
            !rank_to_symbol || rank_to_symbol_bytes != rank_map_bytes ||
            !stream) {
        return nullptr;
    }
    auto *context = new (std::nothrow) lossless_nvcomp_context;
    if (!context) return nullptr;
    context->stream = stream;
    context->raw_object_bytes = raw_object_bytes;
    context->transformed_stride = (raw_object_bytes + 255) & ~size_t{255};
    context->max_objects = max_objects;
    try {
        context->manager = new nvcomp::ANSManager(
            chunk_bytes,
            nvcompBatchedANSCompressDefaultOpts,
            nvcompBatchedANSDecompressDefaultOpts,
            stream,
            nvcomp::ComputeAndNoVerify);
        context->compression_config =
            context->manager->configure_compression(raw_object_bytes);
        context->decompression_config =
            context->manager->configure_decompression(
                context->compression_config);
        context->compressed_ptrs.resize(max_objects);
        context->output_ptrs.resize(max_objects);
        context->configuration_batches.resize(max_objects);
        for (size_t count = 1; count <= max_objects; ++count) {
            context->configuration_batches[count - 1].assign(
                count, context->decompression_config);
        }
        context->lane_done.assign(max_objects, nullptr);
        context->lane_recorded.assign(max_objects, 0);
    } catch (...) {
        lossless_nvcomp_destroy(context);
        return nullptr;
    }
    if (context->decompression_config.decomp_data_size != raw_object_bytes ||
            cudaMemcpyToSymbol(
                g_nvcomp_rank_to_symbol, rank_to_symbol,
                rank_to_symbol_bytes) != cudaSuccess ||
            cudaMalloc(
                reinterpret_cast<void **>(&context->transformed),
                context->transformed_stride * max_objects) != cudaSuccess ||
            cudaMalloc(
                reinterpret_cast<void **>(&context->device_descriptors),
                sizeof(device_descriptor) * max_objects) != cudaSuccess ||
            cudaMallocHost(
                reinterpret_cast<void **>(&context->host_descriptors),
                sizeof(device_descriptor) * max_objects) != cudaSuccess) {
        cudaGetLastError();
        lossless_nvcomp_destroy(context);
        return nullptr;
    }
    for (size_t lane = 0; lane < max_objects; ++lane) {
        if (cudaEventCreateWithFlags(
                &context->lane_done[lane], cudaEventDisableTiming) !=
                cudaSuccess) {
            cudaGetLastError();
            lossless_nvcomp_destroy(context);
            return nullptr;
        }
    }
    return context;
}

void lossless_nvcomp_destroy(lossless_nvcomp_context *context) {
    if (!context) return;
    for (cudaEvent_t event : context->lane_done) {
        if (event) cudaEventDestroy(event);
    }
    cudaFreeHost(context->host_descriptors);
    cudaFree(context->device_descriptors);
    cudaFree(context->transformed);
    delete context->manager;
    delete context;
}

bool lossless_nvcomp_decode_async(
        lossless_nvcomp_context *context,
        const lossless_nvcomp_object_descriptor *descriptors,
        size_t object_count,
        cudaStream_t stream) {
    if (!context || !descriptors || object_count == 0 ||
            object_count > context->max_objects || stream != context->stream) {
        return false;
    }
    uint64_t max_blocks = 0;
    for (size_t object = 0; object < object_count; ++object) {
        if (!descriptors[object].compressed || descriptors[object].role_mask == 0) {
            return false;
        }
        uint64_t object_bytes = 0;
        device_descriptor device{};
        device.role_mask = descriptors[object].role_mask;
        for (size_t role = 0; role < role_count; ++role) {
            if (descriptors[object].blocks[role] == 0 ||
                    descriptors[object].blocks[role] >
                        UINT64_MAX / 17) {
                return false;
            }
            device.destinations[role] = descriptors[object].destinations[role];
            device.blocks[role] = descriptors[object].blocks[role];
            object_bytes += descriptors[object].blocks[role] * 17;
            max_blocks = std::max(max_blocks, descriptors[object].blocks[role]);
        }
        if (object_bytes != context->raw_object_bytes) return false;
        context->compressed_ptrs[object] = descriptors[object].compressed;
        context->output_ptrs[object] = context->transformed +
            object * context->transformed_stride;
        context->host_descriptors[object] = device;
    }
    try {
        context->manager->decompress(
            context->output_ptrs.data(), context->compressed_ptrs.data(),
            context->configuration_batches[object_count - 1]);
    } catch (...) {
        return false;
    }
    if (cudaMemcpyAsync(
            context->device_descriptors, context->host_descriptors,
            sizeof(device_descriptor) * object_count,
            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return false;
    }
    constexpr uint32_t threads = 256;
    const dim3 grid(
        static_cast<unsigned>(
            (max_blocks + threads - 1) / threads),
        static_cast<unsigned>(object_count * role_count));
    remap_mxfp4_rank_kernel<<<grid, threads, 0, stream>>>(
        context->transformed, context->transformed_stride,
        context->device_descriptors, object_count);
    return cudaGetLastError() == cudaSuccess;
}

bool lossless_nvcomp_decode_single_async(
        lossless_nvcomp_context *context,
        const lossless_nvcomp_object_descriptor &descriptor,
        size_t lane,
        cudaStream_t stream) {
    if (!context || lane >= context->max_objects ||
            stream != context->stream || !descriptor.compressed ||
            descriptor.role_mask == 0) {
        return false;
    }
    if (context->lane_recorded[lane]) {
        const cudaError_t query = cudaEventQuery(context->lane_done[lane]);
        if (query == cudaErrorNotReady) {
            if (cudaEventSynchronize(context->lane_done[lane]) != cudaSuccess) {
                return false;
            }
        } else if (query != cudaSuccess) {
            return false;
        }
    }
    uint64_t max_blocks = 0;
    uint64_t object_bytes = 0;
    device_descriptor device{};
    device.role_mask = descriptor.role_mask;
    for (size_t role = 0; role < role_count; ++role) {
        if (descriptor.blocks[role] == 0 ||
                descriptor.blocks[role] > UINT64_MAX / 17) {
            return false;
        }
        device.destinations[role] = descriptor.destinations[role];
        device.blocks[role] = descriptor.blocks[role];
        object_bytes += descriptor.blocks[role] * 17;
        max_blocks = std::max(max_blocks, descriptor.blocks[role]);
    }
    if (object_bytes != context->raw_object_bytes) return false;
    context->compressed_ptrs[lane] = descriptor.compressed;
    context->output_ptrs[lane] = context->transformed +
        lane * context->transformed_stride;
    context->host_descriptors[lane] = device;
    try {
        context->manager->decompress(
            &context->output_ptrs[lane], &context->compressed_ptrs[lane],
            context->configuration_batches[0]);
    } catch (...) {
        return false;
    }
    if (cudaMemcpyAsync(
            context->device_descriptors + lane,
            context->host_descriptors + lane,
            sizeof(device_descriptor), cudaMemcpyHostToDevice,
            stream) != cudaSuccess) {
        return false;
    }
    constexpr uint32_t threads = 256;
    const dim3 grid(
        static_cast<unsigned>(
            (max_blocks + threads - 1) / threads),
        static_cast<unsigned>(role_count));
    remap_mxfp4_rank_kernel<<<grid, threads, 0, stream>>>(
        context->transformed + lane * context->transformed_stride,
        context->transformed_stride,
        context->device_descriptors + lane, 1);
    if (cudaGetLastError() != cudaSuccess ||
            cudaEventRecord(context->lane_done[lane], stream) != cudaSuccess) {
        return false;
    }
    context->lane_recorded[lane] = 1;
    return true;
}

bool lossless_nvcomp_available() {
    return true;
}

#else

struct lossless_nvcomp_context {};

lossless_nvcomp_context *lossless_nvcomp_create(
        size_t, size_t, size_t, const uint8_t *, size_t, cudaStream_t) {
    return nullptr;
}

void lossless_nvcomp_destroy(lossless_nvcomp_context *) {}

bool lossless_nvcomp_decode_async(
        lossless_nvcomp_context *,
        const lossless_nvcomp_object_descriptor *, size_t, cudaStream_t) {
    return false;
}

bool lossless_nvcomp_decode_single_async(
        lossless_nvcomp_context *,
        const lossless_nvcomp_object_descriptor &, size_t, cudaStream_t) {
    return false;
}

bool lossless_nvcomp_available() {
    return false;
}

#endif

} // namespace ggml_moe_common
