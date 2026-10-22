#pragma once

#include <cstddef>
#include <cstdint>

struct CUstream_st;
using cudaStream_t = CUstream_st *;

namespace ggml_moe_common {

struct lossless_nvcomp_context;

struct lossless_nvcomp_object_descriptor {
    const uint8_t *compressed = nullptr;
    uint8_t *destinations[3] = {};
    uint64_t blocks[3] = {};
    uint8_t role_mask = 0;
};

lossless_nvcomp_context *lossless_nvcomp_create(
    size_t chunk_bytes,
    size_t raw_object_bytes,
    size_t max_objects,
    const uint8_t *rank_to_symbol,
    size_t rank_to_symbol_bytes,
    cudaStream_t stream);

void lossless_nvcomp_destroy(lossless_nvcomp_context *context);

bool lossless_nvcomp_decode_async(
    lossless_nvcomp_context *context,
    const lossless_nvcomp_object_descriptor *descriptors,
    size_t object_count,
    cudaStream_t stream);

bool lossless_nvcomp_decode_single_async(
    lossless_nvcomp_context *context,
    const lossless_nvcomp_object_descriptor &descriptor,
    size_t lane,
    cudaStream_t stream);

bool lossless_nvcomp_available();

} // namespace ggml_moe_common
