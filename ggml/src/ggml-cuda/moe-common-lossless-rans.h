#pragma once

#include <cstddef>
#include <cstdint>

struct CUevent_st;
using cudaEvent_t = CUevent_st *;
struct CUstream_st;
using cudaStream_t = CUstream_st *;

namespace ggml_moe_common {

constexpr uint32_t lossless_rans_alphabet = 16;
constexpr uint32_t lossless_rans_streams = 256;
constexpr uint32_t lossless_rans_scale_bits = 14;
constexpr uint32_t lossless_rans_table_size = 1U << lossless_rans_scale_bits;
constexpr uint32_t lossless_rans_q40_table_count = 5;
constexpr uint32_t lossless_rans_q40_scale_planes = 4;
constexpr uint32_t lossless_rans_mxfp4_table_count = 54;
constexpr uint32_t lossless_rans_mxfp4_scale_planes = 2;
constexpr size_t lossless_mxfp4_rank_remap_bytes = 3 * 16 * 16;
constexpr uint32_t lossless_rans_mxfp4_max_chunks = 64;
constexpr uint32_t lossless_rans_mxfp4_max_scale_records =
    lossless_rans_mxfp4_scale_planes * lossless_rans_mxfp4_max_chunks;
constexpr uint32_t lossless_rans_max_table_count = 64;
constexpr uint32_t lossless_crc32_segments = 4096;

enum class lossless_rans_status : uint32_t {
    ok = 0,
    invalid_descriptor = 1,
    invalid_record = 2,
    invalid_stream = 3,
    stream_leftover = 4,
    stream_final_state = 5,
};

enum class lossless_q40_scale_layout : uint32_t {
    repeated_per_row = 0,
    raw_per_block = 1,
};

enum class lossless_q40_scale_codec : uint32_t {
    raw = 0,
    rans_nibble_planes = 1,
    rans_e8m0_nibble_planes = 2,
    sparse_e8m0_binary = 3,
    fixed3_e8m0_group8 = 4,
    fixed3_e8m0_group8_exceptions = 5,
};

struct lossless_rans_table {
    uint32_t freq[lossless_rans_alphabet] = {};
    uint32_t start[lossless_rans_alphabet] = {};
    uint8_t slot_to_symbol[lossless_rans_table_size] = {};
};

// All pointers are device addresses. The rANS record uses the frozen
// int4-rans256-g0 framing; scale bytes are stored separately and copied
// without numeric conversion. The destination is the existing kernel-native
// block_q4_0 slot (2-byte fp16 scale followed by 16 packed quant bytes).
struct lossless_q40_decode_descriptor {
    const uint8_t *record = nullptr;
    uint64_t record_bytes = 0;
    const uint8_t *scales = nullptr;
    uint64_t scale_bytes = 0;
    const uint8_t *scale_records[lossless_rans_q40_scale_planes] = {};
    uint64_t scale_record_bytes[lossless_rans_q40_scale_planes] = {};
    uint64_t scale_value_offset = 0;
    uint64_t scale_value_count = 0;
    uint8_t *destination = nullptr;
    uint64_t destination_bytes = 0;
    uint64_t quant_byte_offset = 0;
    uint32_t rows = 0;
    uint32_t blocks_per_row = 0;
    lossless_q40_scale_layout scale_layout =
        lossless_q40_scale_layout::repeated_per_row;
    lossless_q40_scale_codec scale_codec =
        lossless_q40_scale_codec::raw;
    uint32_t write_scales = 1;
};

// MXFP4 stores one E8M0 scale byte followed by 16 packed quant bytes per
// block. Quant symbols select one of 16 tables using the low scale nibble.
struct lossless_mxfp4_decode_descriptor {
    const uint8_t *records[lossless_rans_mxfp4_max_chunks] = {};
    uint64_t record_bytes[lossless_rans_mxfp4_max_chunks] = {};
    uint64_t quant_byte_offsets[lossless_rans_mxfp4_max_chunks] = {};
    uint64_t packed_bytes[lossless_rans_mxfp4_max_chunks] = {};
    const uint8_t *scale_records[lossless_rans_mxfp4_max_scale_records] = {};
    uint64_t scale_record_bytes[lossless_rans_mxfp4_max_scale_records] = {};
    uint8_t *destination = nullptr;
    uint64_t destination_bytes = 0;
    uint64_t blocks = 0;
    uint32_t role_index = 0;
    uint32_t chunks = 0;
    uint32_t scale_records_count = 0;
    lossless_q40_scale_codec scale_codec =
        lossless_q40_scale_codec::raw;
    uint32_t quant_context_mode = 0;
};

struct lossless_crc32_descriptor {
    const uint8_t *data = nullptr;
    uint64_t bytes = 0;
};

struct lossless_rans_cuda_context;

lossless_rans_cuda_context *lossless_rans_cuda_create(
        const lossless_rans_table &table,
        size_t max_roles);

bool lossless_rans_cuda_set_mxfp4_rank_remap(
        lossless_rans_cuda_context *context,
        const uint8_t *rank_to_symbol,
        size_t bytes);

lossless_rans_cuda_context *lossless_rans_cuda_create(
        const lossless_rans_table *tables,
        size_t table_count,
        size_t max_roles);

void lossless_rans_cuda_destroy(lossless_rans_cuda_context *context);

// Copies descriptors into a fixed device allocation and launches one CUDA
// block per role. No allocation or host synchronization occurs here.
bool lossless_rans_cuda_decode_q40_async(
        lossless_rans_cuda_context *context,
        const lossless_q40_decode_descriptor *descriptors,
        size_t role_count,
        cudaStream_t stream,
        cudaEvent_t completion = nullptr);

bool lossless_rans_cuda_decode_mxfp4_async(
        lossless_rans_cuda_context *context,
        const lossless_mxfp4_decode_descriptor *descriptors,
        size_t role_count,
        cudaStream_t stream,
        cudaEvent_t completion = nullptr);

// Statuses remain in a fixed device allocation. Callers may copy them into
// their own pinned completion record and publish role slots only after every
// status is ok.
const uint32_t *lossless_rans_cuda_status_device(
        const lossless_rans_cuda_context *context);

bool lossless_rans_cuda_crc32_async(
        lossless_rans_cuda_context *context,
        const uint8_t *data,
        size_t bytes,
        cudaStream_t stream);

// Computes one independent segmented CRC for each descriptor in a single
// launch. Results use descriptor-major [count][lossless_crc32_segments]
// layout in the context's fixed partial buffer.
bool lossless_rans_cuda_crc32_batch_async(
        lossless_rans_cuda_context *context,
        const lossless_crc32_descriptor *descriptors,
        size_t count,
        cudaStream_t stream);

const uint32_t *lossless_rans_cuda_crc32_partials_device(
        const lossless_rans_cuda_context *context);

uint32_t lossless_crc32_combine_segments(
        const uint32_t *partials,
        size_t bytes);

uint32_t lossless_crc32_combine(
        uint32_t first,
        uint32_t second,
        size_t second_bytes);

size_t lossless_rans_cuda_capacity(
        const lossless_rans_cuda_context *context);

} // namespace ggml_moe_common
