#include "moe-common-lossless-rans.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <new>
#include <vector>

namespace ggml_moe_common {

namespace {

constexpr uint32_t rans_l = 1U << 23;
constexpr uint32_t q40_block_bytes = 18;
constexpr uint32_t q40_scale_bytes = 2;
constexpr uint32_t q40_quant_bytes = 16;
constexpr uint32_t mxfp4_block_bytes = 17;
constexpr uint32_t mxfp4_quant_bytes = 16;
constexpr uint32_t mxfp4_scale3_group_blocks = 8;
constexpr uint32_t mxfp4_scale3_group_bytes =
    mxfp4_scale3_group_blocks * mxfp4_quant_bytes + 3;
constexpr uint32_t record_header_bytes = 1056;
constexpr uint32_t crc32_polynomial = 0xEDB88320U;

__constant__ uint32_t
    g_rans_freq[lossless_rans_max_table_count][lossless_rans_alphabet];
__constant__ uint32_t
    g_rans_start[lossless_rans_max_table_count][lossless_rans_alphabet];
__constant__ uint8_t g_mxfp4_rank_to_symbol[3][16][16];

__device__ inline uint64_t load_u64_le(const uint8_t *src) {
    uint64_t value = 0;
#pragma unroll
    for (int byte = 7; byte >= 0; --byte) {
        value = (value << 8) | src[byte];
    }
    return value;
}

__device__ inline uint32_t load_u32_le(const uint8_t *src) {
    return static_cast<uint32_t>(src[0]) |
        (static_cast<uint32_t>(src[1]) << 8) |
        (static_cast<uint32_t>(src[2]) << 16) |
        (static_cast<uint32_t>(src[3]) << 24);
}

__device__ inline uint32_t load_u32_be(const uint8_t *src) {
    return (static_cast<uint32_t>(src[0]) << 24) |
        (static_cast<uint32_t>(src[1]) << 16) |
        (static_cast<uint32_t>(src[2]) << 8) |
        static_cast<uint32_t>(src[3]);
}

__device__ inline void record_error(uint32_t *status, lossless_rans_status error) {
    atomicCAS(status, 0U, static_cast<uint32_t>(error));
}

struct rans_stream_state {
    const uint8_t *cursor = nullptr;
    const uint8_t *end = nullptr;
    uint32_t state = 0;
    uint64_t symbols = 0;
};

static __device__ bool initialize_stream(
        const uint8_t *record,
        uint64_t record_bytes,
        uint32_t stream,
        uint64_t n_symbols,
        rans_stream_state &result,
        uint32_t *status) {
    const uint32_t begin = load_u32_le(record + 16 + stream * 4);
    const uint32_t end = load_u32_le(record + 16 + (stream + 1) * 4);
    const uint64_t payload_bytes = record_bytes - record_header_bytes;
    if (begin > end || end > payload_bytes || end - begin < 4) {
        record_error(status, lossless_rans_status::invalid_stream);
        return false;
    }
    result.cursor = record + record_header_bytes + begin;
    result.end = record + record_header_bytes + end;
    result.state = load_u32_be(result.cursor);
    result.cursor += 4;
    result.symbols = n_symbols / lossless_rans_streams +
        (stream < n_symbols % lossless_rans_streams ? 1U : 0U);
    if (result.state < rans_l) {
        record_error(status, lossless_rans_status::invalid_stream);
        return false;
    }
    return true;
}

__device__ inline uint8_t decode_symbol(
        rans_stream_state &stream,
        uint32_t table_index,
        const uint8_t *slot_to_symbol) {
    constexpr uint32_t mask = lossless_rans_table_size - 1;
    const uint32_t slot = stream.state & mask;
    const uint8_t symbol = __ldg(
        slot_to_symbol + table_index * lossless_rans_table_size + slot);
    stream.state = g_rans_freq[table_index][symbol] *
        (stream.state >> lossless_rans_scale_bits) +
        slot - g_rans_start[table_index][symbol];
    while (stream.state < rans_l && stream.cursor < stream.end) {
        stream.state = (stream.state << 8) | *stream.cursor++;
    }
    return symbol;
}

__global__ void decode_q40_kernel(
        const lossless_q40_decode_descriptor *descriptors,
        uint32_t *statuses,
        const uint8_t *slot_to_symbol,
        size_t role_count) {
    const size_t role_index = blockIdx.x;
    if (role_index >= role_count || threadIdx.x >= lossless_rans_streams) return;

    const lossless_q40_decode_descriptor descriptor = descriptors[role_index];
    uint32_t *status = &statuses[role_index];
    const uint64_t blocks =
        static_cast<uint64_t>(descriptor.rows) * descriptor.blocks_per_row;
    const uint64_t role_packed_bytes = blocks * q40_quant_bytes;
    const uint64_t packed_bytes = load_u64_le(descriptor.record + 8);
    const uint64_t destination_bytes = blocks * q40_block_bytes;
    const uint64_t expected_scale_bytes =
        descriptor.scale_layout == lossless_q40_scale_layout::repeated_per_row ?
            static_cast<uint64_t>(descriptor.rows) * q40_scale_bytes :
            blocks * q40_scale_bytes;
    const bool compressed_scales = descriptor.scale_codec ==
        lossless_q40_scale_codec::rans_nibble_planes;
    bool valid_scale_records = true;
    if (compressed_scales && descriptor.write_scales) {
#pragma unroll
        for (uint32_t plane = 0;
                plane < lossless_rans_q40_scale_planes;
                ++plane) {
            valid_scale_records = valid_scale_records &&
                descriptor.scale_records[plane] &&
                descriptor.scale_record_bytes[plane] >= record_header_bytes;
        }
    }
    if (!descriptor.record || !descriptor.destination ||
            descriptor.record_bytes < record_header_bytes ||
            descriptor.rows == 0 || descriptor.blocks_per_row == 0 ||
            descriptor.destination_bytes != destination_bytes ||
            descriptor.quant_byte_offset % q40_quant_bytes != 0 ||
            packed_bytes == 0 || packed_bytes % q40_quant_bytes != 0 ||
            descriptor.quant_byte_offset > role_packed_bytes ||
            packed_bytes > role_packed_bytes - descriptor.quant_byte_offset ||
            load_u64_le(descriptor.record) != packed_bytes * 2 ||
            (descriptor.write_scales &&
                (descriptor.scale_bytes != expected_scale_bytes ||
                 (compressed_scales &&
                    (descriptor.scale_value_count == 0 ||
                     descriptor.scale_value_offset > expected_scale_bytes / q40_scale_bytes ||
                     descriptor.scale_value_count >
                        expected_scale_bytes / q40_scale_bytes -
                            descriptor.scale_value_offset)) ||
                 (descriptor.scale_codec == lossless_q40_scale_codec::raw &&
                    !descriptor.scales) ||
                 (compressed_scales && !valid_scale_records))) ||
            static_cast<uint32_t>(descriptor.scale_codec) >
                static_cast<uint32_t>(
                    lossless_q40_scale_codec::rans_nibble_planes)) {
        if (threadIdx.x == 0) {
            record_error(status, lossless_rans_status::invalid_descriptor);
        }
        return;
    }

    const uint32_t stream_index = threadIdx.x;
    rans_stream_state rans;
    const uint64_t n_symbols = packed_bytes * 2;
    if (!initialize_stream(
                descriptor.record, descriptor.record_bytes,
                stream_index, n_symbols, rans, status)) {
        return;
    }

    const uint64_t rounds = (n_symbols + lossless_rans_streams - 1) /
        lossless_rans_streams;
    for (uint64_t round = 0; round < rounds; ++round) {
        const uint8_t symbol = round < rans.symbols ?
            decode_symbol(rans, 0, slot_to_symbol) : 0;
        const uint8_t adjacent = static_cast<uint8_t>(
            __shfl_down_sync(0xffffffffU, static_cast<uint32_t>(symbol), 1));
        const uint64_t symbol_index =
            round * lossless_rans_streams + stream_index;
        if ((stream_index & 1U) == 0 && symbol_index < n_symbols) {
            const uint64_t byte_index =
                descriptor.quant_byte_offset + symbol_index / 2;
            const uint64_t block = byte_index / q40_quant_bytes;
            const uint64_t quant = byte_index % q40_quant_bytes;
            descriptor.destination[block * q40_block_bytes + q40_scale_bytes + quant] =
                static_cast<uint8_t>(symbol | (adjacent << 4));
        }
    }

    if (rans.cursor != rans.end) {
        record_error(status, lossless_rans_status::stream_leftover);
    }
    if (rans.state != rans_l) {
        record_error(status, lossless_rans_status::stream_final_state);
    }

    if (descriptor.write_scales &&
            descriptor.scale_codec == lossless_q40_scale_codec::raw) {
        for (uint64_t block = threadIdx.x; block < blocks; block += blockDim.x) {
            const uint64_t scale_index =
                descriptor.scale_layout == lossless_q40_scale_layout::repeated_per_row ?
                    block / descriptor.blocks_per_row : block;
            descriptor.destination[block * q40_block_bytes] =
                descriptor.scales[scale_index * q40_scale_bytes];
            descriptor.destination[block * q40_block_bytes + 1] =
                descriptor.scales[scale_index * q40_scale_bytes + 1];
        }
    }
    if (descriptor.write_scales && compressed_scales) {
        const uint64_t scale_values = descriptor.scale_value_count;
#pragma unroll
        for (uint32_t byte_index = 0;
                byte_index < q40_scale_bytes;
                ++byte_index) {
            const uint32_t low_plane = byte_index * 2;
            const uint32_t high_plane = low_plane + 1;
            rans_stream_state low;
            rans_stream_state high;
            if (!initialize_stream(
                        descriptor.scale_records[low_plane],
                        descriptor.scale_record_bytes[low_plane],
                        stream_index, scale_values, low, status) ||
                    !initialize_stream(
                        descriptor.scale_records[high_plane],
                        descriptor.scale_record_bytes[high_plane],
                        stream_index, scale_values, high, status)) {
                return;
            }
            const uint64_t scale_rounds =
                (scale_values + lossless_rans_streams - 1) /
                lossless_rans_streams;
            for (uint64_t round = 0; round < scale_rounds; ++round) {
                const uint8_t low_symbol = round < low.symbols ?
                    decode_symbol(low, low_plane + 1, slot_to_symbol) : 0;
                const uint8_t high_symbol = round < high.symbols ?
                    decode_symbol(high, high_plane + 1, slot_to_symbol) : 0;
                const uint64_t local_scale_index =
                    round * lossless_rans_streams + stream_index;
                if (local_scale_index >= scale_values) continue;
                const uint64_t scale_index =
                    descriptor.scale_value_offset + local_scale_index;
                const uint8_t value = static_cast<uint8_t>(
                    low_symbol | (high_symbol << 4));
                if (descriptor.scale_layout ==
                        lossless_q40_scale_layout::repeated_per_row) {
                    const uint64_t first_block =
                        scale_index * descriptor.blocks_per_row;
                    for (uint32_t row_block = 0;
                            row_block < descriptor.blocks_per_row;
                            ++row_block) {
                        descriptor.destination[
                            (first_block + row_block) * q40_block_bytes + byte_index] = value;
                    }
                } else {
                    descriptor.destination[
                        scale_index * q40_block_bytes + byte_index] = value;
                }
            }
            if (low.cursor != low.end || high.cursor != high.end) {
                record_error(status, lossless_rans_status::stream_leftover);
            }
            if (low.state != rans_l || high.state != rans_l) {
                record_error(status, lossless_rans_status::stream_final_state);
            }
        }
    }
}

__global__ void decode_mxfp4_scales_kernel(
        const lossless_mxfp4_decode_descriptor *descriptors,
        uint32_t *statuses,
        const uint8_t *slot_to_symbol,
        size_t role_count) {
    const size_t chunk_index = blockIdx.x;
    const size_t descriptor_index = blockIdx.y;
    if (descriptor_index >= role_count || threadIdx.x >= lossless_rans_streams) return;

    const lossless_mxfp4_decode_descriptor &descriptor =
        descriptors[descriptor_index];
    uint32_t *status = &statuses[descriptor_index];
    __shared__ uint32_t stream_error;
    if (threadIdx.x == 0) stream_error = 0;
    __syncthreads();

    const uint64_t destination_bytes = descriptor.blocks * mxfp4_block_bytes;
    const uint32_t scale_chunks =
        descriptor.scale_records_count / lossless_rans_mxfp4_scale_planes;
    if (chunk_index >= scale_chunks) return;
    const uint64_t first_block =
        chunk_index * descriptor.blocks / scale_chunks;
    const uint64_t last_block =
        (chunk_index + 1) * descriptor.blocks / scale_chunks;
    const uint64_t chunk_blocks = last_block - first_block;
    const uint32_t low_record =
        static_cast<uint32_t>(chunk_index) * lossless_rans_mxfp4_scale_planes;
    const uint32_t high_record = low_record + 1;
    bool valid = descriptor.destination && descriptor.blocks != 0 &&
        descriptor.destination_bytes == destination_bytes &&
        descriptor.role_index < 3 && descriptor.chunks != 0 &&
        descriptor.chunks <= lossless_rans_mxfp4_max_chunks &&
        descriptor.scale_records_count >= lossless_rans_mxfp4_scale_planes &&
        descriptor.scale_records_count <= lossless_rans_mxfp4_max_scale_records &&
        descriptor.scale_records_count % lossless_rans_mxfp4_scale_planes == 0 &&
        scale_chunks <= lossless_rans_mxfp4_max_chunks &&
        chunk_blocks != 0 &&
        first_block <= descriptor.blocks &&
        chunk_blocks <= descriptor.blocks - first_block &&
        descriptor.quant_context_mode <= 4 &&
        descriptor.scale_records[low_record] &&
        descriptor.scale_records[high_record] &&
        descriptor.scale_record_bytes[low_record] >= record_header_bytes &&
        descriptor.scale_record_bytes[high_record] >= record_header_bytes &&
        load_u64_le(descriptor.scale_records[low_record]) == chunk_blocks &&
        load_u64_le(descriptor.scale_records[high_record]) == chunk_blocks;
    if (!valid) {
        if (threadIdx.x == 0) {
            record_error(status, lossless_rans_status::invalid_descriptor);
        }
        return;
    }

    const uint32_t stream_index = threadIdx.x;
    rans_stream_state low;
    rans_stream_state high;
    const bool scale_streams_valid = initialize_stream(
            descriptor.scale_records[low_record],
            descriptor.scale_record_bytes[low_record],
            stream_index, chunk_blocks, low, status) &&
        initialize_stream(
            descriptor.scale_records[high_record],
            descriptor.scale_record_bytes[high_record],
            stream_index, chunk_blocks, high, status);
    if (!scale_streams_valid) atomicExch(&stream_error, 1U);
    if (scale_streams_valid) {
        const uint64_t rounds =
            (chunk_blocks + lossless_rans_streams - 1) /
            lossless_rans_streams;
        for (uint64_t round = 0; round < rounds; ++round) {
            const uint8_t low_symbol = round < low.symbols ?
                decode_symbol(
                    low, descriptor.role_index * 2, slot_to_symbol) : 0;
            const uint8_t high_symbol = round < high.symbols ?
                decode_symbol(
                    high, descriptor.role_index * 2 + 1, slot_to_symbol) : 0;
            const uint64_t block =
                round * lossless_rans_streams + stream_index;
            if (block < chunk_blocks) {
                const uint8_t value = static_cast<uint8_t>(
                    low_symbol | (high_symbol << 4));
                if (descriptor.quant_context_mode == 4) {
                    descriptor.destination[
                        descriptor.blocks * mxfp4_quant_bytes +
                        first_block + block] = value;
                } else {
                    descriptor.destination[
                        (first_block + block) * mxfp4_block_bytes] = value;
                }
            }
        }
        if (low.cursor != low.end || high.cursor != high.end) {
            record_error(status, lossless_rans_status::stream_leftover);
            atomicExch(&stream_error, 1U);
        }
        if (low.state != rans_l || high.state != rans_l) {
            record_error(status, lossless_rans_status::stream_final_state);
            atomicExch(&stream_error, 1U);
        }
    }
}

__global__ void decode_mxfp4_sparse_scales_kernel(
        const lossless_mxfp4_decode_descriptor *descriptors,
        uint32_t *statuses,
        size_t role_count) {
    const size_t descriptor_index = blockIdx.x;
    if (descriptor_index >= role_count) return;
    const lossless_mxfp4_decode_descriptor &descriptor =
        descriptors[descriptor_index];
    uint32_t *status = &statuses[descriptor_index];
    const uint8_t *bitmap_record = descriptor.scale_records[0];
    const uint8_t *exception_record = descriptor.scale_records[1];
    const uint64_t bitmap_bytes = (descriptor.blocks + 7) / 8;
    bool valid = descriptor.destination && descriptor.blocks != 0 &&
        descriptor.destination_bytes == descriptor.blocks * mxfp4_block_bytes &&
        descriptor.scale_codec == lossless_q40_scale_codec::sparse_e8m0_binary &&
        descriptor.scale_records_count == 2 && bitmap_record && exception_record &&
        descriptor.scale_record_bytes[0] == bitmap_bytes + 2 &&
        descriptor.scale_record_bytes[1] >= sizeof(uint32_t);
    uint32_t exception_count = 0;
    if (valid) {
        exception_count = load_u32_le(exception_record);
        valid = descriptor.scale_record_bytes[1] ==
            sizeof(uint32_t) + static_cast<uint64_t>(exception_count) * 5;
    }
    if (!valid) {
        if (threadIdx.x == 0) {
            record_error(status, lossless_rans_status::invalid_descriptor);
        }
        return;
    }
    const uint8_t base0 = bitmap_record[0];
    const uint8_t base1 = bitmap_record[1];
    const uint8_t *bitmap = bitmap_record + 2;
    const bool planar = descriptor.quant_context_mode == 4;
    for (uint64_t block = threadIdx.x;
            block < descriptor.blocks;
            block += blockDim.x) {
        const uint8_t selector =
            (bitmap[block >> 3] >> (block & 7)) & 1U;
        const uint64_t scale_offset = planar ?
            descriptor.blocks * mxfp4_quant_bytes + block :
            block * mxfp4_block_bytes;
        descriptor.destination[scale_offset] =
            selector ? base1 : base0;
    }
    __syncthreads();
    for (uint32_t exception = threadIdx.x;
            exception < exception_count;
            exception += blockDim.x) {
        const uint8_t *entry = exception_record + sizeof(uint32_t) +
            static_cast<uint64_t>(exception) * 5;
        const uint32_t block = load_u32_le(entry);
        if (block >= descriptor.blocks) {
            record_error(status, lossless_rans_status::invalid_record);
            continue;
        }
        const uint64_t scale_offset = planar ?
            descriptor.blocks * mxfp4_quant_bytes + block :
            static_cast<uint64_t>(block) * mxfp4_block_bytes;
        descriptor.destination[scale_offset] = entry[4];
    }
}

__global__ void decode_mxfp4_quant_kernel(
        const lossless_mxfp4_decode_descriptor *descriptors,
        uint32_t *statuses,
        const uint8_t *slot_to_symbol,
        size_t role_count) {
    const size_t chunk_index = blockIdx.x;
    const size_t descriptor_index = blockIdx.y;
    if (descriptor_index >= role_count || threadIdx.x >= lossless_rans_streams) return;
    const lossless_mxfp4_decode_descriptor &descriptor =
        descriptors[descriptor_index];
    if (chunk_index >= descriptor.chunks) return;

    uint32_t *status = &statuses[descriptor_index];
    const uint8_t *record = descriptor.records[chunk_index];
    const uint64_t record_bytes = descriptor.record_bytes[chunk_index];
    const uint64_t quant_byte_offset =
        descriptor.quant_byte_offsets[chunk_index];
    const uint64_t packed_bytes = descriptor.packed_bytes[chunk_index];
    const uint64_t role_quant_bytes = descriptor.blocks * mxfp4_quant_bytes;
    if (!record || record_bytes < record_header_bytes || packed_bytes == 0 ||
            packed_bytes % mxfp4_quant_bytes != 0 ||
            quant_byte_offset % mxfp4_quant_bytes != 0 ||
            quant_byte_offset > role_quant_bytes ||
            packed_bytes > role_quant_bytes - quant_byte_offset ||
            load_u64_le(record) != packed_bytes * 2) {
        if (threadIdx.x == 0) {
            record_error(status, lossless_rans_status::invalid_descriptor);
        }
        return;
    }

    const uint32_t quant_table_base = 6 + descriptor.role_index * 16;
    const uint32_t stream_index = threadIdx.x;
    const uint32_t warp_index = stream_index / 32;
    const uint32_t lane_index = stream_index & 31U;
    rans_stream_state quant;
    const uint64_t symbols = packed_bytes * 2;
    const uint64_t first_block = quant_byte_offset / mxfp4_quant_bytes;
    if (!initialize_stream(
            record, record_bytes, stream_index, symbols, quant, status)) {
        return;
    }
    const uint64_t rounds =
        (symbols + lossless_rans_streams - 1) / lossless_rans_streams;
    for (uint64_t round = 0; round < rounds; ++round) {
        const uint64_t symbol_index =
            round * lossless_rans_streams + stream_index;
        const uint64_t block = first_block +
            round * (lossless_rans_streams / 32) + warp_index;
        uint32_t context = 0;
        if (descriptor.quant_context_mode != 0) {
            if (lane_index == 0) {
                context = descriptor.destination[
                    block * mxfp4_block_bytes] & 15U;
            }
            context = __shfl_sync(0xffffffffU, context, 0);
        }
        uint8_t symbol = 0;
        if (round < quant.symbols) {
            const uint32_t table_index = descriptor.quant_context_mode == 3 ?
                quant_table_base : quant_table_base + context;
            symbol = decode_symbol(quant, table_index, slot_to_symbol);
            if (descriptor.quant_context_mode == 3) {
                symbol = g_mxfp4_rank_to_symbol[
                    descriptor.role_index][context][symbol];
            }
        }
        const uint8_t adjacent = static_cast<uint8_t>(
            __shfl_down_sync(0xffffffffU, static_cast<uint32_t>(symbol), 1));
        if ((stream_index & 1U) == 0 && symbol_index < symbols) {
            descriptor.destination[
                block * mxfp4_block_bytes + 1 + lane_index / 2] =
                    static_cast<uint8_t>(symbol | (adjacent << 4));
        }
    }
    if (quant.cursor != quant.end) {
        record_error(status, lossless_rans_status::stream_leftover);
    }
    if (quant.state != rans_l) {
        record_error(status, lossless_rans_status::stream_final_state);
    }
}

__global__ void copy_mxfp4_quant_kernel(
        const lossless_mxfp4_decode_descriptor *descriptors,
        uint32_t *statuses,
        size_t role_count) {
    const size_t chunk_index = blockIdx.x;
    const size_t descriptor_index = blockIdx.y;
    if (descriptor_index >= role_count) return;
    const lossless_mxfp4_decode_descriptor &descriptor =
        descriptors[descriptor_index];
    if (chunk_index >= descriptor.chunks) return;
    uint32_t *status = &statuses[descriptor_index];
    const uint8_t *source = descriptor.records[chunk_index];
    const uint64_t source_bytes = descriptor.record_bytes[chunk_index];
    const uint64_t quant_byte_offset =
        descriptor.quant_byte_offsets[chunk_index];
    const uint64_t packed_bytes = descriptor.packed_bytes[chunk_index];
    const uint64_t role_quant_bytes = descriptor.blocks * mxfp4_quant_bytes;
    if (!source || descriptor.quant_context_mode != 2 ||
            source_bytes != packed_bytes || packed_bytes == 0 ||
            packed_bytes % mxfp4_quant_bytes != 0 ||
            quant_byte_offset % mxfp4_quant_bytes != 0 ||
            quant_byte_offset > role_quant_bytes ||
            packed_bytes > role_quant_bytes - quant_byte_offset) {
        if (threadIdx.x == 0) {
            record_error(status, lossless_rans_status::invalid_descriptor);
        }
        return;
    }
    const uint64_t first_block = quant_byte_offset / mxfp4_quant_bytes;
    const uint64_t chunk_blocks = packed_bytes / mxfp4_quant_bytes;
    for (uint64_t local_block = threadIdx.x;
            local_block < chunk_blocks;
            local_block += blockDim.x) {
        const uint8_t *source_block =
            source + local_block * mxfp4_quant_bytes;
        uint8_t *destination_block = descriptor.destination +
            (first_block + local_block) * mxfp4_block_bytes + 1;
#pragma unroll
        for (uint32_t byte = 0; byte < mxfp4_quant_bytes; ++byte) {
            destination_block[byte] = source_block[byte];
        }
    }
}

__global__ void interleave_mxfp4_sparse_kernel(
        const lossless_mxfp4_decode_descriptor *descriptors,
        uint32_t *statuses,
        size_t role_count) {
    const size_t chunk_index = blockIdx.x;
    const size_t descriptor_index = blockIdx.y;
    if (descriptor_index >= role_count) return;
    const lossless_mxfp4_decode_descriptor &descriptor =
        descriptors[descriptor_index];
    if (chunk_index >= descriptor.chunks) return;
    const uint8_t *source = descriptor.records[chunk_index];
    const uint64_t source_bytes = descriptor.record_bytes[chunk_index];
    const uint64_t quant_byte_offset =
        descriptor.quant_byte_offsets[chunk_index];
    const uint64_t packed_bytes = descriptor.packed_bytes[chunk_index];
    const uint8_t *bitmap_record = descriptor.scale_records[0];
    const uint64_t bitmap_bytes = (descriptor.blocks + 7) / 8;
    const bool valid = source && packed_bytes != 0 &&
        source_bytes == packed_bytes &&
        packed_bytes % (mxfp4_scale3_group_blocks * mxfp4_quant_bytes) == 0 &&
        quant_byte_offset % (mxfp4_scale3_group_blocks * mxfp4_quant_bytes) == 0 &&
        quant_byte_offset <= descriptor.blocks * mxfp4_quant_bytes &&
        packed_bytes <= descriptor.blocks * mxfp4_quant_bytes -
            quant_byte_offset &&
        descriptor.destination && descriptor.destination_bytes ==
            descriptor.blocks * mxfp4_block_bytes &&
        descriptor.quant_context_mode == 2 &&
        descriptor.scale_codec ==
            lossless_q40_scale_codec::sparse_e8m0_binary &&
        descriptor.scale_records_count == 2 && bitmap_record &&
        descriptor.scale_record_bytes[0] == bitmap_bytes + 2;
    if (!valid) {
        if (threadIdx.x == 0) {
            record_error(&statuses[descriptor_index],
                lossless_rans_status::invalid_descriptor);
        }
        return;
    }

    constexpr uint32_t warp_size = 32;
    constexpr uint32_t warps_per_block = 8;
    constexpr uint32_t input_group_bytes =
        mxfp4_scale3_group_blocks * mxfp4_quant_bytes;
    constexpr uint32_t output_group_bytes =
        mxfp4_scale3_group_blocks * mxfp4_block_bytes;
    constexpr uint32_t output_words = output_group_bytes / sizeof(uint32_t);
    const uint32_t warp = threadIdx.x / warp_size;
    const uint32_t lane = threadIdx.x % warp_size;
    const uint64_t first_block =
        quant_byte_offset / mxfp4_quant_bytes;
    const uint64_t chunk_groups =
        packed_bytes / input_group_bytes;
    const uint8_t base0 = bitmap_record[0];
    const uint8_t base1 = bitmap_record[1];
    const uint8_t *bitmap = bitmap_record + 2;

    for (uint64_t local_group = warp;
            local_group < chunk_groups;
            local_group += warps_per_block) {
        const uint64_t first_group_block = first_block +
            local_group * mxfp4_scale3_group_blocks;
        const uint8_t selectors = bitmap[first_group_block >> 3];
        const uint8_t *group_source = source +
            local_group * input_group_bytes;
        uint32_t *destination = reinterpret_cast<uint32_t *>(
            descriptor.destination +
            first_group_block * mxfp4_block_bytes);
        for (uint32_t word = lane; word < output_words;
                word += warp_size) {
            uint32_t value = 0;
#pragma unroll
            for (uint32_t byte = 0; byte < sizeof(uint32_t); ++byte) {
                const uint32_t output_byte =
                    word * sizeof(uint32_t) + byte;
                const uint32_t block = output_byte / mxfp4_block_bytes;
                const uint32_t byte_in_block =
                    output_byte - block * mxfp4_block_bytes;
                const uint8_t source_byte = byte_in_block == 0 ?
                    ((selectors >> block) & 1U ? base1 : base0) :
                    group_source[
                        block * mxfp4_quant_bytes + byte_in_block - 1];
                value |= static_cast<uint32_t>(source_byte) << (byte * 8);
            }
            destination[word] = value;
        }
    }
}

__global__ void apply_mxfp4_sparse_exceptions_kernel(
        const lossless_mxfp4_decode_descriptor *descriptors,
        uint32_t *statuses,
        size_t role_count) {
    const size_t descriptor_index = blockIdx.x;
    if (descriptor_index >= role_count) return;
    const lossless_mxfp4_decode_descriptor &descriptor =
        descriptors[descriptor_index];
    const uint8_t *record = descriptor.scale_records[1];
    if (!record || descriptor.scale_record_bytes[1] < sizeof(uint32_t)) {
        if (threadIdx.x == 0) {
            record_error(&statuses[descriptor_index],
                lossless_rans_status::invalid_descriptor);
        }
        return;
    }
    const uint32_t count = load_u32_le(record);
    if (descriptor.scale_record_bytes[1] !=
            sizeof(uint32_t) + static_cast<uint64_t>(count) * 5) {
        if (threadIdx.x == 0) {
            record_error(&statuses[descriptor_index],
                lossless_rans_status::invalid_record);
        }
        return;
    }
    for (uint32_t exception = threadIdx.x;
            exception < count;
            exception += blockDim.x) {
        const uint8_t *entry = record + sizeof(uint32_t) +
            static_cast<uint64_t>(exception) * 5;
        const uint32_t block = load_u32_le(entry);
        if (block >= descriptor.blocks) {
            record_error(&statuses[descriptor_index],
                lossless_rans_status::invalid_record);
            continue;
        }
        descriptor.destination[
            static_cast<uint64_t>(block) * mxfp4_block_bytes] = entry[4];
    }
}

__global__ void expand_mxfp4_scale3_kernel(
        const lossless_mxfp4_decode_descriptor *descriptors,
        uint32_t *statuses,
        size_t role_count) {
    const size_t descriptor_index = blockIdx.y;
    if (descriptor_index >= role_count) return;
    const lossless_mxfp4_decode_descriptor &descriptor =
        descriptors[descriptor_index];
    const uint64_t expected_source_bytes =
        descriptor.blocks / mxfp4_scale3_group_blocks *
        mxfp4_scale3_group_bytes;
    if (!descriptor.records[0] || !descriptor.destination ||
            descriptor.blocks % mxfp4_scale3_group_blocks != 0 ||
            descriptor.record_bytes[0] != expected_source_bytes ||
            descriptor.destination_bytes !=
                descriptor.blocks * mxfp4_block_bytes) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            record_error(&statuses[descriptor_index],
                lossless_rans_status::invalid_record);
        }
        return;
    }

    constexpr uint32_t warp_size = 32;
    constexpr uint32_t output_group_bytes =
        mxfp4_scale3_group_blocks * mxfp4_block_bytes;
    constexpr uint32_t output_words = output_group_bytes / sizeof(uint32_t);
    const uint32_t warp = threadIdx.x / warp_size;
    const uint32_t lane = threadIdx.x % warp_size;
    const uint32_t warps_per_block = blockDim.x / warp_size;
    constexpr uint32_t staging_words_per_warp = 68;
    constexpr uint32_t staging_output_offset = 136;
    __shared__ uint32_t staging[8][staging_words_per_warp];
    const uint64_t group = static_cast<uint64_t>(blockIdx.x) *
        warps_per_block + warp;
    const uint64_t groups = descriptor.blocks /
        mxfp4_scale3_group_blocks;
    if (group >= groups) return;

    const uint8_t *group_source = descriptor.records[0] +
        group * mxfp4_scale3_group_bytes;
    uint8_t *staging_input = reinterpret_cast<uint8_t *>(staging[warp]);
    uint8_t *staging_output = staging_input + staging_output_offset;
    for (uint32_t byte = lane; byte < mxfp4_scale3_group_bytes;
            byte += warp_size) {
        staging_input[byte] = group_source[byte];
    }
    __syncwarp();

    const uint8_t *packed_scales = staging_input +
        mxfp4_scale3_group_blocks * mxfp4_quant_bytes;
    const uint32_t scale_word = static_cast<uint32_t>(packed_scales[0]) |
        (static_cast<uint32_t>(packed_scales[1]) << 8) |
        (static_cast<uint32_t>(packed_scales[2]) << 16);
    if (lane < mxfp4_scale3_group_blocks) {
        uint8_t *output_block = staging_output + lane * mxfp4_block_bytes;
        output_block[0] = static_cast<uint8_t>(
            118U + ((scale_word >> (3U * lane)) & 7U));
#pragma unroll
        for (uint32_t byte = 0; byte < mxfp4_quant_bytes; ++byte) {
            output_block[1 + byte] =
                staging_input[lane * mxfp4_quant_bytes + byte];
        }
    }
    __syncwarp();

    uint8_t *destination_group = descriptor.destination +
        group * output_group_bytes;
    const uint32_t *output_words_source =
        reinterpret_cast<const uint32_t *>(staging_output);
    uint32_t *output_words_destination =
        reinterpret_cast<uint32_t *>(destination_group);
    for (uint32_t output_word = lane; output_word < output_words;
            output_word += warp_size) {
        output_words_destination[output_word] =
            output_words_source[output_word];
    }
}

__global__ void expand_mxfp4_scale3_planar_kernel(
        const lossless_mxfp4_decode_descriptor *descriptors,
        uint32_t *statuses,
        size_t role_count) {
    const size_t descriptor_index = blockIdx.y;
    if (descriptor_index >= role_count) return;
    const lossless_mxfp4_decode_descriptor &descriptor =
        descriptors[descriptor_index];
    const uint64_t group = static_cast<uint64_t>(blockIdx.x) *
        blockDim.x + threadIdx.x;
    const uint64_t groups = descriptor.blocks /
        mxfp4_scale3_group_blocks;
    const uint64_t expected_source_bytes = groups * 3;
    if (!descriptor.records[0] || !descriptor.destination ||
            descriptor.blocks % mxfp4_scale3_group_blocks != 0 ||
            descriptor.record_bytes[0] != expected_source_bytes ||
            descriptor.destination_bytes !=
                descriptor.blocks * mxfp4_block_bytes) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            record_error(&statuses[descriptor_index],
                lossless_rans_status::invalid_record);
        }
        return;
    }
    if (group >= groups) return;

    const uint8_t *packed_scales = descriptor.records[0] + group * 3;
    const uint32_t scale_word =
        static_cast<uint32_t>(packed_scales[0]) |
        (static_cast<uint32_t>(packed_scales[1]) << 8) |
        (static_cast<uint32_t>(packed_scales[2]) << 16);
    uint8_t *scale_destination = descriptor.destination +
        descriptor.blocks * mxfp4_quant_bytes +
        group * mxfp4_scale3_group_blocks;
#pragma unroll
    for (uint32_t block = 0; block < mxfp4_scale3_group_blocks; ++block) {
        scale_destination[block] = static_cast<uint8_t>(
            118U + ((scale_word >> (3U * block)) & 7U));
    }
}

__global__ void apply_mxfp4_scale3_exceptions_kernel(
        const lossless_mxfp4_decode_descriptor *descriptors,
        uint32_t *statuses,
        size_t role_count) {
    const size_t descriptor_index = blockIdx.y;
    if (descriptor_index >= role_count) return;
    const lossless_mxfp4_decode_descriptor &descriptor =
        descriptors[descriptor_index];
    const uint32_t exception = blockIdx.x * blockDim.x + threadIdx.x;
    if (exception >= descriptor.scale_records_count) return;
    const uint8_t *records = descriptor.scale_records[0];
    const uint64_t expected_bytes =
        static_cast<uint64_t>(descriptor.scale_records_count) * 8;
    if (!records || descriptor.scale_record_bytes[0] != expected_bytes) {
        if (exception == 0) {
            record_error(&statuses[descriptor_index],
                lossless_rans_status::invalid_record);
        }
        return;
    }
    const uint8_t *record = records + static_cast<uint64_t>(exception) * 8;
    const uint32_t block = load_u32_le(record);
    if (block >= descriptor.blocks) {
        record_error(&statuses[descriptor_index],
            lossless_rans_status::invalid_record);
        return;
    }
    uint8_t *scale_destination = descriptor.quant_context_mode == 6 ?
        descriptor.destination + descriptor.blocks * mxfp4_quant_bytes + block :
        descriptor.destination + static_cast<uint64_t>(block) * mxfp4_block_bytes;
    *scale_destination = record[4];
}

__global__ void crc32_segments_kernel(
        const uint8_t *data,
        size_t bytes,
        const uint32_t *table,
        uint32_t *partials) {
    const uint32_t segment = blockIdx.x * blockDim.x + threadIdx.x;
    if (segment >= lossless_crc32_segments) return;
    const size_t segment_bytes =
        (bytes + lossless_crc32_segments - 1) / lossless_crc32_segments;
    const size_t begin = static_cast<size_t>(segment) * segment_bytes;
    const size_t end = begin + segment_bytes < bytes ?
        begin + segment_bytes : bytes;
    uint32_t crc = 0xFFFFFFFFU;
    size_t offset = begin;
    for (; offset + 8 <= end; offset += 8) {
        const uint32_t first = crc ^
            static_cast<uint32_t>(data[offset + 0]) ^
            (static_cast<uint32_t>(data[offset + 1]) << 8) ^
            (static_cast<uint32_t>(data[offset + 2]) << 16) ^
            (static_cast<uint32_t>(data[offset + 3]) << 24);
        const uint32_t second =
            static_cast<uint32_t>(data[offset + 4]) ^
            (static_cast<uint32_t>(data[offset + 5]) << 8) ^
            (static_cast<uint32_t>(data[offset + 6]) << 16) ^
            (static_cast<uint32_t>(data[offset + 7]) << 24);
        crc =
            __ldg(table + 7 * 256 + ((first >> 0) & 0xFFU)) ^
            __ldg(table + 6 * 256 + ((first >> 8) & 0xFFU)) ^
            __ldg(table + 5 * 256 + ((first >> 16) & 0xFFU)) ^
            __ldg(table + 4 * 256 + ((first >> 24) & 0xFFU)) ^
            __ldg(table + 3 * 256 + ((second >> 0) & 0xFFU)) ^
            __ldg(table + 2 * 256 + ((second >> 8) & 0xFFU)) ^
            __ldg(table + 1 * 256 + ((second >> 16) & 0xFFU)) ^
            __ldg(table + 0 * 256 + ((second >> 24) & 0xFFU));
    }
    for (; offset < end; ++offset) {
        crc = __ldg(table + ((crc ^ data[offset]) & 0xFFU)) ^ (crc >> 8);
    }
    partials[segment] = crc ^ 0xFFFFFFFFU;
}

__global__ void crc32_segments_batch_kernel(
        const lossless_crc32_descriptor *descriptors,
        size_t count,
        const uint32_t *table,
        uint32_t *partials) {
    const size_t descriptor_index = blockIdx.y;
    const uint32_t segment = blockIdx.x * blockDim.x + threadIdx.x;
    if (descriptor_index >= count || segment >= lossless_crc32_segments) return;
    const lossless_crc32_descriptor descriptor = descriptors[descriptor_index];
    if (!descriptor.data || descriptor.bytes == 0) return;
    const size_t segment_bytes =
        (descriptor.bytes + lossless_crc32_segments - 1) /
        lossless_crc32_segments;
    const size_t begin = static_cast<size_t>(segment) * segment_bytes;
    const size_t end = begin + segment_bytes < descriptor.bytes ?
        begin + segment_bytes : descriptor.bytes;
    uint32_t crc = 0xFFFFFFFFU;
    size_t offset = begin;
    for (; offset + 8 <= end; offset += 8) {
        const uint32_t first = crc ^
            static_cast<uint32_t>(descriptor.data[offset + 0]) ^
            (static_cast<uint32_t>(descriptor.data[offset + 1]) << 8) ^
            (static_cast<uint32_t>(descriptor.data[offset + 2]) << 16) ^
            (static_cast<uint32_t>(descriptor.data[offset + 3]) << 24);
        const uint32_t second =
            static_cast<uint32_t>(descriptor.data[offset + 4]) ^
            (static_cast<uint32_t>(descriptor.data[offset + 5]) << 8) ^
            (static_cast<uint32_t>(descriptor.data[offset + 6]) << 16) ^
            (static_cast<uint32_t>(descriptor.data[offset + 7]) << 24);
        crc =
            __ldg(table + 7 * 256 + ((first >> 0) & 0xFFU)) ^
            __ldg(table + 6 * 256 + ((first >> 8) & 0xFFU)) ^
            __ldg(table + 5 * 256 + ((first >> 16) & 0xFFU)) ^
            __ldg(table + 4 * 256 + ((first >> 24) & 0xFFU)) ^
            __ldg(table + 3 * 256 + ((second >> 0) & 0xFFU)) ^
            __ldg(table + 2 * 256 + ((second >> 8) & 0xFFU)) ^
            __ldg(table + 1 * 256 + ((second >> 16) & 0xFFU)) ^
            __ldg(table + 0 * 256 + ((second >> 24) & 0xFFU));
    }
    for (; offset < end; ++offset) {
        crc = __ldg(table + ((crc ^ descriptor.data[offset]) & 0xFFU)) ^
            (crc >> 8);
    }
    partials[descriptor_index * lossless_crc32_segments + segment] =
        crc ^ 0xFFFFFFFFU;
}

static bool validate_table(const lossless_rans_table &table) {
    uint32_t prefix = 0;
    for (uint32_t symbol = 0; symbol < lossless_rans_alphabet; ++symbol) {
        if (table.start[symbol] != prefix) return false;
        prefix += table.freq[symbol];
    }
    if (prefix != lossless_rans_table_size) return false;
    for (uint32_t slot = 0; slot < lossless_rans_table_size; ++slot) {
        const uint32_t symbol = table.slot_to_symbol[slot];
        if (symbol >= lossless_rans_alphabet ||
                slot < table.start[symbol] ||
                slot >= table.start[symbol] + table.freq[symbol]) {
            return false;
        }
    }
    return true;
}

} // namespace

struct lossless_rans_cuda_context {
    lossless_q40_decode_descriptor *descriptors = nullptr;
    lossless_mxfp4_decode_descriptor *mxfp4_descriptors = nullptr;
    lossless_crc32_descriptor *crc32_descriptors = nullptr;
    uint32_t *statuses = nullptr;
    uint8_t *slot_to_symbol = nullptr;
    uint32_t *crc32_table = nullptr;
    uint32_t *crc32_partials = nullptr;
    size_t capacity = 0;
    size_t table_count = 0;
    bool mxfp4_rank_remap_ready = false;
};

lossless_rans_cuda_context *lossless_rans_cuda_create(
        const lossless_rans_table &table,
        size_t max_roles) {
    return lossless_rans_cuda_create(&table, 1, max_roles);
}

lossless_rans_cuda_context *lossless_rans_cuda_create(
        const lossless_rans_table *tables,
        size_t table_count,
        size_t max_roles) {
    if (!tables || max_roles == 0 || table_count == 0 ||
            table_count > lossless_rans_max_table_count) {
        return nullptr;
    }
    for (size_t index = 0; index < table_count; ++index) {
        if (!validate_table(tables[index])) return nullptr;
    }
    auto *context = new (std::nothrow) lossless_rans_cuda_context;
    if (!context) return nullptr;
    context->capacity = max_roles;
    context->table_count = table_count;
    uint32_t upload_freq[lossless_rans_max_table_count][lossless_rans_alphabet] = {};
    uint32_t upload_start[lossless_rans_max_table_count][lossless_rans_alphabet] = {};
    std::vector<uint8_t> upload_slots(
        static_cast<size_t>(lossless_rans_max_table_count) *
        lossless_rans_table_size);
    for (size_t index = 0; index < lossless_rans_max_table_count; ++index) {
        const lossless_rans_table &table = tables[index < table_count ? index : 0];
        std::copy(
            table.freq, table.freq + lossless_rans_alphabet,
            upload_freq[index]);
        std::copy(
            table.start, table.start + lossless_rans_alphabet,
            upload_start[index]);
        std::copy(
            table.slot_to_symbol,
            table.slot_to_symbol + lossless_rans_table_size,
            upload_slots.data() + index * lossless_rans_table_size);
    }
    uint32_t crc32_table[8][256] = {};
    for (uint32_t value = 0; value < 256; ++value) {
        uint32_t crc = value;
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^
                (crc32_polynomial & (0U - (crc & 1U)));
        }
        crc32_table[0][value] = crc;
    }
    for (size_t slice = 1; slice < 8; ++slice) {
        for (size_t value = 0; value < 256; ++value) {
            const uint32_t previous = crc32_table[slice - 1][value];
            crc32_table[slice][value] =
                crc32_table[0][previous & 0xFFU] ^ (previous >> 8);
        }
    }
    if (cudaMalloc(
                reinterpret_cast<void **>(&context->descriptors),
                max_roles * sizeof(*context->descriptors)) != cudaSuccess ||
            cudaMalloc(
                reinterpret_cast<void **>(&context->statuses),
                max_roles * sizeof(*context->statuses)) != cudaSuccess ||
            cudaMalloc(
                reinterpret_cast<void **>(&context->mxfp4_descriptors),
                max_roles * sizeof(*context->mxfp4_descriptors)) != cudaSuccess ||
            cudaMalloc(
                reinterpret_cast<void **>(&context->slot_to_symbol),
                upload_slots.size()) != cudaSuccess ||
            cudaMalloc(
                reinterpret_cast<void **>(&context->crc32_table),
                sizeof(crc32_table)) != cudaSuccess ||
            cudaMalloc(
                reinterpret_cast<void **>(&context->crc32_descriptors),
                max_roles * sizeof(*context->crc32_descriptors)) != cudaSuccess ||
            cudaMalloc(
                reinterpret_cast<void **>(&context->crc32_partials),
                max_roles * lossless_crc32_segments * sizeof(uint32_t)) !=
                    cudaSuccess ||
            cudaMemcpyToSymbol(
                g_rans_freq, upload_freq, sizeof(upload_freq)) != cudaSuccess ||
            cudaMemcpyToSymbol(
                g_rans_start, upload_start, sizeof(upload_start)) != cudaSuccess ||
            cudaMemcpy(
                context->slot_to_symbol,
                upload_slots.data(),
                upload_slots.size(),
                cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(
                context->crc32_table,
                crc32_table,
                sizeof(crc32_table),
                cudaMemcpyHostToDevice) != cudaSuccess) {
        lossless_rans_cuda_destroy(context);
        return nullptr;
    }
    return context;
}

bool lossless_rans_cuda_set_mxfp4_rank_remap(
        lossless_rans_cuda_context *context,
        const uint8_t *rank_to_symbol,
        size_t bytes) {
    if (!context || !rank_to_symbol ||
            bytes != lossless_mxfp4_rank_remap_bytes) {
        return false;
    }
    if (cudaMemcpyToSymbol(
            g_mxfp4_rank_to_symbol, rank_to_symbol, bytes) != cudaSuccess) {
        return false;
    }
    context->mxfp4_rank_remap_ready = true;
    return true;
}

void lossless_rans_cuda_destroy(lossless_rans_cuda_context *context) {
    if (!context) return;
    cudaFree(context->crc32_partials);
    cudaFree(context->crc32_descriptors);
    cudaFree(context->crc32_table);
    cudaFree(context->slot_to_symbol);
    cudaFree(context->mxfp4_descriptors);
    cudaFree(context->statuses);
    cudaFree(context->descriptors);
    delete context;
}

bool lossless_rans_cuda_decode_q40_async(
        lossless_rans_cuda_context *context,
        const lossless_q40_decode_descriptor *descriptors,
        size_t role_count,
        cudaStream_t stream,
        cudaEvent_t completion) {
    if (!context || !descriptors || role_count == 0 ||
            role_count > context->capacity || !stream) {
        return false;
    }
    if (cudaMemcpyAsync(
                context->descriptors,
                descriptors,
                role_count * sizeof(*descriptors),
                cudaMemcpyHostToDevice,
                stream) != cudaSuccess ||
            cudaMemsetAsync(
                context->statuses, 0, role_count * sizeof(*context->statuses),
                stream) != cudaSuccess) {
        return false;
    }
    decode_q40_kernel<<<static_cast<unsigned>(role_count), lossless_rans_streams, 0, stream>>>(
        context->descriptors, context->statuses, context->slot_to_symbol, role_count);
    if (cudaGetLastError() != cudaSuccess) return false;
    return !completion || cudaEventRecord(completion, stream) == cudaSuccess;
}

bool lossless_rans_cuda_decode_mxfp4_async(
        lossless_rans_cuda_context *context,
        const lossless_mxfp4_decode_descriptor *descriptors,
        size_t role_count,
        cudaStream_t stream,
        cudaEvent_t completion) {
    if (!context || !descriptors || role_count == 0 ||
            role_count > context->capacity || !stream) {
        return false;
    }
    uint32_t max_chunks = 0;
    uint32_t max_scale_chunks = 0;
    const uint32_t quant_context_mode = descriptors[0].quant_context_mode;
    const lossless_q40_scale_codec scale_codec = descriptors[0].scale_codec;
    const bool fixed3 =
        scale_codec == lossless_q40_scale_codec::fixed3_e8m0_group8 ||
        scale_codec ==
            lossless_q40_scale_codec::fixed3_e8m0_group8_exceptions;
    const bool fixed3_exceptions = scale_codec ==
        lossless_q40_scale_codec::fixed3_e8m0_group8_exceptions;
    const bool fixed3_planar = fixed3 && quant_context_mode == 6;
    if (quant_context_mode == 3 && !context->mxfp4_rank_remap_ready) {
        return false;
    }
    if (!fixed3 &&
            context->table_count < lossless_rans_mxfp4_table_count) {
        return false;
    }
    for (size_t role = 0; role < role_count; ++role) {
        if (fixed3) {
            if (descriptors[role].chunks != 1 ||
                    descriptors[role].scale_codec != scale_codec ||
                    descriptors[role].quant_context_mode != quant_context_mode ||
                    (descriptors[role].quant_context_mode != 5 &&
                     descriptors[role].quant_context_mode != 6) ||
                    (!fixed3_exceptions &&
                        descriptors[role].scale_records_count != 0) ||
                    (fixed3_exceptions &&
                        (descriptors[role].scale_records_count != 0 &&
                            (!descriptors[role].scale_records[0] ||
                             descriptors[role].scale_record_bytes[0] !=
                                static_cast<uint64_t>(
                                    descriptors[role].scale_records_count) * 8))) ||
                    descriptors[role].blocks == 0 ||
                    descriptors[role].blocks %
                        mxfp4_scale3_group_blocks != 0) {
                return false;
            }
            max_chunks = std::max(max_chunks, descriptors[role].chunks);
            continue;
        }
        if (descriptors[role].chunks == 0 ||
                descriptors[role].chunks > lossless_rans_mxfp4_max_chunks ||
                descriptors[role].scale_records_count <
                    lossless_rans_mxfp4_scale_planes ||
                descriptors[role].scale_records_count >
                    lossless_rans_mxfp4_max_scale_records ||
                descriptors[role].scale_records_count %
                    lossless_rans_mxfp4_scale_planes != 0 ||
                descriptors[role].scale_codec != scale_codec ||
                descriptors[role].quant_context_mode != quant_context_mode ||
                descriptors[role].quant_context_mode > 4) {
            return false;
        }
        max_chunks = std::max(max_chunks, descriptors[role].chunks);
        max_scale_chunks = std::max(
            max_scale_chunks,
            descriptors[role].scale_records_count /
                lossless_rans_mxfp4_scale_planes);
    }
    cudaError_t error = cudaMemcpyAsync(
                context->mxfp4_descriptors,
                descriptors,
                role_count * sizeof(*descriptors),
                cudaMemcpyHostToDevice,
                stream);
    if (error != cudaSuccess) {
        std::fprintf(stderr, "lossless MXFP4 descriptor H2D failed: %s\n",
            cudaGetErrorString(error));
        return false;
    }
    error = cudaMemsetAsync(
                context->statuses, 0, role_count * sizeof(*context->statuses),
                stream);
    if (error != cudaSuccess) {
        std::fprintf(stderr, "lossless MXFP4 status reset failed: %s\n",
            cudaGetErrorString(error));
        return false;
    }
    if (fixed3) {
        constexpr uint32_t threads = 256;
        constexpr uint32_t warps_per_block = threads / 32;
        uint64_t max_groups = 0;
        for (size_t role = 0; role < role_count; ++role) {
            max_groups = std::max(max_groups,
                descriptors[role].blocks / mxfp4_scale3_group_blocks);
        }
        if (fixed3_planar) {
            const dim3 grid(
                static_cast<unsigned>(
                    (max_groups + threads - 1) / threads),
                static_cast<unsigned>(role_count));
            expand_mxfp4_scale3_planar_kernel<<<grid, threads, 0, stream>>>(
                context->mxfp4_descriptors, context->statuses, role_count);
        } else {
            const dim3 grid(
                static_cast<unsigned>(
                    (max_groups + warps_per_block - 1) / warps_per_block),
                static_cast<unsigned>(role_count));
            expand_mxfp4_scale3_kernel<<<grid, threads, 0, stream>>>(
                context->mxfp4_descriptors, context->statuses, role_count);
        }
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::fprintf(stderr, "lossless MXFP4 scale3 launch failed: %s\n",
                cudaGetErrorString(error));
            return false;
        }
        if (fixed3_exceptions) {
            uint32_t max_exceptions = 0;
            for (size_t role = 0; role < role_count; ++role) {
                max_exceptions = std::max(
                    max_exceptions, descriptors[role].scale_records_count);
            }
            if (max_exceptions != 0) {
                constexpr uint32_t exception_threads = 64;
                const dim3 exception_grid(
                    (max_exceptions + exception_threads - 1) /
                        exception_threads,
                    static_cast<unsigned>(role_count));
                apply_mxfp4_scale3_exceptions_kernel<<<
                    exception_grid, exception_threads, 0, stream>>>(
                        context->mxfp4_descriptors,
                        context->statuses, role_count);
                error = cudaGetLastError();
                if (error != cudaSuccess) {
                    std::fprintf(stderr,
                        "lossless MXFP4 scale3 exception launch failed: %s\n",
                        cudaGetErrorString(error));
                    return false;
                }
            }
        }
        return !completion || cudaEventRecord(completion, stream) == cudaSuccess;
    }
    const bool fused_sparse_interleave = scale_codec ==
            lossless_q40_scale_codec::sparse_e8m0_binary &&
        quant_context_mode == 2;
    if (fused_sparse_interleave) {
        constexpr uint32_t sparse_threads = 256;
        interleave_mxfp4_sparse_kernel<<<
            dim3(max_chunks, static_cast<unsigned>(role_count)),
            sparse_threads, 0, stream>>>(
                context->mxfp4_descriptors, context->statuses, role_count);
        apply_mxfp4_sparse_exceptions_kernel<<<
            static_cast<unsigned>(role_count), sparse_threads, 0, stream>>>(
                context->mxfp4_descriptors, context->statuses, role_count);
    } else if (scale_codec ==
            lossless_q40_scale_codec::sparse_e8m0_binary) {
        constexpr uint32_t sparse_threads = 256;
        decode_mxfp4_sparse_scales_kernel<<<
            static_cast<unsigned>(role_count), sparse_threads, 0, stream>>>(
                context->mxfp4_descriptors, context->statuses, role_count);
    } else if (scale_codec ==
            lossless_q40_scale_codec::rans_e8m0_nibble_planes) {
        decode_mxfp4_scales_kernel<<<
            dim3(max_scale_chunks, static_cast<unsigned>(role_count)),
            lossless_rans_streams, 0, stream>>>(
                context->mxfp4_descriptors, context->statuses,
                context->slot_to_symbol, role_count);
    } else {
        return false;
    }
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::fprintf(stderr, "lossless MXFP4 scale launch failed: %s\n",
            cudaGetErrorString(error));
        return false;
    }
    if (quant_context_mode == 2 && !fused_sparse_interleave) {
        constexpr uint32_t copy_threads = 256;
        copy_mxfp4_quant_kernel<<<
            dim3(max_chunks, static_cast<unsigned>(role_count)),
            copy_threads, 0, stream>>>(
                context->mxfp4_descriptors, context->statuses, role_count);
    } else if (quant_context_mode < 2 || quant_context_mode == 3) {
        decode_mxfp4_quant_kernel<<<
            dim3(max_chunks, static_cast<unsigned>(role_count)),
            lossless_rans_streams, 0, stream>>>(
                context->mxfp4_descriptors, context->statuses,
                context->slot_to_symbol, role_count);
    }
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::fprintf(stderr, "lossless MXFP4 quant launch failed: %s\n",
            cudaGetErrorString(error));
        return false;
    }
    return !completion || cudaEventRecord(completion, stream) == cudaSuccess;
}

const uint32_t *lossless_rans_cuda_status_device(
        const lossless_rans_cuda_context *context) {
    return context ? context->statuses : nullptr;
}

bool lossless_rans_cuda_crc32_async(
        lossless_rans_cuda_context *context,
        const uint8_t *data,
        size_t bytes,
        cudaStream_t stream) {
    if (!context || !data || bytes == 0 || !stream) return false;
    constexpr uint32_t threads = 256;
    constexpr uint32_t blocks =
        (lossless_crc32_segments + threads - 1) / threads;
    crc32_segments_kernel<<<blocks, threads, 0, stream>>>(
        data, bytes, context->crc32_table, context->crc32_partials);
    return cudaGetLastError() == cudaSuccess;
}

bool lossless_rans_cuda_crc32_batch_async(
        lossless_rans_cuda_context *context,
        const lossless_crc32_descriptor *descriptors,
        size_t count,
        cudaStream_t stream) {
    if (!context || !descriptors || count == 0 ||
            count > context->capacity || !stream) {
        return false;
    }
    if (cudaMemcpyAsync(
            context->crc32_descriptors,
            descriptors,
            count * sizeof(*descriptors),
            cudaMemcpyHostToDevice,
            stream) != cudaSuccess) {
        return false;
    }
    constexpr uint32_t threads = 256;
    constexpr uint32_t blocks =
        (lossless_crc32_segments + threads - 1) / threads;
    crc32_segments_batch_kernel<<<
        dim3(blocks, static_cast<unsigned>(count)), threads, 0, stream>>>(
            context->crc32_descriptors,
            count,
            context->crc32_table,
            context->crc32_partials);
    return cudaGetLastError() == cudaSuccess;
}

const uint32_t *lossless_rans_cuda_crc32_partials_device(
        const lossless_rans_cuda_context *context) {
    return context ? context->crc32_partials : nullptr;
}

namespace {

static uint32_t crc32_matrix_times(
        const uint32_t *matrix,
        uint32_t vector) {
    uint32_t sum = 0;
    while (vector != 0) {
        const unsigned int bit = static_cast<unsigned int>(__builtin_ctz(vector));
        sum ^= matrix[bit];
        vector &= vector - 1;
    }
    return sum;
}

static void crc32_matrix_square(
        uint32_t *square,
        const uint32_t *matrix) {
    for (size_t index = 0; index < 32; ++index) {
        square[index] = crc32_matrix_times(matrix, matrix[index]);
    }
}

static void crc32_combine_operator(
        uint32_t *matrix,
        size_t second_bytes) {
    for (size_t bit = 0; bit < 32; ++bit) {
        matrix[bit] = uint32_t{1} << bit;
    }
    if (second_bytes == 0) return;

    uint32_t power[32] = {};
    power[0] = crc32_polynomial;
    uint32_t row = 1;
    for (size_t index = 1; index < 32; ++index) {
        power[index] = row;
        row <<= 1;
    }
    size_t zero_bits = second_bytes * 8;
    while (zero_bits != 0) {
        if (zero_bits & 1U) {
            uint32_t composed[32] = {};
            for (size_t bit = 0; bit < 32; ++bit) {
                composed[bit] = crc32_matrix_times(power, matrix[bit]);
            }
            std::copy(composed, composed + 32, matrix);
        }
        zero_bits >>= 1;
        if (zero_bits == 0) break;
        uint32_t squared[32] = {};
        crc32_matrix_square(squared, power);
        std::copy(squared, squared + 32, power);
    }
}

static uint32_t crc32_combine(
        uint32_t first,
        uint32_t second,
        size_t second_bytes) {
    if (second_bytes == 0) return first;
    uint32_t matrix[32] = {};
    crc32_combine_operator(matrix, second_bytes);
    return crc32_matrix_times(matrix, first) ^ second;
}

} // namespace

uint32_t lossless_crc32_combine(
        uint32_t first,
        uint32_t second,
        size_t second_bytes) {
    return crc32_combine(first, second, second_bytes);
}

uint32_t lossless_crc32_combine_segments(
        const uint32_t *partials,
        size_t bytes) {
    if (!partials || bytes == 0) return 0;
    const size_t segment_bytes =
        (bytes + lossless_crc32_segments - 1) / lossless_crc32_segments;
    uint32_t full_segment_operator[32] = {};
    crc32_combine_operator(full_segment_operator, segment_bytes);
    uint32_t combined = partials[0];
    for (size_t segment = 1;
            segment < lossless_crc32_segments;
            ++segment) {
        const size_t begin = segment * segment_bytes;
        if (begin >= bytes) break;
        const size_t length = std::min(segment_bytes, bytes - begin);
        if (length == segment_bytes) {
            combined = crc32_matrix_times(
                full_segment_operator, combined) ^ partials[segment];
        } else {
            combined = crc32_combine(
                combined, partials[segment], length);
        }
    }
    return combined;
}

size_t lossless_rans_cuda_capacity(
        const lossless_rans_cuda_context *context) {
    return context ? context->capacity : 0;
}

} // namespace ggml_moe_common
