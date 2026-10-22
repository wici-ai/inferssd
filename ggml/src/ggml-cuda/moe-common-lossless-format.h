#pragma once

#include "moe-common-lossless-rans.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ggml_moe_common {

constexpr uint32_t lossless_sidecar_version_v1 = 1;
constexpr uint32_t lossless_sidecar_version = 2;
constexpr uint32_t lossless_sidecar_version_mxfp4 = 3;
constexpr uint32_t lossless_sidecar_alignment = 4096;
constexpr uint32_t lossless_sidecar_q40_layout = 1;
constexpr uint32_t lossless_sidecar_mxfp4_layout = 2;
constexpr uint32_t lossless_sidecar_index_entry_bytes = 512;
constexpr uint32_t lossless_sidecar_role_count = 3;

enum class lossless_role_codec : uint32_t {
    raw = 0,
    rans_q40 = 1,
    rans_mxfp4 = 2,
    compact_mxfp4_scale3 = 3,
    nvcomp_mxfp4_rank = 4,
};

struct lossless_role_record {
    lossless_role_codec codec = lossless_role_codec::raw;
    lossless_q40_scale_layout scale_layout =
        lossless_q40_scale_layout::repeated_per_row;
    uint32_t rows = 0;
    uint32_t blocks_per_row = 0;
    uint32_t chunks = 0;
    uint64_t payload_offset = 0;
    uint64_t payload_bytes = 0;
    uint64_t raw_offset = 0;
    uint64_t raw_bytes = 0;
    uint64_t chunk_index_offset = 0;
    uint64_t chunk_index_bytes = 0;
    uint64_t scales_offset = 0;
    uint64_t scales_bytes = 0;
    lossless_q40_scale_codec scale_codec = lossless_q40_scale_codec::raw;
    uint32_t scale_records = 0;
    uint64_t scale_raw_bytes = 0;
    uint32_t raw_crc32 = 0;
    std::array<uint8_t, 32> raw_sha256{};
};

struct lossless_object_record {
    int32_t layer = -1;
    int32_t expert = -1;
    uint64_t file_offset = 0;
    uint64_t physical_bytes = 0;
    uint64_t payload_bytes = 0;
    uint64_t raw_bytes = 0;
    uint32_t payload_crc32 = 0;
    uint32_t raw_crc32 = 0;
    std::array<lossless_role_record, lossless_sidecar_role_count> roles{};
};

struct lossless_sidecar {
    int fd_buffered = -1;
    int fd_direct = -1;
    std::string path;
    uint64_t file_bytes = 0;
    uint64_t source_pack_bytes = 0;
    uint64_t max_payload_bytes = 0;
    uint64_t max_physical_bytes = 0;
    uint32_t version = 0;
    uint32_t layout = 0;
    uint32_t chunks_per_role = 0;
    uint32_t table_count = 0;
    uint32_t quant_context_mode = 0;
    uint32_t nvcomp_chunk_bytes = 0;
    std::array<uint8_t, 32> source_header_sha256{};
    std::array<uint8_t, 32> source_manifest_sha256{};
    std::array<uint8_t, 32> table_sha256{};
    std::array<lossless_rans_table, lossless_rans_max_table_count> tables{};
    std::array<uint8_t, lossless_mxfp4_rank_remap_bytes>
        mxfp4_rank_to_symbol{};
    std::vector<lossless_object_record> objects;
    std::unordered_map<uint64_t, size_t> object_by_key;
};

struct lossless_chunk_record {
    uint32_t record_offset = 0;
    uint32_t record_bytes = 0;
    uint32_t quant_byte_offset = 0;
    uint32_t packed_bytes = 0;
};

struct lossless_scale_record {
    uint32_t record_offset = 0;
    uint32_t record_bytes = 0;
};

bool lossless_sidecar_load(
        lossless_sidecar &sidecar,
        const char *path,
        uint64_t expected_source_pack_bytes = 0,
        const uint8_t *expected_source_header_sha256 = nullptr,
        const uint8_t *expected_source_manifest_sha256 = nullptr);

void lossless_sidecar_close(lossless_sidecar &sidecar);

const lossless_object_record *lossless_sidecar_find(
        const lossless_sidecar &sidecar,
        int32_t layer,
        int32_t expert);

bool lossless_sidecar_parse_chunks(
        const lossless_role_record &role,
        const uint8_t *object_payload,
        size_t object_payload_bytes,
        lossless_chunk_record *chunks,
        size_t chunk_capacity);

bool lossless_sidecar_parse_scale_records(
        const lossless_role_record &role,
        const uint8_t *object_payload,
        size_t object_payload_bytes,
        lossless_scale_record *records,
        size_t record_capacity);

uint32_t lossless_crc32(const void *data, size_t bytes);

} // namespace ggml_moe_common
