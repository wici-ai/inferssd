#include "moe-common-lossless-format.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <mutex>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ggml_moe_common {

namespace {

constexpr size_t header_bytes = 4096;
constexpr size_t encoded_header_bytes = 184;
constexpr size_t table_v1_prefix_bytes = 144;
constexpr size_t table_v2_prefix_bytes = 32;
constexpr size_t table_v2_entry_bytes =
    2 * lossless_rans_alphabet * sizeof(uint32_t) + lossless_rans_table_size;
constexpr size_t object_header_bytes = 56;
constexpr size_t role_entry_bytes = 144;
constexpr size_t encoded_role_bytes = 120;
constexpr size_t chunk_entry_bytes = 16;
constexpr uint32_t q40_block_bytes = 18;
constexpr uint32_t q40_scale_bytes = 2;
constexpr uint32_t mxfp4_block_bytes = 17;
constexpr uint32_t mxfp4_scale_bytes = 1;
constexpr uint32_t mxfp4_scale3_group_blocks = 8;
constexpr uint32_t mxfp4_scale3_group_bytes =
    mxfp4_scale3_group_blocks * 16 + 3;
constexpr char magic[] = "GGMLMOERANSv1";

static uint32_t load_u32_le(const uint8_t *src) {
    return static_cast<uint32_t>(src[0]) |
        (static_cast<uint32_t>(src[1]) << 8) |
        (static_cast<uint32_t>(src[2]) << 16) |
        (static_cast<uint32_t>(src[3]) << 24);
}

static int32_t load_i32_le(const uint8_t *src) {
    return static_cast<int32_t>(load_u32_le(src));
}

static uint64_t load_u64_le(const uint8_t *src) {
    uint64_t value = 0;
    for (int byte = 7; byte >= 0; --byte) {
        value = (value << 8) | src[byte];
    }
    return value;
}

static bool all_zero(const uint8_t *data, size_t bytes) {
    for (size_t index = 0; index < bytes; ++index) {
        if (data[index] != 0) return false;
    }
    return true;
}

static bool range_valid(uint64_t offset, uint64_t bytes, uint64_t limit) {
    return offset <= limit && bytes <= limit - offset;
}

#if !defined(_WIN32)
static bool pread_all(int fd, void *destination, size_t bytes, uint64_t offset) {
    auto *dst = static_cast<uint8_t *>(destination);
    size_t completed = 0;
    while (completed < bytes) {
        const ssize_t got = pread(
            fd, dst + completed, bytes - completed,
            static_cast<off_t>(offset + completed));
        if (got > 0) {
            completed += static_cast<size_t>(got);
            continue;
        }
        if (got < 0 && errno == EINTR) continue;
        return false;
    }
    return true;
}
#endif

static uint64_t object_key(int32_t layer, int32_t expert) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
        static_cast<uint32_t>(expert);
}

static bool parse_role(
        const uint8_t *source,
        uint32_t sidecar_version,
        uint32_t sidecar_layout,
        lossless_role_record &role) {
    const uint32_t codec = load_u32_le(source);
    const uint32_t scale_layout = load_u32_le(source + 4);
    if (codec > static_cast<uint32_t>(
            lossless_role_codec::nvcomp_mxfp4_rank) ||
            scale_layout > static_cast<uint32_t>(
                lossless_q40_scale_layout::raw_per_block)) {
        return false;
    }
    role.codec = static_cast<lossless_role_codec>(codec);
    role.scale_layout = static_cast<lossless_q40_scale_layout>(scale_layout);
    role.rows = load_u32_le(source + 8);
    role.blocks_per_row = load_u32_le(source + 12);
    role.chunks = load_u32_le(source + 16);
    if (load_u32_le(source + 20) != 0 || role.rows == 0 ||
            role.blocks_per_row == 0) {
        return false;
    }
    role.payload_offset = load_u64_le(source + 24);
    role.payload_bytes = load_u64_le(source + 32);
    role.raw_offset = load_u64_le(source + 40);
    role.raw_bytes = load_u64_le(source + 48);
    role.chunk_index_offset = load_u64_le(source + 56);
    role.chunk_index_bytes = load_u64_le(source + 64);
    role.scales_offset = load_u64_le(source + 72);
    role.scales_bytes = load_u64_le(source + 80);
    std::memcpy(role.raw_sha256.data(), source + 88, role.raw_sha256.size());
    if (sidecar_version == lossless_sidecar_version_v1) {
        if (!all_zero(source + encoded_role_bytes,
                role_entry_bytes - encoded_role_bytes)) {
            return false;
        }
    } else {
        const uint32_t scale_codec = load_u32_le(source + 120);
        if (scale_codec > static_cast<uint32_t>(
                lossless_q40_scale_codec::fixed3_e8m0_group8_exceptions) ||
                load_u32_le(source + 140) != 0) {
            return false;
        }
        role.scale_codec = static_cast<lossless_q40_scale_codec>(scale_codec);
        role.scale_records = load_u32_le(source + 124);
        role.scale_raw_bytes = load_u64_le(source + 128);
        role.raw_crc32 = load_u32_le(source + 136);
    }
    const uint64_t block_count =
        static_cast<uint64_t>(role.rows) * role.blocks_per_row;
    const bool q40 = sidecar_layout == lossless_sidecar_q40_layout;
    const bool mxfp4 = sidecar_layout == lossless_sidecar_mxfp4_layout;
    const uint32_t block_bytes = q40 ? q40_block_bytes : mxfp4_block_bytes;
    const uint32_t scale_bytes = q40 ? q40_scale_bytes : mxfp4_scale_bytes;
    if (role.blocks_per_row != 0 && block_count / role.blocks_per_row != role.rows ||
            (!q40 && !mxfp4) ||
            block_count > std::numeric_limits<uint64_t>::max() / block_bytes ||
            role.raw_bytes != block_count * block_bytes || role.payload_bytes == 0 ||
            (q40 && (role.codec == lossless_role_codec::rans_mxfp4 ||
                    role.codec == lossless_role_codec::compact_mxfp4_scale3)) ||
            (mxfp4 && role.codec != lossless_role_codec::rans_mxfp4 &&
                role.codec != lossless_role_codec::compact_mxfp4_scale3 &&
                role.codec != lossless_role_codec::nvcomp_mxfp4_rank)) {
        return false;
    }
    if (role.codec == lossless_role_codec::raw) {
        return role.chunks == 0 && role.chunk_index_offset == 0 &&
            role.chunk_index_bytes == 0 && role.scales_offset == 0 &&
            role.scales_bytes == 0 && role.payload_bytes == role.raw_bytes &&
            role.scale_codec == lossless_q40_scale_codec::raw &&
            role.scale_records == 0 && role.scale_raw_bytes == 0;
    }
    if (role.codec == lossless_role_codec::nvcomp_mxfp4_rank) {
        return mxfp4 && role.chunks == 1 &&
            role.scale_layout == lossless_q40_scale_layout::raw_per_block &&
            role.chunk_index_offset == 0 && role.chunk_index_bytes == 0 &&
            role.scales_offset == 0 && role.scales_bytes == 0 &&
            role.scale_codec == lossless_q40_scale_codec::raw &&
            role.scale_records == 0 && role.scale_raw_bytes == 0;
    }
    if (role.codec == lossless_role_codec::compact_mxfp4_scale3) {
        if (!mxfp4 ||
                role.scale_layout != lossless_q40_scale_layout::raw_per_block ||
                block_count % mxfp4_scale3_group_blocks != 0 ||
                block_count / mxfp4_scale3_group_blocks >
                    std::numeric_limits<uint64_t>::max() /
                        mxfp4_scale3_group_bytes) {
            return false;
        }
        const uint64_t groups = block_count / mxfp4_scale3_group_blocks;
        const bool grouped_layout =
            role.scales_offset == mxfp4_scale3_group_blocks * 16;
        const bool planar_layout =
            role.scales_offset == block_count * 16;
        const uint64_t packed_bytes = groups * mxfp4_scale3_group_bytes;
        if (role.chunks == 0 || (!grouped_layout && !planar_layout) ||
                role.scales_bytes != groups * 3 ||
                role.scale_raw_bytes != block_count) {
            return false;
        }
        if (role.scale_codec ==
                lossless_q40_scale_codec::fixed3_e8m0_group8) {
            return role.chunk_index_offset == 0 &&
                role.chunk_index_bytes == 0 &&
                role.payload_bytes == packed_bytes &&
                role.scale_records == 0;
        }
        const uint64_t exception_bytes =
            static_cast<uint64_t>(role.scale_records) * 8;
        const uint64_t exception_offset = planar_layout ?
            role.scales_offset + role.scales_bytes : packed_bytes;
        return role.scale_codec ==
                lossless_q40_scale_codec::fixed3_e8m0_group8_exceptions &&
            role.chunk_index_offset == exception_offset &&
            role.chunk_index_bytes == exception_bytes &&
            role.payload_bytes == packed_bytes + exception_bytes &&
            range_valid(role.chunk_index_offset,
                role.chunk_index_bytes, role.payload_bytes);
    }
    if (role.scale_layout != lossless_q40_scale_layout::repeated_per_row &&
            role.scale_layout != lossless_q40_scale_layout::raw_per_block) {
        return false;
    }
    const uint64_t expected_scale_bytes =
        role.scale_layout == lossless_q40_scale_layout::repeated_per_row ?
            static_cast<uint64_t>(role.rows) * scale_bytes :
            block_count * scale_bytes;
    if (role.chunks == 0 ||
            role.chunk_index_bytes !=
                static_cast<uint64_t>(role.chunks) * chunk_entry_bytes ||
            !range_valid(role.chunk_index_offset, role.chunk_index_bytes,
                role.payload_bytes) ||
            !range_valid(role.scales_offset, role.scales_bytes,
                role.payload_bytes)) {
        return false;
    }
    if (role.scale_codec == lossless_q40_scale_codec::raw) {
        return role.scale_records == 0 &&
            (role.scale_raw_bytes == 0 ||
                role.scale_raw_bytes == expected_scale_bytes) &&
            role.scales_bytes == expected_scale_bytes;
    }
    if (role.scale_codec == lossless_q40_scale_codec::rans_nibble_planes) {
        return q40 && sidecar_version >= lossless_sidecar_version &&
            role.scale_records >= lossless_rans_q40_scale_planes &&
            role.scale_records % lossless_rans_q40_scale_planes == 0 &&
            role.scale_records / lossless_rans_q40_scale_planes <= role.chunks &&
            role.scale_raw_bytes == expected_scale_bytes &&
            role.scales_bytes >
                static_cast<uint64_t>(role.scale_records) * sizeof(uint32_t);
    }
    if (role.scale_codec == lossless_q40_scale_codec::sparse_e8m0_binary) {
        return mxfp4 && sidecar_version >= lossless_sidecar_version_mxfp4 &&
            role.scale_records == 2 &&
            role.scale_raw_bytes == expected_scale_bytes &&
            role.scales_bytes > 2 * sizeof(uint32_t);
    }
    return mxfp4 && sidecar_version >= lossless_sidecar_version_mxfp4 &&
        role.scale_records >= lossless_rans_mxfp4_scale_planes &&
        role.scale_records % lossless_rans_mxfp4_scale_planes == 0 &&
        role.scale_records <= lossless_rans_mxfp4_max_scale_records &&
        role.scale_records / lossless_rans_mxfp4_scale_planes <=
            lossless_rans_mxfp4_max_chunks &&
        role.scale_raw_bytes == expected_scale_bytes &&
        role.scales_bytes >
            static_cast<uint64_t>(role.scale_records) * sizeof(uint32_t);
}

static bool validate_table(const lossless_rans_table &table) {
    uint32_t prefix = 0;
    for (uint32_t symbol = 0; symbol < lossless_rans_alphabet; ++symbol) {
        if (table.start[symbol] != prefix) return false;
        prefix += table.freq[symbol];
    }
    if (prefix != lossless_rans_table_size) return false;
    for (uint32_t index = 0; index < lossless_rans_table_size; ++index) {
        const uint32_t symbol = table.slot_to_symbol[index];
        if (symbol >= lossless_rans_alphabet ||
                index < table.start[symbol] ||
                index >= table.start[symbol] + table.freq[symbol]) {
            return false;
        }
    }
    return true;
}

} // namespace

void lossless_sidecar_close(lossless_sidecar &sidecar) {
#if !defined(_WIN32)
    if (sidecar.fd_direct >= 0) close(sidecar.fd_direct);
    if (sidecar.fd_buffered >= 0) close(sidecar.fd_buffered);
#endif
    sidecar = {};
}

bool lossless_sidecar_load(
        lossless_sidecar &sidecar,
        const char *path,
        uint64_t expected_source_pack_bytes,
        const uint8_t *expected_source_header_sha256,
        const uint8_t *expected_source_manifest_sha256) {
    lossless_sidecar_close(sidecar);
#if defined(_WIN32)
    (void)path;
    (void)expected_source_pack_bytes;
    (void)expected_source_header_sha256;
    (void)expected_source_manifest_sha256;
    return false;
#else
    if (!path || !*path) return false;
    sidecar.fd_buffered = open(path, O_RDONLY | O_CLOEXEC);
    if (sidecar.fd_buffered < 0) return false;
    struct stat stat_buffer {};
    if (fstat(sidecar.fd_buffered, &stat_buffer) != 0 || stat_buffer.st_size <= 0) {
        lossless_sidecar_close(sidecar);
        return false;
    }
    sidecar.file_bytes = static_cast<uint64_t>(stat_buffer.st_size);
    std::array<uint8_t, header_bytes> header{};
    if (!pread_all(sidecar.fd_buffered, header.data(), header.size(), 0) ||
            std::memcmp(header.data(), magic, sizeof(magic) - 1) != 0 ||
            !all_zero(header.data() + sizeof(magic) - 1, 16 - (sizeof(magic) - 1)) ||
            load_u32_le(header.data() + 20) != header_bytes ||
            load_u32_le(header.data() + 24) != lossless_sidecar_alignment ||
            !all_zero(header.data() + encoded_header_bytes,
                header.size() - encoded_header_bytes)) {
        lossless_sidecar_close(sidecar);
        return false;
    }
    sidecar.version = load_u32_le(header.data() + 16);
    sidecar.layout = load_u32_le(header.data() + 28);
    if (sidecar.version != lossless_sidecar_version_v1 &&
            sidecar.version != lossless_sidecar_version &&
            sidecar.version != lossless_sidecar_version_mxfp4) {
        lossless_sidecar_close(sidecar);
        return false;
    }
    if ((sidecar.version <= lossless_sidecar_version &&
            sidecar.layout != lossless_sidecar_q40_layout) ||
            (sidecar.version == lossless_sidecar_version_mxfp4 &&
             sidecar.layout != lossless_sidecar_mxfp4_layout)) {
        lossless_sidecar_close(sidecar);
        return false;
    }
    const uint64_t object_count = load_u64_le(header.data() + 32);
    const uint64_t table_offset = load_u64_le(header.data() + 40);
    const uint64_t table_bytes = load_u64_le(header.data() + 48);
    const uint64_t index_offset = load_u64_le(header.data() + 56);
    const uint64_t index_entry_bytes = load_u64_le(header.data() + 64);
    const uint64_t data_offset = load_u64_le(header.data() + 72);
    sidecar.source_pack_bytes = load_u64_le(header.data() + 80);
    std::memcpy(sidecar.source_header_sha256.data(), header.data() + 88, 32);
    std::memcpy(sidecar.source_manifest_sha256.data(), header.data() + 120, 32);
    std::memcpy(sidecar.table_sha256.data(), header.data() + 152, 32);
    if (object_count == 0 || object_count > 1000000 ||
            index_entry_bytes != lossless_sidecar_index_entry_bytes ||
            table_offset % lossless_sidecar_alignment != 0 ||
            table_bytes < table_v1_prefix_bytes + lossless_rans_table_size ||
            table_bytes > 1024 * 1024 ||
            index_offset < table_offset + table_bytes ||
            data_offset % lossless_sidecar_alignment != 0 ||
            index_offset > data_offset ||
            object_count * index_entry_bytes > data_offset - index_offset ||
            data_offset > sidecar.file_bytes ||
            !range_valid(table_offset, table_bytes, sidecar.file_bytes) ||
            object_count > std::numeric_limits<uint64_t>::max() / index_entry_bytes ||
            !range_valid(
                index_offset, object_count * index_entry_bytes,
                sidecar.file_bytes) ||
            (expected_source_pack_bytes != 0 &&
                sidecar.source_pack_bytes != expected_source_pack_bytes) ||
            (expected_source_header_sha256 &&
                std::memcmp(
                    sidecar.source_header_sha256.data(),
                    expected_source_header_sha256, 32) != 0) ||
            (expected_source_manifest_sha256 &&
                std::memcmp(
                    sidecar.source_manifest_sha256.data(),
                    expected_source_manifest_sha256, 32) != 0)) {
        lossless_sidecar_close(sidecar);
        return false;
    }

    std::vector<uint8_t> table_blob(table_bytes);
    if (!pread_all(
            sidecar.fd_buffered, table_blob.data(), table_blob.size(), table_offset) ||
            load_u32_le(table_blob.data()) != lossless_rans_scale_bits ||
            load_u32_le(table_blob.data() + 4) != lossless_rans_streams ||
            load_u32_le(table_blob.data() + 12) != lossless_rans_table_size) {
        lossless_sidecar_close(sidecar);
        return false;
    }
    sidecar.chunks_per_role = load_u32_le(table_blob.data() + 8);
    sidecar.quant_context_mode = sidecar.version == lossless_sidecar_version_mxfp4 ?
        load_u32_le(table_blob.data() + 24) : 0;
    sidecar.nvcomp_chunk_bytes =
        sidecar.version == lossless_sidecar_version_mxfp4 ?
            load_u32_le(table_blob.data() + 28) : 0;
    sidecar.table_count = sidecar.version == lossless_sidecar_version_v1 ?
        1 : load_u32_le(table_blob.data() + 16);
    const size_t table_prefix_bytes =
        sidecar.version == lossless_sidecar_version_v1 ?
            table_v1_prefix_bytes : table_v2_prefix_bytes;
    const size_t table_entry_bytes = table_v2_entry_bytes;
    const size_t required_table_bytes =
        sidecar.version == lossless_sidecar_version_v1 ?
            table_v1_prefix_bytes + lossless_rans_table_size :
            table_v2_prefix_bytes +
                static_cast<size_t>(sidecar.table_count) * table_v2_entry_bytes;
    if (sidecar.chunks_per_role == 0 || sidecar.table_count == 0 ||
            sidecar.table_count > lossless_rans_max_table_count ||
            (sidecar.version == lossless_sidecar_version &&
                (sidecar.table_count != lossless_rans_q40_table_count ||
                 load_u32_le(table_blob.data() + 20) != table_v2_entry_bytes ||
                 load_u32_le(table_blob.data() + 24) != 0 ||
                 load_u32_le(table_blob.data() + 28) != 0)) ||
            (sidecar.version == lossless_sidecar_version_mxfp4 &&
                (sidecar.table_count != lossless_rans_mxfp4_table_count ||
                 sidecar.chunks_per_role > lossless_rans_mxfp4_max_chunks ||
                 load_u32_le(table_blob.data() + 20) != table_v2_entry_bytes ||
                 sidecar.quant_context_mode > 3 ||
                 (sidecar.nvcomp_chunk_bytes != 0 &&
                  (sidecar.nvcomp_chunk_bytes < 16 * 1024 ||
                   sidecar.nvcomp_chunk_bytes > 16 * 1024 * 1024 ||
                   (sidecar.nvcomp_chunk_bytes &
                    (sidecar.nvcomp_chunk_bytes - 1)) != 0)))) ||
            required_table_bytes > table_blob.size()) {
        lossless_sidecar_close(sidecar);
        return false;
    }
    for (uint32_t table_index = 0;
            table_index < sidecar.table_count;
            ++table_index) {
        const uint8_t *entry = sidecar.version == lossless_sidecar_version_v1 ?
            table_blob.data() :
            table_blob.data() + table_prefix_bytes + table_index * table_entry_bytes;
        const size_t freq_offset = sidecar.version == lossless_sidecar_version_v1 ? 16 : 0;
        const size_t start_offset = sidecar.version == lossless_sidecar_version_v1 ? 80 : 64;
        const size_t slot_offset = sidecar.version == lossless_sidecar_version_v1 ?
            table_v1_prefix_bytes : 128;
        lossless_rans_table &table = sidecar.tables[table_index];
        for (uint32_t symbol = 0; symbol < lossless_rans_alphabet; ++symbol) {
            table.freq[symbol] = load_u32_le(entry + freq_offset + symbol * 4);
            table.start[symbol] = load_u32_le(entry + start_offset + symbol * 4);
        }
        std::memcpy(
            table.slot_to_symbol, entry + slot_offset,
            sizeof(table.slot_to_symbol));
        if (!validate_table(table)) {
            lossless_sidecar_close(sidecar);
            return false;
        }
    }
    size_t used_table_bytes = required_table_bytes;
    if (sidecar.version == lossless_sidecar_version_mxfp4 &&
            sidecar.quant_context_mode == 3) {
        if (lossless_mxfp4_rank_remap_bytes >
                table_blob.size() - used_table_bytes) {
            lossless_sidecar_close(sidecar);
            return false;
        }
        std::memcpy(
            sidecar.mxfp4_rank_to_symbol.data(),
            table_blob.data() + used_table_bytes,
            lossless_mxfp4_rank_remap_bytes);
        for (size_t context = 0;
                context < lossless_sidecar_role_count * 16;
                ++context) {
            uint32_t seen = 0;
            for (size_t rank = 0; rank < 16; ++rank) {
                const uint8_t symbol =
                    sidecar.mxfp4_rank_to_symbol[context * 16 + rank];
                if (symbol >= 16 || (seen & (1U << symbol)) != 0) {
                    lossless_sidecar_close(sidecar);
                    return false;
                }
                seen |= 1U << symbol;
            }
            if (seen != 0xffffU) {
                lossless_sidecar_close(sidecar);
                return false;
            }
        }
        used_table_bytes += lossless_mxfp4_rank_remap_bytes;
    }
    if (!all_zero(
            table_blob.data() + used_table_bytes,
            table_blob.size() - used_table_bytes)) {
        lossless_sidecar_close(sidecar);
        return false;
    }

    std::vector<uint8_t> index(object_count * index_entry_bytes);
    if (!pread_all(
            sidecar.fd_buffered, index.data(), index.size(), index_offset)) {
        lossless_sidecar_close(sidecar);
        return false;
    }
    sidecar.objects.reserve(static_cast<size_t>(object_count));
    sidecar.object_by_key.reserve(static_cast<size_t>(object_count));
    uint64_t previous_end = data_offset;
    for (uint64_t object_index = 0; object_index < object_count; ++object_index) {
        const uint8_t *entry = index.data() + object_index * index_entry_bytes;
        lossless_object_record object;
        object.layer = load_i32_le(entry);
        object.expert = load_i32_le(entry + 4);
        const uint32_t role_count = load_u32_le(entry + 8);
        const uint32_t reserved = load_u32_le(entry + 12);
        object.file_offset = load_u64_le(entry + 16);
        object.physical_bytes = load_u64_le(entry + 24);
        object.payload_bytes = load_u64_le(entry + 32);
        object.raw_bytes = load_u64_le(entry + 40);
        object.payload_crc32 = load_u32_le(entry + 48);
        object.raw_crc32 = sidecar.version == lossless_sidecar_version_mxfp4 ?
            load_u32_le(entry + 52) : 0;
        if (role_count != lossless_sidecar_role_count || reserved != 0 ||
                (sidecar.version != lossless_sidecar_version_mxfp4 &&
                 load_u32_le(entry + 52) != 0) ||
                object.layer < 0 || object.expert < 0 ||
                object.file_offset % lossless_sidecar_alignment != 0 ||
                object.physical_bytes == 0 ||
                object.physical_bytes % lossless_sidecar_alignment != 0 ||
                object.file_offset < previous_end ||
                !range_valid(
                    object.file_offset, object.physical_bytes,
                    sidecar.file_bytes) ||
                object.payload_bytes == 0 ||
                object.payload_bytes > object.physical_bytes) {
            lossless_sidecar_close(sidecar);
            return false;
        }
        uint64_t payload_covered = 0;
        uint64_t raw_covered = 0;
        for (size_t role_index = 0; role_index < object.roles.size(); ++role_index) {
            const uint8_t *role_entry =
                entry + object_header_bytes + role_index * role_entry_bytes;
            lossless_role_record &role = object.roles[role_index];
            if (!parse_role(role_entry, sidecar.version, sidecar.layout, role) ||
                    (sidecar.version == lossless_sidecar_version_mxfp4 &&
                     role.chunks != sidecar.chunks_per_role) ||
                    role.payload_offset != payload_covered ||
                    !range_valid(
                        role.raw_offset, role.raw_bytes,
                        sidecar.source_pack_bytes) ||
                    !range_valid(
                        role.payload_offset, role.payload_bytes,
                        object.payload_bytes)) {
                lossless_sidecar_close(sidecar);
                return false;
            }
            payload_covered += role.payload_bytes;
            raw_covered += role.raw_bytes;
        }
        if (payload_covered != object.payload_bytes ||
                raw_covered != object.raw_bytes ||
                !all_zero(
                    entry + object_header_bytes +
                        lossless_sidecar_role_count * role_entry_bytes,
                    index_entry_bytes - object_header_bytes -
                        lossless_sidecar_role_count * role_entry_bytes)) {
            lossless_sidecar_close(sidecar);
            return false;
        }
        const uint64_t key = object_key(object.layer, object.expert);
        if (!sidecar.object_by_key.emplace(
                key, sidecar.objects.size()).second) {
            lossless_sidecar_close(sidecar);
            return false;
        }
        sidecar.objects.push_back(object);
        sidecar.max_payload_bytes = std::max(
            sidecar.max_payload_bytes, object.payload_bytes);
        sidecar.max_physical_bytes = std::max(
            sidecar.max_physical_bytes, object.physical_bytes);
        previous_end = object.file_offset + object.physical_bytes;
    }
    sidecar.fd_direct = open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (sidecar.fd_direct < 0) {
        lossless_sidecar_close(sidecar);
        return false;
    }
    sidecar.path = path;
    return true;
#endif
}

const lossless_object_record *lossless_sidecar_find(
        const lossless_sidecar &sidecar,
        int32_t layer,
        int32_t expert) {
    const auto found = sidecar.object_by_key.find(object_key(layer, expert));
    if (found == sidecar.object_by_key.end() || found->second >= sidecar.objects.size()) {
        return nullptr;
    }
    return &sidecar.objects[found->second];
}

bool lossless_sidecar_parse_chunks(
        const lossless_role_record &role,
        const uint8_t *object_payload,
        size_t object_payload_bytes,
        lossless_chunk_record *chunks,
        size_t chunk_capacity) {
    if ((role.codec != lossless_role_codec::rans_q40 &&
            role.codec != lossless_role_codec::rans_mxfp4) || !object_payload ||
            !chunks || chunk_capacity < role.chunks ||
            !range_valid(role.payload_offset, role.payload_bytes, object_payload_bytes) ||
            !range_valid(
                role.chunk_index_offset, role.chunk_index_bytes,
                role.payload_bytes)) {
        return false;
    }
    const uint8_t *role_payload = object_payload + role.payload_offset;
    const uint8_t *index = role_payload + role.chunk_index_offset;
    uint64_t expected_quant_offset = 0;
    uint64_t expected_record_offset =
        role.chunk_index_offset + role.chunk_index_bytes;
    for (uint32_t chunk_index = 0; chunk_index < role.chunks; ++chunk_index) {
        const uint8_t *entry = index + chunk_index * chunk_entry_bytes;
        lossless_chunk_record &chunk = chunks[chunk_index];
        chunk.record_offset = load_u32_le(entry);
        chunk.record_bytes = load_u32_le(entry + 4);
        chunk.quant_byte_offset = load_u32_le(entry + 8);
        chunk.packed_bytes = load_u32_le(entry + 12);
        if (chunk.packed_bytes == 0 ||
                chunk.quant_byte_offset != expected_quant_offset ||
                chunk.record_offset != expected_record_offset ||
                !range_valid(
                    chunk.record_offset, chunk.record_bytes,
                    role.payload_bytes)) {
            return false;
        }
        expected_quant_offset += chunk.packed_bytes;
        expected_record_offset += chunk.record_bytes;
    }
    const uint64_t block_count =
        static_cast<uint64_t>(role.rows) * role.blocks_per_row;
    return expected_quant_offset == block_count * 16 &&
        expected_record_offset == role.scales_offset;
}

bool lossless_sidecar_parse_scale_records(
        const lossless_role_record &role,
        const uint8_t *object_payload,
        size_t object_payload_bytes,
        lossless_scale_record *records,
        size_t record_capacity) {
    const bool q40_scales = role.codec == lossless_role_codec::rans_q40 &&
        role.scale_codec == lossless_q40_scale_codec::rans_nibble_planes;
    const bool mxfp4_scales = role.codec == lossless_role_codec::rans_mxfp4 &&
        (role.scale_codec ==
            lossless_q40_scale_codec::rans_e8m0_nibble_planes ||
         role.scale_codec == lossless_q40_scale_codec::sparse_e8m0_binary);
    if ((!q40_scales && !mxfp4_scales) ||
            (q40_scales &&
             (role.scale_records < lossless_rans_q40_scale_planes ||
              role.scale_records % lossless_rans_q40_scale_planes != 0)) ||
            (mxfp4_scales &&
             (role.scale_records < lossless_rans_mxfp4_scale_planes ||
              role.scale_records % lossless_rans_mxfp4_scale_planes != 0 ||
              role.scale_records > lossless_rans_mxfp4_max_scale_records)) ||
            !object_payload || !records ||
            record_capacity < role.scale_records ||
            !range_valid(role.payload_offset, role.payload_bytes,
                object_payload_bytes) ||
            !range_valid(role.scales_offset, role.scales_bytes,
                role.payload_bytes) ||
            role.scale_records >
                std::numeric_limits<uint64_t>::max() / sizeof(uint32_t) ||
            role.scales_bytes <=
                static_cast<uint64_t>(role.scale_records) * sizeof(uint32_t)) {
        return false;
    }
    const uint8_t *role_payload = object_payload + role.payload_offset;
    const uint8_t *index = role_payload + role.scales_offset;
    const uint64_t index_bytes =
        static_cast<uint64_t>(role.scale_records) * sizeof(uint32_t);
    uint64_t offset = role.scales_offset + index_bytes;
    for (uint32_t record_index = 0;
            record_index < role.scale_records;
            ++record_index) {
        lossless_scale_record &record = records[record_index];
        record.record_offset = static_cast<uint32_t>(offset);
        record.record_bytes = load_u32_le(index + record_index * sizeof(uint32_t));
        if (record.record_bytes == 0 ||
                !range_valid(record.record_offset, record.record_bytes,
                    role.payload_bytes)) {
            return false;
        }
        offset += record.record_bytes;
    }
    return offset == role.scales_offset + role.scales_bytes;
}

uint32_t lossless_crc32(const void *data, size_t bytes) {
    static std::once_flag initialized;
    static uint32_t table[8][256];
    std::call_once(initialized, [] {
        for (uint32_t value = 0; value < 256; ++value) {
            uint32_t crc = value;
            for (int bit = 0; bit < 8; ++bit) {
                crc = (crc >> 1) ^ (0xEDB88320U &
                    (0U - (crc & 1U)));
            }
            table[0][value] = crc;
        }
        for (uint32_t value = 0; value < 256; ++value) {
            uint32_t crc = table[0][value];
            for (size_t slice = 1; slice < 8; ++slice) {
                crc = table[0][crc & 0xFFU] ^ (crc >> 8);
                table[slice][value] = crc;
            }
        }
    });
    uint32_t crc = 0xFFFFFFFFU;
    const auto *source = static_cast<const uint8_t *>(data);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    while (bytes >= 8) {
        uint32_t first = 0;
        uint32_t second = 0;
        std::memcpy(&first, source, sizeof(first));
        std::memcpy(&second, source + sizeof(first), sizeof(second));
        first ^= crc;
        crc =
            table[7][first & 0xFFU] ^
            table[6][(first >> 8) & 0xFFU] ^
            table[5][(first >> 16) & 0xFFU] ^
            table[4][first >> 24] ^
            table[3][second & 0xFFU] ^
            table[2][(second >> 8) & 0xFFU] ^
            table[1][(second >> 16) & 0xFFU] ^
            table[0][second >> 24];
        source += 8;
        bytes -= 8;
    }
#endif
    for (size_t index = 0; index < bytes; ++index) {
        crc = table[0][(crc ^ source[index]) & 0xFFU] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFU;
}

} // namespace ggml_moe_common
