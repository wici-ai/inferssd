#pragma once
// Per-index owned parser records. Production publishes only a context-owned
// immutable source_index; no singleton or backend registration lives here.
#include "../platform/file.h"
#include <cstdint>
#include <cstdio>
#include <vector>

namespace c0 {
struct expert_pack_entry {
    char tensor[128] = {};
    int32_t expert_idx = -1;
    int32_t source_idx = 0;
    int32_t layer_idx = -1;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
    uint64_t object_offset = 0;
    uint64_t object_nbytes = 0;
    uint64_t object_relative_offset = 0;
    bool alias_source = false;
};

struct expert_pack_source {
    FILE *file = nullptr; // buffered metadata (platform::open_stream)
    int fd_direct = -1;   // page-cache bypassing payload reads (platform::open_direct)
    void *mmap_base = nullptr;
    size_t mmap_size = 0;
    bool mmap_attempted = false;
    bool mmap_enabled = false;
    bool page_cache = false;
    bool direct_required = false;
    uint64_t file_size = 0;
    char path[512] = {};
};

struct parser_state {
    std::vector<expert_pack_source> sources;
    std::vector<expert_pack_entry> entries;
    FILE * file = nullptr; // borrowed first-source aliases used by legacy CUDA
    int fd_direct = -1;
    bool inited = false;
    bool enabled = false;
    int io_backend = 0;
    uint64_t page_cache_sources = 0;
    parser_state() = default;
    parser_state(const parser_state &) = delete;
    parser_state & operator=(const parser_state &) = delete;
    ~parser_state() { reset(); }
    void reset() {
        for (auto & source : sources) {
            if (source.mmap_base && source.mmap_size) { ssd::platform::unmap_view(source.mmap_base, source.mmap_size); }
            if (source.file) { std::fclose(source.file); }
            ssd::platform::close_fd(source.fd_direct);
        }
        sources.clear(); entries.clear(); file = nullptr; fd_direct = -1;
        inited = false; enabled = false; io_backend = 0; page_cache_sources = 0;
    }
};

} // namespace c0
