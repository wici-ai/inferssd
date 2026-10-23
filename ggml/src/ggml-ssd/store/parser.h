#pragma once
#include "catalog.h"
#include <string>
#include <vector>

namespace c0 {
struct description {
    std::vector<std::string> packs;
    std::vector<std::string> aliases;
    std::vector<std::string> page_cache_packs;
    bool replace_duplicates = false;
    int io_backend = 0;
    bool require_direct = false;
};

// Transport capability and accounting stay in the provider. Callbacks run
// synchronously at the original per-source open points, never after publication.
struct source_open_hooks {
    // Non-null only when that provider was built with io_uring support.
    void (*report_iouring_enabled)(const char * label, const char * path) = nullptr;
    void (*page_cache_source_opened)() = nullptr;
};

// Append into the caller-owned parser state. Transport setup and immutable
// publication remain that explicit owner's responsibility.
bool open_data_source(parser_state & state, const char * path, const char * label,
        bool page_cache_source = false, bool direct_required = false, source_open_hooks hooks = {});
bool append_sources(parser_state & state, const description & desc,
        std::vector<expert_pack_entry> & entries, source_open_hooks hooks = {});
bool finalize_entries(std::vector<expert_pack_entry> & entries, bool replace_duplicates);
} // namespace c0
