#pragma once
#include "catalog.h"
#include <string>
#include <vector>

namespace c0 {
// Transport capability and accounting stay in the provider. Callbacks run
// synchronously at the original per-source open points, never after publication.
struct source_open_hooks {
    // Non-null only when that provider was built with io_uring support.
    void (*report_iouring_enabled)(const char * label, const char * path) = nullptr;
    void (*page_cache_source_opened)() = nullptr;
};

struct description {
    std::vector<std::string> packs;
    std::vector<std::string> aliases;
    std::vector<std::string> page_cache_packs;
    bool replace_duplicates = false;
    int io_backend = 0;
    bool require_direct = false;
    std::vector<std::string> ggufs = {}; // a model's GGUF files: their routed experts, indexed from their headers
    // Reads one of them (append_gguf_source, given by the source index): the GGUF reader is ggml's, and this parser
    // links without ggml.
    bool (*read_gguf)(parser_state &, const char *, std::vector<expert_pack_entry> &, bool, source_open_hooks) = nullptr;
};

// Append into the caller-owned parser state. Transport setup and immutable
// publication remain that explicit owner's responsibility.
bool open_data_source(parser_state & state, const char * path, const char * label,
        bool page_cache_source = false, bool direct_required = false, source_open_hooks hooks = {});
bool append_sources(parser_state & state, const description & desc,
        std::vector<expert_pack_entry> & entries, source_open_hooks hooks = {});
// Opens a model's GGUF file as a source and appends an entry for every expert of its routed expert tensors
// (blk.N.ffn_{up,gate,down}_exps.weight), at that expert's bytes in the file, read from the header with ggml's reader.
bool append_gguf_source(parser_state & state, const char * path, std::vector<expert_pack_entry> & entries,
        bool direct_required, source_open_hooks hooks = {});
bool finalize_entries(std::vector<expert_pack_entry> & entries, bool replace_duplicates);
} // namespace c0
