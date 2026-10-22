#pragma once
// Experimental immutable store. The linked C0 option is disabled by default.
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace c0 {
#include "types.inc"

struct description {
    std::vector<std::string> packs;
    std::vector<std::string> aliases;
    std::vector<std::string> page_cache_packs;
    bool replace_duplicates = false;
    int io_backend = 0;
};

// The transport statistics, mmap state, CUDA state and liburing rings do not
// participate in parser behavior. Source handles have one owner here.
struct parser_state {
    std::vector<expert_pack_source> sources;
    std::vector<expert_pack_entry> entries;
    bool inited = false;
    bool enabled = false;
    int io_backend = 0;
    uint64_t page_cache_sources = 0;
    parser_state() = default;
    parser_state(const parser_state &) = delete;
    parser_state & operator=(const parser_state &) = delete;
    ~parser_state() {
        for (auto & source : sources) {
            if (source.file) { std::fclose(source.file); }
            if (source.fd_direct >= 0) { ::close(source.fd_direct); }
        }
    }
};

struct source_identity {
    uint64_t device = 0;
    uint64_t inode = 0;
    uint64_t size = 0;
    bool operator==(const source_identity & other) const {
        return device == other.device && inode == other.inode && size == other.size;
    }
};

inline source_identity identity(const expert_pack_source & source) {
    struct stat info = {};
    if (!source.file || ::fstat(::fileno(source.file), &info) || info.st_size < 0) { return {}; }
    return {uint64_t(info.st_dev), uint64_t(info.st_ino), uint64_t(info.st_size)};
}

struct extent {
    int source_id = -1;
    uint64_t offset = 0;
    uint64_t logical_bytes = 0;
    bool alias = false;
    source_identity file;
};

class expert_index {
public:
    static std::unique_ptr<expert_index> open(const description & desc);
    // False means an absent key or a logical range beyond the opened file.
    // There is no transport padding here. No tensor type is inferred: pack v1
    // does not store it; the caller's tensor context supplies that information.
    bool locate(const char * name, int expert, size_t bytes, extent & result) const {
        if (!state_->enabled || !name || !name[0] || expert < 0 || !bytes) { return false; }
        size_t lo = 0;
        size_t hi = state_->entries.size();
        while (lo < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            const auto & item = state_->entries[mid];
            const int name_cmp = std::strcmp(item.tensor, name);
            const int cmp = name_cmp ? name_cmp : item.expert_idx != expert ?
                (item.expert_idx < expert ? -1 : 1) : item.nbytes != bytes ? (item.nbytes < bytes ? -1 : 1) : 0;
            if (cmp < 0) { lo = mid + 1; } else { hi = mid; }
        }
        if (lo == state_->entries.size()) { return false; }
        const auto & item = state_->entries[lo];
        if (std::strcmp(item.tensor, name) || item.expert_idx != expert || item.nbytes != bytes ||
            item.source_idx < 0 || size_t(item.source_idx) >= state_->sources.size()) { return false; }
        const auto & source = state_->sources[size_t(item.source_idx)];
        if (item.offset > source.file_size || bytes > source.file_size - item.offset) { return false; }
        result = {item.source_idx, item.offset, item.nbytes, item.alias_source, identity(source)};
        return true;
    }
    // Borrowed descriptor, valid until this index is destroyed; never close it.
    // This experiment exposes buffered handles for file-identity tests. The
    // production raw-direct adapter must separately require O_DIRECT on Linux.
    int borrowed_fd(int source_id) const {
        if (source_id < 0 || size_t(source_id) >= state_->sources.size()) { return -1; }
        return ::fileno(state_->sources[size_t(source_id)].file);
    }
    int borrowed_direct_fd(int source_id) const {
        if (source_id < 0 || size_t(source_id) >= state_->sources.size()) { return -1; }
        return state_->sources[size_t(source_id)].fd_direct;
    }
    size_t source_count() const { return state_->sources.size(); }
    const std::vector<expert_pack_entry> & entries_for_test() const { return state_->entries; }
    source_identity source_identity_for_test(size_t i) const { return identity(state_->sources.at(i)); }
private:
    explicit expert_index(std::unique_ptr<parser_state> state) : state_(std::move(state)) {}
    std::unique_ptr<const parser_state> state_;
};

namespace detail {
class builder {
public:
    builder() : owned_(std::make_unique<parser_state>()), g_expert_pack(*owned_) {}
    std::unique_ptr<parser_state> build(const description & desc) {
        g_expert_pack.io_backend = desc.io_backend;
        std::vector<expert_pack_entry> entries;
        for (const auto & path : desc.packs) {
            if (!expert_pack_load_source(path.c_str(), int32_t(g_expert_pack.sources.size()), entries, false)) { return {}; }
        }
        std::unordered_map<std::string, int32_t> aliases;
        for (const auto & path : desc.aliases) {
            if (!expert_pack_load_gguf_alias_tsv(path.c_str(), entries, aliases)) { return {}; }
        }
        for (const auto & path : desc.page_cache_packs) {
            if (!expert_pack_load_source(path.c_str(), int32_t(g_expert_pack.sources.size()), entries, true)) { return {}; }
        }
        if (entries.empty()) { return {}; }
        finalize(entries, desc.replace_duplicates || !desc.page_cache_packs.empty());
        if (!g_expert_pack.enabled) { return {}; }
        g_expert_pack.entries = std::move(entries);
        return std::move(owned_);
    }
private:
    std::unique_ptr<parser_state> owned_;
    parser_state & g_expert_pack;
#include "member-parsers.inc"
    void finalize(std::vector<expert_pack_entry> & entries, bool replace_duplicates) {
#include "finalize.inc"
        g_expert_pack.enabled = true;
        g_expert_pack.inited = true;
    }
};
} // namespace detail

inline std::unique_ptr<expert_index> expert_index::open(const description & desc) {
    detail::builder build;
    auto state = build.build(desc);
    return state ? std::unique_ptr<expert_index>(new expert_index(std::move(state))) : nullptr;
}
} // namespace c0
