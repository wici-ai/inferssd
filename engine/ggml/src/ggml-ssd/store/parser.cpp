#include "parser.h"
#include "../platform/file.h"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace c0 {
namespace {
struct close_file {
    void operator()(FILE * file) const { std::fclose(file); }
};
// Opening a FIFO must not block startup. The platform layer validates the
// opened handle, so path replacement cannot bypass the regular-file check.
FILE * open_regular_file(const char * path) noexcept {
    return ssd::platform::open_stream(path);
}
// A source is borrowed by the catalog only after vector insertion succeeds.
struct pending_source_handles {
    expert_pack_source & source;
    bool transferred = false;
    ~pending_source_handles() {
        if (!transferred) {
            if (source.file) { std::fclose(source.file); }
            ssd::platform::close_fd(source.fd_direct);
        }
    }
};

class source_parser {
public:
    explicit source_parser(parser_state & state, source_open_hooks hooks = {}, bool require_direct = false)
        : g_expert_pack(state), hooks(hooks), force_direct(require_direct) {}
    parser_state & g_expert_pack;
    source_open_hooks hooks;
    const bool force_direct;
static int expert_pack_entry_cmp(const expert_pack_entry &e, const char *tensor_name, int expert_idx, size_t nbytes) {
    const int name_cmp = std::strcmp(e.tensor, tensor_name);
    if (name_cmp != 0) return name_cmp;
    if (e.expert_idx != expert_idx) return e.expert_idx < expert_idx ? -1 : 1;
    if (e.nbytes != nbytes) return e.nbytes < nbytes ? -1 : 1;
    return 0;
}

bool expert_pack_read_exact(FILE *file, void *dst, size_t sz) {
    char *out = (char *)dst;
    size_t done = 0;
    while (done < sz) {
        const size_t chunk = sz - done;
        const size_t got = std::fread(out + done, 1, chunk, file);
        if (got == 0) return false;
        done += got;
    }
    return true;
}

std::string expert_pack_key_string(const char *tensor_name, int expert_idx, uint64_t nbytes) {
    std::string key = tensor_name ? tensor_name : "";
    key.push_back('\n');
    key += std::to_string(expert_idx);
    key.push_back('\n');
    key += std::to_string((unsigned long long)nbytes);
    return key;
}

bool expert_pack_open_data_source(
        const char *path,
        const char *label,
        bool page_cache_source = false,
        bool direct_required = false) {
    if (!path || !path[0]) return false;
    direct_required = direct_required || force_direct;

    FILE *file = open_regular_file(path);
    if (!file) {
        std::fprintf(stderr, "[moe_stream_batch] %s: open failed: %s\n", label, path);
        return false;
    }

    expert_pack_source source;
    source.file = file;
    pending_source_handles pending{source};
    source.page_cache = page_cache_source;
    source.direct_required = direct_required;
    std::snprintf(source.path, sizeof(source.path), "%s", path);
    if (!ssd::platform::file_size(ssd::platform::stream_fd(file), source.file_size)) {
        std::fprintf(stderr, "[moe_stream_batch] %s: stat failed: %s\n", label, path);
        return false;
    }
    source.fd_direct = -1;
    if (!page_cache_source && (g_expert_pack.io_backend == 1 || g_expert_pack.io_backend == 2)) {
        source.fd_direct = ssd::platform::open_direct(path);
        if (source.fd_direct < 0) {
            if (direct_required) {
                std::fprintf(stderr, "[moe_stream_batch] %s: required direct open failed: %s\n", label, path);
                return false;
            }
            std::fprintf(stderr, "[moe_stream_batch] %s: direct open failed; using buffered reads: %s\n", label, path);
            g_expert_pack.io_backend = 0;
        } else if (g_expert_pack.io_backend == 2) {
            if (hooks.report_iouring_enabled) {
                hooks.report_iouring_enabled(label, path);
            } else {
                std::fprintf(stderr, "[moe_stream_batch] %s: io_uring requested but liburing headers are unavailable; %s\n",
                             label, direct_required ? "using direct fallback anyway" : "using direct fallback");
                g_expert_pack.io_backend = 1;
            }
        } else if (g_expert_pack.io_backend == 1) {
            std::fprintf(stderr, "[moe_stream_batch] %s: direct reads enabled: %s\n", label, path);
        }
    } else if (page_cache_source) {
        ++g_expert_pack.page_cache_sources;
        if (hooks.page_cache_source_opened) { hooks.page_cache_source_opened(); }
        std::fprintf(stderr, "[moe_stream_batch] %s: page-cache buffered source enabled: %s\n", label, path);
    }

    g_expert_pack.sources.push_back(source);
    pending.transferred = true;
    return true;
}

bool expert_pack_load_source(const char *path, int32_t source_idx, std::vector<expert_pack_entry> &entries, bool page_cache_source = false) {
    if (!path || !path[0]) return false;

    FILE *probe = open_regular_file(path);
    if (!probe) {
        std::fprintf(stderr, "[moe_stream_batch] expert pack: open failed: %s\n", path);
        return false;
    }

    char magic[16] = {};
    uint32_t version = 0;
    uint32_t header_size = 0;
    uint64_t n_entries = 0;
    uint64_t data_start = 0;
    if (!expert_pack_read_exact(probe, magic, sizeof(magic)) ||
            !expert_pack_read_exact(probe, &version, sizeof(version)) ||
            !expert_pack_read_exact(probe, &header_size, sizeof(header_size)) ||
            !expert_pack_read_exact(probe, &n_entries, sizeof(n_entries)) ||
            !expert_pack_read_exact(probe, &data_start, sizeof(data_start)) ||
            std::memcmp(magic, "GGMLMOEPACKv1", 13) != 0 ||
            version != 1 || header_size < 40 || data_start < header_size || n_entries > 10000000ULL) {
        std::fprintf(stderr, "[moe_stream_batch] expert pack: invalid header: %s\n", path);
        std::fclose(probe);
        return false;
    }
    std::fclose(probe);

    if ((size_t)source_idx != g_expert_pack.sources.size()) {
        std::fprintf(stderr, "[moe_stream_batch] expert pack: internal source index mismatch for %s\n", path);
        return false;
    }
    if (!expert_pack_open_data_source(path, page_cache_source ? "page-cache expert pack" : "expert pack", page_cache_source)) {
        return false;
    }
    expert_pack_source &source = g_expert_pack.sources.back();
    if (std::fseek(source.file, 40, SEEK_SET) != 0) {
        std::fprintf(stderr, "[moe_stream_batch] expert pack: seek index failed: %s\n", path);
        return false;
    }

    const size_t before = entries.size();
    entries.resize(before + (size_t)n_entries);
    for (uint64_t i = 0; i < n_entries; ++i) {
        expert_pack_entry &e = entries[before + (size_t)i];
        uint32_t reserved = 0;
        if (!expert_pack_read_exact(source.file, e.tensor, sizeof(e.tensor)) ||
                !expert_pack_read_exact(source.file, &e.expert_idx, sizeof(e.expert_idx)) ||
                !expert_pack_read_exact(source.file, &reserved, sizeof(reserved)) ||
                !expert_pack_read_exact(source.file, &e.offset, sizeof(e.offset)) ||
                !expert_pack_read_exact(source.file, &e.nbytes, sizeof(e.nbytes))) {
            std::fprintf(stderr, "[moe_stream_batch] expert pack: short index: %s\n", path);
            return false;
        }
        e.tensor[sizeof(e.tensor) - 1] = '\0';
        e.source_idx = source_idx;
    }

    std::fprintf(stderr, "[moe_stream_batch] expert pack: loaded %lu entries from %s\n",
                 (unsigned long)n_entries, path);
    return true;
}

void expert_pack_split_tsv_line(char *line, std::vector<char *> &fields) {
    fields.clear();
    char *p = line;
    while (p && *p) {
        fields.push_back(p);
        char *tab = std::strchr(p, '\t');
        if (!tab) break;
        *tab = '\0';
        p = tab + 1;
    }
    if (!fields.empty()) {
        char *last = fields.back();
        const size_t len = std::strlen(last);
        if (len > 0 && last[len - 1] == '\n') last[len - 1] = '\0';
        const size_t len2 = std::strlen(last);
        if (len2 > 0 && last[len2 - 1] == '\r') last[len2 - 1] = '\0';
    }
}

int expert_pack_tsv_column(const std::vector<char *> &header, const char *name) {
    for (size_t i = 0; i < header.size(); ++i) {
        if (std::strcmp(header[i], name) == 0) return (int)i;
    }
    return -1;
}

bool expert_pack_load_gguf_alias_tsv(const char *path, std::vector<expert_pack_entry> &entries,
                                            std::unordered_map<std::string, int32_t> &alias_sources) {
    if (!path || !path[0]) return true;
    FILE *file = open_regular_file(path);
    if (!file) {
        std::fprintf(stderr, "[moe_stream_batch] GGUF alias: open failed: %s\n", path);
        return false;
    }
    std::unique_ptr<FILE, close_file> input(file);

    std::unordered_set<std::string> existing_keys;
    existing_keys.reserve(entries.size() * 2 + 1024);
    for (const expert_pack_entry &e : entries) {
        existing_keys.insert(expert_pack_key_string(e.tensor, e.expert_idx, e.nbytes));
    }

    char line[4096];
    std::vector<char *> fields;
    if (!std::fgets(line, sizeof(line), file)) {
        std::fprintf(stderr, "[moe_stream_batch] GGUF alias: empty TSV: %s\n", path);
        return false;
    }
    expert_pack_split_tsv_line(line, fields);
    const int col_tensor = expert_pack_tsv_column(fields, "tensor");
    int col_expert = expert_pack_tsv_column(fields, "expert_idx");
    if (col_expert < 0) {
        col_expert = expert_pack_tsv_column(fields, "expert");
    }
    const int col_nbytes = expert_pack_tsv_column(fields, "nbytes");
    const int col_source = expert_pack_tsv_column(fields, "source_path");
    int col_offset = expert_pack_tsv_column(fields, "offset");
    if (col_offset < 0) {
        col_offset = expert_pack_tsv_column(fields, "model_offset");
    }
    if (col_offset < 0) {
        col_offset = expert_pack_tsv_column(fields, "offset_bytes");
    }
    if (col_tensor < 0 || col_expert < 0 || col_nbytes < 0 || col_source < 0 || col_offset < 0) {
        std::fprintf(stderr, "[moe_stream_batch] GGUF alias: required TSV columns missing: %s\n", path);
        return false;
    }

    uint64_t rows = 0;
    uint64_t loaded = 0;
    uint64_t skipped_duplicates = 0;
    uint64_t bad_rows = 0;
    while (std::fgets(line, sizeof(line), file)) {
        expert_pack_split_tsv_line(line, fields);
        ++rows;
        const int need_cols = std::max(std::max(col_tensor, col_expert), std::max(col_nbytes, std::max(col_source, col_offset))) + 1;
        if ((int)fields.size() < need_cols) {
            ++bad_rows;
            continue;
        }
        const char *tensor = fields[(size_t)col_tensor];
        const char *source_path = fields[(size_t)col_source];
        const int expert_idx = (int)std::strtol(fields[(size_t)col_expert], nullptr, 10);
        const uint64_t nbytes = (uint64_t)std::strtoull(fields[(size_t)col_nbytes], nullptr, 10);
        const uint64_t offset = (uint64_t)std::strtoull(fields[(size_t)col_offset], nullptr, 10);
        if (!tensor || !tensor[0] || !source_path || !source_path[0] ||
                expert_idx < 0 || nbytes == 0 || std::strlen(tensor) >= 128) {
            ++bad_rows;
            continue;
        }
        const std::string key = expert_pack_key_string(tensor, expert_idx, nbytes);
        if (existing_keys.find(key) != existing_keys.end()) {
            ++skipped_duplicates;
            continue;
        }

        int32_t source_idx = -1;
        auto it = alias_sources.find(source_path);
        if (it != alias_sources.end()) {
            source_idx = it->second;
        } else {
            source_idx = (int32_t)g_expert_pack.sources.size();
            if (!expert_pack_open_data_source(source_path, "GGUF alias", false,
                    force_direct)) {
                return false;
            }
            alias_sources[source_path] = source_idx;
        }

        expert_pack_entry e = {};
        std::snprintf(e.tensor, sizeof(e.tensor), "%s", tensor);
        e.expert_idx = expert_idx;
        e.source_idx = source_idx;
        e.offset = offset;
        e.nbytes = nbytes;
        e.alias_source = true;
        entries.push_back(e);
        existing_keys.insert(key);
        ++loaded;
    }
    input.reset();

    std::fprintf(stderr,
            "[moe_stream_batch] GGUF alias: loaded %lu entries from %s sources=%zu skipped_duplicates=%lu bad_rows=%lu rows=%lu\n",
            (unsigned long)loaded, path, alias_sources.size(),
            (unsigned long)skipped_duplicates, (unsigned long)bad_rows, (unsigned long)rows);
    return bad_rows == 0;
}
};
} // namespace

bool open_data_source(parser_state & state, const char * path, const char * label,
        bool page_cache_source, bool direct_required, source_open_hooks hooks) {
    return source_parser(state, hooks).expert_pack_open_data_source(path, label, page_cache_source, direct_required);
}

bool append_sources(parser_state & state, const description & desc,
        std::vector<expert_pack_entry> & entries, source_open_hooks hooks) {
    source_parser parser(state, hooks, desc.require_direct);
    for (const auto & path : desc.packs) {
        if (!parser.expert_pack_load_source(path.c_str(), int32_t(state.sources.size()), entries, false)) { return false; }
    }
    std::unordered_map<std::string, int32_t> aliases;
    for (const auto & path : desc.aliases) {
        if (!parser.expert_pack_load_gguf_alias_tsv(path.c_str(), entries, aliases)) { return false; }
    }
    for (const auto & path : desc.page_cache_packs) {
        if (!parser.expert_pack_load_source(path.c_str(), int32_t(state.sources.size()), entries, true)) { return false; }
    }
    return true;
}

bool finalize_entries(std::vector<expert_pack_entry> & entries, bool replace_duplicates) {
    std::sort(entries.begin(), entries.end(),
        [](const expert_pack_entry &a, const expert_pack_entry &b) {
            const int name_cmp = std::strcmp(a.tensor, b.tensor);
            if (name_cmp != 0) return name_cmp < 0;
            if (a.expert_idx != b.expert_idx) return a.expert_idx < b.expert_idx;
            if (a.nbytes != b.nbytes) return a.nbytes < b.nbytes;
            return a.source_idx < b.source_idx;
        });
    uint64_t replaced_duplicates = 0;
    if (replace_duplicates && !entries.empty()) {
        std::vector<expert_pack_entry> deduped;
        deduped.reserve(entries.size());
        for (const expert_pack_entry &entry : entries) {
            if (!deduped.empty() &&
                    source_parser::expert_pack_entry_cmp(deduped.back(), entry.tensor, entry.expert_idx, entry.nbytes) == 0) {
                if (entry.source_idx >= deduped.back().source_idx) {
                    deduped.back() = entry;
                }
                ++replaced_duplicates;
            } else {
                deduped.push_back(entry);
            }
        }
        entries = std::move(deduped);
        if (replaced_duplicates > 0) {
            std::fprintf(stderr,
                "[moe_stream_batch] expert pack: replaced %lu duplicate keys with later pack sources\n",
                (unsigned long)replaced_duplicates);
        }
    }
    for (size_t i = 1; i < entries.size(); ++i) {
        if (source_parser::expert_pack_entry_cmp(entries[i - 1], entries[i].tensor, entries[i].expert_idx, entries[i].nbytes) == 0) {
            std::fprintf(stderr, "[moe_stream_batch] expert pack: duplicate key across packs: %s expert=%d bytes=%lu\n",
                         entries[i].tensor, entries[i].expert_idx, (unsigned long)entries[i].nbytes);
            return false;
        }
    }

    return true;
}

} // namespace c0
