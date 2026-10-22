#include "mac-expert-pack-cache.h"

#if defined(__APPLE__) || defined(GGML_MAC_EXPERT_CACHE_TESTING)

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr size_t alignment = 4096;
constexpr uint64_t max_entries = 1000000;

#pragma pack(push, 1)
struct pack_header {
    char magic[16];
    uint32_t version;
    uint32_t header_size;
    uint64_t entry_count;
    uint64_t data_start;
};
struct pack_entry {
    char name[128];
    int32_t expert;
    uint32_t reserved;
    uint64_t offset;
    uint64_t bytes;
};
#pragma pack(pop)
static_assert(sizeof(pack_header) == 40);
static_assert(sizeof(pack_entry) == 152);

struct source_entry {
    int fd;
    uint64_t offset;
    size_t bytes;
    size_t read_bytes;
};

struct source_file {
    int fd;
    uint64_t bytes;
    explicit source_file(const char * path) : fd(open(path, O_RDONLY)) {
        if (fd < 0) { throw std::runtime_error("cannot open expert source: " + std::string(std::strerror(errno))); }
        struct stat info {};
        if (fstat(fd, &info) != 0 || info.st_size < 0
#if defined(__APPLE__)
                || fcntl(fd, F_NOCACHE, 1) == -1
#endif
                ) {
            const std::string reason = std::strerror(errno);
            close(fd);
            throw std::runtime_error("cannot prepare expert source: " + reason);
        }
        bytes = uint64_t(info.st_size);
    }
    ~source_file() { close(fd); }
    source_file(const source_file &) = delete;
    source_file & operator=(const source_file &) = delete;
};

static uint64_t parse_uint(const std::string & text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error("invalid unsigned integer in expert alias index");
    }
    size_t used = 0;
    const auto value = std::stoull(text, &used);
    if (used != text.size()) { throw std::runtime_error("invalid expert alias integer"); }
    return value;
}

struct key {
    std::string name;
    int32_t expert;
    bool operator==(const key & other) const {
        return expert == other.expert && name == other.name;
    }
};
struct key_hash {
    size_t operator()(const key & value) const {
        return std::hash<std::string>{}(value.name) ^ (size_t(uint32_t(value.expert)) * 0x9e3779b97f4a7c15ULL);
    }
};
struct slot {
    void * data = nullptr;
    size_t allocated = 0;
    uint64_t last_use = 0;
    unsigned pins = 0;
    bool admitted = false;
    slot() = default;
    slot(const slot &) = delete;
    slot & operator=(const slot &) = delete;
    slot(slot && other) noexcept : data(other.data), allocated(other.allocated),
            last_use(other.last_use), pins(other.pins), admitted(other.admitted) { other.data = nullptr; }
    slot & operator=(slot &&) = delete;
    ~slot() { free(data); }
};

static size_t round_up(size_t value) {
    if (value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        throw std::runtime_error("expert slice length overflows alignment");
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

static void pread_exact(int fd, void * data, size_t bytes, uint64_t offset) {
    auto * target = static_cast<char *>(data);
    while (bytes) {
        const size_t chunk = std::min(bytes, size_t(64) << 20);
        const ssize_t got = pread(fd, target, chunk, static_cast<off_t>(offset));
        if (got < 0 && errno == EINTR) { continue; }
        if (got <= 0) { throw std::runtime_error("short/failed expert-pack read: " + std::string(std::strerror(errno))); }
        target += got;
        offset += uint64_t(got);
        bytes -= size_t(got);
    }
}

class pack_cache {
public:
    explicit pack_cache(const char * path, bool alias) {
        if (const char * value = std::getenv("SSD_LLAMA_MAC_EXPERT_TRACE")) { trace_path = value; }
        if (alias) { load_alias(path); } else { load_pack(path); }
        const char * budget_env = std::getenv("SSD_LLAMA_MAC_EXPERT_CACHE_MIB");
        char * end = nullptr;
        errno = 0;
        unsigned long long mib = budget_env ? std::strtoull(budget_env, &end, 10) : 4096;
        if ((budget_env && (errno || end == budget_env || *end)) || mib > 16384) {
            throw std::runtime_error("SSD_LLAMA_MAC_EXPERT_CACHE_MIB must be 0..16384");
        }
        budget = size_t(mib) << 20;
        if (const char * value = std::getenv("SSD_LLAMA_MAC_EXPERT_CACHE_POLICY")) { policy = value; }
        if (policy != "lfu" && policy != "lru") {
            throw std::runtime_error("SSD_LLAMA_MAC_EXPERT_CACHE_POLICY must be lfu or lru");
        }
        std::fprintf(stderr, "[mac_expert_pack] path=%s entries=%zu cache_budget_mib=%llu cache_policy=%s source=%s F_NOCACHE=1\n",
                     path, index.size(), mib, policy.c_str(), alias ? "gguf-alias" : "pack");
    }

    ~pack_cache() {
        log_stats("shutdown");
        if (!trace_path.empty()) {
            if (FILE * output = std::fopen(trace_path.c_str(), "w")) {
                for (const auto & item : access_trace) {
                    std::fprintf(output, "%s %d\n", item.name.c_str(), item.expert);
                }
                std::fclose(output);
            }
        }
        free(scratch_data);
    }

    const void * acquire(const char * name, int32_t expert, size_t logical_bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        const key id { name, expert };
        if (!trace_path.empty()) { access_trace.push_back(id); }
        const auto source = index.find(id);
        if (source == index.end() || source->second.bytes != logical_bytes) {
            throw std::runtime_error("expert pack missing/mismatched tensor " + id.name + "[" + std::to_string(expert) + "]");
        }
        auto & frequency = heat[id];
        if (frequency < UINT16_MAX) { ++frequency; }
        if ((clock & 16383) == 0 && clock != 0) {
            for (auto & item : heat) { item.second >>= 1; }
        }
        auto hit = slots.find(id);
        if (hit != slots.end()) {
            ++hits;
            hit->second.pins++;
            hit->second.last_use = ++clock;
            maybe_log_stats();
            return hit->second.data;
        }
        ++misses;
        const size_t allocation = round_up(logical_bytes);
        bool admit = budget >= allocation && !tensor_names.empty();
        const size_t per_tensor_slots = admit ? budget / tensor_names.size() / allocation : 0;
        admit = admit && per_tensor_slots > 0;
        const bool use_lru = policy == "lru";
        size_t same_tensor = 0;
        auto same_tensor_victim = slots.end();
        for (auto it = slots.begin(); it != slots.end(); ++it) {
            if (it->first.name != id.name || !it->second.admitted) { continue; }
            ++same_tensor;
            if (it->second.pins != 0) { continue; }
            if (same_tensor_victim == slots.end() ||
                    (use_lru ? it->second.last_use < same_tensor_victim->second.last_use :
                    heat[it->first] < heat[same_tensor_victim->first] ||
                    (heat[it->first] == heat[same_tensor_victim->first] &&
                     it->second.last_use < same_tensor_victim->second.last_use))) {
                same_tensor_victim = it;
            }
        }
        if (admit && same_tensor >= per_tensor_slots) {
            if (same_tensor_victim == slots.end() || (!use_lru && frequency <= heat[same_tensor_victim->first])) {
                admit = false;
            } else {
                allocated_bytes -= same_tensor_victim->second.allocated;
                slots.erase(same_tensor_victim);
                ++evictions;
            }
        }
        while (admit && allocated_bytes + allocation > budget && !slots.empty()) {
            auto victim = slots.end();
            for (auto it = slots.begin(); it != slots.end(); ++it) {
                if (it->second.pins == 0 && (victim == slots.end() ||
                        heat[it->first] < heat[victim->first] ||
                        (heat[it->first] == heat[victim->first] && it->second.last_use < victim->second.last_use))) {
                    victim = it;
                }
            }
            if (victim == slots.end()) { admit = false; break; }
            allocated_bytes -= victim->second.allocated;
            slots.erase(victim);
            ++evictions;
        }
        slot fresh;
        if (!admit && scratch_data && scratch_allocated != allocation) {
            free(scratch_data);
            scratch_data = nullptr;
        }
        if (!admit && scratch_data) {
            fresh.data = scratch_data;
            scratch_data = nullptr;
            ++scratch_reuses;
        } else if (posix_memalign(&fresh.data, alignment, allocation) != 0) {
            throw std::runtime_error("cannot allocate expert-pack read buffer");
        }
        fresh.allocated = allocation;
        fresh.admitted = admit;
        const auto read_start = std::chrono::steady_clock::now();
        pread_exact(source->second.fd, fresh.data, source->second.read_bytes, source->second.offset);
        read_us += std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - read_start).count();
        ++read_requests;
        read_bytes += source->second.read_bytes;
        fresh.pins = 1;
        fresh.last_use = ++clock;
        auto inserted = slots.emplace(id, std::move(fresh));
        allocated_bytes += allocation;
        peak_bytes = std::max(peak_bytes, allocated_bytes);
        maybe_log_stats();
        return inserted.first->second.data;
    }

    void release(const void * data) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto it = slots.begin(); it != slots.end(); ++it) {
            if (it->second.data != data) { continue; }
            if (it->second.pins == 0) { throw std::runtime_error("expert-pack slice released twice"); }
            --it->second.pins;
            if (!it->second.admitted && it->second.pins == 0) {
                allocated_bytes -= it->second.allocated;
                if (!scratch_data) {
                    scratch_data = it->second.data;
                    scratch_allocated = it->second.allocated;
                    it->second.data = nullptr;
                }
                slots.erase(it);
            }
            return;
        }
        throw std::runtime_error("unknown expert-pack slice released");
    }

private:
    void load_pack(const char * path) {
        sources.emplace_back(std::make_unique<source_file>(path));
        const auto & file = *sources.back();
        if (file.bytes < sizeof(pack_header)) { throw std::runtime_error("truncated expert pack"); }
        pack_header header {};
        pread_exact(file.fd, &header, sizeof(header), 0);
        if (std::memcmp(header.magic, "GGMLMOEPACKv1\0\0\0", 16) != 0 ||
                header.version != 1 || header.header_size != sizeof(header) ||
                header.entry_count == 0 || header.entry_count > max_entries ||
                header.data_start < sizeof(header) || header.data_start > file.bytes ||
                header.data_start % alignment != 0 ||
                header.entry_count > (header.data_start - sizeof(header)) / sizeof(pack_entry)) {
            throw std::runtime_error("invalid v1 expert-pack header");
        }
        index.reserve(size_t(header.entry_count));
        for (uint64_t i = 0; i < header.entry_count; ++i) {
            pack_entry entry {};
            pread_exact(file.fd, &entry, sizeof(entry), sizeof(header) + i * sizeof(entry));
            if (!std::memchr(entry.name, '\0', sizeof(entry.name)) || entry.expert < 0 || entry.reserved != 0 ||
                    entry.bytes == 0 || entry.bytes > SIZE_MAX || entry.offset < header.data_start ||
                    entry.offset % alignment != 0 || entry.offset > file.bytes ||
                    round_up(size_t(entry.bytes)) > file.bytes - entry.offset) {
                throw std::runtime_error("invalid expert-pack index entry " + std::to_string(i));
            }
            add_entry(entry.name, entry.expert, {file.fd, entry.offset, size_t(entry.bytes), round_up(size_t(entry.bytes))});
        }
    }

    void add_entry(const std::string & name, int32_t expert, source_entry entry) {
        if (!index.emplace(key{name, expert}, entry).second) {
            throw std::runtime_error("duplicate expert-source index entry");
        }
        tensor_names.insert(name);
    }

    void load_alias(const char * path) {
        std::ifstream input(path);
        if (!input) { throw std::runtime_error("cannot open expert alias index"); }
        std::unordered_map<std::string, source_file *> opened;
        std::string line;
        if (!std::getline(input, line)) { throw std::runtime_error("empty expert alias index"); }
        const std::string header = "tensor\texpert_idx\tnbytes\tsource_path\toffset";
        if (!line.empty() && line.back() == '\r') { line.pop_back(); }
        if (line != header && line.compare(0, header.size() + 1, header + "\t") != 0) {
            throw std::runtime_error("invalid expert alias header");
        }
        while (std::getline(input, line)) {
            if (!line.empty() && line.back() == '\r') { line.pop_back(); }
            if (line.empty()) { throw std::runtime_error("empty expert alias row"); }
            if (index.size() >= max_entries || line.size() > 65536) { throw std::runtime_error("expert alias index too large"); }
            std::istringstream row(line);
            std::string name, expert_text, bytes_text, file_path, offset_text;
            if (!std::getline(row, name, '\t') || !std::getline(row, expert_text, '\t') ||
                    !std::getline(row, bytes_text, '\t') || !std::getline(row, file_path, '\t') ||
                    !std::getline(row, offset_text, '\t') || name.empty() || name.size() >= 128 ||
                    file_path.empty() || file_path[0] != '/') {
                throw std::runtime_error("invalid expert alias row (source paths must be absolute)");
            }
            const uint64_t expert = parse_uint(expert_text), bytes = parse_uint(bytes_text), offset = parse_uint(offset_text);
            if (expert > INT32_MAX || bytes == 0 || bytes > SIZE_MAX || offset > uint64_t(INT64_MAX)) {
                throw std::runtime_error("expert alias entry is out of range");
            }
            auto found = opened.find(file_path);
            if (found == opened.end()) {
                sources.emplace_back(std::make_unique<source_file>(file_path.c_str()));
                found = opened.emplace(file_path, sources.back().get()).first;
            }
            const auto & file = *found->second;
            if (offset > file.bytes || bytes > file.bytes - offset) {
                throw std::runtime_error("expert alias slice exceeds source file");
            }
            // GGUF slices need not be page aligned. Unlike padded packs, read exactly
            // the logical extent; in particular, never read beyond the final tensor.
            add_entry(name, int32_t(expert), {file.fd, offset, size_t(bytes), size_t(bytes)});
        }
        if (!input.eof() || index.empty()) { throw std::runtime_error("invalid expert alias index"); }
    }

    void maybe_log_stats() const {
        if ((hits + misses) % 1024 == 0) { log_stats("periodic"); }
    }

    void log_stats(const char * reason) const {
        const uint64_t calls = hits + misses;
        std::fprintf(stderr,
                "[mac_expert_pack] reason=%s requests=%llu hits=%llu misses=%llu hit_rate=%.4f"
                " read_requests=%llu read_bytes=%llu read_wall_ms=%.1f evictions=%llu scratch_reuses=%llu"
                " cache_bytes=%zu peak_cache_bytes=%zu\n",
                reason, static_cast<unsigned long long>(calls),
                static_cast<unsigned long long>(hits), static_cast<unsigned long long>(misses),
                calls ? double(hits) / double(calls) : 0.0,
                static_cast<unsigned long long>(read_requests), static_cast<unsigned long long>(read_bytes),
                read_us / 1000.0, static_cast<unsigned long long>(evictions),
                static_cast<unsigned long long>(scratch_reuses),
                allocated_bytes, peak_bytes);
    }

    std::vector<std::unique_ptr<source_file>> sources;
    size_t budget = 0;
    size_t allocated_bytes = 0;
    size_t peak_bytes = 0;
    uint64_t clock = 0, hits = 0, misses = 0, read_requests = 0, read_bytes = 0, evictions = 0, scratch_reuses = 0;
    double read_us = 0.0;
    std::mutex mutex;
    std::unordered_map<key, source_entry, key_hash> index;
    std::unordered_set<std::string> tensor_names;
    std::unordered_map<key, uint16_t, key_hash> heat;
    std::unordered_map<key, slot, key_hash> slots;
    std::string trace_path;
    std::string policy = "lfu";
    std::vector<key> access_trace;
    void * scratch_data = nullptr;
    size_t scratch_allocated = 0;
};

static pack_cache * instance() {
    static std::unique_ptr<pack_cache> cache = []() -> std::unique_ptr<pack_cache> {
        const char * path = std::getenv("SSD_LLAMA_MAC_EXPERT_PACK");
        const char * alias = std::getenv("GGML_MOE_EXPERT_GGUF_ALIAS_TSV");
        const bool has_pack = path && *path, has_alias = alias && *alias;
        if (has_pack && has_alias) { throw std::runtime_error("select one expert source: pack or GGUF alias"); }
        if (!has_pack && !has_alias) { return nullptr; }
        return std::make_unique<pack_cache>(has_alias ? alias : path, has_alias);
    }();
    return cache.get();
}

} // namespace

extern "C" bool ggml_mac_expert_pack_enabled(void) {
    const char * path = std::getenv("SSD_LLAMA_MAC_EXPERT_PACK");
    const char * alias = std::getenv("GGML_MOE_EXPERT_GGUF_ALIAS_TSV");
    return (path && *path) || (alias && *alias);
}

extern "C" const void * ggml_mac_expert_pack_acquire(const char * name, int32_t expert, size_t bytes) {
    try { return instance()->acquire(name, expert, bytes); }
    catch (const std::exception & e) {
        std::fprintf(stderr, "[mac_expert_pack] fatal: %s\n", e.what());
        std::abort();
    }
}

extern "C" void ggml_mac_expert_pack_release(const void * data) {
    try { instance()->release(data); }
    catch (const std::exception & e) {
        std::fprintf(stderr, "[mac_expert_pack] fatal: %s\n", e.what());
        std::abort();
    }
}

#endif
