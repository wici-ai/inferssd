#include "mac-expert-source.h"
#include "mac-runtime.h"
#include "mac-runtime-fixture.h"
#include "../../ggml-ssd/runtime-services.h"
#include "../../ggml-ssd/settings/settings.h"
#include "../../ggml-ssd/settings/json.h"
#include "../../ggml-ssd/telemetry/telemetry.h"
#include "io/pread-nocache.h"

#if defined(__APPLE__) || defined(GGML_MAC_EXPERT_CACHE_TESTING)

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <thread>
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

using ssd::io::source_file;
using ssd::io::pread_exact;

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
    size_t prefix = 0;
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
    size_t prefix = 0;
    uint64_t last_use = 0;
    unsigned pins = 0;
    bool admitted = false;
    slot() = default;
    slot(const slot &) = delete;
    slot & operator=(const slot &) = delete;
    slot(slot && other) noexcept : data(other.data), allocated(other.allocated), prefix(other.prefix),
            last_use(other.last_use), pins(other.pins), admitted(other.admitted) { other.data = nullptr; }
    slot & operator=(slot &&) = delete;
    ~slot() { free(data); }
    const void * payload() const { return static_cast<const char *>(data) + prefix; }
};

static size_t round_up(size_t value) {
    if (value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        throw std::runtime_error("expert slice length overflows alignment");
    }
    return (value + alignment - 1) & ~(alignment - 1);
}



struct pending_read {
    key id;
    source_entry source;
    slot buffer;
    std::promise<void> completion;
    std::future<void> ready = completion.get_future();
};

struct expert_batch {
    const void * owner = nullptr;
    std::string name;
    size_t logical_bytes;
    std::vector<int32_t> experts;
    size_t next = 0;
    size_t demand = 0;
    size_t pending_bytes = 0;
    bool cache_updates = true;
    std::unordered_map<int32_t, std::shared_ptr<pending_read>> pending;
};

struct request_trace_entry {
    key id;
    uint64_t bytes, begin_ns;
    bool hit;
    bool cache_updates;
};

struct io_trace_entry {
    int fd;
    uint64_t offset, bytes, logical_bytes, begin_ns, duration_ns;
};

struct phase_trace_entry {
    uint64_t end_ns, requests;
    double acquire_us, read_wait_us, compute_us, barrier_us, lookup_us, policy_us;
    double allocation_us, release_us, lock_wait_us;
};

struct elapsed_counter {
    double & total;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~elapsed_counter() {
        total += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    }
};

// Published once with its owning cache. Source handles and lookup geometry
// never change while workers may read them; no process-global registry exists.
class source_index {
public:
    source_index(const std::vector<std::string> & paths, size_t requested_alignment)
        : read_alignment(requested_alignment) {
        if (read_alignment != 1 && read_alignment != 4096 && read_alignment != 16384) {
            throw std::runtime_error("invalid explicit Mac source alignment");
        }
        for (const auto & path : paths) {
            source_file probe(path.c_str());
            char magic[16] {};
            if (probe.bytes >= sizeof(magic)) { pread_exact(probe.fd, magic, sizeof(magic), 0); }
            if (std::memcmp(magic, "GGMLMOEPACKv1\0\0\0", sizeof(magic)) == 0) {
                load_pack(path.c_str());
            } else {
                load_alias(path.c_str());
            }
        }
    }
    std::vector<std::unique_ptr<source_file>> sources;
    std::unordered_map<key, source_entry, key_hash> index;
    std::unordered_set<std::string> tensor_names;
private:
    const size_t read_alignment;
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
        struct stat info {};
        if (stat(path, &info) != 0 || !S_ISREG(info.st_mode)) {
            throw std::runtime_error("expert alias index must be a regular file");
        }
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
            // Include surrounding source bytes only for I/O alignment. The kernel
            // receives the original logical slice; the last read stops at EOF.
            const uint64_t read_offset = offset / read_alignment * read_alignment;
            const uint64_t read_end = std::min((offset + bytes + read_alignment - 1) / read_alignment * read_alignment, file.bytes);
            add_entry(name, int32_t(expert), {file.fd, read_offset, size_t(bytes), size_t(read_end - read_offset), size_t(offset - read_offset)});
        }
        if (!input.eof() || index.empty()) { throw std::runtime_error("invalid expert alias index"); }
    }

};

struct cache_options {
    size_t budget = size_t(4096) << 20;
    size_t read_ahead = 2;
    size_t read_alignment = 4096;
    bool lru = false;
    bool prefill = true;
    std::string fixture_trace;
    ssd::telemetry::writer * events = nullptr;
    ssd::telemetry::debug_channels debug;
#ifdef GGML_MAC_EXPERT_CACHE_TESTING
    std::shared_future<void> read_gate;
#endif
};

class pack_cache {
public:
    explicit pack_cache(const std::vector<std::string> & paths, cache_options options)
        : source(paths, options.read_alignment), sources(source.sources), index(source.index),
          tensor_names(source.tensor_names), budget(options.budget), read_ahead(options.read_ahead),
          read_alignment(options.read_alignment), trace_path(std::move(options.fixture_trace)),
          policy(options.lru ? "lru" : "lfu"), prefill(options.prefill),
          events(options.events), debug(std::move(options.debug)) {
        if (read_ahead > 4 || (read_alignment != 1 && read_alignment != 4096 && read_alignment != 16384)) {
            throw std::runtime_error("invalid explicit Mac reader window/alignment");
        }
#ifdef GGML_MAC_EXPERT_CACHE_TESTING
        fixture_gate = options.read_gate;
#endif
        try {
            for (size_t i = 0; i < read_ahead; ++i) { readers.emplace_back([this] { reader_main(); }); }
        } catch (...) {
            stop_readers();
            throw;
        }
        std::fprintf(stderr, "[mac_expert_pack] sources=%zu entries=%zu cache_budget_mib=%llu cache_policy=%s read_ahead=%zu read_alignment=%zu F_NOCACHE=1\n",
                     paths.size(), index.size(), (unsigned long long)(budget >> 20), policy.c_str(), read_ahead, read_alignment);
    }

    bool prefill_enabled() const noexcept { return prefill; }

    ~pack_cache() {
        stop_readers();
        log_stats("shutdown");
        if (publication_failed.load(std::memory_order_acquire) &&
                !terminal_failed.load(std::memory_order_acquire)) {
            // A shutdown-only publication failure still cannot report success.
            // stop_readers() above has already joined every publisher/reader.
            std::fprintf(stderr, "[mac_expert_pack] fatal: telemetry failed after reader drain\n");
            std::abort();
        }
        if (!trace_path.empty()) {
            if (FILE * output = std::fopen(trace_path.c_str(), "w")) {
                for (const auto & item : access_trace) {
                    std::fprintf(output, "%s %d\n", item.id.name.c_str(), item.id.expert);
                }
                std::fclose(output);
            }
            if (FILE * output = std::fopen((trace_path + ".requests.tsv").c_str(), "w")) {
                std::fprintf(output, "tensor\texpert\tlogical_bytes\tbegin_ns\thit\tcache_updates\n");
                for (const auto & item : access_trace) {
                    std::fprintf(output, "%s\t%d\t%llu\t%llu\t%d\t%d\n", item.id.name.c_str(), item.id.expert,
                            (unsigned long long)item.bytes, (unsigned long long)item.begin_ns, int(item.hit), int(item.cache_updates));
                }
                std::fclose(output);
            }
            if (FILE * output = std::fopen((trace_path + ".reads.tsv").c_str(), "w")) {
                std::fprintf(output, "source_path\toffset\tread_bytes\tlogical_bytes\tbegin_ns\tduration_ns\n");
                std::unordered_map<int, std::string> paths;
                for (const auto & file : sources) { paths.emplace(file->fd, file->path); }
                for (const auto & item : io_trace) {
                    std::fprintf(output, "%s\t%llu\t%llu\t%llu\t%llu\t%llu\n", paths.at(item.fd).c_str(),
                            (unsigned long long)item.offset, (unsigned long long)item.bytes,
                            (unsigned long long)item.logical_bytes, (unsigned long long)item.begin_ns,
                            (unsigned long long)item.duration_ns);
                }
                std::fclose(output);
            }
            if (FILE * output = std::fopen((trace_path + ".phases.tsv").c_str(), "w")) {
                std::fprintf(output, "end_ns\trequests\tacquire_us\tread_wait_us\tcompute_us\tbarrier_us\tlookup_us\tpolicy_us\tallocation_us\trelease_us\tlock_wait_us\n");
                for (const auto & item : phase_trace) {
                    std::fprintf(output, "%llu\t%llu\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
                            (unsigned long long)item.end_ns, (unsigned long long)item.requests,
                            item.acquire_us, item.read_wait_us, item.compute_us, item.barrier_us,
                            item.lookup_us, item.policy_us, item.allocation_us, item.release_us, item.lock_wait_us);
                }
                std::fclose(output);
            }
        }
        free(scratch_data);
    }

    const void * acquire(const char * name, int32_t expert, size_t logical_bytes, expert_batch * batch = nullptr) {
        if (batch && batch->owner != this) { throw std::runtime_error("foreign Mac cache batch"); }
        const auto lock_start = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(mutex);
        lock_wait_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - lock_start).count();
        check_publication_locked();
        elapsed_counter elapsed{acquire_us};
        const auto lookup_start = std::chrono::steady_clock::now();
        if (batch && (batch->name != name || batch->logical_bytes != logical_bytes ||
                batch->demand >= batch->experts.size() || batch->experts[batch->demand++] != expert)) {
            throw std::runtime_error("expert batch acquired out of order");
        }
        const key id { name, expert };
        const auto source = index.find(id);
        if (source == index.end() || source->second.bytes != logical_bytes) {
            throw std::runtime_error("expert pack missing/mismatched tensor " + id.name + "[" + std::to_string(expert) + "]");
        }
        const bool cache_updates = !batch || batch->cache_updates;
        uint16_t frequency = 0;
        if (cache_updates) {
            auto & current = heat[id];
            if (current < UINT16_MAX) { ++current; }
            if ((clock & 16383) == 0 && clock != 0) {
                for (auto & item : heat) { item.second >>= 1; }
            }
            frequency = current;
        }
        auto hit = slots.find(id);
        if (debug.trace) {
            publish("ssd.mac.expert_request", [&] { return "{\"tensor\":" + ssd::json::quote(id.name) +
                    ",\"expert\":" + std::to_string(expert) + ",\"bytes\":" + std::to_string(logical_bytes) +
                    ",\"hit\":" + (hit != slots.end() ? "true" : "false") +
                    ",\"cache_updates\":" + (cache_updates ? "true" : "false") + "}"; });
        }
        if (!trace_path.empty()) {
            access_trace.push_back({id, logical_bytes,
                    uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(lookup_start.time_since_epoch()).count()),
                    hit != slots.end(), cache_updates});
        }
        check_publication_locked();
        lookup_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - lookup_start).count();
        if (hit != slots.end()) {
            ++hits;
            hit->second.pins++;
            if (cache_updates) { hit->second.last_use = ++clock; }
            if (batch) {
                discard_pending(*batch, expert);
                schedule(*batch);
            }
            maybe_log_stats();
            check_publication_locked();
            return hit->second.payload();
        }
        ++misses;
        const auto policy_start = std::chrono::steady_clock::now();
        const size_t allocation = round_up(source->second.read_bytes);
        bool admit = cache_updates && budget >= allocation && !tensor_names.empty();
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
        policy_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - policy_start).count();
        slot fresh;
        auto staged = batch ? batch->pending.find(expert) : std::unordered_map<int32_t, std::shared_ptr<pending_read>>::iterator{};
        if (batch && staged != batch->pending.end()) {
            auto job = staged->second;
            {
                elapsed_counter wait{read_wait_us};
                job->ready.get();
                check_publication_locked();
            }
            fresh.data = job->buffer.data;
            job->buffer.data = nullptr;
            fresh.allocated = job->buffer.allocated;
            batch->pending_bytes -= fresh.allocated;
            batch->pending.erase(staged);
            ++prefetch_used;
        } else {
            allocate_buffer(fresh, allocation, !admit);
            elapsed_counter wait{read_wait_us};
            read_source(source->second, fresh.data);
            check_publication_locked();
        }
        fresh.allocated = allocation;
        fresh.prefix = source->second.prefix;
        fresh.admitted = admit;
        fresh.pins = 1;
        fresh.last_use = cache_updates ? ++clock : clock;
        auto inserted = slots.emplace(id, std::move(fresh));
        allocated_bytes += allocation;
        peak_bytes = std::max(peak_bytes, allocated_bytes);
        if (batch) { schedule(*batch); }
        maybe_log_stats();
        check_publication_locked();
        return inserted.first->second.payload();
    }

    void release(const void * data) {
        std::lock_guard<std::mutex> lock(mutex);
        elapsed_counter elapsed{release_us};
        for (auto it = slots.begin(); it != slots.end(); ++it) {
            if (it->second.payload() != data) { continue; }
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
            check_publication_locked();
            return;
        }
        throw std::runtime_error("unknown expert-pack slice released");
    }

    expert_batch * begin(const char * name, const int64_t * counts, int n, size_t bytes, bool cache_updates) {
        if (publication_failed.load(std::memory_order_acquire) || terminal_failed.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(mutex); check_publication_locked();
        }
        if (!read_ahead && cache_updates) { return nullptr; }
        auto batch = std::make_unique<expert_batch>();
        batch->owner = this;
        batch->name = name;
        batch->logical_bytes = bytes;
        batch->cache_updates = cache_updates;
        for (int i = 0; i < n; ++i) {
            if (counts[i] > 0) { batch->experts.push_back(i); }
        }
        std::lock_guard<std::mutex> lock(mutex);
        check_publication_locked();
        schedule(*batch);
        check_publication_locked();
        return batch.release();
    }

    void end(expert_batch * raw) {
        if (!raw) {
            if (publication_failed.load(std::memory_order_acquire) || terminal_failed.load(std::memory_order_acquire)) {
                std::lock_guard<std::mutex> lock(mutex); check_publication_locked();
            }
            return;
        }
        if (raw->owner != this) { throw std::runtime_error("foreign Mac cache batch"); }
        std::unique_ptr<expert_batch> batch(raw);
        std::lock_guard<std::mutex> lock(mutex);
        while (!batch->pending.empty()) { discard_pending(*batch, batch->pending.begin()->first); }
        check_publication_locked();
        if (batch->demand != batch->experts.size()) { throw std::runtime_error("incomplete expert batch"); }
    }

    void record_compute(uint64_t us, uint64_t barriers) {
        std::lock_guard<std::mutex> lock(mutex);
        compute_us += us;
        barrier_us += barriers;
        ++computed_experts;
        if (debug.profile) {
            publish("ssd.mac.compute", [&] { return "{\"compute_us\":" + std::to_string(us) +
                    ",\"barrier_us\":" + std::to_string(barriers) + "}"; });
        }
        if (!trace_path.empty()) {
            phase_trace.push_back({uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count()), hits + misses,
                    acquire_us, read_wait_us, double(compute_us), double(barrier_us), lookup_us, policy_us,
                    allocation_us, release_us, lock_wait_us});
        }
        check_publication_locked();
    }

    // Called by C ABI catches after the failing method has released its lock.
    // No failure path may abort while a reader still borrows a source/buffer.
    void drain_after_failure() noexcept {
        terminal_failed.store(true, std::memory_order_release);
        try {
            std::lock_guard<std::mutex> lock(mutex);
            stop_readers();
        } catch (...) {
            std::fprintf(stderr, "[mac_expert_pack] fatal: failed to join Mac readers\n");
            std::abort();
        }
    }
#ifdef GGML_MAC_EXPERT_CACHE_TESTING
    bool fixture_readers_joined() {
        std::lock_guard<std::mutex> lock(queue_mutex);
        return queue.empty() && std::all_of(readers.begin(), readers.end(), [](const std::thread & r) { return !r.joinable(); });
    }
    bool fixture_publication_failed() const { return publication_failed.load(std::memory_order_acquire); }
    uint64_t fixture_read_count() const { return read_requests.load(std::memory_order_acquire); }
    uint64_t fixture_submitted_reads() const { return prefetch_submitted; }
#endif
private:
    // Caller holds mutex; reader threads never take it and only record failure.
    void check_publication_locked() {
        if (!publication_failed.load(std::memory_order_acquire) && !terminal_failed.load(std::memory_order_acquire)) { return; }
        terminal_failed.store(true, std::memory_order_release);
        stop_readers();
        throw std::runtime_error(publication_failed.load(std::memory_order_acquire) ?
                "Mac telemetry publication failed after reader drain" : "Mac cache closed after reader drain");
    }

    void allocate_buffer(slot & buffer, size_t allocation, bool reuse_scratch) {
        elapsed_counter elapsed{allocation_us};
        if (reuse_scratch && scratch_data && scratch_allocated != allocation) {
            free(scratch_data);
            scratch_data = nullptr;
        }
        if (reuse_scratch && scratch_data) {
            buffer.data = scratch_data;
            scratch_data = nullptr;
            ++scratch_reuses;
        } else if (posix_memalign(&buffer.data, std::max(alignment, read_alignment), allocation) != 0) {
            throw std::runtime_error("cannot allocate expert read buffer");
        }
        buffer.allocated = allocation;
    }

    void read_source(const source_entry & source, void * data) {
        const auto start = std::chrono::steady_clock::now();
        pread_exact(source.fd, data, source.read_bytes, source.offset);
        const auto duration = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count());
        read_ns.fetch_add(duration, std::memory_order_relaxed);
        if (debug.trace || debug.profile) {
            publish("ssd.mac.read", [&] { return "{\"offset\":" + std::to_string(source.offset) +
                    ",\"bytes\":" + std::to_string(source.read_bytes) +
                    ",\"logical_bytes\":" + std::to_string(source.bytes) +
                    ",\"duration_ns\":" + std::to_string(duration) + "}"; });
        }
        if (!trace_path.empty()) {
            std::lock_guard<std::mutex> trace_lock(trace_mutex);
            io_trace.push_back({source.fd, source.offset, source.read_bytes, source.bytes,
                    uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count()), duration});
        }
        read_requests.fetch_add(1, std::memory_order_relaxed);
        read_bytes.fetch_add(source.read_bytes, std::memory_order_relaxed);
    }

    void reader_main() {
        for (;;) {
            std::shared_ptr<pending_read> job;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_ready.wait(lock, [this] { return stopping || !queue.empty(); });
                if (queue.empty()) { return; }
                job = std::move(queue.front());
                queue.pop_front();
            }
            try {
#ifdef GGML_MAC_EXPERT_CACHE_TESTING
                // Only deterministic fixture scheduling; the actual pread and writer stay real.
                while (fixture_gate.valid() && fixture_gate.wait_for(std::chrono::milliseconds(1)) != std::future_status::ready) {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    if (stopping) { break; }
                }
#endif
                read_source(job->source, job->buffer.data);
                job->completion.set_value();
            } catch (...) {
                job->completion.set_exception(std::current_exception());
            }
        }
    }

    void stop_readers() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stopping = true;
        }
        queue_ready.notify_all();
        for (auto & reader : readers) { if (reader.joinable()) { reader.join(); } }
    }

    void discard_pending(expert_batch & batch, int32_t expert) {
        const auto found = batch.pending.find(expert);
        if (found == batch.pending.end()) { return; }
        found->second->ready.get();
        batch.pending_bytes -= found->second->buffer.allocated;
        batch.pending.erase(found);
        ++prefetch_unused;
    }

    void schedule(expert_batch & batch) {
        // Only selected experts in the current op are read. Admission and heat
        // updates still happen in demand order, exactly as in synchronous mode.
        constexpr size_t staging_limit = size_t(64) << 20;
        while (batch.pending.size() < read_ahead && batch.next < batch.experts.size()) {
            const auto expert = batch.experts[batch.next++];
            const key id{batch.name, expert};
            if (slots.find(id) != slots.end()) { continue; }
            const auto source = index.find(id);
            if (source == index.end() || source->second.bytes != batch.logical_bytes) {
                throw std::runtime_error("expert batch missing/mismatched source");
            }
            const auto allocation = round_up(source->second.read_bytes);
            if (allocation > staging_limit - batch.pending_bytes) { continue; }
            auto job = std::make_shared<pending_read>();
            job->id = id;
            job->source = source->second;
            allocate_buffer(job->buffer, allocation, true);
            batch.pending_bytes += allocation;
            peak_staging_bytes = std::max(peak_staging_bytes, batch.pending_bytes);
            batch.pending.emplace(expert, job);
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                queue.push_back(std::move(job));
            }
            ++prefetch_submitted;
            queue_ready.notify_one();
        }
    }

    template<class Payload> void publish(const char * name, Payload payload) const noexcept {
        if (!events) { return; }
        try {
            if (!events->event(name, payload())) { publication_failed.store(true, std::memory_order_release); }
        } catch (...) { publication_failed.store(true, std::memory_order_release); }
    }

    void maybe_log_stats() const {
        if ((hits + misses) % 1024 == 0) { log_stats("periodic"); }
    }

    void log_stats(const char * reason) const {
        const uint64_t calls = hits + misses;
        publish("ssd.mac.cache", [&] { return "{\"reason\":" + ssd::json::quote(reason) +
                ",\"requests\":" + std::to_string(calls) + ",\"hits\":" + std::to_string(hits) +
                ",\"misses\":" + std::to_string(misses) + ",\"cache_bytes\":" + std::to_string(allocated_bytes) +
                ",\"budget_bytes\":" + std::to_string(budget) + ",\"read_bytes\":" + std::to_string(read_bytes.load()) +
                ",\"peak_staging_bytes\":" + std::to_string(peak_staging_bytes) + "}"; });
        std::fprintf(stderr,
                "[mac_expert_pack] reason=%s requests=%llu hits=%llu misses=%llu hit_rate=%.4f"
                " read_requests=%llu read_bytes=%llu read_call_ms=%.1f evictions=%llu scratch_reuses=%llu"
                " cache_bytes=%zu peak_cache_bytes=%zu acquire_ms=%.1f read_wait_ms=%.1f compute_ms=%.1f"
                " barrier_ms=%.1f lookup_ms=%.1f policy_ms=%.1f allocation_ms=%.1f release_ms=%.1f lock_wait_ms=%.1f"
                " computed_experts=%llu prefetch_submitted=%llu prefetch_used=%llu prefetch_unused=%llu"
                " peak_staging_bytes=%zu\n",
                reason, static_cast<unsigned long long>(calls),
                static_cast<unsigned long long>(hits), static_cast<unsigned long long>(misses),
                calls ? double(hits) / double(calls) : 0.0,
                static_cast<unsigned long long>(read_requests.load()), static_cast<unsigned long long>(read_bytes.load()),
                double(read_ns.load()) / 1000000.0, static_cast<unsigned long long>(evictions),
                static_cast<unsigned long long>(scratch_reuses),
                allocated_bytes, peak_bytes, acquire_us / 1000.0, read_wait_us / 1000.0, compute_us / 1000.0,
                double(barrier_us) / 1000.0, lookup_us / 1000.0, policy_us / 1000.0, allocation_us / 1000.0,
                release_us / 1000.0, lock_wait_us / 1000.0,
                static_cast<unsigned long long>(computed_experts), static_cast<unsigned long long>(prefetch_submitted),
                static_cast<unsigned long long>(prefetch_used), static_cast<unsigned long long>(prefetch_unused), peak_staging_bytes);
    }

    const source_index source;
    const std::vector<std::unique_ptr<source_file>> & sources;
    const std::unordered_map<key, source_entry, key_hash> & index;
    const std::unordered_set<std::string> & tensor_names;
    const size_t budget;
    size_t allocated_bytes = 0;
    size_t peak_bytes = 0;
    uint64_t clock = 0, hits = 0, misses = 0, evictions = 0, scratch_reuses = 0;
    std::atomic<uint64_t> read_requests{0}, read_bytes{0}, read_ns{0};
#ifdef GGML_MAC_EXPERT_CACHE_TESTING
    std::shared_future<void> fixture_gate;
#endif
    mutable std::atomic<bool> publication_failed{false};
    std::atomic<bool> terminal_failed{false};
    double acquire_us = 0.0, read_wait_us = 0.0, lookup_us = 0.0, policy_us = 0.0;
    double allocation_us = 0.0, release_us = 0.0, lock_wait_us = 0.0;
    uint64_t compute_us = 0, barrier_us = 0, computed_experts = 0;
    uint64_t prefetch_submitted = 0, prefetch_used = 0, prefetch_unused = 0;
    const size_t read_ahead;
    size_t peak_staging_bytes = 0;
    const size_t read_alignment;
    std::mutex queue_mutex;
    std::condition_variable queue_ready;
    std::deque<std::shared_ptr<pending_read>> queue;
    std::vector<std::thread> readers;
    bool stopping = false;
    std::mutex mutex;
    std::unordered_map<key, uint16_t, key_hash> heat;
    std::unordered_map<key, slot, key_hash> slots;
    std::string trace_path;
    const std::string policy;
    const bool prefill;
    ssd::telemetry::writer * const events;
    const ssd::telemetry::debug_channels debug;
    std::vector<request_trace_entry> access_trace;
    std::mutex trace_mutex;
    std::vector<io_trace_entry> io_trace;
    std::vector<phase_trace_entry> phase_trace;
    void * scratch_data = nullptr;
    size_t scratch_allocated = 0;
};

pack_cache & require_cache(void * handle);

} // namespace

struct ggml_ssd_mac_cache {
    std::unique_ptr<pack_cache> value;
};

namespace {
void error_text(char * error, size_t size, const char * text) noexcept {
    if (error && size) { std::snprintf(error, size, "%s", text); }
}
pack_cache & require_cache(void * handle) {
    if (!handle) { throw std::runtime_error("Mac cache handle is null"); }
    return *static_cast<ggml_ssd_mac_cache *>(handle)->value;
}
ggml_ssd_mac_cache * create_cache(const std::vector<std::string> & paths, cache_options options,
                                char * error, size_t size) noexcept {
    error_text(error, size, "");
    try {
        if (paths.empty()) { return nullptr; }
        auto result = std::make_unique<ggml_ssd_mac_cache>();
        result->value = std::make_unique<pack_cache>(paths, std::move(options));
        return result.release();
    } catch (const std::exception & e) { error_text(error, size, e.what()); }
      catch (...) { error_text(error, size, "unknown Mac cache construction failure"); }
    return nullptr;
}
[[noreturn]] void drain_and_fatal(void * handle, const char * message) noexcept {
    if (handle) { static_cast<ggml_ssd_mac_cache *>(handle)->value->drain_after_failure(); }
    std::fprintf(stderr, "[mac_expert_pack] fatal: %s\n", message);
    std::abort();
}
} // namespace

namespace ssd { namespace mac {
ggml_ssd_mac_cache * create_configured(const settings::snapshot * values, const runtime_services * services,
                                     char * error, size_t size) noexcept {
    error_text(error, size, "");
    try {
        if (!values || !values->sources() || !values->integer("host.expert_cache_mib")) {
            error_text(error, size, "Mac cache requires a complete settings snapshot"); return nullptr;
        }
        const auto mib = *values->integer("host.expert_cache_mib");
        if (mib > (std::numeric_limits<size_t>::max() >> 20)) {
            error_text(error, size, "Mac cache MiB exceeds size_t"); return nullptr;
        }
        cache_options options; options.budget = size_t(mib) << 20;
        if (services) {
            options.events = services->events;
            if (services->debug) { options.debug = *services->debug; }
        }
        return create_cache(*values->sources(), std::move(options), error, size);
    } catch (const std::exception & e) { error_text(error, size, e.what()); }
      catch (...) { error_text(error, size, "unknown Mac settings failure"); }
    return nullptr;
}
void destroy(ggml_ssd_mac_cache * cache) noexcept { delete cache; }
namespace testing {
ggml_ssd_mac_cache * create(const std::vector<std::string> & paths, const options & input,
                           char * error, size_t size) noexcept {
    try {
        cache_options options; options.budget = input.budget_bytes;
        options.read_ahead = input.read_ahead; options.read_alignment = input.read_alignment;
        options.lru = input.lru; options.prefill = input.prefill; options.fixture_trace = input.trace_path;
        return create_cache(paths, std::move(options), error, size);
    } catch (const std::exception & e) { error_text(error, size, e.what()); }
      catch (...) { error_text(error, size, "unknown Mac fixture failure"); }
    return nullptr;
}
} // namespace testing
} } // namespace ssd::mac

extern "C" bool ggml_mac_expert_pack_enabled(void * cache) { return cache != nullptr; }
extern "C" bool ggml_mac_expert_pack_prefill_enabled(void * cache) {
    return cache && static_cast<ggml_ssd_mac_cache *>(cache)->value->prefill_enabled();
}
extern "C" const void * ggml_mac_expert_pack_acquire(void * cache, const char * name, int32_t expert, size_t bytes) {
    try { return require_cache(cache).acquire(name, expert, bytes); }
    catch (const std::exception & e) { drain_and_fatal(cache, e.what()); } catch (...) { drain_and_fatal(cache, "unknown acquire failure"); }
}
extern "C" void * ggml_mac_expert_pack_begin_batch(void * cache, const char * name, const int64_t * counts, int n, size_t bytes, bool cache_updates) {
    try { return require_cache(cache).begin(name, counts, n, bytes, cache_updates); }
    catch (const std::exception & e) { drain_and_fatal(cache, e.what()); } catch (...) { drain_and_fatal(cache, "unknown begin failure"); }
}
extern "C" const void * ggml_mac_expert_pack_acquire_batch(void * cache, void * batch, const char * name, int32_t expert, size_t bytes) {
    try { return require_cache(cache).acquire(name, expert, bytes, static_cast<expert_batch *>(batch)); }
    catch (const std::exception & e) { drain_and_fatal(cache, e.what()); } catch (...) { drain_and_fatal(cache, "unknown batch acquire failure"); }
}
extern "C" void ggml_mac_expert_pack_end_batch(void * cache, void * batch) {
    try { require_cache(cache).end(static_cast<expert_batch *>(batch)); }
    catch (const std::exception & e) { drain_and_fatal(cache, e.what()); } catch (...) { drain_and_fatal(cache, "unknown end failure"); }
}
extern "C" void ggml_mac_expert_pack_record_compute(void * cache, uint64_t us, uint64_t barriers) {
    try { require_cache(cache).record_compute(us, barriers); }
    catch (const std::exception & e) { drain_and_fatal(cache, e.what()); } catch (...) { drain_and_fatal(cache, "unknown compute trace failure"); }
}
extern "C" void ggml_mac_expert_pack_release(void * cache, const void * data) {
    try { require_cache(cache).release(data); }
    catch (const std::exception & e) { drain_and_fatal(cache, e.what()); } catch (...) { drain_and_fatal(cache, "unknown release failure"); }
}

#endif
