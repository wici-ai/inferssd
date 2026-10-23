#pragma once
// Host-memory ledger: one arbiter for the DRAM of the machine.
//
// Every host allocation the engine makes on purpose asks for room first: RAM expert cache slots, server
// checkpoints and prompt-cache blobs, host compute buffers (and on a unified-memory GPU the device compute buffers,
// which come out of the same DRAM). Room for `extra` more bytes is the smaller of two terms.
//  - cgroup: the tightest finite memory.max among this process's cgroup and its ancestors, less the cgroup reserve
//    (GGML_KIMI_CPU_SINK_CACHE_RESERVE_BYTES, at least 512 MiB). It is what the RAM cache was always checked
//    against, and on a discrete GPU it is the term that binds.
//  - system: MemAvailable less the floor. On a unified-memory GPU (DGX Spark, Jetson) a device allocation comes out
//    of the same DRAM without being charged to the cgroup, so only this term sees the GPU tier, the KV cache, the
//    dense weights and pinned buffers.
// floor = file reserve + margin + explicit reserves:
//  - file reserve (unified memory only): MemAvailable counts the server's own page cache as available, but the
//    kernel evicts it before it can touch the tier's pinned blocks. Keep the file pages the server holds when the
//    ledger first measures them, plus one page for every page the kernel read back after evicting it (a refault
//    means the file working set did not fit). It only grows.
//  - margin: headroom for growth nobody accounts. It starts at the cgroup reserve; on unified memory it is at least
//    the tier's former GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB, learns that growth (end_interval) and only grows.
//    The cgroup term keeps its own reserve: a memory.max already leaves the deployment's headroom out, and growth
//    outside the cgroup (CUDA graphs and pools on unified memory) does not count against it.
//  - explicit reserves, deployment facts: GGML_KIMI_HOST_RESERVE_MIB for other processes and, on unified memory,
//    GGML_CUDA_MEMORY_RESERVE_MIB, which reserves the same DRAM.
// The GPU tier of a unified device receives the floor and keeps it, with the RAM cache's uncommitted slots, free.
//
// On unified memory MemAvailable also moves with nothing allocated: it keeps back the zones' low watermarks, and the
// kernel boosts them after fragmenting allocations by up to 1.5 times the high watermark (kimi-memory-boost.h).
// Whoever takes memory up to the floor (the tier's growth, an allocation outside the caches) counts MemAvailable as
// it reads at the full boost, so that the boost coming back cannot take MemAvailable under the floor.
//
// A prefill reads rows of weights by index straight from the model's file mapping (token embeddings, DeepSeek-V4.1's
// Engram tables): their pages come into the page cache while it runs, charged to the cgroup, and they must stay there
// until the layer that reads them has run. File pages count as room (CACHE_RECLAIMABLE_FILE, MemAvailable), so nothing
// kept room for them: in a full cgroup the kernel took the rows read ahead back before they were used, and the prefill
// read them again one synchronous fault at a time. The prefill's graph prices them (prefill_pages) before it runs; both
// terms hold that room from then until the first decode token, so the RAM cache gives it back at once at the prefill's
// start and takes it back at the first decode token.
#include "budget-linux.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <unistd.h>

struct kimi_host_ledger {
    enum room_state { room_ok, room_pressure, room_unreadable };
    // Both terms for one request. A term that cannot be read does not bind; the RAM cache still admits only under
    // a readable finite cgroup (cgroup == room_unreadable stops it).
    struct room {
        room_state cgroup = room_unreadable;
        uint64_t cgroup_shortage = 0, cgroup_surplus = 0;
        bool system = false; // MemAvailable was read
        // available: MemAvailable, less boost_headroom when measured at the full boost
        uint64_t available = 0, floor = 0, system_shortage = 0, system_surplus = 0, boost_headroom = 0;
        bool fits() const { return !cgroup_shortage && !system_shortage; }
        uint64_t shortage() const { return std::max(cgroup_shortage, system_shortage); }
        // What both readable terms leave; UINT64_MAX when neither binds.
        uint64_t surplus() const {
            if (!fits()) { return 0; }
            uint64_t s = UINT64_MAX;
            if (cgroup != room_unreadable) { s = std::min(s, cgroup_surplus); }
            if (system) { s = std::min(s, system_surplus); }
            return s;
        }
    };
    // The files the ledger reads stay open: a cgroup or /proc file is generated again by every read from offset 0,
    // and opening one costs several times reading it (the RAM cache asks once or twice per layer and token). Copies
    // of a ledger share them.
    struct open_files {
        std::mutex mutex;
        std::map<std::string, int> fds;
        ~open_files() { for (const auto & f : fds) { close(f.second); } }
        bool read(const std::string & path, std::string & text) {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = fds.find(path);
            if (it == fds.end()) {
                const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
                if (fd < 0) { return false; }
                it = fds.emplace(path, fd).first;
            }
            text.clear();
            char buffer[4096];
            for (off_t at = 0;;) {
                const ssize_t n = pread(it->second, buffer, sizeof(buffer), at);
                if (n < 0) { close(it->second); fds.erase(it); return false; } // a removed cgroup: open again next time
                if (!n || text.size() > (1u << 20)) { break; }
                text.append(buffer, size_t(n));
                at += n;
            }
            return true;
        }
    };
    std::shared_ptr<open_files> files = std::make_shared<open_files>();
    // Tests inject their own files.
    using reader = std::function<bool(const std::string & path, std::string & text)>;
    reader read;
    bool get(const std::string & path, std::string & text) const { return read ? read(path, text) : files->read(path, text); }
    std::string cgroup;             // "/sys/fs/cgroup/<this process's cgroup>", empty when unknown
    bool reclaimable_file = false;  // cgroup term: file pages count as headroom (CACHE_RECLAIMABLE_FILE)
    bool unified = false;           // the GPU allocates out of this DRAM
    uint64_t reserve = 2ull << 30;  // the cgroup term's headroom
    uint64_t margin = 2ull << 30;   // the floor's headroom for untracked growth
    uint64_t explicit_reserve = 0;
    // The open prefill's file pages (the weight rows its graph gathers), held in both terms until the first decode.
    uint64_t prefill_pages = 0;
    uint64_t file_reserve = 0, refaults = 0, page_size = 4096, file_updates = 0;
    bool file_measured = false;
    // Host memory make_room has granted so far (callers allocate it right after), for the margin's intervals.
    uint64_t granted = 0;
    // The open interval (begin_interval): MemAvailable and the bytes the ledger knows were taken at its start.
    bool interval_open = false;
    uint64_t interval_available = 0, interval_known = 0, margin_updates = 0;

    static uint64_t add(uint64_t a, uint64_t b) { return b > UINT64_MAX - a ? UINT64_MAX : a + b; }
    uint64_t floor() const { return add(add(file_reserve, margin), explicit_reserve); }

    static bool decimal(const std::string & text, uint64_t & value) {
        if (text.empty()) { return false; }
        value = 0;
        for (const char c : text) {
            if (c < '0' || c > '9' || value > (UINT64_MAX - uint64_t(c - '0')) / 10) { return false; }
            value = value * 10 + uint64_t(c - '0');
        }
        return true;
    }
    static std::string first_token(const std::string & text) {
        const size_t begin = text.find_first_not_of(" \t\n");
        if (begin == std::string::npos) { return ""; }
        return text.substr(begin, text.find_first_of(" \t\n", begin) - begin);
    }
    // The value of `key` in "key value" lines (memory.stat, /proc/vmstat) or "key: value kB" lines (/proc/meminfo).
    static bool field(const std::string & text, const char * key, uint64_t & value) {
        const size_t length = std::strlen(key);
        for (size_t at = 0; at < text.size();) {
            const size_t end = std::min(text.find('\n', at), text.size());
            if (end - at > length && !text.compare(at, length, key) &&
                    (text[at + length] == ' ' || text[at + length] == ':')) {
                size_t begin = at + length + 1;
                while (begin < end && (text[begin] == ' ' || text[begin] == '\t')) { ++begin; }
                size_t stop = begin;
                while (stop < end && text[stop] >= '0' && text[stop] <= '9') { ++stop; }
                if (!decimal(text.substr(begin, stop - begin), value)) { return false; }
                if (text.compare(stop, 3, " kB") == 0) {
                    if (value > UINT64_MAX / 1024) { return false; }
                    value *= 1024;
                }
                return true;
            }
            at = end + 1;
        }
        return false;
    }

    static uint64_t env_mib(const char * name) {
        const char * text = std::getenv(name);
        uint64_t mib = 0;
        if (!text || !*text) { return 0; }
        if (!decimal(text, mib) || mib > (UINT64_MAX >> 20)) { throw std::runtime_error(std::string("invalid ") + name); }
        return mib << 20;
    }
    // Configuration from the environment, the one place every ledger of the process reads it. The cgroup reserve is
    // GGML_KIMI_CPU_SINK_CACHE_RESERVE_BYTES (default 2 GiB, at least 512 MiB) and the margin starts there. On unified
    // memory the tier's former GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB named the same margin, so the larger of the
    // two applies to it; a discrete GPU never read it. GGML_KIMI_CPU_SINK_CACHE_RECLAIMABLE_FILE=1 counts the
    // cgroup's file pages as its headroom.
    void configure(bool is_unified) {
        unified = is_unified;
        reserve = 2ull << 30;
        if (const char * value = std::getenv("GGML_KIMI_CPU_SINK_CACHE_RESERVE_BYTES")) {
            if (!decimal(value, reserve)) { throw std::runtime_error("invalid GGML_KIMI_CPU_SINK_CACHE_RESERVE_BYTES"); }
            if (reserve < (512ull << 20)) { throw std::runtime_error("CACHE_RESERVE_BYTES must be at least 512 MiB"); }
        }
        margin = unified ? std::max(reserve, env_mib("GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB")) : reserve;
        reclaimable_file = false;
        if (const char * value = std::getenv("GGML_KIMI_CPU_SINK_CACHE_RECLAIMABLE_FILE")) {
            if (!std::strcmp(value, "1")) { reclaimable_file = true; }
            else if (std::strcmp(value, "0")) {
                throw std::runtime_error("GGML_KIMI_CPU_SINK_CACHE_RECLAIMABLE_FILE must be 0 or 1");
            }
        }
        explicit_reserve = add(env_mib("GGML_KIMI_HOST_RESERVE_MIB"), unified ? env_mib("GGML_CUDA_MEMORY_RESERVE_MIB") : 0);
        const long page = sysconf(_SC_PAGESIZE);
        page_size = page > 0 ? uint64_t(page) : 4096;
        std::string groups;
        cgroup.clear();
        if (get("/proc/self/cgroup", groups)) {
            for (size_t at = 0; at < groups.size();) {
                const size_t end = std::min(groups.find('\n', at), groups.size());
                if (!groups.compare(at, 3, "0::")) { cgroup = "/sys/fs/cgroup" + groups.substr(at + 3, end - at - 3); break; }
                at = end + 1;
            }
        }
    }

    // MemAvailable from /proc/meminfo.
    bool memory_available(uint64_t & bytes) const {
        std::string text;
        return get("/proc/meminfo", text) && field(text, "MemAvailable", bytes);
    }
    // Unified memory: MemAvailable as it reads at every zone's largest watermark boost, from the same /proc/meminfo,
    // and what that keeps back (headroom); elsewhere MemAvailable. An unreadable /proc/zoneinfo or boost factor keeps
    // nothing back.
    bool memory_available_at_full_boost(uint64_t & bytes, uint64_t & headroom) const {
        std::string text, zones, factor;
        headroom = 0;
        if (!get("/proc/meminfo", text) || !field(text, "MemAvailable", bytes)) { return false; }
        if (unified && get("/proc/zoneinfo", zones) && get("/proc/sys/vm/watermark_boost_factor", factor)) {
            headroom = std::min(bytes, kimi_memory_boost::headroom(text, zones, factor, page_size));
        }
        bytes -= headroom;
        return true;
    }
    // cgroup term: every finite memory.max ancestor must keep the reserve and the open prefill's pages free after
    // `extra`. Unreadable controller files below the root disable the term (and with it RAM cache admission)
    // conservatively.
    void cgroup_term(uint64_t extra, room & r) const {
        if (cgroup.empty()) { return; }
        const size_t root = std::strlen("/sys/fs/cgroup");
        std::string path = cgroup, text;
        bool bounded = false;
        uint64_t missing = 0, spare = UINT64_MAX;
        while (path.size() >= root) {
            const std::string maximum = get(path + "/memory.max", text) ? first_token(text) : "";
            if (maximum.empty()) {
                // The cgroup-v2 root has no memory.max.
                if (path != "/sys/fs/cgroup" && path != "/sys/fs/cgroup/") { return; }
            } else if (maximum != "max") {
                bounded = true;
                uint64_t limit = 0, used = 0;
                if (!decimal(maximum, limit) || !get(path + "/memory.current", text) || !decimal(first_token(text), used)) { return; }
                if (reclaimable_file || prefill_pages) {
                    // File-backed page cache is reclaimable before anonymous memory (dirty pages may first need
                    // writeback). With CACHE_RECLAIMABLE_FILE both LRU lists count as headroom, and the open prefill's
                    // pages are required beside the reserve whether they have come in yet or not. Without it file
                    // pages are use, except up to the prefill's pages: those are the ones it holds room for (or
                    // pages the kernel takes back to make that room), not more use. Unreadable stats fail closed.
                    uint64_t active = 0, inactive = 0;
                    if (!get(path + "/memory.stat", text) || !field(text, "active_file", active) ||
                            !field(text, "inactive_file", inactive)) { return; }
                    const uint64_t file = add(active, inactive);
                    used -= std::min(used, reclaimable_file ? file : std::min(file, prefill_pages));
                }
                const uint64_t required = add(add(reserve, prefill_pages), extra);
                uint64_t deficit = 0;
                if (used >= limit) { deficit = add(used - limit, required); }
                else if (required > limit - used) { deficit = required - (limit - used); }
                missing = std::max(missing, deficit);
                spare = std::min(spare, deficit ? uint64_t(0) : limit - used - required);
            }
            const size_t slash = path.find_last_of('/');
            if (slash == std::string::npos) { break; }
            path.resize(slash);
        }
        if (!bounded) { return; }
        r.cgroup = missing ? room_pressure : room_ok;
        r.cgroup_shortage = missing;
        r.cgroup_surplus = missing ? 0 : spare;
    }
    // What an allocation takes from this DRAM: host memory always, device memory only where the GPU shares it.
    uint64_t dram_bytes(uint64_t host, uint64_t device) const { return add(host, unified ? device : 0); }
    // Which term was short ("cgroup", "system", "both") or "none", for the receipts.
    static const char * short_term(const room & r) {
        return r.cgroup_shortage && r.system_shortage ? "both" : r.cgroup_shortage ? "cgroup" : r.system_shortage ? "system" : "none";
    }
    // full_boost: the system term counts MemAvailable at the full watermark boost, for memory about to be taken up
    // to the floor (make_room). The RAM cache's per-layer guard reads MemAvailable as it is: it commits only the
    // slots whose room the tier keeps free as the cache's demand, with the boost headroom beside it. MemAvailable
    // counts page cache as available, so the open prefill's pages are required beside the floor, as in the cgroup term.
    room measure(uint64_t extra, bool full_boost = false) const {
        room r;
        cgroup_term(extra, r);
        if (full_boost ? memory_available_at_full_boost(r.available, r.boost_headroom) : memory_available(r.available)) {
            r.system = true;
            r.floor = floor();
            const uint64_t need = add(add(r.floor, prefill_pages), extra);
            r.system_shortage = need > r.available ? need - r.available : 0;
            r.system_surplus = r.available > need ? r.available - need : 0;
        }
        return r;
    }

    // The server's file pages and refault count: its cgroup's memory.stat, else the system's counters.
    bool file_working_set(uint64_t & file, uint64_t & refault) const {
        std::string text;
        if (!cgroup.empty() && get(cgroup + "/memory.stat", text) && field(text, "file", file) &&
                field(text, "workingset_refault_file", refault)) { return true; }
        uint64_t pages = 0;
        if (get("/proc/vmstat", text) && field(text, "nr_file_pages", pages) && field(text, "workingset_refault_file", refault)) {
            file = pages > UINT64_MAX / page_size ? UINT64_MAX : pages * page_size;
            return true;
        }
        return false;
    }
    // Unified memory: measures the file reserve the first time, then adds the refaults since. True when it grew.
    bool update_file_reserve() {
        if (!unified) { return false; }
        uint64_t file = 0, refault = 0;
        if (!file_working_set(file, refault)) { return false; }
        uint64_t next = file_reserve;
        if (!file_measured) { next = std::max(next, file); file_measured = true; }
        else if (refault > refaults) { next = add(next, (refault - refaults) > UINT64_MAX / page_size ? UINT64_MAX : (refault - refaults) * page_size); }
        refaults = refault;
        if (next == file_reserve) { return false; }
        file_reserve = next;
        ++file_updates;
        return true;
    }

    // What make_room may ask of the expert caches, in this order.
    struct actions {
        // Publishes the GPU tier's finished deferred uploads and returns their RAM leases; waits for copies in flight.
        std::function<void()> collect_deferred;
        // Unified memory: the GPU tier gives blocks back so that an allocation of `bytes` keeps its free memory above
        // its low watermark. False when it could not (a layer is live, nothing left to give).
        std::function<bool(uint64_t bytes)> reclaim_tier;
        // The RAM cache gives slots back until room(bytes) fits (false when it cannot).
        std::function<bool(uint64_t bytes)> reclaim_cache;
    };
    // Room for `extra` bytes the caller is about to allocate outside the expert caches, MemAvailable counted at the
    // full watermark boost. A tier block on unified memory is system memory the cgroup does not see, so it relieves
    // only the system term; it is taken first because the tier grows back into memory that is free again at a later
    // decode token, while a RAM cache slot given back stays disabled until the first decode token after the prefill
    // takes back what the room allows then. The RAM cache relieves both terms. The tier keeps the open prefill's pages
    // free beside the allocation.
    room make_room(uint64_t extra, const actions & a) {
        room r = measure(extra, true);
        if (!r.fits() && r.system_shortage && unified && a.reclaim_tier) {
            if (a.collect_deferred) { a.collect_deferred(); }
            a.reclaim_tier(add(extra, prefill_pages));
            r = measure(extra, true);
        }
        if (!r.fits() && a.reclaim_cache) {
            a.reclaim_cache(extra);
            r = measure(extra, true);
        }
        if (r.fits()) { granted = add(granted, extra); }
        return r;
    }

    // The margin learns growth nobody accounts (unified memory), one decode token at a time. `held` is the bytes the
    // engine knows it holds in this DRAM (GPU tier blocks, committed RAM cache slots). An interval begins after a
    // decode token boundary has resized the caches and ends at the next boundary before it resizes anything, so the
    // boundaries' own resizes (and a MemAvailable that lags them) fall outside every interval, while a block the OOM
    // reclaimer gives a ggml allocation inside a token pairs with that allocation. Within the interval, a drop in
    // MemAvailable beyond what `held` and the ledger's grants explain was taken by something untracked: CUDA graphs
    // instantiated for a new shape, pool and driver allocations. When one interval's untracked drop exceeds the
    // margin, the margin becomes it; it never shrinks. MemAvailable is read at the full watermark boost, so that a boost
    // the kernel raises or drops within the token is not growth.
    void begin_interval(uint64_t held) {
        uint64_t headroom = 0;
        interval_open = unified && memory_available_at_full_boost(interval_available, headroom);
        interval_known = add(held, granted);
    }
    // consecutive=false (a prefill or a skipped token came between) discards the open interval. True when the
    // margin grew.
    bool end_interval(bool consecutive, uint64_t held, uint64_t & untracked) {
        untracked = 0;
        const bool open = interval_open;
        interval_open = false;
        uint64_t available = 0, headroom = 0;
        if (!unified || !open || !consecutive || !memory_available_at_full_boost(available, headroom)) { return false; }
        // drop - (known - interval_known), with either difference possibly negative
        const uint64_t lhs = add(interval_available, interval_known), rhs = add(available, add(held, granted));
        untracked = lhs > rhs ? lhs - rhs : 0;
        if (untracked <= margin) { return false; }
        margin = untracked;
        ++margin_updates;
        return true;
    }
};
