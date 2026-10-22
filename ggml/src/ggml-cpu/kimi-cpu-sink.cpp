#include "kimi-cpu-sink.h"
#include "kimi-cpu-sink-internal.h"
#include "kimi-prefill-cost.h"
#include "kimi-host-lock.h"
#include "kimi-cache-location-api.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C" size_t kimi_cpu_sink_source_size(void) { return sizeof(kimi_cpu_sink_source); }

static void sink_error(char * dst, size_t size, const char * message) {
    if (dst && size) {
        std::snprintf(dst, size, "%s", message);
    }
}

#if defined(__linux__)

#include <cerrno>
#include <fcntl.h>
#include <dlfcn.h>
#include "kimi-host-ledger.h"
#include "kimi-vram-tier-api.h"
#include "kimi-vram-location-api.h"
#include <linux/io_uring.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>
#include <time.h>

namespace {

using sink_clock = std::chrono::steady_clock;
static double elapsed_us(sink_clock::time_point start) {
    return std::chrono::duration<double, std::micro>(sink_clock::now() - start).count();
}

static int sink_poll_us_env() {
    const char * value = std::getenv("GGML_KIMI_CPU_SINK_POLL_US");
    if (!value || !*value) { return 0; }
    char * end = nullptr;
    errno = 0;
    const long n = std::strtol(value, &end, 10);
    if (errno || end == value || *end || n < 0 || n > 1000) {
        throw std::runtime_error("GGML_KIMI_CPU_SINK_POLL_US must be an integer in [0,1000]");
    }
    return int(n);
}

template <typename Predicate>
static void sink_poll(int microseconds, Predicate ready) {
    const auto deadline = sink_clock::now() + std::chrono::microseconds(microseconds);
    do {
        for (int i = 0; i < 32; ++i) {
            if (ready()) { return; }
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#elif defined(__aarch64__)
            __asm__ __volatile__("yield" ::: "memory");
#endif
        }
    } while (sink_clock::now() < deadline);
}

static std::runtime_error sys_error(const char * operation) {
    return std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
}

static bool sink_read_fixed_from_env() {
    const char * value = std::getenv("GGML_KIMI_CPU_SINK_READ_FIXED");
    if (!value || !value[0] || !std::strcmp(value, "0")) { return false; }
    if (!std::strcmp(value, "1")) { return true; }
    throw std::runtime_error("GGML_KIMI_CPU_SINK_READ_FIXED must be 0 or 1");
}

// Single producer/consumer, owned by the persistent I/O thread. Direct syscalls
// avoid adding a liburing dependency to the CPU backend.
struct sink_ring {
    int fd = -1;
    void * sq = MAP_FAILED;
    void * cq = MAP_FAILED;
    io_uring_sqe * sqes = static_cast<io_uring_sqe *>(MAP_FAILED);
    size_t sq_size = 0, cq_size = 0, sqes_size = 0;
    unsigned * sq_head = nullptr, * sq_tail = nullptr, * sq_mask = nullptr, * sq_array = nullptr;
    unsigned * cq_head = nullptr, * cq_tail = nullptr, * cq_mask = nullptr;
    io_uring_cqe * cqes = nullptr;
    unsigned sq_entries = 0;
    uint64_t pending = 0; // published SQEs whose CQEs have not been consumed
    uint64_t completed = 0;
    bool read_fixed = false;
    bool buffers_registered = false;
    std::vector<iovec> fixed_buffers;
    uint64_t register_count = 0, unregister_count = 0;

    ~sink_ring() { close_ring(); }

    void close_ring() {
        // close(io_uring_fd) starts asynchronous cancellation; it does not make
        // freeing user buffers safe. Every runtime error must first drain them.
        if (pending) {
            std::fprintf(stderr, "[kimi_cpu_sink] undrained I/O at ring destruction; aborting\n");
            std::abort();
        }
        try {
            unregister_buffers();
        } catch (const std::exception & e) {
            std::fprintf(stderr, "[kimi_cpu_sink_io] cannot unregister before buffer release: %s; aborting\n", e.what());
            std::abort();
        }
        if (sqes != MAP_FAILED) { munmap(sqes, sqes_size); }
        if (cq != MAP_FAILED && cq != sq) { munmap(cq, cq_size); }
        if (sq != MAP_FAILED) { munmap(sq, sq_size); }
        if (fd >= 0) { close(fd); }
        fd = -1;
        sq = cq = MAP_FAILED;
        sqes = static_cast<io_uring_sqe *>(MAP_FAILED);
    }

    void init(unsigned depth, bool fixed) {
        read_fixed = fixed;
        io_uring_params p = {};
        fd = static_cast<int>(syscall(__NR_io_uring_setup, depth, &p));
        if (fd < 0) { throw sys_error("io_uring_setup (no fallback)"); }
        sq_size = p.sq_off.array + p.sq_entries * sizeof(unsigned);
        cq_size = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
        if (p.features & IORING_FEAT_SINGLE_MMAP) { sq_size = cq_size = std::max(sq_size, cq_size); }
        sq = mmap(nullptr, sq_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, IORING_OFF_SQ_RING);
        if (sq == MAP_FAILED) { throw sys_error("mmap io_uring SQ"); }
        cq = (p.features & IORING_FEAT_SINGLE_MMAP) ? sq :
            mmap(nullptr, cq_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, IORING_OFF_CQ_RING);
        if (cq == MAP_FAILED) { throw sys_error("mmap io_uring CQ"); }
        sqes_size = p.sq_entries * sizeof(io_uring_sqe);
        sqes = static_cast<io_uring_sqe *>(mmap(nullptr, sqes_size, PROT_READ | PROT_WRITE,
                                              MAP_SHARED, fd, IORING_OFF_SQES));
        if (sqes == MAP_FAILED) { throw sys_error("mmap io_uring SQEs"); }
        auto sq_ptr = [this](unsigned off) { return reinterpret_cast<unsigned *>(static_cast<char *>(sq) + off); };
        auto cq_ptr = [this](unsigned off) { return reinterpret_cast<unsigned *>(static_cast<char *>(cq) + off); };
        sq_head = sq_ptr(p.sq_off.head); sq_tail = sq_ptr(p.sq_off.tail);
        sq_mask = sq_ptr(p.sq_off.ring_mask); sq_array = sq_ptr(p.sq_off.array);
        cq_head = cq_ptr(p.cq_off.head); cq_tail = cq_ptr(p.cq_off.tail); cq_mask = cq_ptr(p.cq_off.ring_mask);
        cqes = reinterpret_cast<io_uring_cqe *>(static_cast<char *>(cq) + p.cq_off.cqes);
        sq_entries = p.sq_entries;
    }

    void register_buffers(const std::vector<iovec> & buffers) {
        if (!read_fixed || buffers_registered || pending || buffers.empty() || buffers.size() > UINT16_MAX) {
            throw std::runtime_error("invalid CPU sink fixed-buffer registration state");
        }
        // Retain the indexed ranges for submission checks. Allocate before the
        // syscall so a C++ allocation failure cannot lose a live registration.
        fixed_buffers = buffers;
        int result;
        do {
            result = static_cast<int>(syscall(__NR_io_uring_register, fd, IORING_REGISTER_BUFFERS,
                                              fixed_buffers.data(), static_cast<unsigned>(fixed_buffers.size())));
        } while (result < 0 && errno == EINTR);
        if (result < 0) {
            const auto error = sys_error("IORING_REGISTER_BUFFERS (check RLIMIT_MEMLOCK; no fallback)");
            fixed_buffers.clear();
            throw error;
        }
        buffers_registered = true;
        ++register_count;
        size_t bytes = 0;
        for (const auto & buffer : fixed_buffers) { bytes += buffer.iov_len; }
        std::fprintf(stderr, "[kimi_cpu_sink_io] read_fixed=1 registered=%zu bytes=%zu register_count=%llu unregister_count=%llu\n",
                     fixed_buffers.size(), bytes, static_cast<unsigned long long>(register_count),
                     static_cast<unsigned long long>(unregister_count));
    }

    void unregister_buffers() {
        if (!buffers_registered) { return; }
        if (pending) { throw std::runtime_error("cannot unregister CPU sink buffers with pending I/O"); }
        int result;
        do {
            result = static_cast<int>(syscall(__NR_io_uring_register, fd, IORING_UNREGISTER_BUFFERS, nullptr, 0));
        } while (result < 0 && errno == EINTR);
        if (result < 0) { throw sys_error("IORING_UNREGISTER_BUFFERS"); }
        buffers_registered = false;
        fixed_buffers.clear();
        ++unregister_count;
        std::fprintf(stderr, "[kimi_cpu_sink_io] read_fixed=1 register_count=%llu unregister_count=%llu\n",
                     static_cast<unsigned long long>(register_count), static_cast<unsigned long long>(unregister_count));
    }

    void queue(const kimi_cpu_sink_source & source, void * data, uint64_t tag, bool fixed) {
        if (!data || reinterpret_cast<uintptr_t>(data) % 4096) {
            throw std::runtime_error("CPU sink read target is not 4096-byte aligned");
        }
        if (fixed && (!read_fixed || !buffers_registered || tag >= fixed_buffers.size() ||
            data != fixed_buffers[tag].iov_base || source.nbytes > fixed_buffers[tag].iov_len)) {
            throw std::runtime_error("READ_FIXED buffer is absent or outside its registered range (no fallback)");
        }
        const unsigned head = __atomic_load_n(sq_head, __ATOMIC_ACQUIRE);
        const unsigned tail = *sq_tail;
        if (tail - head >= sq_entries) { throw std::runtime_error("CPU sink SQ unexpectedly full"); }
        const unsigned index = tail & *sq_mask;
        io_uring_sqe & entry = sqes[index];
        std::memset(&entry, 0, sizeof(entry));
        entry.opcode = fixed ? IORING_OP_READ_FIXED : IORING_OP_READ;
        if (fixed) { entry.buf_index = static_cast<uint16_t>(tag); }
        entry.fd = source.fd_direct;
        entry.off = source.offset;
        entry.addr = reinterpret_cast<uint64_t>(data);
        entry.len = static_cast<unsigned>(source.nbytes);
        entry.user_data = tag;
        sq_array[index] = index;
        __atomic_store_n(sq_tail, tail + 1, __ATOMIC_RELEASE);
        ++pending;
    }

    void submit(unsigned count) {
        while (count) {
            const int result = static_cast<int>(syscall(__NR_io_uring_enter, fd, count, 0, 0, nullptr, 0));
            if (result < 0 && errno == EINTR) { continue; }
            if (result < 0) { throw sys_error("io_uring_enter submit"); }
            if (!result) { throw std::runtime_error("io_uring submission made no progress"); }
            count -= static_cast<unsigned>(result);
        }
    }

    bool pop(io_uring_cqe & result) {
        const unsigned head = *cq_head;
        if (head == __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE)) { return false; }
        result = cqes[head & *cq_mask];
        __atomic_store_n(cq_head, head + 1, __ATOMIC_RELEASE);
        if (!pending) { throw std::runtime_error("CPU sink received an untracked completion"); }
        --pending;
        ++completed;
        return true;
    }

    void wait() {
        int result;
        do {
            result = static_cast<int>(syscall(__NR_io_uring_enter, fd, 0, 1, IORING_ENTER_GETEVENTS, nullptr, 0));
        } while (result < 0 && errno == EINTR);
        if (result < 0) { throw sys_error("io_uring_enter wait"); }
    }

    void drain(void * userdata, void (*on_completion)(void *, const io_uring_cqe &)) {
        if (!pending) { return; }
        // An interrupted/failed submission may have consumed only part of the
        // published SQEs. Submit the rest before waiting for their completions.
        const unsigned remaining = *sq_tail - __atomic_load_n(sq_head, __ATOMIC_ACQUIRE);
        submit(remaining);
        io_uring_cqe cqe;
        while (pending) {
            while (pop(cqe)) { on_completion(userdata, cqe); }
            if (pending) { wait(); }
        }
    }
};

struct sink_buffer {
    void * data = nullptr;
    size_t size = 0;
    ~sink_buffer() { std::free(data); }
    sink_buffer() = default;
    sink_buffer(const sink_buffer &) = delete;
    sink_buffer & operator=(const sink_buffer &) = delete;
    sink_buffer(sink_buffer && other) noexcept : data(other.data), size(other.size) { other.data = nullptr; other.size = 0; }
    void reserve(size_t n) {
        if (size >= n) { return; }
        void * next = nullptr;
        if (posix_memalign(&next, 4096, n) != 0) { throw std::runtime_error("CPU sink aligned allocation failed"); }
        std::free(data);
        data = next; size = n;
    }
};

// Weight types whose CPU MUL_MAT_ID is one vec_dot per weight row against an
// input row quantized by from_float of vec_dot_type, with whole blocks per row.
static bool sink_expert_type_supported(ggml_type type, int64_t columns) {
    if (int(type) < 0 || int(type) >= int(GGML_TYPE_COUNT) || ggml_blck_size(type) <= 0) { return false; }
    const auto * traits = ggml_get_type_traits_cpu(type);
    if (!traits || !traits->vec_dot || traits->vec_dot_type == GGML_TYPE_F32) { return false; }
    const auto * input = ggml_get_type_traits_cpu(traits->vec_dot_type);
    return input && input->from_float && ggml_blck_size(traits->vec_dot_type) > 0 &&
        columns % ggml_blck_size(type) == 0 && columns % ggml_blck_size(traits->vec_dot_type) == 0;
}

static bool sink_bool_env(const char * name, bool fallback) {
    const char * value = std::getenv(name);
    if (!value || !*value) { return fallback; }
    if (std::strcmp(value, "0") == 0) { return false; }
    if (std::strcmp(value, "1") == 0) { return true; }
    throw std::runtime_error(std::string(name) + " must be 0 or 1");
}

static int sink_dynamic_chunk_rows_env() {
    const char * value = std::getenv("GGML_KIMI_CPU_SINK_DYNAMIC_CHUNK_ROWS");
    if (!value || !*value) { return 64; }
    char * end = nullptr;
    errno = 0;
    const long n = std::strtol(value, &end, 10);
    // A float output cache line is 16 rows. Requiring whole lines lets the
    // scheduler give every worker disjoint cache lines after its aligned prefix.
    if (errno || end == value || *end || n < 16 || n > 16384 || n % 16) {
        throw std::runtime_error("GGML_KIMI_CPU_SINK_DYNAMIC_CHUNK_ROWS must be a multiple of 16 in [16,16384]");
    }
    return int(n);
}

static std::vector<double> sink_weights_env(int count) {
    const char * p = std::getenv("GGML_KIMI_CPU_SINK_ROW_WEIGHTS");
    std::vector<double> result;
    if (!p || !*p) { return result; }
    for (;;) {
        char * end = nullptr;
        errno = 0;
        const double value = std::strtod(p, &end);
        if (end == p || errno || !std::isfinite(value) || value <= 0 || (*end && *end != ',')) {
            throw std::runtime_error("ROW_WEIGHTS requires comma-separated positive finite weights");
        }
        result.push_back(value);
        if (!*end) { break; }
        p = end + 1;
    }
    if (result.size() != size_t(count)) {
        throw std::runtime_error("ROW_WEIGHTS requires total_threads - 1 entries, including the caller");
    }
    return result;
}

static std::vector<int> sink_cpus_env(const char * name, int count) {
    const char * p = std::getenv(name);
    std::vector<int> result;
    if (!p || !*p) { return result; }
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) { throw sys_error("get sink affinity"); }
    for (;;) {
        char * end = nullptr;
        errno = 0;
        const long cpu = std::strtol(p, &end, 10);
        if (end == p || errno || cpu < 0 || cpu >= CPU_SETSIZE || (*end && *end != ',') ||
            !CPU_ISSET(cpu, &allowed) || std::find(result.begin(), result.end(), cpu) != result.end()) {
            throw std::runtime_error(std::string(name) + " requires distinct CPUs within the caller's allowed mask");
        }
        result.push_back(int(cpu));
        if (!*end) { break; }
        p = end + 1;
    }
    if (count && result.size() != size_t(count)) {
        throw std::runtime_error(std::string(name) + " requires total_threads - 1 CPUs, including the caller");
    }
    return result;
}

static void sink_pin(pthread_t thread, const std::vector<int> & cpus) {
    if (cpus.empty()) { return; }
    cpu_set_t wanted, actual;
    CPU_ZERO(&wanted);
    for (int cpu : cpus) { CPU_SET(cpu, &wanted); }
    const int result = pthread_setaffinity_np(thread, sizeof(wanted), &wanted);
    if (result || pthread_getaffinity_np(thread, sizeof(actual), &actual) || !CPU_EQUAL(&wanted, &actual)) {
        throw std::runtime_error("CPU sink affinity could not be applied exactly (no fallback)");
    }
}

// The graph caller also runs unrelated work. Pin only for this sink call and
// restore its original mask on success and every exception path.
struct sink_caller_affinity {
    cpu_set_t saved;
    bool active = false;
    explicit sink_caller_affinity(const std::vector<int> & cpus) {
        if (cpus.empty()) { return; }
        if (pthread_getaffinity_np(pthread_self(), sizeof(saved), &saved)) {
            throw std::runtime_error("cannot save sink caller affinity");
        }
        try { sink_pin(pthread_self(), { cpus[0] }); }
        catch (...) { pthread_setaffinity_np(pthread_self(), sizeof(saved), &saved); throw; }
        active = true;
    }
    ~sink_caller_affinity() {
        if (active && pthread_setaffinity_np(pthread_self(), sizeof(saved), &saved)) {
            std::fprintf(stderr, "[kimi_cpu_sink] cannot restore caller affinity; aborting\n");
            std::abort();
        }
    }
};

static double sink_thread_cpu_us() {
    timespec value;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) { return 0.0; }
    return double(value.tv_sec) * 1e6 + double(value.tv_nsec) * 1e-3;
}

struct sink_worker_profile {
    double up_gate_wall_us = 0, down_wall_us = 0;
    double up_gate_cpu_us = 0, down_cpu_us = 0;
    double parallel_wall_us = 0, parallel_cpu_us = 0; // generic prefill callbacks
    double dispatch_us = 0, max_dispatch_us = 0;
    uint64_t phases = 0, no_work_wakeups = 0;
};

struct sink_request {
    kimi_cpu_sink_source role[3];
    float weight;
    int32_t id = -1;
    int cache_hit = -1;
    int cache_admit = -1;
    int vram_slot = -1;
    bool vram_transient = false;
    bool vram_ssd_transient = false;
    bool vram_ssd_streaming = false;
    int vram_ssd_staging = -1;
    bool vram_cold_admit = false;
    int vram_cold_admit_routes = 0;
    double lane_start_us = 0.0; // MXFP4 grouped prefill: predicted start on its lane (CPU or GPU)
    int prefill_routes = 0;     // grouped prefill: the expert's distinct tokens
    bool cache_lease = false;
    bool trace_ram_resident_before = false;
    size_t io_slot = 0;
    uint64_t read_bytes = 0; // positive CQE bytes; only the I/O thread writes
    void * read_target[3] = {}; // fixed before publishing the I/O job
    bool cache_only = false; // must never allocate a read target or submit I/O
    uint64_t ready_us = 0; // all successful CQEs, published under io_mutex
    uint64_t events[8] = {}; // cache-compute audit; read only after I/O drain
};

// One bounded, ordinary anonymous mapping. Entries never point into the I/O
// ring and are never registered with io_uring or CUDA. The run mutex owns all
// metadata; workers only borrow immutable bytes during their phase barriers.
// CPUs this process may run on that belong to the fastest cluster of a heterogeneous CPU (Arm big.LITTLE,
// Intel P/E cores), from the kernel's per-CPU capacity, else the maximum frequency. Clusters are split at
// the largest relative step between sorted values; a step under 10% means one cluster, and the result is
// then every CPU. Only a candidate: decode measures it against all threads (sink_thread_choice).
static int sink_fast_cpus() {
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) { return 0; }
    std::vector<double> values;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) { continue; }
        double value = 0;
        for (const char * file : { "cpu_capacity", "cpufreq/cpuinfo_max_freq" }) {
            std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/" + file);
            if (in >> value && value > 0) { break; }
            value = 0;
        }
        if (value <= 0) { return CPU_COUNT(&allowed); }
        values.push_back(value);
    }
    if (values.empty()) { return 0; }
    std::sort(values.begin(), values.end(), std::greater<double>());
    size_t cut = values.size();
    double step = 0.10;
    for (size_t i = 1; i < values.size(); ++i) {
        const double drop = 1.0 - values[i] / values[i - 1];
        if (drop > step) { step = drop; cut = i; }
    }
    return int(cut);
}

// Whether the GPU shares system memory (DGX Spark, Jetson), from the CUDA backend if it is loaded.
static bool sink_unified_memory() {
    using unified_fn = bool (*)(int);
    static const unified_fn is_unified =
        reinterpret_cast<unified_fn>(dlsym(RTLD_DEFAULT, "ggml_backend_cuda_device_is_unified"));
    return is_unified && is_unified(0);
}

struct sink_cache {
    struct source_key {
        uint64_t device, inode, size, offset, bytes;
        int64_t mtime_s, mtime_ns, ctime_s, ctime_ns;
        bool operator==(const source_key & o) const {
            return device == o.device && inode == o.inode && size == o.size && offset == o.offset &&
                bytes == o.bytes && mtime_s == o.mtime_s && mtime_ns == o.mtime_ns &&
                ctime_s == o.ctime_s && ctime_ns == o.ctime_ns;
        }
    };
    struct entry {
        source_key source[3] = {};
        int32_t id = -1;
        size_t layer_index = 0;
        uint64_t last_touch = 0;
        bool valid = false, reserved = false, resident = false, enabled = false, was_released = false, protected_now = false;
        bool prefetched = false; // read ahead by the next-layer prefetch and not requested yet
        bool inflight = false;   // a prefetch read writes into the slot; nothing else may take it
        uint32_t leases = 0;
        uint64_t generation = 0; // cache generation at a prefetch reservation
    };
    struct layer {
        int32_t id;
        size_t begin = 0, count = 0;
        uint64_t token = 0;
        bool seen = false;
        std::map<int32_t, uint64_t> frequency;
        std::map<int32_t, double> count_frequency;
        uint64_t count_step = 0;
    };
    std::vector<layer> layers;
    std::vector<entry> entries;
    kimi_cpu_sink_cache_stats stats = {};
    char * arena = nullptr;
    size_t stride = 0, role_bytes[3] = {};
    int quota = 8;
    uint64_t epoch = 0, token = 0;
    bool active = false, has_token = false, paused = false, elastic_prefill = false;
    enum policy_type { window, lru, lfu, decay, count_decay2 } policy = window;
    // Room for the cache comes from the host-memory ledger, which every other host allocation asks too. Its cgroup
    // reserve is CACHE_RESERVE_BYTES; opt-in CACHE_RECLAIMABLE_FILE counts file-backed pages as cgroup headroom.
    kimi_host_ledger ledger;
    // The last answer check_room got and the rejections per short term, for the receipts: on a discrete GPU the
    // cgroup term is meant to bind, so a system-term rejection there points at memory the cgroup does not see.
    mutable kimi_host_ledger::room last_room;
    uint64_t cgroup_rejections = 0, system_rejections = 0;
    void count_rejection() {
        cgroup_rejections += last_room.cgroup_shortage != 0;
        system_rejections += last_room.system_shortage != 0;
    }
    using room_state = kimi_host_ledger::room_state;
    static constexpr room_state room_ok = kimi_host_ledger::room_ok, room_pressure = kimi_host_ledger::room_pressure,
        room_unreadable = kimi_host_ledger::room_unreadable;
    bool keep_across_requests = false;
    bool adopt = false; // the arena was made at load (reserve_at_load): the first decode keeps it, whatever its epoch
    // An automatic budget's prices off unified memory (geometry): a slot, and the metadata room kept beside the slots.
    uint64_t slot_cost_bytes = 0, metadata_bytes = 0;
    bool prefill_read = false;
    bool elastic = false;
    bool partial_reclaim = false;
    size_t prefill_slots = 0;
    enum allocation_type { priority_order, uniform, global } allocation = priority_order;
    uint64_t clock = 0, age_steps = 0;
    uint64_t prefetch_hits = 0, prefetch_unused = 0;
    // CACHE_BYTES unset: the cache is sized from this machine when the first decode token learns the slot size.
    static constexpr uint64_t auto_budget = UINT64_MAX;
    int staging_per_layer = 1; // slots per layer the next-layer prefetch lands in
    // Changes whenever release() drops every entry. A hook lets deferred GPU uploads that borrow entries finish first.
    uint64_t generation = 0;
    void (*before_release)(void *) = nullptr;
    void * before_release_arg = nullptr;
    void (*before_unmap)(void *) = nullptr; // waits for prefetch reads into the arena
    void * before_unmap_arg = nullptr;
    // Before slots' pages are discarded (MADV_DONTNEED) or unmapped: entries [first, first + count). The decode
    // registrar's page-locks must go first, or the device would keep the old pages.
    void (*before_discard)(void *, size_t first, size_t count) = nullptr;
    void * before_discard_arg = nullptr;
    // Optional fixed geometry learned from earlier requests. This only assigns
    // physical RAM slots; source identity, admission and leases stay unchanged.
    std::map<int32_t, size_t> layer_quotas;
    const uint64_t page_size = uint64_t(sysconf(_SC_PAGESIZE));

    static uint64_t add(uint64_t a, uint64_t b) {
        if (b > UINT64_MAX - a) { throw std::runtime_error("CPU sink cache size addition overflow"); }
        return a + b;
    }
    static uint64_t multiply(uint64_t a, uint64_t b) {
        if (a && b > UINT64_MAX / a) { throw std::runtime_error("CPU sink cache size multiplication overflow"); }
        return a * b;
    }
    static uint64_t ceil_div(uint64_t a, uint64_t b) { return a / b + (a % b != 0); }
    uint64_t page_tables(uint64_t bytes) const {
        if (!bytes) { return 0; }
        // Five levels, with an extra boundary table at each level. Summing
        // this independently for each new slot overcounts shared tables.
        uint64_t pages = ceil_div(bytes, page_size), tables = 1;
        for (int level = 0; level < 4; ++level) {
            pages = add(ceil_div(pages, page_size / sizeof(uint64_t)), 1);
            tables = add(tables, pages);
        }
        return multiply(tables, page_size);
    }
    uint64_t allocation_cost(uint64_t bytes) const {
        if (!bytes) { return 0; }
        // std::vector/map storage belongs to libc. Allow allocator headers,
        // unaligned ends and PMD-sized THP at both boundaries. Payload itself
        // is MADV_NOHUGEPAGE and uses the much smaller ordinary-page bound.
        const uint64_t huge = multiply(page_size, page_size / sizeof(uint64_t));
        const uint64_t resident = add(multiply(ceil_div(add(bytes, 64), huge), huge), huge);
        return add(add(resident, page_tables(resident)), 64 * 1024);
    }
    bool guard(uint64_t extra) {
        stats.last_guard_extra_bytes = extra;
        stats.peak_guard_extra_bytes = std::max(stats.peak_guard_extra_bytes, extra);
        return check_room(extra, nullptr) == room_ok;
    }
    // term: the ledger term that was short when the pause is a rejection for room ("none" otherwise).
    void pause(std::vector<sink_request> & requests, kimi_cpu_sink_stats & current, const char * reason,
               const char * term = "none") {
        // No I/O job or read_target has been published yet. In particular,
        // hits planned before the pressure check must become ring misses.
        for (auto & r : requests) { r.cache_hit = r.cache_admit = -1; }
        current.cache_hits = current.cache_new_slot_bytes = 0;
        release();
        std::vector<entry>().swap(entries);
        stride = 0; std::fill(role_bytes, role_bytes + 3, 0);
        stats.capacity_bytes = stats.enabled_bytes = 0;
        paused = true; ++stats.pressure_stops;
        std::fprintf(stderr, "[kimi_cpu_sink_cache] paused=1 reason=%s term=%s reserve_bytes=%llu extra_bytes=%llu\n",
                reason, term, (unsigned long long) ledger.reserve,
                (unsigned long long) stats.last_guard_extra_bytes);
    }

    const char * policy_name() const {
        switch (policy) {
            case window: return "window";
            case lru:    return "lru";
            case lfu:    return "lfu";
            case decay:  return "decay";
            case count_decay2: return "count_decay2";
        }
        return "unknown";
    }
    const char * allocation_name() const {
        switch (allocation) {
            case priority_order: return "priority";
            case uniform:        return "uniform";
            case global:         return "global";
        }
        return "unknown";
    }

    sink_cache() {
        if (page_size < 4096 || page_size > 65536 || (page_size & (page_size - 1))) {
            throw std::runtime_error("CPU sink cache requires a supported system page size");
        }
        if (const char * value = std::getenv("GGML_KIMI_CPU_SINK_CACHE_POLICY")) {
            if (!std::strcmp(value, "window")) { policy = window; }
            else if (!std::strcmp(value, "lru")) { policy = lru; }
            else if (!std::strcmp(value, "lfu")) { policy = lfu; }
            else if (!std::strcmp(value, "decay")) { policy = decay; }
            else if (!std::strcmp(value, "count_decay2")) { policy = count_decay2; }
            else { throw std::runtime_error("GGML_KIMI_CPU_SINK_CACHE_POLICY must be window, lru, lfu, decay, or count_decay2"); }
        }
        if (const char * value = std::getenv("GGML_KIMI_CPU_SINK_CACHE_KEEP_ACROSS_REQUESTS")) {
            if (!std::strcmp(value, "1")) { keep_across_requests = true; }
            else if (std::strcmp(value, "0")) {
                throw std::runtime_error("GGML_KIMI_CPU_SINK_CACHE_KEEP_ACROSS_REQUESTS must be 0 or 1");
            }
        }
        if (keep_across_requests && policy == window) {
            throw std::runtime_error("CACHE_KEEP_ACROSS_REQUESTS requires a persistent cache policy");
        }
        if (const char * value = std::getenv("GGML_KIMI_CPU_SINK_CACHE_PREFILL_READ")) {
            if (!std::strcmp(value, "1")) { prefill_read = true; }
            else if (std::strcmp(value, "0")) {
                throw std::runtime_error("GGML_KIMI_CPU_SINK_CACHE_PREFILL_READ must be 0 or 1");
            }
        }
        if (prefill_read && !keep_across_requests) {
            throw std::runtime_error("CACHE_PREFILL_READ requires CACHE_KEEP_ACROSS_REQUESTS=1");
        }
        if (const char * value = std::getenv("GGML_KIMI_CPU_SINK_CACHE_ELASTIC")) {
            if (!std::strcmp(value, "1")) { elastic = true; }
            else if (std::strcmp(value, "0")) {
                throw std::runtime_error("GGML_KIMI_CPU_SINK_CACHE_ELASTIC must be 0 or 1");
            }
        }
        if (const char * value = std::getenv("GGML_KIMI_CPU_SINK_CACHE_PARTIAL_RECLAIM")) {
            if (!std::strcmp(value, "1")) { partial_reclaim = true; }
            else if (std::strcmp(value, "0")) {
                throw std::runtime_error("GGML_KIMI_CPU_SINK_CACHE_PARTIAL_RECLAIM must be 0 or 1");
            }
        }
        if (const char * value = std::getenv("GGML_KIMI_CPU_SINK_CACHE_ALLOCATION")) {
            if (!std::strcmp(value, "priority")) { allocation = priority_order; }
            else if (!std::strcmp(value, "uniform")) { allocation = uniform; }
            else if (!std::strcmp(value, "global")) { allocation = global; }
            else { throw std::runtime_error("GGML_KIMI_CPU_SINK_CACHE_ALLOCATION must be priority, uniform, or global"); }
        }
        if (policy == window && allocation == global) {
            throw std::runtime_error("window cache policy requires priority or uniform allocation");
        }
        auto decimal = [](const std::string & text, uint64_t maximum) {
            uint64_t result = 0;
            if (text.empty()) { throw std::runtime_error("empty CPU sink cache setting"); }
            for (const char c : text) {
                if (c < '0' || c > '9' || uint64_t(c - '0') > maximum ||
                        result > (maximum - uint64_t(c - '0')) / 10) {
                    throw std::runtime_error("invalid CPU sink cache integer setting");
                }
                result = result * 10 + uint64_t(c - '0');
            }
            return result;
        };
        const char * prefill_value = std::getenv("GGML_KIMI_CPU_SINK_CACHE_PREFILL_SLOTS");
        if (prefill_value) {
            prefill_slots = size_t(decimal(prefill_value, 256ull * 896));
        }
        if (elastic) {
            if (!prefill_value) {
                throw std::runtime_error("CACHE_ELASTIC requires CACHE_PREFILL_SLOTS");
            }
            if (!keep_across_requests || policy == window || allocation != uniform) {
                throw std::runtime_error("CACHE_ELASTIC requires retention, a persistent policy, and uniform allocation");
            }
        } else if (prefill_value) {
            throw std::runtime_error("CACHE_PREFILL_SLOTS requires CACHE_ELASTIC=1");
        }
        if (partial_reclaim && (!keep_across_requests || policy == window || allocation != uniform)) {
            throw std::runtime_error("CACHE_PARTIAL_RECLAIM requires retention, a persistent policy, and uniform allocation");
        }
        if (const char * value = std::getenv("GGML_KIMI_CPU_SINK_CACHE_LAYER_QUOTAS")) {
            // A bounded inline list is included in the normal environment
            // receipt. Reject partial/duplicate profiles instead of falling
            // back to a different placement silently.
            const size_t length = strnlen(value, 4097);
            if (!length || length > 4096 || allocation != uniform) {
                throw std::runtime_error("CACHE_LAYER_QUOTAS requires uniform allocation and a nonempty list of at most 4096 bytes");
            }
            const std::string profile(value, length);
            size_t start = 0;
            do {
                const size_t end = profile.find(',', start);
                const std::string pair = profile.substr(start, end == std::string::npos ? end : end - start);
                const size_t colon = pair.find(':');
                if (colon == std::string::npos || layer_quotas.size() == 256) {
                    throw std::runtime_error("CACHE_LAYER_QUOTAS must contain at most 256 layer:slots pairs");
                }
                const int32_t id = int32_t(decimal(pair.substr(0, colon), 1023));
                const size_t slots = size_t(decimal(pair.substr(colon + 1), 896));
                if (!layer_quotas.emplace(id, slots).second) {
                    throw std::runtime_error("CACHE_LAYER_QUOTAS contains a duplicate layer");
                }
                if (end == std::string::npos) { break; }
                start = end + 1;
            } while (true);
        }
        if ((elastic || partial_reclaim) && !layer_quotas.empty()) {
            throw std::runtime_error("CACHE_ELASTIC and CACHE_PARTIAL_RECLAIM cannot be combined with CACHE_LAYER_QUOTAS");
        }
        ledger.configure(sink_unified_memory());
        std::fprintf(stderr, "[kimi_host_ledger] unified=%d reserve=%llu margin=%llu explicit_reserve=%llu reclaimable_file=%d cgroup=%s\n",
                int(ledger.unified), (unsigned long long) ledger.reserve, (unsigned long long) ledger.margin,
                (unsigned long long) ledger.explicit_reserve, int(ledger.reclaimable_file),
                ledger.cgroup.empty() ? "-" : ledger.cgroup.c_str());
    }
    ~sink_cache() { release(); }
    void release() {
        if (arena && before_release) { before_release(before_release_arg); }
        if (arena && before_unmap) { before_unmap(before_unmap_arg); }
        if (arena && before_discard) { before_discard(before_discard_arg, 0, entries.size()); }
        ++generation;
        if (arena) {
            if (munmap(arena, stats.mapped_bytes) != 0) { std::abort(); }
            arena = nullptr;
            ++stats.clears;
        }
        stats.mapped_bytes = stats.committed_bytes = stats.enabled_bytes = stats.valid_bytes = stats.page_table_bytes = 0;
        elastic_prefill = adopt = false;
        for (auto & e : entries) { e = {}; e.enabled = true; }
        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            auto & l = layers[layer_index];
            l.seen = false; l.frequency.clear(); l.count_frequency.clear(); l.count_step = 0;
            if (allocation != global) {
                for (size_t index = l.begin; index < l.begin + l.count && index < entries.size(); ++index) {
                    entries[index].layer_index = layer_index;
                }
            }
        }
        if (!entries.empty()) { stats.enabled_bytes = stats.capacity_bytes; }
        clock = age_steps = 0;
    }
    void configure(uint64_t budget, const int32_t * priority, int n, int per_layer) {
        if (budget > std::numeric_limits<size_t>::max() ||
            n < 0 || n > 256 || (n && !priority) ||
            (budget && !n) || per_layer < 1 || per_layer > 896) {
            throw std::runtime_error("invalid CPU sink cache budget, layers or quota");
        }
        std::vector<layer> next;
        next.reserve(size_t(n)); // at most 256 small configuration records
        for (int i = 0; i < n; ++i) {
            if (priority[i] < 0 || priority[i] > 1023 || std::any_of(next.begin(), next.end(),
                    [&](const layer & l) { return l.id == priority[i]; })) {
                throw std::runtime_error("CPU sink cache layers must be distinct integers in [0,1023]");
            }
            layer item; item.id = priority[i]; next.push_back(item);
        }
        if (!layer_quotas.empty()) {
            if (layer_quotas.size() != next.size()) {
                throw std::runtime_error("CACHE_LAYER_QUOTAS must name every configured layer exactly once");
            }
            for (const auto & l : next) {
                const auto item = layer_quotas.find(l.id);
                if (item == layer_quotas.end() || item->second > size_t(per_layer)) {
                    throw std::runtime_error("CACHE_LAYER_QUOTAS contains an unknown layer or exceeds CACHE_MAX_PER_LAYER");
                }
            }
        }
        release();
        layers = std::move(next);
        std::vector<entry>().swap(entries);
        stats = {}; stats.budget_bytes = budget;
        stride = 0; std::fill(role_bytes, role_bytes + 3, 0);
        quota = per_layer; active = has_token = paused = elastic_prefill = false;
        if (ledger.reclaimable_file) {
            std::fprintf(stderr, "[kimi_cpu_sink_cache] reclaimable_file=1 active_file_counts_as_headroom=1 inactive_file_counts_as_headroom=1\n");
        }
        if (!layer_quotas.empty() || ledger.reserve != 2ull * 1024 * 1024 * 1024) {
            std::fprintf(stderr, "[kimi_cpu_sink_cache] fixed_layer_quotas=%zu reserve_bytes=%llu\n",
                    layer_quotas.size(), (unsigned long long) ledger.reserve);
        }
        if (elastic) {
            std::fprintf(stderr,
                    "[kimi_cpu_sink_cache] elastic=1 prefill_slots=%zu reserve_bytes=%llu mapping=retained\n",
                    prefill_slots, (unsigned long long) ledger.reserve);
        }
        if (partial_reclaim) {
            std::fprintf(stderr,
                    "[kimi_cpu_sink_cache] partial_reclaim=1 reserve_bytes=%llu mapping=retained_on_known_pressure\n",
                    (unsigned long long) ledger.reserve);
        }
    }
    void begin(uint64_t next_epoch, uint64_t next_token, bool decode) {
        if (!decode) {
            // No run can be in flight under run_mutex. The optional read-only
            // prefill path may lease retained valid entries, but never trains,
            // admits or evicts here. Recheck headroom before each grouped
            // workspace/stream; an error or explicit configure still unmaps.
            if (elastic && arena) {
                shrink_to(prefill_slots, false, "prefill_phase", true);
                elastic_prefill = true;
                const bool room = partial_reclaim
                        ? reclaim_for_guard(0, "prefill_headroom") : guard(0);
                if (room) {
                    std::fprintf(stderr,
                            "[kimi_cpu_sink_cache] retained_prefill=1 elastic=1 enabled_bytes=%llu resident_bytes=%llu valid_bytes=%llu\n",
                            (unsigned long long) stats.enabled_bytes,
                            (unsigned long long) stats.committed_bytes,
                            (unsigned long long) stats.valid_bytes);
                    for (auto & l : layers) { l.seen = false; }
                } else {
                    ++stats.pressure_stops; count_rejection();
                    std::fprintf(stderr, "[kimi_cpu_sink_cache] paused=1 reason=prefill_headroom term=%s reserve_bytes=%llu\n",
                            kimi_host_ledger::short_term(last_room), (unsigned long long) ledger.reserve);
                    release();
                }
            } else if (keep_across_requests && arena &&
                    (partial_reclaim ? reclaim_for_guard(0, "prefill_headroom") : guard(0))) {
                if (active) {
                    std::fprintf(stderr, "[kimi_cpu_sink_cache] retained_prefill=1 valid_bytes=%llu\n",
                            (unsigned long long) stats.valid_bytes);
                }
                for (auto & l : layers) { l.seen = false; }
            } else {
                if (keep_across_requests && arena) {
                    ++stats.pressure_stops; count_rejection();
                    std::fprintf(stderr, "[kimi_cpu_sink_cache] paused=1 reason=prefill_headroom term=%s reserve_bytes=%llu\n",
                            kimi_host_ledger::short_term(last_room), (unsigned long long) ledger.reserve);
                }
                release();
            }
            active = has_token = paused = false;
            return;
        }
        // A retained mapping is bound to the model/context epoch even after
        // prefill cleared has_token. Per-entry fstat keys also reject fd reuse.
        if (keep_across_requests && arena && next_epoch != epoch && !adopt) { release(); }
        if (adopt) { regrow_adopted(); }
        adopt = false;
        if (has_token && (next_epoch != epoch || token == UINT64_MAX || next_token != token + 1)) { release(); }
        if (elastic) { grow_decode(); }
        epoch = next_epoch; token = next_token; active = has_token = true;
        if (policy == decay || policy == count_decay2) { ++age_steps; }
        if (policy == decay && age_steps % 32 == 0) {
            for (auto & l : layers) {
                for (auto & item : l.frequency) { item.second >>= 1; }
            }
        }
    }
    // Room for the cache's own admission: both ledger terms, and only under a readable finite cgroup. full_boost: the
    // system term at the kernel's full watermark boost, as the ledger's make_room measures it (kimi_host_ledger::measure).
    room_state check_room(uint64_t extra, uint64_t * shortage, uint64_t * surplus = nullptr, bool full_boost = false) const {
        if (shortage) { *shortage = 0; }
        if (surplus) { *surplus = 0; }
        const auto r = ledger.measure(extra, full_boost);
        last_room = r;
        if (r.cgroup == room_unreadable) { return room_unreadable; }
        if (shortage) { *shortage = r.shortage(); }
        if (surplus) { *surplus = r.surplus(); }
        return r.fits() ? room_ok : room_pressure;
    }
    std::pair<uint64_t, double> score(size_t layer_id, int32_t expert) const {
        if (policy == count_decay2) {
            const auto & item = layers.at(layer_id);
            const auto found = item.count_frequency.find(expert);
            const double value = found == item.count_frequency.end() ? 0.0 : found->second;
            return { 0, value * std::exp2(-double(age_steps - item.count_step) / 2.0) };
        }
        const auto & counts = layers.at(layer_id).frequency;
        const auto found = counts.find(expert);
        return { found == counts.end() ? uint64_t(0) : found->second, 0.0 };
    }
    bool colder(size_t left, size_t right) const {
        const entry & a = entries[left];
        const entry & b = entries[right];
        if (a.valid != b.valid) { return !a.valid; }
        const auto a_score = score(a.layer_index, a.id);
        const auto b_score = score(b.layer_index, b.id);
        if (policy != lru && a_score != b_score) { return a_score < b_score; }
        if (a.last_touch != b.last_touch) { return a.last_touch < b.last_touch; }
        return left < right;
    }
    size_t enabled_slots() const {
        return size_t(std::count_if(entries.begin(), entries.end(), [](const entry & e) { return e.enabled; }));
    }
    size_t shrink_to(size_t target, bool guard_reclaim, const char * reason, bool strict) {
        if ((!elastic && !partial_reclaim) || target > entries.size() || !stride || stride % page_size) {
            throw std::runtime_error("invalid CPU sink cache shrink");
        }
        const size_t before = enabled_slots();
        if (target >= before) { return before; }
        const uint64_t started = ggml_time_us();
        const uint64_t released_before = stats.released_bytes;
        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            const auto & item = layers[layer_index];
            const size_t wanted = target / layers.size() + (layer_index < target % layers.size());
            if (wanted > item.count) { throw std::runtime_error("cache shrink target exceeds a layer's maximum quota"); }
            size_t have = 0;
            std::vector<size_t> candidates;
            candidates.reserve(item.count);
            for (size_t index = item.begin; index < item.begin + item.count; ++index) {
                entry & e = entries[index];
                if (!e.enabled) { continue; }
                ++have;
                if (!e.leases && !e.reserved && !e.protected_now && !e.inflight) { candidates.push_back(index); }
            }
            if (have <= wanted) { continue; }
            std::sort(candidates.begin(), candidates.end(), [&](size_t a, size_t b) { return colder(a, b); });
            const size_t count = std::min(have - wanted, candidates.size());
            for (size_t offset = 0; offset < count; ++offset) {
                entry & e = entries[candidates[offset]];
                if (e.resident) {
                    if (arena && before_discard) { before_discard(before_discard_arg, candidates[offset], 1); }
                    if (!arena || madvise(arena + candidates[offset] * stride, stride, MADV_DONTNEED) != 0) {
                        const auto error = sys_error("CPU sink cache MADV_DONTNEED");
                        release();
                        throw error;
                    }
                    e.resident = false;
                    e.was_released = true;
                    stats.committed_bytes -= stride;
                    stats.released_bytes = add(stats.released_bytes, stride);
                }
                if (e.valid) {
                    e.valid = false;
                    stats.valid_bytes -= stride;
                    ++stats.evictions;
                }
                e.enabled = false;
                stats.enabled_bytes -= stride;
            }
        }
        const size_t after = enabled_slots();
        if (strict && after != target) {
            throw std::runtime_error("elastic CPU sink cache shrink found a busy slot at a phase boundary");
        }
        if (after < before) {
            ++stats.shrink_ops;
            if (guard_reclaim) { ++stats.guard_reclaims; }
            stats.shrink_us = add(stats.shrink_us, ggml_time_us() - started);
            std::fprintf(stderr,
                    "[kimi_cpu_sink_cache] cache_shrink=1 reason=%s target_slots=%zu before_slots=%zu after_slots=%zu released_bytes=%llu valid_bytes=%llu\n",
                    reason, target, before, after,
                    (unsigned long long) (stats.released_bytes - released_before),
                    (unsigned long long) stats.valid_bytes);
        }
        return after;
    }
    void grow_decode() {
        if (!elastic || !elastic_prefill || entries.empty()) { elastic_prefill = false; return; }
        const size_t before = enabled_slots();
        for (auto & e : entries) { e.enabled = true; }
        stats.enabled_bytes = multiply(entries.size(), stride);
        elastic_prefill = false;
        if (before != entries.size()) {
            ++stats.phase_grows;
            std::fprintf(stderr,
                    "[kimi_cpu_sink_cache] elastic_grow=1 before_slots=%zu after_slots=%zu resident_bytes=%llu\n",
                    before, entries.size(), (unsigned long long) stats.committed_bytes);
        }
    }
    // Slots given back until the room for `extra` fits; full_boost for the ledger's make_room, which then measures at the
    // full watermark boost: stopping at MemAvailable as it is would leave make_room short by the boost's headroom.
    bool reclaim_for_guard(uint64_t extra, const char * reason, bool full_boost = false) {
        stats.last_guard_extra_bytes = extra;
        stats.peak_guard_extra_bytes = std::max(stats.peak_guard_extra_bytes, extra);
        uint64_t shortage = 0;
        room_state state = check_room(extra, &shortage, nullptr, full_boost);
        if (state == room_ok) { return true; }
        if (state == room_unreadable || !partial_reclaim || !arena || !stride) { return false; }
        size_t before = enabled_slots();
        while (state == room_pressure && before) {
            const size_t slots = size_t(std::min<uint64_t>(before, ceil_div(shortage, stride)));
            const size_t after = shrink_to(before - std::max<size_t>(slots, 1), true, reason, false);
            if (after >= before) { break; }
            before = after;
            state = check_room(extra, &shortage, nullptr, full_boost);
        }
        return state == room_ok;
    }
    void reject_admission(std::vector<sink_request> & requests, kimi_cpu_sink_stats & current, const char * reason) {
        count_rejection();
        if (!partial_reclaim) { pause(requests, current, reason, kimi_host_ledger::short_term(last_room)); return; }
        for (auto & r : requests) {
            if (r.cache_admit >= 0) { entries[size_t(r.cache_admit)].reserved = false; }
            r.cache_hit = r.cache_admit = -1;
        }
        for (auto & e : entries) { e.protected_now = false; }
        current.cache_hits = current.cache_new_slot_bytes = 0;
        ++stats.pressure_stops;
        std::fprintf(stderr,
                "[kimi_cpu_sink_cache] admission_rejected=1 reason=%s term=%s mapping_retained=1 reserve_bytes=%llu extra_bytes=%llu enabled_slots=%zu\n",
                reason, kimi_host_ledger::short_term(last_room), (unsigned long long) ledger.reserve,
                (unsigned long long) stats.last_guard_extra_bytes, enabled_slots());
    }
    static source_key key(const kimi_cpu_sink_source & s) {
        struct stat st;
        if (fstat(s.fd_direct, &st) != 0) { throw sys_error("CPU sink cache fstat"); }
        const auto v = kimi_vram_source_identity(&st, s.offset + s.data_offset, s.payload_nbytes ? s.payload_nbytes : s.nbytes);
        return { v.device, v.inode, v.size, v.offset, v.bytes, v.mtime_s, v.mtime_ns, v.ctime_s, v.ctime_ns };
    }
    static bool matches(const entry & e, const sink_request & r) {
        for (int role = 0; role < 3; ++role) { if (!(e.source[role] == key(r.role[role]))) { return false; } }
        return true;
    }
    bool geometry(const size_t * requested, const ggml_tensor * tensor, size_t max_requests) {
        // Native GGUF roles can differ only in page-prefix padding across
        // layers. Keep the largest buffers instead of clearing the cache as
        // those physical read lengths alternate.
        const size_t bytes[3] = { std::max(requested[0], role_bytes[0]),
            std::max(requested[1], role_bytes[1]), std::max(requested[2], role_bytes[2]) };
        if (std::equal(bytes, bytes + 3, role_bytes)) { return true; }
        const uint64_t sum = add(add(bytes[0], bytes[1]), bytes[2]);
        if (!sum || sum > std::numeric_limits<size_t>::max()) { throw std::runtime_error("invalid CPU sink cache stride"); }
        const size_t next_stride = size_t(sum);
        uint64_t budget = stats.budget_bytes;
        if (budget == auto_budget) {
            // With a unified-memory GPU the GPU tier keeps the same experts in the same DRAM and computes
            // them faster, so the RAM cache keeps only the slots the next-layer prefetch lands in. Otherwise
            // it takes what the cgroup guard would admit now.
            // The receipt names both terms: on a discrete GPU the cgroup term should be the smaller one.
            uint64_t surplus = 0;
            const bool readable = check_room(0, nullptr, &surplus) == room_ok;
            budget = sink_unified_memory()
                    ? multiply(multiply(layers.size(), uint64_t(std::max(1, staging_per_layer))), next_stride)
                    : (readable ? surplus : 0);
            const auto & r = last_room;
            std::fprintf(stderr, "[kimi_cpu_sink_cache] auto_budget=%llu unified=%d slot_bytes=%zu cgroup_surplus=%llu system_surplus=%llu available=%llu floor=%llu binding=%s\n",
                    (unsigned long long) budget, int(sink_unified_memory()), next_stride,
                    (unsigned long long) (r.cgroup == room_unreadable ? 0 : r.cgroup_surplus), (unsigned long long) r.system_surplus,
                    (unsigned long long) r.available, (unsigned long long) r.floor,
                    !readable ? "unreadable" : !r.system || r.cgroup_surplus <= r.system_surplus ? "cgroup" : "system");
        }
        size_t count = static_cast<size_t>(std::min(budget / next_stride,
                                                  multiply(layers.size(), uint64_t(quota))));
        if (stats.budget_bytes == auto_budget && !sink_unified_memory()) {
            // Price the same page rounding/tables and allocation guards used
            // by admission. Payload-only division can spend their headroom,
            // making an automatically chosen arena pause before it fills.
            const uint64_t pages = add(ceil_div(next_stride, page_size), next_stride % page_size != 0);
            const uint64_t payload = multiply(pages, page_size);
            const uint64_t slot_cost = add(payload, page_tables(payload));
            const bool frequencies = policy == count_decay2 || policy == lfu || policy == decay;
            const uint64_t node = sizeof(std::pair<const int32_t, uint64_t>) + 4 * sizeof(void *);
            const uint64_t node_storage = frequencies
                    ? allocation_cost(multiply(multiply(layers.size(), uint64_t(tensor->ne[2])), node)) : 0;
            const uint64_t node_guard = frequencies ? multiply(max_requests, allocation_cost(node)) : 0;
            const auto cost = [&](size_t slots) {
                if (!slots) { return uint64_t(0); }
                return add(add(multiply(slots, slot_cost), allocation_cost(multiply(slots, sizeof(entry)))),
                           add(add(node_storage, node_guard), 64 * 1024));
            };
            size_t low = 0, high = count;
            while (low < high) {
                const size_t middle = low + (high - low + 1) / 2;
                if (cost(middle) <= budget) { low = middle; }
                else { high = middle - 1; }
            }
            count = low;
            slot_cost_bytes = slot_cost;
            metadata_bytes = add(add(node_storage, node_guard), 64 * 1024);
            std::fprintf(stderr,
                    "[kimi_cpu_sink_cache] auto_capacity_bytes=%llu capacity_cost_bytes=%llu surplus_bytes=%llu slot_cost_bytes=%llu metadata_node_storage_bytes=%llu metadata_guard_bytes=%llu\n",
                    (unsigned long long) multiply(count, next_stride), (unsigned long long) cost(count),
                    (unsigned long long) budget, (unsigned long long) slot_cost,
                    (unsigned long long) node_storage, (unsigned long long) node_guard);
        }
        const uint64_t capacity = multiply(count, next_stride);
        if ((elastic || partial_reclaim) && next_stride % page_size) {
            throw std::runtime_error("CACHE_ELASTIC and CACHE_PARTIAL_RECLAIM require a page-aligned expert slot stride");
        }
        if (elastic && prefill_slots > count) {
            throw std::runtime_error("CACHE_PREFILL_SLOTS exceeds the decode cache slot count");
        }
        if (!layer_quotas.empty()) {
            uint64_t total = 0;
            for (const auto & item : layer_quotas) { total = add(total, item.second); }
            if (total != count) {
                throw std::runtime_error("CACHE_LAYER_QUOTAS sum must equal the cache budget's physical slot count");
            }
        }
        if (count > size_t(INT_MAX) || count > entries.max_size() ||
                capacity > uint64_t(std::numeric_limits<ptrdiff_t>::max())) {
            throw std::runtime_error("CPU sink cache geometry exceeds addressable storage");
        }
        // Check before vector value-initialization can touch metadata pages.
        // Existing allocations are already included in memory.current.
        if (!reclaim_for_guard(allocation_cost(multiply(count, sizeof(entry))), "cgroup_metadata_headroom")) { return false; }
        std::vector<entry> next(count);
        release();
        // Geometry is first learned during the first decode token's prepare.
        age_steps = active ? 1 : 0;
        std::copy(bytes, bytes + 3, role_bytes);
        stride = next_stride;
        entries = std::move(next);
        size_t assigned = 0;
        for (size_t i = 0; i < layers.size(); ++i) {
            auto & l = layers[i];
            l.begin = assigned;
            l.count = allocation == uniform ? count / layers.size() + (i < count % layers.size()) :
                std::min(size_t(quota), count - assigned);
            if (!layer_quotas.empty()) { l.count = layer_quotas.at(l.id); }
            if (allocation == global) { l.begin = 0; l.count = count; }
            if (allocation == global) { continue; }
            for (size_t index = l.begin; index < l.begin + l.count; ++index) {
                entries[index].layer_index = i;
            }
            assigned += l.count;
        }
        for (auto & e : entries) { e.enabled = true; }
        stats.capacity_bytes = capacity;
        stats.enabled_bytes = capacity;
        return true;
    }
    void prepare(std::vector<sink_request> & requests, const ggml_tensor * const * tensors,
                 const size_t * bytes, kimi_cpu_sink_stats & current) {
        if (!active || paused || !stats.budget_bytes) { return; }
        int id = -1, consumed = 0;
        if (std::sscanf(tensors[0]->name, "blk.%d.ffn_up_exps.weight%n", &id, &consumed) != 1 ||
            !consumed || tensors[0]->name[consumed]) { return; }
        char name[GGML_MAX_NAME];
        std::snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", id);
        if (std::strcmp(name, tensors[1]->name)) { return; }
        std::snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", id);
        if (std::strcmp(name, tensors[2]->name)) { return; }
        auto found = std::find_if(layers.begin(), layers.end(), [&](const layer & l) { return l.id == id; });
        if (found == layers.end()) { return; }
        try {
            if (!requests.empty() && !geometry(bytes, tensors[0], requests.size())) { reject_admission(requests, current, "cgroup_metadata_headroom"); return; }
        } catch (const std::bad_alloc &) { pause(requests, current, "metadata_allocation_failed"); return; }
        layer & l = *found;
        const size_t layer_index = size_t(found - layers.begin());
        if (!l.count || (l.seen && l.token == token)) { return; }
        // Entries from a skipped token are not a previous-token window.
        const bool previous = l.seen && l.token != UINT64_MAX && l.token + 1 == token;
        for (size_t i = l.begin; i < l.begin + l.count; ++i) {
            auto & e = entries[i];
            if (e.leases) { throw std::runtime_error("CPU sink prepare with an outstanding cache lease"); }
            e.protected_now = false;
            if (!e.enabled) { e.reserved = false; continue; }
            if (!e.valid) { e.reserved = false; continue; }
            if (e.layer_index != layer_index) { continue; }
            auto r = std::find_if(requests.begin(), requests.end(), [&](const sink_request & request) { return request.id == e.id; });
            if ((policy == window && (!previous || r == requests.end())) ||
                    (r != requests.end() && !matches(e, *r))) {
                e.valid = e.reserved = false; ++stats.evictions; stats.valid_bytes -= stride;
            }
        }
        l.seen = true; l.token = token;
        if (policy == count_decay2) {
            size_t nodes = 0;
            for (const auto & r : requests) { nodes += !r.cache_only && l.count_frequency.find(r.id) == l.count_frequency.end(); }
            if (!reclaim_for_guard(multiply(nodes, allocation_cost(sizeof(std::pair<const int32_t, double>) + 4 * sizeof(void *))),
                    "cgroup_count_frequency_headroom")) {
                reject_admission(requests, current, "cgroup_count_frequency_headroom"); return;
            }
            const double factor = std::exp2(-double(age_steps - l.count_step) / 2.0);
            for (auto & item : l.count_frequency) { item.second *= factor; }
            l.count_step = age_steps;
            try {
                // Selection frequency, independent of routing weight. Count a
                // duplicated ID once per layer/token, as the offline set did.
                for (size_t j = 0; j < requests.size(); ++j) {
                    if (requests[j].cache_only) { continue; }
                    const auto duplicate = std::find_if(requests.begin(), requests.begin() + j,
                        [&](const sink_request & r) { return r.id == requests[j].id; });
                    if (duplicate == requests.begin() + j) { l.count_frequency[requests[j].id] += 1.0; }
                }
            } catch (const std::bad_alloc &) { pause(requests, current, "count_frequency_allocation_failed"); return; }
        }
        if (policy == lfu || policy == decay) {
            size_t nodes = 0;
            for (const auto & r : requests) { nodes += !r.cache_only && l.frequency.find(r.id) == l.frequency.end(); }
            // Each map node may fault a different allocator page/THP. Duplicate
            // IDs deliberately overcount; no allocation occurs before this guard.
            if (!reclaim_for_guard(multiply(nodes, allocation_cost(sizeof(std::pair<const int32_t, uint64_t>) + 4 * sizeof(void *))),
                    "cgroup_frequency_headroom")) {
                reject_admission(requests, current, "cgroup_frequency_headroom"); return;
            }
            try {
                for (const auto & r : requests) {
                    if (r.cache_only) { continue; }
                    auto & frequency = l.frequency[r.id];
                    if (frequency != UINT64_MAX) { ++frequency; }
                }
            } catch (const std::bad_alloc &) { pause(requests, current, "frequency_allocation_failed"); return; }
        }
        // Protect every resident hit before reserving any direct-read target.
        // A miss at the front of the batch must never overwrite a later hit.
        for (auto & r : requests) {
            for (size_t i = l.begin; i < l.begin + l.count; ++i) {
                if (entries[i].enabled && entries[i].valid && entries[i].layer_index == layer_index &&
                        entries[i].id == r.id && matches(entries[i], r)) {
                    r.cache_hit = int(i);
                    entries[i].protected_now = true;
                    if (entries[i].prefetched && !r.cache_only) { entries[i].prefetched = false; ++prefetch_hits; }
                    if (!r.cache_only) { entries[i].last_touch = ++clock; }
                    break;
                }
            }
            if (r.cache_hit >= 0 && !r.cache_only) {
                ++current.cache_hits;
                current.cache_hit_bytes = add(current.cache_hit_bytes, stride);
            }
        }
        for (auto & r : requests) {
            if (r.cache_hit >= 0 || r.cache_only) { continue; }
            const bool already_reserved = std::any_of(entries.begin() + l.begin, entries.begin() + l.begin + l.count,
                [&](const entry & e) { return e.enabled && e.reserved && e.layer_index == layer_index && e.id == r.id; });
            if (already_reserved) { ++stats.admission_skips; continue; }
            auto free = std::find_if(entries.begin() + l.begin, entries.begin() + l.begin + l.count,
                [](const entry & e) { return e.enabled && !e.valid && !e.reserved && !e.inflight; });
            if (free == entries.begin() + l.begin + l.count && policy != window) {
                auto frequency = [&](const entry & e) { return score(e.layer_index, e.id); };
                for (auto candidate = entries.begin() + l.begin; candidate != entries.begin() + l.begin + l.count; ++candidate) {
                    if (!candidate->enabled || candidate->protected_now || candidate->reserved || candidate->inflight) { continue; }
                    if (free == entries.begin() + l.begin + l.count ||
                            (policy != lru && frequency(*candidate) < frequency(*free)) ||
                            ((policy == lru || frequency(*candidate) == frequency(*free)) && candidate->last_touch < free->last_touch)) {
                        free = candidate;
                    }
                }
                if (free != entries.begin() + l.begin + l.count && policy != lru && score(layer_index, r.id) < frequency(*free)) {
                    free = entries.begin() + l.begin + l.count;
                }
                if (free != entries.begin() + l.begin + l.count) {
                    free->valid = false;
                    ++stats.evictions; stats.valid_bytes -= stride;
                    if (free->prefetched) { free->prefetched = false; ++prefetch_unused; }
                }
            }
            if (free == entries.begin() + l.begin + l.count) { ++stats.admission_skips; continue; }
            free->id = r.id; free->layer_index = layer_index; free->reserved = true; free->last_touch = ++clock;
            for (int role = 0; role < 3; ++role) { free->source[role] = key(r.role[role]); }
            r.cache_admit = int(free - entries.begin());
        }
        uint64_t new_slots = 0;
        for (const auto & r : requests) {
            if (r.cache_admit >= 0 && !entries[r.cache_admit].resident) { ++new_slots; }
        }
        // The arena is page aligned; a non-page-aligned stride can straddle one
        // additional page. No unused slot or whole-layer quota is charged here.
        const uint64_t slot_pages = add(ceil_div(stride, page_size), stride % page_size != 0);
        const uint64_t payload_pages = multiply(slot_pages, page_size);
        const uint64_t new_tables = multiply(new_slots, page_tables(payload_pages));
        const uint64_t extra = add(add(multiply(new_slots, payload_pages), new_tables), arena ? 0 : 64 * 1024);
        if (!reclaim_for_guard(extra, "cgroup_headroom")) {
            reject_admission(requests, current, "cgroup_headroom"); return;
        }
        if (!arena && stats.capacity_bytes && !requests.empty()) {
            if (const char * failed = map_arena()) { pause(requests, current, failed); return; }
        }
        // MADV_DONTNEED may retain upper-level page tables. Keep a lifetime
        // upper bound, while capping refaults so repeated phase churn cannot
        // make the diagnostic grow beyond one per mapped slot.
        const uint64_t maximum_tables = multiply(entries.size(), page_tables(payload_pages));
        stats.page_table_bytes = std::min(stats.page_table_bytes, maximum_tables);
        stats.page_table_bytes += std::min(new_tables, maximum_tables - stats.page_table_bytes);
        for (const auto & r : requests) {
            if (r.cache_admit < 0 || entries[r.cache_admit].resident) { continue; }
            if (entries[r.cache_admit].was_released) {
                stats.refaulted_bytes = add(stats.refaulted_bytes, stride);
                entries[r.cache_admit].was_released = false;
            }
            entries[r.cache_admit].resident = true;
            stats.committed_bytes = add(stats.committed_bytes, stride);
            current.cache_new_slot_bytes = add(current.cache_new_slot_bytes, stride);
        }
        stats.peak_bytes = std::max(stats.peak_bytes, stats.committed_bytes);
    }
    // The arena, address space only: the guard admits each resident slot against the cgroup, so the kernel's
    // overcommit heuristic must not refuse a mapping larger than RAM plus swap. Ordinary pages: partial reclaim gives
    // single slots back (MADV_DONTNEED). nullptr, or why it failed.
    const char * map_arena() {
        void * memory = mmap(nullptr, stats.capacity_bytes, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (memory == MAP_FAILED) { return "mmap_failed"; }
        arena = static_cast<char *>(memory); stats.mapped_bytes = stats.capacity_bytes;
        return madvise(arena, stats.mapped_bytes, MADV_NOHUGEPAGE) != 0 ? "nohugepage_failed" : nullptr;
    }
    // A cache that can give room back to the ledger: partial reclaim, or elastic (its prefill shrinks to its prefill
    // slots).
    bool gives_room_back() const { return partial_reclaim || elastic; }
    // At load (the sink's load hook, off unified memory): the geometry from the expert tensors' strides (bytes: each
    // role as a decode request reserves it; max_requests: the model's experts a token, as the first decode layer's
    // requests price the frequency nodes its admissions add), the arena, and the slots the guard admits now (all of
    // them when the cache was sized from this room) committed and paged in from `threads` threads, so that no decode
    // token pays their first touch (13.8 ms a token over decode tokens 1-128 on 125, 1.77 s a cold start). Only a
    // cache kept across requests keeps them through the first prefill, which may give slots back through the ledger
    // as every later request's prefill does; the first decode adopts the arena whatever its epoch, and takes back
    // what the room allows then (regrow_adopted). So only a cache that can give room back takes its storage here
    // (partial reclaim, or elastic): an arena that cannot would refuse the first prefill's ledger requests, where a
    // cache sized at the first decode token comes after them. Returns the slots committed.
    size_t reserve_at_load(const size_t * bytes, const ggml_tensor * tensor, size_t max_requests, int threads, double & us) {
        us = 0;
        if (!keep_across_requests || !gives_room_back() || paused || !stats.budget_bytes || arena || layers.empty()) {
            return 0;
        }
        std::vector<std::thread> pool;
        pool.reserve(size_t(std::max(1, threads))); // before anything changes: a failure here leaves the cache as it was
        if (!geometry(bytes, tensor, std::max<size_t>(1, max_requests)) || !stats.capacity_bytes) { return 0; }
        const uint64_t slot_pages = add(ceil_div(stride, page_size), stride % page_size != 0);
        const uint64_t payload_pages = multiply(slot_pages, page_size), tables = page_tables(payload_pages);
        auto cost = [&](size_t n) { return add(multiply(n, add(payload_pages, tables)), 64 * 1024); };
        uint64_t surplus = 0;
        if (check_room(0, nullptr, &surplus) == room_unreadable) { return 0; }
        size_t n = std::min<size_t>(entries.size(), surplus > 64 * 1024 ? (surplus - 64 * 1024) / add(payload_pages, tables) : 0);
        if (!n || !guard(cost(n))) { return 0; }
        if (map_arena()) { release(); return 0; }
        const auto start = sink_clock::now();
        const size_t workers = size_t(std::max(1, std::min(threads, int(n))));
        std::atomic<bool> touched{false};
        auto populate = [&](size_t w) {
            char * begin = arena + n * w / workers * stride, * end = arena + n * (w + 1) / workers * stride;
#ifdef MADV_POPULATE_WRITE
            if (begin == end || madvise(begin, size_t(end - begin), MADV_POPULATE_WRITE) == 0) { return; }
#endif
            touched = true;
            for (char * page = begin; page < end; page += page_size) { *reinterpret_cast<volatile char *>(page) = 0; }
        };
        // this thread takes part 0 and whatever no started thread took (the pool's storage is reserved, so only a
        // thread's creation can fail, and then nothing was added)
        size_t started = 1;
        try {
            for (; started < workers; ++started) { pool.emplace_back(populate, started); }
        } catch (...) {}
        for (size_t w = started; w < workers; ++w) { populate(w); }
        populate(0);
        for (auto & t : pool) { t.join(); }
        us = elapsed_us(start);
        for (size_t i = 0; i < n; ++i) { entries[i].resident = true; }
        stats.committed_bytes = multiply(n, stride);
        stats.page_table_bytes = multiply(n, tables);
        stats.peak_bytes = std::max(stats.peak_bytes, stats.committed_bytes);
        adopt = true;
        std::fprintf(stderr, "[kimi_cpu_sink_cache] reason=load slots=%zu of=%zu slot_bytes=%zu committed_bytes=%llu us=%.0f threads=%zu pages=%s\n",
                n, entries.size(), stride, (unsigned long long) stats.committed_bytes, us, workers,
                touched ? "touched" : "populated");
        return n;
    }
    // The first decode token after reserve_at_load (partial reclaim, not elastic): the slots the first prefill gave
    // back come back as far as the room allows now (an automatic budget: geometry's price of a slot past the
    // metadata room; a fixed one: all of them), so the cache holds what one sized at this token would, within the
    // arena's slots. They are paged in at their admission, as every slot was without the load step. So are the
    // enabled slots the load step did not commit (the room it saw was smaller than the one geometry sized from):
    // with an automatic budget they take their room first (owed).
    void regrow_adopted() {
        if (!partial_reclaim || elastic || !arena || !stride || paused) { return; }
        const size_t before = enabled_slots();
        uint64_t more = entries.size() - before, surplus = 0, owed = 0;
        if (!more) { return; }
        if (stats.budget_bytes == auto_budget) {
            if (check_room(0, nullptr, &surplus) == room_unreadable || !slot_cost_bytes) { return; }
            for (const auto & e : entries) { owed += e.enabled && !e.resident; }
            const uint64_t room = surplus > metadata_bytes ? (surplus - metadata_bytes) / slot_cost_bytes : 0;
            more = std::min<uint64_t>(more, room > owed ? room - owed : 0);
        }
        const size_t target = before + size_t(more);
        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            const auto & item = layers[layer_index];
            const size_t wanted = target / layers.size() + (layer_index < target % layers.size());
            size_t have = 0;
            for (size_t i = item.begin; i < item.begin + item.count; ++i) { have += entries[i].enabled; }
            for (size_t i = item.begin; i < item.begin + item.count && have < wanted; ++i) {
                if (entries[i].enabled) { continue; }
                entries[i].enabled = true; ++have;
                stats.enabled_bytes = add(stats.enabled_bytes, stride);
            }
        }
        std::fprintf(stderr, "[kimi_cpu_sink_cache] load_regrow=1 before_slots=%zu after_slots=%zu of=%zu owed_slots=%llu surplus_bytes=%llu committed_bytes=%llu\n",
                before, enabled_slots(), entries.size(), (unsigned long long) owed, (unsigned long long) surplus,
                (unsigned long long) stats.committed_bytes);
    }
    // Host memory the cache may still commit before a reset: its enabled but not yet resident slots (an elastic
    // cache enables all of them again at the next decode phase), or before the first decode token maps the arena,
    // what geometry() will size it to for this slot size. Off unified memory the automatic size depends on the
    // cgroup at that time, and only the GPU tier of a unified device asks.
    uint64_t pending_bytes(size_t slot_bytes) const {
        if (!stats.budget_bytes || paused) { return 0; }
        if (arena) {
            const uint64_t usable = elastic ? stats.capacity_bytes : stats.enabled_bytes;
            return usable > stats.committed_bytes ? usable - stats.committed_bytes : 0;
        }
        if (!slot_bytes || layers.empty()) { return 0; }
        uint64_t budget = stats.budget_bytes;
        if (budget == auto_budget) {
            if (!sink_unified_memory()) { return 0; }
            budget = multiply(multiply(layers.size(), uint64_t(std::max(1, staging_per_layer))), slot_bytes);
        }
        return multiply(std::min(budget / slot_bytes, multiply(layers.size(), uint64_t(quota))), slot_bytes);
    }
    char * data(int index, int role) const {
        size_t offset = size_t(index) * stride;
        for (int r = 0; r < role; ++r) { offset += role_bytes[r]; }
        return arena + offset;
    }
    // Whether the layer's slots hold the expert or a read of it is on its way. Changes nothing: the next-layer
    // prefetch's statistics ask it for every rank of a hint, read or not.
    bool holds(int32_t layer_id, int32_t expert) const {
        if (!active || !arena) { return false; }
        const auto found = std::find_if(layers.begin(), layers.end(), [&](const layer & l) { return l.id == layer_id; });
        if (found == layers.end()) { return false; }
        const size_t layer_index = size_t(found - layers.begin());
        for (size_t i = found->begin; i < found->begin + found->count; ++i) {
            const auto & e = entries[i];
            if (e.enabled && (e.valid || e.reserved || e.inflight) && e.layer_index == layer_index && e.id == expert) { return true; }
        }
        return false;
    }
    // Next-layer prefetch: take a resident, idle slot of the layer (a free one first, else the coldest) for an
    // expert that is not cached. The slot is inflight, invisible to lookups and allocation, until prefetch_done
    // publishes it (or prefetch_activate publishes it early as a hit that its run waits for).
    // Returns -2 when the expert is already cached or on its way, -1 when there is no slot.
    int prefetch_reserve(int32_t layer_id, int32_t expert, const kimi_cpu_sink_source * roles) {
        if (!active || paused || !stats.budget_bytes || !arena || !stride) { return -1; }
        const auto found = std::find_if(layers.begin(), layers.end(), [&](const layer & l) { return l.id == layer_id; });
        if (found == layers.end() || !found->count) { return -1; }
        const size_t layer_index = size_t(found - layers.begin());
        for (int role = 0; role < 3; ++role) { if (roles[role].nbytes > role_bytes[role]) { return -1; } }
        source_key keys[3];
        for (int role = 0; role < 3; ++role) { keys[role] = key(roles[role]); }
        int victim = -1;
        for (size_t i = found->begin; i < found->begin + found->count; ++i) {
            const auto & e = entries[i];
            if (!e.enabled) { continue; }
            if ((e.valid || e.reserved || e.inflight) && e.layer_index == layer_index && e.id == expert &&
                    e.source[0] == keys[0] && e.source[1] == keys[1] && e.source[2] == keys[2]) { return -2; }
            if (!e.resident || e.reserved || e.inflight || e.leases || e.protected_now) { continue; }
            if (victim < 0 || (entries[size_t(victim)].valid && (!e.valid || colder(i, size_t(victim))))) { victim = int(i); }
        }
        if (victim < 0) { return -1; }
        auto & e = entries[size_t(victim)];
        if (e.valid) {
            e.valid = false; ++stats.evictions; stats.valid_bytes -= stride;
            if (e.prefetched) { ++prefetch_unused; }
        }
        e.id = expert; e.layer_index = layer_index; e.inflight = true; e.prefetched = false;
        e.generation = generation; e.last_touch = ++clock;
        for (int role = 0; role < 3; ++role) { e.source[role] = keys[role]; }
        return victim;
    }
    void prefetch_done(int index, uint64_t reserved_generation, bool ok) {
        if (index < 0 || size_t(index) >= entries.size() || reserved_generation != generation) { return; }
        auto & e = entries[size_t(index)];
        if (!e.inflight || e.generation != reserved_generation) { return; }
        e.inflight = false;
        if (e.valid) { // published early as a hit
            if (!ok) { e.valid = false; e.prefetched = false; e.id = -1; stats.valid_bytes -= stride; }
            return;
        }
        if (ok && e.enabled && e.resident) {
            e.valid = true; e.prefetched = true; e.last_touch = ++clock;
            stats.valid_bytes += stride;
        } else {
            e.id = -1;
        }
    }
    bool prefetch_activate(int index, uint64_t reserved_generation) {
        if (index < 0 || size_t(index) >= entries.size() || reserved_generation != generation) { return false; }
        auto & e = entries[size_t(index)];
        if (!e.inflight || e.valid || e.generation != reserved_generation || !e.enabled || !e.resident) { return false; }
        e.valid = true; e.prefetched = true; e.last_touch = ++clock;
        stats.valid_bytes += stride;
        return true;
    }
    void prepare_prefill_read(std::vector<sink_request> & requests,
            const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * down,
            kimi_cpu_sink_stats & current) {
        if (!prefill_read || !arena || !stride || !stats.valid_bytes || requests.empty()) { return; }
        int layer_id = -1, consumed = 0;
        if (std::sscanf(up->name, "blk.%d.ffn_up_exps.weight%n", &layer_id, &consumed) != 1 ||
                !consumed || up->name[consumed]) { return; }
        char name[GGML_MAX_NAME];
        std::snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", layer_id);
        if (std::strcmp(name, gate->name)) { return; }
        std::snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", layer_id);
        if (std::strcmp(name, down->name)) { return; }
        const auto found = std::find_if(layers.begin(), layers.end(),
            [&](const layer & l) { return l.id == layer_id; });
        if (found == layers.end() || !found->count) { return; }
        const size_t layer_index = size_t(found - layers.begin());
        size_t bytes[3] = {};
        for (int role = 0; role < 3; ++role) {
            for (const auto & request : requests) {
                bytes[role] = std::max(bytes[role], request.role[role].nbytes);
            }
        }
        if (bytes[0] > role_bytes[0] || bytes[1] > role_bytes[1] || bytes[2] > role_bytes[2]) { return; }

        for (auto & request : requests) {
            if (request.vram_slot >= 0 || request.cache_hit >= 0) { continue; }
            for (size_t index = found->begin; index < found->begin + found->count; ++index) {
                auto & e = entries[index];
                if (!e.enabled || !e.valid || !e.resident || e.layer_index != layer_index ||
                        e.id != request.id || !matches(e, request)) { continue; }
                acquire_lease(int(index), request);
                request.cache_hit = int(index);
                request.cache_lease = true;
                e.protected_now = true;
                ++current.cache_hits;
                current.cache_hit_bytes = add(current.cache_hit_bytes, stride);
                ++stats.prefill_hits;
                stats.prefill_hit_bytes = add(stats.prefill_hit_bytes, stride);
                ++stats.prefill_leases;
                break;
            }
        }
    }
    void release_prefill_leases(std::vector<sink_request> & requests) {
        for (auto & request : requests) {
            if (!request.cache_lease) { continue; }
            release_lease(request.cache_hit);
            request.cache_lease = false;
        }
    }
    void admit(const sink_request & request, kimi_cpu_sink_stats & current) {
        if (request.cache_admit < 0) { return; }
        auto & e = entries[request.cache_admit];
        if (!e.enabled || !e.resident) { throw std::runtime_error("CPU sink cache admitted a disabled slot"); }
        // All three reads completed and no compute worker retains this slot.
        // Publication changes metadata only; the expert payload never moves.
        e.valid = true; e.reserved = false;
        stats.valid_bytes += stride;
        stats.admitted_bytes += stride;
        ++current.cache_admitted; current.cache_admitted_bytes += stride;
    }
    void acquire_lease(int index, const sink_request & request) {
        auto & e = entries.at(size_t(index));
        if (!e.enabled || (!e.valid && !e.reserved) || e.id != request.id ||
                !matches(e, request) || e.leases == UINT32_MAX) {
            throw std::runtime_error("CPU sink cache lease identity mismatch");
        }
        ++e.leases;
    }
    void release_lease(int index) {
        auto & e = entries.at(size_t(index));
        if (!e.leases || e.reserved) { throw std::runtime_error("CPU sink cache lease released before completion"); }
        if (!--e.leases) { e.protected_now = false; }
    }
    // Called only while the other tier holds a versioned ready lease. Return
    // -1 if an executor still owns this RAM copy, 0 if absent, 1 if dropped.
    int drop_duplicate(int layer_id, const sink_request & request) {
        if (entries.empty()) { return 0; } // RAM admission may have paused.
        const auto found = std::find_if(layers.begin(), layers.end(), [&](const layer & l) { return l.id == layer_id; });
        if (found == layers.end()) { return 0; }
        const size_t layer_index = size_t(found - layers.begin());
        for (size_t i = found->begin; i < found->begin + found->count; ++i) {
            auto & e = entries[i];
            if (!e.valid || e.id != request.id || e.layer_index != layer_index || !matches(e, request)) { continue; }
            if (e.leases || e.protected_now || e.reserved || e.inflight) { return -1; }
            e.valid = false;
            stats.valid_bytes -= stride;
            // Keep resident and committed/page-table accounting: the physical
            // slot is reused in place, without a second payload allocation.
            return 1;
        }
        return 0;
    }
};

#include "kimi-cpu-sink-vram.inc"

struct sink_slot {
    sink_buffer role[3];
    enum state_type { empty, reading, ready } state = empty;
    unsigned remaining = 0;
    unsigned ready_roles = 0; // successful CQEs, protected by io_mutex
    size_t request = 0;
};

// Q2 prefill keeps dedicated, page-aligned staging storage through finish.
// MXFP4 prefill instead returns each ring credit after its GPU upload.
struct sink_prefill_staging {
    sink_buffer role[3];
};

// Diagnostic only. All times use the scheduler's CLOCK_MONOTONIC microseconds.
// An I/O completion is when this process observes the CQE, not a device timestamp.
// The I/O thread and each worker write disjoint fields; print only after io_done
// and the compute barriers. No extra synchronization is added to the data path.
struct alignas(64) sink_timeline_worker {
    uint64_t begin = 0, end = 0;
    double cpu_us = 0;
    int core = -1;
};

// The frequently modified row cursor gets a cache line of its own. alignas
// also rounds sizeof up to 64, so adjacent immutable scheduler fields cannot
// share its line.
struct alignas(64) sink_dynamic_row_counter {
    std::atomic<int> next{0};
};
static_assert(sizeof(sink_dynamic_row_counter) == 64, "dynamic row cursor must occupy one cache line");
struct sink_timeline_expert {
    std::array<uint64_t, 6> io = {}; // queue, submit begin/end, up/gate/down CQE
    std::array<uint64_t, 8> cpu = {}; // wait, up/gate, hidden quant, down: begin/end
    std::vector<sink_timeline_worker> workers; // up/gate then down, indexed by ith
};

} // namespace

static void sink_location_provider_initialize(kimi_cpu_sink * sink);


// The median and the mean of a duration without keeping the samples: 32 bins per octave of 1 + us (2% wide).
struct sink_duration_histogram {
    std::array<uint64_t, 32 * 27> bins = {}; // up to 2^27 us
    uint64_t count = 0;
    double sum_us = 0;
    void add(double us) {
        us = std::max(0.0, us);
        ++bins[std::min(bins.size() - 1, size_t(32.0 * std::log2(1.0 + us)))];
        ++count;
        sum_us += us;
    }
    double median_us() const {
        uint64_t seen = 0;
        for (size_t i = 0; i < bins.size() && count; ++i) {
            if (2 * (seen += bins[i]) >= count) { return std::exp2((double(i) + 0.5) / 32.0) - 1.0; }
        }
        return 0;
    }
    double mean_us() const { return count ? sum_us / double(count) : 0; }
};

// Online miss-ratio curve of the decode expert stream. Each routed layer keeps an LRU stack of the experts it has
// seen (most recent first) and a histogram of stack distances, so the hit rate an LRU cache of any per-layer size
// would get is known while serving, for any model and without an offline trace. The k experts of one token are one
// access group: their distances are taken before any of them moves to the front. Cost: a linear scan of at most
// n_experts entries per selected expert. GGML_KIMI_CPU_SINK_MRC=1 enables it; a summary line is printed every
// GGML_KIMI_CPU_SINK_MRC_EVERY decode tokens (default 256) and when a new request starts.
struct sink_mrc {
    bool enabled = false;
    uint64_t report_every = 256;
    std::vector<std::vector<int32_t>> stack;      // per layer
    std::vector<std::vector<uint64_t>> hist;      // per layer; bin d = stack distance d (first touches in cold)
    std::vector<int> layers;                      // layer indices seen, in first-seen order
    uint64_t tokens = 0, reported_tokens = 0;
    int first_layer = -1;
    std::vector<int> scratch;

    explicit sink_mrc(bool on) : enabled(on) {
        if (const char * v = std::getenv("GGML_KIMI_CPU_SINK_MRC_EVERY")) {
            const long n = std::strtol(v, nullptr, 10);
            if (n > 0) { report_every = uint64_t(n); }
        }
    }

    void observe(int layer, const int32_t * ids, int n) {
        if (!enabled || layer < 0 || !ids || n <= 0) { return; }
        if (size_t(layer) >= stack.size()) { stack.resize(size_t(layer) + 1); hist.resize(size_t(layer) + 1); }
        if (first_layer < 0) { first_layer = layer; }
        if (layer == first_layer) { ++tokens; }
        auto & st = stack[size_t(layer)];
        auto & h = hist[size_t(layer)];
        if (st.empty() && h.empty()) { layers.push_back(layer); }
        scratch.assign(size_t(n), -1);
        for (int i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            if (e < 0) { continue; }
            const auto it = std::find(st.begin(), st.end(), e);
            scratch[size_t(i)] = it == st.end() ? -1 : int(it - st.begin());
        }
        for (int i = 0; i < n; ++i) {
            if (ids[i] < 0) { continue; }
            const size_t bin = scratch[size_t(i)] < 0 ? SIZE_MAX : size_t(scratch[size_t(i)]);
            if (bin == SIZE_MAX) {
                cold_add(layer);  // first touch: a miss at every size
            } else {
                if (h.size() <= bin) { h.resize(bin + 1); }
                ++h[bin];
            }
        }
        for (int i = n - 1; i >= 0; --i) {
            const int32_t e = ids[i];
            if (e < 0) { continue; }
            const auto it = std::find(st.begin(), st.end(), e);
            if (it != st.end()) { st.erase(it); }
            st.insert(st.begin(), e);
        }
        if (tokens - reported_tokens >= report_every && layer == first_layer) { report("periodic"); }
    }

    std::vector<uint64_t> cold;
    void cold_add(int layer) {
        if (cold.size() <= size_t(layer)) { cold.resize(size_t(layer) + 1); }
        ++cold[size_t(layer)];
    }

    // hits an LRU of per-layer quota q would have had, summed over layers
    uint64_t hits_at(size_t q, int layer) const {
        const auto & h = hist[size_t(layer)];
        uint64_t sum = 0;
        for (size_t d = 0; d < h.size() && d < q; ++d) { sum += h[d]; }
        return sum;
    }
    uint64_t accesses(int layer) const {
        uint64_t sum = size_t(layer) < cold.size() ? cold[size_t(layer)] : 0;
        for (auto c : hist[size_t(layer)]) { sum += c; }
        return sum;
    }

    void report(const char * why) {
        if (!enabled || layers.empty() || tokens == reported_tokens) { return; }
        reported_tokens = tokens;
        static const size_t grid[] = { 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512 };
        uint64_t total = 0;
        for (int l : layers) { total += accesses(l); }
        std::fprintf(stderr, "[kimi_cpu_sink_mrc] {\"schema\":1,\"why\":\"%s\",\"tokens\":%llu,\"layers\":%zu,"
                "\"accesses\":%llu,\"quota\":[", why, (unsigned long long) tokens, layers.size(),
                (unsigned long long) total);
        for (size_t i = 0; i < sizeof(grid) / sizeof(grid[0]); ++i) { std::fprintf(stderr, "%s%zu", i ? "," : "", grid[i]); }
        std::fprintf(stderr, "],\"hit_uniform\":[");
        for (size_t i = 0; i < sizeof(grid) / sizeof(grid[0]); ++i) {
            uint64_t hits = 0;
            for (int l : layers) { hits += hits_at(grid[i], l); }
            std::fprintf(stderr, "%s%.4f", i ? "," : "", total ? double(hits) / double(total) : 0.0);
        }
        std::fprintf(stderr, "],\"per_layer_hit_q16\":[");
        bool first = true;
        for (int l : layers) {
            const uint64_t a = accesses(l);
            std::fprintf(stderr, "%s%.3f", first ? "" : ",", a ? double(hits_at(16, l)) / double(a) : 0.0);
            first = false;
        }
        std::fprintf(stderr, "]}\n");
    }
};

struct kimi_cpu_sink {
    const int compute_threads;
    const int ring_experts;
    kimi_cpu_sink_resolver resolver;
    void * userdata;
    std::mutex run_mutex;
    sink_ring ring;
    std::vector<sink_slot> slots;
    std::vector<sink_prefill_staging> prefill_ssd_staging;
    std::vector<sink_request> requests;
    std::vector<size_t> io_requests;
    sink_cache cache;
    sink_vram vram;
    uint64_t ledger_refusals = 0; // make_room answers that were no
    bool margin_boundary = false; // the last token boundary the margin saw was a decode token
    uint64_t margin_epoch = 0, margin_token = 0;

    // Next-layer expert prefetch (GGML_KIMI_CPU_SINK_PREFETCH=k, off by default). The graph applies the next
    // layer's router to this layer's expert input and hands its experts over as a hint, best first; the first k
    // that are neither cached nor held by the GPU tier now (its residency ABI) are read into RAM cache slots of the
    // next layer once this layer's own reads are done, so the drive works through the gap before the next router
    // runs. At any depth, 0 included, every rank of the hint is counted (prefetch_report): how often its expert was
    // novel, how often the next layer selected a novel one, and how often a read one was used.
    struct prefetch_job {
        int32_t layer = -1, expert = -1;
        int rank = -1;        // in the hint
        bool checked = false; // seen by the run of its layer (used_r)
        int entry = -1;
        uint64_t generation = 0;
        kimi_cpu_sink_source role[3] = {};
        char * target[3] = {};
        int done = 0, running = 0, queued = 0; // roles read / being read / waiting, under the prefetch mutex
        bool failed = false;
        bool activated = false; // published as a hit of a run that waits for the rest of its reads
        bool started = false;   // a read of one of its roles started (started_r)
        // The only field a run's scheduler reads without the prefetch mutex: 0 while the reads may still complete,
        // 1 once all three roles are read, 2 once the job cannot complete (a read failed or a role was dropped).
        std::atomic<int> state{0};
    };
    struct prefetch_state {
        int topk = 0;
        int32_t hint[64] = {}; // the predicted experts of layer hint_layer, best first, for the coming run
        int hint_n = 0;
        int32_t hint_layer = -1;
        std::mutex mutex;
        std::condition_variable cv;
        std::vector<std::unique_ptr<prefetch_job>> jobs; // unfinished jobs; a reader holds one only while it reads
        std::deque<std::pair<prefetch_job *, int>> queue;  // (job, role) of the next layer, read after this run's reads
        std::deque<std::pair<prefetch_job *, int>> urgent; // roles of experts the current run uses, read at once
        std::vector<std::pair<int, prefetch_job *>> pending; // (cache entry, job): hits of the current run still reading
        std::vector<std::thread> readers;
        bool gate_open = false, stop = false;
        int running = 0;
        uint64_t fail_every = 0, reads = 0; // GGML_KIMI_CPU_SINK_PREFETCH_FAIL_EVERY: fault injection
        // GGML_KIMI_CPU_SINK_PREFETCH_THROW_{DRAIN,SCHEDULE}=n: the n-th allocation there fails (fault injection)
        uint64_t throw_drain = 0, throw_schedule = 0, allocs_drain = 0, allocs_schedule = 0;
        uint64_t layers = 0, predicted = 0, issued = 0, ready = 0, cancelled = 0, failed = 0,
                 skip_vram = 0, skip_cached = 0, no_slot = 0, unresolved = 0, carried = 0, activated = 0;
        // Reads of experts the next run selected but the GPU tier came to hold since they were scheduled: not needed,
        // the tier serves them. And the converse: experts skipped because the tier held them that the next run
        // selected after the tier gave them back (a reclaim between the two runs), read on demand instead.
        uint64_t vram_late = 0, skip_vram_lost = 0;
        // Per rank of the hint: given, novel (neither held by the GPU tier nor in the RAM cache), novel and selected
        // by the next layer, read, and read and taken from the RAM cache by the next layer.
        uint64_t predicted_r[64] = {}, novel_r[64] = {}, novel_routed_r[64] = {}, issued_r[64] = {}, used_r[64] = {};
        int width = 0; // ranks of the widest hint
        // The reads that the current run selected (rank, expert), until its prepare says where it takes them from.
        std::pair<int, int32_t> selected_reads[64] = {};
        int selected_n = 0;
        // The last decode run's hint and which of its ranks were novel, until the run of the layer it predicts.
        int32_t last_layer = -1;
        int last_n = 0;
        int32_t last_hint[64] = {};
        bool last_novel[64] = {}, last_skipped[64] = {}; // last_skipped: within the depth and held by the tier
        // The drive's idle window per hinted layer: from the end of a run's own reads (its schedule, without any)
        // to the start of the run of the layer its hint predicts. The window the prefetch reads have, at any depth.
        sink_clock::time_point window_start;
        bool window_open = false;
        sink_duration_histogram window;
        // The readers: bytes read, and the wall time with at least one read running.
        uint64_t reader_bytes = 0;
        double reader_busy_us = 0;
        sink_clock::time_point busy_since;
        // What the reads cost the demand reads: the read each reader runs now, and whether it spills, i.e. runs past
        // the window's close for an expert the run that closed it does not take (an urgent read, or one of an expert
        // that run computes on the CPU, is that run's own read). spill_us is the wall time with at least one spilling
        // read; spills counts the reads that spilled, each once however many windows it runs past (spilled), spill_us_r
        // adds each one's own time past a close to the rank of its expert in the hint, and started_r counts per rank
        // the reads (experts) that began.
        struct read_state {
            prefetch_job * job = nullptr;
            bool urgent = false, spill = false, spilled = false;
            sink_clock::time_point since;
        } reading[3];
        int spilling = 0;
        double spill_us = 0;
        uint64_t spills = 0;
        sink_clock::time_point spill_since;
        uint64_t started_r[64] = {}, spill_us_r[64] = {};
    } prefetch;

    // Unified memory: the ledger's floor at every point, not only at decode token boundaries (floor_watch_loop).
    std::mutex watch_mutex;
    std::condition_variable watch_cv;
    std::thread watch_thread;
    bool watch_stop = false;

    std::mutex io_mutex;
    std::condition_variable io_cv;
    std::thread io_thread;
    bool io_stop = false, io_job = false, io_done = true, io_cancel = false, poisoned = false;
    std::string io_error;
    kimi_cpu_sink_stats current = {};

    std::mutex compute_mutex;
    std::condition_variable compute_cv;
    std::condition_variable compute_done_cv;
    const bool split_cv;
    const bool expert_job;
    const bool ready_first;
    bool cache_compute_active = false;
    kimi_cache_location_api locations = {};
    void * locations_context = nullptr; // run_mutex; published with io_job
    const bool early_up_gate;
    const bool role_major;
    const bool nonblocking_down_sched;
    const int poll_us;
    alignas(64) std::atomic<uint64_t> poll_generation{0};
    alignas(64) std::atomic<int> poll_pending{0};
    const bool dynamic_rows;
    const int dynamic_chunk_rows;
    sink_dynamic_row_counter dynamic_row_counter;
    const std::vector<double> row_weights;
    const std::vector<int> compute_cpus, io_cpus;
    std::vector<int> up_bounds, down_bounds;
    std::vector<float> expert_output;
    std::vector<uint8_t> expert_computed;
    std::vector<uint8_t> expert_stage, expert_upload_started;
    // Measured CPU time of each decode expert of the layer, and its running mean on this machine.
    std::vector<double> expert_cpu_us;
    double cpu_expert_us = 0;
    // The drive's rate for decode reads (bytes per us of the I/O job), a running mean over decode layers with reads.
    double decode_read_bytes_per_us = 0;
    // The decode plan of the current layer: which experts the GPU computes, from the evaluator that also places
    // grouped prefill (kimi-prefill-cost.h, the in-ring list schedule) over the measured lines.
    struct decode_plan_state {
        bool valid = false;
        std::vector<uint8_t> gpu;          // per request: the GPU computes it
        double predicted_us = 0, cpu_only_us = 0;
        size_t residents = 0, ram = 0, misses = 0, candidates = 0, moves = 0, locked = 0;
        sink_clock::time_point start;      // the layer's reads are issued
    } plan;
    // Decode's lines beyond the CPU's time per expert and the tier's: the drive per layer (read time against MB, the
    // last 64 layers with reads), the share of an expert's CPU time before its down role (early up/gate), a layer's
    // time outside its lanes (input quantization, launch, reduction), and the CPU's time per expert in layers without
    // reads, apart (cpu_alone_us) and beside the GPU's host jobs (cpu_shared_us): their copies, or the GPU's reads in
    // place, share the DRAM with the CPU. The plan scales the CPU's work of a layer with moves by their ratio (at
    // least 1; 1 until both are measured).
    struct decode_line_state {
        std::deque<std::pair<double, double>> reads;
        double early = 0, fixed_us = 0, cpu_alone_us = 0, cpu_shared_us = 0;
        double cpu_moved() const {
            return cpu_alone_us > 0 && cpu_shared_us > 0 ? std::max(1.0, cpu_shared_us / cpu_alone_us) : 1.0;
        }
    } decode_lines;
    // Receipts: planned layers, tokens, experts the GPU computed from host memory, refused batches, and the relative
    // error of the predicted lanes' time against the measured one, in 1% bins (the last bin is 100% and more).
    struct decode_place_stats {
        uint64_t layers = 0, tokens = 0, gpu_experts = 0, refused = 0, moves_planned = 0, candidates = 0, wanted = 0;
        uint64_t last_token = UINT64_MAX;
        std::array<uint64_t, 101> error = {};
        double predicted_us = 0, measured_us = 0, cpu_only_us = 0;
    } place;
    // Page-locking of the RAM cache's slots for the GPU tier's copies, on a discrete GPU (kimi-host-lock.h): the slots
    // the decode plan would move an expert from, one at a time, in the drive's shadow; unlocked before their pages go
    // (sink_cache::before_discard).
    kimi_host_lock host_lock;
    // Diagnostic switches for A/B runs; both on by default.
    const bool host_gpu_enabled = sink_bool_env("GGML_KIMI_CPU_SINK_HOST_GPU", true);
    const bool decode_thread_choice = sink_bool_env("GGML_KIMI_CPU_SINK_DECODE_THREAD_CHOICE", true);
    std::vector<uint8_t> staged_quant_hidden;
    bool compute_expert = false; // published with generation under compute_mutex
    std::mutex expert_mutex;
    std::condition_variable expert_ready_cv, expert_done_cv;
    alignas(64) std::atomic<int> expert_up_pending{0};
    alignas(64) std::atomic<int> expert_down_pending{0};
    alignas(64) std::atomic<int> expert_hidden_pending{0};
    const bool profile_enabled;
    const bool trace_enabled;
    const bool timeline_enabled;
    bool timeline_active = false;
    size_t timeline_expert = 0;
    std::array<uint64_t, 6> timeline_call = {}; // enter, prepare end, quant end, job, I/O end, exit
    std::vector<sink_timeline_expert> timeline;
    uint64_t trace_epoch = 0, trace_token = 0, trace_call = 0;
    bool trace_has_token = false, trace_decode = false;
    int trace_layer = -1;
    sink_mrc mrc{sink_bool_env("GGML_KIMI_CPU_SINK_MRC", false)};
    double caller_cpu_start = 0;
    sink_clock::time_point phase_dispatch;
    std::vector<sink_worker_profile> worker_profiles;
    std::vector<std::thread> workers;
    bool compute_stop = false;
    uint64_t generation = 0;
    int pending = 0;
    bool down_phase = false;
    // Participants of a compute phase: all for prefill; for decode, the choice below.
    int active_threads = 0;
    // Decode compute is a few short matvecs per expert, so a slow core's last chunk can hold every expert
    // back, while prefill wants every core. Decode compares the fastest cluster's participants with all of
    // them on live experts (mean CPU time per expert over probe_experts each) and keeps the faster one,
    // re-checking every recheck_experts.
    struct thread_choice {
        static constexpr uint64_t probe_experts = 400, recheck_experts = 40000;
        int candidates[2] = {}, current = 0, decided = -1;
        bool locked = false; // fixed by the environment: no comparison, no recheck
        double us[2] = {};
        uint64_t samples[2] = {}, since_decision = 0;
    } decode_threads;
    kimi_cpu_sink_parallel_fn parallel_fn = nullptr;
    void * parallel_userdata = nullptr;
    int latent = 0, hidden = 0;
    size_t row_bytes_up = 0, row_bytes_down = 0;
    const char * up_data = nullptr, * gate_data = nullptr, * down_data = nullptr;
    float beta = 0, linear_beta = 0, weight = 0;
    const float * post_exp_norm = nullptr;
    int activation_kind = KIMI_CPU_SINK_ACTIVATION_SITU;
    float swiglu_limit = 0;
    float * output = nullptr;
    std::vector<float> activation;
    std::vector<uint8_t> quant_input, quant_hidden;
    const uint8_t * down_quant_data = nullptr;
    size_t quant_hidden_stride = 0;
    // Selected from the expert tensor types by prepare(); up and gate share one.
    ggml_vec_dot_t dot_up = nullptr, dot_down = nullptr;
    ggml_from_float_t quantize_input = nullptr, quantize_hidden = nullptr;

    kimi_cpu_sink(int threads, int depth, kimi_cpu_sink_resolver resolve, void * user) :
        compute_threads(threads - 1), ring_experts(depth), resolver(resolve), userdata(user), slots(depth),
        split_cv(sink_bool_env("GGML_KIMI_CPU_SINK_SPLIT_CV", false)),
        expert_job(sink_bool_env("GGML_KIMI_CPU_SINK_EXPERT_JOB", false)),
        ready_first(sink_bool_env("GGML_KIMI_CPU_SINK_READY_FIRST", false)),
        early_up_gate(sink_bool_env("GGML_KIMI_CPU_SINK_EARLY_UP_GATE", false)),
        role_major(sink_bool_env("GGML_KIMI_CPU_SINK_ROLE_MAJOR", false)),
        nonblocking_down_sched(sink_bool_env("GGML_KIMI_CPU_SINK_NONBLOCKING_DOWN_SCHED", false)),
        poll_us(sink_poll_us_env()),
        dynamic_rows(sink_bool_env("GGML_KIMI_CPU_SINK_DYNAMIC_ROWS", false)),
        dynamic_chunk_rows(sink_dynamic_chunk_rows_env()),
        row_weights(sink_weights_env(threads - 1)),
        compute_cpus(sink_cpus_env("GGML_KIMI_CPU_SINK_COMPUTE_CPUS", threads - 1)),
        io_cpus(sink_cpus_env("GGML_KIMI_CPU_SINK_IO_CPUS", 0)),
        profile_enabled(sink_bool_env("GGML_KIMI_CPU_SINK_PROFILE", false)),
        trace_enabled(sink_bool_env("GGML_KIMI_CPU_SINK_TRACE", false)),
        timeline_enabled(sink_bool_env("GGML_KIMI_CPU_SINK_TIMELINE", false)), worker_profiles(threads - 1) {
        // Kernels are selected per call from the expert tensor types in prepare().
        if (dynamic_rows && !row_weights.empty()) {
            throw std::runtime_error("DYNAMIC_ROWS and ROW_WEIGHTS are mutually exclusive row schedulers");
        }
        vram.initialize(ready_first);
        refresh_floor("init");
        if (vram.deferred_upload) {
            cache.before_release = [](void * p) { static_cast<sink_vram *>(p)->wait_deferred(); };
            cache.before_release_arg = &vram;
        }
        cache.before_discard = [](void * p, size_t first, size_t count) { static_cast<kimi_cpu_sink *>(p)->host_unlock(first, count); };
        cache.before_discard_arg = this;
        vram.ram_slot_locked = [this](int entry) { return entry >= 0 && host_lock.is_locked(size_t(entry)); };
        sink_location_provider_initialize(this);
        // Expert jobs publish hidden readiness directly to the down workers.
        // Keep that lifecycle unchanged until it has its own I/O error barrier.
        if (nonblocking_down_sched && expert_job) {
            throw std::runtime_error("NONBLOCKING_DOWN_SCHED requires EXPERT_JOB=0");
        }
        if (early_up_gate && expert_job) {
            throw std::runtime_error("EARLY_UP_GATE requires EXPERT_JOB=0");
        }
        if (early_up_gate && vram.location_api) {
            throw std::runtime_error("EARLY_UP_GATE cannot share the cache-location provider");
        }
        if (role_major && !early_up_gate) {
            throw std::runtime_error("ROLE_MAJOR requires EARLY_UP_GATE=1");
        }
        if (nonblocking_down_sched && !role_major) {
            throw std::runtime_error("NONBLOCKING_DOWN_SCHED requires ROLE_MAJOR=1");
        }
        if (nonblocking_down_sched && !ready_first) {
            throw std::runtime_error("NONBLOCKING_DOWN_SCHED requires READY_FIRST=1");
        }
        if (poll_us && expert_job) {
            throw std::runtime_error("POLL_US requires EXPERT_JOB=0");
        }
        ring.init(static_cast<unsigned>(3 * depth), sink_read_fixed_from_env());
        // Reserve before starting threads so vector growth cannot strand a thread.
        workers.reserve(static_cast<size_t>(std::max(0, compute_threads - 1)));
        try {
            for (int ith = 1; ith < compute_threads; ++ith) {
                workers.emplace_back([this, ith] { compute_loop(ith); });
                if (!compute_cpus.empty()) { sink_pin(workers.back().native_handle(), { compute_cpus[ith] }); }
            }
            io_thread = std::thread([this] { io_loop(); });
            sink_pin(io_thread.native_handle(), io_cpus);
            if (cache.ledger.unified && vram.context) {
                watch_thread = std::thread([this] { floor_watch_loop(); });
                sink_pin(watch_thread.native_handle(), io_cpus);
            }
        } catch (...) {
            stop_threads();
            throw;
        }
        {
            // One CPU of the fastest cluster stays with the I/O thread, as THREADS counts it.
            const int fast = sink_fast_cpus();
            const int fast_participants = std::max(1, std::min(compute_threads, fast - 1));
            decode_threads.candidates[0] = compute_threads;
            decode_threads.candidates[1] = fast > 0 && fast_participants < compute_threads ? fast_participants : compute_threads;
            if (decode_threads.candidates[1] == compute_threads || !decode_thread_choice) {
                decode_threads.decided = 0;
                decode_threads.locked = true;
            }
            // Per-phase pools: decode on the first N participants (the fastest CPUs when COMPUTE_CPUS lists them
            // first), prefill on all.
            if (const char * value = std::getenv("GGML_KIMI_CPU_SINK_DECODE_PARTICIPANTS")) {
                char * end = nullptr;
                errno = 0;
                const long n = std::strtol(value, &end, 10);
                if (errno || end == value || *end || n < 1 || n > compute_threads) {
                    throw std::runtime_error("GGML_KIMI_CPU_SINK_DECODE_PARTICIPANTS must be in 1..THREADS-1");
                }
                decode_threads.candidates[1] = int(n);
                decode_threads.decided = 1;
                decode_threads.locked = true;
            }
            std::fprintf(stderr, "[kimi_cpu_sink_threads] participants=%d fast_cluster_cpus=%d decode_candidates=%d,%d"
                    " decode_fixed=%d\n", compute_threads, fast, decode_threads.candidates[0],
                    decode_threads.candidates[1], decode_threads.locked ? decode_threads.candidates[decode_threads.decided] : 0);
        }
        std::fprintf(stderr, "[kimi_cpu_sink_schedule] ready_first=%d early_up_gate=%d role_major=%d nonblocking_down_sched=%d poll_us=%d dynamic_rows=%d dynamic_chunk_rows=%d row_weights=",
                     ready_first ? 1 : 0, early_up_gate ? 1 : 0, role_major ? 1 : 0,
                     nonblocking_down_sched ? 1 : 0, poll_us, dynamic_rows ? 1 : 0, dynamic_chunk_rows);
        for (size_t i = 0; i < row_weights.size(); ++i) { std::fprintf(stderr, "%s%.9g", i ? "," : "", row_weights[i]); }
        std::fprintf(stderr, " compute_cpus=");
        for (size_t i = 0; i < compute_cpus.size(); ++i) { std::fprintf(stderr, "%s%d", i ? "," : "", compute_cpus[i]); }
        std::fprintf(stderr, " io_cpus=");
        for (size_t i = 0; i < io_cpus.size(); ++i) { std::fprintf(stderr, "%s%d", i ? "," : "", io_cpus[i]); }
        std::fprintf(stderr, "\n");
        try {
            prefetch_start();
        } catch (...) {
            prefetch_stop();
            stop_threads();
            throw;
        }
        if (prefetch.topk) { std::fprintf(stderr, "[kimi_cpu_sink_prefetch] enabled topk=%d readers=3\n", prefetch.topk); }
    }

    void prefetch_start() {
        const char * value = std::getenv("GGML_KIMI_CPU_SINK_PREFETCH");
        if (!value || !value[0]) { return; }
        char * end = nullptr;
        const long k = std::strtol(value, &end, 10);
        // at most the hint's width: the graph hints the model's top-k, up to the ranks the hint keeps
        const long width = long(sizeof(prefetch.hint) / sizeof(prefetch.hint[0]));
        if (end == value || *end || k < 0 || k > width) {
            throw std::runtime_error("GGML_KIMI_CPU_SINK_PREFETCH must be 0.." + std::to_string(width) + " (the hint's width)");
        }
        prefetch.topk = int(k);
        cache.staging_per_layer = std::max(1, prefetch.topk);
        if (!prefetch.topk) { return; }
        // a prefetched hit may still be reading when its run starts; only this scheduler waits for it
        if (!ready_first || !nonblocking_down_sched) {
            throw std::runtime_error("GGML_KIMI_CPU_SINK_PREFETCH requires GGML_KIMI_CPU_SINK_READY_FIRST=1 and "
                "GGML_KIMI_CPU_SINK_NONBLOCKING_DOWN_SCHED=1");
        }
        if (const char * every = std::getenv("GGML_KIMI_CPU_SINK_PREFETCH_FAIL_EVERY")) {
            prefetch.fail_every = std::strtoull(every, nullptr, 10);
        }
        if (const char * n = std::getenv("GGML_KIMI_CPU_SINK_PREFETCH_THROW_DRAIN")) { prefetch.throw_drain = std::strtoull(n, nullptr, 10); }
        if (const char * n = std::getenv("GGML_KIMI_CPU_SINK_PREFETCH_THROW_SCHEDULE")) { prefetch.throw_schedule = std::strtoull(n, nullptr, 10); }
        cache.before_unmap = [](void * p) { static_cast<kimi_cpu_sink *>(p)->prefetch_wait_reads(); };
        cache.before_unmap_arg = this;
        for (int i = 0; i < 3; ++i) { prefetch.readers.emplace_back([this, i] { prefetch_reader(i); }); }
    }
    void prefetch_stop() {
        {
            std::lock_guard<std::mutex> lock(prefetch.mutex);
            prefetch.stop = true;
            prefetch.queue.clear();
        }
        prefetch.cv.notify_all();
        for (auto & t : prefetch.readers) { if (t.joinable()) { t.join(); } }
        prefetch.readers.clear();
    }
    void prefetch_reader(int reader) {
        std::unique_lock<std::mutex> lock(prefetch.mutex);
        auto & self = prefetch.reading[reader];
        for (;;) {
            prefetch.cv.wait(lock, [&] {
                return prefetch.stop || !prefetch.urgent.empty() || (prefetch.gate_open && !prefetch.queue.empty());
            });
            if (prefetch.stop) { return; }
            const bool urgent = !prefetch.urgent.empty();
            auto & from = urgent ? prefetch.urgent : prefetch.queue;
            const auto item = from.front();
            from.pop_front();
            prefetch_job & job = *item.first;
            if (!job.started) { job.started = true; ++prefetch.started_r[job.rank]; }
            self.job = &job; self.urgent = urgent; self.spill = self.spilled = false;
            const kimi_cpu_sink_source source = job.role[item.second];
            char * target = job.target[item.second];
            if (!prefetch.running++) { prefetch.busy_since = sink_clock::now(); }
            ++job.running;
            --job.queued;
            const bool inject = prefetch.fail_every && ++prefetch.reads % prefetch.fail_every == 0;
            lock.unlock();
            bool ok = !inject;
            size_t done = 0;
            while (ok && done < source.nbytes) {
                const ssize_t got = pread(source.fd_direct, target + done, source.nbytes - done, off_t(source.offset + done));
                if (got < 0 && errno == EINTR) { continue; }
                if (got <= 0) { ok = false; break; }
                done += size_t(got);
                if (source.payload_nbytes && done >= source.data_offset + source.payload_nbytes) { break; }
            }
            lock.lock();
            if (self.spill) { prefetch_end_spill(self, sink_clock::now()); }
            self.job = nullptr;
            if (ok) { ++job.done; } else { job.failed = true; }
            prefetch.reader_bytes += done;
            if (!--prefetch.running) { prefetch.reader_busy_us += elapsed_us(prefetch.busy_since); }
            --job.running;
            prefetch_settle(job);
            prefetch.cv.notify_all();
            // a run may wait for this expert as a cache hit
            lock.unlock();
            { std::lock_guard<std::mutex> io_lock(io_mutex); }
            io_cv.notify_all();
            lock.lock();
        }
    }
    static void prefetch_alloc(uint64_t at, uint64_t & count) {
        if (at && ++count == at) { throw std::bad_alloc(); }
    }
    // Under the prefetch mutex: a spilling read ends (it completed, or a new window opened while it ran).
    void prefetch_end_spill(prefetch_state::read_state & r, sink_clock::time_point now) {
        r.spill = false;
        prefetch.spill_us_r[r.job->rank] += uint64_t(std::llround(std::chrono::duration<double, std::micro>(now - r.since).count()));
        if (!--prefetch.spilling) {
            prefetch.spill_us += std::chrono::duration<double, std::micro>(now - prefetch.spill_since).count();
        }
    }
    // Under the prefetch mutex, after any change of a job's counts: publish its terminal state once it has one.
    static void prefetch_settle(prefetch_job & job) {
        if (job.state.load() != 0) { return; }
        if (job.failed) { job.state = 2; }
        else if (job.done == 3) { job.state = 1; }
        else if (job.running == 0 && job.queued == 0) { job.state = 2; }
    }
    // The I/O thread calls this when a run's own reads are done, a decode run without any when it schedules: the
    // drive's idle window starts, and the reads of the next layer may begin.
    void prefetch_open_gate() {
        {
            std::lock_guard<std::mutex> lock(prefetch.mutex);
            prefetch.window_start = sink_clock::now();
            prefetch.window_open = true;
            if (!prefetch.topk) { return; }
            prefetch.gate_open = true;
            // a read still running from before is in this window now
            for (auto & r : prefetch.reading) { if (r.spill) { prefetch_end_spill(r, prefetch.window_start); } }
        }
        prefetch.cv.notify_all();
    }
    // Every entry that holds the run mutex starts here. Reads of the next layer not started yet are dropped, and
    // jobs whose reads are over are published. A decode run (layer and ids given) does not wait: an expert it
    // computes on the CPU (it selects it and the GPU tier does not hold it) that is still being read becomes a hit
    // it waits for like one of its own reads, its remaining roles read at once; other reads go on in the background
    // into inflight slots. Every other entry waits for all reads; that full drain allocates nothing, as it runs
    // outside the entries' try blocks and in their error handling.
    void prefetch_drain(int32_t layer = -1, const int32_t * ids = nullptr, int nids = 0) {
        uint8_t held[1024] = {}; // per id: the tier holds it, so this run's prepare pins it as a hit
        if (prefetch.topk && ids && nids > 0 && nids <= 1024) { vram.held(layer, ids, nids, held); }
        std::unique_lock<std::mutex> lock(prefetch.mutex);
        prefetch_check(layer, ids, nids, held);
        if (!prefetch.topk) { return; }
        prefetch.pending.clear();
        prefetch.gate_open = false;
        auto needed = [&](const prefetch_job & job) {
            if (!ids || job.layer != layer || job.failed) { return false; }
            for (int i = 0; i < nids; ++i) { if (ids[i] == job.expert && !held[i]) { return true; } }
            return false;
        };
        // the window closes: the reads running for experts this entry does not take spill from now on (a read still
        // running from an earlier window spills again, and counts once)
        const auto closed = sink_clock::now();
        for (auto & r : prefetch.reading) {
            if (!r.job || r.urgent || r.spill || needed(*r.job)) { continue; }
            r.spill = true; r.since = closed;
            if (!r.spilled) { r.spilled = true; ++prefetch.spills; }
            if (!prefetch.spilling++) { prefetch.spill_since = closed; }
        }
        if (ids) {
            // every allocation comes before the first change, so a failure leaves the queues for the full drain
            std::deque<std::pair<prefetch_job *, int>> urgent;
            for (auto * q : { &prefetch.urgent, &prefetch.queue }) {
                for (const auto & item : *q) {
                    if (needed(*item.first)) {
                        prefetch_alloc(prefetch.throw_drain, prefetch.allocs_drain);
                        urgent.push_back(item);
                    }
                }
            }
            prefetch.pending.reserve(prefetch.jobs.size());
            for (auto * q : { &prefetch.urgent, &prefetch.queue }) {
                for (const auto & item : *q) {
                    if (!needed(*item.first)) { --item.first->queued; prefetch_settle(*item.first); }
                }
                q->clear();
            }
            prefetch.urgent.swap(urgent);
        } else {
            for (auto * q : { &prefetch.urgent, &prefetch.queue }) {
                for (const auto & item : *q) { --item.first->queued; prefetch_settle(*item.first); }
                q->clear();
            }
            prefetch.cv.wait(lock, [&] { return prefetch.running == 0; });
        }
        for (auto it = prefetch.jobs.begin(); it != prefetch.jobs.end();) {
            prefetch_job & job = **it;
            if (job.running || job.queued) {
                if (needed(job) && !job.activated && job.state.load() == 0 && job.done + job.running + job.queued == 3 &&
                        cache.prefetch_activate(job.entry, job.generation)) {
                    job.activated = true;
                    prefetch.pending.emplace_back(job.entry, &job);
                    ++prefetch.activated;
                }
                ++prefetch.carried;
                ++it;
                continue;
            }
            const bool ok = !job.failed && job.done == 3;
            if (ok) { ++prefetch.ready; } else if (job.failed) { ++prefetch.failed; } else { ++prefetch.cancelled; }
            cache.prefetch_done(job.entry, job.generation, ok);
            it = prefetch.jobs.erase(it);
        }
        const bool urgent_reads = !prefetch.urgent.empty();
        lock.unlock();
        if (urgent_reads) { prefetch.cv.notify_all(); }
    }
    // Under the prefetch mutex, as an entry starts: a decode run (layer and ids given, held[i] whether the GPU tier
    // holds ids[i]) counts the ranks of the last hint that were novel, notes the reads that it selects, and ends the
    // drive's idle window; any other entry ends the layer sequence.
    void prefetch_check(int32_t layer, const int32_t * ids, int nids, const uint8_t * held) {
        auto selected = [&](int32_t expert) { return expert >= 0 && std::find(ids, ids + nids, expert) != ids + nids; };
        auto tier_holds = [&](int32_t expert) {
            const ptrdiff_t at = std::find(ids, ids + nids, expert) - ids;
            return at < nids && at < 1024 && held[at];
        };
        if (ids && prefetch.last_layer == layer) {
            for (int i = 0; i < prefetch.last_n; ++i) {
                prefetch.novel_routed_r[i] += prefetch.last_novel[i] && selected(prefetch.last_hint[i]);
                prefetch.skip_vram_lost += prefetch.last_skipped[i] && selected(prefetch.last_hint[i]) &&
                    !tier_holds(prefetch.last_hint[i]);
            }
            if (prefetch.window_open) { prefetch.window.add(elapsed_us(prefetch.window_start)); }
        }
        if (ids) {
            prefetch.selected_n = 0;
            for (auto & job : prefetch.jobs) {
                if (job->layer != layer || job->checked) { continue; }
                job->checked = true;
                if (job->failed || !selected(job->expert)) { continue; }
                if (tier_holds(job->expert)) { ++prefetch.vram_late; continue; }
                if (prefetch.selected_n < 64) { prefetch.selected_reads[prefetch.selected_n++] = { job->rank, job->expert }; }
            }
        }
        prefetch.last_layer = -1;
        prefetch.window_open = false;
    }
    // For the run's scheduler, under the I/O mutex: whether a cache hit still waits for its prefetch reads.
    // For the run's scheduler, under the I/O mutex: the state of the prefetch behind a hit (1 for a hit that is
    // not a prefetch of this run), and whether any of them is still reading or failed.
    int prefetch_hit_state(int entry) const {
        for (const auto & p : prefetch.pending) { if (p.first == entry) { return p.second->state.load(); } }
        return 1;
    }
    bool prefetch_hits_pending() const {
        for (const auto & p : prefetch.pending) { if (p.second->state.load() == 0) { return true; } }
        return false;
    }
    bool prefetch_hit_failed() const {
        for (const auto & p : prefetch.pending) { if (p.second->state.load() == 2) { return true; } }
        return false;
    }
    // After the run's prepare: keep only the prefetched slots that are hits of its CPU experts.
    void prefetch_keep_used_hits() {
        if (prefetch.pending.empty()) { return; }
        std::vector<std::pair<int, prefetch_job *>> used;
        for (const auto & p : prefetch.pending) {
            for (const auto & r : requests) {
                if (r.cache_hit == p.first && r.vram_slot < 0 && !r.cache_only) { used.push_back(p); break; }
            }
        }
        prefetch.pending.swap(used);
    }
    // After the run's prepare: a read it selected is used when the run takes it from the RAM cache.
    void prefetch_count_used() {
        for (int i = 0; i < prefetch.selected_n; ++i) {
            for (const auto & r : requests) {
                if (r.id != prefetch.selected_reads[i].second || r.cache_only) { continue; }
                prefetch.used_r[prefetch.selected_reads[i].first] += r.cache_hit >= 0 && r.vram_slot < 0;
                break;
            }
        }
        prefetch.selected_n = 0;
    }
    // Called by the cache before it unmaps its arena, possibly in the middle of a run.
    void prefetch_wait_reads() {
        std::unique_lock<std::mutex> lock(prefetch.mutex);
        for (auto * q : { &prefetch.urgent, &prefetch.queue }) {
            for (const auto & item : *q) { --item.first->queued; prefetch_settle(*item.first); }
            q->clear();
        }
        prefetch.cv.wait(lock, [&] { return prefetch.running == 0; });
    }
    void prefetch_report(const char * reason) {
        if (!prefetch.topk && !prefetch.layers) { return; }
        auto ranks = [&](const char * name, const uint64_t * counts) {
            std::fprintf(stderr, " %s=", name);
            for (int i = 0; i < prefetch.width; ++i) { std::fprintf(stderr, "%s%llu", i ? "," : "", (unsigned long long) counts[i]); }
        };
        uint64_t reader_bytes = 0, spills = 0, started_r[64], spill_us_r[64];
        double reader_busy_us = 0, spill_us = 0;
        {
            std::lock_guard<std::mutex> lock(prefetch.mutex);
            reader_bytes = prefetch.reader_bytes;
            reader_busy_us = prefetch.reader_busy_us;
            spill_us = prefetch.spill_us; spills = prefetch.spills;
            std::copy(prefetch.started_r, prefetch.started_r + 64, started_r);
            std::copy(prefetch.spill_us_r, prefetch.spill_us_r + 64, spill_us_r);
            // a periodic report while reads spill: their time past the close so far
            const auto now = sink_clock::now();
            if (prefetch.spilling) { spill_us += std::chrono::duration<double, std::micro>(now - prefetch.spill_since).count(); }
            for (const auto & r : prefetch.reading) {
                if (r.spill) { spill_us_r[r.job->rank] += uint64_t(std::llround(std::chrono::duration<double, std::micro>(now - r.since).count())); }
            }
        }
        flockfile(stderr);
        std::fprintf(stderr, "[kimi_cpu_sink_prefetch] reason=%s topk=%d layers=%llu predicted=%llu skip_vram=%llu skip_cached=%llu"
                " vram_late=%llu skip_vram_lost=%llu unresolved=%llu no_slot=%llu issued=%llu ready=%llu cancelled=%llu failed=%llu carried=%llu"
                " activated=%llu hits=%llu unused=%llu windows=%llu window_us_p50=%.0f window_us_mean=%.0f"
                " reader_bytes=%llu reader_busy_us=%.0f spill_us=%.0f spills=%llu width=%d",
                reason, prefetch.topk, (unsigned long long) prefetch.layers, (unsigned long long) prefetch.predicted,
                (unsigned long long) prefetch.skip_vram, (unsigned long long) prefetch.skip_cached,
                (unsigned long long) prefetch.vram_late, (unsigned long long) prefetch.skip_vram_lost,
                (unsigned long long) prefetch.unresolved, (unsigned long long) prefetch.no_slot,
                (unsigned long long) prefetch.issued,
                (unsigned long long) prefetch.ready, (unsigned long long) prefetch.cancelled,
                (unsigned long long) prefetch.failed, (unsigned long long) prefetch.carried,
                (unsigned long long) prefetch.activated,
                (unsigned long long) cache.prefetch_hits,
                (unsigned long long) cache.prefetch_unused, (unsigned long long) prefetch.window.count,
                prefetch.window.median_us(), prefetch.window.mean_us(), (unsigned long long) reader_bytes,
                reader_busy_us, spill_us, (unsigned long long) spills, prefetch.width);
        ranks("predicted_r", prefetch.predicted_r); ranks("novel_r", prefetch.novel_r);
        ranks("novel_routed_r", prefetch.novel_routed_r); ranks("issued_r", prefetch.issued_r); ranks("used_r", prefetch.used_r);
        ranks("started_r", started_r); ranks("spill_us_r", spill_us_r);
        std::fprintf(stderr, "\n");
        funlockfile(stderr);
    }
    // Called by a decode run after its own reads were published: the statistics of the hint it got for the next
    // layer, then the reads of its first k ranks. A run without reads of its own opens the drive's idle window here,
    // with a hint or without one.
    void prefetch_schedule(const ggml_tensor * up, bool demand_reads) {
        int layer = -1, consumed = 0;
        const int nhint = prefetch.hint_n;
        prefetch.hint_n = 0;
        if (std::sscanf(up->name, "blk.%d.ffn_up_exps.weight%n", &layer, &consumed) != 1 || !consumed || up->name[consumed]) {
            return;
        }
        if (!nhint || prefetch.hint_layer != layer + 1) {
            // no hint for the next layer (the last MoE layer's run): nothing to read, but a run without reads of its
            // own still leaves the drive idle now, so that the reads still running from earlier stop spilling here
            if (!demand_reads) { prefetch_open_gate(); }
            return;
        }
        ++prefetch.layers;
        if (prefetch.layers % 8192 == 0) { prefetch_report("periodic"); }
        // what the GPU tier holds of the next layer now: its prepare will pin those as hits
        uint8_t held[64] = {};
        vram.held(layer + 1, prefetch.hint, nhint, held);
        for (int i = 0; i < nhint; ++i) {
            const int32_t expert = prefetch.hint[i];
            const bool novel = expert >= 0 && !held[i] && !cache.holds(layer + 1, expert);
            ++prefetch.predicted_r[i];
            prefetch.novel_r[i] += novel;
            prefetch.last_novel[i] = novel;
            prefetch.last_skipped[i] = expert >= 0 && held[i] && i < prefetch.topk;
            prefetch.last_hint[i] = expert;
        }
        prefetch.width = std::max(prefetch.width, nhint);
        prefetch.last_layer = layer + 1;
        prefetch.last_n = nhint;
        // without reads of its own, this run leaves the drive idle now
        if (!demand_reads) { prefetch_open_gate(); }
        const int k = std::min(prefetch.topk, nhint);
        if (!k) { return; }
        char names[3][GGML_MAX_NAME];
        std::snprintf(names[0], sizeof(names[0]), "blk.%d.ffn_up_exps.weight", layer + 1);
        std::snprintf(names[1], sizeof(names[1]), "blk.%d.ffn_gate_exps.weight", layer + 1);
        std::snprintf(names[2], sizeof(names[2]), "blk.%d.ffn_down_exps.weight", layer + 1);
        std::vector<std::unique_ptr<prefetch_job>> jobs;
        jobs.reserve(size_t(k));
        for (int i = 0; i < k; ++i) {
            const int32_t expert = prefetch.hint[i];
            ++prefetch.predicted;
            if (expert < 0) { continue; }
            if (held[i]) { ++prefetch.skip_vram; continue; }
            auto job = std::make_unique<prefetch_job>(); // before the slot is reserved
            job->layer = layer + 1; job->expert = expert; job->rank = i;
            bool resolved = true;
            for (int role = 0; role < 3 && resolved; ++role) {
                resolved = resolver(userdata, names[role], expert, &job->role[role]) &&
                    job->role[role].offset % 4096 == 0 && job->role[role].nbytes % 4096 == 0;
            }
            if (!resolved) { ++prefetch.unresolved; continue; }
            job->entry = cache.prefetch_reserve(layer + 1, expert, job->role);
            if (job->entry == -2) { ++prefetch.skip_cached; continue; }
            if (job->entry < 0) { ++prefetch.no_slot; continue; }
            job->generation = cache.generation;
            for (int role = 0; role < 3; ++role) { job->target[role] = cache.data(job->entry, role); }
            jobs.push_back(std::move(job));
        }
        if (jobs.empty()) { return; }
        {
            std::lock_guard<std::mutex> lock(prefetch.mutex);
            prefetch.jobs.reserve(prefetch.jobs.size() + jobs.size());
            for (auto & job : jobs) {
                prefetch_job * queued = job.get();
                prefetch.jobs.push_back(std::move(job));
                ++prefetch.issued;
                ++prefetch.issued_r[queued->rank];
                // counted as queued, so after a failed allocation the drain settles the job as dropped
                for (int role = 0; role < 3; ++role) {
                    prefetch_alloc(prefetch.throw_schedule, prefetch.allocs_schedule);
                    prefetch.queue.emplace_back(queued, role);
                    ++queued->queued;
                }
            }
        }
        prefetch.cv.notify_all();
    }

    ~kimi_cpu_sink() {
        prefetch_stop();
        host_lock_stop(); // the tier and the arena are both still here
        place_report("destroy");
        prefetch_report("destroy");
        {
            const auto & l = cache.ledger;
            std::fprintf(stderr, "[kimi_host_ledger] reason=destroy unified=%d reserve=%llu margin=%llu floor=%llu file_reserve=%llu explicit_reserve=%llu granted=%llu refused=%llu cache_cgroup_rejections=%llu cache_system_rejections=%llu margin_updates=%llu\n",
                    int(l.unified), (unsigned long long) l.reserve, (unsigned long long) l.margin, (unsigned long long) l.floor(),
                    (unsigned long long) l.file_reserve, (unsigned long long) l.explicit_reserve, (unsigned long long) l.granted,
                    (unsigned long long) ledger_refusals, (unsigned long long) cache.cgroup_rejections,
                    (unsigned long long) cache.system_rejections, (unsigned long long) l.margin_updates);
        }
        cache.before_unmap = nullptr;
        cache.before_release = nullptr; // vram is destroyed before cache
        stop_threads();
        // Explicit unregister is synchronous. Do this before member destruction
        // frees slots; closing only in sink_ring's destructor would be too late.
        ring.close_ring();
    }

    // On unified memory the floor moves between token boundaries: a prefill refaults file pages (the ledger adds
    // them to its file reserve), and allocations outside the ledger take what the tier left. Decode restores it at
    // every token boundary, but a prefill on the graph passes no boundary for minutes. Ten times a second the
    // ledger measures again and, below the floor, the RAM cache's demand and 64 MiB, the tier gives back blocks until
    // MemAvailable keeps the floor, the demand and the tier's low watermark. A sink run in progress keeps run_mutex
    // and is not waited for: the watcher acts between a decode's layers and outside the sink's runs. MemAvailable is
    // read at the full watermark boost, as the tier reads its free memory (kimi-memory-boost.h).
    void floor_watch_loop() {
        std::unique_lock<std::mutex> lock(watch_mutex);
        while (!watch_cv.wait_for(lock, std::chrono::milliseconds(100), [&] { return watch_stop; })) {
            lock.unlock();
            {
                std::unique_lock<std::mutex> run(run_mutex, std::try_to_lock);
                if (run.owns_lock()) {
                    try {
                        refresh_floor("watch");
                        uint64_t available = 0, headroom = 0;
                        if (cache.ledger.memory_available_at_full_boost(available, headroom) && available < kimi_host_ledger::add(cache.ledger.floor(),
                                cache.pending_bytes(cache.stride)) + (uint64_t(64) << 20)) {
                            vram.reclaim_host(0);
                        }
                    } catch (const std::exception & e) {
                        std::fprintf(stderr, "[kimi_host_ledger] event=watch status=fail error=%s\n", e.what());
                    }
                }
            }
            lock.lock();
        }
    }

    void stop_threads() {
        {
            std::lock_guard<std::mutex> lock(watch_mutex);
            watch_stop = true;
        }
        watch_cv.notify_all();
        if (watch_thread.joinable()) { watch_thread.join(); }
        {
            std::lock_guard<std::mutex> lock(compute_mutex);
            compute_stop = true;
        }
        compute_cv.notify_all();
        {
            std::lock_guard<std::mutex> lock(io_mutex);
            io_stop = true;
        }
        io_cv.notify_all();
        if (io_thread.joinable()) { io_thread.join(); }
        for (auto & worker : workers) { if (worker.joinable()) { worker.join(); } }
    }

    int dynamic_first_row(bool is_down) const {
        const int rows = is_down ? latent : hidden;
        const float * target = is_down ? output : activation.data();
        const uintptr_t misalignment = reinterpret_cast<uintptr_t>(target) % 64;
        const int prefix = int((64 - misalignment) % 64) / int(sizeof(float));
        return std::min(rows, prefix);
    }

    void begin_dynamic_phase(bool is_down) {
        if (dynamic_rows && !parallel_fn) {
            dynamic_row_counter.next.store(dynamic_first_row(is_down), std::memory_order_relaxed);
        }
    }

    void compute_rows(int ith, bool is_down) {
        sink_timeline_worker * timing = timeline_active ?
            &timeline[timeline_expert].workers[(is_down ? compute_threads : 0) + ith] : nullptr;
        if (timing) { timing->begin = ggml_time_us(); timing->core = sched_getcpu(); }
        const auto profile_start = profile_enabled ? sink_clock::now() : sink_clock::time_point();
        const double cpu_start = profile_enabled ? sink_thread_cpu_us() : 0.0;
        const int rows = is_down ? latent : hidden;
        const auto & bounds = is_down ? down_bounds : up_bounds;
        if (parallel_fn) {
            parallel_fn(parallel_userdata, ith, compute_threads);
        } else {
            auto compute_range = [&](int begin, int end) {
                if (!is_down && activation_kind == KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP) {
                    // The scalar expression of ggml_compute_forward_swiglu_clamp_f32
                    // (gate=src0, up=src1): each row is bit-identical to the graph op.
                    const float limit = swiglu_limit;
                    for (int row = begin; row < end; ++row) {
                        float u, g;
                        dot_up(latent, &u, 0, up_data + row * row_bytes_up, 0, quant_input.data(), 0, 1);
                        dot_up(latent, &g, 0, gate_data + row * row_bytes_up, 0, quant_input.data(), 0, 1);
                        const float gate_value = std::min(g, limit);
                        const float up_value = std::clamp(u, -limit, limit);
                        activation[row] = gate_value / (1.f + expf(-gate_value)) * up_value;
                    }
                } else if (!is_down) {
                    for (int row = begin; row < end; ++row) {
                        float u, g;
                        dot_up(latent, &u, 0, up_data + row * row_bytes_up, 0, quant_input.data(), 0, 1);
                        dot_up(latent, &g, 0, gate_data + row * row_bytes_up, 0, quant_input.data(), 0, 1);
                        const float a = (beta * std::tanh(g * (1.0f / beta))) * (1.0f / (1.0f + std::exp(-g)));
                        if (linear_beta > 0.0f) { u = linear_beta * std::tanh(u * (1.0f / linear_beta)); }
                        activation[row] = a * u;
                    }
                } else {
                    for (int row = begin; row < end; ++row) {
                        float value;
                        dot_down(hidden, &value, 0, down_data + row * row_bytes_down, 0, down_quant_data, 0, 1);
                        // Preserve the separate multiply followed by the graph's slot-order
                        // addition even when the compiler enables floating point contraction.
                        volatile float scaled = post_exp_norm ? value : value * weight;
                        if (ready_first) { output[row] = scaled; }
                        else { output[row] += scaled; }
                    }
                }
            };
            if (dynamic_rows) {
                const int first = dynamic_first_row(is_down);
                // The caller owns the short unaligned prefix. Every claimed
                // block thereafter begins and ends on an output cache line.
                if (ith == 0 && first) { compute_range(0, first); }
                for (;;) {
                    const int begin = dynamic_row_counter.next.fetch_add(dynamic_chunk_rows, std::memory_order_relaxed);
                    if (begin >= rows) { break; }
                    compute_range(begin, std::min(rows, begin + dynamic_chunk_rows));
                }
            } else {
                const int begin = row_weights.empty() ? rows * ith / active_threads : bounds[ith];
                const int end = row_weights.empty() ? rows * (ith + 1) / active_threads : bounds[ith + 1];
                compute_range(begin, end);
            }
        }
        if (timing) { timing->end = ggml_time_us(); }
        if (profile_enabled) {
            auto & p = worker_profiles[ith];
            const double cpu_us = sink_thread_cpu_us() - cpu_start;
            if (timing) { timing->cpu_us = cpu_us; }
            const double wall_us = elapsed_us(profile_start);
            const double dispatch_us = std::chrono::duration<double, std::micro>(profile_start - phase_dispatch).count();
            p.dispatch_us += dispatch_us;
            p.max_dispatch_us = std::max(p.max_dispatch_us, dispatch_us);
            ++p.phases;
            if (parallel_fn) { p.parallel_wall_us += wall_us; p.parallel_cpu_us += cpu_us; }
            else if (is_down) { p.down_wall_us += wall_us; p.down_cpu_us += cpu_us; }
            else { p.up_gate_wall_us += wall_us; p.up_gate_cpu_us += cpu_us; }
        }
    }

    void compute_loop(int ith) {
        uint64_t seen = 0;
        bool poll_next = false;
        std::unique_lock<std::mutex> lock(compute_mutex);
        for (;;) {
            if (poll_next && !compute_stop && generation == seen) {
                // Poll outside the sleep mutex. Always reacquire it before
                // inspecting the job, so publication and CV wakeup rules stay
                // identical. Long I/O waits and idle requests still park.
                lock.unlock();
                sink_poll(poll_us, [&] { return poll_generation.load(std::memory_order_acquire) != seen; });
                lock.lock();
            }
            while (!compute_stop && generation == seen) {
                compute_cv.wait(lock);
                if (profile_enabled && !compute_stop && generation == seen) {
                    ++worker_profiles[ith].no_work_wakeups;
                }
            }
            if (compute_stop) { return; }
            seen = generation;
            const bool job = compute_expert;
            poll_next = poll_us && !parallel_fn;
            if (!job && ith >= active_threads) { continue; }
            lock.unlock();
            if (job) {
                compute_rows(ith, false);
                expert_arrive(expert_up_pending);
                expert_wait(expert_hidden_pending, expert_ready_cv);
                compute_rows(ith, true);
                expert_arrive(expert_down_pending);
                lock.lock();
                continue;
            }
            compute_rows(ith, down_phase);
            lock.lock();
            if (--pending == 0) {
                if (poll_us) { poll_pending.store(0, std::memory_order_release); }
                if (split_cv) { compute_done_cv.notify_one(); }
                else { compute_cv.notify_all(); }
            }
        }
    }

    // At load, storage before any request, so that no request waits for its page population. On unified memory the
    // GPU tier's (sink_vram::reserve), with the floor and the RAM cache's demand published first, as the first decode
    // layer publishes them before the tier sizes itself there; the RAM cache there only lands the prefetch and follows
    // at the first decode token, from the demand the tier left. Elsewhere the RAM cache's (reserve_cache), with a GPU
    // tier or without one, when the cache can give room back (partial reclaim or elastic). The legacy Q3_K tier has
    // no format ABI and allocates at its first decode layer.
    void reserve_tier(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * down, int experts_used) {
        if (!cache.ledger.unified) { reserve_cache(up, gate, down, experts_used); }
        if (!vram.context || cache.layers.empty() ||
                (up->type == GGML_TYPE_Q3_K && down->type == GGML_TYPE_Q3_K && activation_kind == KIMI_CPU_SINK_ACTIVATION_SITU)) { return; }
        refresh_floor("reserve");
        if (!cache.arena) { vram.set_host_demand(cache.pending_bytes(size_t(up->nb[2] + gate->nb[2] + down->nb[2]))); }
        std::vector<int32_t> layer_ids;
        layer_ids.reserve(cache.layers.size());
        for (const auto & l : cache.layers) { layer_ids.push_back(l.id); }
        vram.reserve(up, gate, down, layer_ids);
    }
    // The RAM cache's arena and pages at load (sink_cache::reserve_at_load), each role as a decode request reserves
    // it (its bytes and a page prefix, in whole pages), after the read ring the first decode charges before the cache
    // sizes itself from the room left; experts_used prices the frequency nodes as the first decode layer does. A cache
    // that cannot give room back takes neither here: the ring and the arena both wait for the first decode token.
    void reserve_cache(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * down, int experts_used) {
        if (cache.layers.empty() || !cache.keep_across_requests || !cache.gives_room_back() || cache.arena) { return; }
        const ggml_tensor * tensors[] = { up, gate, down };
        size_t bytes[3];
        for (int role = 0; role < 3; ++role) {
            bytes[role] = (ggml_row_size(tensors[role]->type, tensors[role]->ne[0]) * tensors[role]->ne[1] + 8190) / 4096 * 4096;
        }
        if (cache.stats.budget_bytes == sink_cache::auto_budget) { reserve_ring(bytes, true); }
        double us = 0;
        cache.reserve_at_load(bytes, up, size_t(std::max(1, experts_used)), compute_threads + 1, us);
    }
    // The ledger's floor moves with the file reserve on unified memory; the GPU tier keeps it free.
    void refresh_floor(const char * reason) {
        auto & ledger = cache.ledger;
        const uint64_t before = ledger.floor();
        if (ledger.update_file_reserve()) {
            // A higher floor is published once the tier has given back what it adds, as far as the tier can now (the
            // watcher repeats it): the file pages that came back keep coming back while the tier holds their memory.
            if (ledger.floor() > before) { vram.reclaim_host(ledger.floor() - before); }
            std::fprintf(stderr, "[kimi_host_ledger] reason=%s floor=%llu file_reserve=%llu margin=%llu explicit_reserve=%llu refaults=%llu updates=%llu\n",
                    reason, (unsigned long long) ledger.floor(), (unsigned long long) ledger.file_reserve,
                    (unsigned long long) ledger.margin, (unsigned long long) ledger.explicit_reserve,
                    (unsigned long long) ledger.refaults, (unsigned long long) ledger.file_updates);
        }
        vram.set_host_floor(ledger.floor());
    }
    // DRAM the engine knows it holds: GPU tier blocks on unified memory, committed RAM cache slots.
    uint64_t held_bytes() { return kimi_host_ledger::add(vram.host_bytes(), cache.stats.committed_bytes); }
    // At a token boundary, before anything resizes: the margin learns what the last decode token took that nobody
    // accounts (unified memory). A boundary that does not follow the previous decode token ends nothing.
    void end_margin_interval(bool decode, uint64_t epoch, uint64_t token) {
        const bool consecutive = decode && margin_boundary && epoch == margin_epoch && token == margin_token + 1;
        margin_boundary = decode; margin_epoch = epoch; margin_token = token;
        auto & ledger = cache.ledger;
        if (!ledger.unified) { return; }
        const uint64_t before = ledger.margin, held = held_bytes();
        uint64_t untracked = 0;
        if (ledger.end_interval(consecutive, held, untracked)) {
            std::fprintf(stderr, "[kimi_host_ledger] event=margin_grow token=%llu untracked=%llu margin_before=%llu margin=%llu held=%llu granted=%llu updates=%llu\n",
                    (unsigned long long) token, (unsigned long long) untracked, (unsigned long long) before,
                    (unsigned long long) ledger.margin, (unsigned long long) held, (unsigned long long) ledger.granted,
                    (unsigned long long) ledger.margin_updates);
        }
    }
    // After the boundary resized the GPU tier and the RAM cache: the next decode token's interval starts here.
    void begin_margin_interval(bool decode) {
        if (decode && cache.ledger.unified) { cache.ledger.begin_interval(held_bytes()); }
    }
    // Room for an allocation outside the expert caches (llama_host_reserve, compute buffers): `host` bytes of host
    // memory and `device` bytes of GPU memory, which count only where the GPU shares the DRAM. No run is in flight:
    // the caller holds run_mutex and has drained the prefetch.
    bool make_room(uint64_t host, uint64_t device) {
        const uint64_t extra = cache.ledger.dram_bytes(host, device);
        if (!extra) { return true; } // nothing comes out of this DRAM (device memory of a discrete GPU)
        refresh_floor("make_room");
        kimi_host_ledger::actions actions;
        actions.collect_deferred = [this] { vram.collect_deferred(cache, -1); };
        actions.reclaim_tier = [this](uint64_t bytes) { return vram.reclaim_host(bytes); };
        actions.reclaim_cache = [this](uint64_t bytes) { return cache.arena && cache.reclaim_for_guard(bytes, "host_reserve", true); };
        const auto room = cache.ledger.make_room(extra, actions);
        if (!room.fits()) {
            ++ledger_refusals;
            std::fprintf(stderr, "[kimi_host_ledger] event=refused extra=%llu term=%s cgroup_shortage=%llu system_shortage=%llu available=%llu floor=%llu\n",
                    (unsigned long long) extra, kimi_host_ledger::short_term(room), (unsigned long long) room.cgroup_shortage,
                    (unsigned long long) room.system_shortage, (unsigned long long) room.available,
                    (unsigned long long) room.floor);
        }
        return room.fits();
    }
    // The layer's plan, once its reads are issued. Lanes (kimi_place_ring_time): the CPU at the measured time per
    // decode expert, of which the up/gate share may start once an expert's first two roles are in; the tier's
    // residents' call, then host calls (its measured lines); on a discrete GPU each moved expert's copy at the measured
    // page-locked rate, behind the copies already queued; the reads landing in issue order (role-major) on the drive's
    // decode line. Candidates: host-ready RAM hits, then misses by landing; on a discrete GPU only experts whose RAM
    // slot is page-locked and that have a device target (their admission reservation, else one of the tier's free
    // scratch slots). The choice: the fewest moves with the lowest
    // predicted time (kimi_place_ring_best_prefix, the list schedule's choice for prefill too). A RAM hit the plan
    // would have moved had its slot been page-locked goes to the front of the registrar's queue.
    void plan_decode_layer() {
        plan.valid = false;
        plan.gpu.assign(requests.size(), 0);
        if (!host_gpu_enabled || !nonblocking_down_sched || !vram.host_ready() || !(cpu_expert_us > 0) ||
                io_requests.size() > slots.size()) { return; }
        kimi_vram_host_costs c = {};
        vram.host_api->costs2(vram.context, &c);
        const bool in_place = c.in_place != 0;
        if (!in_place) { host_lock_decide(); }
        // On a discrete GPU an expert can move only from a page-locked slot, and only once a page-locked copy has been
        // measured (the first page-locked slots' admissions and copies measure it).
        const bool copies = !in_place && host_lock.enabled && c.copy_us_per_mb > 0;
        if (!in_place && !host_lock.enabled) { return; }
        double d0 = 0, us_per_mb = decode_read_bytes_per_us > 0 ? 1e6 / decode_read_bytes_per_us : 0;
        if (decode_lines.reads.size() >= 2 && us_per_mb > 0) {
            kimi_prefill_fit fit;
            for (const auto & x : decode_lines.reads) { fit.add(x.first, x.second); }
            kimi_prefill_fit_line(fit, 0.0, us_per_mb, d0, us_per_mb);
        }
        const double cpu = cpu_expert_us, early = std::clamp(decode_lines.early, 0.0, 1.0) * cpu_expert_us;
        const double resident_b = c.resident_expert_us, host_b = c.host_expert_us > 0 ? c.host_expert_us : c.resident_expert_us;
        auto bytes = [&](const sink_request & r, int role) { return double(r.role[role].nbytes); };
        auto expert_mb = [&](const sink_request & r) { return (bytes(r, 0) + bytes(r, 1) + bytes(r, 2)) / 1e6; };
        std::vector<double> gate_land(requests.size(), 0.0), down_land(requests.size(), 0.0);
        double mb = 0, reads_end = 0;
        for (size_t i : io_requests) { mb += (bytes(requests[i], 0) + bytes(requests[i], 1)) / 1e6; gate_land[i] = d0 + mb * us_per_mb; }
        for (size_t i : io_requests) { mb += bytes(requests[i], 2) / 1e6; down_land[i] = d0 + mb * us_per_mb; }
        if (!io_requests.empty()) { reads_end = d0 + mb * us_per_mb; }
        std::vector<kimi_place_expert> experts(requests.size());
        std::vector<size_t> hits, misses, unlocked;
        std::vector<uint8_t> locked;
        if (!in_place) { locked = host_lock.snapshot(); }
        plan.residents = plan.ram = plan.misses = plan.locked = 0;
        for (size_t i = 0; i < requests.size(); ++i) {
            const auto & r = requests[i];
            auto & x = experts[i];
            if (r.vram_slot >= 0) { x.resident = true; x.gpu = resident_b; ++plan.residents; continue; }
            if (r.cache_only) { continue; }
            x.cpu = cpu; x.early = early; x.gpu = host_b;
            x.link = in_place ? 0.0 : expert_mb(r) * c.copy_us_per_mb;
            if (r.cache_hit >= 0) {
                ++plan.ram;
                if (prefetch_hit_state(r.cache_hit) != 1) { continue; } // still reading: the CPU's, whenever it lands
                const bool page_locked = size_t(r.cache_hit) < locked.size() && locked[size_t(r.cache_hit)];
                plan.locked += page_locked;
                if (in_place || (page_locked && copies)) { hits.push_back(i); }
                else if (!page_locked) { unlocked.push_back(i); }
                continue;
            }
            ++plan.misses;
            x.early_ready = gate_land[i]; x.ready = down_land[i];
            const bool page_locked = r.cache_admit >= 0 && size_t(r.cache_admit) < locked.size() && locked[size_t(r.cache_admit)];
            if (in_place || (page_locked && copies)) { misses.push_back(i); }
        }
        std::stable_sort(misses.begin(), misses.end(), [&](size_t a, size_t b) { return down_land[a] < down_land[b]; });
        // On a discrete GPU a moved expert needs a device target: its admission reservation, else a scratch slot.
        auto targets = [&](std::vector<size_t> & list) {
            if (in_place) { return; }
            size_t scratch = size_t(std::max(0, c.slots));
            std::vector<size_t> kept;
            for (size_t i : list) {
                if (i < 64 && (c.reserved_rows >> i) & 1u) { kept.push_back(i); }
                else if (scratch) { --scratch; kept.push_back(i); }
            }
            list.swap(kept);
        };
        std::vector<size_t> candidates(hits);
        candidates.insert(candidates.end(), misses.begin(), misses.end());
        targets(candidates);
        kimi_place_lanes lanes = {};
        lanes.resident_call = c.resident_call_us;
        lanes.host_call = c.host_call_us > 0 ? c.host_call_us : c.resident_call_us;
        lanes.reads_end = reads_end;
        lanes.link_free = c.link_busy_us;
        lanes.cpu_moved = decode_lines.cpu_moved();
        double time = 0;
        const size_t moves = kimi_place_ring_best_prefix(lanes, experts.data(), experts.size(), candidates.data(),
            candidates.size(), &time);
        for (size_t k = 0; k < moves; ++k) { plan.gpu[candidates[k]] = 1; }
        std::vector<uint8_t> none(requests.size(), 0);
        plan.cpu_only_us = kimi_place_ring_time(lanes, experts.data(), experts.size(), none.data());
        plan.predicted_us = time;
        plan.candidates = candidates.size();
        plan.moves = moves;
        plan.valid = true;
        // Page-locking a slot is bought once, the move it allows pays per use: rent or buy (kimi_host_lock::missed). The
        // plan prices the moves it would make were the unlocked RAM hits' slots page-locked; each such slot accrues its
        // share of the time they would save, and goes to the registrar once that reaches one registration (the
        // registrar's measured mean, before the first one slot at 13.6 GB/s). Before a page-locked copy is measured a
        // copy is priced free, which is no price: nothing accrues, and only the first slot even a free copy would move
        // from is locked, whose registration measures the copy (the tier times one copy of the first range). On 125 a
        // registration in the server takes 2-8 ms against a move's 0.1-0.4 ms: only slots the plan keeps wanting are
        // locked, and a short decode locks few.
        if (!in_place && !unlocked.empty()) {
            std::vector<size_t> all(hits);
            all.insert(all.end(), unlocked.begin(), unlocked.end());
            targets(all);
            double would_time = 0;
            const size_t would = kimi_place_ring_best_prefix(lanes, experts.data(), experts.size(), all.data(), all.size(), &would_time);
            std::vector<size_t> missed;
            std::vector<uint64_t> occupants;
            for (size_t k = 0; k < would; ++k) {
                if (std::find(unlocked.begin(), unlocked.end(), all[k]) == unlocked.end()) { continue; }
                const size_t entry = size_t(requests[all[k]].cache_hit);
                missed.push_back(entry);
                const auto & e = cache.entries[entry];
                occupants.push_back(uint64_t(e.layer_index) << 32 | uint32_t(e.id));
            }
            place.wanted += host_lock.missed(missed, occupants, time - would_time, double(cache.stride) / 13600.0,
                cache.entries.size(), c.copy_us_per_mb > 0);
        }
    }
    // After the layer: its measured lanes' time (reads issued to the tier's finish) against the plan's, the drive's
    // and the layer's fixed time as samples, the receipts.
    void observe_decode_layer(double lanes_us, double total_us) {
        if (current.read_bytes && current.io_us > 0) {
            decode_lines.reads.emplace_back(double(current.read_bytes) / 1e6, current.io_us);
            if (decode_lines.reads.size() > 64) { decode_lines.reads.pop_front(); }
        }
        const double fixed = std::max(0.0, total_us - lanes_us);
        decode_lines.fixed_us = decode_lines.fixed_us > 0 ? 0.9 * decode_lines.fixed_us + 0.1 * fixed : fixed;
        if (!plan.valid) { return; }
        ++place.layers;
        if (vram.token != place.last_token) { ++place.tokens; place.last_token = vram.token; }
        place.moves_planned += plan.moves;
        place.candidates += plan.candidates;
        place.predicted_us += plan.predicted_us; place.measured_us += lanes_us; place.cpu_only_us += plan.cpu_only_us;
        if (lanes_us > 0) {
            const double error = std::fabs(plan.predicted_us - lanes_us) / lanes_us;
            ++place.error[std::min<size_t>(100, size_t(error * 100.0))];
        }
        if (profile_enabled) {
            std::fprintf(stderr, "[kimi_decode_place] token=%llu layer=%d resident=%zu ram=%zu miss=%zu locked=%zu candidates=%zu "
                "planned=%zu predicted_us=%.1f cpu_only_us=%.1f measured_us=%.1f fixed_us=%.1f total_us=%.1f\n",
                (unsigned long long) vram.token, vram.requests.empty() ? -1 : vram.requests[0].layer,
                plan.residents, plan.ram, plan.misses, plan.locked, plan.candidates, plan.moves, plan.predicted_us,
                plan.cpu_only_us, lanes_us, fixed, total_us);
        }
    }
    void place_report(const char * reason) {
        if (!place.layers) { return; }
        auto quantile = [&](double q) {
            uint64_t total = 0, seen = 0;
            for (uint64_t n : place.error) { total += n; }
            for (size_t b = 0; b < place.error.size(); ++b) { seen += place.error[b]; if (double(seen) >= q * double(total)) { return double(b); } }
            return 100.0;
        };
        std::fprintf(stderr, "[kimi_decode_place] reason=%s layers=%llu tokens=%llu gpu_experts=%llu gpu_per_token=%.2f planned=%llu "
            "candidates=%llu refused=%llu wanted=%llu error_p50_pct=%.0f error_p90_pct=%.0f predicted_ms=%.1f measured_ms=%.1f cpu_only_ms=%.1f "
            "cpu_expert_us=%.1f early=%.3f fixed_us=%.1f decode_read_gbs=%.2f cpu_alone_us=%.1f cpu_shared_us=%.1f cpu_moved=%.3f\n",
            reason, (unsigned long long) place.layers, (unsigned long long) place.tokens, (unsigned long long) place.gpu_experts,
            place.tokens ? double(place.gpu_experts) / double(place.tokens) : 0.0, (unsigned long long) place.moves_planned,
            (unsigned long long) place.candidates, (unsigned long long) place.refused, (unsigned long long) place.wanted,
            quantile(0.5), quantile(0.9), place.predicted_us / 1e3, place.measured_us / 1e3, place.cpu_only_us / 1e3,
            cpu_expert_us, decode_lines.early, decode_lines.fixed_us, decode_read_bytes_per_us / 1e3, decode_lines.cpu_alone_us,
            decode_lines.cpu_shared_us, decode_lines.cpu_moved());
    }
    // Whether the registrar runs: a GPU tier whose host path copies (not in place), the host path on
    // (GGML_KIMI_CPU_SINK_HOST_GPU), a page-aligned slot stride, a mapped arena. Decided once the arena exists.
    void host_lock_decide() {
        if (host_lock.decided || !cache.arena || !cache.stride || !vram.context || !vram.host_api) { return; }
        host_lock.decided = true;
        kimi_vram_host_costs c = {};
        vram.host_api->costs2(vram.context, &c);
        const bool enabled = host_gpu_enabled && !c.in_place && cache.stride % cache.page_size == 0 &&
            uintptr_t(cache.arena) % cache.page_size == 0;
        std::fprintf(stderr, "[kimi_host_register] reason=decided enabled=%d in_place=%d host_gpu=%d stride=%zu\n",
            int(enabled), int(c.in_place), int(host_gpu_enabled), cache.stride);
        if (!enabled) { return; }
        host_lock.lock_pages = [this](const void * p, size_t bytes, std::string * why) { return vram.register_host(p, bytes, why); };
        host_lock.unlock_pages = [this](const void * p) { return vram.unregister_host(p); };
        host_lock.start();
    }
    // In a decode layer's wait for reads (io_mutex held): the reads left, at the drive's decode rate, as the shadow a
    // registration may take (kimi_host_lock::offer; before the first registration its price is one slot at 13.6 GB/s).
    // Only wanted slots: a page-locked slot's admission upload is a DMA copy, cheaper, and the tier's admission rule
    // then admits more; with every slot page-locked (25cbe36) 125 admitted 15% more, evicted 25% more and, the RAM
    // copies of admitted experts dropped (exclusive), read 2.5% more experts from the drive.
    void host_lock_in_shadow() {
        host_lock_decide();
        if (!host_lock.enabled || !(decode_read_bytes_per_us > 0)) { return; }
        double left = 0;
        for (size_t index : io_requests) {
            const auto & request = requests[index];
            const auto & slot = slots[request.io_slot];
            for (int role = 0; role < 3; ++role) {
                if (slot.request != index || !(slot.ready_roles & (1u << role))) { left += double(request.role[role].nbytes); }
            }
        }
        host_lock.offer(left / decode_read_bytes_per_us, double(cache.stride) / 13600.0, cache.arena, cache.stride,
            cache.entries.size(), [&](size_t i) { return cache.entries[i].resident; });
    }
    // sink_cache::before_discard: unlock entries [first, first + count) before their pages go.
    void host_unlock(size_t first, size_t count) { host_lock.unlock(first, count, cache.arena, cache.stride); }
    void host_lock_stop() {
        host_lock.stop();
        if (cache.arena) { host_unlock(0, host_lock.snapshot().size()); }
        if (host_lock.decided) {
            std::lock_guard<std::mutex> lock(host_lock.mutex);
            host_lock.report(stderr, "destroy", cache.stride);
        }
        cache.before_discard = nullptr;
    }
    // What an expert the GPU tier does not hold costs this layer: computed by the faster processor that can
    // take it, and for an SSD miss no less than its share of the layer's reads. Feeds the tier's admission.
    void report_miss_cost() {
        size_t ram = 0, ssd = 0;
        for (const auto & r : requests) {
            if (r.vram_slot >= 0 || r.cache_only) { continue; }
            if (r.cache_hit >= 0) { ++ram; } else { ++ssd; }
        }
        if (!(ram + ssd) || !(cpu_expert_us > 0)) { return; }
        double compute = cpu_expert_us;
        // In place a host job's measured time is its whole cost; on a discrete GPU it leaves out the copy.
        kimi_vram_host_costs costs = {};
        if (vram.host_ready()) { vram.host_api->costs2(vram.context, &costs); }
        if (costs.in_place) {
            double resident = 0, host = 0;
            vram.host_costs(resident, host);
            if (host > 0) { compute = std::min(compute, host); }
        }
        const double read = ssd ? current.io_us / double(ssd) : 0.0;
        vram.set_miss_cost((double(ram) * compute + double(ssd) * std::max(compute, read)) / double(ram + ssd));
    }
    void observe_decode_threads(double expert_us) {
        auto & c = decode_threads;
        if (c.locked) { return; }
        if (c.decided >= 0) {
            if (c.candidates[0] == c.candidates[1] || ++c.since_decision < c.recheck_experts) { return; }
            c.decided = -1; c.current = 0; c.us[0] = c.us[1] = 0; c.samples[0] = c.samples[1] = 0;
            return;
        }
        c.us[c.current] += expert_us;
        if (++c.samples[c.current] < c.probe_experts) { return; }
        if (c.current == 0) { c.current = 1; return; }
        const double mean0 = c.us[0] / double(c.samples[0]), mean1 = c.us[1] / double(c.samples[1]);
        c.decided = mean1 < mean0 ? 1 : 0;
        c.since_decision = 0;
        std::fprintf(stderr, "[kimi_cpu_sink_threads] decode_participants=%d us_per_expert=%.1f (%d participants: %.1f)\n",
                c.candidates[c.decided], c.decided ? mean1 : mean0, c.candidates[1 - c.decided], c.decided ? mean0 : mean1);
    }
    void compute_phase(bool is_down) {
        {
            std::lock_guard<std::mutex> lock(compute_mutex);
            compute_expert = false;
            down_phase = is_down;
            begin_dynamic_phase(is_down);
            // Row weights fix one bound per participant, so they keep every thread.
            active_threads = parallel_fn || !row_weights.empty() ? compute_threads :
                decode_threads.candidates[decode_threads.decided >= 0 ? decode_threads.decided : decode_threads.current];
            pending = active_threads - 1;
            if (poll_us) { poll_pending.store(pending, std::memory_order_relaxed); }
            if (profile_enabled) { phase_dispatch = sink_clock::now(); }
            ++generation;
            if (poll_us) { poll_generation.store(generation, std::memory_order_release); }
        }
        compute_cv.notify_all();
        compute_rows(0, is_down);
        wait_workers();
    }

    void wait_workers() {
        if (compute_expert) {
            expert_wait(expert_down_pending, expert_done_cv);
            return;
        }
        if (poll_us && !parallel_fn) {
            sink_poll(poll_us, [&] { return poll_pending.load(std::memory_order_acquire) == 0; });
            if (poll_pending.load(std::memory_order_acquire) == 0) { return; }
        }
        std::unique_lock<std::mutex> lock(compute_mutex);
        if (split_cv) { compute_done_cv.wait(lock, [&] { return pending == 0; }); }
        else { compute_cv.wait(lock, [&] { return pending == 0; }); }
    }

    void expert_wait(std::atomic<int> & remaining, std::condition_variable & cv) {
        // Bounded polling only within a ready expert. Idle workers still park
        // on compute_cv throughout SSD waits, prefill and between requests.
        for (int spin = 0; spin < 128; ++spin) {
            if (remaining.load(std::memory_order_acquire) == 0) { return; }
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#elif defined(__aarch64__)
            __asm__ __volatile__("yield" ::: "memory");
#endif
        }
        std::unique_lock<std::mutex> lock(expert_mutex);
        cv.wait(lock, [&] { return remaining.load(std::memory_order_acquire) == 0; });
    }

    void expert_arrive(std::atomic<int> & remaining) {
        // The RMW release sequence publishes every worker's rows. Only the last
        // worker takes the sleep mutex; locking also prevents a lost CV wakeup.
        if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lock(expert_mutex);
            expert_done_cv.notify_one();
        }
    }

    void quantize_activation() {
        const auto start = sink_clock::now();
        const double cpu_start = profile_enabled ? sink_thread_cpu_us() : 0.0;
        quantize_hidden(activation.data(), quant_hidden.data(), hidden);
        current.hidden_quantize_us += elapsed_us(start);
        if (profile_enabled) { current.compute_cpu_us += sink_thread_cpu_us() - cpu_start; }
    }

    void compute_expert_job() {
        auto * timing = timeline_active ? &timeline[timeline_expert] : nullptr;
        auto phase_start = sink_clock::now();
        if (timing) { timing->cpu[2] = ggml_time_us(); }
        {
            std::lock_guard<std::mutex> lock(compute_mutex);
            compute_expert = true;
            active_threads = compute_threads;
            begin_dynamic_phase(false);
            expert_up_pending.store(compute_threads - 1, std::memory_order_relaxed);
            expert_down_pending.store(compute_threads - 1, std::memory_order_relaxed);
            expert_hidden_pending.store(1, std::memory_order_relaxed);
            if (profile_enabled) { phase_dispatch = phase_start; }
            ++generation;
        }
        compute_cv.notify_all();
        compute_rows(0, false);
        expert_wait(expert_up_pending, expert_done_cv);
        if (timing) { timing->cpu[3] = ggml_time_us(); }
        current.up_gate_us += elapsed_us(phase_start);
        if (timing) { timing->cpu[4] = ggml_time_us(); }
        quantize_activation(); // complete up/gate output is required by Q8_K
        if (timing) { timing->cpu[5] = ggml_time_us(); }
        phase_start = sink_clock::now();
        if (timing) { timing->cpu[6] = ggml_time_us(); }
        if (profile_enabled) { phase_dispatch = phase_start; }
        {
            std::lock_guard<std::mutex> lock(expert_mutex);
            begin_dynamic_phase(true);
            expert_hidden_pending.store(0, std::memory_order_release);
        }
        expert_ready_cv.notify_all();
        compute_rows(0, true);
        expert_wait(expert_down_pending, expert_done_cv);
        if (timing) { timing->cpu[7] = ggml_time_us(); }
        current.down_reduce_us += elapsed_us(phase_start);
        // Caller may now admit/release this expert's cache/ring slot. No worker
        // retains weight/output pointers after its down completion publication.
    }

    void cancel_and_wait() {
        // A provider cancellation must retain its native leases until the
        // consumer has drained and explicitly released them.
        if (!cache_compute_active || !locations.version) { vram.cancel(); }
        {
            std::unique_lock<std::mutex> lock(io_mutex);
            if (!io_done) {
                io_cancel = true;
                io_cv.notify_all();
                io_cv.wait(lock, [&] { return io_done; });
            }
        }
        wait_workers();
    }

    void begin_observation(const ggml_tensor * up, bool decode) {
        timeline_active = timeline_enabled && decode;
        timeline_call = {};
        timeline.clear();
        if (timeline_active) { timeline_call[0] = ggml_time_us(); }
        if (!trace_enabled && !profile_enabled && !timeline_enabled) { return; }
        ++trace_call;
        trace_decode = decode;
        if (!decode) { trace_has_token = false; }
        trace_layer = -1;
        int consumed = 0, layer = -1;
        if (up && std::sscanf(up->name, "blk.%d.ffn_up_exps.weight%n", &layer, &consumed) == 1 &&
                consumed && !up->name[consumed]) { trace_layer = layer; }
        if (profile_enabled) {
            caller_cpu_start = sink_thread_cpu_us();
            std::lock_guard<std::mutex> lock(compute_mutex);
            std::fill(worker_profiles.begin(), worker_profiles.end(), sink_worker_profile());
        }
    }

    void finish_observation() {
        if (!profile_enabled) { return; }
        // Called after the worker barrier and io_done acquire. These counters
        // are separate thread CPU samples, never additions to elapsed wall time.
        current.caller_cpu_us = sink_thread_cpu_us() - caller_cpu_start;
        for (size_t i = trace_decode ? 0 : 1; i < worker_profiles.size(); ++i) {
            const auto & p = worker_profiles[i];
            current.compute_cpu_us += p.up_gate_cpu_us + p.down_cpu_us + p.parallel_cpu_us;
        }
        current.other_us = std::max(0.0, current.total_us - current.compute_us - current.wait_io_us);
    }

    void print_observation(bool success) {
        if (!trace_enabled && !profile_enabled && !timeline_enabled) { return; }
        // Keep each JSON record intact even when server logging uses stderr.
        flockfile(stderr);
        if (timeline_active) {
            std::fprintf(stderr, "[kimi_cpu_sink_timeline] {\"schema\":1,\"clock\":\"CLOCK_MONOTONIC_us\","
                "\"epoch\":%llu,\"token\":%llu,\"layer\":%d,\"call\":%llu,\"success\":%s,\"t\":[",
                (unsigned long long) trace_epoch, (unsigned long long) trace_token, trace_layer,
                (unsigned long long) trace_call, success ? "true" : "false");
            for (size_t i = 0; i < timeline_call.size(); ++i) {
                std::fprintf(stderr, "%s%llu", i ? "," : "", (unsigned long long) timeline_call[i]);
            }
            std::fprintf(stderr, "],\"ready_first\":%s,\"ready_scratch_bytes\":%llu,"
                "\"ready_out_of_order_experts\":%llu,\"ordered_reduce_us\":%.3f,\"experts\":[",
                ready_first ? "true" : "false", (unsigned long long) current.ready_scratch_bytes,
                (unsigned long long) current.ready_out_of_order_experts, current.ordered_reduce_us);
            for (size_t i = 0; i < timeline.size(); ++i) {
                const auto & t = timeline[i];
                std::fprintf(stderr, "%s{\"id\":%d,\"hit\":%s,\"io\":[", i ? "," : "",
                    requests[i].id, requests[i].cache_hit >= 0 ? "true" : "false");
                for (size_t j = 0; j < t.io.size(); ++j) {
                    std::fprintf(stderr, "%s%llu", j ? "," : "", (unsigned long long) t.io[j]);
                }
                std::fprintf(stderr, "],\"cpu\":[");
                for (size_t j = 0; j < t.cpu.size(); ++j) {
                    std::fprintf(stderr, "%s%llu", j ? "," : "", (unsigned long long) t.cpu[j]);
                }
                std::fprintf(stderr, "],\"workers\":[");
                for (size_t j = 0; j < t.workers.size(); ++j) {
                    const auto & w = t.workers[j];
                    std::fprintf(stderr, "%s[%llu,%llu,%.3f,%d]", j ? "," : "",
                        (unsigned long long) w.begin, (unsigned long long) w.end, w.cpu_us, w.core);
                }
                std::fprintf(stderr, "]}");
            }
            std::fprintf(stderr, "]}\n");
        }
        if (trace_enabled) {
            std::fprintf(stderr, "[kimi_cpu_sink_trace] {\"schema\":1,\"event\":\"layer\",\"phase\":\"%s\","
                "\"has_token\":%s,\"epoch\":%llu,\"token\":%llu,\"layer\":%d,\"call\":%llu,\"success\":%s,"
                "\"selected_experts\":[", trace_decode ? "decode" : "prefill", trace_has_token ? "true" : "false",
                (unsigned long long) trace_epoch, (unsigned long long) trace_token, trace_layer,
                (unsigned long long) trace_call, success ? "true" : "false");
            for (size_t i = 0; i < requests.size(); ++i) { std::fprintf(stderr, "%s%d", i ? "," : "", requests[i].id); }
            // the hint this decode run got for the next layer, best first
            std::fprintf(stderr, "],\"hint\":[");
            const int hint_n = trace_decode && prefetch.last_layer == trace_layer + 1 ? prefetch.last_n : 0;
            for (int i = 0; i < hint_n; ++i) { std::fprintf(stderr, "%s%d", i ? "," : "", prefetch.last_hint[i]); }
            std::fprintf(stderr, "],\"expert_bytes\":[");
            for (size_t i = 0; i < requests.size(); ++i) {
                const auto & r = requests[i];
                std::fprintf(stderr, "%s%llu", i ? "," : "", (unsigned long long)
                    (r.role[0].nbytes + r.role[1].nbytes + r.role[2].nbytes));
            }
            std::fprintf(stderr, "],\"cache_hit\":[");
            for (size_t i = 0; i < requests.size(); ++i) {
                std::fprintf(stderr, "%s%s", i ? "," : "", requests[i].cache_hit >= 0 ? "true" : "false");
            }
            std::fprintf(stderr, "],\"ram_resident_before\":[");
            for (size_t i = 0; i < requests.size(); ++i) {
                std::fprintf(stderr, "%s%s", i ? "," : "", requests[i].trace_ram_resident_before ? "true" : "false");
            }
            std::fprintf(stderr, "],\"vram_hit\":[");
            for (size_t i = 0; i < requests.size(); ++i) {
                std::fprintf(stderr, "%s%s", i ? "," : "", requests[i].vram_slot >= 0 ? "true" : "false");
            }
            std::fprintf(stderr, "],\"expert_read_bytes\":[");
            for (size_t i = 0; i < requests.size(); ++i) {
                std::fprintf(stderr, "%s%llu", i ? "," : "", (unsigned long long) requests[i].read_bytes);
            }
            std::fprintf(stderr, "],\"hits\":%llu,\"misses\":%llu,\"read_bytes\":%llu,\"demand_bytes\":%llu,"
                "\"cache_budget_bytes\":%llu,\"cache_capacity_bytes\":%llu,\"cache_max_experts_per_layer\":%d,"
                "\"cache_policy\":\"%s\",\"cache_allocation\":\"%s\",\"cache_layer_priority\":[",
                (unsigned long long) current.cache_hits, (unsigned long long) current.cache_misses,
                (unsigned long long) current.read_bytes, (unsigned long long) current.demand_bytes,
                (unsigned long long) cache.stats.budget_bytes, (unsigned long long) cache.stats.capacity_bytes,
                cache.quota, cache.policy_name(), cache.allocation_name());
            for (size_t i = 0; i < cache.layers.size(); ++i) { std::fprintf(stderr, "%s%d", i ? "," : "", cache.layers[i].id); }
            std::fprintf(stderr, "]}\n");
        }
        if (profile_enabled) {
            std::fprintf(stderr, "[kimi_cpu_sink_profile] {\"schema\":1,\"event\":\"layer\",\"phase\":\"%s\","
                "\"scope\":\"%s\",\"has_token\":%s,\"epoch\":%llu,\"token\":%llu,\"layer\":%d,\"call\":%llu,\"success\":%s,"
                "\"experts\":%llu,\"threads\":%d,\"split_cv\":%d,\"expert_job\":%d,\"total_us\":%.3f,\"compute_wall_us\":%.3f,"
                "\"wait_io_us\":%.3f,\"other_wall_us\":%.3f,\"io_job_wall_us\":%.3f,\"io_wait_wall_us\":%.3f,"
                "\"io_credit_wait_wall_us\":%.3f,\"compute_thread_cpu_us\":%.3f,\"caller_thread_cpu_us\":%.3f,"
                "\"io_thread_cpu_us\":%.3f,\"input_quantize_us\":%.3f,\"hidden_quantize_us\":%.3f,"
                "\"up_gate_us\":%.3f,\"down_reduce_us\":%.3f,\"workers\":[",
                trace_decode ? "decode" : "prefill", trace_decode ? "sink_run" : "sink_stream",
                trace_has_token ? "true" : "false", (unsigned long long) trace_epoch,
                (unsigned long long) trace_token, trace_layer,
                (unsigned long long) trace_call, success ? "true" : "false", (unsigned long long) current.experts,
                compute_threads + 1, split_cv ? 1 : 0, expert_job ? 1 : 0, current.total_us, current.compute_us, current.wait_io_us,
                current.other_us, current.io_us, current.io_wait_us, current.io_credit_wait_us,
                current.compute_cpu_us, current.caller_cpu_us, current.io_cpu_us, current.input_quantize_us,
                current.hidden_quantize_us, current.up_gate_us, current.down_reduce_us);
            std::lock_guard<std::mutex> lock(compute_mutex);
            for (int i = 0; i < compute_threads; ++i) {
                const auto & p = worker_profiles[i];
                std::fprintf(stderr, "%s{\"ith\":%d,\"phases\":%llu,\"up_gate_wall_us\":%.3f,\"down_wall_us\":%.3f,"
                    "\"up_gate_cpu_us\":%.3f,\"down_cpu_us\":%.3f,\"parallel_wall_us\":%.3f,\"parallel_cpu_us\":%.3f,"
                    "\"dispatch_us\":%.3f,\"max_dispatch_us\":%.3f,\"no_work_wakeups\":%llu}",
                    i ? "," : "", i, (unsigned long long) p.phases, p.up_gate_wall_us, p.down_wall_us,
                    p.up_gate_cpu_us, p.down_cpu_us, p.parallel_wall_us, p.parallel_cpu_us,
                    p.dispatch_us, p.max_dispatch_us, (unsigned long long) p.no_work_wakeups);
            }
            std::fprintf(stderr, "]}\n");
        }
        funlockfile(stderr);
    }

    void account_read_bytes(const io_uring_cqe & cqe) {
        if (cqe.user_data >= 3 * slots.size()) { throw std::runtime_error("invalid CPU sink CQE target"); }
        if (cqe.res <= 0) { return; }
        auto & request = requests[slots[cqe.user_data / 3].request];
        if (trace_enabled || cache_compute_active) { request.read_bytes += static_cast<uint64_t>(cqe.res); }
        current.read_bytes += static_cast<uint64_t>(cqe.res);
        if (request.cache_admit >= 0) { current.cache_fill_read_bytes += cqe.res; }
        else { current.ring_read_bytes += cqe.res; }
    }

    void io_loop() {
        std::unique_lock<std::mutex> lock(io_mutex);
        for (;;) {
            io_cv.wait(lock, [&] { return io_stop || io_job; });
            if (io_stop) { return; }
            io_job = false;
            lock.unlock();
            const auto start = sink_clock::now();
            const double cpu_start = profile_enabled ? sink_thread_cpu_us() : 0.0;
            try {
                io_execute();
            } catch (const std::exception & e) {
                // A broken ring cannot be reused. If it cannot even drain, end
                // the process rather than return user buffers still owned by I/O.
                try {
                    ring.drain(this, [](void * userdata, const io_uring_cqe & cqe) {
                        static_cast<kimi_cpu_sink *>(userdata)->account_read_bytes(cqe);
                    });
                } catch (const std::exception & drain_error) {
                    std::fprintf(stderr, "[kimi_cpu_sink] cannot drain I/O after %s: %s; aborting\n",
                                 e.what(), drain_error.what());
                    std::abort();
                }
                ring.close_ring();
                std::lock_guard<std::mutex> error_lock(io_mutex);
                poisoned = true;
                io_error = e.what();
                ++current.read_errors;
            }
            lock.lock();
            current.io_us = elapsed_us(start);
            if (profile_enabled) { current.io_cpu_us = sink_thread_cpu_us() - cpu_start; }
            if (timeline_active) { timeline_call[4] = ggml_time_us(); }
            io_done = true;
            prefetch_open_gate();
            io_cv.notify_all();
        }
    }

    void io_execute() {
        size_t next = 0;
        unsigned inflight = 0;
        bool failed = false;
        while (next < io_requests.size() || inflight) {
            const size_t batch_begin = next;
            unsigned queued = 0;
            {
                std::lock_guard<std::mutex> lock(io_mutex);
                failed = failed || io_cancel;
                auto queue_role = [&](size_t ordinal, unsigned role) {
                    const size_t index = ordinal % slots.size();
                    auto & request = requests[io_requests[ordinal]];
                    const bool cache_fill = request.cache_admit >= 0;
                    const bool external_target = cache_fill ||
                        (request.vram_ssd_transient && !request.vram_ssd_streaming);
                    const bool fixed = ring.read_fixed && !external_target;
                    ring.queue(request.role[role], request.read_target[role], 3 * index + role, fixed);
                    ++queued;
                    ++current.read_requests;
                    if (cache_fill) { ++current.cache_fill_read_requests; }
                    else { ++current.ring_read_requests; }
                    if (fixed) { ++current.fixed_read_requests; }
                };
                while (!failed && next < io_requests.size()) {
                    const size_t index = next % slots.size();
                    sink_slot & slot = slots[index];
                    if (slot.state != sink_slot::empty) { break; }
                    slot.state = sink_slot::reading;
                    slot.remaining = 3;
                    slot.ready_roles = 0;
                    slot.request = io_requests[next];
                    if (timeline_active) { timeline[slot.request].io[0] = ggml_time_us(); }
                    auto & request = requests[slot.request];
                    if (cache_compute_active) { request.events[0] = ggml_time_us(); }
                    if (!role_major) { for (unsigned role = 0; role < 3; ++role) { queue_role(next, role); } }
                    ++next;
                }
                if (role_major) {
                    // Keep one submit for the whole batch, but put every
                    // expert's up/gate before any down so early-up-gate can
                    // overlap the CPU work with the streamed down reads.
                    for (size_t j = batch_begin; j < next; ++j) {
                        queue_role(j, 0);
                        queue_role(j, 1);
                    }
                    for (size_t j = batch_begin; j < next; ++j) { queue_role(j, 2); }
                }
            }
            if (queued) {
                const uint64_t submit_begin = timeline_active || cache_compute_active ? ggml_time_us() : 0;
                ring.submit(queued); inflight += queued;
                if (cache_compute_active) {
                    const uint64_t end = ggml_time_us();
                    for (size_t j = batch_begin; j < next; ++j) {
                        requests[io_requests[j]].events[1] = submit_begin;
                        requests[io_requests[j]].events[2] = end;
                    }
                }
                if (timeline_active) {
                    const uint64_t submit_end = ggml_time_us();
                    for (size_t j = batch_begin; j < next; ++j) {
                        auto & t = timeline[io_requests[j]];
                        t.io[1] = submit_begin; t.io[2] = submit_end;
                    }
                }
            }
            io_uring_cqe cqe;
            bool consumed = false;
            while (ring.pop(cqe)) {
                consumed = true;
                if (!inflight || cqe.user_data >= 3 * slots.size()) {
                    throw std::runtime_error("invalid CPU sink CQE");
                }
                --inflight;
                const size_t index = cqe.user_data / 3;
                const unsigned role = static_cast<unsigned>(cqe.user_data % 3);
                std::lock_guard<std::mutex> lock(io_mutex);
                sink_slot & slot = slots[index];
                if (timeline_active) { timeline[slot.request].io[3 + role] = ggml_time_us(); }
                if (cache_compute_active) { requests[slot.request].events[3 + role] = ggml_time_us(); }
                const auto & range = requests[slot.request].role[role];
                const size_t expected = range.nbytes;
                const size_t required = range.payload_nbytes ? range.data_offset + range.payload_nbytes : expected;
                account_read_bytes(cqe);
                // A final GGUF payload may end before the rounded physical
                // read does. Accept only a complete logical payload, never a
                // truncated weight, and preserve the actual CQE byte count.
                if (cqe.res < 0 || size_t(cqe.res) < required || size_t(cqe.res) > expected) {
                    ++current.read_errors;
                    failed = true;
                    if (io_error.empty()) {
                        io_error = "O_DIRECT expert read failed/short: result=" + std::to_string(cqe.res) +
                                   " expected=" + std::to_string(expected);
                    }
                } else {
                    slot.ready_roles |= 1u << role;
                }
                if (!--slot.remaining) {
                    slot.state = sink_slot::ready;
                    if (cache_compute_active && slot.ready_roles == 7u) { requests[slot.request].ready_us = ggml_time_us(); }
                }
                io_cv.notify_all();
            }
            if (failed && !inflight) { return; }
            if (inflight && !consumed) {
                const auto wait_start = profile_enabled ? sink_clock::now() : sink_clock::time_point();
                ring.wait();
                if (profile_enabled) { current.io_wait_us += elapsed_us(wait_start); }
            } else if (!inflight && next < io_requests.size()) {
                // Always timed: the drive is idle here, and the prefill cost model takes the drive rate
                // from the read bytes over the I/O time without these waits.
                const auto wait_start = sink_clock::now();
                std::unique_lock<std::mutex> lock(io_mutex);
                io_cv.wait(lock, [&] { return io_cancel || slots[next % slots.size()].state == sink_slot::empty; });
                current.io_credit_wait_us += elapsed_us(wait_start);
            }
        }
    }

    size_t reserve_ring(const size_t * max_bytes, bool resident) {
        size_t allocated = 0;
        bool needs_resize = false;
        for (const auto & slot : slots) {
            for (int role = 0; role < 3; ++role) {
                allocated += std::max(slot.role[role].size, max_bytes[role]);
                needs_resize |= slot.role[role].size < max_bytes[role];
            }
        }
        // One alignment page per role is separate from the payload limit.
        const size_t alignment_bytes = slots.size() * 3 * 4096;
        if (allocated > 768ull * 1024 * 1024 + alignment_bytes) {
            throw std::runtime_error("CPU sink ring payload exceeds 768 MiB limit");
        }
        // Serialized runs drain previous consumers before reusing these slots.
        if (ring.pending) { throw std::runtime_error("CPU sink reserve with pending ring I/O"); }
        if (needs_resize) { ring.unregister_buffers(); }
        for (auto & slot : slots) {
            for (int role = 0; role < 3; ++role) {
                slot.role[role].reserve(max_bytes[role]);
                if (resident) { std::memset(slot.role[role].data, 0, slot.role[role].size); }
            }
        }
        return allocated;
    }

    void prepare(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * down,
                 const int32_t * ids, const float * weights, int nslots, bool use_cache = true,
                 const uint8_t * cache_only = nullptr) {
        const ggml_tensor * tensors[] = { up, gate, down };
        for (const auto * tensor : tensors) {
            if (!tensor || !ggml_is_contiguous(tensor) ||
                tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] <= 0 || tensor->ne[3] != 1 ||
                tensor->ne[0] > 16384 || tensor->ne[1] > 16384 ||
                !sink_expert_type_supported(tensor->type, tensor->ne[0])) {
                throw std::runtime_error("CPU sink requires contiguous expert metadata with valid dimensions and a quantized CPU vec_dot type");
            }
        }
        if (up->type != gate->type) {
            throw std::runtime_error("CPU sink requires up and gate experts of one type");
        }
        if (up->ne[0] != gate->ne[0] || up->ne[1] != gate->ne[1] || up->ne[2] != gate->ne[2] ||
            down->ne[0] != up->ne[1] || down->ne[1] != up->ne[0] || down->ne[2] != up->ne[2]) {
            throw std::runtime_error("CPU sink expert shape mismatch");
        }
        // Q3_K SiTU experts use the tier's original ABI; other formats need its
        // explicit-format extension, checked when the tier prepares the layer.
        const bool vram_legacy = up->type == GGML_TYPE_Q3_K && down->type == GGML_TYPE_Q3_K &&
            activation_kind == KIMI_CPU_SINK_ACTIVATION_SITU;
        latent = static_cast<int>(up->ne[0]); hidden = static_cast<int>(up->ne[1]);
        // The graph's MUL_MAT_ID kernels: vec_dot of the weight type against the
        // input quantized to its vec_dot_type (Q3_K/Q8_K, MXFP4/Q8_0, ...).
        const auto * up_traits = ggml_get_type_traits_cpu(up->type);
        const auto * down_traits = ggml_get_type_traits_cpu(down->type);
        dot_up = up_traits->vec_dot;
        dot_down = down_traits->vec_dot;
        quantize_input = ggml_get_type_traits_cpu(up_traits->vec_dot_type)->from_float;
        quantize_hidden = ggml_get_type_traits_cpu(down_traits->vec_dot_type)->from_float;
        row_bytes_up = ggml_row_size(up->type, latent);
        row_bytes_down = ggml_row_size(down->type, hidden);
        requests.clear();
        size_t max_bytes[3] = {};
        for (int slot = 0; slot < nslots; ++slot) {
            if (ids[slot] < 0) { continue; }
            if (ids[slot] >= up->ne[2] || !std::isfinite(weights[slot]) || weights[slot] < 0.0f) {
                throw std::runtime_error("invalid CPU sink expert id or weight");
            }
            sink_request request = {};
            request.weight = weights[slot];
            request.id = ids[slot];
            request.cache_only = cache_only && cache_only[slot];
            for (int role = 0; role < 3; ++role) {
                auto & source = request.role[role];
                if (!resolver(userdata, tensors[role]->name, ids[slot], &source)) {
                    throw std::runtime_error("CPU sink pack resolver failed for " + std::string(tensors[role]->name));
                }
                const size_t bytes = ggml_row_size(tensors[role]->type, tensors[role]->ne[0]) * tensors[role]->ne[1];
                const int flags = fcntl(source.fd_direct, F_GETFL);
                if (flags < 0 || !(flags & O_DIRECT) || source.offset % 4096 || source.nbytes % 4096 ||
                    (source.payload_nbytes && source.payload_nbytes != bytes) || source.data_offset >= 4096 ||
                    source.data_offset > source.nbytes || bytes > source.nbytes - source.data_offset ||
                    source.nbytes > source.data_offset + bytes + 4095 ||
                    source.nbytes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                    source.offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - source.nbytes) {
                    throw std::runtime_error("CPU sink requires an O_DIRECT fd and aligned, exact padded expert range");
                }
                // A later native tensor can have another page prefix. Reserve
                // that maximum once, while submitting only the actual read.
                const size_t capacity = source.payload_nbytes ?
                    std::max(source.nbytes, (bytes + 8190) / 4096 * 4096) : source.nbytes;
                max_bytes[role] = std::max(max_bytes[role], capacity);
                if (!request.cache_only) { current.demand_bytes += source.nbytes; }
            }
            requests.push_back(request);
        }
        if (use_cache && ready_first) {
            // nslots <= 1024 and latent <= 16384: at most 64 MiB. Allocate
            // before the cache admission check so memory.current includes it.
            expert_output.resize(requests.size() * size_t(latent));
            expert_computed.assign(requests.size(), 0);
            current.ready_scratch_bytes = expert_output.capacity() * sizeof(float) + expert_computed.capacity();
        }
        if (!row_weights.empty()) {
            const double scale = *std::max_element(row_weights.begin(), row_weights.end());
            double total = 0, cumulative = 0;
            for (double w : row_weights) { total += w / scale; }
            up_bounds.resize(compute_threads + 1); down_bounds.resize(compute_threads + 1);
            up_bounds[0] = down_bounds[0] = 0;
            for (int ith = 0; ith < compute_threads; ++ith) {
                cumulative += row_weights[ith] / scale;
                up_bounds[ith + 1] = std::min(hidden, int(hidden * cumulative / total));
                down_bounds[ith + 1] = std::min(latent, int(latent * cumulative / total));
            }
            up_bounds.back() = hidden; down_bounds.back() = latent;
        }
        if (use_cache) {
            if (trace_enabled) {
                // Observe immutable RAM residency before either tier changes
                // its metadata. This does not touch or protect the entry.
                for (auto & r : requests) {
                    for (const auto & e : cache.entries) {
                        if (e.valid && cache.layers[e.layer_index].id == trace_layer &&
                                e.id == r.id && sink_cache::matches(e, r)) {
                            r.trace_ram_resident_before = true;
                            break;
                        }
                    }
                }
            }
            const bool provider_path = cache_compute_active && locations.version;
            if (provider_path) { vram.hits = 0; }
            else {
                std::vector<int32_t> vram_layers;
                if (vram.context && !vram_legacy) {
                    vram_layers.reserve(cache.layers.size());
                    for (const auto & l : cache.layers) { vram_layers.push_back(l.id); }
                }
                // The tier may size itself in this call (its first decode layer) while the RAM cache has not mapped
                // its slots yet: tell it first what the cache will commit, which on unified memory is the same DRAM.
                if (vram.context && !cache.arena) {
                    vram.set_host_demand(cache.pending_bytes(max_bytes[0] + max_bytes[1] + max_bytes[2]));
                }
                vram.prepare(requests, up, down, latent, hidden, vram_legacy, vram_layers, cache, post_exp_norm != nullptr);
            }
            // Auto sizing reads memory.current. After GPU prefill, the CPU
            // read ring may still be absent or only virtually allocated. Its
            // first read/registration would otherwise consume the headroom
            // just assigned to the optional cache. Charge this mandatory
            // staging before learning cache geometry, while no cache entry
            // or current I/O job exists. CPU prefill normally touched it already.
            if (cache.active && !cache.paused && !cache.stride &&
                    cache.stats.budget_bytes == sink_cache::auto_budget &&
                    std::any_of(requests.begin(), requests.end(),
                        [](const sink_request & r) { return r.vram_slot < 0; })) {
                const size_t allocated = reserve_ring(max_bytes, true);
                std::fprintf(stderr, "[kimi_cpu_sink_io] auto_cache_ring_resident=1 bytes=%zu\n", allocated);
            }
            if (vram.hits) {
                // GPU hits do not consume RAM cache quota or SSD ring credits.
                std::vector<sink_request> cpu_requests;
                for (const auto & r : requests) { if (r.vram_slot < 0) { cpu_requests.push_back(r); } }
                cache.prepare(cpu_requests, tensors, max_bytes, current);
                size_t j = 0;
                for (auto & r : requests) { if (r.vram_slot < 0) { r = cpu_requests[j++]; } }
                for (size_t i = 0; i < requests.size(); ++i) { expert_computed[i] = requests[i].vram_slot >= 0; }
            } else {
                cache.prepare(requests, tensors, max_bytes, current);
            }
            vram.pin_ram_batch(cache, requests);
        }
        io_requests.clear();
        for (size_t i = 0; i < requests.size(); ++i) {
            if (requests[i].cache_hit >= 0 || requests[i].vram_slot >= 0 || requests[i].cache_only) { continue; }
            requests[i].io_slot = io_requests.size() % slots.size();
            io_requests.push_back(i);
        }
        current.cache_misses = io_requests.size();
        if (use_cache) {
            cache.stats.hits += current.cache_hits;
            cache.stats.misses += current.cache_misses;
        }
        const size_t allocated = reserve_ring(max_bytes, false);
        for (auto & slot : slots) {
            slot.state = sink_slot::empty;
            slot.ready_roles = 0;
        }
        bool has_ring_reads = false;
        for (size_t index : io_requests) {
            auto & request = requests[index];
            has_ring_reads |= request.cache_admit < 0;
            for (int role = 0; role < 3; ++role) {
                request.read_target[role] = request.cache_admit >= 0 ?
                    cache.data(request.cache_admit, role) : slots[request.io_slot].role[role].data;
            }
        }
        if (ring.read_fixed && has_ring_reads && !ring.buffers_registered) {
            std::vector<iovec> buffers;
            buffers.reserve(3 * slots.size());
            for (const auto & slot : slots) {
                for (const auto & buffer : slot.role) { buffers.push_back({ buffer.data, buffer.size }); }
            }
            ring.register_buffers(buffers);
        }
        current.ring_bytes = allocated;
        current.experts = requests.size();
        activation.resize(hidden);
        quant_input.resize(ggml_row_size(up_traits->vec_dot_type, latent));
        quant_hidden_stride = ggml_row_size(down_traits->vec_dot_type, hidden);
        quant_hidden.resize(quant_hidden_stride);
        down_quant_data = quant_hidden.data();
        if (nonblocking_down_sched) {
            if (quant_hidden_stride && requests.size() > std::numeric_limits<size_t>::max() / quant_hidden_stride) {
                throw std::runtime_error("nonblocking down scheduler scratch overflow");
            }
            staged_quant_hidden.resize(requests.size() * quant_hidden_stride);
            expert_stage.assign(requests.size(), 0);
            expert_upload_started.assign(requests.size(), 0);
            for (size_t i = 0; i < requests.size(); ++i) {
                if (requests[i].vram_slot >= 0) { expert_stage[i] = 2; }
            }
            current.ready_scratch_bytes = expert_output.capacity() * sizeof(float) + expert_computed.capacity() +
                staged_quant_hidden.capacity() + expert_stage.capacity() + expert_upload_started.capacity();
        }
    }

    // Role-major makes many up/gate pairs ready before their down reads. Treat
    // those halves as independent work: never reserve the caller on one
    // expert's late down CQE while another expert's up/gate can run.
    void run_nonblocking_down_schedule() {
        enum work_type { no_work, up_gate_work, down_work, gpu_work };
        const size_t cpu_experts = std::count_if(expert_stage.begin(), expert_stage.end(),
            [](uint8_t stage) { return stage != 2; });
        size_t finished = 0, down_ordinal = 0;
        // Where an expert runs: the layer's plan (plan_decode_layer). A planned expert goes to the GPU once it is whole,
        // with every other whole planned expert in one engine batch; on a discrete GPU the tier copies them first. A
        // refused batch puts the rest of the layer on the CPU.
        bool host_gpu = plan.valid && std::find(plan.gpu.begin(), plan.gpu.end(), uint8_t(1)) != plan.gpu.end();
        auto planned = [&](size_t i) { return host_gpu && i < plan.gpu.size() && plan.gpu[i]; };
        bool moved = false; // host jobs launched in this layer
        std::fill(expert_cpu_us.begin(), expert_cpu_us.end(), 0.0);
        expert_cpu_us.resize(requests.size(), 0.0);
        while (finished < cpu_experts) {
            size_t expert = requests.size();
            work_type work = no_work;
            bool upload_source_complete = false;
            const auto ready_wait_start = sink_clock::now();
            const uint64_t ready_wait_us = timeline_active ? ggml_time_us() : 0;
            {
                std::unique_lock<std::mutex> lock(io_mutex);
                auto choose = [&] {
                    // Up/gate work always wins. Fully ready experts do not
                    // preempt a partial expert, because neither choice blocks.
                    for (size_t i = 0; i < requests.size(); ++i) {
                        if (expert_stage[i] != 0) { continue; }
                        const auto & r = requests[i];
                        if (r.cache_hit >= 0) {
                            if (prefetch_hit_state(r.cache_hit) != 1) { continue; }
                            expert = i; work = planned(i) ? gpu_work : up_gate_work; upload_source_complete = true; return;
                        }
                        const auto & slot = slots[r.io_slot];
                        if (slot.request == i && (slot.ready_roles & 7u) == 7u && planned(i)) {
                            expert = i; work = gpu_work; upload_source_complete = true; return;
                        }
                        // The GPU takes whole experts only; leave a partial planned one to it rather than start it here.
                        if (slot.request == i && (slot.ready_roles & 3u) == 3u && !planned(i)) {
                            expert = i; work = up_gate_work;
                            upload_source_complete = r.cache_admit >= 0 && (slot.ready_roles & 4u);
                            return;
                        }
                    }
                    for (size_t i = 0; i < requests.size(); ++i) {
                        if (expert_stage[i] != 1) { continue; }
                        const auto & r = requests[i];
                        if (r.cache_hit >= 0) { expert = i; work = down_work; return; }
                        const auto & slot = slots[r.io_slot];
                        if (slot.request == i && (slot.ready_roles & 4u)) {
                            expert = i; work = down_work; return;
                        }
                    }
                };
                io_cv.wait(lock, [&] {
                    if (!io_error.empty()) { return true; }
                    expert = requests.size(); work = no_work; upload_source_complete = false;
                    choose();
                    if (work != no_work || prefetch_hit_failed()) { return true; }
                    if (!io_done || prefetch_hits_pending()) { host_lock_in_shadow(); return false; }
                    // a prefetch read may have completed since choose() looked at its hit
                    choose();
                    return true;
                });
                current.wait_io_us += elapsed_us(ready_wait_start);
                if (!io_error.empty()) { break; }
                if (work == no_work && prefetch_hit_failed()) {
                    throw std::runtime_error("CPU sink prefetch read of a used expert failed");
                }
                if (work == no_work) {
                    for (size_t i = 0; i < requests.size(); ++i) {
                        const auto & r = requests[i];
                        std::fprintf(stderr, "[kimi_cpu_sink] stuck expert=%zu id=%d stage=%d cache_hit=%d cache_admit=%d vram_slot=%d"
                            " slot_request=%zu ready_roles=%u prefetch_pending=%d\n", i, r.id, int(expert_stage[i]), r.cache_hit,
                            r.cache_admit, r.vram_slot, slots[r.io_slot].request, unsigned(slots[r.io_slot].ready_roles),
                            r.cache_hit >= 0 ? prefetch_hit_state(r.cache_hit) : -1);
                    }
                    throw std::runtime_error("nonblocking down scheduler has no runnable work");
                }
                if (timeline_active && work == up_gate_work) {
                    timeline[expert].cpu[0] = ready_wait_us;
                    timeline[expert].cpu[1] = ggml_time_us();
                }
            }

            const auto & request = requests[expert];
            const void * data[3];
            for (int role = 0; role < 3; ++role) {
                data[role] = static_cast<const char *>(request.cache_hit >= 0 ? cache.data(request.cache_hit, role) : request.read_target[role]) + request.role[role].data_offset;
            }
            const bool borrow_upload = request.cache_hit >= 0 || request.cache_admit >= 0;
            up_data = static_cast<const char *>(data[0]);
            gate_data = static_cast<const char *>(data[1]);
            down_data = static_cast<const char *>(data[2]);
            weight = request.weight;
            output = expert_output.data() + expert * size_t(latent);
            timeline_expert = expert;

            if (work == gpu_work) {
                // Every other whole planned expert goes in the same engine batch.
                std::vector<size_t> batch = { expert };
                {
                    std::lock_guard<std::mutex> lock(io_mutex);
                    for (size_t i = 0; i < requests.size(); ++i) {
                        if (i == expert || expert_stage[i] != 0 || !planned(i)) { continue; }
                        const auto & r = requests[i];
                        const bool whole = r.cache_hit >= 0 ? prefetch_hit_state(r.cache_hit) == 1 :
                            (slots[r.io_slot].request == i && (slots[r.io_slot].ready_roles & 7u) == 7u);
                        if (whole) { batch.push_back(i); }
                    }
                }
                std::vector<kimi_vram_host_job> jobs;
                std::vector<std::array<const void *, 3>> payloads;
                for (size_t i : batch) {
                    const auto & r = requests[i];
                    std::array<const void *, 3> payload;
                    for (int role = 0; role < 3; ++role) {
                        payload[role] = static_cast<const char *>(r.cache_hit >= 0 ? cache.data(r.cache_hit, role) : r.read_target[role]) + r.role[role].data_offset;
                    }
                    payloads.push_back(payload);
                    // A post-norm model weights each expert after its norm (the reduction below), as the
                    // resident tier requests do: the GPU returns the unweighted output.
                    jobs.push_back({ int32_t(i), post_exp_norm ? 1.0f : r.weight, { payload[0], payload[1], payload[2] } });
                }
                if (vram.launch_host(jobs)) {
                    place.gpu_experts += jobs.size();
                    moved = true;
                    for (size_t k = 0; k < batch.size(); ++k) {
                        const size_t i = batch[k];
                        const auto & r = requests[i];
                        // Tier admission as on the CPU path, from the same complete payload, after the launch: a discrete
                        // GPU's job may have taken the expert's reservation, and then this admit has nothing to do.
                        const bool borrow = r.cache_hit >= 0 || r.cache_admit >= 0;
                        if (vram.async_upload && borrow) { vram.admit(i, payloads[k].data(), true); }
                        else { vram.admit(i, payloads[k].data(), borrow); }
                        expert_upload_started[i] = 1;
                        cache.admit(requests[i], current);
                        expert_stage[i] = 2;
                        expert_computed[i] = 1;
                        ++finished;
                    }
                    continue;
                }
                // The tier took nothing for this layer: compute them here, and the rest of the layer too.
                ++place.refused;
                host_gpu = false;
                work = up_gate_work;
            }
            if (work == up_gate_work) {
                const auto compute_start = sink_clock::now();
                if (vram.async_upload && borrow_upload && upload_source_complete && !expert_upload_started[expert]) {
                    vram.admit(expert, data, true);
                    expert_upload_started[expert] = 1;
                }
                auto phase_start = sink_clock::now();
                if (timeline_active) { timeline[expert].cpu[2] = ggml_time_us(); }
                compute_phase(false);
                if (timeline_active) { timeline[expert].cpu[3] = ggml_time_us(); }
                current.up_gate_us += elapsed_us(phase_start);

                phase_start = sink_clock::now();
                const double quant_cpu_start = profile_enabled ? sink_thread_cpu_us() : 0.0;
                if (timeline_active) { timeline[expert].cpu[4] = ggml_time_us(); }
                quantize_hidden(activation.data(), staged_quant_hidden.data() + expert * quant_hidden_stride, hidden);
                if (timeline_active) { timeline[expert].cpu[5] = ggml_time_us(); }
                current.hidden_quantize_us += elapsed_us(phase_start);
                if (profile_enabled) { current.compute_cpu_us += sink_thread_cpu_us() - quant_cpu_start; }
                const double up_gate_us = elapsed_us(compute_start);
                current.compute_us += up_gate_us;
                expert_cpu_us[expert] += up_gate_us;
                expert_stage[expert] = 1;
                continue;
            }

            const auto compute_start = sink_clock::now();
            if (!expert_upload_started[expert] && vram.async_upload && borrow_upload) {
                // Down readiness proves that all three immutable payloads are
                // complete. The upload and CPU down projection may borrow them
                // concurrently, exactly as in the blocking path.
                vram.admit(expert, data, true);
                expert_upload_started[expert] = 1;
            }
            down_quant_data = staged_quant_hidden.data() + expert * quant_hidden_stride;
            const auto phase_start = sink_clock::now();
            if (timeline_active) { timeline[expert].cpu[6] = ggml_time_us(); }
            compute_phase(true);
            if (timeline_active) { timeline[expert].cpu[7] = ggml_time_us(); }
            current.down_reduce_us += elapsed_us(phase_start);
            current.compute_us += elapsed_us(compute_start);
            // This machine's CPU time per decode expert, the cost the GPU placement compares against.
            const double expert_us = expert_cpu_us[expert] + elapsed_us(phase_start);
            cpu_expert_us = cpu_expert_us > 0 ? 0.9 * cpu_expert_us + 0.1 * expert_us : expert_us;
            if (expert_us > 0) {
                const double early = expert_cpu_us[expert] / expert_us; // the up/gate share
                decode_lines.early = decode_lines.early > 0 ? 0.9 * decode_lines.early + 0.1 * early : early;
                if (io_requests.empty()) {
                    double & line = moved ? decode_lines.cpu_shared_us : decode_lines.cpu_alone_us;
                    line = line > 0 ? 0.9 * line + 0.1 * expert_us : expert_us;
                }
            }
            observe_decode_threads(expert_us);
            if (!expert_upload_started[expert]) { vram.admit(expert, data, borrow_upload); }
            cache.admit(request, current);
            if (request.cache_hit < 0) {
                std::lock_guard<std::mutex> lock(io_mutex);
                slots[request.io_slot].state = sink_slot::empty;
            }
            io_cv.notify_one();
            expert_stage[expert] = 2;
            expert_computed[expert] = 1;
            current.ready_out_of_order_experts += expert != down_ordinal;
            ++down_ordinal;
            ++finished;
        }
    }

    // io_mutex must be held. Cache hits were pinned for this whole batch by
    // prepare(). A ready ring slot may belong to a different wrap generation.
    size_t next_ready() const {
        for (size_t i = 0; i < requests.size(); ++i) {
            if (expert_computed[i]) { continue; }
            const auto & r = requests[i];
            const auto & slot = slots[r.io_slot];
            if ((r.cache_hit >= 0 && prefetch_hit_state(r.cache_hit) == 1) ||
                    (slot.state == sink_slot::ready && slot.request == i)) { return i; }
        }
        // Use partial readiness only after all fully ready experts. Otherwise
        // a partial miss could delay an already cached expert behind its down I/O.
        if (early_up_gate) {
            for (size_t i = 0; i < requests.size(); ++i) {
                const auto & slot = slots[requests[i].io_slot];
                if (!expert_computed[i] && slot.state == sink_slot::reading && slot.request == i &&
                        (slot.ready_roles & 3u) == 3u) { return i; }
            }
        }
        return requests.size();
    }

    double wait_down(size_t expert) {
        if (!early_up_gate || requests[expert].cache_hit >= 0) { return 0; }
        const auto start = sink_clock::now();
        std::unique_lock<std::mutex> lock(io_mutex);
        const auto & slot = slots[requests[expert].io_slot];
        io_cv.wait(lock, [&] {
            return (slot.request == expert && (slot.ready_roles & 4u)) || !io_error.empty();
        });
        const double elapsed = elapsed_us(start);
        current.wait_io_us += elapsed;
        // No down worker has started. Error unwinding can drain all remaining
        // reads before either cache or ring storage is released.
        if (!io_error.empty()) { throw std::runtime_error(io_error); }
        return elapsed;
    }
};

// A prefill consumer runs while the stream owns run_mutex, so these workers
// and the role buffers cannot concurrently be used by decode.
void kimi_cpu_sink_parallel(struct kimi_cpu_sink * sink,
        kimi_cpu_sink_parallel_fn function, void * userdata) {
    sink->parallel_fn = function;
    sink->parallel_userdata = userdata;
    try { sink->compute_phase(false); }
    catch (...) {
        sink->wait_workers();
        sink->parallel_fn = nullptr;
        sink->parallel_userdata = nullptr;
        throw;
    }
    sink->parallel_fn = nullptr;
    sink->parallel_userdata = nullptr;
}

static bool kimi_cpu_sink_stream_impl(struct kimi_cpu_sink * sink,
        const struct ggml_tensor * up, const struct ggml_tensor * gate,
        const struct ggml_tensor * down, const int32_t * ids, int nexperts,
        struct kimi_cpu_sink_prefill_batch * batch,
        kimi_cpu_sink_consumer_fn consumer, void * userdata,
        struct kimi_cpu_sink_stats * stats, char * error, size_t error_size) {
    sink_error(error, error_size, "");
    if (stats) { *stats = {}; }
    if (!sink) { sink_error(error, error_size, "null CPU sink"); return false; }
    std::lock_guard<std::mutex> serial(sink->run_mutex); sink->prefetch_drain();
    const auto start = sink_clock::now();
    const uint64_t completed_before = sink->ring.completed;
    sink->current = {};
    sink->requests.clear();
    sink->begin_observation(up, false);
    // The tier's grouped API reduces routed outputs before returning them. A
    // per-expert normalization must happen before that reduction, so retain
    // native CPU grouped computation for that capability, even when the caller
    // enables GPU prefill globally. Decode's per-expert tier API remains usable.
    struct restore_prefill_flag {
        bool & flag;
        bool saved;
        ~restore_prefill_flag() { flag = saved; }
    } restore { sink->vram.prefill_enabled, sink->vram.prefill_enabled };
    if (batch && batch->activation.post_exp_norm) { sink->vram.prefill_enabled = false; }
    try {
        sink_caller_affinity affinity(sink->compute_cpus);
        // The default remains ring-only. CACHE_PREFILL_READ may borrow retained
        // decode entries under explicit leases after the phase headroom check.
        sink->refresh_floor("prefill");
        sink->vram.begin(0, 0, false, sink->cache);
        sink->cache.begin(0, 0, false);
        if (!ids || !consumer || nexperts < 0 || nexperts > 1024 ||
                (batch && (!batch->quantized_input || !batch->f32_input || !batch->output ||
                    (batch->nslots && (!batch->route_request_index || !batch->weights)) || batch->nslots < 0 || batch->nslots > 16 ||
                    batch->ntokens < 1 || (batch->nslots &&
                    batch->ntokens > std::numeric_limits<int>::max() / batch->nslots)))) {
            throw std::runtime_error("invalid CPU expert stream arguments");
        }
        if (sink->poisoned) { throw std::runtime_error("CPU sink ring is poisoned after an I/O failure"); }
        if (batch) {
            sink->post_exp_norm = batch->activation.post_exp_norm;
            sink->activation_kind = batch->activation.kind;
            sink->beta = batch->activation.situ_beta;
            sink->linear_beta = batch->activation.situ_linear_beta;
            sink->swiglu_limit = batch->activation.swiglu_limit;
            batch->gpu_experts = batch->gpu_routes = 0;
            batch->gpu_transient_experts = batch->gpu_transient_routes = 0;
            batch->gpu_transient_upload_bytes = 0;
            batch->gpu_transient_upload_us = batch->gpu_worker_us = 0.0;
            batch->gpu_expert_us = batch->gpu_read_wait_us = 0.0;
            batch->gpu_ssd_transient_candidates = batch->gpu_ssd_transient_experts = 0;
            batch->gpu_ssd_transient_routes = batch->gpu_ssd_transient_upload_bytes = 0;
            batch->gpu_ssd_transient_upload_us = 0.0;
            batch->gpu_cold_admit_candidates = batch->gpu_cold_admit_experts = 0;
            batch->gpu_cold_admit_routes = batch->gpu_cold_admit_upload_bytes = 0;
            batch->gpu_cold_admit_upload_us = 0.0;
            batch->ram_experts = batch->ram_routes = batch->ram_bytes = 0;
            batch->ssd_experts = batch->ssd_routes = batch->ssd_bytes = 0;
        }
        std::vector<float> weights(static_cast<size_t>(nexperts), 1.0f);
        sink->prepare(up, gate, down, ids, weights.data(), nexperts, false);
        if (sink->requests.size() != static_cast<size_t>(nexperts)) {
            throw std::runtime_error("CPU expert stream requires nonnegative expert IDs");
        }
        std::vector<size_t> ssd_transient_requests;
        // Set once prepare_prefill has seen this layer's format: before it, the first MXFP4 call of a
        // process would still see no format and stage its transients the Q2 way.
        bool streaming_ssd = false;
        auto streaming_stage = +[](void * context, int32_t request_index, int phase) -> int {
            auto * owner = static_cast<kimi_cpu_sink *>(context);
            if (request_index < 0 || size_t(request_index) >= owner->requests.size()) { return 0; }
            const auto & request = owner->requests[size_t(request_index)];
            auto & slot = owner->slots[request.io_slot];
            std::unique_lock<std::mutex> lock(owner->io_mutex);
            if (phase == 0) {
                owner->io_cv.wait(lock, [&] {
                    return (slot.state == sink_slot::ready && slot.request == size_t(request_index)) ||
                        !owner->io_error.empty() || owner->io_cancel || owner->io_done;
                });
                return slot.state == sink_slot::ready && slot.request == size_t(request_index) &&
                    owner->io_error.empty() && !owner->io_cancel;
            }
            if (phase == 2) {
                return slot.state == sink_slot::ready && slot.request == size_t(request_index) &&
                    owner->io_error.empty() && !owner->io_cancel;
            }
            if (phase != 1 || slot.state != sink_slot::ready ||
                    slot.request != size_t(request_index)) { return 0; }
            slot.state = sink_slot::empty;
            lock.unlock();
            owner->io_cv.notify_all();
            return 1;
        };
        if (batch) {
            std::vector<int32_t> vram_layers;
            vram_layers.reserve(sink->cache.layers.size());
            for (const auto & layer : sink->cache.layers) { vram_layers.push_back(layer.id); }
            batch->gpu_experts = sink->vram.prepare_prefill(
                sink->requests, up, down, sink->latent, sink->hidden, vram_layers);
            streaming_ssd = sink->vram.format_mxfp4_input && sink->vram.mxfp4_batch_enabled;
            if (batch->gpu_experts) {
                const size_t route_count = size_t(batch->nslots) * size_t(batch->ntokens);
                for (size_t route = 0; route < route_count; ++route) {
                    const int request = batch->route_request_index[route];
                    if (request >= 0 && size_t(request) < sink->requests.size() &&
                            sink->requests[size_t(request)].vram_slot >= 0) {
                        ++batch->gpu_routes;
                    }
                }
            }
            // Source order is strict: retained VRAM, then a read-only leased
            // RAM entry, then O_DIRECT. This lookup does not train or admit the
            // decode cache and an SSD miss never receives a duplicate RAM slot.
            sink->cache.prepare_prefill_read(sink->requests, up, gate, down, sink->current);
            // Streamed MXFP4 transients hold a ring slot and a device slot each, whatever their number, so
            // only the optional cap limits them; Q2 staging owns one host buffer per transient.
            const int streaming_cap = sink->vram.mxfp4_prefill_ssd_gpu_max_experts_per_layer;
            const size_t staging_limit = !streaming_ssd ? sink->slots.size() :
                streaming_cap < 0 ? sink->requests.size() : size_t(streaming_cap);
            if (streaming_ssd) {
                std::fprintf(stderr,
                    "[kimi_mxfp4_prefill_staging] selected_cap=%zu ring_buffers=%zu streaming=1\n",
                    staging_limit, sink->slots.size());
            }
            sink->vram.select_prefill_transient(
                sink->requests, *batch, sink->cache, staging_limit, sink->slots.size());

            size_t ssd_transient_count = 0;
            for (const auto & request : sink->requests) {
                if (!request.vram_ssd_transient) { continue; }
                if (request.vram_ssd_staging < 0) {
                    throw std::runtime_error("Q2 SSD prefill staging index is absent");
                }
                ssd_transient_count = std::max(
                    ssd_transient_count, size_t(request.vram_ssd_staging) + 1);
            }
            if (ssd_transient_count > staging_limit) {
                throw std::runtime_error("SSD prefill staging exceeds the allowed capacity");
            }
            if (!streaming_ssd && sink->prefill_ssd_staging.size() < ssd_transient_count) {
                sink->prefill_ssd_staging.resize(ssd_transient_count);
            }
            ssd_transient_requests.assign(ssd_transient_count, std::numeric_limits<size_t>::max());
            for (size_t i = 0; i < sink->requests.size(); ++i) {
                auto & request = sink->requests[i];
                if (!request.vram_ssd_transient) { continue; }
                const size_t staging = size_t(request.vram_ssd_staging);
                if (staging >= ssd_transient_requests.size() ||
                        ssd_transient_requests[staging] != std::numeric_limits<size_t>::max()) {
                    throw std::runtime_error("invalid duplicate Q2 SSD prefill staging index");
                }
                ssd_transient_requests[staging] = i;
                if (!streaming_ssd) {
                    const void * data[3];
                    for (int role = 0; role < 3; ++role) {
                        auto & buffer = sink->prefill_ssd_staging[staging].role[role];
                        buffer.reserve(request.role[role].nbytes);
                        request.read_target[role] = buffer.data;
                        data[role] = static_cast<const char *>(buffer.data) + request.role[role].data_offset;
                    }
                    sink->vram.bind_prefill_ssd_transient(i, data);
                } else {
                    request.vram_ssd_streaming = true;
                }
            }
            if (std::find(ssd_transient_requests.begin(), ssd_transient_requests.end(),
                    std::numeric_limits<size_t>::max()) != ssd_transient_requests.end()) {
                throw std::runtime_error("non-contiguous Q2 SSD prefill staging indices");
            }

            sink->io_requests.clear();
            auto append_io_request = [&](size_t i) {
                auto & request = sink->requests[i];
                request.io_slot = sink->io_requests.size() % sink->slots.size();
                if (!request.vram_ssd_transient || request.vram_ssd_streaming) {
                    for (int role = 0; role < 3; ++role) {
                        request.read_target[role] = sink->slots[request.io_slot].role[role].data;
                    }
                }
                sink->io_requests.push_back(i);
            };
            std::vector<size_t> cpu_ssd_requests;
            for (size_t i = 0; i < sink->requests.size(); ++i) {
                const auto & request = sink->requests[i];
                if (request.vram_slot < 0 && request.cache_hit < 0 && !request.vram_transient) {
                    cpu_ssd_requests.push_back(i);
                }
            }
            if (streaming_ssd && !sink->vram.mxfp4_prefill_legacy_schedule) {
                // Issue the reads in the order the two consumers need them: by each expert's predicted start
                // on its lane (the makespan rule's lane_start_us), so that both lanes run through the whole
                // layer. Each consumer keeps its own order, so a ring credit is always returned by an earlier
                // read of a consumer that is not waiting on a later one.
                size_t gpu_next = 0, cpu_next = 0;
                while (gpu_next < ssd_transient_requests.size() || cpu_next < cpu_ssd_requests.size()) {
                    const bool gpu_first = cpu_next == cpu_ssd_requests.size() ||
                        (gpu_next < ssd_transient_requests.size() &&
                         sink->requests[ssd_transient_requests[gpu_next]].lane_start_us <=
                         sink->requests[cpu_ssd_requests[cpu_next]].lane_start_us);
                    append_io_request(gpu_first ? ssd_transient_requests[gpu_next++] : cpu_ssd_requests[cpu_next++]);
                }
            } else if (streaming_ssd) {
                // Keep each consumer's request order while sharing the eight
                // ring credits. CPU reads start before the GPU backlog drains.
                size_t gpu_next = 0, cpu_next = 0;
                while (gpu_next < ssd_transient_requests.size() || cpu_next < cpu_ssd_requests.size()) {
                    for (int j = 0; j < 4 && gpu_next < ssd_transient_requests.size(); ++j) {
                        append_io_request(ssd_transient_requests[gpu_next++]);
                    }
                    for (int j = 0; j < 4 && cpu_next < cpu_ssd_requests.size(); ++j) {
                        append_io_request(cpu_ssd_requests[cpu_next++]);
                    }
                }
            } else {
                // Q2 staging owns its payload through finish. Reads release
                // only the ring credit before CPU expert processing starts.
                for (size_t i : ssd_transient_requests) { append_io_request(i); }
                for (size_t i : cpu_ssd_requests) { append_io_request(i); }
            }
            if (streaming_ssd) {
                for (size_t i : ssd_transient_requests) {
                    const auto & request = sink->requests[i];
                    const void * data[3] = {
                        static_cast<const char *>(request.read_target[0]) + request.role[0].data_offset,
                        static_cast<const char *>(request.read_target[1]) + request.role[1].data_offset,
                        static_cast<const char *>(request.read_target[2]) + request.role[2].data_offset,
                    };
                    sink->vram.bind_prefill_ssd_transient(i, data, streaming_stage, sink);
                }
            }
            sink->current.cache_misses = sink->io_requests.size();
            sink->cache.stats.prefill_misses += sink->current.cache_misses;
            batch->ram_experts = sink->current.cache_hits;
            batch->ram_bytes = sink->current.cache_hit_bytes;
            batch->ssd_experts = sink->io_requests.size();
            const size_t route_count = size_t(batch->nslots) * size_t(batch->ntokens);
            for (size_t route = 0; route < route_count; ++route) {
                const int request = batch->route_request_index[route];
                if (request < 0 || size_t(request) >= sink->requests.size()) { continue; }
                const auto & source = sink->requests[size_t(request)];
                if (source.vram_slot >= 0) { continue; }
                if (source.cache_hit >= 0) { ++batch->ram_routes; }
                else { ++batch->ssd_routes; }
            }
            if (ssd_transient_requests.empty()) {
                sink->vram.launch_prefill(batch->quantized_input, batch->quantized_stride,
                    batch->f32_input, batch->f32_stride, batch->route_request_index, batch->weights,
                    batch->nslots, batch->ntokens, batch->activation);
            }
        }
        {
            std::lock_guard<std::mutex> lock(sink->io_mutex);
            sink->io_error.clear(); sink->io_cancel = false;
            if (!sink->io_requests.empty()) { sink->io_done = false; sink->io_job = true; }
        }
        sink->io_cv.notify_one();
        if (batch && streaming_ssd && !ssd_transient_requests.empty()) {
            sink->vram.launch_prefill(batch->quantized_input, batch->quantized_stride,
                batch->f32_input, batch->f32_stride, batch->route_request_index, batch->weights,
                batch->nslots, batch->ntokens, batch->activation);
        }
        if (batch && !ssd_transient_requests.empty() && !streaming_ssd) {
            // launch_prefill() borrows the staging pointers until finish. Wait
            // for every selected read, then return only the I/O credits; never
            // expose recyclable ring storage to the asynchronous GPU worker.
            for (size_t request_index : ssd_transient_requests) {
                auto & request = sink->requests[request_index];
                sink_slot & slot = sink->slots[request.io_slot];
                const auto wait_start = sink_clock::now();
                {
                    std::unique_lock<std::mutex> lock(sink->io_mutex);
                    sink->io_cv.wait(lock, [&] {
                        return (slot.state == sink_slot::ready && slot.request == request_index) ||
                            !sink->io_error.empty();
                    });
                    sink->current.wait_io_us += elapsed_us(wait_start);
                    if (!sink->io_error.empty()) { throw std::runtime_error(sink->io_error); }
                    slot.state = sink_slot::empty;
                }
                sink->io_cv.notify_one();
            }
            sink->vram.launch_prefill(batch->quantized_input, batch->quantized_stride,
                batch->f32_input, batch->f32_stride, batch->route_request_index, batch->weights,
                batch->nslots, batch->ntokens, batch->activation);
        }
        for (size_t expert = 0; expert < sink->requests.size(); ++expert) {
            auto & request = sink->requests[expert];
            if (request.vram_slot >= 0 || request.vram_transient) { continue; }
            sink_slot * slot = nullptr;
            if (request.cache_hit < 0) {
                slot = &sink->slots[request.io_slot];
                const auto wait_start = sink_clock::now();
                {
                    std::unique_lock<std::mutex> lock(sink->io_mutex);
                    sink->io_cv.wait(lock, [&] {
                        return (slot->state == sink_slot::ready && slot->request == expert) ||
                            !sink->io_error.empty();
                    });
                    sink->current.wait_io_us += elapsed_us(wait_start);
                    if (!sink->io_error.empty()) { break; }
                }
            }
            const auto compute_start = sink_clock::now();
            const double cpu_start = sink->profile_enabled ? sink_thread_cpu_us() : 0.0;
            const void * data[3];
            for (int role = 0; role < 3; ++role) {
                data[role] = static_cast<const char *>(request.cache_hit >= 0 ? sink->cache.data(request.cache_hit, role) : request.read_target[role]) + request.role[role].data_offset;
            }
            try {
                consumer(userdata, expert, data[0], data[1], data[2]);
            } catch (...) {
                if (request.cache_lease) {
                    sink->cache.release_lease(request.cache_hit);
                    request.cache_lease = false;
                }
                throw;
            }
            if (request.cache_lease) {
                sink->cache.release_lease(request.cache_hit);
                request.cache_lease = false;
            }
            const double compute_us = elapsed_us(compute_start);
            sink->current.compute_us += compute_us;
            if (batch) { sink->vram.observe_prefill_cpu(request.prefill_routes, compute_us); }
            if (sink->profile_enabled) { sink->current.compute_cpu_us += sink_thread_cpu_us() - cpu_start; }
            // A cold admission copies synchronously before this O_DIRECT ring
            // slot is returned to the I/O worker. It is timed separately from
            // CPU expert compute and never borrows the ring pointer.
            if (batch && request.vram_cold_admit) {
                sink->vram.admit_cold_prefill(expert, data, request, *batch);
            }
            if (slot) {
                std::lock_guard<std::mutex> lock(sink->io_mutex);
                slot->state = sink_slot::empty;
            }
            if (slot) { sink->io_cv.notify_one(); }
        }
        {
            const auto wait_start = sink_clock::now();
            std::unique_lock<std::mutex> lock(sink->io_mutex);
            sink->io_cv.wait(lock, [&] { return sink->io_done; });
            sink->current.wait_io_us += elapsed_us(wait_start);
            if (!sink->io_error.empty()) { throw std::runtime_error(sink->io_error); }
        }
        if (batch && (batch->gpu_experts || !sink->vram.prefill_transient.empty())) {
            sink->vram.finish_prefill(batch->output, size_t(sink->latent), *batch);
        }
        if (batch) {
            sink->vram.calibrate_prefill(sink->current.compute_us, *batch, sink->current.read_bytes,
                sink->current.io_us - sink->current.io_credit_wait_us, elapsed_us(start));
        }
        if (!sink->requests.empty()) {
            const auto & r = sink->requests.front();
            sink->vram.set_host_demand(sink->cache.pending_bytes(r.role[0].nbytes + r.role[1].nbytes + r.role[2].nbytes));
        }
        if (batch) {
            // CPU-consumed RAM leases were released immediately. Transient GPU
            // sources remain leased until finish drained H2D and compute.
            sink->cache.release_prefill_leases(sink->requests);
            batch->ssd_bytes = sink->current.read_bytes;
            std::fprintf(stderr,
                "[kimi_cpu_prefill_sources] tensor=%s vram_experts=%llu vram_routes=%llu "
                "ram_gpu_experts=%llu ram_gpu_routes=%llu ram_gpu_upload_bytes=%llu ram_gpu_upload_us=%.3f "
                "ssd_gpu_candidates=%llu ssd_gpu_experts=%llu ssd_gpu_routes=%llu "
                "ssd_gpu_upload_bytes=%llu ssd_gpu_upload_us=%.3f gpu_worker_us=%.3f "
                "gpu_expert_us=%.3f gpu_read_wait_us=%.3f cold_admit_candidates=%llu cold_admit_experts=%llu "
                "cold_admit_routes=%llu cold_admit_upload_bytes=%llu cold_admit_upload_us=%.3f "
                "ram_experts=%llu ram_routes=%llu ram_bytes=%llu "
                "ssd_experts=%llu ssd_routes=%llu ssd_bytes=%llu ssd_read_us=%.1f\n",
                up->name, (unsigned long long) batch->gpu_experts,
                (unsigned long long) batch->gpu_routes,
                (unsigned long long) batch->gpu_transient_experts,
                (unsigned long long) batch->gpu_transient_routes,
                (unsigned long long) batch->gpu_transient_upload_bytes,
                batch->gpu_transient_upload_us,
                (unsigned long long) batch->gpu_ssd_transient_candidates,
                (unsigned long long) batch->gpu_ssd_transient_experts,
                (unsigned long long) batch->gpu_ssd_transient_routes,
                (unsigned long long) batch->gpu_ssd_transient_upload_bytes,
                batch->gpu_ssd_transient_upload_us,batch->gpu_worker_us,
                batch->gpu_expert_us,batch->gpu_read_wait_us,
                (unsigned long long) batch->gpu_cold_admit_candidates,
                (unsigned long long) batch->gpu_cold_admit_experts,
                (unsigned long long) batch->gpu_cold_admit_routes,
                (unsigned long long) batch->gpu_cold_admit_upload_bytes,
                batch->gpu_cold_admit_upload_us,
                (unsigned long long) batch->ram_experts,
                (unsigned long long) batch->ram_routes,
                (unsigned long long) batch->ram_bytes, (unsigned long long) batch->ssd_experts,
                (unsigned long long) batch->ssd_routes,
                (unsigned long long) batch->ssd_bytes, sink->current.io_us - sink->current.io_credit_wait_us);
        }
        if (sink->timeline_active) { sink->timeline_call[5] = ggml_time_us(); }
        sink->current.total_us = elapsed_us(start);
        sink->current.read_completions = sink->ring.completed - completed_before;
        sink->current.pending_reads = sink->ring.pending;
        sink->finish_observation();
        if (stats) { *stats = sink->current; }
        sink->print_observation(true);
        return true;
    } catch (const std::exception & e) {
        // Stop new submissions and drain before returning, including an
        // unexpected exception in a consumer. No user buffer may outlive I/O.
        sink->cancel_and_wait();
        if (batch) { sink->vram.cancel_prefill(); }
        if (batch) { sink->cache.release_prefill_leases(sink->requests); }
        sink->cache.release();
        if (sink->timeline_active) { sink->timeline_call[5] = ggml_time_us(); }
        sink->current.total_us = elapsed_us(start);
        sink->current.read_completions = sink->ring.completed - completed_before;
        sink->current.pending_reads = sink->ring.pending;
        sink->finish_observation();
        if (stats) { *stats = sink->current; }
        sink->print_observation(false);
        sink_error(error, error_size, e.what());
        return false;
    }
}

bool kimi_cpu_sink_stream(struct kimi_cpu_sink * sink,
        const struct ggml_tensor * up, const struct ggml_tensor * gate,
        const struct ggml_tensor * down, const int32_t * ids, int nexperts,
        kimi_cpu_sink_consumer_fn consumer, void * userdata,
        struct kimi_cpu_sink_stats * stats, char * error, size_t error_size) {
    return kimi_cpu_sink_stream_impl(sink, up, gate, down, ids, nexperts, nullptr,
            consumer, userdata, stats, error, error_size);
}

bool kimi_cpu_sink_stream_hybrid(struct kimi_cpu_sink * sink,
        const struct ggml_tensor * up, const struct ggml_tensor * gate,
        const struct ggml_tensor * down, const int32_t * ids, int nexperts,
        struct kimi_cpu_sink_prefill_batch * batch,
        kimi_cpu_sink_consumer_fn consumer, void * userdata,
        struct kimi_cpu_sink_stats * stats, char * error, size_t error_size) {
    return kimi_cpu_sink_stream_impl(sink, up, gate, down, ids, nexperts, batch,
            consumer, userdata, stats, error, error_size);
}

extern "C" struct kimi_cpu_sink * kimi_cpu_sink_create(
        int total_threads, int ring_experts, kimi_cpu_sink_resolver resolver,
        void * userdata, char * error, size_t error_size) {
    sink_error(error, error_size, "");
    try {
        if (total_threads < 2 || total_threads > 256 || ring_experts < 1 || ring_experts > 32 || !resolver) {
            throw std::runtime_error("CPU sink requires 2..256 total threads, 1..32 ring experts, and a resolver");
        }
        return new kimi_cpu_sink(total_threads, ring_experts, resolver, userdata);
    } catch (const std::exception & e) {
        sink_error(error, error_size, e.what());
        return nullptr;
    }
}

extern "C" void kimi_cpu_sink_destroy(struct kimi_cpu_sink * sink) { delete sink; }

extern "C" void kimi_cpu_sink_prefetch_hint(struct kimi_cpu_sink * sink, int32_t layer, const int32_t * experts, int n) {
    if (!sink || !experts || n <= 0) { return; }
    std::lock_guard<std::mutex> serial(sink->run_mutex);
    sink->prefetch.hint_n = std::min(n, 64);
    std::copy(experts, experts + sink->prefetch.hint_n, sink->prefetch.hint);
    sink->prefetch.hint_layer = layer;
}

extern "C" void kimi_cpu_sink_cache_reset(struct kimi_cpu_sink * sink) {
    if (!sink) { return; }
    std::lock_guard<std::mutex> serial(sink->run_mutex); sink->prefetch_drain();
    sink->vram.cancel();
    sink->cache.release();
    sink->cache.active = sink->cache.has_token = sink->cache.paused = false;
}

extern "C" bool kimi_cpu_sink_cache_configure(struct kimi_cpu_sink * sink, uint64_t budget,
        const int32_t * priority, int n, int quota, char * error, size_t error_size) {
    sink_error(error, error_size, "");
    if (!sink) { sink_error(error, error_size, "null CPU sink"); return false; }
    std::lock_guard<std::mutex> serial(sink->run_mutex); sink->prefetch_drain();
    try { sink->vram.begin(0, 0, false, sink->cache); sink->cache.configure(budget, priority, n, quota); return true; }
    catch (const std::exception & e) { sink_error(error, error_size, e.what()); return false; }
}

extern "C" bool kimi_cpu_sink_cache_begin(struct kimi_cpu_sink * sink, uint64_t epoch,
        uint64_t token, bool decode, char * error, size_t error_size) {
    sink_error(error, error_size, "");
    if (!sink) { sink_error(error, error_size, "null CPU sink"); return false; }
    std::lock_guard<std::mutex> serial(sink->run_mutex); sink->prefetch_drain();
    try {
        sink->end_margin_interval(decode, epoch, token);
        sink->refresh_floor(decode ? "decode" : "prefill");
        sink->vram.begin(epoch, token, decode, sink->cache);
        sink->cache.begin(epoch, token, decode);
        sink->begin_margin_interval(decode);
    } catch (const std::exception & e) {
        sink->vram.cancel(); sink->cache.release();
        sink_error(error, error_size, e.what()); return false;
    }
    if (!decode) { sink->mrc.report("request"); }
    sink->trace_epoch = epoch;
    sink->trace_token = token;
    sink->trace_has_token = decode;
    sink->trace_decode = decode;
    if (sink->trace_enabled && !decode) {
        std::fprintf(stderr, "[kimi_cpu_sink_trace] {\"schema\":1,\"event\":\"reset\",\"phase\":\"prefill\","
            "\"epoch\":%llu,\"token\":%llu}\n", (unsigned long long) epoch, (unsigned long long) token);
    }
    return true;
}

extern "C" bool kimi_cpu_sink_vram_phase_complete(
        struct kimi_cpu_sink * sink, bool decode, char * error, size_t error_size) {
    sink_error(error, error_size, "");
    if (!sink) { sink_error(error, error_size, "null CPU sink"); return false; }
    std::lock_guard<std::mutex> serial(sink->run_mutex); sink->prefetch_drain();
    try {
        sink->vram.phase_complete(decode);
        return true;
    } catch (const std::exception & e) {
        sink_error(error, error_size, e.what());
        return false;
    }
}

extern "C" bool kimi_cpu_sink_vram_reserve_accumulator(
        struct kimi_cpu_sink * sink, int tokens, char * error, size_t error_size) {
    sink_error(error, error_size, "");
    if (!sink) { sink_error(error, error_size, "null CPU sink"); return false; }
    std::lock_guard<std::mutex> serial(sink->run_mutex); sink->prefetch_drain();
    try { sink->vram.reserve_accumulate(tokens); return true; }
    catch (const std::exception & e) { sink_error(error, error_size, e.what()); return false; }
}

extern "C" bool kimi_cpu_sink_vram_reserve(struct kimi_cpu_sink * sink, const struct ggml_tensor * up,
        const struct ggml_tensor * gate, const struct ggml_tensor * down, int experts_used, char * error, size_t error_size) {
    sink_error(error, error_size, "");
    if (!sink || !up || !gate || !down) { sink_error(error, error_size, "null CPU sink or expert tensor"); return false; }
    std::lock_guard<std::mutex> serial(sink->run_mutex); sink->prefetch_drain();
    try {
        // The tier's threads take the affinity of the thread that creates them: that of a sink call, as when
        // the first decode or prefill call creates them.
        sink_caller_affinity affinity(sink->compute_cpus);
        sink->reserve_tier(up, gate, down, experts_used);
        return true;
    }
    catch (const std::exception & e) { sink_error(error, error_size, e.what()); return false; }
}

extern "C" void kimi_cpu_sink_cache_get_stats(struct kimi_cpu_sink * sink, kimi_cpu_sink_cache_stats * stats) {
    if (!stats) { return; }
    *stats = {};
    if (!sink) { return; }
    std::lock_guard<std::mutex> serial(sink->run_mutex);
    *stats = sink->cache.stats;
}

extern "C" bool kimi_cpu_sink_cache_make_room(struct kimi_cpu_sink * sink, uint64_t host_bytes, uint64_t device_bytes) {
    if (!sink) {
        // Before the first graph creates the sink: the ledger alone, from this process's environment and /proc.
        try {
            kimi_host_ledger ledger;
            ledger.configure(sink_unified_memory());
            const uint64_t extra = ledger.dram_bytes(host_bytes, device_bytes);
            if (!extra) { return true; }
            ledger.update_file_reserve();
            return ledger.measure(extra, true).fits();
        } catch (const std::exception & e) {
            std::fprintf(stderr, "[kimi_host_ledger] make_room failed: %s\n", e.what());
            return false;
        }
    }
    std::lock_guard<std::mutex> serial(sink->run_mutex); sink->prefetch_drain();
    try { return sink->make_room(host_bytes, device_bytes); }
    catch (const std::exception & e) {
        std::fprintf(stderr, "[kimi_cpu_sink_cache] make_room failed: %s\n", e.what());
        return false;
    }
}

static bool sink_activation_valid(const kimi_cpu_sink_activation * activation) {
    if (!activation) { return false; }
    switch (activation->kind) {
        case KIMI_CPU_SINK_ACTIVATION_SITU:
            return std::isfinite(activation->situ_beta) && activation->situ_beta > 0.0f &&
                std::isfinite(activation->situ_linear_beta);
        case KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP:
            return !std::isnan(activation->swiglu_limit) && activation->swiglu_limit > 1e-6f;
        default:
            return false;
    }
}

extern "C" bool kimi_cpu_sink_run_activation(
        struct kimi_cpu_sink * sink, const struct ggml_tensor * up, const struct ggml_tensor * gate,
        const struct ggml_tensor * down, const float * input, const int32_t * ids, const float * weights,
        int nslots, const struct kimi_cpu_sink_activation * activation, float * output,
        struct kimi_cpu_sink_stats * stats, char * error, size_t error_size) {
    sink_error(error, error_size, "");
    if (stats) { *stats = {}; }
    if (!sink) { sink_error(error, error_size, "null CPU sink"); return false; }
    std::lock_guard<std::mutex> serial(sink->run_mutex);
    const auto start = sink_clock::now();
    const uint64_t completed_before = sink->ring.completed;
    sink->current = {};
    sink->requests.clear();
    sink->begin_observation(up, true);
    try {
        sink_caller_affinity affinity(sink->compute_cpus);
        if (!input || !ids || !weights || !output || nslots < 0 || nslots > 1024 ||
            !sink_activation_valid(activation)) {
            throw std::runtime_error(activation && activation->kind == KIMI_CPU_SINK_ACTIVATION_SITU ?
                "invalid CPU sink input or SiTU parameters" : "invalid CPU sink input or activation parameters");
        }
        if (sink->poisoned) { throw std::runtime_error("CPU sink ring is poisoned after an I/O failure"); }
        {
            // with the ids checked: an in-flight prefetch of an expert this run uses becomes a hit it waits for
            int layer = -1, consumed = 0;
            if (up && std::sscanf(up->name, "blk.%d.ffn_up_exps.weight%n", &layer, &consumed) == 1 && consumed &&
                    !up->name[consumed]) {
                sink->prefetch_drain(layer, ids, nslots);
                sink->mrc.observe(layer, ids, nslots);
            } else {
                sink->prefetch_drain();
            }
        }
        sink->post_exp_norm = activation->post_exp_norm;
        if (sink->post_exp_norm && (!sink->ready_first || !std::isfinite(activation->post_exp_norm_eps) ||
                activation->post_exp_norm_eps <= 0.0f)) {
            throw std::runtime_error("per-expert post norm requires READY_FIRST=1 and positive finite epsilon");
        }
        sink->activation_kind = activation->kind;
        sink->beta = activation->situ_beta; sink->linear_beta = activation->situ_linear_beta;
        sink->swiglu_limit = activation->swiglu_limit;
        sink->prepare(up, gate, down, ids, weights, nslots);
        sink->prefetch_keep_used_hits();
        sink->prefetch_count_used();
        if (sink->timeline_active) {
            sink->timeline.resize(sink->requests.size());
            for (auto & t : sink->timeline) { t.workers.resize(2 * sink->compute_threads); }
            sink->timeline_call[1] = ggml_time_us();
        }
        sink->output = output;
        auto compute_start = sink_clock::now();
        double quant_cpu_start = sink->profile_enabled ? sink_thread_cpu_us() : 0.0;
        sink->quantize_input(input, sink->quant_input.data(), sink->latent);
        std::fill(output, output + sink->latent, 0.0f);
        sink->current.input_quantize_us = elapsed_us(compute_start);
        if (sink->profile_enabled) { sink->current.compute_cpu_us += sink_thread_cpu_us() - quant_cpu_start; }
        sink->current.compute_us += sink->current.input_quantize_us;
        sink->vram.launch(sink->quant_input.data(), sink->quant_input.size(), input,
                          size_t(sink->latent) * sizeof(float), *activation);
        if (sink->timeline_active) { sink->timeline_call[2] = ggml_time_us(); }
        sink->plan.start = sink_clock::now();
        {
            std::lock_guard<std::mutex> lock(sink->io_mutex);
            sink->io_error.clear();
            sink->io_cancel = false;
            if (sink->timeline_active) { sink->timeline_call[3] = ggml_time_us(); }
            if (!sink->io_requests.empty()) { sink->io_done = false; sink->io_job = true; }
        }
        sink->io_cv.notify_one();
        sink->prefetch_schedule(up, !sink->io_requests.empty());
        // The plan reads nothing the reads change (its landing times start at plan.start): planned once they run.
        sink->plan_decode_layer();
        if (sink->nonblocking_down_sched) {
            sink->run_nonblocking_down_schedule();
        }
        const size_t blocking_experts = sink->nonblocking_down_sched ? 0 :
            sink->requests.size() - size_t(sink->vram.hits);
        for (size_t ordinal = 0; ordinal < blocking_experts; ++ordinal) {
            size_t expert = ordinal;
            const auto ready_wait_start = sink->ready_first ? sink_clock::now() : sink_clock::time_point();
            const uint64_t ready_wait_us = sink->timeline_active ? ggml_time_us() : 0;
            if (sink->ready_first) {
                std::unique_lock<std::mutex> lock(sink->io_mutex);
                sink->io_cv.wait(lock, [&] {
                    expert = sink->next_ready();
                    return expert < sink->requests.size() || !sink->io_error.empty();
                });
                sink->current.wait_io_us += elapsed_us(ready_wait_start);
                if (!sink->io_error.empty()) { break; }
                sink->expert_computed[expert] = 1;
                sink->current.ready_out_of_order_experts += expert != ordinal;
                sink->output = sink->expert_output.data() + expert * size_t(sink->latent);
                if (sink->timeline_active) {
                    sink->timeline[expert].cpu[0] = ready_wait_us;
                    sink->timeline[expert].cpu[1] = ggml_time_us();
                }
            }
            const auto & request = sink->requests[expert];
            sink_slot & slot = sink->slots[request.io_slot];
            const auto wait_start = sink_clock::now();
            if (request.cache_hit < 0 && !sink->ready_first) {
                if (sink->timeline_active) { sink->timeline[expert].cpu[0] = ggml_time_us(); }
                std::unique_lock<std::mutex> lock(sink->io_mutex);
                sink->io_cv.wait(lock, [&] {
                    return slot.state == sink_slot::ready ||
                        (sink->early_up_gate && slot.request == expert && (slot.ready_roles & 3u) == 3u) ||
                        !sink->io_error.empty();
                });
                sink->current.wait_io_us += elapsed_us(wait_start);
                if (!sink->io_error.empty()) { break; }
                if (sink->timeline_active) { sink->timeline[expert].cpu[1] = ggml_time_us(); }
            }
            compute_start = sink_clock::now();
            const void * data[3];
            for (int role = 0; role < 3; ++role) {
                data[role] = static_cast<const char *>(request.cache_hit >= 0 ? sink->cache.data(request.cache_hit, role) : request.read_target[role]) + request.role[role].data_offset;
            }
            const bool borrow_upload = request.cache_hit >= 0 || request.cache_admit >= 0;
            // A cache hit has all three roles. A cache admission is also an I/O
            // target, and early-up-gate may expose it while down is still in
            // flight. Do not let an asynchronous upload borrow that incomplete
            // payload; wait_down() supplies the next safe admission point.
            bool upload_source_complete = request.cache_hit >= 0;
            if (request.cache_admit >= 0) {
                std::lock_guard<std::mutex> lock(sink->io_mutex);
                upload_source_complete = slot.request == expert && (slot.ready_roles & 4u);
            }
            bool upload_started = sink->vram.async_upload && borrow_upload && upload_source_complete;
            if (upload_started) { sink->vram.admit(expert, data, true); }
            sink->up_data = static_cast<const char *>(data[0]);
            sink->gate_data = static_cast<const char *>(data[1]);
            sink->down_data = static_cast<const char *>(data[2]);
            sink->weight = request.weight;
            double down_wait_us = 0;
            if (sink->timeline_active) { sink->timeline_expert = expert; }
            if (sink->expert_job) {
                sink->compute_expert_job();
            } else {
                auto phase_start = sink_clock::now();
                if (sink->timeline_active) { sink->timeline_expert = expert; sink->timeline[expert].cpu[2] = ggml_time_us(); }
                sink->compute_phase(false);
                if (sink->timeline_active) { sink->timeline[expert].cpu[3] = ggml_time_us(); }
                sink->current.up_gate_us += elapsed_us(phase_start);
                phase_start = sink_clock::now();
                quant_cpu_start = sink->profile_enabled ? sink_thread_cpu_us() : 0.0;
                if (sink->timeline_active) { sink->timeline[expert].cpu[4] = ggml_time_us(); }
                sink->quantize_hidden(sink->activation.data(), sink->quant_hidden.data(), sink->hidden);
                if (sink->timeline_active) { sink->timeline[expert].cpu[5] = ggml_time_us(); }
                sink->current.hidden_quantize_us += elapsed_us(phase_start);
                if (sink->profile_enabled) { sink->current.compute_cpu_us += sink_thread_cpu_us() - quant_cpu_start; }
                down_wait_us = sink->wait_down(expert);
                if (!upload_started && sink->vram.async_upload && borrow_upload) {
                    // wait_down() returned only after a successful down CQE, so
                    // the immutable three-role cache payload can now be borrowed
                    // while CPU down computation reads it too.
                    sink->vram.admit(expert, data, true);
                    upload_started = true;
                }
                phase_start = sink_clock::now();
                if (sink->timeline_active) { sink->timeline[expert].cpu[6] = ggml_time_us(); }
                sink->compute_phase(true);
                if (sink->timeline_active) { sink->timeline[expert].cpu[7] = ggml_time_us(); }
                sink->current.down_reduce_us += elapsed_us(phase_start);
            }
            sink->current.compute_us += elapsed_us(compute_start) - down_wait_us;
            if (!upload_started) { sink->vram.admit(expert, data, borrow_upload); }
            sink->cache.admit(request, sink->current);
            if (request.cache_hit < 0) {
                std::lock_guard<std::mutex> lock(sink->io_mutex);
                slot.state = sink_slot::empty;
            }
            sink->io_cv.notify_one();
        }
        {
            const auto wait_start = sink_clock::now();
            std::unique_lock<std::mutex> lock(sink->io_mutex);
            sink->io_cv.wait(lock, [&] { return sink->io_done; });
            sink->current.wait_io_us += elapsed_us(wait_start);
            if (!sink->io_error.empty()) { throw std::runtime_error(sink->io_error); }
        }
        if (sink->current.read_bytes && sink->current.io_us > 0) {
            const double rate = double(sink->current.read_bytes) / sink->current.io_us;
            sink->decode_read_bytes_per_us = sink->decode_read_bytes_per_us > 0 ? 0.9 * sink->decode_read_bytes_per_us + 0.1 * rate : rate;
        }
        sink->vram.finish(sink->expert_output.data(), size_t(sink->latent), sink->cache, sink->requests);
        const double lanes_us = elapsed_us(sink->plan.start);
        sink->report_miss_cost();
        sink->vram.set_host_demand(sink->cache.pending_bytes(sink->cache.stride));
        if (sink->ready_first) {
            const auto reduce_start = sink_clock::now();
            const double cpu_start = sink->profile_enabled ? sink_thread_cpu_us() : 0;
            // Each stored value is the same rounded multiply as the old path.
            // Preserve every addition, including the initial addition to +0.
            for (size_t expert = 0; expert < sink->requests.size(); ++expert) {
                float * contribution = sink->expert_output.data() + expert * size_t(sink->latent);
                if (activation->post_exp_norm) {
                    kimi_cpu_sink_normalize_expert(contribution, sink->latent,
                        activation->post_exp_norm, activation->post_exp_norm_eps);
                    for (int row = 0; row < sink->latent; ++row) {
                        volatile float scaled = contribution[row] * sink->requests[expert].weight;
                        output[row] += scaled;
                    }
                } else {
                    for (int row = 0; row < sink->latent; ++row) { output[row] += contribution[row]; }
                }
            }
            sink->current.ordered_reduce_us = elapsed_us(reduce_start);
            sink->current.compute_us += sink->current.ordered_reduce_us;
            if (sink->profile_enabled) { sink->current.compute_cpu_us += sink_thread_cpu_us() - cpu_start; }
        }
        if (sink->timeline_active) { sink->timeline_call[5] = ggml_time_us(); }
        sink->current.total_us = elapsed_us(start);
        sink->current.read_completions = sink->ring.completed - completed_before;
        sink->current.pending_reads = sink->ring.pending;
        sink->observe_decode_layer(lanes_us, sink->current.total_us);
        sink->finish_observation();
        if (stats) { *stats = sink->current; }
        sink->print_observation(true);
        return true;
    } catch (const std::exception & e) {
        // Cache slots are kernel read targets now. Stop producing new reads and
        // wait for every published SQE before unmapping, even if the caller
        // throws while the I/O thread is waiting for a compute credit to free.
        sink->cancel_and_wait();
        sink->prefetch_drain();
        sink->cache.release();
        if (sink->timeline_active) { sink->timeline_call[5] = ggml_time_us(); }
        sink->current.total_us = elapsed_us(start);
        sink->current.read_completions = sink->ring.completed - completed_before;
        sink->current.pending_reads = sink->ring.pending;
        sink->finish_observation();
        if (stats) { *stats = sink->current; }
        sink->print_observation(false);
        sink_error(error, error_size, e.what());
        return false;
    }
}

extern "C" bool kimi_cpu_sink_run(
        struct kimi_cpu_sink * sink, const struct ggml_tensor * up, const struct ggml_tensor * gate,
        const struct ggml_tensor * down, const float * input, const int32_t * ids, const float * weights,
        int nslots, float situ_beta, float situ_linear_beta, float * output,
        struct kimi_cpu_sink_stats * stats, char * error, size_t error_size) {
    const kimi_cpu_sink_activation activation = { KIMI_CPU_SINK_ACTIVATION_SITU, situ_beta, situ_linear_beta, 0.0f };
    return kimi_cpu_sink_run_activation(sink, up, gate, down, input, ids, weights, nslots, &activation,
        output, stats, error, error_size);
}

#include "kimi-cpu-cache-executor.inc"

#else

void kimi_cpu_sink_parallel(struct kimi_cpu_sink *, kimi_cpu_sink_parallel_fn, void *) {}
bool kimi_cpu_sink_stream(struct kimi_cpu_sink *, const struct ggml_tensor *, const struct ggml_tensor *,
        const struct ggml_tensor *, const int32_t *, int, kimi_cpu_sink_consumer_fn, void *,
        struct kimi_cpu_sink_stats * stats, char * error, size_t error_size) {
    if (stats) { *stats = {}; }
    sink_error(error, error_size, "CPU prefill requires Linux io_uring and O_DIRECT (no fallback)");
    return false;
}
bool kimi_cpu_sink_stream_hybrid(struct kimi_cpu_sink *, const struct ggml_tensor *, const struct ggml_tensor *,
        const struct ggml_tensor *, const int32_t *, int, struct kimi_cpu_sink_prefill_batch *,
        kimi_cpu_sink_consumer_fn, void *, struct kimi_cpu_sink_stats * stats, char * error, size_t error_size) {
    if (stats) { *stats = {}; }
    sink_error(error, error_size, "CPU prefill requires Linux io_uring and O_DIRECT (no fallback)");
    return false;
}

extern "C" struct kimi_cpu_sink * kimi_cpu_sink_create(
        int, int, kimi_cpu_sink_resolver, void *, char * error, size_t error_size) {
    sink_error(error, error_size, "CPU sink requires Linux io_uring and O_DIRECT (no fallback)");
    return nullptr;
}
extern "C" void kimi_cpu_sink_destroy(struct kimi_cpu_sink *) {}
extern "C" void kimi_cpu_sink_prefetch_hint(struct kimi_cpu_sink *, int32_t, const int32_t *, int) {}
extern "C" void kimi_cpu_sink_cache_reset(struct kimi_cpu_sink *) {}
extern "C" bool kimi_cpu_sink_cache_configure(struct kimi_cpu_sink *, uint64_t,
        const int32_t *, int, int, char * error, size_t error_size) {
    sink_error(error, error_size, "CPU sink cache requires Linux"); return false;
}
extern "C" bool kimi_cpu_sink_cache_begin(struct kimi_cpu_sink *, uint64_t,
        uint64_t, bool, char * error, size_t error_size) {
    sink_error(error, error_size, "CPU sink cache requires Linux"); return false;
}
extern "C" bool kimi_cpu_sink_vram_phase_complete(
        struct kimi_cpu_sink *, bool, char * error, size_t error_size) {
    sink_error(error, error_size, "CPU sink VRAM lifecycle requires Linux"); return false;
}
extern "C" bool kimi_cpu_sink_vram_reserve_accumulator(
        struct kimi_cpu_sink *, int, char * error, size_t error_size) {
    sink_error(error, error_size, "CPU sink VRAM lifecycle requires Linux"); return false;
}
extern "C" bool kimi_cpu_sink_vram_reserve(struct kimi_cpu_sink *, const struct ggml_tensor *,
        const struct ggml_tensor *, const struct ggml_tensor *, int, char * error, size_t error_size) {
    sink_error(error, error_size, "CPU sink VRAM lifecycle requires Linux"); return false;
}
extern "C" void kimi_cpu_sink_cache_get_stats(struct kimi_cpu_sink *, kimi_cpu_sink_cache_stats * stats) {
    if (stats) { *stats = {}; }
}
extern "C" bool kimi_cpu_sink_cache_make_room(struct kimi_cpu_sink *, uint64_t, uint64_t) { return false; }
extern "C" bool kimi_cpu_sink_run(
        struct kimi_cpu_sink *, const struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *,
        const float *, const int32_t *, const float *, int, float, float, float *,
        struct kimi_cpu_sink_stats * stats, char * error, size_t error_size) {
    if (stats) { *stats = {}; }
    sink_error(error, error_size, "CPU sink requires Linux io_uring and O_DIRECT (no fallback)");
    return false;
}
extern "C" bool kimi_cpu_sink_run_activation(
        struct kimi_cpu_sink *, const struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *,
        const float *, const int32_t *, const float *, int, const struct kimi_cpu_sink_activation *, float *,
        struct kimi_cpu_sink_stats * stats, char * error, size_t error_size) {
    if (stats) { *stats = {}; }
    sink_error(error, error_size, "CPU sink requires Linux io_uring and O_DIRECT (no fallback)");
    return false;
}

extern "C" bool kimi_cpu_sink_run_cache_compute(kimi_cpu_sink *,
        const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const float *,
        const int32_t *, const float *, int, const int32_t *, const float *, int, float, bool,
        const kimi_cache_compute_config *, float, float, float *, kimi_cpu_sink_stats * stats,
        kimi_cache_compute_stats * decision, char * error, size_t error_size) {
    if (stats) { *stats = {}; } if (decision) { *decision = {}; }
    sink_error(error, error_size, "cache-compute requires Linux io_uring and O_DIRECT (no fallback)"); return false;
}

extern "C" bool kimi_cpu_sink_set_cache_locations(kimi_cpu_sink *, const kimi_cache_location_api *, void *, char * error, size_t size) {
    sink_error(error,size,"cache locations require Linux"); return false;
}

#endif
