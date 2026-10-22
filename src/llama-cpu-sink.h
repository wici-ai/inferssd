#pragma once

#include "kimi-cpu-sink.h"
#include "llama-cpu-prefill.h"

#include <cerrno>
#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#endif

// Only metadata is retained here. Expert tensors are deliberately absent from
// the custom op's graph sources, so the scheduler cannot copy their payloads.
struct llama_kimi_cpu_sink_runtime;

struct llama_kimi_cpu_sink_node {
    ggml_tensor up;
    ggml_tensor gate;
    ggml_tensor down;
    ggml_backend_reg_t cpu_reg;
    llama_kimi_cpu_sink_runtime * runtime;
    float situ_beta;
    float situ_linear_beta;
    // SITU nodes keep the K3 entries (run, cache compute, grouped prefill).
    // Other activations are decode-only through kimi_cpu_sink_run_activation.
    kimi_cpu_sink_activation activation = { KIMI_CPU_SINK_ACTIVATION_SITU, 0.0f, 0.0f, 0.0f };
    int layer;
    bool post_exp_norm = false;
    float post_exp_norm_eps = 0.0f;
    bool prefill = false;
    bool cache_compute = false;
    bool guarded = false;
    float routed_scale = 1.0f;
};

struct llama_kimi_cpu_sink_runtime {
    // Resolve the logical native range; the callback below aligns its physical I/O.
    // resolve_raw_direct_shape(name, expert, logical bytes, n_experts, fd, logical offset, logical bytes)
    using resolve_fn = bool (*)(const char *, int, size_t, int64_t, int *, uint64_t *, size_t *);

    std::mutex mutex;
    kimi_cpu_sink * sink = nullptr;
    llama_kimi_cpu_prefill_runtime prefill;
    decltype(&kimi_cpu_sink_run) run = nullptr;
    decltype(&kimi_cpu_sink_run_activation) run_activation = nullptr;
    decltype(&kimi_cpu_sink_run_cache_compute) run_cache_compute = nullptr;
    bool cache_compute_enabled = false;
    kimi_cache_compute_config cache_compute_config = {};
    decltype(&kimi_cpu_sink_destroy) destroy = nullptr;
    decltype(&kimi_cpu_sink_cache_configure) cache_configure = nullptr;
    decltype(&kimi_cpu_sink_cache_begin) cache_begin = nullptr;
    decltype(&kimi_cpu_sink_vram_phase_complete) vram_phase_complete = nullptr;
    decltype(&kimi_cpu_sink_cache_get_stats) cache_get_stats = nullptr;
    decltype(&kimi_cpu_sink_cache_make_room) cache_make_room = nullptr; // optional
    decltype(&kimi_cpu_sink_cache_prefill_room) cache_prefill_room = nullptr; // optional
    resolve_fn resolve = nullptr;
    ggml_backend_reg_t cpu_reg = nullptr;
    const llama_kimi_cpu_sink_node * active_node = nullptr;
    uint64_t calls = 0;
    uint64_t read_requests = 0;
    uint64_t read_bytes = 0;
    uint64_t read_errors = 0;
    uint64_t fallback_count = 0;
    double total_us = 0.0;
    const uint64_t epoch;
    uint64_t cache_budget = 0;
    uint64_t vram_budget = 0;
    int cache_max_per_layer = 8;
    std::vector<int32_t> layer_priority;
    // Expert tensor shapes of every layer, so that the sink's next-layer prefetch can resolve the next layer's
    // experts while the current layer's node is active: name -> (bytes per expert, experts).
    std::map<std::string, std::pair<size_t, int64_t>> expert_shapes;
    decltype(&kimi_cpu_sink_prefetch_hint) prefetch_hint = nullptr; // optional
    bool sequence_known = false;
    llama_seq_id sequence = 0;
    bool cache_decode_active = false;
    // A prefill phase is open from a prefill begin() until phase_complete(true) or a decode begin(); the graph
    // reserves inside it repeat the request-boundary transition and are skipped.
    bool prefill_open = false;
    bool cache_keep_across_requests = false;
    // The file pages the weight rows of the last prefill graph reserved gather (prefill_room): the load step leaves
    // them uncommitted for the first prefill.
    uint64_t prefill_pages = 0;

    static uint64_t unsigned_env(const char * name, uint64_t fallback, uint64_t maximum) {
        const char * value = std::getenv(name);
        if (!value) { return fallback; }
        uint64_t parsed = 0;
        if (!*value) { throw std::runtime_error(std::string(name) + " requires decimal digits"); }
        for (const char * p = value; *p; ++p) {
            if (*p < '0' || *p > '9' || parsed > maximum / 10 ||
                    (parsed == maximum / 10 && uint64_t(*p - '0') > maximum % 10)) {
                throw std::runtime_error(std::string(name) + " is not a valid unsigned decimal value in range");
            }
            parsed = parsed * 10 + uint64_t(*p - '0');
        }
        return parsed;
    }

    // Kimi-K3 layout: frozen K3 cache priority table and 896 routed experts.
    llama_kimi_cpu_sink_runtime(int n_layers, uint64_t context_epoch)
        : llama_kimi_cpu_sink_runtime(n_layers, context_epoch, true, 896, {}) {}

    // n_expert is the routed-expert tensor ne[2] and bounds CACHE_MAX_PER_LAYER.
    // Other layouts default to routed_layers (model order) as cache priority and
    // refuse the K3-only cache-compute policies. The VRAM tier needs its
    // explicit-format extension for them (checked by the executor).
    llama_kimi_cpu_sink_runtime(int n_layers, uint64_t context_epoch, bool kimi_k3_layout,
            int64_t n_expert, const std::vector<int32_t> & routed_layers) : epoch(context_epoch) {
        if (n_expert <= 0) {
            throw std::runtime_error("CPU sink requires a positive routed expert count");
        }
        const char * cache_compute = std::getenv("GGML_KIMI_CPU_SINK_CACHE_COMPUTE");
        cache_compute_enabled = cache_compute && std::strcmp(cache_compute, "off") != 0;
        if (cache_compute_enabled && std::strcmp(cache_compute, "bonus") &&
                std::strcmp(cache_compute, "hybrid") && std::strcmp(cache_compute, "shrink")) {
            throw std::runtime_error("GGML_KIMI_CPU_SINK_CACHE_COMPUTE must be off, bonus, hybrid or shrink");
        }
        const char * locations=std::getenv("GGML_KIMI_CPU_SINK_VRAM_CACHE_LOCATIONS");
        if(locations && std::strcmp(locations,"0") && std::strcmp(locations,"1")) {
            throw std::runtime_error("GGML_KIMI_CPU_SINK_VRAM_CACHE_LOCATIONS must be 0 or 1");
        }
        // The provider may execute the unchanged S set through the shared
        // consumer. Optional shrink/bonus policy stays off unless requested.
        cache_compute_enabled=cache_compute_enabled || (locations&&!std::strcmp(locations,"1"));
        cache_keep_across_requests = unsigned_env("GGML_KIMI_CPU_SINK_CACHE_KEEP_ACROSS_REQUESTS", 0, 1) != 0;
        // Unset: the sink sizes the cache from this machine at the first decode token (UINT64_MAX).
        cache_budget = unsigned_env("GGML_KIMI_CPU_SINK_CACHE_BYTES", std::numeric_limits<uint64_t>::max(),
            std::numeric_limits<size_t>::max());
        // auto: the tier sizes itself (the sink parses the value); any nonzero value turns it on.
        const char * vram_bytes = std::getenv("GGML_KIMI_CPU_SINK_VRAM_BYTES");
        vram_budget = vram_bytes && !std::strcmp(vram_bytes, "auto") ? std::numeric_limits<uint64_t>::max() :
            unsigned_env("GGML_KIMI_CPU_SINK_VRAM_BYTES", 0, 64ull << 30);
        if (!kimi_k3_layout && cache_compute_enabled) {
            throw std::runtime_error("GGML_KIMI_CPU_SINK_CACHE_COMPUTE and GGML_KIMI_CPU_SINK_VRAM_CACHE_LOCATIONS "
                "require Kimi-K3 SiTU experts; use off and 0");
        }
        // Bounded by the routed expert count and the executor's per-layer ABI maximum.
        const uint64_t max_per_layer = uint64_t(std::min<int64_t>(n_expert, 896));
        // An automatic cache is bounded by memory alone; a fixed one keeps the old default of eight.
        const bool automatic = cache_budget == std::numeric_limits<uint64_t>::max();
        cache_max_per_layer = static_cast<int>(unsigned_env("GGML_KIMI_CPU_SINK_CACHE_MAX_PER_LAYER",
            automatic ? max_per_layer : std::min<uint64_t>(8, max_per_layer), max_per_layer));
        if (!cache_max_per_layer) {
            throw std::runtime_error("GGML_KIMI_CPU_SINK_CACHE_MAX_PER_LAYER must be in [1, " +
                std::to_string(max_per_layer) + "]");
        }
        // Frozen ordering: true top16 decode trace, first 127 of 255 tokens.
        const int32_t defaults[] = {
            57,46,56,90,12,86,73,53,87,58,43,55,45,59,81,88,42,84,47,49,76,44,85,54,
            91,89,52,74,79,77,78,41,9,38,50,72,10,83,39,37,75,30,19,31,51,36,4,48,
            29,40,33,3,82,2,60,80,18,23,11,34,27,32,26,22,7,17,35,25,20,16,24,67,
            14,15,13,62,28,69,64,92,66,21,8,61,71,63,70,5,65,68,1,6,
        };
        const char * priority = std::getenv("GGML_KIMI_CPU_SINK_CACHE_LAYER_PRIORITY");
        if (priority) {
            const char * p = priority;
            do {
                char * end = nullptr;
                errno = 0;
                const long value = std::strtol(p, &end, 10);
                if (errno || end == p || value < 0 || value >= n_layers ||
                        (*end && *end != ',') ||
                        std::find(layer_priority.begin(), layer_priority.end(), value) != layer_priority.end()) {
                    throw std::runtime_error("GGML_KIMI_CPU_SINK_CACHE_LAYER_PRIORITY requires unique comma-separated model layer IDs");
                }
                layer_priority.push_back(static_cast<int32_t>(value));
                if (!*end) { break; }
                p = end + 1;
            } while (true);
        } else if (kimi_k3_layout) {
            for (int32_t layer : defaults) { if (layer < n_layers) { layer_priority.push_back(layer); } }
        } else {
            for (int32_t layer : routed_layers) {
                if (layer >= 0 && layer < n_layers &&
                        std::find(layer_priority.begin(), layer_priority.end(), layer) == layer_priority.end()) {
                    layer_priority.push_back(layer);
                }
            }
        }
        if (cache_budget && layer_priority.empty()) {
            throw std::runtime_error("CPU sink cache requires at least one eligible model layer");
        }
    }

    ~llama_kimi_cpu_sink_runtime() {
        prefill.reset();
        if (sink) {
            std::fprintf(stderr,
                "[kimi_cpu_sink_total] calls=%" PRIu64 " read_requests=%" PRIu64
                " read_bytes=%" PRIu64 " total_us=%.0f read_errors=%" PRIu64
                " fallback_count=%" PRIu64 "\n",
                calls, read_requests, read_bytes, total_us, read_errors, fallback_count);
            print_cache_stats("before_destroy", cache_decode_active, 0);
            destroy(sink);
        }
    }

    static bool resolve_source(void * userdata, const char * tensor_name, int32_t expert,
            kimi_cpu_sink_source * source) {
        auto & runtime = *static_cast<llama_kimi_cpu_sink_runtime *>(userdata);
        if (!runtime.active_node || !runtime.resolve || !source) {
            return false;
        }
        const auto & node = *runtime.active_node;
        auto range = [&](size_t logical_nbytes, int64_t experts) {
            *source = {};
            if (!runtime.resolve(tensor_name, expert, logical_nbytes, experts,
                    &source->fd_direct, &source->offset, &source->nbytes)) { return false; }
            source->data_offset = source->offset % 4096;
            source->payload_nbytes = logical_nbytes;
            if (logical_nbytes > SIZE_MAX - source->data_offset - 4095) { return false; }
            source->offset -= source->data_offset;
            source->nbytes = (source->data_offset + logical_nbytes + 4095) / 4096 * 4096;
            return true;
        };
        for (const ggml_tensor * tensor : { &node.up, &node.gate, &node.down }) {
            if (std::strcmp(tensor_name, tensor->name) == 0) {
                return range(tensor->nb[2], tensor->ne[2]);
            }
        }
        const auto shape = runtime.expert_shapes.find(tensor_name);
        if (shape != runtime.expert_shapes.end()) {
            return range(shape->second.first, shape->second.second);
        }
        return false;
    }

    void initialize(ggml_backend_reg_t registry) {
        if (sink) {
            GGML_ASSERT(registry == cpu_reg);
            return;
        }
        cpu_reg = registry;
        auto create = reinterpret_cast<decltype(&kimi_cpu_sink_create)>(
            ggml_backend_reg_get_proc_address(registry, "kimi_cpu_sink_create"));
        run = reinterpret_cast<decltype(run)>(
            ggml_backend_reg_get_proc_address(registry, "kimi_cpu_sink_run"));
        destroy = reinterpret_cast<decltype(destroy)>(
            ggml_backend_reg_get_proc_address(registry, "kimi_cpu_sink_destroy"));
        cache_configure = reinterpret_cast<decltype(cache_configure)>(
            ggml_backend_reg_get_proc_address(registry, "kimi_cpu_sink_cache_configure"));
        cache_begin = reinterpret_cast<decltype(cache_begin)>(
            ggml_backend_reg_get_proc_address(registry, "kimi_cpu_sink_cache_begin"));
        vram_phase_complete = reinterpret_cast<decltype(vram_phase_complete)>(
            ggml_backend_reg_get_proc_address(registry, "kimi_cpu_sink_vram_phase_complete"));
        cache_get_stats = reinterpret_cast<decltype(cache_get_stats)>(
            ggml_backend_reg_get_proc_address(registry, "kimi_cpu_sink_cache_get_stats"));
#if defined(__linux__)
        const auto source_size = reinterpret_cast<size_t (*)(void)>(
            dlsym(RTLD_DEFAULT, "kimi_cpu_sink_source_size"));
        if (!source_size || source_size() != sizeof(kimi_cpu_sink_source)) {
            GGML_ABORT("CPU sink native-range ABI mismatch; rebuild the CPU backend and model together");
        }
        resolve = reinterpret_cast<resolve_fn>(
            dlsym(RTLD_DEFAULT, "ggml_cuda_moe_expert_pack_resolve_raw_direct_shape"));
        // Exported by the same CPU backend library as the registry entries
        // above; looked up like the resolver, leaving the proc-address table as is.
        run_activation = reinterpret_cast<decltype(run_activation)>(
            dlsym(RTLD_DEFAULT, "kimi_cpu_sink_run_activation"));
        cache_make_room = reinterpret_cast<decltype(cache_make_room)>(
            dlsym(RTLD_DEFAULT, "kimi_cpu_sink_cache_make_room"));
        cache_prefill_room = reinterpret_cast<decltype(cache_prefill_room)>(
            dlsym(RTLD_DEFAULT, "kimi_cpu_sink_cache_prefill_room"));
        prefetch_hint = reinterpret_cast<decltype(prefetch_hint)>(
            dlsym(RTLD_DEFAULT, "kimi_cpu_sink_prefetch_hint"));
#endif
        if (!create || !run || !run_activation || !destroy || !resolve || !cache_configure || !cache_begin ||
                !vram_phase_complete || !cache_get_stats) {
            GGML_ABORT("CPU sink requires the CPU executor and direct expert-pack resolver; no fallback");
        }

        int threads = 16;
        if (const char * value = std::getenv("GGML_KIMI_CPU_SINK_THREADS")) {
            char * end = nullptr;
            errno = 0;
            const long parsed = std::strtol(value, &end, 10);
            if (errno || end == value || *end != '\0' || parsed < 2 || parsed > 256) {
                GGML_ABORT("GGML_KIMI_CPU_SINK_THREADS must be an integer in [2, 256]");
            }
            threads = static_cast<int>(parsed);
        }
        int ring_experts = 8;
        if (const char * value = std::getenv("GGML_KIMI_CPU_SINK_RING_EXPERTS")) {
            char * end = nullptr;
            errno = 0;
            const long parsed = std::strtol(value, &end, 10);
            if (errno || end == value || *end != '\0' || parsed < 1 || parsed > 32) {
                GGML_ABORT("GGML_KIMI_CPU_SINK_RING_EXPERTS must be an integer in [1, 32]");
            }
            ring_experts = static_cast<int>(parsed);
        }
        char error[512] = {};
        if (cache_compute_enabled) {
            run_cache_compute = reinterpret_cast<decltype(run_cache_compute)>(
                ggml_backend_reg_get_proc_address(registry, "kimi_cpu_sink_run_cache_compute"));
            auto config = reinterpret_cast<decltype(&kimi_cache_compute_config_from_env)>(
                ggml_backend_reg_get_proc_address(registry, "kimi_cache_compute_config_from_env"));
            if (!run_cache_compute || !config || !config(&cache_compute_config, error, sizeof(error))) {
                GGML_ABORT("cache-compute configuration failed: %s; no fallback", error);
            }
            std::fprintf(stderr, "[kimi_cache_compute_config] version=%u mode=%d bonus_max=%d s_fraction=%.6g min_s=%d max_s=%d deadline_us=%llu protect_mass=%.6g protect_min=%d apply_guard=%d trace=%d\n",
                cache_compute_config.version, cache_compute_config.mode, cache_compute_config.bonus_max,
                cache_compute_config.s_fraction, cache_compute_config.min_s, cache_compute_config.max_s,
                (unsigned long long) cache_compute_config.deadline_us, cache_compute_config.protect_mass,
                cache_compute_config.protect_min, cache_compute_config.apply_guard, cache_compute_config.trace);
        }
        sink = create(threads, ring_experts, resolve_source, this, error, sizeof(error));
        if (!sink) {
            GGML_ABORT("CPU sink initialization failed: %s; no fallback", error);
        }
        if (!cache_configure(sink, cache_budget, layer_priority.data(),
                static_cast<int>(layer_priority.size()), cache_max_per_layer, error, sizeof(error))) {
            GGML_ABORT("CPU sink cache configuration failed: %s; no fallback", error);
        }
        std::fprintf(stderr,
            "[kimi_cpu_sink] enabled threads=%d ring_experts=%d n_tokens=1 expert_h2d_bytes=0"
            " cache_budget=%" PRIu64 " cache_max_experts_per_layer=%d epoch=%" PRIu64
            " cache_admission=direct_read fixed_scope=ring_only\n",
            threads, ring_experts, cache_budget, cache_max_per_layer, epoch);
    }

    void print_cache_stats(const char * reason, bool decode, uint64_t token) {
        if (!sink || !cache_budget) { return; }
        kimi_cpu_sink_cache_stats stats = {};
        cache_get_stats(sink, &stats);
        std::fprintf(stderr,
            "[kimi_cpu_sink_cache] reason=%s epoch=%" PRIu64 " token=%" PRIu64 " decode=%d"
            " budget_bytes=%" PRIu64 " capacity_bytes=%" PRIu64 " mapped_bytes=%" PRIu64
            " committed_bytes=%" PRIu64 " enabled_bytes=%" PRIu64 " valid_bytes=%" PRIu64 " peak_bytes=%" PRIu64
            " hits=%" PRIu64 " misses=%" PRIu64 " admitted_bytes=%" PRIu64 " evictions=%" PRIu64
            " admission_skips=%" PRIu64 " clears=%" PRIu64 " pressure_stops=%" PRIu64 " copy_us=%.0f"
            " page_table_estimate_bytes=%" PRIu64 " last_guard_extra_bytes=%" PRIu64
            " peak_guard_extra_bytes=%" PRIu64
            " shrink_ops=%" PRIu64 " phase_grows=%" PRIu64
            " released_bytes=%" PRIu64 " refaulted_bytes=%" PRIu64
            " guard_reclaims=%" PRIu64 " shrink_us=%" PRIu64
            " prefill_hits=%" PRIu64 " prefill_misses=%" PRIu64
            " prefill_hit_bytes=%" PRIu64 " prefill_leases=%" PRIu64 "\n",
            reason, epoch, token, decode ? 1 : 0, stats.budget_bytes, stats.capacity_bytes,
            stats.mapped_bytes, stats.committed_bytes, stats.enabled_bytes, stats.valid_bytes, stats.peak_bytes,
            stats.hits, stats.misses, stats.admitted_bytes, stats.evictions,
            stats.admission_skips, stats.clears, stats.pressure_stops, stats.copy_us,
            stats.page_table_bytes, stats.last_guard_extra_bytes, stats.peak_guard_extra_bytes,
            stats.shrink_ops, stats.phase_grows, stats.released_bytes,
            stats.refaulted_bytes, stats.guard_reclaims, stats.shrink_us,
            stats.prefill_hits, stats.prefill_misses, stats.prefill_hit_bytes, stats.prefill_leases);
    }

    void begin(ggml_backend_reg_t registry, uint64_t token, bool decode, llama_seq_id seq = 0) {
        std::lock_guard<std::mutex> lock(mutex);
        const char * trace = std::getenv("GGML_KIMI_CPU_SINK_TRACE");
        const char * profile = std::getenv("GGML_KIMI_CPU_SINK_PROFILE");
        const bool observe = (trace && !std::strcmp(trace, "1")) || (profile && !std::strcmp(profile, "1"));
        if (!cache_budget && !vram_budget && !observe) { return; }
        if (decode) { initialize(registry); }
        if (!sink || (!decode && prefill_open)) { return; }
        char error[512] = {};
        if (decode && sequence_known && sequence != seq) {
            if (!cache_begin(sink, epoch, token, false, error, sizeof(error))) {
                GGML_ABORT("CPU sink cache sequence reset failed: %s; no fallback", error);
            }
        }
        if (!cache_begin(sink, epoch, token, decode, error, sizeof(error))) {
            GGML_ABORT("CPU sink cache phase/token transition failed: %s; no fallback", error);
        }
        sequence_known = decode;
        sequence = seq;
        cache_decode_active = decode;
        prefill_open = !decode;
        if (!decode) {
            kimi_cpu_sink_cache_stats stats = {};
            cache_get_stats(sink, &stats);
            if (!cache_keep_across_requests && (stats.mapped_bytes || stats.committed_bytes || stats.valid_bytes)) {
                GGML_ABORT("CPU sink cache retained payload at prefill/unsupported batch boundary");
            }
            print_cache_stats(stats.mapped_bytes ? "retain" : "release", false, token);
        } else {
            const char * trace = std::getenv("GGML_KIMI_CPU_SINK_TRACE");
            if (trace && trace[0] && trace[0] != '0') { print_cache_stats("token_begin", true, token); }
        }
    }

    // At load, after the context reserved its worst-case graph: the sink, and on unified memory the GPU tier's
    // storage for these routed experts, so that no request waits for its page population (the tier decides; a
    // discrete GPU keeps its arena after the first prefill). Off unified memory the RAM cache takes its pages here
    // instead, with a GPU tier or without one, when it can give room back to the first prefill's ledger requests
    // (CACHE_PARTIAL_RECLAIM or CACHE_ELASTIC; the sink decides). The sink exists from here on, so the first request's
    // prefill goes through cache_begin and make_room as every later one does; a cache that cannot give room back is
    // still sized at the first decode token. A failure leaves both to the first decode.
    void reserve_tier(ggml_backend_reg_t registry, const ggml_tensor * up, const ggml_tensor * gate,
            const ggml_tensor * down, int experts_used) {
        std::lock_guard<std::mutex> lock(mutex);
        if ((!vram_budget && !cache_budget) || !registry) { return; }
        initialize(registry);
#if defined(__linux__)
        auto reserve = reinterpret_cast<decltype(&kimi_cpu_sink_vram_reserve)>(
            dlsym(RTLD_DEFAULT, "kimi_cpu_sink_vram_reserve"));
        // the load-time graph's pages, held from here to the first decode token: the load step leaves their room
        if (prefill_pages && cache_prefill_room) { cache_prefill_room(sink, prefill_pages, 0, 0); }
        char error[512] = {};
        if (reserve && !reserve(sink, up, gate, down, experts_used, error, sizeof(error))) {
            std::fprintf(stderr, "[kimi_vram_elastic] event=reserve status=fail error=%s\n", error);
        }
#endif
    }

    // Room for `host_bytes` of host memory and `device_bytes` of GPU memory the caller is about to allocate, from
    // the CPU backend's host-memory ledger, which counts device memory only where the GPU shares the DRAM: the
    // expert caches give back what it needs (on unified memory the GPU tier first). Before the first graph creates
    // the sink the ledger alone decides.
    bool make_room(uint64_t host_bytes, uint64_t device_bytes = 0) {
        std::lock_guard<std::mutex> lock(mutex);
#if defined(__linux__)
        if (!cache_make_room) {
            cache_make_room = reinterpret_cast<decltype(cache_make_room)>(dlsym(RTLD_DEFAULT, "kimi_cpu_sink_cache_make_room"));
        }
#endif
        return !cache_make_room || cache_make_room(sink, host_bytes, device_bytes);
    }

    // Room for a prefill graph about to be allocated (kimi_cpu_sink_cache_prefill_room): its compute buffers and the
    // file pages of the weight rows it gathers, which the ledger holds until the first decode token.
    bool prefill_room(uint64_t pages, uint64_t host_bytes, uint64_t device_bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        prefill_pages = pages;
#if defined(__linux__)
        if (!cache_prefill_room) {
            cache_prefill_room = reinterpret_cast<decltype(cache_prefill_room)>(
                dlsym(RTLD_DEFAULT, "kimi_cpu_sink_cache_prefill_room"));
        }
#endif
        return !cache_prefill_room || cache_prefill_room(sink, pages, host_bytes, device_bytes);
    }

    bool phase_complete(bool decode) {
        std::lock_guard<std::mutex> lock(mutex);
        if (decode) { prefill_open = false; }
        if (!sink) { return true; }
        char error[512] = {};
        if (!vram_phase_complete(sink, decode, error, sizeof(error))) {
            std::fprintf(stderr, "[kimi_vram_elastic] phase completion failed: %s\n", error);
            return false;
        }
        return true;
    }
};

inline void llama_kimi_cpu_sink_compute(ggml_tensor * dst, int ith, int nth, void * userdata) {
    // The sink owns its expert pool. Other ggml workers wait at the graph's
    // outer barrier; they may compute CPU dense matrices in the same graph.
    GGML_ASSERT(ith >= 0 && ith < nth);
    if (ith != 0) { return; }
    auto & node = *static_cast<llama_kimi_cpu_sink_node *>(userdata);
    GGML_ASSERT(node.runtime);
    auto & runtime = *node.runtime;
    std::lock_guard<std::mutex> lock(runtime.mutex);
    runtime.initialize(node.cpu_reg);
    runtime.active_node = &node;

    const ggml_tensor * input = dst->src[0];
    const ggml_tensor * ids = dst->src[1];
    const ggml_tensor * weights = dst->src[2];
    GGML_ASSERT(input->type == GGML_TYPE_F32 && ggml_is_contiguous(input));
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids));
    GGML_ASSERT(weights->type == GGML_TYPE_F32 && ggml_is_contiguous(weights));
    GGML_ASSERT(ggml_nelements(ids) == ggml_nelements(weights));
    const int ntokens = static_cast<int>(dst->ne[1]);
    GGML_ASSERT(ntokens >= 1);
    GGML_ASSERT(ggml_nelements(input) == node.up.ne[0] * ntokens);
    GGML_ASSERT(ggml_nelements(ids) % ntokens == 0);
    const int nslots = static_cast<int>(ggml_nelements(ids) / ntokens);

    kimi_cpu_sink_stats stats = {};
    kimi_cpu_prefill_stats grouped_stats = {};
    char error[512] = {};
    bool ok;
    kimi_cpu_sink_activation activation = node.activation;
    if (node.post_exp_norm) {
        const ggml_tensor * norm = dst->src[3];
        GGML_ASSERT(norm && norm->type == GGML_TYPE_F32 && ggml_is_contiguous(norm));
        GGML_ASSERT(ggml_nelements(norm) == node.down.ne[1]);
        activation.post_exp_norm = static_cast<const float *>(norm->data);
        activation.post_exp_norm_eps = node.post_exp_norm_eps;
    }
    const bool situ = node.activation.kind == KIMI_CPU_SINK_ACTIVATION_SITU;
    GGML_ASSERT(situ || !node.cache_compute);
    if (node.prefill) {
        runtime.prefill.initialize(node.cpu_reg, runtime.sink);
        ok = runtime.prefill.run_activation(runtime.prefill.context, &node.up, &node.gate, &node.down,
            static_cast<const float *>(input->data), static_cast<const int32_t *>(ids->data),
            static_cast<const float *>(weights->data), nslots, ntokens,
            &activation, static_cast<float *>(dst->data),
            &grouped_stats, error, sizeof(error));
        stats = grouped_stats.stream;
    } else if (node.cache_compute) {
        GGML_ASSERT(ntokens == 1 && runtime.run_cache_compute);
        const ggml_tensor * native_ids = dst->src[3];
        const ggml_tensor * native_weights = dst->src[4];
        GGML_ASSERT(native_ids && native_ids->type == GGML_TYPE_I32 && ggml_is_contiguous(native_ids));
        GGML_ASSERT(native_weights && native_weights->type == GGML_TYPE_F32 && ggml_is_contiguous(native_weights));
        GGML_ASSERT(ggml_nelements(native_ids) == ggml_nelements(native_weights));
        ok = runtime.run_cache_compute(runtime.sink, &node.up, &node.gate, &node.down,
            static_cast<const float *>(input->data), static_cast<const int32_t *>(ids->data),
            static_cast<const float *>(weights->data), nslots,
            static_cast<const int32_t *>(native_ids->data), static_cast<const float *>(native_weights->data),
            int(ggml_nelements(native_ids)), node.routed_scale, node.guarded, &runtime.cache_compute_config,
            node.situ_beta, node.situ_linear_beta, static_cast<float *>(dst->data), &stats, nullptr, error, sizeof(error));
    } else if (situ) {
        GGML_ASSERT(ntokens == 1);
        ok = runtime.run(runtime.sink, &node.up, &node.gate, &node.down,
            static_cast<const float *>(input->data), static_cast<const int32_t *>(ids->data),
            static_cast<const float *>(weights->data), nslots,
            node.situ_beta, node.situ_linear_beta, static_cast<float *>(dst->data),
            &stats, error, sizeof(error));
    } else {
        GGML_ASSERT(ntokens == 1);
        // the next layer's predicted experts, for the sink's prefetch
        if (const ggml_tensor * hint = dst->src[5]; hint && runtime.prefetch_hint) {
            GGML_ASSERT(hint->type == GGML_TYPE_I32 && ggml_is_contiguous(hint));
            runtime.prefetch_hint(runtime.sink, node.layer + 1, static_cast<const int32_t *>(hint->data),
                static_cast<int>(ggml_nelements(hint)));
        }
        ok = runtime.run_activation(runtime.sink, &node.up, &node.gate, &node.down,
            static_cast<const float *>(input->data), static_cast<const int32_t *>(ids->data),
            static_cast<const float *>(weights->data), nslots, &activation,
            static_cast<float *>(dst->data), &stats, error, sizeof(error));
    }
    runtime.active_node = nullptr;
    if (!ok) {
        GGML_ABORT("CPU sink layer=%d failed: %s read_errors=%" PRIu64
            " fallback_count=%" PRIu64 "; output discarded", node.layer, error,
            stats.read_errors, stats.fallback_count);
    }
    runtime.calls++;
    runtime.read_requests += stats.read_requests;
    runtime.read_bytes += stats.read_bytes;
    runtime.read_errors += stats.read_errors;
    runtime.fallback_count += stats.fallback_count;
    runtime.total_us += stats.total_us;
    const char * trace = std::getenv("GGML_KIMI_CPU_SINK_TRACE");
    if (runtime.calls == 1 || (trace && trace[0] && trace[0] != '0')) {
        if (node.prefill) {
            std::fprintf(stderr,
                "[kimi_cpu_prefill_layer] layer=%d n_tokens=%d slots=%d unique_experts=%" PRIu64
                " routed_slots=%" PRIu64 " expert_token_pairs=%" PRIu64 " max_group_tokens=%" PRIu64
                " workspace_bytes=%" PRIu64 " gpu_experts=%" PRIu64 " gpu_routes=%" PRIu64
                " gpu_transient_experts=%" PRIu64 " gpu_transient_routes=%" PRIu64
                " gpu_transient_upload_bytes=%" PRIu64 " gpu_transient_upload_us=%.3f gpu_worker_us=%.3f"
                " gpu_ssd_transient_candidates=%" PRIu64 " gpu_ssd_transient_experts=%" PRIu64
                " gpu_ssd_transient_routes=%" PRIu64 " gpu_ssd_transient_upload_bytes=%" PRIu64
                " gpu_ssd_transient_upload_us=%.3f"
                " gpu_cold_admit_candidates=%" PRIu64 " gpu_cold_admit_experts=%" PRIu64
                " gpu_cold_admit_routes=%" PRIu64 " gpu_cold_admit_upload_bytes=%" PRIu64
                " gpu_cold_admit_upload_us=%.3f"
                " ram_experts=%" PRIu64 " ram_routes=%" PRIu64 " ram_bytes=%" PRIu64
                " ssd_experts=%" PRIu64 " ssd_routes=%" PRIu64 " ssd_bytes=%" PRIu64 "\n",
                node.layer, ntokens, nslots, stats.experts, grouped_stats.routed_slots,
                grouped_stats.expert_token_pairs, grouped_stats.max_group_tokens, grouped_stats.workspace_bytes,
                grouped_stats.gpu_experts, grouped_stats.gpu_routes,
                grouped_stats.gpu_transient_experts, grouped_stats.gpu_transient_routes,
                grouped_stats.gpu_transient_upload_bytes, grouped_stats.gpu_transient_upload_us,
                grouped_stats.gpu_worker_us,
                grouped_stats.gpu_ssd_transient_candidates, grouped_stats.gpu_ssd_transient_experts,
                grouped_stats.gpu_ssd_transient_routes, grouped_stats.gpu_ssd_transient_upload_bytes,
                grouped_stats.gpu_ssd_transient_upload_us,
                grouped_stats.gpu_cold_admit_candidates, grouped_stats.gpu_cold_admit_experts,
                grouped_stats.gpu_cold_admit_routes, grouped_stats.gpu_cold_admit_upload_bytes,
                grouped_stats.gpu_cold_admit_upload_us,
                grouped_stats.ram_experts, grouped_stats.ram_routes, grouped_stats.ram_bytes,
                grouped_stats.ssd_experts, grouped_stats.ssd_routes, grouped_stats.ssd_bytes);
        }
        std::fprintf(stderr,
            "[kimi_cpu_sink_layer] layer=%d calls=%" PRIu64 " experts=%" PRIu64
            " reads=%" PRIu64 " bytes=%" PRIu64 " ring_bytes=%" PRIu64
            " total_us=%.0f io_us=%.0f compute_us=%.0f wait_io_us=%.0f"
            " input_quantize_us=%.0f hidden_quantize_us=%.0f up_gate_us=%.0f down_reduce_us=%.0f"
            " demand_bytes=%" PRIu64 " cache_hits=%" PRIu64 " cache_misses=%" PRIu64
            " cache_admitted=%" PRIu64 " cache_admitted_bytes=%" PRIu64 " cache_copy_us=%.0f"
            " cache_fill_read_requests=%" PRIu64 " cache_fill_read_bytes=%" PRIu64
            " ring_read_requests=%" PRIu64 " ring_read_bytes=%" PRIu64
            " fixed_read_requests=%" PRIu64 " read_completions=%" PRIu64 " pending_reads=%" PRIu64
            " cache_new_slot_bytes=%" PRIu64
            " read_errors=%" PRIu64 " fallback_count=%" PRIu64 "\n",
            node.layer, runtime.calls, stats.experts, stats.read_requests,
            stats.read_bytes, stats.ring_bytes, stats.total_us, stats.io_us,
            stats.compute_us, stats.wait_io_us, stats.input_quantize_us, stats.hidden_quantize_us,
            stats.up_gate_us, stats.down_reduce_us, stats.demand_bytes, stats.cache_hits, stats.cache_misses,
            stats.cache_admitted, stats.cache_admitted_bytes, stats.cache_copy_us,
            stats.cache_fill_read_requests, stats.cache_fill_read_bytes, stats.ring_read_requests, stats.ring_read_bytes,
            stats.fixed_read_requests, stats.read_completions, stats.pending_reads, stats.cache_new_slot_bytes,
            stats.read_errors, stats.fallback_count);
    }
}

inline ggml_tensor llama_kimi_cpu_sink_metadata(const ggml_tensor * tensor) {
    ggml_tensor result = *tensor;
    result.data = nullptr;
    result.buffer = nullptr;
    result.extra = nullptr;
    result.view_src = nullptr;
    std::memset(result.src, 0, sizeof(result.src));
    return result;
}
