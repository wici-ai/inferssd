// Included inside the common adapter namespace after pack/profile helpers.
// Payloads and CUDA events have the same process lifetime as the static tier.
static constexpr size_t COMMON_TIER_Q3_ROLE_BYTES = 4730880;
// A host copy must fit inside one CUDA registration. Round the pin chunk down
// to whole slots so no expert straddles two adjacent registered ranges.
static constexpr size_t COMMON_TIER_PIN_CHUNK_BYTES =
        (256ULL * 1024 * 1024 / COMMON_TIER_Q3_ROLE_BYTES) * COMMON_TIER_Q3_ROLE_BYTES;

struct common_dynamic_tier_state {
    using lru_type = ggml_moe_common::common_dynamic_tier_lru;
    struct transfer {
        cudaEvent_t event;
        lru_type::lease held;
    };
    lru_type lru;
    std::atomic<bool> active{false};
    void * base = nullptr;
    size_t bytes = 0;
    std::mutex transfers_mu;
    std::vector<transfer> pending;
    std::vector<cudaEvent_t> free_events;
    FILE * trace = nullptr;
    uint64_t sequence = 0;
    uint64_t preloaded = 0;
    std::atomic<uint64_t> skipped_fills{0};
};

static common_dynamic_tier_state g_common_tier;

static bool common_dynamic_tier_key(const expert_pack_entry * entry, size_t bytes, size_t & key) {
    if (!entry || bytes != COMMON_TIER_Q3_ROLE_BYTES || entry->nbytes != bytes ||
            (!std::strstr(entry->tensor, ".ffn_up_exps.") &&
             !std::strstr(entry->tensor, ".ffn_gate_exps.") &&
             !std::strstr(entry->tensor, ".ffn_down_exps."))) {
        return false;
    }
    // Pack entries are immutable and sorted by (tensor, expert, bytes).
    const uintptr_t begin = (uintptr_t) g_expert_pack.entries.data();
    const uintptr_t address = (uintptr_t) entry;
    if (address < begin || (address - begin) % sizeof(expert_pack_entry) != 0) return false;
    key = (address - begin) / sizeof(expert_pack_entry);
    return key < g_expert_pack.entries.size();
}

// transfers_mu must be held. Each reader has its own event, including readers
// of the same slot on different streams; a newer copy cannot hide an old one.
static void common_dynamic_tier_reap() {
    auto & tier = g_common_tier;
    for (size_t i = 0; i < tier.pending.size();) {
        auto & transfer = tier.pending[i];
        const cudaError_t status = cudaEventQuery(transfer.event);
        if (status == cudaErrorNotReady) {
            ++i;
            continue;
        }
        CUDA_CHECK(status);
        GGML_ASSERT(tier.lru.release(transfer.held));
        tier.free_events.push_back(transfer.event);
        tier.pending[i] = tier.pending.back();
        tier.pending.pop_back();
    }
}

static void common_dynamic_tier_fill(const expert_pack_entry * entry, const void * data, size_t bytes) {
    auto & tier = g_common_tier;
    size_t key;
    if (!tier.active.load(std::memory_order_acquire) || !data ||
            !common_dynamic_tier_key(entry, bytes, key)) return;
    {
        std::lock_guard<std::mutex> lock(tier.transfers_mu);
        common_dynamic_tier_reap();
    }
    common_dynamic_tier_state::lru_type::reservation fill;
    if (!tier.lru.reserve(key, fill)) {
        ++tier.skipped_fills;
        return;
    }
    // The caller still owns the completed SSD staging slice. Publication is
    // synchronous: neither a partial read nor an in-flight write is visible.
    std::memcpy((char *) tier.base + fill.slot * COMMON_TIER_Q3_ROLE_BYTES, data, bytes);
    GGML_ASSERT(tier.lru.publish(fill));
}

static void common_dynamic_tier_init(size_t budget) {
    auto & tier = g_common_tier;
    const size_t slots = budget / COMMON_TIER_Q3_ROLE_BYTES;
    if (slots == 0) return;
    bool has_q3 = false;
    for (const auto & entry : g_expert_pack.entries) {
        size_t key;
        if (common_dynamic_tier_key(&entry, entry.nbytes, key)) {
            has_q3 = true;
            break;
        }
    }
    if (!has_q3) {
        std::fprintf(stderr, "[moe_stream_batch] dynamic RAM tier: no Q3 role entries, disabled\n");
        return;
    }
#if !defined(_WIN32)
    const size_t bytes = slots * COMMON_TIER_Q3_ROLE_BYTES;
    void * base = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
        std::fprintf(stderr, "[moe_stream_batch] dynamic RAM tier: mmap failed, disabled\n");
        return;
    }
    // Each chunk and the final remainder contain only whole page-aligned slots.
    const size_t chunk_bytes = COMMON_TIER_PIN_CHUNK_BYTES;
    size_t pinned = 0;
    while (pinned < bytes) {
        const size_t chunk = std::min(chunk_bytes, bytes - pinned);
        const cudaError_t error = cudaHostRegister((char *) base + pinned, chunk, cudaHostRegisterDefault);
        if (error != cudaSuccess) {
            std::fprintf(stderr, "[moe_stream_batch] dynamic RAM tier: pin failed at %zu MiB: %s; disabled\n",
                    pinned / (1024 * 1024), cudaGetErrorString(error));
            cudaGetLastError(); // Optional cache failure must not poison SSD fallback.
            for (size_t offset = 0; offset < pinned; offset += chunk_bytes) {
                CUDA_CHECK(cudaHostUnregister((char *) base + offset));
            }
            ::munmap(base, bytes);
            cudaGetLastError();
            return;
        }
        pinned += chunk;
    }
    GGML_ASSERT(tier.lru.reset(slots));
    tier.base = base;
    tier.bytes = bytes;
    g_expert_pack.ram_tier_base = base;
    g_expert_pack.ram_tier_bytes = bytes;
    g_expert_pack.ram_tier_pinned_bytes = bytes;
    const char * trace_path = std::getenv("GGML_MOE_RAM_TIER_TRACE");
    if (trace_path && trace_path[0]) {
        tier.trace = std::fopen(trace_path, "w");
        if (tier.trace) {
            std::fprintf(tier.trace, "seq,tensor,expert,result\n");
        } else {
            std::fprintf(stderr, "[moe_stream_batch] dynamic RAM tier: cannot open trace %s\n", trace_path);
        }
    }
    tier.active.store(true, std::memory_order_release);

    // Optional static warm seed occupies the SAME slots and budget. Avoid
    // expert_pack_read_entry here: it would recursively enter ram_tier_init.
    const char * profile_path = std::getenv("GGML_MOE_RAM_TIER_PROFILE");
    std::vector<profile_entry> profile;
    if (profile_path && profile_path[0] && load_profile_file(profile_path, profile, "dynamic RAM seed")) {
        const char * skip_env = std::getenv("GGML_MOE_RAM_TIER_SKIP");
        const size_t skip = skip_env ? (size_t) std::max(0L, std::atol(skip_env)) : 0;
        std::vector<const expert_pack_entry *> seeds;
        for (size_t i = skip; i < profile.size() && seeds.size() < slots; ++i) {
            const auto & p = profile[i];
            const expert_pack_entry * entry = expert_pack_lookup(p.tensor, p.expert_idx, p.expert_bytes);
            size_t key;
            if (common_dynamic_tier_key(entry, p.expert_bytes, key)) seeds.push_back(entry);
        }
        // Reverse the hottest-first profile so its hottest seed ends at MRU.
        for (auto it = seeds.rbegin(); it != seeds.rend(); ++it) {
            const expert_pack_entry * entry = *it;
            size_t key;
            common_dynamic_tier_key(entry, entry->nbytes, key);
            common_dynamic_tier_state::lru_type::reservation fill;
            if (!tier.lru.reserve(key, fill)) continue;
            void * dst = (char *) base + fill.slot * COMMON_TIER_Q3_ROLE_BYTES;
            bool ok = expert_pack_direct_read_entry_to_host(entry, dst, entry->nbytes);
            if (!ok) {
                const expert_pack_source * source = expert_pack_source_for_entry(entry);
                std::lock_guard<std::mutex> lock(g_expert_pack.mu);
                ok = source && source->file && ::fseeko(source->file, (off_t) entry->offset, SEEK_SET) == 0 &&
                    expert_pack_read_exact(source->file, dst, entry->nbytes);
            }
            if (ok) {
                GGML_ASSERT(tier.lru.publish(fill));
                ++tier.preloaded;
            } else {
                GGML_ASSERT(tier.lru.cancel(fill));
            }
        }
    }
    std::fprintf(stderr, "[moe_stream_batch] dynamic RAM tier: slots=%zu slot_bytes=%zu pinned=%.2f MiB "
            "pin_chunk=%.2f MiB preloaded=%llu roles=up,gate,down policy=LRU\n",
            slots, COMMON_TIER_Q3_ROLE_BYTES, bytes / (1024.0 * 1024.0),
            chunk_bytes / (1024.0 * 1024.0),
            (unsigned long long) tier.preloaded);
#else
    std::fprintf(stderr, "[moe_stream_batch] dynamic RAM tier: unsupported platform, disabled\n");
#endif
}

static bool common_dynamic_tier_copy_h2d(
        const expert_pack_entry * entry, void * dst, size_t bytes, cudaStream_t stream) {
    auto & tier = g_common_tier;
    size_t key;
    if (!tier.active.load(std::memory_order_acquire) || !common_dynamic_tier_key(entry, bytes, key)) return false;
    std::lock_guard<std::mutex> lock(tier.transfers_mu);
    common_dynamic_tier_reap();
    common_dynamic_tier_state::lru_type::lease held;
    const bool hit = tier.lru.acquire(key, held);
    ++g_expert_pack.ram_tier_total;
    if (tier.trace) {
        std::fprintf(tier.trace, "%llu,%s,%d,%s\n", (unsigned long long) ++tier.sequence,
                entry->tensor, entry->expert_idx, hit ? "hit" : "miss");
    }
    if (!hit) return false;
    cudaEvent_t event;
    if (tier.free_events.empty()) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    } else {
        event = tier.free_events.back();
        tier.free_events.pop_back();
    }
    const void * src = (const char *) tier.base + held.slot * COMMON_TIER_Q3_ROLE_BYTES;
    GGML_ASSERT((held.slot * COMMON_TIER_Q3_ROLE_BYTES) % COMMON_TIER_PIN_CHUNK_BYTES + bytes <=
            COMMON_TIER_PIN_CHUNK_BYTES);
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(event, stream));
    tier.pending.push_back({event, held});
    ++g_expert_pack.ram_tier_hits;
    ++g_expert_pack.ram_tier_h2d_enqueues;
    g_expert_pack.ram_tier_h2d_bytes.fetch_add(bytes);
    return true;
}

static void common_dynamic_tier_report() {
    auto & tier = g_common_tier;
    if (!tier.active.load(std::memory_order_acquire)) return;
    const auto stats = tier.lru.get_stats();
    const uint64_t hits = g_expert_pack.ram_tier_hits.load();
    const uint64_t total = g_expert_pack.ram_tier_total.load();
    std::fprintf(stderr, "[moe_stream_batch] RAM tier: hits=%llu total=%llu hit_rate=%.1f%% "
            "resident=%.2f MiB pinned=%.2f MiB h2d_bytes=%llu admissions=%llu evictions=%llu "
            "preloaded=%llu skipped_fills=%llu dynamic=1 policy=LRU\n",
            (unsigned long long) hits, (unsigned long long) total, total ? 100.0 * hits / total : 0.0,
            stats.ready * COMMON_TIER_Q3_ROLE_BYTES / (1024.0 * 1024.0), tier.bytes / (1024.0 * 1024.0),
            (unsigned long long) g_expert_pack.ram_tier_h2d_bytes.load(),
            (unsigned long long) stats.admissions, (unsigned long long) stats.evictions,
            (unsigned long long) tier.preloaded, (unsigned long long) tier.skipped_fills.load());
    std::lock_guard<std::mutex> lock(tier.transfers_mu);
    if (tier.trace) std::fflush(tier.trace);
}

static void common_dynamic_tier_phase_report(const char * label, uint64_t sequence) {
    auto & tier = g_common_tier;
    if (!tier.active.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lock(tier.transfers_mu);
    const auto stats = tier.lru.get_stats();
    std::fprintf(stderr, "[moe_stream_batch_phase] label=%s seq=%llu common_dynamic_tier=1 "
            "ram_hits=%llu ram_total=%llu ram_h2d_bytes=%llu ram_admissions=%llu ram_evictions=%llu\n",
            label, (unsigned long long) sequence,
            (unsigned long long) g_expert_pack.ram_tier_hits.load(),
            (unsigned long long) g_expert_pack.ram_tier_total.load(),
            (unsigned long long) g_expert_pack.ram_tier_h2d_bytes.load(),
            (unsigned long long) stats.admissions, (unsigned long long) stats.evictions);
    if (tier.trace) std::fflush(tier.trace);
}
