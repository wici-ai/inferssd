// Included inside the common adapter namespace, after pack/transport helpers.
// A separate host bank keeps future reads out of demand ring.next reuse.
static bool common_cross_role_enabled() {
    static const bool enabled = expert_pack_env_bool("GGML_MOE_COMMON_CROSS_ROLE_PREFETCH", false);
    return enabled;
}

struct common_cross_role_state {
    std::mutex mu;
    pinned_stage_ring ring;
    ggml_moe_common::cross_role_window window;
    std::vector<int> fds;
    std::vector<int> experts;
    std::string names[2];
    std::thread submitter;
    size_t next_role = 0;
    uint64_t generations = 0;
    uint64_t reads = 0;
    uint64_t read_bytes = 0;
    uint64_t copies = 0;
    uint64_t copy_bytes = 0;
    uint64_t skips = 0;
    bool registered = false;
};

static common_cross_role_state g_common_cross_role;

static bool common_cross_role_expected_tensor(const char *tensor) {
    if (!common_cross_role_enabled() || !tensor) return false;
    auto &s = g_common_cross_role;
    std::lock_guard<std::mutex> lk(s.mu);
    return !s.window.idle() && (s.names[0] == tensor || s.names[1] == tensor);
}

[[noreturn]] static void common_cross_role_fail(const char *reason) {
    ++g_expert_pack.io_stats.fallbacks;
    GGML_ABORT("common cross-role prefetch: %s; refusing duplicate-read fallback", reason);
}

static ggml_moe_common::cross_role_key common_cross_role_key(const expert_pack_entry &entry) {
    return {entry.tensor, entry.source_idx, entry.offset, (size_t)entry.nbytes, entry.expert_idx};
}

static bool common_cross_role_profile_may_load(
        const char *tensor, batch_vram_cache *cache, const int *experts, int count, size_t bytes) {
    if (!g_profile_enabled || !profile_preload_enabled()) return false;
    std::lock_guard<std::mutex> lk(g_profile_mu);
    char lookup[96] = {};
    profile_lookup_name(tensor, lookup, sizeof(lookup));
    const bool use_lookup = !profile_has_tensor_locked(g_profile, tensor);
    const int cache_id = batch_cache_id_for_size(bytes);
    size_t seen = 0;
    const size_t budget = profile_preload_slot_budget(cache);
    // Match the actual bounded scan in preload_profile_entries_for_tensor.
    // A loaded profile alone need not imply that this future role is loaded.
    for (const auto &entry : g_profile) {
        if (entry.expert_bytes != 0 && batch_cache_id_for_size(entry.expert_bytes) != cache_id) continue;
        if (seen++ >= budget) break;
        if (std::strcmp(entry.tensor, tensor) != 0 &&
                (!use_lookup || std::strcmp(entry.tensor, lookup) != 0)) continue;
        if (std::find(experts, experts + count, entry.expert_idx) != experts + count) return true;
    }
    return false;
}

// Caller has joined submitter and owns state.mu. All CQEs belong to this generation.
static void common_cross_role_wait_one() {
    auto &s = g_common_cross_role;
    ggml_moe_common::object_io_completion cqe;
    const auto start = std::chrono::steady_clock::now();
    ++g_expert_pack.io_stats.wait_calls;
    const bool ok = ggml_moe_common::object_io_ring_wait(s.ring, cqe);
    g_expert_pack.io_stats.wait_us.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count());
    if (!ok) common_cross_role_fail("CQE wait failed");
    const size_t slot = cqe.user_data == 0 ? SIZE_MAX : (size_t)cqe.user_data - 1;
    const bool valid = slot < s.window.keys.size();
    const size_t bytes = valid ? s.window.keys[slot].bytes : 0;
    if (ggml_moe_common::object_io_completion_validate(s.ring, cqe, valid, bytes) !=
                ggml_moe_common::object_io_completion_status::complete ||
            !s.window.complete(slot, s.window.generation, (size_t)cqe.result)) {
        common_cross_role_fail("invalid, duplicate, or failed read completion");
    }
    ++g_expert_pack.io_stats.cqes;
    ++g_expert_pack.io_stats.reads;
    g_expert_pack.io_stats.bytes.fetch_add(bytes);
    ++s.reads;
    s.read_bytes += bytes;
}

static void common_cross_role_shutdown() {
    auto &s = g_common_cross_role;
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.submitter.joinable()) s.submitter.join();
    // Joining only waits for submission, not for the kernel's buffer writes.
    while (s.window.reading() > 0) common_cross_role_wait_one();
    for (auto &slot : s.ring.slots) {
        if (!ggml_moe_common::object_stage_slot_synchronize(s.ring, slot)) {
            common_cross_role_fail("shutdown H2D wait failed");
        }
    }
    std::fprintf(stderr,
            "[moe_stream_batch] common cross-role summary: generations=%lu reads=%lu read_bytes=%lu "
            "h2d=%lu h2d_bytes=%lu unused_reads=%lu skips=%lu\n",
            (unsigned long)s.generations, (unsigned long)s.reads, (unsigned long)s.read_bytes,
            (unsigned long)s.copies, (unsigned long)s.copy_bytes,
            (unsigned long)(s.reads - s.copies), (unsigned long)s.skips);
    ggml_moe_common::object_stage_ring_release(s.ring);
}

// No CUDA calls or batch mutex acquisition in this thread. One submit per role
// retains small-batch depth; the current gate's demand worker runs independently.
static void common_cross_role_submit() {
    auto &s = g_common_cross_role;
    const size_t role_jobs = s.experts.size();
    const bool sqpoll = expert_pack_env_bool("GGML_MOE_IO_SQPOLL", false);
    for (size_t role = 0; role < 2; ++role) {
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = role * role_jobs; i < (role + 1) * role_jobs; ++i) {
            const auto &key = s.window.keys[i];
            if (!ggml_moe_common::object_io_ring_prepare_read(
                        s.ring, s.fds[i], s.ring.slots[i].host, key.bytes, key.offset, i + 1)) {
                common_cross_role_fail("prepare future read failed");
            }
        }
        size_t remaining = role_jobs;
        while (remaining > 0) {
            const int submitted = ggml_moe_common::object_io_submit_chunk_result(
                    ggml_moe_common::object_io_ring_submit_count(s.ring), remaining, sqpoll);
            if (submitted <= 0 || (size_t)submitted > remaining) {
                common_cross_role_fail("submit future read failed");
            }
            remaining -= (size_t)submitted;
            ++g_expert_pack.io_stats.submit_calls;
        }
        g_expert_pack.io_stats.submit_us.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start).count());
    }
}

// Called with g_batch_mu held, before launching either demand stage thread.
// All-miss, single-token Q3 is deliberate: speculation must replace a demand
// read, not add traffic or compete with another prefetch/cache admission path.
static bool common_cross_role_plan(
        const char *tensor, ggml_type type, const int *experts, const int32_t *tokens,
        int count, size_t bytes, size_t demand_jobs, bool eligible) {
    if (!common_cross_role_enabled()) return false;
    auto &s = g_common_cross_role;
    std::lock_guard<std::mutex> lk(s.mu);
    const std::string name = tensor ? tensor : "";
    if (!s.window.idle()) {
        if (s.next_role >= 2 || name != s.names[s.next_role] ||
                count != (int)s.experts.size() || demand_jobs != (size_t)count || !eligible) {
            common_cross_role_fail("unexpected role or demand count in active generation");
        }
        for (int i = 0; i < count; ++i) {
            if (experts[i] != s.experts[i] || tokens[i] != 0 ||
                    bytes != s.window.keys[(size_t)i].bytes) {
                common_cross_role_fail("route or payload changed within layer");
            }
        }
        return true;
    }
    const size_t marker = name.find(".ffn_gate_exps.");
    if (marker == std::string::npos) return false;
    auto skip = [&](const char *reason) {
        if (s.skips++ < 8) {
            std::fprintf(stderr, "[moe_stream_batch] common cross-role skipped: tensor=%s reason=%s\n", tensor, reason);
        }
        return false;
    };
    if (!eligible || type != GGML_TYPE_Q3_K || count <= 0 || count > 16 ||
            bytes == 0 || bytes > UINT_MAX || bytes % expert_pack_direct_alignment() != 0 ||
            g_expert_pack.io_backend != 2 || g_expert_pack.lossless_standalone) {
        return skip("requires ordinary single-token Q3 io_uring staging (up to 16 experts)");
    }
    for (int i = 0; i < count; ++i) {
        if (tokens[i] != 0 || std::find(experts, experts + i, experts[i]) != experts + i) {
            return skip("multi-token or duplicate routes");
        }
    }
    if (expert_pack_env_size("GGML_MOE_RAM_TIER_MIB", 0, 0, SIZE_MAX) != 0 ||
            g_expert_pack.ram_tier_base) {
        return skip("RAM tier");
    }
    for (const char *env : {
            "GGML_MOE_COMMON_TIER_DYNAMIC", "GGML_MOE_HOST_PREFETCH", "GGML_MOE_TRACE_PREFETCH",
            "GGML_MOE_PLANNED_HOST_PREFETCH", "GGML_MOE_PREDICTIVE_HOST_PREFETCH",
            "GGML_MOE_PREFETCH_DOWN", "GGML_MOE_GATE_UPDOWN_COSUBMIT", "GGML_MOE_UPDOWN_PAIRED_READ",
            "GGML_MOE_ROUTE_GROUP_UPDOWN_QUEUE",
            "GGML_MOE_CURRENT_DOWN_OVERLAP", "GGML_MOE_IO_ADJACENT_COALESCE"}) {
        if (expert_pack_env_bool(env, false)) return skip(env);
    }
    batch_vram_cache *cache = batch_cache_get(bytes);
    if (!cache) return skip("missing demand cache");
    std::vector<ggml_moe_common::cross_role_key> keys;
    std::vector<int> fds;
    std::string names[2];
    for (size_t role = 0; role < 2; ++role) {
        names[role] = name.substr(0, marker) + (role == 0 ? ".ffn_up_exps." : ".ffn_down_exps.") +
            name.substr(marker + std::strlen(".ffn_gate_exps."));
        if (common_cross_role_profile_may_load(names[role].c_str(), cache, experts, count, bytes)) {
            return skip("future demand may be loaded by profile preload");
        }
        for (int i = 0; i < count; ++i) {
            if (batch_cache_find_slot(cache, batch_key_hash(names[role].c_str(), experts[i])) >= 0) {
                return skip("future role already cached");
            }
            const expert_pack_entry *entry = expert_pack_lookup_impl(names[role].c_str(), experts[i], bytes, false);
            const expert_pack_source *source = entry ? expert_pack_source_for_entry(entry) : nullptr;
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING) && !defined(_WIN32)
            if (!source || source->page_cache || source->fd_direct < 0 || entry->alias_source ||
                    entry->offset % expert_pack_direct_alignment() != 0) {
                return skip("future role lacks aligned direct pack entry");
            }
            fds.push_back(source->fd_direct);
#else
            (void)source;
            return skip("liburing unavailable");
#endif
            keys.push_back(common_cross_role_key(*entry));
        }
    }
    if (s.submitter.joinable()) s.submitter.join();
    const size_t need_slots = keys.size();
    // Allocate only the high-water mark (32 future slots for top-16); keep the
    // configured demand rings intact. No allocation at all while disabled.
    if (s.ring.slots.size() < need_slots || s.ring.slot_sz < bytes) {
        ggml_moe_common::object_stage_ring_release(s.ring);
        if (!ggml_moe_common::object_stage_ring_allocate(s.ring, bytes, need_slots, false)) {
            common_cross_role_fail("pinned host allocation failed");
        }
    }
    const unsigned flags = expert_pack_env_bool("GGML_MOE_IO_SQPOLL", false) ?
        ggml_moe_common::object_io_sqpoll_setup_flag() : 0;
    // The producer publishes both roles before any consumer joins it.
    if (!ggml_moe_common::object_io_ring_ensure(s.ring, s.ring.slots.size(), flags, nullptr) ||
            !s.window.begin(std::move(keys))) {
        common_cross_role_fail("ring or generation initialization failed");
    }
    s.fds = std::move(fds);
    s.experts.assign(experts, experts + count);
    s.names[0] = std::move(names[0]);
    s.names[1] = std::move(names[1]);
    s.next_role = 0;
    if (!s.registered) {
        std::atexit(common_cross_role_shutdown);
        s.registered = true;
        std::fprintf(stderr,
                "[moe_stream_batch] common cross-role active: gate=%s experts=%d future_slots=%zu pinned_bytes=%zu\n",
                tensor, count, s.ring.slots.size(), s.ring.slot_sz * s.ring.slots.size());
    }
    ++s.generations;
    for (int role = 0; role < 2; ++role) {
        expert_pack_record_iouring_batch((size_t)count);
        ++s.ring.iouring_batches;
        s.ring.iouring_jobs += (size_t)count;
        ++s.ring.iouring_batch_hist[expert_pack_iouring_batch_bucket((size_t)count)];
    }
    s.submitter = std::thread(common_cross_role_submit);
    return true;
}

static bool common_cross_role_copy_h2d(
        const expert_pack_entry *entry, void *dst, size_t bytes, cudaStream_t stream) {
    if (!common_cross_role_enabled() || !entry) return false;
    auto &s = g_common_cross_role;
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.window.idle()) return false;
    const size_t slot = s.window.find(common_cross_role_key(*entry));
    if (slot == SIZE_MAX) {
        if (s.names[0] == entry->tensor || s.names[1] == entry->tensor) {
            common_cross_role_fail("future demand key does not match planned read");
        }
        return false;
    }
    if (bytes != s.window.keys[slot].bytes) common_cross_role_fail("future demand size mismatch");
    if (s.submitter.joinable()) s.submitter.join();
    while (s.window.phases[slot] == ggml_moe_common::cross_role_phase::reading) common_cross_role_wait_one();
    if (s.window.phases[slot] != ggml_moe_common::cross_role_phase::ready ||
            cudaMemcpyAsync(dst, s.ring.slots[slot].host, bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
            !ggml_moe_common::object_stage_slot_record(s.ring.slots[slot], stream) ||
            !s.window.copied(slot)) {
        common_cross_role_fail("duplicate demand or future H2D failed");
    }
    ++g_expert_pack.iouring_h2d_enqueues;
    ++s.ring.copies;
    ++s.copies;
    s.copy_bytes += bytes;
    return true;
}

// The caller retains the existing per-role stream synchronization. Release the
// host lease only after the event on its actual copy stream also completes.
static void common_cross_role_finish(const char *tensor) {
    if (!common_cross_role_enabled()) return;
    auto &s = g_common_cross_role;
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.window.idle() || s.next_role >= 2 || s.names[s.next_role] != tensor) return;
    for (size_t i = 0; i < s.window.keys.size(); ++i) {
        if (s.window.keys[i].tensor != tensor) continue;
        if (s.window.phases[i] != ggml_moe_common::cross_role_phase::h2d_pending ||
                !ggml_moe_common::object_stage_slot_synchronize(s.ring, s.ring.slots[i]) ||
                !s.window.release(i)) {
            common_cross_role_fail("unconsumed future read or H2D release failed");
        }
    }
    ++s.next_role;
}
