// Kimi-specific CPU MoE mul_mat_id path.
// This file is included by ggml-cpu.c so it can use that translation unit's
// internal helpers, macros, and static CUDA MoE bridge symbols.

#if defined(__gnu_linux__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((weak)) extern bool llama_mmap_dontneed_expert_host_range(
        const char * tensor_name, const void * addr, size_t size, size_t * advised_out);
#endif

static bool ggml_model_type_is_kimi(void) {
    const char * model_type = getenv("MODEL_TYPE");
    if (!model_type) {
        return false;
    }

    while (*model_type == ' ' || *model_type == '\t') {
        ++model_type;
    }

    const char kimi[] = "kimi";
    for (size_t i = 0; kimi[i] != '\0'; ++i) {
        char c = model_type[i];
        if (c >= 'A' && c <= 'Z') {
            c = (char) (c - 'A' + 'a');
        }
        if (c != kimi[i]) {
            return false;
        }
    }

    const char end = model_type[sizeof(kimi) - 1];
    return end == '\0' || end == ' ' || end == '\t' || end == ',' || end == ':';
}

struct ggml_kimi_phase1_mmap_reclaim_state {
    atomic_uint_fast64_t calls;
    atomic_uint_fast64_t prompt_calls;
    atomic_uint_fast64_t ranges;
    atomic_uint_fast64_t bytes;
    atomic_uint_fast64_t file_bytes;
    atomic_uint_fast64_t pageout_failures;
    atomic_uint_fast64_t dontneed_failures;
};

static struct ggml_kimi_phase1_mmap_reclaim_state ggml_kimi_phase1_mmap_reclaim = {
    ATOMIC_VAR_INIT(0),
    ATOMIC_VAR_INIT(0),
    ATOMIC_VAR_INIT(0),
    ATOMIC_VAR_INIT(0),
    ATOMIC_VAR_INIT(0),
    ATOMIC_VAR_INIT(0),
    ATOMIC_VAR_INIT(0),
};

static bool ggml_kimi_phase1_mmap_reclaim_enabled(void) {
#if defined(__gnu_linux__)
    const char * phase1 = getenv("GGML_MOE_PREFILL_PHASE1");
    const char * reclaim = getenv("GGML_MOE_PHASE1_MMAP_RECLAIM");
    return phase1 && phase1[0] && phase1[0] != '0' &&
           reclaim && reclaim[0] && reclaim[0] != '0';
#else
    return false;
#endif
}

// Keep this ownership check in sync with llama_is_expert_tensor_name() in
// llama-model-loader.cpp.  It deliberately accepts only routed-expert weights;
// dense FFN, attention, norm, and router tensors cannot reach madvise below.
static bool ggml_kimi_phase1_reclaim_is_expert_tensor(const char * name) {
    if (!name || strncmp(name, "blk.", 4) != 0) {
        return false;
    }

    const char * p = name + 4;
    if (*p < '0' || *p > '9') {
        return false;
    }
    while (*p >= '0' && *p <= '9') {
        ++p;
    }
    if (strncmp(p, ".ffn_", 5) != 0) {
        return false;
    }
    p += 5;

    if (strncmp(p, "gate_up_", 8) == 0) {
        p += 8;
    } else if (strncmp(p, "down_", 5) == 0 || strncmp(p, "gate_", 5) == 0) {
        p += 5;
    } else if (strncmp(p, "up_", 3) == 0) {
        p += 3;
    } else {
        return false;
    }

    if (strncmp(p, "ch", 2) == 0) {
        p += 2;
    }
    return strncmp(p, "exps.", 5) == 0;
}

static void ggml_kimi_phase1_mmap_reclaim_tensor(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * ids,
        int64_t n_as,
        size_t nb02,
        size_t expert_bytes) {
    if (!ggml_kimi_phase1_mmap_reclaim_enabled() || ids->ne[1] <= 1) {
        return;
    }

    if (!ggml_kimi_phase1_reclaim_is_expert_tensor(src0->name)) {
        if (params->ith == 0) {
            fprintf(stderr,
                    "[moe_phase1_mmap_reclaim] rejected_non_expert tensor=%s\n",
                    src0->name);
        }
        return;
    }

    // Every worker may still be reading src0 during the CPU fallback.  The
    // first barrier closes that lifetime; the second keeps the next graph node
    // from faulting pages while MADV_DONTNEED is still walking this mapping.
    ggml_barrier(params->threadpool);
    if (params->ith == 0 && src0->data && n_as > 0 && nb02 > 0 && expert_bytes > 0 &&
            (uint64_t) (n_as - 1) <= (SIZE_MAX - expert_bytes) / nb02) {
#if defined(__gnu_linux__)
        const long page_size = sysconf(_SC_PAGESIZE);
        const bool file_api = llama_mmap_dontneed_expert_host_range != NULL;
        uint64_t reclaim_ranges = 0;
        uint64_t reclaim_bytes = 0;
        uint64_t reclaim_file_bytes = 0;
        uint64_t pageout_failures = 0;
        uint64_t dontneed_failures = 0;

        for (int64_t range = 0; page_size > 0 && range < 1; ++range) {
            const size_t offset = 0;
            const size_t bytes = (size_t) (n_as - 1) * nb02 + expert_bytes;
            const uintptr_t base = (uintptr_t) src0->data;
            if (offset > UINTPTR_MAX - base || bytes > UINTPTR_MAX - (base + offset)) {
                ++dontneed_failures;
                continue;
            }
            const uintptr_t begin = base + offset;
            const uintptr_t page = (uintptr_t) page_size;
            const uintptr_t end = begin + bytes;
            const uintptr_t begin_remainder = begin % page;
            if (begin_remainder != 0 && begin > UINTPTR_MAX - (page - begin_remainder)) {
                ++dontneed_failures;
                continue;
            }
            // Only advise pages wholly owned by this expert tensor.  Expanding
            // outward could reclaim a neighbouring dense tensor's boundary
            // page when GGUF offsets are not page-aligned.
            const uintptr_t aligned_begin = begin_remainder == 0 ?
                begin : begin + page - begin_remainder;
            const uintptr_t aligned_end = end - end % page;
            if (aligned_end <= aligned_begin) {
                ++dontneed_failures;
                continue;
            }

            int pageout_failed = 0;
#if defined(MADV_PAGEOUT)
            pageout_failed = madvise(
                    (void *) aligned_begin, aligned_end - aligned_begin, MADV_PAGEOUT) != 0;
#endif
            size_t file_bytes = 0;
            const int dontneed_failed = !file_api ||
                !llama_mmap_dontneed_expert_host_range(
                    src0->name, (const void *) begin, bytes, &file_bytes);
            ++reclaim_ranges;
            reclaim_bytes += aligned_end - aligned_begin;
            reclaim_file_bytes += file_bytes;
            pageout_failures += pageout_failed != 0;
            dontneed_failures += dontneed_failed != 0;
        }

        if (reclaim_ranges > 0) {
            const uint64_t call = atomic_fetch_add_explicit(
                    &ggml_kimi_phase1_mmap_reclaim.calls, 1, memory_order_relaxed) + 1;
            const uint64_t phase_call = atomic_fetch_add_explicit(
                    &ggml_kimi_phase1_mmap_reclaim.prompt_calls,
                    1, memory_order_relaxed) + 1;
            const uint64_t total_ranges = atomic_fetch_add_explicit(
                    &ggml_kimi_phase1_mmap_reclaim.ranges,
                    reclaim_ranges,
                    memory_order_relaxed) + reclaim_ranges;
            const uint64_t total_bytes = atomic_fetch_add_explicit(
                    &ggml_kimi_phase1_mmap_reclaim.bytes,
                    reclaim_bytes,
                    memory_order_relaxed) + reclaim_bytes;
            const uint64_t total_file_bytes = atomic_fetch_add_explicit(
                    &ggml_kimi_phase1_mmap_reclaim.file_bytes,
                    reclaim_file_bytes,
                    memory_order_relaxed) + reclaim_file_bytes;
            atomic_fetch_add_explicit(
                    &ggml_kimi_phase1_mmap_reclaim.pageout_failures,
                    pageout_failures,
                    memory_order_relaxed);
            atomic_fetch_add_explicit(
                    &ggml_kimi_phase1_mmap_reclaim.dontneed_failures,
                    dontneed_failures,
                    memory_order_relaxed);
            if (call == 1 || call % 276 == 0 || pageout_failures || dontneed_failures) {
                fprintf(stderr,
                        "[moe_phase1_mmap_reclaim] calls=%" PRIu64 " ranges=%" PRIu64
                        " bytes=%" PRIu64 " file_bytes=%" PRIu64 " file_api=%d phase=%s"
                        " ownership=expert_tensor_whitelist"
                        " phase_calls=%" PRIu64 " tensor=%s tensor_ranges=%" PRIu64
                        " tensor_bytes=%" PRIu64 " pageout_failures=%" PRIuFAST64
                        " dontneed_failures=%" PRIuFAST64 "\n",
                        call,
                        total_ranges,
                        total_bytes,
                        total_file_bytes,
                        file_api ? 1 : 0,
                        "prompt",
                        phase_call,
                        src0->name,
                        reclaim_ranges,
                        reclaim_bytes,
                        atomic_load_explicit(
                            &ggml_kimi_phase1_mmap_reclaim.pageout_failures, memory_order_relaxed),
                        atomic_load_explicit(
                            &ggml_kimi_phase1_mmap_reclaim.dontneed_failures, memory_order_relaxed));
            }
        }
#endif
    }
    ggml_barrier(params->threadpool);
}

static bool ggml_kimi_moe_mixed_iq2_iq3_pair(enum ggml_type up_type, enum ggml_type gate_type) {
    return ggml_model_type_is_kimi() &&
           ((up_type == GGML_TYPE_IQ2_S && gate_type == GGML_TYPE_IQ3_XXS) ||
            (up_type == GGML_TYPE_IQ3_XXS && gate_type == GGML_TYPE_IQ2_S));
}

struct ggml_kimi_cpu_moe_profile_op {
    uint64_t calls;
    uint64_t setup_us;
    uint64_t convert_us;
    uint64_t route_us;
    uint64_t route_tail_us;
    uint64_t route_barrier_us;
    uint64_t eligibility_us;
    uint64_t cuda_batch_us;
    uint64_t batch_profile_bookkeeping_us;
    uint64_t batch_post_us;
    uint64_t cuda_single_us;
    uint64_t post_cuda_barrier_us;
    uint64_t post_cuda_tail_us;
    uint64_t fallback_us;
    uint64_t total_us;
    uint64_t cuda_batch_accepted;
    uint64_t cuda_batch_declined;
    uint64_t cuda_single_accepted;
    uint64_t cuda_single_declined;
    uint64_t decode_calls;
    uint64_t decode_total_us;
    uint64_t decode_cuda_batch_us;
    uint64_t decode_cuda_single_us;
    uint64_t decode_fallback_us;
    uint64_t prompt_calls;
    uint64_t prompt_total_us;
    uint64_t prompt_cuda_batch_us;
    uint64_t prompt_cuda_single_us;
    uint64_t prompt_fallback_us;
};

#define GGML_KIMI_CPU_MOE_NAME_PROFILE_MAX 512
#define GGML_KIMI_CPU_MOE_NAME_PROFILE_LEN 96
#define GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_MAX 32768

enum ggml_kimi_cpu_moe_eligibility_reason {
    GGML_KIMI_CPU_MOE_ELIGIBLE = 0,
    GGML_KIMI_CPU_MOE_INELIG_ENV,
    GGML_KIMI_CPU_MOE_INELIG_BATCH_FN,
    GGML_KIMI_CPU_MOE_INELIG_AVAILABLE_FN,
    GGML_KIMI_CPU_MOE_INELIG_AVAILABLE_FALSE,
    GGML_KIMI_CPU_MOE_INELIG_UNSUPPORTED,
    GGML_KIMI_CPU_MOE_INELIG_SRC1_TYPE,
    GGML_KIMI_CPU_MOE_INELIG_NE13,
    GGML_KIMI_CPU_MOE_INELIG_DST_TYPE,
    GGML_KIMI_CPU_MOE_ELIGIBILITY_REASON_COUNT,
};

struct ggml_kimi_cpu_moe_name_profile_entry {
    char name[GGML_KIMI_CPU_MOE_NAME_PROFILE_LEN];
    uint64_t calls;
    uint64_t total_us;
    uint64_t fallback_us;
    uint64_t decode_calls;
    uint64_t decode_total_us;
    uint64_t decode_fallback_us;
    uint64_t prompt_calls;
    uint64_t prompt_total_us;
    uint64_t prompt_fallback_us;
    uint64_t cuda_batch_eligible;
    uint64_t cuda_batch_accepted;
    uint64_t cuda_batch_declined;
    int src0_type;
    uint64_t eligibility[GGML_KIMI_CPU_MOE_ELIGIBILITY_REASON_COUNT];
    uint64_t eligibility_decode[GGML_KIMI_CPU_MOE_ELIGIBILITY_REASON_COUNT];
    uint64_t eligibility_prompt[GGML_KIMI_CPU_MOE_ELIGIBILITY_REASON_COUNT];
};

struct ggml_kimi_cpu_moe_profile_state {
    bool registered;
    bool enabled;
    bool name_enabled;
    struct ggml_kimi_cpu_moe_profile_op up_gate;
    struct ggml_kimi_cpu_moe_profile_op down;
    struct ggml_kimi_cpu_moe_name_profile_entry names[GGML_KIMI_CPU_MOE_NAME_PROFILE_MAX];
    int n_names;
};

static struct ggml_kimi_cpu_moe_profile_state ggml_kimi_cpu_moe_profile;

static void ggml_kimi_cpu_moe_profile_record_phase(
        struct ggml_kimi_cpu_moe_profile_op * op,
        bool prompt_phase,
        uint64_t total_us,
        uint64_t cuda_batch_us,
        uint64_t cuda_single_us,
        uint64_t fallback_us) {
    if (prompt_phase) {
        op->prompt_calls++;
        op->prompt_total_us += total_us;
        op->prompt_cuda_batch_us += cuda_batch_us;
        op->prompt_cuda_single_us += cuda_single_us;
        op->prompt_fallback_us += fallback_us;
    } else {
        op->decode_calls++;
        op->decode_total_us += total_us;
        op->decode_cuda_batch_us += cuda_batch_us;
        op->decode_cuda_single_us += cuda_single_us;
        op->decode_fallback_us += fallback_us;
    }
}

struct ggml_kimi_cpu_moe_fallback_profile_entry {
    char name[GGML_KIMI_CPU_MOE_NAME_PROFILE_LEN];
    int expert_idx;
    int src0_type;
    bool prompt_phase;
    uint64_t count;
    uint64_t calls;
    uint64_t fallback_us;
    uint64_t touch_us;
    uint64_t pack_mmap_calls;
    uint64_t gguf_calls;
    size_t expert_bytes;
};

struct ggml_kimi_cpu_moe_fallback_profile_state {
    bool initialized;
    bool registered;
    bool enabled;
    const char * out;
    struct ggml_kimi_cpu_moe_fallback_profile_entry entries[GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_MAX];
    int n_entries;
    uint64_t dropped;
};

static struct ggml_kimi_cpu_moe_fallback_profile_state ggml_kimi_cpu_moe_fallback_profile;

struct ggml_kimi_cpu_fallback_pack_mmap_state {
    bool initialized;
    bool enabled;
    bool registered;
    uint64_t hits;
    uint64_t misses;
    uint64_t bytes;
    uint64_t fallback_gguf;
};

static struct ggml_kimi_cpu_fallback_pack_mmap_state ggml_kimi_cpu_fallback_pack_mmap;

#define GGML_KIMI_CPU_FALLBACK_SOURCE_PROFILE_MAX 128
#define GGML_KIMI_CPU_FALLBACK_SOURCE_PATH_MAX 256

struct ggml_kimi_cpu_fallback_source_profile_entry {
    char source_path[GGML_KIMI_CPU_FALLBACK_SOURCE_PATH_MAX];
    int source_idx;
    int source_is_gguf;
    int src0_type;
    bool prompt_phase;
    char role[8];
    uint64_t calls;
    uint64_t rows;
    uint64_t bytes;
};

struct ggml_kimi_cpu_fallback_source_profile_state {
    bool initialized;
    bool registered;
    bool enabled;
    const char * out;
    struct ggml_kimi_cpu_fallback_source_profile_entry entries[GGML_KIMI_CPU_FALLBACK_SOURCE_PROFILE_MAX];
    int n_entries;
    uint64_t dropped;
};

static struct ggml_kimi_cpu_fallback_source_profile_state ggml_kimi_cpu_fallback_source_profile;

static void ggml_kimi_cpu_fallback_source_profile_report(void) {
    if (!ggml_kimi_cpu_fallback_source_profile.enabled ||
            !ggml_kimi_cpu_fallback_source_profile.out ||
            !ggml_kimi_cpu_fallback_source_profile.out[0]) {
        return;
    }

    FILE * f = fopen(ggml_kimi_cpu_fallback_source_profile.out, "w");
    if (!f) {
        fprintf(stderr,
                "[kimi_cpu_fallback_source_profile] open failed: %s\n",
                ggml_kimi_cpu_fallback_source_profile.out);
        return;
    }

    fprintf(f, "rank,phase,src0_type,role,source_idx,source_is_gguf,calls,rows,bytes,source_path\n");
    for (int i = 0; i < ggml_kimi_cpu_fallback_source_profile.n_entries; ++i) {
        const struct ggml_kimi_cpu_fallback_source_profile_entry * e =
            &ggml_kimi_cpu_fallback_source_profile.entries[i];
        fprintf(f, "%d,%s,%d,%s,%d,%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%s\n",
                i + 1,
                e->prompt_phase ? "prompt" : "decode",
                e->src0_type,
                e->role,
                e->source_idx,
                e->source_is_gguf,
                e->calls,
                e->rows,
                e->bytes,
                e->source_path);
    }
    fclose(f);

    fprintf(stderr,
            "[kimi_cpu_fallback_source_profile] written: %s entries=%d dropped=%" PRIu64 "\n",
            ggml_kimi_cpu_fallback_source_profile.out,
            ggml_kimi_cpu_fallback_source_profile.n_entries,
            ggml_kimi_cpu_fallback_source_profile.dropped);
}

static bool ggml_kimi_cpu_fallback_source_profile_enabled(void) {
    if (!ggml_model_type_is_kimi()) {
        return false;
    }
    if (!ggml_kimi_cpu_fallback_source_profile.initialized) {
        ggml_kimi_cpu_fallback_source_profile.initialized = true;
        ggml_kimi_cpu_fallback_source_profile.out = getenv("GGML_MOE_CPU_FALLBACK_SOURCE_PROFILE_OUT");
        ggml_kimi_cpu_fallback_source_profile.enabled =
            ggml_kimi_cpu_fallback_source_profile.out &&
            ggml_kimi_cpu_fallback_source_profile.out[0];
        if (ggml_kimi_cpu_fallback_source_profile.enabled &&
                !ggml_kimi_cpu_fallback_source_profile.registered) {
            ggml_kimi_cpu_fallback_source_profile.registered = true;
            atexit(ggml_kimi_cpu_fallback_source_profile_report);
        }
    }

    return ggml_kimi_cpu_fallback_source_profile.enabled;
}

static const char * ggml_kimi_cpu_moe_tensor_role(const char * tensor_name) {
    if (!tensor_name) {
        return "other";
    }
    if (strstr(tensor_name, "ffn_down_exps")) {
        return "down";
    }
    if (strstr(tensor_name, "ffn_up_exps")) {
        return "up";
    }
    if (strstr(tensor_name, "ffn_gate_exps")) {
        return "gate";
    }
    return "other";
}

static void ggml_kimi_cpu_fallback_source_profile_record(
        const char * tensor_name,
        int src0_type,
        bool prompt_phase,
        int source_idx,
        const char * source_path,
        int source_is_gguf,
        int64_t rows,
        size_t expert_bytes) {
    if (!ggml_kimi_cpu_fallback_source_profile_enabled() || rows <= 0) {
        return;
    }

    const char * role = ggml_kimi_cpu_moe_tensor_role(tensor_name);
    const char * safe_path = source_path ? source_path : "<unknown>";
    int idx = -1;

    for (int i = 0; i < ggml_kimi_cpu_fallback_source_profile.n_entries; ++i) {
        struct ggml_kimi_cpu_fallback_source_profile_entry * e =
            &ggml_kimi_cpu_fallback_source_profile.entries[i];
        if (e->source_idx == source_idx &&
                e->source_is_gguf == source_is_gguf &&
                e->src0_type == src0_type &&
                e->prompt_phase == prompt_phase &&
                strncmp(e->role, role, sizeof(e->role)) == 0 &&
                strncmp(e->source_path, safe_path, GGML_KIMI_CPU_FALLBACK_SOURCE_PATH_MAX) == 0) {
            idx = i;
            break;
        }
    }

    if (idx < 0) {
        if (ggml_kimi_cpu_fallback_source_profile.n_entries >= GGML_KIMI_CPU_FALLBACK_SOURCE_PROFILE_MAX) {
            ggml_kimi_cpu_fallback_source_profile.dropped++;
            return;
        }
        idx = ggml_kimi_cpu_fallback_source_profile.n_entries++;
        struct ggml_kimi_cpu_fallback_source_profile_entry * e =
            &ggml_kimi_cpu_fallback_source_profile.entries[idx];
        snprintf(e->source_path, sizeof(e->source_path), "%s", safe_path);
        snprintf(e->role, sizeof(e->role), "%s", role);
        e->source_idx = source_idx;
        e->source_is_gguf = source_is_gguf;
        e->src0_type = src0_type;
        e->prompt_phase = prompt_phase;
    }

    struct ggml_kimi_cpu_fallback_source_profile_entry * e =
        &ggml_kimi_cpu_fallback_source_profile.entries[idx];
    e->calls++;
    e->rows += (uint64_t) rows;
    e->bytes += (uint64_t) rows * (uint64_t) expert_bytes;
}

static void ggml_kimi_cpu_fallback_pack_mmap_report(void) {
    if (!ggml_kimi_cpu_fallback_pack_mmap.enabled &&
            ggml_kimi_cpu_fallback_pack_mmap.hits == 0 &&
            ggml_kimi_cpu_fallback_pack_mmap.misses == 0) {
        return;
    }
    fprintf(stderr,
            "[kimi_cpu_fallback_pack_mmap] enabled=%d hits=%" PRIu64 " misses=%" PRIu64
            " bytes=%" PRIu64 " fallback_gguf=%" PRIu64 "\n",
            ggml_kimi_cpu_fallback_pack_mmap.enabled ? 1 : 0,
            ggml_kimi_cpu_fallback_pack_mmap.hits,
            ggml_kimi_cpu_fallback_pack_mmap.misses,
            ggml_kimi_cpu_fallback_pack_mmap.bytes,
            ggml_kimi_cpu_fallback_pack_mmap.fallback_gguf);
}

static bool ggml_kimi_cpu_fallback_pack_mmap_enabled(void) {
    if (!ggml_model_type_is_kimi()) {
        return false;
    }
    if (!ggml_kimi_cpu_fallback_pack_mmap.initialized) {
        ggml_kimi_cpu_fallback_pack_mmap.initialized = true;
        const char * env = getenv("GGML_MOE_CPU_FALLBACK_PACK_MMAP");
        ggml_kimi_cpu_fallback_pack_mmap.enabled = env && env[0] && env[0] != '0';
        if (ggml_kimi_cpu_fallback_pack_mmap.enabled && !ggml_kimi_cpu_fallback_pack_mmap.registered) {
            ggml_kimi_cpu_fallback_pack_mmap.registered = true;
            atexit(ggml_kimi_cpu_fallback_pack_mmap_report);
        }
    }
    return ggml_kimi_cpu_fallback_pack_mmap.enabled && ggml_cuda_moe_expert_pack_mmap_ptr != NULL;
}

static bool ggml_kimi_cpu_fallback_pack_mmap_prompt_enabled(void) {
    static int initialized = 0;
    static bool enabled = false;
    if (!initialized) {
        initialized = 1;
        const char * env = getenv("GGML_MOE_CPU_FALLBACK_PACK_MMAP_PROMPT");
        enabled = env && env[0] && env[0] != '0';
    }
    return enabled;
}

static bool ggml_kimi_cpu_fallback_miss_trace_enabled(void) {
    static int initialized = 0;
    static bool enabled = false;
    if (!initialized) {
        initialized = 1;
        const char * env = getenv("GGML_MOE_CPU_FALLBACK_MISS_TRACE");
        enabled = env && env[0] && env[0] != '0';
    }
    return enabled;
}

static void ggml_kimi_cpu_fallback_pack_mmap_prepare(
        const char * tensor_name,
        int src0_type,
        bool prompt_phase,
        int64_t n_as,
        const int64_t * matrix_row_counts,
        size_t expert_bytes,
        const void ** mmap_ptrs) {
    for (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        mmap_ptrs[cur_a] = NULL;
    }
    if ((prompt_phase && !ggml_kimi_cpu_fallback_pack_mmap_prompt_enabled()) ||
            !ggml_kimi_cpu_fallback_pack_mmap_enabled() ||
            !tensor_name || !tensor_name[0] || expert_bytes == 0) {
        return;
    }
    const bool trace_misses = ggml_kimi_cpu_fallback_miss_trace_enabled();
    const bool source_profile = ggml_kimi_cpu_fallback_source_profile_enabled() &&
        ggml_cuda_moe_expert_pack_mmap_ptr_source_debug != NULL;
    for (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        if (matrix_row_counts[cur_a] == 0) {
            continue;
        }
        const char * reason = "entry_missing";
        size_t entry_nbytes = 0;
        uint64_t entry_offset = 0;
        int source_idx = -1;
        int source_is_gguf = 0;
        const char * source_path = NULL;
        const void * ptr = NULL;
        if (source_profile) {
            ptr = ggml_cuda_moe_expert_pack_mmap_ptr_source_debug(
                    tensor_name,
                    (int) cur_a,
                    expert_bytes,
                    &reason,
                    &entry_nbytes,
                    &entry_offset,
                    &source_idx,
                    &source_path,
                    &source_is_gguf);
        } else if (trace_misses && ggml_cuda_moe_expert_pack_mmap_ptr_debug != NULL) {
            ptr = ggml_cuda_moe_expert_pack_mmap_ptr_debug(
                    tensor_name,
                    (int) cur_a,
                    expert_bytes,
                    &reason,
                    &entry_nbytes,
                    &entry_offset);
        } else {
            ptr = ggml_cuda_moe_expert_pack_mmap_ptr(tensor_name, (int) cur_a, expert_bytes);
        }
        if (ptr) {
            mmap_ptrs[cur_a] = ptr;
            ggml_kimi_cpu_fallback_pack_mmap.hits++;
            ggml_kimi_cpu_fallback_pack_mmap.bytes += expert_bytes;
            if (source_profile) {
                ggml_kimi_cpu_fallback_source_profile_record(
                        tensor_name,
                        src0_type,
                        prompt_phase,
                        source_idx,
                        source_path,
                        source_is_gguf,
                        matrix_row_counts[cur_a],
                        expert_bytes);
            }
        } else {
            ggml_kimi_cpu_fallback_pack_mmap.misses++;
            ggml_kimi_cpu_fallback_pack_mmap.fallback_gguf++;
            if (trace_misses) {
                fprintf(stderr,
                        "[kimi_cpu_fallback_pack_mmap_miss] phase=%s tensor=%s expert=%" PRId64
                        " src0_type=%d expert_bytes=%zu active_rows=%" PRId64
                        " reason=%s entry_bytes=%zu entry_offset=%" PRIu64 "\n",
                        prompt_phase ? "prompt" : "decode",
                        tensor_name,
                        cur_a,
                        src0_type,
                        expert_bytes,
                        matrix_row_counts[cur_a],
                        reason ? reason : "unknown",
                        entry_nbytes,
                        entry_offset);
            }
        }
    }
}

static bool ggml_kimi_cpu_moe_eligibility_profile_enabled(void) {
    if (!ggml_model_type_is_kimi()) {
        return false;
    }
    static int initialized = 0;
    static bool enabled = false;

    if (!initialized) {
        initialized = 1;
        enabled = getenv("GGML_KIMI_CPU_MOE_ELIGIBILITY_PROFILE") != NULL;
    }

    return enabled;
}

static int ggml_kimi_cpu_moe_name_profile_top(void) {
    const char * env = getenv("GGML_KIMI_CPU_MOE_NAME_PROFILE_TOP");
    if (!env || !env[0]) {
        return 40;
    }

    char * end = NULL;
    long value = strtol(env, &end, 10);
    if (end == env || value <= 0) {
        return 40;
    }
    if (value > GGML_KIMI_CPU_MOE_NAME_PROFILE_MAX) {
        value = GGML_KIMI_CPU_MOE_NAME_PROFILE_MAX;
    }
    return (int) value;
}

static void ggml_kimi_cpu_moe_fallback_profile_report(void) {
    if (!ggml_kimi_cpu_moe_fallback_profile.enabled ||
            !ggml_kimi_cpu_moe_fallback_profile.out ||
            !ggml_kimi_cpu_moe_fallback_profile.out[0]) {
        return;
    }

    FILE * f = fopen(ggml_kimi_cpu_moe_fallback_profile.out, "w");
    if (!f) {
        fprintf(stderr,
                "[kimi_cpu_moe_fallback_profile] open failed: %s\n",
                ggml_kimi_cpu_moe_fallback_profile.out);
        return;
    }

    fprintf(f, "rank,count,calls,fallback_us,touch_us,expert_bytes,src0_type,phase,expert_idx,pack_mmap_calls,gguf_calls,tensor\n");
    for (int i = 0; i < ggml_kimi_cpu_moe_fallback_profile.n_entries; ++i) {
        const struct ggml_kimi_cpu_moe_fallback_profile_entry * e =
            &ggml_kimi_cpu_moe_fallback_profile.entries[i];
        fprintf(f, "%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%zu,%d,%s,%d,%" PRIu64 ",%" PRIu64 ",%s\n",
                i + 1,
                e->count,
                e->calls,
                e->fallback_us,
                e->touch_us,
                e->expert_bytes,
                e->src0_type,
                e->prompt_phase ? "prompt" : "decode",
                e->expert_idx,
                e->pack_mmap_calls,
                e->gguf_calls,
                e->name);
    }
    fclose(f);

    fprintf(stderr,
            "[kimi_cpu_moe_fallback_profile] written: %s entries=%d dropped=%" PRIu64 "\n",
            ggml_kimi_cpu_moe_fallback_profile.out,
            ggml_kimi_cpu_moe_fallback_profile.n_entries,
            ggml_kimi_cpu_moe_fallback_profile.dropped);
}

static bool ggml_kimi_cpu_moe_fallback_profile_enabled(void) {
    if (!ggml_model_type_is_kimi()) {
        return false;
    }
    if (!ggml_kimi_cpu_moe_fallback_profile.initialized) {
        ggml_kimi_cpu_moe_fallback_profile.initialized = true;
        ggml_kimi_cpu_moe_fallback_profile.out = getenv("GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT");
        ggml_kimi_cpu_moe_fallback_profile.enabled =
            ggml_kimi_cpu_moe_fallback_profile.out &&
            ggml_kimi_cpu_moe_fallback_profile.out[0];
        if (ggml_kimi_cpu_moe_fallback_profile.enabled &&
                !ggml_kimi_cpu_moe_fallback_profile.registered) {
            ggml_kimi_cpu_moe_fallback_profile.registered = true;
            atexit(ggml_kimi_cpu_moe_fallback_profile_report);
        }
    }

    return ggml_kimi_cpu_moe_fallback_profile.enabled;
}

static const char * ggml_kimi_cpu_moe_eligibility_reason_name(enum ggml_kimi_cpu_moe_eligibility_reason reason) {
    switch (reason) {
        case GGML_KIMI_CPU_MOE_ELIGIBLE: return "eligible";
        case GGML_KIMI_CPU_MOE_INELIG_ENV: return "batch_env_missing";
        case GGML_KIMI_CPU_MOE_INELIG_BATCH_FN: return "batch_fn_missing";
        case GGML_KIMI_CPU_MOE_INELIG_AVAILABLE_FN: return "available_fn_missing";
        case GGML_KIMI_CPU_MOE_INELIG_AVAILABLE_FALSE: return "stream_available_false";
        case GGML_KIMI_CPU_MOE_INELIG_UNSUPPORTED: return "batch_unsupported";
        case GGML_KIMI_CPU_MOE_INELIG_SRC1_TYPE: return "src1_not_f32";
        case GGML_KIMI_CPU_MOE_INELIG_NE13: return "ne13_not1";
        case GGML_KIMI_CPU_MOE_INELIG_DST_TYPE: return "dst_not_f32";
        default: return "unknown";
    }
}

static void ggml_kimi_cpu_moe_fallback_profile_record(
        const char * name,
        int src0_type,
        bool prompt_phase,
        int expert_idx,
        int64_t count,
        size_t expert_bytes,
        uint64_t fallback_us,
        uint64_t touch_us,
        bool pack_mmap_source) {
    if (!ggml_kimi_cpu_moe_fallback_profile_enabled() || count <= 0) {
        return;
    }

    const char * safe_name = name ? name : "<unnamed>";
    int idx = -1;

    for (int i = 0; i < ggml_kimi_cpu_moe_fallback_profile.n_entries; ++i) {
        struct ggml_kimi_cpu_moe_fallback_profile_entry * e =
            &ggml_kimi_cpu_moe_fallback_profile.entries[i];
        if (e->expert_idx == expert_idx &&
                e->src0_type == src0_type &&
                e->prompt_phase == prompt_phase &&
                strncmp(e->name, safe_name, GGML_KIMI_CPU_MOE_NAME_PROFILE_LEN) == 0) {
            idx = i;
            break;
        }
    }

    if (idx < 0) {
        if (ggml_kimi_cpu_moe_fallback_profile.n_entries >= GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_MAX) {
            ggml_kimi_cpu_moe_fallback_profile.dropped++;
            return;
        }

        idx = ggml_kimi_cpu_moe_fallback_profile.n_entries++;
        struct ggml_kimi_cpu_moe_fallback_profile_entry * e =
            &ggml_kimi_cpu_moe_fallback_profile.entries[idx];
        snprintf(e->name, GGML_KIMI_CPU_MOE_NAME_PROFILE_LEN, "%s", safe_name);
        e->expert_idx = expert_idx;
        e->src0_type = src0_type;
        e->prompt_phase = prompt_phase;
        e->expert_bytes = expert_bytes;
    }

    struct ggml_kimi_cpu_moe_fallback_profile_entry * e =
        &ggml_kimi_cpu_moe_fallback_profile.entries[idx];
    e->count += (uint64_t) count;
    e->calls++;
    e->fallback_us += fallback_us;
    e->touch_us += touch_us;
    if (pack_mmap_source) {
        e->pack_mmap_calls++;
    } else {
        e->gguf_calls++;
    }
    e->expert_bytes = expert_bytes;
}

static void ggml_kimi_cpu_moe_profile_report_op(const char * name, const struct ggml_kimi_cpu_moe_profile_op * op) {
    if (op->calls == 0) {
        return;
    }

    const uint64_t decode_cuda_us = op->decode_cuda_batch_us + op->decode_cuda_single_us;
    const uint64_t prompt_cuda_us = op->prompt_cuda_batch_us + op->prompt_cuda_single_us;
    fprintf(stderr,
            "[kimi_cpu_moe_profile] %s calls=%" PRIu64
            " total=%.3f ms/call convert_t0=%.3f route=%.3f route_barrier=%.3f"
            " cuda_batch=%.3f cuda_single=%.3f post_cuda_barrier=%.3f fallback_t0=%.3f"
            " batch_accept=%" PRIu64 " batch_decline=%" PRIu64
            " single_accept=%" PRIu64 " single_decline=%" PRIu64
            " decode_calls=%" PRIu64 " decode_total=%.3f ms/call decode_cuda=%.3f ms/call"
            " decode_cuda_batch=%.3f ms/call decode_cuda_single=%.3f ms/call decode_fallback=%.3f ms/call"
            " prompt_calls=%" PRIu64 " prompt_total=%.3f ms/call prompt_cuda=%.3f ms/call"
            " prompt_cuda_batch=%.3f ms/call prompt_cuda_single=%.3f ms/call prompt_fallback=%.3f ms/call\n",
            name,
            op->calls,
            (double) op->total_us / 1000.0 / (double) op->calls,
            (double) op->convert_us / 1000.0 / (double) op->calls,
            (double) op->route_us / 1000.0 / (double) op->calls,
            (double) op->route_barrier_us / 1000.0 / (double) op->calls,
            (double) op->cuda_batch_us / 1000.0 / (double) op->calls,
            (double) op->cuda_single_us / 1000.0 / (double) op->calls,
            (double) op->post_cuda_barrier_us / 1000.0 / (double) op->calls,
            (double) op->fallback_us / 1000.0 / (double) op->calls,
            op->cuda_batch_accepted,
            op->cuda_batch_declined,
            op->cuda_single_accepted,
            op->cuda_single_declined,
            op->decode_calls,
            op->decode_calls ? (double) op->decode_total_us / 1000.0 / (double) op->decode_calls : 0.0,
            op->decode_calls ? (double) decode_cuda_us / 1000.0 / (double) op->decode_calls : 0.0,
            op->decode_calls ? (double) op->decode_cuda_batch_us / 1000.0 / (double) op->decode_calls : 0.0,
            op->decode_calls ? (double) op->decode_cuda_single_us / 1000.0 / (double) op->decode_calls : 0.0,
            op->decode_calls ? (double) op->decode_fallback_us / 1000.0 / (double) op->decode_calls : 0.0,
            op->prompt_calls,
            op->prompt_calls ? (double) op->prompt_total_us / 1000.0 / (double) op->prompt_calls : 0.0,
            op->prompt_calls ? (double) prompt_cuda_us / 1000.0 / (double) op->prompt_calls : 0.0,
            op->prompt_calls ? (double) op->prompt_cuda_batch_us / 1000.0 / (double) op->prompt_calls : 0.0,
            op->prompt_calls ? (double) op->prompt_cuda_single_us / 1000.0 / (double) op->prompt_calls : 0.0,
            op->prompt_calls ? (double) op->prompt_fallback_us / 1000.0 / (double) op->prompt_calls : 0.0);
    fprintf(stderr,
            "[kimi_cpu_moe_profile_raw] %s calls=%" PRIu64
            " total_us=%" PRIu64 " setup_us=%" PRIu64 " convert_us=%" PRIu64
            " route_us=%" PRIu64 " route_tail_us=%" PRIu64
            " route_barrier_us=%" PRIu64 " eligibility_us=%" PRIu64
            " cuda_batch_us=%" PRIu64 " batch_profile_bookkeeping_us=%" PRIu64
            " batch_post_us=%" PRIu64 " cuda_single_us=%" PRIu64
            " post_cuda_barrier_us=%" PRIu64 " post_cuda_tail_us=%" PRIu64
            " fallback_us=%" PRIu64 "\n",
            name,
            op->calls,
            op->total_us,
            op->setup_us,
            op->convert_us,
            op->route_us,
            op->route_tail_us,
            op->route_barrier_us,
            op->eligibility_us,
            op->cuda_batch_us,
            op->batch_profile_bookkeeping_us,
            op->batch_post_us,
            op->cuda_single_us,
            op->post_cuda_barrier_us,
            op->post_cuda_tail_us,
            op->fallback_us);
}

static void ggml_kimi_cpu_moe_profile_report(void) {
    if (!ggml_kimi_cpu_moe_profile.enabled) {
        return;
    }

    ggml_kimi_cpu_moe_profile_report_op("up_gate", &ggml_kimi_cpu_moe_profile.up_gate);
    ggml_kimi_cpu_moe_profile_report_op("down",    &ggml_kimi_cpu_moe_profile.down);

    if (ggml_kimi_cpu_moe_profile.name_enabled) {
        bool printed[GGML_KIMI_CPU_MOE_NAME_PROFILE_MAX] = { false };
        const int top_n = MIN(ggml_kimi_cpu_moe_name_profile_top(), ggml_kimi_cpu_moe_profile.n_names);

        for (int rank = 0; rank < top_n; ++rank) {
            int best = -1;
            for (int i = 0; i < ggml_kimi_cpu_moe_profile.n_names; ++i) {
                if (printed[i]) {
                    continue;
                }
                if (best < 0 ||
                    ggml_kimi_cpu_moe_profile.names[i].total_us > ggml_kimi_cpu_moe_profile.names[best].total_us) {
                    best = i;
                }
            }

            if (best < 0) {
                break;
            }

            printed[best] = true;
            const struct ggml_kimi_cpu_moe_name_profile_entry * e = &ggml_kimi_cpu_moe_profile.names[best];
            fprintf(stderr,
                    "[kimi_cpu_moe_name_profile] top%d name=%s calls=%" PRIu64
                    " total=%.3f ms/call fallback_t0=%.3f ms/call"
                    " batch_eligible=%" PRIu64 " batch_accept=%" PRIu64 " batch_decline=%" PRIu64
                    " decode_calls=%" PRIu64 " decode_total=%.3f ms/call decode_fallback=%.3f ms/call"
                    " prompt_calls=%" PRIu64 " prompt_total=%.3f ms/call prompt_fallback=%.3f ms/call\n",
                    rank + 1,
                    e->name,
                    e->calls,
                    (double) e->total_us / 1000.0 / (double) e->calls,
                    (double) e->fallback_us / 1000.0 / (double) e->calls,
                    e->cuda_batch_eligible,
                    e->cuda_batch_accepted,
                    e->cuda_batch_declined,
                    e->decode_calls,
                    e->decode_calls ? (double) e->decode_total_us / 1000.0 / (double) e->decode_calls : 0.0,
                    e->decode_calls ? (double) e->decode_fallback_us / 1000.0 / (double) e->decode_calls : 0.0,
                    e->prompt_calls,
                    e->prompt_calls ? (double) e->prompt_total_us / 1000.0 / (double) e->prompt_calls : 0.0,
                    e->prompt_calls ? (double) e->prompt_fallback_us / 1000.0 / (double) e->prompt_calls : 0.0);
            if (ggml_kimi_cpu_moe_eligibility_profile_enabled()) {
                fprintf(stderr,
                        "[kimi_cpu_moe_eligibility_profile] top%d name=%s"
                        " src0_type=%d"
                        " eligible=%" PRIu64
                        " env_missing=%" PRIu64
                        " batch_fn_missing=%" PRIu64
                        " available_fn_missing=%" PRIu64
                        " available_false=%" PRIu64
                        " unsupported=%" PRIu64
                        " src1_not_f32=%" PRIu64
                        " ne13_not1=%" PRIu64
                        " dst_not_f32=%" PRIu64
                        " decode_eligible=%" PRIu64
                        " decode_unsupported=%" PRIu64
                        " prompt_eligible=%" PRIu64
                        " prompt_unsupported=%" PRIu64 "\n",
                        rank + 1,
                        e->name,
                        e->src0_type,
                        e->eligibility[GGML_KIMI_CPU_MOE_ELIGIBLE],
                        e->eligibility[GGML_KIMI_CPU_MOE_INELIG_ENV],
                        e->eligibility[GGML_KIMI_CPU_MOE_INELIG_BATCH_FN],
                        e->eligibility[GGML_KIMI_CPU_MOE_INELIG_AVAILABLE_FN],
                        e->eligibility[GGML_KIMI_CPU_MOE_INELIG_AVAILABLE_FALSE],
                        e->eligibility[GGML_KIMI_CPU_MOE_INELIG_UNSUPPORTED],
                        e->eligibility[GGML_KIMI_CPU_MOE_INELIG_SRC1_TYPE],
                        e->eligibility[GGML_KIMI_CPU_MOE_INELIG_NE13],
                        e->eligibility[GGML_KIMI_CPU_MOE_INELIG_DST_TYPE],
                        e->eligibility_decode[GGML_KIMI_CPU_MOE_ELIGIBLE],
                        e->eligibility_decode[GGML_KIMI_CPU_MOE_INELIG_UNSUPPORTED],
                        e->eligibility_prompt[GGML_KIMI_CPU_MOE_ELIGIBLE],
                        e->eligibility_prompt[GGML_KIMI_CPU_MOE_INELIG_UNSUPPORTED]);
            }
        }
    }
}

static bool ggml_kimi_cpu_moe_profile_enabled(void) {
    static int initialized = 0;

    if (!initialized) {
        initialized = 1;
        ggml_kimi_cpu_moe_profile.enabled =
            ggml_model_type_is_kimi() && getenv("GGML_KIMI_CPU_MOE_PROFILE") != NULL;
        ggml_kimi_cpu_moe_profile.name_enabled =
            ggml_kimi_cpu_moe_profile.enabled && getenv("GGML_KIMI_CPU_MOE_NAME_PROFILE") != NULL;
        if (ggml_kimi_cpu_moe_profile.enabled && !ggml_kimi_cpu_moe_profile.registered) {
            ggml_kimi_cpu_moe_profile.registered = true;
            atexit(ggml_kimi_cpu_moe_profile_report);
        }
    }

    return ggml_kimi_cpu_moe_profile.enabled;
}

static void ggml_kimi_cpu_moe_name_profile_record(
        const char * name,
        uint64_t total_us,
        uint64_t fallback_us,
        bool prompt_phase,
        bool cuda_batch_eligible,
        bool cuda_batch_accepted) {
    if (!ggml_kimi_cpu_moe_profile.name_enabled) {
        return;
    }

    const char * safe_name = name ? name : "<unnamed>";
    int idx = -1;

    for (int i = 0; i < ggml_kimi_cpu_moe_profile.n_names; ++i) {
        if (strncmp(ggml_kimi_cpu_moe_profile.names[i].name, safe_name, GGML_KIMI_CPU_MOE_NAME_PROFILE_LEN) == 0) {
            idx = i;
            break;
        }
    }

    if (idx < 0) {
        if (ggml_kimi_cpu_moe_profile.n_names >= GGML_KIMI_CPU_MOE_NAME_PROFILE_MAX) {
            idx = GGML_KIMI_CPU_MOE_NAME_PROFILE_MAX - 1;
        } else {
            idx = ggml_kimi_cpu_moe_profile.n_names++;
            snprintf(ggml_kimi_cpu_moe_profile.names[idx].name,
                     GGML_KIMI_CPU_MOE_NAME_PROFILE_LEN,
                     "%s",
                     safe_name);
        }
    }

    struct ggml_kimi_cpu_moe_name_profile_entry * e = &ggml_kimi_cpu_moe_profile.names[idx];
    e->calls++;
    e->total_us += total_us;
    e->fallback_us += fallback_us;
    if (prompt_phase) {
        e->prompt_calls++;
        e->prompt_total_us += total_us;
        e->prompt_fallback_us += fallback_us;
    } else {
        e->decode_calls++;
        e->decode_total_us += total_us;
        e->decode_fallback_us += fallback_us;
    }
    if (cuda_batch_eligible) {
        e->cuda_batch_eligible++;
        if (cuda_batch_accepted) {
            e->cuda_batch_accepted++;
        } else {
            e->cuda_batch_declined++;
        }
    }
}

static void ggml_kimi_cpu_moe_name_profile_record_eligibility(
        const char * name,
        int src0_type,
        bool prompt_phase,
        enum ggml_kimi_cpu_moe_eligibility_reason reason) {
    if (!ggml_kimi_cpu_moe_profile.name_enabled ||
            !ggml_kimi_cpu_moe_eligibility_profile_enabled()) {
        return;
    }
    if (reason < 0 || reason >= GGML_KIMI_CPU_MOE_ELIGIBILITY_REASON_COUNT) {
        return;
    }

    const char * safe_name = name ? name : "<unnamed>";
    int idx = -1;

    for (int i = 0; i < ggml_kimi_cpu_moe_profile.n_names; ++i) {
        if (strncmp(ggml_kimi_cpu_moe_profile.names[i].name, safe_name, GGML_KIMI_CPU_MOE_NAME_PROFILE_LEN) == 0) {
            idx = i;
            break;
        }
    }

    if (idx < 0) {
        if (ggml_kimi_cpu_moe_profile.n_names >= GGML_KIMI_CPU_MOE_NAME_PROFILE_MAX) {
            idx = GGML_KIMI_CPU_MOE_NAME_PROFILE_MAX - 1;
        } else {
            idx = ggml_kimi_cpu_moe_profile.n_names++;
            snprintf(ggml_kimi_cpu_moe_profile.names[idx].name,
                     GGML_KIMI_CPU_MOE_NAME_PROFILE_LEN,
                     "%s",
                     safe_name);
        }
    }

    ggml_kimi_cpu_moe_profile.names[idx].eligibility[reason]++;
    if (prompt_phase) {
        ggml_kimi_cpu_moe_profile.names[idx].eligibility_prompt[reason]++;
    } else {
        ggml_kimi_cpu_moe_profile.names[idx].eligibility_decode[reason]++;
    }
    ggml_kimi_cpu_moe_profile.names[idx].src0_type = src0_type;
}

#if !defined(_WIN32)

#define GGML_KIMI_A2_MAX_EXPERTS 1024
#define GGML_KIMI_A2_MAX_WAVES  128

enum ggml_kimi_a2_call_mode {
    GGML_KIMI_A2_CALL_INVALID = -1,
    GGML_KIMI_A2_CALL_TRIPLET = 0,
    GGML_KIMI_A2_CALL_REPLAY  = 1,
    GGML_KIMI_A2_CALL_DOWN    = 2,
};

struct ggml_kimi_a2_pipeline_state {
    char tensor_name[128];
    int32_t expert_ids[GGML_KIMI_A2_MAX_EXPERTS];
    int32_t wave_sizes[GGML_KIMI_A2_MAX_WAVES];
    int32_t wave_first[GGML_KIMI_A2_MAX_WAVES];
    ggml_moe_cpu_bank_result wave_results[GGML_KIMI_A2_MAX_WAVES];
    bool wave_ready[GGML_KIMI_A2_MAX_WAVES];
    bool wave_released[GGML_KIMI_A2_MAX_WAVES];
    int n_experts;
    int n_waves;
    size_t slice_bytes;
    size_t triplet_bytes;
    bool active;
    bool cancel;
    bool provider_done;
    bool provider_ok;
    bool call_ok;
    pthread_t provider_thread;
    uint64_t logical_bytes;
    uint64_t physical_bytes;
    uint64_t provider_us;
    uint64_t kernel_us;
    uint64_t iouring_reads;
    uint64_t ram_admissions;
    uint64_t vram_admissions;
    uint64_t direct_fallbacks;
    size_t ring_bytes;
};

struct ggml_kimi_a2_generation_state {
    bool active;
    int layer;
    int prompt_tokens;
    int routes;
    int n_experts;
    int first_role;
    int companion_role;
    uint64_t route_hash;
    uint64_t generation;
    unsigned consumed_role_mask;
    size_t slice_bytes;
    int32_t expert_ids[GGML_KIMI_A2_MAX_EXPERTS];
    int32_t expert_positions[GGML_KIMI_A2_MAX_EXPERTS];
    void * companion_output;
    size_t companion_output_bytes;
    void * down_weights;
    size_t down_weight_bytes;
    uint64_t logical_bytes;
    uint64_t physical_bytes;
    uint64_t iouring_reads;
};

static pthread_mutex_t ggml_kimi_a2_run_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t ggml_kimi_a2_state_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ggml_kimi_a2_state_cv = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t ggml_kimi_a2_trace_mu = PTHREAD_MUTEX_INITIALIZER;
static struct ggml_kimi_a2_pipeline_state ggml_kimi_a2_pipeline;
static struct ggml_kimi_a2_generation_state ggml_kimi_a2_generation;
static uint64_t ggml_kimi_a2_next_generation;
static uint64_t ggml_kimi_a2_trace_seq;
static bool ggml_kimi_a2_trace_header;
static uint64_t ggml_kimi_a2_role_calls[3];
static uint64_t ggml_kimi_a2_triplets;
static uint64_t ggml_kimi_a2_total_logical_bytes;
static uint64_t ggml_kimi_a2_total_physical_bytes;
static uint64_t ggml_kimi_a2_total_reads;
static uint64_t ggml_kimi_a2_peak_generation_bytes;
static bool ggml_kimi_a2_report_registered;

static int ggml_kimi_tensor_layer(const char * tensor_name) {
    if (!tensor_name || strncmp(tensor_name, "blk.", 4) != 0) {
        return -1;
    }
    char * end = NULL;
    const long layer = strtol(tensor_name + 4, &end, 10);
    return end != tensor_name + 4 && end && *end == '.' && layer >= 0 && layer <= INT_MAX ?
        (int) layer : -1;
}

static bool ggml_kimi_layer_list_contains(const char * layers, int layer) {
    if (!layers || !layers[0] || layer < 0) {
        return false;
    }
    bool found = false;
    const char * current = layers;
    while (*current) {
        while (*current == ' ' || *current == '\t' || *current == ',') {
            ++current;
        }
        if (!*current) {
            break;
        }
        char * end = NULL;
        const long value = strtol(current, &end, 10);
        if (end == current || value < 0 || value > INT_MAX) {
            return false;
        }
        found = found || (int) value == layer;
        current = end;
        while (*current == ' ' || *current == '\t') {
            ++current;
        }
        if (*current && *current != ',') {
            return false;
        }
    }
    return found;
}

static bool ggml_kimi_a2_enabled(const char * tensor_name) {
    const char * env = getenv("GGML_MOE_A2_CPU_GROUPED_PREFILL");
    if (!env || !env[0] || env[0] == '0' ||
            !ggml_cpu_kimi_a2_request_phase_is_active()) {
        return false;
    }

    // Task 33 can restrict the already proven A2 direct/refill provider to a
    // diagnostic set of blocks.  An unset list preserves the original A2
    // all-layer behavior; setting the list alone cannot enable A2.
    const char * layers = getenv("GGML_KIMI_D1_EXPERT_ODIRECT_LAYERS");
    return !layers || !layers[0] ||
        ggml_kimi_layer_list_contains(layers, ggml_kimi_tensor_layer(tensor_name));
}

static int ggml_kimi_a2_expected_threads(void) {
    const char * env = getenv("GGML_MOE_A2_THREADS");
    if (!env || !env[0]) {
        return 16;
    }
    char * end = NULL;
    const long value = strtol(env, &end, 10);
    return end != env && *end == '\0' && value > 0 && value <= 256 ? (int) value : -1;
}

static int ggml_kimi_a2_role_id(const char * name) {
    if (!name) {
        return -1;
    }
    if (strstr(name, ".ffn_up_exps.")) {
        return 0;
    }
    if (strstr(name, ".ffn_gate_exps.")) {
        return 1;
    }
    if (strstr(name, ".ffn_down_exps.")) {
        return 2;
    }
    return -1;
}

static const char * ggml_kimi_a2_role_name(int role) {
    static const char * names[3] = {"up", "gate", "down"};
    return role >= 0 && role < 3 ? names[role] : "invalid";
}

static const char * ggml_kimi_a2_mode_name(enum ggml_kimi_a2_call_mode mode) {
    static const char * names[3] = {"triplet", "replay", "down_reservoir"};
    return mode >= GGML_KIMI_A2_CALL_TRIPLET && mode <= GGML_KIMI_A2_CALL_DOWN ?
        names[mode] : "invalid";
}

static uint64_t ggml_kimi_a2_route_hash(
        const int64_t * matrix_row_counts,
        const struct mmid_row_mapping * matrix_rows,
        int64_t n_as,
        int64_t rows_stride) {
    uint64_t hash = 1469598103934665603ULL;
    for (int64_t expert = 0; expert < n_as; ++expert) {
        for (int64_t row = 0; row < matrix_row_counts[expert]; ++row) {
            const struct mmid_row_mapping mapping = matrix_rows[expert * rows_stride + row];
            const uint32_t values[3] = {
                (uint32_t) expert, (uint32_t) mapping.i1, (uint32_t) mapping.i2,
            };
            for (int value = 0; value < 3; ++value) {
                for (int byte = 0; byte < 4; ++byte) {
                    hash ^= (uint8_t) (values[value] >> (byte * 8));
                    hash *= 1099511628211ULL;
                }
            }
        }
    }
    return hash;
}

static bool ggml_kimi_a2_aligned_alloc(void ** ptr, size_t bytes) {
    if (!ptr || bytes == 0 || bytes > SIZE_MAX - 4095) {
        return false;
    }
    const size_t padded = (bytes + 4095) & ~(size_t) 4095;
    void * memory = NULL;
    if (posix_memalign(&memory, 4096, padded) != 0 || !memory) {
        return false;
    }
    memset(memory, 0, padded);
    *ptr = memory;
    return true;
}

static void ggml_kimi_a2_generation_release(void) {
    free(ggml_kimi_a2_generation.companion_output);
    free(ggml_kimi_a2_generation.down_weights);
    memset(&ggml_kimi_a2_generation, 0, sizeof(ggml_kimi_a2_generation));
}

static bool ggml_kimi_a2_generation_begin(
        int layer,
        int role,
        int prompt_tokens,
        int routes,
        uint64_t route_hash,
        size_t slice_bytes,
        size_t companion_output_bytes,
        const int64_t * matrix_row_counts,
        int64_t n_as) {
    struct ggml_kimi_a2_generation_state * generation = &ggml_kimi_a2_generation;
    if (generation->active || (role != 0 && role != 1) ||
            n_as > GGML_KIMI_A2_MAX_EXPERTS || slice_bytes == 0 ||
            companion_output_bytes == 0) {
        return false;
    }
    memset(generation, 0, sizeof(*generation));
    for (int i = 0; i < GGML_KIMI_A2_MAX_EXPERTS; ++i) {
        generation->expert_positions[i] = -1;
    }
    generation->layer = layer;
    generation->prompt_tokens = prompt_tokens;
    generation->routes = routes;
    generation->first_role = role;
    generation->companion_role = 1 - role;
    generation->route_hash = route_hash;
    generation->generation = ++ggml_kimi_a2_next_generation;
    generation->slice_bytes = slice_bytes;
    generation->companion_output_bytes = companion_output_bytes;
    for (int64_t expert = 0; expert < n_as; ++expert) {
        if (matrix_row_counts[expert] <= 0) {
            continue;
        }
        if (generation->n_experts >= GGML_KIMI_A2_MAX_EXPERTS) {
            ggml_kimi_a2_generation_release();
            return false;
        }
        const int position = generation->n_experts++;
        generation->expert_ids[position] = (int32_t) expert;
        generation->expert_positions[expert] = position;
    }
    if (generation->n_experts <= 0 ||
            (size_t) generation->n_experts > SIZE_MAX / slice_bytes) {
        ggml_kimi_a2_generation_release();
        return false;
    }
    generation->down_weight_bytes = (size_t) generation->n_experts * slice_bytes;
    if (!ggml_kimi_a2_aligned_alloc(
                &generation->companion_output, generation->companion_output_bytes) ||
            !ggml_kimi_a2_aligned_alloc(
                &generation->down_weights, generation->down_weight_bytes)) {
        ggml_kimi_a2_generation_release();
        return false;
    }
    generation->active = true;
    const uint64_t generation_bytes =
        (uint64_t) generation->companion_output_bytes + generation->down_weight_bytes;
    ggml_kimi_a2_peak_generation_bytes = MAX(
        ggml_kimi_a2_peak_generation_bytes, generation_bytes);
    return true;
}

static bool ggml_kimi_a2_generation_matches(
        int layer,
        int role,
        int prompt_tokens,
        int routes,
        int unique_experts,
        uint64_t route_hash,
        size_t slice_bytes,
        size_t output_bytes,
        const int64_t * matrix_row_counts,
        int64_t n_as) {
    const struct ggml_kimi_a2_generation_state * generation = &ggml_kimi_a2_generation;
    if (!generation->active || generation->layer != layer ||
            generation->prompt_tokens != prompt_tokens || generation->routes != routes ||
            generation->n_experts != unique_experts || generation->route_hash != route_hash ||
            generation->slice_bytes != slice_bytes) {
        return false;
    }
    if (role == generation->companion_role && output_bytes != generation->companion_output_bytes) {
        return false;
    }
    int position = 0;
    for (int64_t expert = 0; expert < n_as; ++expert) {
        if (matrix_row_counts[expert] <= 0) {
            continue;
        }
        if (position >= generation->n_experts ||
                generation->expert_ids[position] != expert ||
                generation->expert_positions[expert] != position) {
            return false;
        }
        ++position;
    }
    return position == generation->n_experts;
}

static enum ggml_kimi_a2_call_mode ggml_kimi_a2_call_mode_for_role(int role) {
    const struct ggml_kimi_a2_generation_state * generation = &ggml_kimi_a2_generation;
    if (!generation->active) {
        return role == 0 || role == 1 ? GGML_KIMI_A2_CALL_TRIPLET : GGML_KIMI_A2_CALL_INVALID;
    }
    const unsigned first_mask = 1U << generation->first_role;
    const unsigned companion_mask = 1U << generation->companion_role;
    if (role == generation->companion_role && generation->consumed_role_mask == first_mask) {
        return GGML_KIMI_A2_CALL_REPLAY;
    }
    if (role == 2 && generation->consumed_role_mask == (first_mask | companion_mask)) {
        return GGML_KIMI_A2_CALL_DOWN;
    }
    return GGML_KIMI_A2_CALL_INVALID;
}

static void ggml_kimi_a2_report(void) {
    if (ggml_kimi_a2_generation.active || ggml_kimi_a2_role_calls[0] ||
            ggml_kimi_a2_role_calls[1] || ggml_kimi_a2_role_calls[2]) {
        fprintf(stderr,
                "[kimi_a2] summary role_calls_up=%" PRIu64 " role_calls_gate=%" PRIu64
                " role_calls_down=%" PRIu64 " triplets=%" PRIu64
                " logical_bytes=%" PRIu64 " physical_bytes=%" PRIu64
                " iouring_reads=%" PRIu64 " peak_generation_bytes=%" PRIu64
                " incomplete_generation=%d\n",
                ggml_kimi_a2_role_calls[0], ggml_kimi_a2_role_calls[1],
                ggml_kimi_a2_role_calls[2], ggml_kimi_a2_triplets,
                ggml_kimi_a2_total_logical_bytes, ggml_kimi_a2_total_physical_bytes,
                ggml_kimi_a2_total_reads, ggml_kimi_a2_peak_generation_bytes,
                ggml_kimi_a2_generation.active ? 1 : 0);
    }
}

static bool ggml_kimi_a2_wave_ready(
        void * opaque,
        int wave_index,
        const ggml_moe_cpu_bank_result * result) {
    struct ggml_kimi_a2_pipeline_state * state = (struct ggml_kimi_a2_pipeline_state *) opaque;
    if (state != &ggml_kimi_a2_pipeline || !result || wave_index < 0) {
        return false;
    }
    pthread_mutex_lock(&ggml_kimi_a2_state_mu);
    const bool valid = state->active && !state->cancel &&
        wave_index < state->n_waves && !state->wave_ready[wave_index] &&
        result->bank_ptr != NULL && result->bank_index == wave_index % 2 &&
        result->n_experts == state->wave_sizes[wave_index] &&
        result->slot_stride == state->triplet_bytes &&
        result->ring_bytes == 16ULL * state->triplet_bytes &&
        result->logical_bytes == (uint64_t) state->wave_sizes[wave_index] * state->triplet_bytes &&
        result->physical_read_bytes == result->logical_bytes &&
        result->iouring_reads == (uint64_t) state->wave_sizes[wave_index] &&
        result->ram_admissions == 0 && result->vram_admissions == 0 &&
        result->direct_fallbacks == 0;
    if (valid) {
        state->wave_results[wave_index] = *result;
        state->wave_ready[wave_index] = true;
    } else {
        state->cancel = true;
    }
    pthread_cond_broadcast(&ggml_kimi_a2_state_cv);
    pthread_mutex_unlock(&ggml_kimi_a2_state_mu);
    return valid;
}

static bool ggml_kimi_a2_wave_released(void * opaque, int wave_index, bool wait) {
    struct ggml_kimi_a2_pipeline_state * state = (struct ggml_kimi_a2_pipeline_state *) opaque;
    if (state != &ggml_kimi_a2_pipeline || wave_index < 0) {
        return false;
    }
    pthread_mutex_lock(&ggml_kimi_a2_state_mu);
    if (wave_index >= state->n_waves) {
        pthread_mutex_unlock(&ggml_kimi_a2_state_mu);
        return false;
    }
    while (wait && !state->wave_released[wave_index] && !state->cancel) {
        pthread_cond_wait(&ggml_kimi_a2_state_cv, &ggml_kimi_a2_state_mu);
    }
    const bool released = state->wave_released[wave_index] && !state->cancel;
    pthread_mutex_unlock(&ggml_kimi_a2_state_mu);
    return released;
}

static thread_ret_t ggml_kimi_a2_provider_main(void * opaque) {
    struct ggml_kimi_a2_pipeline_state * state = (struct ggml_kimi_a2_pipeline_state *) opaque;
    const bool ok = state == &ggml_kimi_a2_pipeline && ggml_cuda_moe_a2_read_pipeline &&
        ggml_cuda_moe_a2_read_pipeline(
            state->tensor_name,
            state->expert_ids,
            state->wave_sizes,
            state->n_waves,
            state->slice_bytes,
            state,
            ggml_kimi_a2_wave_ready,
            ggml_kimi_a2_wave_released);
    pthread_mutex_lock(&ggml_kimi_a2_state_mu);
    state->provider_ok = ok;
    state->provider_done = true;
    pthread_cond_broadcast(&ggml_kimi_a2_state_cv);
    pthread_mutex_unlock(&ggml_kimi_a2_state_mu);
    return NULL;
}

static bool ggml_kimi_a2_start(const char * tensor_name) {
    struct ggml_kimi_a2_pipeline_state * state = &ggml_kimi_a2_pipeline;
    const struct ggml_kimi_a2_generation_state * generation = &ggml_kimi_a2_generation;
    pthread_mutex_lock(&ggml_kimi_a2_state_mu);
    memset(state, 0, sizeof(*state));
    state->active = true;
    state->call_ok = true;
    state->slice_bytes = generation->slice_bytes;
    state->triplet_bytes = 3 * generation->slice_bytes;
    state->n_experts = generation->n_experts;
    memcpy(state->expert_ids, generation->expert_ids,
            (size_t) generation->n_experts * sizeof(generation->expert_ids[0]));
    snprintf(state->tensor_name, sizeof(state->tensor_name), "%s", tensor_name);
    for (int first = 0; state->call_ok && first < state->n_experts; first += 8) {
        if (state->n_waves >= GGML_KIMI_A2_MAX_WAVES) {
            state->call_ok = false;
            break;
        }
        state->wave_first[state->n_waves] = first;
        state->wave_sizes[state->n_waves] = (int32_t) MIN(8, state->n_experts - first);
        ++state->n_waves;
    }
    state->call_ok = state->call_ok && state->n_experts > 0 && state->n_waves > 0;
    pthread_mutex_unlock(&ggml_kimi_a2_state_mu);
    if (!state->call_ok || pthread_create(
            &state->provider_thread, NULL, ggml_kimi_a2_provider_main, state) != 0) {
        pthread_mutex_lock(&ggml_kimi_a2_state_mu);
        state->call_ok = false;
        state->active = false;
        pthread_mutex_unlock(&ggml_kimi_a2_state_mu);
        return false;
    }
    return true;
}

static bool ggml_kimi_a2_wait_wave(int wave_index) {
    struct ggml_kimi_a2_pipeline_state * state = &ggml_kimi_a2_pipeline;
    pthread_mutex_lock(&ggml_kimi_a2_state_mu);
    while (!state->wave_ready[wave_index] && !state->provider_done && !state->cancel) {
        pthread_cond_wait(&ggml_kimi_a2_state_cv, &ggml_kimi_a2_state_mu);
    }
    const bool ok = state->wave_ready[wave_index] && !state->cancel;
    state->call_ok = state->call_ok && ok;
    pthread_mutex_unlock(&ggml_kimi_a2_state_mu);
    return ok;
}

static void ggml_kimi_a2_release_wave(int wave_index) {
    struct ggml_kimi_a2_pipeline_state * state = &ggml_kimi_a2_pipeline;
    pthread_mutex_lock(&ggml_kimi_a2_state_mu);
    state->wave_released[wave_index] = true;
    pthread_cond_broadcast(&ggml_kimi_a2_state_cv);
    pthread_mutex_unlock(&ggml_kimi_a2_state_mu);
}

static bool ggml_kimi_a2_finish(void) {
    struct ggml_kimi_a2_pipeline_state * state = &ggml_kimi_a2_pipeline;
    pthread_mutex_lock(&ggml_kimi_a2_state_mu);
    if (!state->call_ok) {
        state->cancel = true;
        pthread_cond_broadcast(&ggml_kimi_a2_state_cv);
    }
    pthread_mutex_unlock(&ggml_kimi_a2_state_mu);
    pthread_join(state->provider_thread, NULL);
    pthread_mutex_lock(&ggml_kimi_a2_state_mu);
    const bool ok = state->call_ok && state->provider_done && state->provider_ok && !state->cancel;
    state->active = false;
    pthread_mutex_unlock(&ggml_kimi_a2_state_mu);
    return ok;
}

static void ggml_kimi_a2_trace(
        enum ggml_kimi_a2_call_mode mode,
        const char * tensor_name,
        int layer,
        int role,
        int prompt_tokens,
        int routes,
        int unique_experts,
        uint64_t route_hash,
        int waves,
        uint64_t logical_bytes,
        uint64_t physical_bytes,
        uint64_t provider_us,
        uint64_t kernel_us,
        uint64_t wall_us,
        uint64_t iouring_reads,
        uint64_t ram_admissions,
        uint64_t vram_admissions,
        uint64_t direct_fallbacks,
        size_t ring_bytes) {
    const char * path = getenv("GGML_MOE_A2_TRACE_OUT");
    const struct ggml_kimi_a2_generation_state * generation = &ggml_kimi_a2_generation;
    if (!path || !path[0]) {
        return;
    }
    pthread_mutex_lock(&ggml_kimi_a2_trace_mu);
    FILE * file = fopen(path, "a");
    if (file) {
        if (!ggml_kimi_a2_trace_header) {
            fprintf(file,
                    "seq,status,mode,layer,role,tensor,prompt_tokens,routes,unique_experts,route_hash,waves,"
                    "logical_bytes,physical_bytes,provider_us,kernel_us,wall_us,iouring_reads,"
                    "ram_admissions,vram_admissions,direct_fallbacks,ring_bytes,"
                    "triplet_generation,triplet_role_mask,triplet_logical_bytes,triplet_physical_bytes,"
                    "companion_output_bytes,down_reservoir_bytes\n");
            ggml_kimi_a2_trace_header = true;
        }
        fprintf(file,
                "%" PRIu64 ",accepted,%s,%d,%s,%s,%d,%d,%d,%016" PRIx64 ",%d,"
                "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
                "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%zu,"
                "%" PRIu64 ",%u,%" PRIu64 ",%" PRIu64 ",%zu,%zu\n",
                ++ggml_kimi_a2_trace_seq, ggml_kimi_a2_mode_name(mode), layer,
                ggml_kimi_a2_role_name(role), tensor_name, prompt_tokens, routes,
                unique_experts, route_hash, waves, logical_bytes, physical_bytes,
                provider_us, kernel_us, wall_us, iouring_reads, ram_admissions,
                vram_admissions, direct_fallbacks, ring_bytes, generation->generation,
                generation->consumed_role_mask, generation->logical_bytes,
                generation->physical_bytes, generation->companion_output_bytes,
                generation->down_weight_bytes);
        fclose(file);
    }
    pthread_mutex_unlock(&ggml_kimi_a2_trace_mu);
}

static void ggml_kimi_a2_compute_expert(
        const struct ggml_compute_params * params,
        struct ggml_tensor * target_dst,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * ids,
        int64_t cur_a,
        const int64_t * matrix_row_counts,
        const struct mmid_row_mapping * matrix_rows,
        char (*atomic_current_chunk)[CACHE_LINE_SIZE],
        const void * src0_cur,
        size_t row_size,
        bool src1_cont,
        const void * wdata) {
    const int ith = params->ith;
    const int nth = params->nth;
    const int64_t nr0 = src0->ne[1];
    const int64_t nr1 = matrix_row_counts[cur_a];
    int chunk_size = (nr0 == 1 || nr1 == 1) ? 64 : 16;
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;
    if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        nchunk0 = nr0 > nr1 ? nth : 1;
        nchunk1 = nr0 > nr1 ? 1 : nth;
    }
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;
    int current_chunk = ith;
    atomic_int * current_chunk_ctr = (atomic_int *) (atomic_current_chunk + cur_a);
    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;
        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);
        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);
        ggml_compute_forward_mul_mat_id_one_chunk(
            target_dst, src0, src1, ids, cur_a,
            ir0_start, ir0_end, ir1_start, ir1_end,
            src0_cur, matrix_rows, row_size, src1_cont, wdata);
        if (nth >= nchunk0 * nchunk1) {
            break;
        }
        current_chunk = atomic_fetch_add_explicit(
            current_chunk_ctr, 1, memory_order_relaxed);
    }
}

static bool ggml_kimi_a2_grouped_prefill(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * ids,
        int64_t * matrix_row_counts,
        struct mmid_row_mapping * matrix_rows,
        char (*atomic_current_chunk)[CACHE_LINE_SIZE],
        bool src1_cont) {
    if (!ggml_kimi_a2_enabled(src0->name) || ids->ne[1] <= 1) {
        return false;
    }
    const int role = ggml_kimi_a2_role_id(src0->name);
    if (role < 0) {
        return false;
    }
    const int ith = params->ith;
    const int nth = params->nth;
    const int expected_threads = ggml_kimi_a2_expected_threads();
    const int layer = ggml_ds4_grouped_retained_parse_layer(src0->name);
    const int64_t n_as = src0->ne[2];
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const size_t slice_bytes = (size_t) ne01 * src0->nb[1];
    const size_t output_bytes = ggml_nbytes(dst);
    const int64_t rows_stride = ids->ne[0] * ids->ne[1];
    int routes = 0;
    int unique_experts = 0;
    for (int64_t expert = 0; expert < n_as; ++expert) {
        if (matrix_row_counts[expert] > 0) {
            ++unique_experts;
            routes += (int) matrix_row_counts[expert];
        }
    }
    const bool shape_ok = src0->type == GGML_TYPE_Q2_K && src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 && src1_cont && expected_threads == nth &&
        layer >= 1 && layer <= 92 && n_as == 896 && ids->ne[0] == 16 && ids->ne[1] > 1 &&
        ((role < 2 && ne00 == 3584 && ne01 == 3072) ||
         (role == 2 && ne00 == 3072 && ne01 == 3584)) &&
        slice_bytes == 3612672 && routes == rows_stride && unique_experts > 0 &&
        unique_experts <= GGML_KIMI_A2_MAX_EXPERTS && ggml_cuda_moe_a2_read_pipeline;
    if (!shape_ok) {
        if (ith == 0) {
            fprintf(stderr,
                    "[kimi_a2] fatal eligibility mismatch tensor=%s role=%s layer=%d type=%d "
                    "threads=%d expected_threads=%d n_as=%" PRId64 " ids=%" PRId64 "x%" PRId64
                    " shape=%" PRId64 "x%" PRId64 " slice_bytes=%zu routes=%d unique=%d src1_cont=%d\n",
                    src0->name, ggml_kimi_a2_role_name(role), layer, src0->type,
                    nth, expected_threads, n_as, ids->ne[0], ids->ne[1], ne00, ne01,
                    slice_bytes, routes, unique_experts, src1_cont ? 1 : 0);
        }
        GGML_ABORT("A2 grouped prefill eligibility invariant failed");
    }

    const uint64_t route_hash = ggml_kimi_a2_route_hash(
        matrix_row_counts, matrix_rows, n_as, rows_stride);
    const uint64_t wall_start_us = ith == 0 ? ggml_time_us() : 0;
    if (ith == 0) {
        pthread_mutex_lock(&ggml_kimi_a2_run_mu);
    }
    ggml_barrier(params->threadpool);

    const enum ggml_kimi_a2_call_mode mode = ggml_kimi_a2_call_mode_for_role(role);
    const bool generation_match = mode == GGML_KIMI_A2_CALL_TRIPLET ||
        ggml_kimi_a2_generation_matches(
            layer, role, (int) ids->ne[1], routes, unique_experts, route_hash,
            slice_bytes, output_bytes, matrix_row_counts, n_as);
    if (mode == GGML_KIMI_A2_CALL_INVALID || !generation_match) {
        if (ith == 0) {
            fprintf(stderr,
                    "[kimi_a2] fatal generation mismatch mode=%d layer=%d role=%s tokens=%" PRId64
                    " routes=%d unique=%d route_hash=%016" PRIx64 " active=%d generation=%" PRIu64
                    " consumed_mask=%u\n",
                    mode, layer, ggml_kimi_a2_role_name(role), ids->ne[1], routes,
                    unique_experts, route_hash, ggml_kimi_a2_generation.active ? 1 : 0,
                    ggml_kimi_a2_generation.generation,
                    ggml_kimi_a2_generation.consumed_role_mask);
        }
        GGML_ABORT("A2 triplet role/route generation mismatch");
    }

    if (mode == GGML_KIMI_A2_CALL_REPLAY) {
        const size_t begin = output_bytes * (size_t) ith / (size_t) nth;
        const size_t end = output_bytes * (size_t) (ith + 1) / (size_t) nth;
        memcpy((char *) dst->data + begin,
               (const char *) ggml_kimi_a2_generation.companion_output + begin,
               end - begin);
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            ggml_kimi_a2_generation.consumed_role_mask |= 1U << role;
            ++ggml_kimi_a2_role_calls[role];
            ggml_kimi_a2_trace(
                mode, src0->name, layer, role, (int) ids->ne[1], routes,
                unique_experts, route_hash, 0, 0, 0, 0, 0,
                ggml_time_us() - wall_start_us, 0, 0, 0, 0, 0);
            free(ggml_kimi_a2_generation.companion_output);
            ggml_kimi_a2_generation.companion_output = NULL;
            ggml_kimi_a2_generation.companion_output_bytes = 0;
            memset(matrix_row_counts, 0, (size_t) n_as * sizeof(int64_t));
        }
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            pthread_mutex_unlock(&ggml_kimi_a2_run_mu);
        }
        ggml_barrier(params->threadpool);
        return true;
    }

    const enum ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
    const void * wdata = src1->type == vec_dot_type ? src1->data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne00);

    if (mode == GGML_KIMI_A2_CALL_DOWN) {
        const uint64_t kernel_start_us = ith == 0 ? ggml_time_us() : 0;
        for (int position = 0; position < ggml_kimi_a2_generation.n_experts; ++position) {
            const int64_t cur_a = ggml_kimi_a2_generation.expert_ids[position];
            const void * src0_cur = (const char *) ggml_kimi_a2_generation.down_weights +
                (size_t) position * slice_bytes;
            ggml_kimi_a2_compute_expert(
                params, dst, src0, src1, ids, cur_a, matrix_row_counts,
                matrix_rows, atomic_current_chunk, src0_cur, row_size, src1_cont, wdata);
        }
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            const uint64_t kernel_us = ggml_time_us() - kernel_start_us;
            ggml_kimi_a2_generation.consumed_role_mask |= 1U << role;
            const uint64_t expected_bytes =
                (uint64_t) unique_experts * 3ULL * slice_bytes;
            if (ggml_kimi_a2_generation.consumed_role_mask != 7U ||
                    ggml_kimi_a2_generation.logical_bytes != expected_bytes ||
                    ggml_kimi_a2_generation.physical_bytes != expected_bytes ||
                    ggml_kimi_a2_generation.iouring_reads != (uint64_t) unique_experts) {
                GGML_ABORT("A2 physical triplet union invariant failed");
            }
            ++ggml_kimi_a2_role_calls[role];
            ++ggml_kimi_a2_triplets;
            ggml_kimi_a2_trace(
                mode, src0->name, layer, role, (int) ids->ne[1], routes,
                unique_experts, route_hash, 0, 0, 0, 0, kernel_us,
                ggml_time_us() - wall_start_us, 0, 0, 0, 0, 0);
            memset(matrix_row_counts, 0, (size_t) n_as * sizeof(int64_t));
            ggml_kimi_a2_generation_release();
        }
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            pthread_mutex_unlock(&ggml_kimi_a2_run_mu);
        }
        ggml_barrier(params->threadpool);
        return true;
    }

    if (ith == 0) {
        const bool begun = ggml_kimi_a2_generation_begin(
            layer, role, (int) ids->ne[1], routes, route_hash, slice_bytes,
            output_bytes, matrix_row_counts, n_as);
        if (!begun || !ggml_kimi_a2_start(src0->name)) {
            GGML_ABORT("A2 triplet generation/provider start failed");
        }
    }
    ggml_barrier(params->threadpool);

    struct ggml_kimi_a2_pipeline_state * state = &ggml_kimi_a2_pipeline;
    struct ggml_tensor companion_dst = *dst;
    companion_dst.data = ggml_kimi_a2_generation.companion_output;
    for (int wave = 0; wave < state->n_waves; ++wave) {
        uint64_t kernel_start_us = 0;
        if (ith == 0) {
            state->call_ok = ggml_kimi_a2_wait_wave(wave);
            kernel_start_us = ggml_time_us();
        }
        ggml_barrier(params->threadpool);
        if (state->call_ok) {
            const ggml_moe_cpu_bank_result * result = &state->wave_results[wave];
            for (int local = 0; local < state->wave_sizes[wave]; ++local) {
                const int expert_pos = state->wave_first[wave] + local;
                const int64_t cur_a = state->expert_ids[expert_pos];
                const char * triplet = (const char *) result->bank_ptr +
                    (size_t) local * result->slot_stride;
                if (ith == 0) {
                    memcpy((char *) ggml_kimi_a2_generation.down_weights +
                               (size_t) expert_pos * slice_bytes,
                           triplet + 2 * slice_bytes, slice_bytes);
                }
                ggml_kimi_a2_compute_expert(
                    params, dst, src0, src1, ids, cur_a, matrix_row_counts,
                    matrix_rows, atomic_current_chunk,
                    triplet + (size_t) role * slice_bytes,
                    row_size, src1_cont, wdata);
                ggml_barrier(params->threadpool);
                if (ith == 0) {
                    atomic_int * current_chunk_ctr =
                        (atomic_int *) (atomic_current_chunk + cur_a);
                    atomic_store_explicit(current_chunk_ctr, nth, memory_order_relaxed);
                }
                ggml_barrier(params->threadpool);
                ggml_kimi_a2_compute_expert(
                    params, &companion_dst, src0, src1, ids, cur_a, matrix_row_counts,
                    matrix_rows, atomic_current_chunk,
                    triplet + (size_t) ggml_kimi_a2_generation.companion_role * slice_bytes,
                    row_size, src1_cont, wdata);
                ggml_barrier(params->threadpool);
            }
        }
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            if (state->call_ok) {
                const ggml_moe_cpu_bank_result * result = &state->wave_results[wave];
                state->logical_bytes += result->logical_bytes;
                state->physical_bytes += result->physical_read_bytes;
                state->provider_us += result->stage_us;
                state->kernel_us += ggml_time_us() - kernel_start_us;
                state->iouring_reads += result->iouring_reads;
                state->ram_admissions += result->ram_admissions;
                state->vram_admissions += result->vram_admissions;
                state->direct_fallbacks += result->direct_fallbacks;
                state->ring_bytes = MAX(state->ring_bytes, result->ring_bytes);
                ggml_kimi_a2_release_wave(wave);
            } else {
                pthread_mutex_lock(&ggml_kimi_a2_state_mu);
                state->cancel = true;
                pthread_cond_broadcast(&ggml_kimi_a2_state_cv);
                pthread_mutex_unlock(&ggml_kimi_a2_state_mu);
            }
        }
        ggml_barrier(params->threadpool);
        if (!state->call_ok) {
            break;
        }
    }

    if (ith == 0) {
        state->call_ok = state->call_ok && ggml_kimi_a2_finish();
        const uint64_t expected_bytes =
            (uint64_t) unique_experts * 3ULL * slice_bytes;
        state->call_ok = state->call_ok &&
            state->logical_bytes == expected_bytes && state->physical_bytes == expected_bytes &&
            state->iouring_reads == (uint64_t) unique_experts &&
            state->ram_admissions == 0 && state->vram_admissions == 0 &&
            state->direct_fallbacks == 0 &&
            state->ring_bytes == 16ULL * 3ULL * slice_bytes;
        if (state->call_ok) {
            ggml_kimi_a2_generation.logical_bytes = state->logical_bytes;
            ggml_kimi_a2_generation.physical_bytes = state->physical_bytes;
            ggml_kimi_a2_generation.iouring_reads = state->iouring_reads;
            ggml_kimi_a2_generation.consumed_role_mask = 1U << role;
            ++ggml_kimi_a2_role_calls[role];
            ggml_kimi_a2_total_logical_bytes += state->logical_bytes;
            ggml_kimi_a2_total_physical_bytes += state->physical_bytes;
            ggml_kimi_a2_total_reads += state->iouring_reads;
            if (!ggml_kimi_a2_report_registered) {
                ggml_kimi_a2_report_registered = true;
                atexit(ggml_kimi_a2_report);
            }
            ggml_kimi_a2_trace(
                mode, src0->name, layer, role, (int) ids->ne[1], routes,
                unique_experts, route_hash, state->n_waves, state->logical_bytes,
                state->physical_bytes, state->provider_us, state->kernel_us,
                ggml_time_us() - wall_start_us, state->iouring_reads,
                state->ram_admissions, state->vram_admissions,
                state->direct_fallbacks, state->ring_bytes);
            memset(matrix_row_counts, 0, (size_t) n_as * sizeof(int64_t));
        }
    }
    ggml_barrier(params->threadpool);
    if (!state->call_ok) {
        GGML_ABORT("A2 grouped prefill triplet provider/compute invariant failed");
    }
    if (ith == 0) {
        pthread_mutex_unlock(&ggml_kimi_a2_run_mu);
    }
    ggml_barrier(params->threadpool);
    return true;
}

#else

static bool ggml_kimi_a2_grouped_prefill(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * ids,
        int64_t * matrix_row_counts,
        struct mmid_row_mapping * matrix_rows,
        char (*atomic_current_chunk)[CACHE_LINE_SIZE],
        bool src1_cont) {
    (void) params;
    (void) dst;
    (void) src0;
    (void) src1;
    (void) ids;
    (void) matrix_row_counts;
    (void) matrix_rows;
    (void) atomic_current_chunk;
    (void) src1_cont;
    return false;
}

#endif

static atomic_uint_fast64_t ggml_kimi_cpu_moe_full_coverage_fast_returns = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t ggml_kimi_cpu_moe_full_coverage_fallback_rebuilds = ATOMIC_VAR_INIT(0);
static atomic_bool ggml_kimi_cpu_moe_full_coverage_registered = ATOMIC_VAR_INIT(false);

static void ggml_kimi_cpu_moe_full_coverage_report(void) {
    fprintf(stderr,
            "[kimi_cpu_moe_full_coverage_fast_path] returns=%" PRIu64
            " fallback_rebuilds=%" PRIu64 "\n",
            (uint64_t) atomic_load_explicit(
                &ggml_kimi_cpu_moe_full_coverage_fast_returns, memory_order_relaxed),
            (uint64_t) atomic_load_explicit(
                &ggml_kimi_cpu_moe_full_coverage_fallback_rebuilds, memory_order_relaxed));
}

static bool ggml_kimi_cpu_moe_full_coverage_fast_path_enabled(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * ids,
        const struct ggml_tensor * dst) {
    const char * env = getenv("GGML_KIMI_CPU_MOE_FULL_COVERAGE_FAST_PATH");
    return env != NULL && env[0] != '\0' && env[0] != '0' &&
        src0 != NULL && src1 != NULL && ids != NULL && dst != NULL &&
        src0->type == GGML_TYPE_IQ1_S && src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 && src0->ne[2] == 896 && src0->ne[3] == 1 &&
        ids->ne[1] == 1 && (size_t) src0->ne[1] * src0->nb[1] == 2150400;
}

static void ggml_kimi_cpu_moe_full_coverage_record(bool fast_return) {
    if (!atomic_exchange_explicit(
            &ggml_kimi_cpu_moe_full_coverage_registered,
            true, memory_order_relaxed)) {
        atexit(ggml_kimi_cpu_moe_full_coverage_report);
    }
    atomic_fetch_add_explicit(
        fast_return ? &ggml_kimi_cpu_moe_full_coverage_fast_returns :
                      &ggml_kimi_cpu_moe_full_coverage_fallback_rebuilds,
        1, memory_order_relaxed);
}

static inline void ggml_kimi_cpu_moe_quantize_src1(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src1,
        enum ggml_type vec_dot_type,
        ggml_from_float_t from_float,
        int64_t ne10,
        int64_t ne11,
        int64_t ne12,
        int64_t ne13,
        size_t nb10,
        size_t nb11,
        size_t nb12,
        size_t nb13) {
    char * wdata = params->wdata;
    const int ith = params->ith;
    const int nth = params->nth;
    const size_t nbw0 = ggml_type_size(vec_dot_type);
    const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
    const size_t nbw2 = nbw1*ne11;
    const size_t nbw3 = nbw2*ne12;

    assert(params->wsize >= ne13*nbw3);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t i11 = 0; i11 < ne11; ++i11) {
                const size_t bs = ggml_blck_size(vec_dot_type);
                const int64_t ne10_block_start = (ith * ne10/bs) / nth;
                const int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                from_float(
                    (float *) ((char *) src1->data + i13*nb13 + i12*nb12 +
                               i11*nb11 + ne10_block_start*bs*nb10),
                    (void *) (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 +
                              ne10_block_start*nbw0),
                    (ne10_block_end - ne10_block_start) * bs);
            }
        }
    }
}

static void ggml_compute_forward_mul_mat_id_kimi(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;
    const bool kimi_cpu_moe_profile = ggml_kimi_cpu_moe_profile_enabled();
    const bool ds4_sparse_fused_mmvq_membership = ggml_ds4_sparse_fused_mmvq_membership_enabled();
    const uint64_t kimi_cpu_moe_total_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
    uint64_t kimi_cpu_moe_cuda_batch_this_us = 0;
    uint64_t kimi_cpu_moe_cuda_single_this_us = 0;

    const enum ggml_type type = src0->type;

    const bool src1_cont = ggml_is_contiguous(src1);

    enum ggml_type    const vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t const from_float      = type_traits_cpu[vec_dot_type].from_float;
    const bool kimi_cpu_moe_full_coverage_fast_path =
        ggml_kimi_cpu_moe_full_coverage_fast_path_enabled(src0, src1, ids, dst);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    // The Kimi-specialized bridge runs instead of the generic matmul-id
    // bridge, so it must make the same early adapter selection before its
    // first availability/cache probe.
    if (ggml_cuda_moe_adapter_select_model_shape) {
        ggml_cuda_moe_adapter_select_model_shape(n_as, (size_t) ne01 * nb01);
    }

    void * wdata_cur = params->wdata;

    if (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), sizeof(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[CACHE_LINE_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

    const void ** fallback_pack_mmap_ptrs =
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(void *), sizeof(void *));

    uint64_t * fallback_touch_us =
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(uint64_t), sizeof(uint64_t));

    int64_t * gpu_tail_row_counts =
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * gpu_tail_matrix_rows =
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

    const uint64_t kimi_cpu_moe_convert_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
    if (kimi_cpu_moe_profile && ith == 0) {
        ggml_kimi_cpu_moe_profile.down.setup_us +=
            kimi_cpu_moe_convert_start - kimi_cpu_moe_total_start;
    }
    if (!kimi_cpu_moe_full_coverage_fast_path && src1->type != vec_dot_type) {
        ggml_kimi_cpu_moe_quantize_src1(
            params, src1, vec_dot_type, from_float,
            ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13);
    }
    if (kimi_cpu_moe_profile && ith == 0) {
        ggml_kimi_cpu_moe_profile.down.convert_us += ggml_time_us() - kimi_cpu_moe_convert_start;
    }

    if (ith == 0) {
        const uint64_t kimi_cpu_moe_route_start = kimi_cpu_moe_profile ? ggml_time_us() : 0;
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));
        const int keep_topk_updown = ggml_moe_keep_topk_for_tensor(src0->name);
        const bool prune_updown = keep_topk_updown > 0;

        // group rows by src0 matrix
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                if (i02 == -1) {
                    memset((char *) dst->data + id * nb1 + iid1 * nb2, 0, (size_t) ne01 * sizeof(float));
                    continue;
                }
                assert(i02 >= 0 && i02 < n_as);

                if (prune_updown && id >= keep_topk_updown) {
                    memset((char *) dst->data + id * nb1 + iid1 * nb2, 0, (size_t) ne01 * sizeof(float));
                    continue;
                }

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
        if (kimi_cpu_moe_profile) {
            ggml_kimi_cpu_moe_profile.down.route_us += ggml_time_us() - kimi_cpu_moe_route_start;
        }
        ggml_ds4_grouped_retained_route_profile_record(
                src0->name,
                src0->type,
                src0->data,
                nb02,
                ids->ne[1] > 1,
                matrix_row_counts,
                n_as,
                (size_t)ne01 * nb01);
        ggml_ds4_grouped_retained_handoff_record_down(
                src0->name,
                src1->data,
                ne10,
                ne01,
                matrix_row_counts,
                n_as);
    }

    const uint64_t kimi_cpu_moe_route_tail_start =
        (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;

    // reset current_chunk
    if (!kimi_cpu_moe_full_coverage_fast_path) {
        for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
            atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
            *current_chunk_ctr = nth;
        }
    }

    const uint64_t kimi_cpu_moe_route_barrier_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
    if (kimi_cpu_moe_profile && ith == 0) {
        ggml_kimi_cpu_moe_profile.down.route_tail_us +=
            kimi_cpu_moe_route_barrier_start - kimi_cpu_moe_route_tail_start;
    }
    ggml_barrier(params->threadpool);
    if (kimi_cpu_moe_profile && ith == 0) {
        ggml_kimi_cpu_moe_profile.down.route_barrier_us += ggml_time_us() - kimi_cpu_moe_route_barrier_start;
    }

    const uint64_t kimi_cpu_moe_eligibility_start =
        (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;

    if (ggml_kimi_a2_grouped_prefill(
            params, dst, src0, src1, ids,
            matrix_row_counts, matrix_rows, atomic_current_chunk, src1_cont)) {
        return;
    }

    enum ggml_kimi_cpu_moe_eligibility_reason kimi_cpu_moe_batch_reason = GGML_KIMI_CPU_MOE_ELIGIBLE;
    if (!ggml_moe_gpu_provider_enabled(ids)) {
        kimi_cpu_moe_batch_reason = GGML_KIMI_CPU_MOE_INELIG_ENV;
    } else if (getenv("GGML_MOE_STREAM_DOWN_BATCH") == NULL) {
        kimi_cpu_moe_batch_reason = GGML_KIMI_CPU_MOE_INELIG_ENV;
    } else if (!ggml_cuda_moe_stream_batch) {
        kimi_cpu_moe_batch_reason = GGML_KIMI_CPU_MOE_INELIG_BATCH_FN;
    } else if (!ggml_cuda_moe_stream_available) {
        kimi_cpu_moe_batch_reason = GGML_KIMI_CPU_MOE_INELIG_AVAILABLE_FN;
    } else if (!ggml_cuda_moe_stream_available()) {
        kimi_cpu_moe_batch_reason = GGML_KIMI_CPU_MOE_INELIG_AVAILABLE_FALSE;
    } else if (src0->type == GGML_TYPE_Q4_0 && ids->ne[1] > 1 &&
            !(getenv("GGML_MOE_Q4_PROMPT_DOWN_BATCH") != NULL &&
              getenv("GGML_MOE_Q4_PROMPT_DOWN_BATCH")[0] != '0')) {
        kimi_cpu_moe_batch_reason = GGML_KIMI_CPU_MOE_INELIG_UNSUPPORTED;
    } else if (!ggml_cuda_moe_stream_supports_down_batch(src0->type, src0->name)) {
        kimi_cpu_moe_batch_reason = GGML_KIMI_CPU_MOE_INELIG_UNSUPPORTED;
    } else if (src1->type != GGML_TYPE_F32) {
        kimi_cpu_moe_batch_reason = GGML_KIMI_CPU_MOE_INELIG_SRC1_TYPE;
    } else if (ne13 != 1) {
        kimi_cpu_moe_batch_reason = GGML_KIMI_CPU_MOE_INELIG_NE13;
    } else if (dst->type != GGML_TYPE_F32) {
        kimi_cpu_moe_batch_reason = GGML_KIMI_CPU_MOE_INELIG_DST_TYPE;
    }

    if (kimi_cpu_moe_profile && ith == 0) {
        ggml_kimi_cpu_moe_name_profile_record_eligibility(
                src0->name, src0->type, ids->ne[1] > 1, kimi_cpu_moe_batch_reason);
    }

    const bool use_gpu_stream_batch =
        kimi_cpu_moe_batch_reason == GGML_KIMI_CPU_MOE_ELIGIBLE;
    bool kimi_cpu_moe_batch_done = false;
    int64_t kimi_cpu_moe_single_attempts = 0;
    int64_t kimi_cpu_moe_single_accepts = 0;

    if (use_gpu_stream_batch) {
        if (ith == 0) {
            const uint64_t kimi_cpu_moe_cuda_start = kimi_cpu_moe_profile ? ggml_time_us() : 0;
            if (kimi_cpu_moe_profile) {
                ggml_kimi_cpu_moe_profile.down.eligibility_us +=
                    kimi_cpu_moe_cuda_start - kimi_cpu_moe_eligibility_start;
            }
            const int gpu_keep_topk =
                ggml_moe_gpu_keep_topk_applies(src0->name) ? ggml_moe_gpu_keep_topk_updown() : 0;
            const bool split_gpu_cpu_tail =
                gpu_keep_topk > 0 && gpu_keep_topk < ids->ne[0];
            if (split_gpu_cpu_tail) {
                memset(gpu_tail_row_counts, 0, n_as*sizeof(int64_t));
                for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                    const int64_t original_count = matrix_row_counts[cur_a];
                    int64_t gpu_count = 0;
                    int64_t tail_count = 0;
                    for (int64_t k = 0; k < original_count; ++k) {
                        const struct mmid_row_mapping row =
                            matrix_rows[cur_a * ids->ne[0] * ids->ne[1] + k];
                        if (row.i1 < gpu_keep_topk) {
                            matrix_rows[cur_a * ids->ne[0] * ids->ne[1] + gpu_count++] = row;
                        } else {
                            gpu_tail_matrix_rows[cur_a * ids->ne[0] * ids->ne[1] + tail_count++] = row;
                        }
                    }
                    matrix_row_counts[cur_a] = gpu_count;
                    gpu_tail_row_counts[cur_a] = tail_count;
                }
            }
            const void * src1_q8_0_batch = (src1->type == vec_dot_type) ? src1->data : params->wdata;
            const size_t src1_q8_0_row_size_batch = ggml_row_size(vec_dot_type, ne10);
            // The batch-path route-group lease is part of the K3-local
            // adapter.  The common adapter already owns its historical
            // single-path role-cache lease/event schedule; injecting the K3
            // lease here changes Kimi 2.x cache ownership and routing output.
            if (ggml_kimi_uses_k3_local_adapter(n_as, (size_t) ne01 * nb01) &&
                    ggml_cuda_moe_stream_route_group_preload_updown_from_gate &&
                    strstr(src0->name, ".ffn_gate_exps.") != NULL) {
                const int updown_prefetch_jobs = ggml_cuda_moe_stream_route_group_preload_updown_from_gate(
                        src0->name,
                        n_as,
                        matrix_row_counts);
                (void) updown_prefetch_jobs;
            }
            const bool done = ggml_cuda_moe_stream_batch(
                src0->type,
                src0->name,
                src0->data,
                n_as,
                ne01, ne00, nb01, nb02,
                (const float *) src1->data,
                ne11,
                nb11, nb12,
                src1_q8_0_batch,
                src1_q8_0_row_size_batch,
                ne11,
                (float *) dst->data,
                nb1, nb2,
                matrix_row_counts,
                (const ggml_moe_stream_row_mapping *) matrix_rows,
                ids->ne[0]*ids->ne[1]);
            kimi_cpu_moe_batch_done = done;
            const uint64_t kimi_cpu_moe_cuda_end = kimi_cpu_moe_profile ? ggml_time_us() : 0;
            if (kimi_cpu_moe_profile) {
                const uint64_t elapsed_us = kimi_cpu_moe_cuda_end - kimi_cpu_moe_cuda_start;
                ggml_kimi_cpu_moe_profile.down.cuda_batch_us += elapsed_us;
                kimi_cpu_moe_cuda_batch_this_us += elapsed_us;
                if (done) {
                    ggml_kimi_cpu_moe_profile.down.cuda_batch_accepted++;
                } else {
                    ggml_kimi_cpu_moe_profile.down.cuda_batch_declined++;
                }
            }

            const uint64_t kimi_cpu_moe_batch_post_start =
                kimi_cpu_moe_profile ? ggml_time_us() : 0;
            if (kimi_cpu_moe_profile) {
                ggml_kimi_cpu_moe_profile.down.batch_profile_bookkeeping_us +=
                    kimi_cpu_moe_batch_post_start - kimi_cpu_moe_cuda_end;
            }

            if (done) {
                if (ggml_moe_stream_compare_cpu_enabled()) {
                    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                        const int64_t cne1 = matrix_row_counts[cur_a];
                        if (cne1 == 0) {
                            continue;
                        }
                        const char * src0_cur = (const char *) src0->data + cur_a * nb02;
                        ggml_moe_stream_compare_cpu_result(
                            dst, src0, src1, cur_a, cne1,
                            src0_cur,
                            matrix_rows + cur_a * ids->ne[0] * ids->ne[1],
                            ggml_row_size(type_traits_cpu[src0->type].vec_dot_type, ne10),
                            src1_cont,
                            src1->type == type_traits_cpu[src0->type].vec_dot_type ? src1->data : params->wdata);
                    }
                }
                if (split_gpu_cpu_tail) {
                    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                        const int64_t tail_count = gpu_tail_row_counts[cur_a];
                        for (int64_t k = 0; k < tail_count; ++k) {
                            matrix_rows[cur_a * ids->ne[0] * ids->ne[1] + k] =
                                gpu_tail_matrix_rows[cur_a * ids->ne[0] * ids->ne[1] + k];
                        }
                        matrix_row_counts[cur_a] = tail_count;
                    }
                } else {
                    memset(matrix_row_counts, 0, n_as*sizeof(int64_t));
                }
            } else if (split_gpu_cpu_tail) {
                for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                    const int64_t gpu_count = matrix_row_counts[cur_a];
                    const int64_t tail_count = gpu_tail_row_counts[cur_a];
                    for (int64_t k = 0; k < tail_count; ++k) {
                        matrix_rows[cur_a * ids->ne[0] * ids->ne[1] + gpu_count + k] =
                            gpu_tail_matrix_rows[cur_a * ids->ne[0] * ids->ne[1] + k];
                    }
                    matrix_row_counts[cur_a] = gpu_count + tail_count;
                }
            }
            if (kimi_cpu_moe_profile) {
                ggml_kimi_cpu_moe_profile.down.batch_post_us +=
                    ggml_time_us() - kimi_cpu_moe_batch_post_start;
            }
        }

        const uint64_t kimi_cpu_moe_post_cuda_barrier_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
        ggml_barrier(params->threadpool);
        if (kimi_cpu_moe_profile && ith == 0) {
            ggml_kimi_cpu_moe_profile.down.post_cuda_barrier_us += ggml_time_us() - kimi_cpu_moe_post_cuda_barrier_start;
        }

    }

    if (kimi_cpu_moe_full_coverage_fast_path) {
        bool has_residual_rows = false;
        for (int cur_a = 0; cur_a < n_as; ++cur_a) {
            if (matrix_row_counts[cur_a] != 0) {
                has_residual_rows = true;
                break;
            }
        }
        if (!has_residual_rows) {
            if (ith == 0) {
                ggml_kimi_cpu_moe_full_coverage_record(true);
            }
            return;
        }

        if (src1->type != vec_dot_type) {
            ggml_kimi_cpu_moe_quantize_src1(
                params, src1, vec_dot_type, from_float,
                ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13);
        }
        for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
            atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
            *current_chunk_ctr = nth;
        }
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            ggml_kimi_cpu_moe_full_coverage_record(false);
        }
    }

    const uint64_t kimi_cpu_moe_post_cuda_tail_start =
        (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;

    const bool use_gpu_stream =
        !use_gpu_stream_batch &&
        ggml_moe_gpu_provider_enabled(ids) &&
        getenv("GGML_MOE_STREAM_BATCH_ONLY") == NULL &&
        ggml_cuda_moe_stream_one &&
        ggml_cuda_moe_stream_available &&
        ggml_cuda_moe_stream_available() &&
        ggml_cuda_moe_stream_supports_one_type(src0->type) &&
        src1->type == GGML_TYPE_F32 &&
        ne13 == 1 &&
        dst->type == GGML_TYPE_F32;

    if (use_gpu_stream) {
        if (ith == 0) {
            const void * wdata_stream = (src1->type == vec_dot_type) ? src1->data : params->wdata;
            const size_t row_size_stream = ggml_row_size(vec_dot_type, ne10);
            const char * route_group_native_parity_env = getenv("GGML_MOE_ROUTE_GROUP_NATIVE_PARITY");
            const bool route_group_native_parity =
                route_group_native_parity_env && route_group_native_parity_env[0] && route_group_native_parity_env[0] != '0';
            if (ggml_moe_gate_batch_prefetch_enabled() &&
                    !route_group_native_parity &&
                    ggml_cuda_moe_stream_batch_preload_active_from_pack &&
                    src0->name && strstr(src0->name, ".ffn_gate_exps.") != NULL) {
                static bool gate_batch_prefetch_logged = false;
                if (!gate_batch_prefetch_logged) {
                    fprintf(stderr, "[moe_stream_cpu] gate batch prefetch requested: tensor=%s n_as=%" PRId64 " expert_bytes=%zu\n",
                            src0->name, (int64_t) n_as, (size_t) ne01 * nb01);
                    gate_batch_prefetch_logged = true;
                }
                const int gate_prefetch_jobs = ggml_cuda_moe_stream_batch_preload_active_from_pack(
                        src0->type,
                        src0->name,
                        n_as,
                        (size_t) ne01 * nb01,
                        matrix_row_counts);
                (void) gate_prefetch_jobs;
            }
            if (ggml_cuda_moe_stream_route_group_preload_updown_from_gate &&
                    src0->name && strstr(src0->name, ".ffn_gate_exps.") != NULL) {
                const int updown_prefetch_jobs = ggml_cuda_moe_stream_route_group_preload_updown_from_gate(
                        src0->name,
                        n_as,
                        matrix_row_counts);
                (void) updown_prefetch_jobs;
            }
            const char * q4_colibri_pilot_prefetch_env = getenv("GGML_MOE_Q4_COLIBRI_PILOT_PREFETCH");
            if (q4_colibri_pilot_prefetch_env &&
                    q4_colibri_pilot_prefetch_env[0] &&
                    q4_colibri_pilot_prefetch_env[0] != '0' &&
                    ggml_cuda_moe_stream_q4_colibri_pilot_prefetch_from_gate &&
                    src0->name && strstr(src0->name, ".ffn_gate_exps.") != NULL) {
                const int pilot_prefetch_jobs = ggml_cuda_moe_stream_q4_colibri_pilot_prefetch_from_gate(
                        src0->type,
                        src0->name,
                        n_as,
                        (size_t) ne01 * nb01,
                        matrix_row_counts);
                (void) pilot_prefetch_jobs;
            }

            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];

                if (cne1 == 0) {
                    continue;
                }

                const char * src0_cur = (const char *) src0->data + cur_a * nb02;
                const uint64_t kimi_cpu_moe_cuda_start = kimi_cpu_moe_profile ? ggml_time_us() : 0;
                kimi_cpu_moe_single_attempts++;
                const bool done = ggml_cuda_moe_stream_one(
                    src0->type,
                    src0->name,
                    cur_a,
                    src0_cur,
                    ne01, ne00, nb01,
                    (const float *) src1->data,
                    nb11, nb12,
                    ne11,
                    cne1,
                    NULL, 0,
                    (float *) dst->data,
                    nb1, nb2,
                    (const ggml_moe_stream_row_mapping *) (matrix_rows + cur_a * ids->ne[0] * ids->ne[1]));
                if (kimi_cpu_moe_profile) {
                    const uint64_t elapsed_us = ggml_time_us() - kimi_cpu_moe_cuda_start;
                    ggml_kimi_cpu_moe_profile.down.cuda_single_us += elapsed_us;
                    kimi_cpu_moe_cuda_single_this_us += elapsed_us;
                    if (done) {
                        ggml_kimi_cpu_moe_profile.down.cuda_single_accepted++;
                    } else {
                        ggml_kimi_cpu_moe_profile.down.cuda_single_declined++;
                    }
                }

                if (done) {
                    kimi_cpu_moe_single_accepts++;
                    if (ggml_moe_stream_compare_cpu_enabled()) {
                        ggml_moe_stream_compare_cpu_result(
                            dst, src0, src1, cur_a, cne1,
                            src0_cur,
                            matrix_rows + cur_a * ids->ne[0] * ids->ne[1],
                            ggml_row_size(type_traits_cpu[src0->type].vec_dot_type, ne10),
                            src1_cont,
                            src1->type == type_traits_cpu[src0->type].vec_dot_type ? src1->data : params->wdata);
                    }
                    matrix_row_counts[cur_a] = 0;
                }
            }

            if (ggml_cuda_moe_stream_sync) {
                ggml_cuda_moe_stream_sync();
            }
        }

        const uint64_t kimi_cpu_moe_post_cuda_barrier_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
        ggml_barrier(params->threadpool);
        if (kimi_cpu_moe_profile && ith == 0) {
            ggml_kimi_cpu_moe_profile.down.post_cuda_barrier_us += ggml_time_us() - kimi_cpu_moe_post_cuda_barrier_start;
        }
    }

    if (ggml_moe_stream_one_gpu_only_filter_matches(src0->name)) {
        int64_t gpu_only_rows = 0;
        for (int cur_a = 0; cur_a < n_as; ++cur_a) {
            gpu_only_rows += matrix_row_counts[cur_a];
        }
        if (gpu_only_rows > 0) {
            fprintf(stderr,
                    "[moe_stream] GPU-only filter matched but CPU fallback remains: tensor=%s rows=%" PRId64 " batch_reason=%s single_attempts=%" PRId64 " single_accepts=%" PRId64 "\n",
                    src0->name ? src0->name : "", gpu_only_rows,
                    ggml_kimi_cpu_moe_eligibility_reason_name(kimi_cpu_moe_batch_reason),
                    kimi_cpu_moe_single_attempts, kimi_cpu_moe_single_accepts);
            abort();
        }
    }

    if (ggml_moe_cpu_willneed_enabled()) {
        if (ith == 0) {
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                if (matrix_row_counts[cur_a] == 0) {
                    continue;
                }
                ggml_moe_cpu_willneed_pages((const char *) src0->data + cur_a * nb02, (size_t) ne01 * nb01);
            }
        }
        ggml_barrier(params->threadpool);
    }

    if (ith == 0) {
        ggml_kimi_cpu_fallback_pack_mmap_prepare(
                src0->name,
                src0->type,
                ids->ne[1] > 1,
                n_as,
                matrix_row_counts,
                (size_t) ne01 * nb01,
                fallback_pack_mmap_ptrs);
        memset(fallback_touch_us, 0, n_as*sizeof(uint64_t));
        if (ggml_moe_cpu_fallback_touch_profile_enabled()) {
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                if (matrix_row_counts[cur_a] == 0) {
                    continue;
                }
                const char * src0_cur = fallback_pack_mmap_ptrs[cur_a] ?
                    (const char *) fallback_pack_mmap_ptrs[cur_a] :
                    (const char *) src0->data + cur_a * nb02;
                fallback_touch_us[cur_a] =
                    ggml_moe_cpu_touch_pages_us(src0_cur, (size_t) ne01 * nb01);
            }
        }
    }
    ggml_barrier(params->threadpool);

    const bool use_cpu_batch_microprobe =
        ggml_moe_cpu_batch_microprobe_enabled() &&
        src0->type == GGML_TYPE_MXFP4 &&
        src1->type != vec_dot_type &&
        vec_dot_type == GGML_TYPE_Q8_0 &&
        ne13 == 1 &&
        dst->type == GGML_TYPE_F32 &&
        nth <= GGML_MOE_CPU_BATCH_MICROPROBE_MAX_THREADS &&
        ggml_moe_cpu_batch_microprobe_name_allows(src0->name);

    int cpu_batch_microprobe_cur_a = -1;
    int cpu_batch_microprobe_seq = -1;
    if (use_cpu_batch_microprobe) {
        if (ith == 0) {
            int selected = -1;
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                if (matrix_row_counts[cur_a] > 0) {
                    selected = cur_a;
                    break;
                }
            }

            const int seq = atomic_fetch_add_explicit(&ggml_moe_cpu_batch_microprobe_next_seq, 1, memory_order_relaxed);
            const int max_calls = ggml_moe_cpu_batch_microprobe_env_int("GGML_MOE_CPU_BATCH_MICROPROBE_MAX_CALLS", 1);
            if (selected >= 0 && (max_calls <= 0 || seq < max_calls)) {
                atomic_store_explicit(&ggml_moe_cpu_batch_microprobe_active_seq, seq, memory_order_relaxed);
                atomic_store_explicit(&ggml_moe_cpu_batch_microprobe_active_expert, selected, memory_order_relaxed);
            } else {
                atomic_store_explicit(&ggml_moe_cpu_batch_microprobe_active_seq, -1, memory_order_relaxed);
                atomic_store_explicit(&ggml_moe_cpu_batch_microprobe_active_expert, -1, memory_order_relaxed);
            }
        }
        ggml_barrier(params->threadpool);
        cpu_batch_microprobe_seq = atomic_load_explicit(&ggml_moe_cpu_batch_microprobe_active_seq, memory_order_relaxed);
        cpu_batch_microprobe_cur_a = atomic_load_explicit(&ggml_moe_cpu_batch_microprobe_active_expert, memory_order_relaxed);
    }

    if (ggml_moe_stream_q80_skip_enabled() &&
            ggml_cuda_moe_stream_q80_skip &&
            src0->type == GGML_TYPE_MXFP4 &&
            src1->type != vec_dot_type &&
            vec_dot_type == GGML_TYPE_Q8_0 &&
            ne13 == 1 &&
            dst->type == GGML_TYPE_F32) {
        if (ith == 0) {
            const void * q80_base = params->wdata;
            const size_t q80_row_size = ggml_row_size(vec_dot_type, ne10);
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0) {
                    continue;
                }
                const char * src0_cur = fallback_pack_mmap_ptrs[cur_a] ?
                    (const char *) fallback_pack_mmap_ptrs[cur_a] :
                    (const char *) src0->data + cur_a * nb02;
                const bool done = ggml_cuda_moe_stream_q80_skip(
                        src0->type,
                        src0->name,
                        cur_a,
                        src0_cur,
                        ne01, ne00, nb01,
                        q80_base,
                        q80_row_size,
                        ne11,
                        cne1,
                        (float *) dst->data,
                        nb1, nb2,
                        (const ggml_moe_stream_row_mapping *) (matrix_rows + cur_a * ids->ne[0] * ids->ne[1]));
                if (done) {
                    matrix_row_counts[cur_a] = 0;
                }
            }
        }
        ggml_barrier(params->threadpool);
    }

    const bool moe_fallback_reason_profile = ggml_moe_fallback_reason_profile_enabled();
    const uint64_t kimi_cpu_moe_fallback_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
    if (kimi_cpu_moe_profile && ith == 0) {
        ggml_kimi_cpu_moe_profile.down.post_cuda_tail_us +=
            kimi_cpu_moe_fallback_start - kimi_cpu_moe_post_cuda_tail_start;
    }
    const uint64_t moe_fallback_reason_start = (moe_fallback_reason_profile && ith == 0) ? ggml_time_us() : 0;
    const uint64_t ds4_sparse_fused_mmvq_fallback_start =
        (ds4_sparse_fused_mmvq_membership && ith == 0) ? ggml_time_us() : 0;
    if (ith == 0) {
        ggml_kimi_full_coverage_cpu_fallback_record(
            src0, ids, matrix_row_counts, n_as);
    }
    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        const char * src0_cur = fallback_pack_mmap_ptrs[cur_a] ?
            (const char *) fallback_pack_mmap_ptrs[cur_a] :
            (const char *) src0->data + cur_a * nb02;
        const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        const int64_t nr0 = ne01;
        const int64_t nr1 = cne1;

        int chunk_size = 16;
        if (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        if (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        int current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            const bool trace_cpu_chunk = ggml_moe_cpu_chunk_trace_enabled();
            const double trace_t0_ms = trace_cpu_chunk ? ggml_moe_cpu_trace_now_ms() : 0.0;

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            if (trace_cpu_chunk) {
                const double trace_t1_ms = ggml_moe_cpu_trace_now_ms();
                ggml_moe_cpu_chunk_trace_write(
                    ggml_moe_tensor_role(src0->name),
                    src0->name,
                    src0->type,
                    cur_a,
                    ith,
                    nth,
                    cne1,
                    ir0_start,
                    ir0_end,
                    ir1_start,
                    ir1_end,
                    (size_t) ne01 * nb01,
                    trace_t1_ms - trace_t0_ms);
            }

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }

        if (cpu_batch_microprobe_cur_a == cur_a) {
            const int repeats_env = ggml_moe_cpu_batch_microprobe_env_int("GGML_MOE_CPU_BATCH_MICROPROBE_REPEATS", 1);
            const int repeats = repeats_env > 0 ? repeats_env : 1;
            const int64_t max_cols_env = (int64_t) ggml_moe_cpu_batch_microprobe_env_int("GGML_MOE_CPU_BATCH_MICROPROBE_MAX_COLS", 0);
            const int64_t max_rows_env = (int64_t) ggml_moe_cpu_batch_microprobe_env_int("GGML_MOE_CPU_BATCH_MICROPROBE_MAX_ROWS", 1);
            const int64_t probe_cols = max_cols_env > 0 ? MIN(nr0, max_cols_env) : nr0;
            const int64_t probe_rows = max_rows_env > 0 ? MIN(nr1, max_rows_env) : nr1;

            ggml_barrier(params->threadpool);
            if (ith == 0) {
                *current_chunk_ctr = nth;
            }
            if (ith >= 0 && ith < GGML_MOE_CPU_BATCH_MICROPROBE_MAX_THREADS) {
                ggml_moe_cpu_batch_microprobe_dot_calls[ith] = 0;
                ggml_moe_cpu_batch_microprobe_diff_count[ith] = 0;
                ggml_moe_cpu_batch_microprobe_src0_bytes[ith] = 0;
                ggml_moe_cpu_batch_microprobe_q80_bytes[ith] = 0;
                ggml_moe_cpu_batch_microprobe_max_abs[ith] = 0.0;
                ggml_moe_cpu_batch_microprobe_sum_abs[ith] = 0.0;
                ggml_moe_cpu_batch_microprobe_sink[ith] = 0.0;
            }
            ggml_barrier(params->threadpool);

            uint64_t microprobe_start_us = 0;
            if (ith == 0) {
                microprobe_start_us = ggml_time_us();
            }

            current_chunk = ith;
            while (current_chunk < nchunk0 * nchunk1) {
                const int64_t ith0 = current_chunk % nchunk0;
                const int64_t ith1 = current_chunk / nchunk0;

                const int64_t ir0_start_probe = dr0 * ith0;
                const int64_t ir0_end_probe = MIN(ir0_start_probe + dr0, probe_cols);

                const int64_t ir1_start_probe = dr1 * ith1;
                const int64_t ir1_end_probe = MIN(ir1_start_probe + dr1, probe_rows);

                if (ir0_start_probe < ir0_end_probe && ir1_start_probe < ir1_end_probe) {
                    ggml_compute_forward_mul_mat_id_microprobe_one_chunk(
                        dst, src0, src1, ids, cur_a,
                        ir0_start_probe, ir0_end_probe, ir1_start_probe, ir1_end_probe,
                        src0_cur, matrix_rows, row_size, src1_cont, wdata, repeats, ith
                    );
                }

                if (nth >= nchunk0 * nchunk1) {
                    break;
                }

                current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
            }

            ggml_barrier(params->threadpool);

            if (ith == 0) {
                const uint64_t wall_us = ggml_time_us() - microprobe_start_us;
                uint64_t dot_calls = 0;
                uint64_t diff_count = 0;
                uint64_t src0_bytes = 0;
                uint64_t q80_bytes = 0;
                double max_abs = 0.0;
                double sum_abs = 0.0;
                double sink = 0.0;
                for (int it = 0; it < nth && it < GGML_MOE_CPU_BATCH_MICROPROBE_MAX_THREADS; ++it) {
                    dot_calls += ggml_moe_cpu_batch_microprobe_dot_calls[it];
                    diff_count += ggml_moe_cpu_batch_microprobe_diff_count[it];
                    src0_bytes += ggml_moe_cpu_batch_microprobe_src0_bytes[it];
                    q80_bytes += ggml_moe_cpu_batch_microprobe_q80_bytes[it];
                    if (ggml_moe_cpu_batch_microprobe_max_abs[it] > max_abs) {
                        max_abs = ggml_moe_cpu_batch_microprobe_max_abs[it];
                    }
                    sum_abs += ggml_moe_cpu_batch_microprobe_sum_abs[it];
                    sink += ggml_moe_cpu_batch_microprobe_sink[it];
                }
                ggml_moe_cpu_batch_microprobe_write(
                    cpu_batch_microprobe_seq,
                    ggml_moe_tensor_role(src0->name),
                    src0->name,
                    src0->type,
                    cur_a,
                    nth,
                    repeats,
                    cne1,
                    probe_cols,
                    probe_rows,
                    dot_calls,
                    diff_count,
                    src0_bytes,
                    q80_bytes,
                    wall_us,
                    max_abs,
                    sum_abs,
                    sink);
            }
            ggml_barrier(params->threadpool);
        }
    }
    if (moe_fallback_reason_profile && ith == 0) {
        const uint64_t moe_fallback_reason_us = ggml_time_us() - moe_fallback_reason_start;
        int64_t fallback_rows = 0;
        for (int cur_a = 0; cur_a < n_as; ++cur_a) {
            fallback_rows += matrix_row_counts[cur_a];
        }
        const char * batch_reason_name = ggml_kimi_cpu_moe_eligibility_reason_name(kimi_cpu_moe_batch_reason);
        const char * single_reason_name = "not_attempted";
        const char * final_reason_name = "cpu_fallback";
        if (use_gpu_stream_batch) {
            single_reason_name = "not_attempted_batch_path_selected";
            final_reason_name = kimi_cpu_moe_batch_done ? "post_batch_residual" : "batch_declined_internal_no_single_retry";
        } else if (getenv("GGML_MOE_STREAM_BATCH_ONLY") != NULL) {
            single_reason_name = "stream_batch_only";
            final_reason_name = "batch_only_cpu_fallback";
        } else if (!ggml_cuda_moe_stream_one) {
            single_reason_name = "one_fn_missing";
            final_reason_name = "one_fn_missing";
        } else if (!ggml_cuda_moe_stream_available) {
            single_reason_name = "available_fn_missing";
            final_reason_name = "available_fn_missing";
        } else if (!ggml_cuda_moe_stream_available()) {
            single_reason_name = "stream_available_false";
            final_reason_name = "stream_available_false";
        } else if (!ggml_cuda_moe_stream_supports_one_type(src0->type)) {
            single_reason_name = "one_unsupported_type";
            final_reason_name = "one_unsupported_type";
        } else if (src1->type != GGML_TYPE_F32) {
            single_reason_name = "src1_not_f32";
            final_reason_name = "src1_not_f32";
        } else if (ne13 != 1) {
            single_reason_name = "ne13_not1";
            final_reason_name = "ne13_not1";
        } else if (dst->type != GGML_TYPE_F32) {
            single_reason_name = "dst_not_f32";
            final_reason_name = "dst_not_f32";
        } else if (!ggml_moe_stream_one_name_filter_would_allow(src0->name)) {
            single_reason_name = "one_name_filter";
            final_reason_name = "one_name_filter";
        } else {
            single_reason_name = "one_declined_internal";
            final_reason_name = "one_declined_internal";
        }
        if (fallback_rows > 0) {
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0) {
                    continue;
                }
                const uint64_t expert_fallback_us =
                    (uint64_t) (((double) moe_fallback_reason_us * (double) cne1) / (double) fallback_rows);
                ggml_moe_fallback_reason_profile_record(
                        ggml_moe_tensor_role(src0->name),
                        src0->name,
                        src0->type,
                        ids->ne[1] > 1,
                        cur_a,
                        cne1,
                        (size_t) nb02,
                        expert_fallback_us,
                        batch_reason_name,
                        single_reason_name,
                        final_reason_name,
                        use_gpu_stream_batch,
                        kimi_cpu_moe_batch_done,
                        kimi_cpu_moe_single_attempts > 0,
                        false);
                ggml_moe_fallback_source_probe_record(
                        ggml_moe_tensor_role(src0->name),
                        src0->name,
                        ids->ne[1] > 1,
                        cur_a,
                        cne1,
                        expert_fallback_us,
                        batch_reason_name,
                        single_reason_name,
                        final_reason_name,
                        src0,
                        src1,
                        ids,
                        dst);
            }
        }
    }
    if (ds4_sparse_fused_mmvq_membership && ith == 0) {
        const uint64_t ds4_sparse_fused_mmvq_fallback_us =
            ggml_time_us() - ds4_sparse_fused_mmvq_fallback_start;
        int64_t fallback_rows = 0;
        for (int cur_a = 0; cur_a < n_as; ++cur_a) {
            fallback_rows += matrix_row_counts[cur_a];
        }
        if (fallback_rows > 0) {
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0) {
                    continue;
                }
                const uint64_t expert_fallback_us =
                    (uint64_t) (((double) ds4_sparse_fused_mmvq_fallback_us * (double) cne1) / (double) fallback_rows);
                ggml_ds4_sparse_fused_mmvq_membership_record(
                        src0->name,
                        src0->type,
                        ids->ne[1] > 1,
                        cur_a,
                        cne1,
                        (size_t) nb02,
                        expert_fallback_us,
                        fallback_touch_us[cur_a],
                        fallback_pack_mmap_ptrs[cur_a] != NULL);
            }
        }
    }
    if (kimi_cpu_moe_profile && ith == 0) {
        const uint64_t kimi_cpu_moe_fallback_us = ggml_time_us() - kimi_cpu_moe_fallback_start;
        const uint64_t kimi_cpu_moe_total_us = ggml_time_us() - kimi_cpu_moe_total_start;
        if (ggml_kimi_cpu_moe_fallback_profile_enabled()) {
            int64_t fallback_rows = 0;
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                fallback_rows += matrix_row_counts[cur_a];
            }
            if (fallback_rows > 0) {
                for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                    const int64_t cne1 = matrix_row_counts[cur_a];
                    if (cne1 == 0) {
                        continue;
                    }
                    const uint64_t expert_fallback_us =
                        (uint64_t) (((double) kimi_cpu_moe_fallback_us * (double) cne1) / (double) fallback_rows);
                    ggml_kimi_cpu_moe_fallback_profile_record(
                            src0->name,
                            src0->type,
                            ids->ne[1] > 1,
                            cur_a,
                            cne1,
                            (size_t) nb02,
                            expert_fallback_us,
                            fallback_touch_us[cur_a],
                            fallback_pack_mmap_ptrs[cur_a] != NULL);
                }
            }
        }
        ggml_kimi_cpu_moe_profile.down.fallback_us += kimi_cpu_moe_fallback_us;
        ggml_kimi_cpu_moe_profile.down.total_us += kimi_cpu_moe_total_us;
        ggml_kimi_cpu_moe_profile.down.calls++;
        ggml_kimi_cpu_moe_profile_record_phase(
                &ggml_kimi_cpu_moe_profile.down,
                ids->ne[1] > 1,
                kimi_cpu_moe_total_us,
                kimi_cpu_moe_cuda_batch_this_us,
                kimi_cpu_moe_cuda_single_this_us,
                kimi_cpu_moe_fallback_us);
        ggml_kimi_cpu_moe_name_profile_record(
            src0->name,
            kimi_cpu_moe_total_us,
            kimi_cpu_moe_fallback_us,
            ids->ne[1] > 1,
            use_gpu_stream_batch,
            kimi_cpu_moe_batch_done);
    }

    if (ggml_moe_stream_q80_probe_enabled() &&
            ggml_cuda_moe_stream_q80_probe &&
            src0->type == GGML_TYPE_MXFP4 &&
            src1->type != vec_dot_type &&
            vec_dot_type == GGML_TYPE_Q8_0 &&
            ne13 == 1 &&
            dst->type == GGML_TYPE_F32) {
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            const void * q80_base = params->wdata;
            const size_t q80_row_size = ggml_row_size(vec_dot_type, ne10);
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0) {
                    continue;
                }
                const char * src0_cur = fallback_pack_mmap_ptrs[cur_a] ?
                    (const char *) fallback_pack_mmap_ptrs[cur_a] :
                    (const char *) src0->data + cur_a * nb02;
                ggml_cuda_moe_stream_q80_probe(
                        src0->type,
                        src0->name,
                        cur_a,
                        src0_cur,
                        ne01, ne00, nb01,
                        q80_base,
                        q80_row_size,
                        ne11,
                        cne1,
                        (const float *) dst->data,
                        nb1, nb2,
                        (const ggml_moe_stream_row_mapping *) (matrix_rows + cur_a * ids->ne[0] * ids->ne[1]));
            }
        }
        ggml_barrier(params->threadpool);
    }

    if (ggml_moe_stream_q80_hot_batch_probe_enabled() &&
            ggml_cuda_moe_stream_q80_hot_batch_probe &&
            src0->type == GGML_TYPE_MXFP4 &&
            src1->type != vec_dot_type &&
            vec_dot_type == GGML_TYPE_Q8_0 &&
            ne13 == 1 &&
            dst->type == GGML_TYPE_F32) {
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            const void * q80_base = params->wdata;
            const size_t q80_row_size = ggml_row_size(vec_dot_type, ne10);
            ggml_cuda_moe_stream_q80_hot_batch_probe(
                    src0->type,
                    src0->name,
                    n_as,
                    ne01, ne00, nb01,
                    q80_base,
                    q80_row_size,
                    ne11,
                    (const float *) src1->data,
                    nb11,
                    nb12,
                    matrix_row_counts,
                    (const ggml_moe_stream_row_mapping *) matrix_rows,
                    ids->ne[0] * ids->ne[1],
                    (const float *) dst->data,
                    nb1, nb2);
        }
        ggml_barrier(params->threadpool);
    }

    if (ggml_moe_stream_q80_write_enabled() &&
            ggml_cuda_moe_stream_q80_write &&
            src0->type == GGML_TYPE_MXFP4 &&
            src1->type != vec_dot_type &&
            vec_dot_type == GGML_TYPE_Q8_0 &&
            ne13 == 1 &&
            dst->type == GGML_TYPE_F32) {
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            const void * q80_base = params->wdata;
            const size_t q80_row_size = ggml_row_size(vec_dot_type, ne10);
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0) {
                    continue;
                }
                const char * src0_cur = fallback_pack_mmap_ptrs[cur_a] ?
                    (const char *) fallback_pack_mmap_ptrs[cur_a] :
                    (const char *) src0->data + cur_a * nb02;
                ggml_cuda_moe_stream_q80_write(
                        src0->type,
                        src0->name,
                        cur_a,
                        src0_cur,
                        ne01, ne00, nb01,
                        q80_base,
                        q80_row_size,
                        ne11,
                        cne1,
                        (float *) dst->data,
                        nb1, nb2,
                        (const ggml_moe_stream_row_mapping *) (matrix_rows + cur_a * ids->ne[0] * ids->ne[1]));
            }
        }
        ggml_barrier(params->threadpool);
    }

    ggml_kimi_phase1_mmap_reclaim_tensor(
            params, src0, ids, n_as, nb02, (size_t) ne01 * nb01);
}
