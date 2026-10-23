// August-5 common CPU adapter, mechanically name-isolated for per-shape dispatch.
static int64_t ggml_common_moe_prompt_hybrid_cpu_max_rows(void) {
    static bool initialized = false;
    static int64_t max_rows = 0;
    if (!initialized) {
        initialized = true;
        const char * env = getenv("GGML_MOE_PROMPT_HYBRID_CPU_MAX_ROWS");
        if (env && env[0]) {
            char * end = NULL;
            const long value = strtol(env, &end, 10);
            if (end != env && value > 0 && value <= INT32_MAX) {
                max_rows = value;
            }
        }
    }
    return max_rows;
}

static bool ggml_common_moe_prompt_hybrid_cpu_expert(
        const struct ggml_tensor * ids,
        int64_t rows) {
    const int64_t max_rows = ggml_common_moe_prompt_hybrid_cpu_max_rows();
    return ids && ids->ne[1] > 1 && max_rows > 0 && rows > 0 && rows <= max_rows;
}


// Kimi-specific CPU MoE mul_mat_id path.
// This file is included by ggml-cpu.c so it can use that translation unit's
// internal helpers, macros, and static CUDA MoE bridge symbols.

static bool ggml_common_model_type_is_kimi(void) {
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

static bool ggml_common_kimi_moe_mixed_iq2_iq3_pair(enum ggml_type up_type, enum ggml_type gate_type) {
    return ggml_common_model_type_is_kimi() &&
           ((up_type == GGML_TYPE_IQ2_S && gate_type == GGML_TYPE_IQ3_XXS) ||
            (up_type == GGML_TYPE_IQ3_XXS && gate_type == GGML_TYPE_IQ2_S));
}

struct ggml_common_kimi_prompt_hybrid_stats {
    bool registered;
    uint64_t up_gate_calls;
    uint64_t down_calls;
    uint64_t cpu_experts;
    uint64_t gpu_experts;
    uint64_t cpu_rows;
    uint64_t gpu_rows;
};

static struct ggml_common_kimi_prompt_hybrid_stats ggml_common_kimi_prompt_hybrid_stats;

static void ggml_common_kimi_prompt_hybrid_report(void) {
    const struct ggml_common_kimi_prompt_hybrid_stats * s = &ggml_common_kimi_prompt_hybrid_stats;
    fprintf(stderr,
            "[kimi_prompt_hybrid] max_cpu_rows=%" PRId64
            " up_gate_calls=%" PRIu64 " down_calls=%" PRIu64
            " cpu_experts=%" PRIu64 " gpu_experts=%" PRIu64
            " cpu_rows=%" PRIu64 " gpu_rows=%" PRIu64 "\n",
            ggml_common_moe_prompt_hybrid_cpu_max_rows(),
            s->up_gate_calls, s->down_calls,
            s->cpu_experts, s->gpu_experts,
            s->cpu_rows, s->gpu_rows);
}

static void ggml_common_kimi_prompt_hybrid_partition(
        const struct ggml_tensor * ids,
        int64_t n_as,
        int64_t rows_stride,
        int64_t * gpu_row_counts,
        struct mmid_row_mapping * gpu_matrix_rows,
        int64_t * cpu_row_counts,
        struct mmid_row_mapping * cpu_matrix_rows,
        bool up_gate) {
    memset(cpu_row_counts, 0, (size_t) n_as * sizeof(int64_t));

    uint64_t cpu_experts = 0;
    uint64_t gpu_experts = 0;
    uint64_t cpu_rows = 0;
    uint64_t gpu_rows = 0;
    for (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t rows = gpu_row_counts[cur_a];
        if (rows <= 0) {
            continue;
        }
        if (ggml_common_moe_prompt_hybrid_cpu_expert(ids, rows)) {
            cpu_row_counts[cur_a] = rows;
            memcpy(cpu_matrix_rows + cur_a * rows_stride,
                   gpu_matrix_rows + cur_a * rows_stride,
                   (size_t) rows * sizeof(struct mmid_row_mapping));
            gpu_row_counts[cur_a] = 0;
            ++cpu_experts;
            cpu_rows += (uint64_t) rows;
        } else {
            ++gpu_experts;
            gpu_rows += (uint64_t) rows;
        }
    }

    struct ggml_common_kimi_prompt_hybrid_stats * s = &ggml_common_kimi_prompt_hybrid_stats;
    if (!s->registered) {
        s->registered = true;
        atexit(ggml_common_kimi_prompt_hybrid_report);
    }
    if (up_gate) {
        ++s->up_gate_calls;
    } else {
        ++s->down_calls;
    }
    s->cpu_experts += cpu_experts;
    s->gpu_experts += gpu_experts;
    s->cpu_rows += cpu_rows;
    s->gpu_rows += gpu_rows;
}

struct ggml_common_kimi_cpu_moe_profile_op {
    uint64_t calls;
    uint64_t convert_us;
    uint64_t route_us;
    uint64_t route_barrier_us;
    uint64_t cuda_batch_us;
    uint64_t cuda_single_us;
    uint64_t post_cuda_barrier_us;
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

#define GGML_COMMON_KIMI_CPU_MOE_NAME_PROFILE_MAX 512
#define GGML_COMMON_KIMI_CPU_MOE_NAME_PROFILE_LEN 96
#define GGML_COMMON_KIMI_CPU_MOE_FALLBACK_PROFILE_MAX 32768

enum ggml_common_kimi_cpu_moe_eligibility_reason {
    GGML_COMMON_KIMI_CPU_MOE_ELIGIBLE = 0,
    GGML_COMMON_KIMI_CPU_MOE_INELIG_ENV,
    GGML_COMMON_KIMI_CPU_MOE_INELIG_BATCH_FN,
    GGML_COMMON_KIMI_CPU_MOE_INELIG_AVAILABLE_FN,
    GGML_COMMON_KIMI_CPU_MOE_INELIG_AVAILABLE_FALSE,
    GGML_COMMON_KIMI_CPU_MOE_INELIG_UNSUPPORTED,
    GGML_COMMON_KIMI_CPU_MOE_INELIG_SRC1_TYPE,
    GGML_COMMON_KIMI_CPU_MOE_INELIG_NE13,
    GGML_COMMON_KIMI_CPU_MOE_INELIG_DST_TYPE,
    GGML_COMMON_KIMI_CPU_MOE_ELIGIBILITY_REASON_COUNT,
};

struct ggml_common_kimi_cpu_moe_name_profile_entry {
    char name[GGML_COMMON_KIMI_CPU_MOE_NAME_PROFILE_LEN];
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
    uint64_t eligibility[GGML_COMMON_KIMI_CPU_MOE_ELIGIBILITY_REASON_COUNT];
    uint64_t eligibility_decode[GGML_COMMON_KIMI_CPU_MOE_ELIGIBILITY_REASON_COUNT];
    uint64_t eligibility_prompt[GGML_COMMON_KIMI_CPU_MOE_ELIGIBILITY_REASON_COUNT];
};

struct ggml_common_kimi_cpu_moe_profile_state {
    bool registered;
    bool enabled;
    bool name_enabled;
    struct ggml_common_kimi_cpu_moe_profile_op up_gate;
    struct ggml_common_kimi_cpu_moe_profile_op down;
    struct ggml_common_kimi_cpu_moe_name_profile_entry names[GGML_COMMON_KIMI_CPU_MOE_NAME_PROFILE_MAX];
    int n_names;
};

static struct ggml_common_kimi_cpu_moe_profile_state ggml_common_kimi_cpu_moe_profile;

static void ggml_common_kimi_cpu_moe_profile_record_phase(
        struct ggml_common_kimi_cpu_moe_profile_op * op,
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

struct ggml_common_kimi_cpu_moe_fallback_profile_entry {
    char name[GGML_COMMON_KIMI_CPU_MOE_NAME_PROFILE_LEN];
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

struct ggml_common_kimi_cpu_moe_fallback_profile_state {
    bool initialized;
    bool registered;
    bool enabled;
    const char * out;
    struct ggml_common_kimi_cpu_moe_fallback_profile_entry entries[GGML_COMMON_KIMI_CPU_MOE_FALLBACK_PROFILE_MAX];
    int n_entries;
    uint64_t dropped;
};

static struct ggml_common_kimi_cpu_moe_fallback_profile_state ggml_common_kimi_cpu_moe_fallback_profile;

struct ggml_common_kimi_cpu_fallback_pack_mmap_state {
    bool initialized;
    bool enabled;
    bool registered;
    uint64_t hits;
    uint64_t misses;
    uint64_t bytes;
    uint64_t fallback_gguf;
};

static struct ggml_common_kimi_cpu_fallback_pack_mmap_state ggml_common_kimi_cpu_fallback_pack_mmap;

#define GGML_COMMON_KIMI_CPU_FALLBACK_SOURCE_PROFILE_MAX 128
#define GGML_COMMON_KIMI_CPU_FALLBACK_SOURCE_PATH_MAX 256

struct ggml_common_kimi_cpu_fallback_source_profile_entry {
    char source_path[GGML_COMMON_KIMI_CPU_FALLBACK_SOURCE_PATH_MAX];
    int source_idx;
    int source_is_gguf;
    int src0_type;
    bool prompt_phase;
    char role[8];
    uint64_t calls;
    uint64_t rows;
    uint64_t bytes;
};

struct ggml_common_kimi_cpu_fallback_source_profile_state {
    bool initialized;
    bool registered;
    bool enabled;
    const char * out;
    struct ggml_common_kimi_cpu_fallback_source_profile_entry entries[GGML_COMMON_KIMI_CPU_FALLBACK_SOURCE_PROFILE_MAX];
    int n_entries;
    uint64_t dropped;
};

static struct ggml_common_kimi_cpu_fallback_source_profile_state ggml_common_kimi_cpu_fallback_source_profile;

static void ggml_common_kimi_cpu_fallback_source_profile_report(void) {
    if (!ggml_common_kimi_cpu_fallback_source_profile.enabled ||
            !ggml_common_kimi_cpu_fallback_source_profile.out ||
            !ggml_common_kimi_cpu_fallback_source_profile.out[0]) {
        return;
    }

    FILE * f = fopen(ggml_common_kimi_cpu_fallback_source_profile.out, "w");
    if (!f) {
        fprintf(stderr,
                "[kimi_cpu_fallback_source_profile] open failed: %s\n",
                ggml_common_kimi_cpu_fallback_source_profile.out);
        return;
    }

    fprintf(f, "rank,phase,src0_type,role,source_idx,source_is_gguf,calls,rows,bytes,source_path\n");
    for (int i = 0; i < ggml_common_kimi_cpu_fallback_source_profile.n_entries; ++i) {
        const struct ggml_common_kimi_cpu_fallback_source_profile_entry * e =
            &ggml_common_kimi_cpu_fallback_source_profile.entries[i];
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
            ggml_common_kimi_cpu_fallback_source_profile.out,
            ggml_common_kimi_cpu_fallback_source_profile.n_entries,
            ggml_common_kimi_cpu_fallback_source_profile.dropped);
}

static bool ggml_common_kimi_cpu_fallback_source_profile_enabled(void) {
    if (!ggml_common_model_type_is_kimi()) {
        return false;
    }
    if (!ggml_common_kimi_cpu_fallback_source_profile.initialized) {
        ggml_common_kimi_cpu_fallback_source_profile.initialized = true;
        ggml_common_kimi_cpu_fallback_source_profile.out = getenv("GGML_MOE_CPU_FALLBACK_SOURCE_PROFILE_OUT");
        ggml_common_kimi_cpu_fallback_source_profile.enabled =
            ggml_common_kimi_cpu_fallback_source_profile.out &&
            ggml_common_kimi_cpu_fallback_source_profile.out[0];
        if (ggml_common_kimi_cpu_fallback_source_profile.enabled &&
                !ggml_common_kimi_cpu_fallback_source_profile.registered) {
            ggml_common_kimi_cpu_fallback_source_profile.registered = true;
            atexit(ggml_common_kimi_cpu_fallback_source_profile_report);
        }
    }

    return ggml_common_kimi_cpu_fallback_source_profile.enabled;
}

static const char * ggml_common_kimi_cpu_moe_tensor_role(const char * tensor_name) {
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

static void ggml_common_kimi_cpu_fallback_source_profile_record(
        const char * tensor_name,
        int src0_type,
        bool prompt_phase,
        int source_idx,
        const char * source_path,
        int source_is_gguf,
        int64_t rows,
        size_t expert_bytes) {
    if (!ggml_common_kimi_cpu_fallback_source_profile_enabled() || rows <= 0) {
        return;
    }

    const char * role = ggml_common_kimi_cpu_moe_tensor_role(tensor_name);
    const char * safe_path = source_path ? source_path : "<unknown>";
    int idx = -1;

    for (int i = 0; i < ggml_common_kimi_cpu_fallback_source_profile.n_entries; ++i) {
        struct ggml_common_kimi_cpu_fallback_source_profile_entry * e =
            &ggml_common_kimi_cpu_fallback_source_profile.entries[i];
        if (e->source_idx == source_idx &&
                e->source_is_gguf == source_is_gguf &&
                e->src0_type == src0_type &&
                e->prompt_phase == prompt_phase &&
                strncmp(e->role, role, sizeof(e->role)) == 0 &&
                strncmp(e->source_path, safe_path, GGML_COMMON_KIMI_CPU_FALLBACK_SOURCE_PATH_MAX) == 0) {
            idx = i;
            break;
        }
    }

    if (idx < 0) {
        if (ggml_common_kimi_cpu_fallback_source_profile.n_entries >= GGML_COMMON_KIMI_CPU_FALLBACK_SOURCE_PROFILE_MAX) {
            ggml_common_kimi_cpu_fallback_source_profile.dropped++;
            return;
        }
        idx = ggml_common_kimi_cpu_fallback_source_profile.n_entries++;
        struct ggml_common_kimi_cpu_fallback_source_profile_entry * e =
            &ggml_common_kimi_cpu_fallback_source_profile.entries[idx];
        snprintf(e->source_path, sizeof(e->source_path), "%s", safe_path);
        snprintf(e->role, sizeof(e->role), "%s", role);
        e->source_idx = source_idx;
        e->source_is_gguf = source_is_gguf;
        e->src0_type = src0_type;
        e->prompt_phase = prompt_phase;
    }

    struct ggml_common_kimi_cpu_fallback_source_profile_entry * e =
        &ggml_common_kimi_cpu_fallback_source_profile.entries[idx];
    e->calls++;
    e->rows += (uint64_t) rows;
    e->bytes += (uint64_t) rows * (uint64_t) expert_bytes;
}

static void ggml_common_kimi_cpu_fallback_pack_mmap_report(void) {
    if (!ggml_common_kimi_cpu_fallback_pack_mmap.enabled &&
            ggml_common_kimi_cpu_fallback_pack_mmap.hits == 0 &&
            ggml_common_kimi_cpu_fallback_pack_mmap.misses == 0) {
        return;
    }
    fprintf(stderr,
            "[kimi_cpu_fallback_pack_mmap] enabled=%d hits=%" PRIu64 " misses=%" PRIu64
            " bytes=%" PRIu64 " fallback_gguf=%" PRIu64 "\n",
            ggml_common_kimi_cpu_fallback_pack_mmap.enabled ? 1 : 0,
            ggml_common_kimi_cpu_fallback_pack_mmap.hits,
            ggml_common_kimi_cpu_fallback_pack_mmap.misses,
            ggml_common_kimi_cpu_fallback_pack_mmap.bytes,
            ggml_common_kimi_cpu_fallback_pack_mmap.fallback_gguf);
}

static bool ggml_common_kimi_cpu_fallback_pack_mmap_enabled(void) {
    if (!ggml_common_model_type_is_kimi()) {
        return false;
    }
    if (!ggml_common_kimi_cpu_fallback_pack_mmap.initialized) {
        ggml_common_kimi_cpu_fallback_pack_mmap.initialized = true;
        const char * env = getenv("GGML_MOE_CPU_FALLBACK_PACK_MMAP");
        ggml_common_kimi_cpu_fallback_pack_mmap.enabled = env && env[0] && env[0] != '0';
        if (ggml_common_kimi_cpu_fallback_pack_mmap.enabled && !ggml_common_kimi_cpu_fallback_pack_mmap.registered) {
            ggml_common_kimi_cpu_fallback_pack_mmap.registered = true;
            atexit(ggml_common_kimi_cpu_fallback_pack_mmap_report);
        }
    }
    return ggml_common_kimi_cpu_fallback_pack_mmap.enabled && ggml_cuda_moe_expert_pack_mmap_ptr != NULL;
}

static bool ggml_common_kimi_cpu_fallback_pack_mmap_prompt_enabled(void) {
    static int initialized = 0;
    static bool enabled = false;
    if (!initialized) {
        initialized = 1;
        const char * env = getenv("GGML_MOE_CPU_FALLBACK_PACK_MMAP_PROMPT");
        enabled = env && env[0] && env[0] != '0';
    }
    return enabled;
}

static bool ggml_common_kimi_cpu_fallback_miss_trace_enabled(void) {
    static int initialized = 0;
    static bool enabled = false;
    if (!initialized) {
        initialized = 1;
        const char * env = getenv("GGML_MOE_CPU_FALLBACK_MISS_TRACE");
        enabled = env && env[0] && env[0] != '0';
    }
    return enabled;
}

static void ggml_common_kimi_cpu_fallback_pack_mmap_prepare(
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
    if ((prompt_phase && !ggml_common_kimi_cpu_fallback_pack_mmap_prompt_enabled()) ||
            !ggml_common_kimi_cpu_fallback_pack_mmap_enabled() ||
            !tensor_name || !tensor_name[0] || expert_bytes == 0) {
        return;
    }
    const bool trace_misses = ggml_common_kimi_cpu_fallback_miss_trace_enabled();
    const bool source_profile = ggml_common_kimi_cpu_fallback_source_profile_enabled() &&
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
            ggml_common_kimi_cpu_fallback_pack_mmap.hits++;
            ggml_common_kimi_cpu_fallback_pack_mmap.bytes += expert_bytes;
            if (source_profile) {
                ggml_common_kimi_cpu_fallback_source_profile_record(
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
            ggml_common_kimi_cpu_fallback_pack_mmap.misses++;
            ggml_common_kimi_cpu_fallback_pack_mmap.fallback_gguf++;
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

static bool ggml_common_kimi_cpu_moe_eligibility_profile_enabled(void) {
    if (!ggml_common_model_type_is_kimi()) {
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

static int ggml_common_kimi_cpu_moe_name_profile_top(void) {
    const char * env = getenv("GGML_KIMI_CPU_MOE_NAME_PROFILE_TOP");
    if (!env || !env[0]) {
        return 40;
    }

    char * end = NULL;
    long value = strtol(env, &end, 10);
    if (end == env || value <= 0) {
        return 40;
    }
    if (value > GGML_COMMON_KIMI_CPU_MOE_NAME_PROFILE_MAX) {
        value = GGML_COMMON_KIMI_CPU_MOE_NAME_PROFILE_MAX;
    }
    return (int) value;
}

static void ggml_common_kimi_cpu_moe_fallback_profile_report(void) {
    if (!ggml_common_kimi_cpu_moe_fallback_profile.enabled ||
            !ggml_common_kimi_cpu_moe_fallback_profile.out ||
            !ggml_common_kimi_cpu_moe_fallback_profile.out[0]) {
        return;
    }

    FILE * f = fopen(ggml_common_kimi_cpu_moe_fallback_profile.out, "w");
    if (!f) {
        fprintf(stderr,
                "[kimi_cpu_moe_fallback_profile] open failed: %s\n",
                ggml_common_kimi_cpu_moe_fallback_profile.out);
        return;
    }

    fprintf(f, "rank,count,calls,fallback_us,touch_us,expert_bytes,src0_type,phase,expert_idx,pack_mmap_calls,gguf_calls,tensor\n");
    for (int i = 0; i < ggml_common_kimi_cpu_moe_fallback_profile.n_entries; ++i) {
        const struct ggml_common_kimi_cpu_moe_fallback_profile_entry * e =
            &ggml_common_kimi_cpu_moe_fallback_profile.entries[i];
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
            ggml_common_kimi_cpu_moe_fallback_profile.out,
            ggml_common_kimi_cpu_moe_fallback_profile.n_entries,
            ggml_common_kimi_cpu_moe_fallback_profile.dropped);
}

static bool ggml_common_kimi_cpu_moe_fallback_profile_enabled(void) {
    if (!ggml_common_model_type_is_kimi()) {
        return false;
    }
    if (!ggml_common_kimi_cpu_moe_fallback_profile.initialized) {
        ggml_common_kimi_cpu_moe_fallback_profile.initialized = true;
        ggml_common_kimi_cpu_moe_fallback_profile.out = getenv("GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT");
        ggml_common_kimi_cpu_moe_fallback_profile.enabled =
            ggml_common_kimi_cpu_moe_fallback_profile.out &&
            ggml_common_kimi_cpu_moe_fallback_profile.out[0];
        if (ggml_common_kimi_cpu_moe_fallback_profile.enabled &&
                !ggml_common_kimi_cpu_moe_fallback_profile.registered) {
            ggml_common_kimi_cpu_moe_fallback_profile.registered = true;
            atexit(ggml_common_kimi_cpu_moe_fallback_profile_report);
        }
    }

    return ggml_common_kimi_cpu_moe_fallback_profile.enabled;
}

static const char * ggml_common_kimi_cpu_moe_eligibility_reason_name(enum ggml_common_kimi_cpu_moe_eligibility_reason reason) {
    switch (reason) {
        case GGML_COMMON_KIMI_CPU_MOE_ELIGIBLE: return "eligible";
        case GGML_COMMON_KIMI_CPU_MOE_INELIG_ENV: return "batch_env_missing";
        case GGML_COMMON_KIMI_CPU_MOE_INELIG_BATCH_FN: return "batch_fn_missing";
        case GGML_COMMON_KIMI_CPU_MOE_INELIG_AVAILABLE_FN: return "available_fn_missing";
        case GGML_COMMON_KIMI_CPU_MOE_INELIG_AVAILABLE_FALSE: return "stream_available_false";
        case GGML_COMMON_KIMI_CPU_MOE_INELIG_UNSUPPORTED: return "batch_unsupported";
        case GGML_COMMON_KIMI_CPU_MOE_INELIG_SRC1_TYPE: return "src1_not_f32";
        case GGML_COMMON_KIMI_CPU_MOE_INELIG_NE13: return "ne13_not1";
        case GGML_COMMON_KIMI_CPU_MOE_INELIG_DST_TYPE: return "dst_not_f32";
        default: return "unknown";
    }
}

static void ggml_common_kimi_cpu_moe_fallback_profile_record(
        const char * name,
        int src0_type,
        bool prompt_phase,
        int expert_idx,
        int64_t count,
        size_t expert_bytes,
        uint64_t fallback_us,
        uint64_t touch_us,
        bool pack_mmap_source) {
    if (!ggml_common_kimi_cpu_moe_fallback_profile_enabled() || count <= 0) {
        return;
    }

    const char * safe_name = name ? name : "<unnamed>";
    int idx = -1;

    for (int i = 0; i < ggml_common_kimi_cpu_moe_fallback_profile.n_entries; ++i) {
        struct ggml_common_kimi_cpu_moe_fallback_profile_entry * e =
            &ggml_common_kimi_cpu_moe_fallback_profile.entries[i];
        if (e->expert_idx == expert_idx &&
                e->src0_type == src0_type &&
                e->prompt_phase == prompt_phase &&
                strncmp(e->name, safe_name, GGML_COMMON_KIMI_CPU_MOE_NAME_PROFILE_LEN) == 0) {
            idx = i;
            break;
        }
    }

    if (idx < 0) {
        if (ggml_common_kimi_cpu_moe_fallback_profile.n_entries >= GGML_COMMON_KIMI_CPU_MOE_FALLBACK_PROFILE_MAX) {
            ggml_common_kimi_cpu_moe_fallback_profile.dropped++;
            return;
        }

        idx = ggml_common_kimi_cpu_moe_fallback_profile.n_entries++;
        struct ggml_common_kimi_cpu_moe_fallback_profile_entry * e =
            &ggml_common_kimi_cpu_moe_fallback_profile.entries[idx];
        snprintf(e->name, GGML_COMMON_KIMI_CPU_MOE_NAME_PROFILE_LEN, "%s", safe_name);
        e->expert_idx = expert_idx;
        e->src0_type = src0_type;
        e->prompt_phase = prompt_phase;
        e->expert_bytes = expert_bytes;
    }

    struct ggml_common_kimi_cpu_moe_fallback_profile_entry * e =
        &ggml_common_kimi_cpu_moe_fallback_profile.entries[idx];
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

static void ggml_common_kimi_cpu_moe_profile_report_op(const char * name, const struct ggml_common_kimi_cpu_moe_profile_op * op) {
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
}

static void ggml_common_kimi_cpu_moe_profile_report(void) {
    if (!ggml_common_kimi_cpu_moe_profile.enabled) {
        return;
    }

    ggml_common_kimi_cpu_moe_profile_report_op("up_gate", &ggml_common_kimi_cpu_moe_profile.up_gate);
    ggml_common_kimi_cpu_moe_profile_report_op("down",    &ggml_common_kimi_cpu_moe_profile.down);

    if (ggml_common_kimi_cpu_moe_profile.name_enabled) {
        bool printed[GGML_COMMON_KIMI_CPU_MOE_NAME_PROFILE_MAX] = { false };
        const int top_n = MIN(ggml_common_kimi_cpu_moe_name_profile_top(), ggml_common_kimi_cpu_moe_profile.n_names);

        for (int rank = 0; rank < top_n; ++rank) {
            int best = -1;
            for (int i = 0; i < ggml_common_kimi_cpu_moe_profile.n_names; ++i) {
                if (printed[i]) {
                    continue;
                }
                if (best < 0 ||
                    ggml_common_kimi_cpu_moe_profile.names[i].total_us > ggml_common_kimi_cpu_moe_profile.names[best].total_us) {
                    best = i;
                }
            }

            if (best < 0) {
                break;
            }

            printed[best] = true;
            const struct ggml_common_kimi_cpu_moe_name_profile_entry * e = &ggml_common_kimi_cpu_moe_profile.names[best];
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
            if (ggml_common_kimi_cpu_moe_eligibility_profile_enabled()) {
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
                        e->eligibility[GGML_COMMON_KIMI_CPU_MOE_ELIGIBLE],
                        e->eligibility[GGML_COMMON_KIMI_CPU_MOE_INELIG_ENV],
                        e->eligibility[GGML_COMMON_KIMI_CPU_MOE_INELIG_BATCH_FN],
                        e->eligibility[GGML_COMMON_KIMI_CPU_MOE_INELIG_AVAILABLE_FN],
                        e->eligibility[GGML_COMMON_KIMI_CPU_MOE_INELIG_AVAILABLE_FALSE],
                        e->eligibility[GGML_COMMON_KIMI_CPU_MOE_INELIG_UNSUPPORTED],
                        e->eligibility[GGML_COMMON_KIMI_CPU_MOE_INELIG_SRC1_TYPE],
                        e->eligibility[GGML_COMMON_KIMI_CPU_MOE_INELIG_NE13],
                        e->eligibility[GGML_COMMON_KIMI_CPU_MOE_INELIG_DST_TYPE],
                        e->eligibility_decode[GGML_COMMON_KIMI_CPU_MOE_ELIGIBLE],
                        e->eligibility_decode[GGML_COMMON_KIMI_CPU_MOE_INELIG_UNSUPPORTED],
                        e->eligibility_prompt[GGML_COMMON_KIMI_CPU_MOE_ELIGIBLE],
                        e->eligibility_prompt[GGML_COMMON_KIMI_CPU_MOE_INELIG_UNSUPPORTED]);
            }
        }
    }
}

static bool ggml_common_kimi_cpu_moe_profile_enabled(void) {
    static int initialized = 0;

    if (!initialized) {
        initialized = 1;
        ggml_common_kimi_cpu_moe_profile.enabled =
            ggml_common_model_type_is_kimi() && getenv("GGML_KIMI_CPU_MOE_PROFILE") != NULL;
        ggml_common_kimi_cpu_moe_profile.name_enabled =
            ggml_common_kimi_cpu_moe_profile.enabled && getenv("GGML_KIMI_CPU_MOE_NAME_PROFILE") != NULL;
        if (ggml_common_kimi_cpu_moe_profile.enabled && !ggml_common_kimi_cpu_moe_profile.registered) {
            ggml_common_kimi_cpu_moe_profile.registered = true;
            atexit(ggml_common_kimi_cpu_moe_profile_report);
        }
    }

    return ggml_common_kimi_cpu_moe_profile.enabled;
}

static void ggml_common_kimi_cpu_moe_name_profile_record(
        const char * name,
        uint64_t total_us,
        uint64_t fallback_us,
        bool prompt_phase,
        bool cuda_batch_eligible,
        bool cuda_batch_accepted) {
    if (!ggml_common_kimi_cpu_moe_profile.name_enabled) {
        return;
    }

    const char * safe_name = name ? name : "<unnamed>";
    int idx = -1;

    for (int i = 0; i < ggml_common_kimi_cpu_moe_profile.n_names; ++i) {
        if (strncmp(ggml_common_kimi_cpu_moe_profile.names[i].name, safe_name, GGML_COMMON_KIMI_CPU_MOE_NAME_PROFILE_LEN) == 0) {
            idx = i;
            break;
        }
    }

    if (idx < 0) {
        if (ggml_common_kimi_cpu_moe_profile.n_names >= GGML_COMMON_KIMI_CPU_MOE_NAME_PROFILE_MAX) {
            idx = GGML_COMMON_KIMI_CPU_MOE_NAME_PROFILE_MAX - 1;
        } else {
            idx = ggml_common_kimi_cpu_moe_profile.n_names++;
            snprintf(ggml_common_kimi_cpu_moe_profile.names[idx].name,
                     GGML_COMMON_KIMI_CPU_MOE_NAME_PROFILE_LEN,
                     "%s",
                     safe_name);
        }
    }

    struct ggml_common_kimi_cpu_moe_name_profile_entry * e = &ggml_common_kimi_cpu_moe_profile.names[idx];
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

static void ggml_common_kimi_cpu_moe_name_profile_record_eligibility(
        const char * name,
        int src0_type,
        bool prompt_phase,
        enum ggml_common_kimi_cpu_moe_eligibility_reason reason) {
    if (!ggml_common_kimi_cpu_moe_profile.name_enabled ||
            !ggml_common_kimi_cpu_moe_eligibility_profile_enabled()) {
        return;
    }
    if (reason < 0 || reason >= GGML_COMMON_KIMI_CPU_MOE_ELIGIBILITY_REASON_COUNT) {
        return;
    }

    const char * safe_name = name ? name : "<unnamed>";
    int idx = -1;

    for (int i = 0; i < ggml_common_kimi_cpu_moe_profile.n_names; ++i) {
        if (strncmp(ggml_common_kimi_cpu_moe_profile.names[i].name, safe_name, GGML_COMMON_KIMI_CPU_MOE_NAME_PROFILE_LEN) == 0) {
            idx = i;
            break;
        }
    }

    if (idx < 0) {
        if (ggml_common_kimi_cpu_moe_profile.n_names >= GGML_COMMON_KIMI_CPU_MOE_NAME_PROFILE_MAX) {
            idx = GGML_COMMON_KIMI_CPU_MOE_NAME_PROFILE_MAX - 1;
        } else {
            idx = ggml_common_kimi_cpu_moe_profile.n_names++;
            snprintf(ggml_common_kimi_cpu_moe_profile.names[idx].name,
                     GGML_COMMON_KIMI_CPU_MOE_NAME_PROFILE_LEN,
                     "%s",
                     safe_name);
        }
    }

    ggml_common_kimi_cpu_moe_profile.names[idx].eligibility[reason]++;
    if (prompt_phase) {
        ggml_common_kimi_cpu_moe_profile.names[idx].eligibility_prompt[reason]++;
    } else {
        ggml_common_kimi_cpu_moe_profile.names[idx].eligibility_decode[reason]++;
    }
    ggml_common_kimi_cpu_moe_profile.names[idx].src0_type = src0_type;
}

static void ggml_compute_forward_mul_mat_id_common_kimi(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;
    const bool kimi_cpu_moe_profile = ggml_common_kimi_cpu_moe_profile_enabled();
    const bool ds4_sparse_fused_mmvq_membership = ggml_ds4_sparse_fused_mmvq_membership_enabled();
    const uint64_t kimi_cpu_moe_total_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
    uint64_t kimi_cpu_moe_cuda_batch_this_us = 0;
    uint64_t kimi_cpu_moe_cuda_single_this_us = 0;

    const enum ggml_type type = src0->type;

    const bool src1_cont = ggml_is_contiguous(src1);

    enum ggml_type    const vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t const from_float      = type_traits_cpu[vec_dot_type].from_float;

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
    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }
    if (kimi_cpu_moe_profile && ith == 0) {
        ggml_common_kimi_cpu_moe_profile.down.convert_us += ggml_time_us() - kimi_cpu_moe_convert_start;
    }

    const bool prompt_hybrid =
        ids->ne[1] > 1 &&
        ggml_common_moe_prompt_hybrid_cpu_max_rows() > 0 &&
        ggml_moe_gpu_provider_enabled(ids) &&
        getenv("GGML_MOE_STREAM_DOWN_BATCH") != NULL &&
        ggml_cuda_moe_stream_batch &&
        ggml_cuda_moe_stream_available &&
        ggml_cuda_moe_stream_available() &&
        ggml_cuda_moe_stream_supports_down_batch(src0->type, src0->name) &&
        src1->type == GGML_TYPE_F32 && ne13 == 1 && dst->type == GGML_TYPE_F32 &&
        (src0->type != GGML_TYPE_Q4_0 ||
         (getenv("GGML_MOE_Q4_PROMPT_DOWN_BATCH") != NULL &&
          getenv("GGML_MOE_Q4_PROMPT_DOWN_BATCH")[0] != '0'));

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
            ggml_common_kimi_cpu_moe_profile.down.route_us += ggml_time_us() - kimi_cpu_moe_route_start;
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
        if (prompt_hybrid) {
            ggml_common_kimi_prompt_hybrid_partition(
                    ids, n_as, ids->ne[0] * ids->ne[1],
                    matrix_row_counts, matrix_rows,
                    gpu_tail_row_counts, gpu_tail_matrix_rows,
                    false);
            memset(fallback_pack_mmap_ptrs, 0, (size_t) n_as * sizeof(void *));
            memset(fallback_touch_us, 0, (size_t) n_as * sizeof(uint64_t));
        }
    }

    // reset current_chunk
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    const uint64_t kimi_cpu_moe_route_barrier_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
    ggml_barrier(params->threadpool);
    if (kimi_cpu_moe_profile && ith == 0) {
        ggml_common_kimi_cpu_moe_profile.down.route_barrier_us += ggml_time_us() - kimi_cpu_moe_route_barrier_start;
    }

    enum ggml_common_kimi_cpu_moe_eligibility_reason kimi_cpu_moe_batch_reason = GGML_COMMON_KIMI_CPU_MOE_ELIGIBLE;
    if (!ggml_moe_gpu_provider_enabled(ids)) {
        kimi_cpu_moe_batch_reason = GGML_COMMON_KIMI_CPU_MOE_INELIG_ENV;
    } else if (getenv("GGML_MOE_STREAM_DOWN_BATCH") == NULL) {
        kimi_cpu_moe_batch_reason = GGML_COMMON_KIMI_CPU_MOE_INELIG_ENV;
    } else if (!ggml_cuda_moe_stream_batch) {
        kimi_cpu_moe_batch_reason = GGML_COMMON_KIMI_CPU_MOE_INELIG_BATCH_FN;
    } else if (!ggml_cuda_moe_stream_available) {
        kimi_cpu_moe_batch_reason = GGML_COMMON_KIMI_CPU_MOE_INELIG_AVAILABLE_FN;
    } else if (!ggml_cuda_moe_stream_available()) {
        kimi_cpu_moe_batch_reason = GGML_COMMON_KIMI_CPU_MOE_INELIG_AVAILABLE_FALSE;
    } else if (src0->type == GGML_TYPE_Q4_0 && ids->ne[1] > 1 &&
            !(getenv("GGML_MOE_Q4_PROMPT_DOWN_BATCH") != NULL &&
              getenv("GGML_MOE_Q4_PROMPT_DOWN_BATCH")[0] != '0')) {
        kimi_cpu_moe_batch_reason = GGML_COMMON_KIMI_CPU_MOE_INELIG_UNSUPPORTED;
    } else if (!ggml_cuda_moe_stream_supports_down_batch(src0->type, src0->name)) {
        kimi_cpu_moe_batch_reason = GGML_COMMON_KIMI_CPU_MOE_INELIG_UNSUPPORTED;
    } else if (src1->type != GGML_TYPE_F32) {
        kimi_cpu_moe_batch_reason = GGML_COMMON_KIMI_CPU_MOE_INELIG_SRC1_TYPE;
    } else if (ne13 != 1) {
        kimi_cpu_moe_batch_reason = GGML_COMMON_KIMI_CPU_MOE_INELIG_NE13;
    } else if (dst->type != GGML_TYPE_F32) {
        kimi_cpu_moe_batch_reason = GGML_COMMON_KIMI_CPU_MOE_INELIG_DST_TYPE;
    }

    if (kimi_cpu_moe_profile && ith == 0) {
        ggml_common_kimi_cpu_moe_name_profile_record_eligibility(
                src0->name, src0->type, ids->ne[1] > 1, kimi_cpu_moe_batch_reason);
    }

    const bool use_gpu_stream_batch =
        kimi_cpu_moe_batch_reason == GGML_COMMON_KIMI_CPU_MOE_ELIGIBLE;
    bool kimi_cpu_moe_batch_done = false;
    int64_t kimi_cpu_moe_single_attempts = 0;
    int64_t kimi_cpu_moe_single_accepts = 0;

    if (use_gpu_stream_batch) {
        if (ith == 0) {
            const uint64_t kimi_cpu_moe_cuda_start = kimi_cpu_moe_profile ? ggml_time_us() : 0;
            const int gpu_keep_topk =
                ggml_moe_gpu_keep_topk_applies(src0->name) ? ggml_moe_gpu_keep_topk_updown() : 0;
            const bool split_gpu_cpu_tail =
                !prompt_hybrid && gpu_keep_topk > 0 && gpu_keep_topk < ids->ne[0];
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
            if (prompt_hybrid && !done) {
                GGML_ABORT("Kimi prompt hybrid GPU down batch declined");
            }
            if (kimi_cpu_moe_profile) {
                const uint64_t elapsed_us = ggml_time_us() - kimi_cpu_moe_cuda_start;
                ggml_common_kimi_cpu_moe_profile.down.cuda_batch_us += elapsed_us;
                kimi_cpu_moe_cuda_batch_this_us += elapsed_us;
                if (done) {
                    ggml_common_kimi_cpu_moe_profile.down.cuda_batch_accepted++;
                } else {
                    ggml_common_kimi_cpu_moe_profile.down.cuda_batch_declined++;
                }
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
        }

        if (!prompt_hybrid) {
            const uint64_t kimi_cpu_moe_post_cuda_barrier_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
            ggml_barrier(params->threadpool);
            if (kimi_cpu_moe_profile && ith == 0) {
                ggml_common_kimi_cpu_moe_profile.down.post_cuda_barrier_us += ggml_time_us() - kimi_cpu_moe_post_cuda_barrier_start;
            }
        }
    }

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
                    ggml_common_kimi_cpu_moe_profile.down.cuda_single_us += elapsed_us;
                    kimi_cpu_moe_cuda_single_this_us += elapsed_us;
                    if (done) {
                        ggml_common_kimi_cpu_moe_profile.down.cuda_single_accepted++;
                    } else {
                        ggml_common_kimi_cpu_moe_profile.down.cuda_single_declined++;
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
            ggml_common_kimi_cpu_moe_profile.down.post_cuda_barrier_us += ggml_time_us() - kimi_cpu_moe_post_cuda_barrier_start;
        }
    }

    if (!prompt_hybrid && ggml_moe_stream_one_gpu_only_filter_matches(src0->name)) {
        int64_t gpu_only_rows = 0;
        for (int cur_a = 0; cur_a < n_as; ++cur_a) {
            gpu_only_rows += matrix_row_counts[cur_a];
        }
        if (gpu_only_rows > 0) {
            fprintf(stderr,
                    "[moe_stream] GPU-only filter matched but CPU fallback remains: tensor=%s rows=%" PRId64 " batch_reason=%s single_attempts=%" PRId64 " single_accepts=%" PRId64 "\n",
                    src0->name ? src0->name : "", gpu_only_rows,
                    ggml_common_kimi_cpu_moe_eligibility_reason_name(kimi_cpu_moe_batch_reason),
                    kimi_cpu_moe_single_attempts, kimi_cpu_moe_single_accepts);
            abort();
        }
    }

    if (!prompt_hybrid && ggml_moe_cpu_willneed_enabled()) {
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

    if (!prompt_hybrid) {
        if (ith == 0) {
            ggml_common_kimi_cpu_fallback_pack_mmap_prepare(
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
    }

    const bool use_cpu_batch_microprobe =
        !prompt_hybrid && ggml_moe_cpu_batch_microprobe_enabled() &&
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

    const int64_t * cpu_row_counts = prompt_hybrid ? gpu_tail_row_counts : matrix_row_counts;
    const struct mmid_row_mapping * cpu_matrix_rows =
        prompt_hybrid ? gpu_tail_matrix_rows : matrix_rows;

    const bool moe_fallback_reason_profile = ggml_moe_fallback_reason_profile_enabled();
    const uint64_t kimi_cpu_moe_fallback_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
    const uint64_t moe_fallback_reason_start = (moe_fallback_reason_profile && ith == 0) ? ggml_time_us() : 0;
    const uint64_t ds4_sparse_fused_mmvq_fallback_start =
        (ds4_sparse_fused_mmvq_membership && ith == 0) ? ggml_time_us() : 0;
    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = cpu_row_counts[cur_a];

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
                src0_cur, cpu_matrix_rows, row_size, src1_cont, wdata
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
    if (prompt_hybrid) {
        ggml_barrier(params->threadpool);
    }
    if (moe_fallback_reason_profile && ith == 0) {
        const uint64_t moe_fallback_reason_us = ggml_time_us() - moe_fallback_reason_start;
        int64_t fallback_rows = 0;
        for (int cur_a = 0; cur_a < n_as; ++cur_a) {
            fallback_rows += matrix_row_counts[cur_a];
        }
        const char * batch_reason_name = ggml_common_kimi_cpu_moe_eligibility_reason_name(kimi_cpu_moe_batch_reason);
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
        if (ggml_common_kimi_cpu_moe_fallback_profile_enabled()) {
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
                    ggml_common_kimi_cpu_moe_fallback_profile_record(
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
        ggml_common_kimi_cpu_moe_profile.down.fallback_us += kimi_cpu_moe_fallback_us;
        ggml_common_kimi_cpu_moe_profile.down.total_us += kimi_cpu_moe_total_us;
        ggml_common_kimi_cpu_moe_profile.down.calls++;
        ggml_common_kimi_cpu_moe_profile_record_phase(
                &ggml_common_kimi_cpu_moe_profile.down,
                ids->ne[1] > 1,
                kimi_cpu_moe_total_us,
                kimi_cpu_moe_cuda_batch_this_us,
                kimi_cpu_moe_cuda_single_this_us,
                kimi_cpu_moe_fallback_us);
        ggml_common_kimi_cpu_moe_name_profile_record(
            src0->name,
            kimi_cpu_moe_total_us,
            kimi_cpu_moe_fallback_us,
            ids->ne[1] > 1,
            use_gpu_stream_batch,
            kimi_cpu_moe_batch_done);
    }

}

static float ggml_common_moe_up_gate_activate(float x, enum ggml_unary_op op) {
    switch (op) {
        case GGML_UNARY_OP_SILU:
            return ggml_silu_f32(x);
        default:
            GGML_ABORT("unsupported MoE fused up/gate CPU activation");
    }
}

static float ggml_common_moe_up_gate_clamp(float x, float lo, float hi) {
    return MIN(hi, MAX(lo, x));
}

static float ggml_common_moe_up_gate_fuse_value(float up, float gate, enum ggml_unary_op op, float limit) {
    if (op == GGML_UNARY_OP_SILU && limit > 1.0e-6f) {
        const float gate_v = ggml_silu_f32(MIN(gate, limit));
        const float up_v = ggml_common_moe_up_gate_clamp(up, -limit, limit);
        return up_v * gate_v;
    }
    return up * ggml_common_moe_up_gate_activate(gate, op);
}

static void ggml_compute_forward_moe_up_gate_common_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0_up,
    const struct ggml_tensor * src0_gate,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_up_cur,
    const char * src0_gate_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata,
    enum ggml_unary_op op,
    float limit) {

    const enum ggml_type type_up = src0_up->type;
    const enum ggml_type type_gate = src0_gate->type;

    ggml_vec_dot_t const vec_dot_up      = type_traits_cpu[type_up].vec_dot;
    ggml_vec_dot_t const vec_dot_gate    = type_traits_cpu[type_gate].vec_dot;
    enum ggml_type const vec_dot_type_up = type_traits_cpu[type_up].vec_dot_type;
    enum ggml_type const vec_dot_type_gate = type_traits_cpu[type_gate].vec_dot_type;

    const int64_t ne00 = src0_up->ne[0];
    const int64_t ne11 = src1->ne[1];

    const size_t up_nb01 = src0_up->nb[1];
    const size_t gate_nb01 = src0_gate->nb[1];
    const size_t nb11 = src1->nb[1];
    const size_t nb12 = src1->nb[2];
    const size_t nb1  = dst->nb[1];
    const size_t nb2  = dst->nb[2];

    GGML_ASSERT(vec_dot_type_up == vec_dot_type_gate);
    const enum ggml_type vec_dot_type = vec_dot_type_up;

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float up_tmp[16];
    float gate_tmp[16];
    float fused_tmp[16];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);
                const int id = row_mapping.i1;

                const int64_t i11 = id % ne11;
                const int64_t i12 = row_mapping.i2;

                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11 + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                float * dst_col = (float *) ((char *) dst->data + (id*nb1 + i12*nb2));

                const int64_t ir0_stop = MIN(iir0 + blck_0, ir0_end);
                for (int64_t ir0 = iir0; ir0 < ir0_stop; ++ir0) {
                    vec_dot_up(ne00,   &up_tmp[ir0 - iir0],   0, src0_up_cur   + ir0*up_nb01,   0, src1_col, 0, 1);
                    vec_dot_gate(ne00, &gate_tmp[ir0 - iir0], 0, src0_gate_cur + ir0*gate_nb01, 0, src1_col, 0, 1);
                }

                const int64_t n_cur = ir0_stop - iir0;
                if (op == GGML_UNARY_OP_SILU && limit > 1.0e-6f) {
                    for (int64_t i = 0; i < n_cur; ++i) {
                        gate_tmp[i] = MIN(gate_tmp[i], limit);
                        up_tmp[i]   = ggml_common_moe_up_gate_clamp(up_tmp[i], -limit, limit);
                    }
                    ggml_vec_swiglu_f32((int) n_cur, fused_tmp, gate_tmp, up_tmp);
                } else {
                    for (int64_t i = 0; i < n_cur; ++i) {
                        fused_tmp[i] = ggml_common_moe_up_gate_fuse_value(up_tmp[i], gate_tmp[i], op, limit);
                    }
                }

                memcpy(&dst_col[iir0], fused_tmp, n_cur*sizeof(float));
            }
        }
    }
}

static void ggml_compute_forward_moe_up_gate_common(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0_up   = dst->src[0];
    const struct ggml_tensor * src0_gate = dst->src[1];
    const struct ggml_tensor * src1      = dst->src[2];
    const struct ggml_tensor * ids       = dst->src[3];

    const int ith = params->ith;
    const int nth = params->nth;
    const bool kimi_cpu_moe_profile = ggml_common_kimi_cpu_moe_profile_enabled();
    const uint64_t kimi_cpu_moe_total_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
    uint64_t kimi_cpu_moe_cuda_batch_this_us = 0;

    const enum ggml_type up_type = src0_up->type;
    const enum ggml_type gate_type = src0_gate->type;
    enum ggml_type    const vec_dot_type = type_traits_cpu[up_type].vec_dot_type;
    enum ggml_type    const gate_vec_dot_type = type_traits_cpu[gate_type].vec_dot_type;
    ggml_from_float_t const from_float   = type_traits_cpu[vec_dot_type].from_float;
    const bool same_weight_type = up_type == gate_type;
    const bool scoped_mixed_pair = ggml_common_kimi_moe_mixed_iq2_iq3_pair(up_type, gate_type);
    const float fused_limit = ggml_get_op_params_f32(dst, 1);

    GGML_ASSERT(src0_gate != NULL);
    GGML_ASSERT(same_weight_type || scoped_mixed_pair);
    GGML_ASSERT(vec_dot_type == gate_vec_dot_type);
    GGML_ASSERT(ggml_are_same_shape(src0_up, src0_gate));
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == vec_dot_type);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_get_op_params_i32(dst, 0) == GGML_UNARY_OP_SILU);

    const int64_t ne00 = src0_up->ne[0];
    const int64_t ne01 = src0_up->ne[1];
    const int64_t n_as = src0_up->ne[2];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const size_t nb01 = src0_up->nb[1];
    const size_t nb02 = src0_up->nb[2];
    const size_t gate_nb02 = src0_gate->nb[2];
    const size_t nb11 = src1->nb[1];
    const size_t nb12 = src1->nb[2];
    const size_t nb1  = dst->nb[1];
    const size_t nb2  = dst->nb[2];

    GGML_ASSERT(ne00 == ne10);
    GGML_ASSERT(ids->ne[0] % ne11 == 0);
    GGML_ASSERT(ids->ne[1] == ne12);
    GGML_ASSERT(ne13 == 1);

    const bool src1_cont = ggml_is_contiguous(src1);
    const int n_ids = ids->ne[0];

    void * wdata_cur = params->wdata;

    if (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), sizeof(int64_t));
    }

    int64_t * matrix_row_counts =
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows =
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[CACHE_LINE_SIZE] =
        incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

    int64_t * cpu_row_counts =
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * cpu_matrix_rows =
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));
    const uint64_t kimi_cpu_moe_convert_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*src1->nb[3] + i12*nb12 + i11*nb11 + ne10_block_start*bs*src1->nb[0]),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
    }
    if (kimi_cpu_moe_profile && ith == 0) {
        ggml_common_kimi_cpu_moe_profile.up_gate.convert_us += ggml_time_us() - kimi_cpu_moe_convert_start;
    }

    const bool prompt_hybrid =
        ggml_common_model_type_is_kimi() &&
        ids->ne[1] > 1 &&
        ggml_common_moe_prompt_hybrid_cpu_max_rows() > 0 &&
        ggml_moe_gpu_provider_enabled(ids) &&
        ggml_cuda_moe_stream_up_gate_batch &&
        ggml_cuda_moe_stream_available &&
        ggml_cuda_moe_stream_available() &&
        ggml_cuda_moe_stream_supports_type(up_type) &&
        ggml_cuda_moe_stream_supports_type(gate_type) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    if (ith == 0) {
        const uint64_t kimi_cpu_moe_route_start = kimi_cpu_moe_profile ? ggml_time_us() : 0;
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                if (i02 == -1) {
                    memset((char *) dst->data + id * nb1 + iid1 * nb2, 0, (size_t) ne01 * sizeof(float));
                    continue;
                }
                assert(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
        if (kimi_cpu_moe_profile) {
            ggml_common_kimi_cpu_moe_profile.up_gate.route_us += ggml_time_us() - kimi_cpu_moe_route_start;
        }
        const size_t up_expert_bytes = (size_t)ne01 * nb01;
        const size_t gate_expert_bytes = (size_t)src0_gate->ne[1] * src0_gate->nb[1];
        ggml_ds4_grouped_retained_route_profile_record(
                src0_up->name,
                src0_up->type,
                src0_up->data,
                src0_up->nb[2],
                ids->ne[1] > 1,
                matrix_row_counts,
                n_as,
                up_expert_bytes);
        ggml_ds4_grouped_retained_route_profile_record(
                src0_gate->name,
                src0_gate->type,
                src0_gate->data,
                src0_gate->nb[2],
                ids->ne[1] > 1,
                matrix_row_counts,
                n_as,
                gate_expert_bytes);
        ggml_ds4_grouped_retained_route_profile_record_up_gate(
                src0_up->name,
                src0_gate->name,
                src0_up->type,
                src0_gate->type,
                src0_up->data,
                src0_up->nb[2],
                src0_gate->data,
                src0_gate->nb[2],
                ids->ne[1] > 1,
                matrix_row_counts,
                n_as,
                up_expert_bytes,
                gate_expert_bytes);
        ggml_ds4_grouped_retained_handoff_mark_up_gate(
                src0_up->name,
                src0_gate->name,
                dst->data,
                ne01,
                matrix_row_counts,
                n_as,
                ids->ne[0]*ids->ne[1]);
        if (prompt_hybrid) {
            ggml_common_kimi_prompt_hybrid_partition(
                    ids, n_as, ids->ne[0] * ids->ne[1],
                    matrix_row_counts, matrix_rows,
                    cpu_row_counts, cpu_matrix_rows,
                    true);
        }
    }

    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    const uint64_t kimi_cpu_moe_route_barrier_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
    ggml_barrier(params->threadpool);
    if (kimi_cpu_moe_profile && ith == 0) {
        ggml_common_kimi_cpu_moe_profile.up_gate.route_barrier_us += ggml_time_us() - kimi_cpu_moe_route_barrier_start;
    }

    const bool use_gpu_stream =
        ggml_moe_gpu_provider_enabled(ids) &&
        ggml_cuda_moe_stream_up_gate_batch &&
        ggml_cuda_moe_stream_available &&
        ggml_cuda_moe_stream_available() &&
        ggml_cuda_moe_stream_supports_type(up_type) &&
        ggml_cuda_moe_stream_supports_type(gate_type) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32;

    if (use_gpu_stream) {
        if (ith == 0) {
            const uint64_t kimi_cpu_moe_cuda_start = kimi_cpu_moe_profile ? ggml_time_us() : 0;
            if (ggml_moe_gate_batch_prefetch_enabled() &&
                    ggml_cuda_moe_stream_batch_preload_active_from_pack) {
                static bool same_layer_prefetch_logged = false;
                if (!same_layer_prefetch_logged) {
                    fprintf(stderr, "[moe_stream_cpu] same-layer prefetch before up/gate batch: gate=%s n_as=%" PRId64 " expert_bytes=%zu\n",
                            src0_gate->name, (int64_t) n_as,
                            (size_t) src0_gate->ne[1] * src0_gate->nb[1]);
                    same_layer_prefetch_logged = true;
                }
                const int same_layer_jobs = ggml_cuda_moe_stream_batch_preload_active_from_pack(
                        gate_type,
                        src0_gate->name,
                        n_as,
                        (size_t) src0_gate->ne[1] * src0_gate->nb[1],
                        matrix_row_counts);
                (void) same_layer_jobs;
            }
            const char * q4_colibri_pilot_prefetch_env = getenv("GGML_MOE_Q4_COLIBRI_PILOT_PREFETCH");
            if (q4_colibri_pilot_prefetch_env &&
                    q4_colibri_pilot_prefetch_env[0] &&
                    q4_colibri_pilot_prefetch_env[0] != '0' &&
                    ggml_cuda_moe_stream_q4_colibri_pilot_prefetch_from_gate) {
                const int pilot_prefetch_jobs = ggml_cuda_moe_stream_q4_colibri_pilot_prefetch_from_gate(
                        gate_type,
                        src0_gate->name,
                        n_as,
                        (size_t) src0_gate->ne[1] * src0_gate->nb[1],
                        matrix_row_counts);
                (void) pilot_prefetch_jobs;
            }
            const bool done = ggml_cuda_moe_stream_up_gate_batch(
                up_type,
                gate_type,
                src0_up->name, src0_up->data,
                src0_gate->name, src0_gate->data,
                n_as,
                ne01, ne00, nb01, nb02,
                (size_t) ne01 * nb01,
                src0_gate->nb[1], src0_gate->nb[2],
                (size_t) src0_gate->ne[1] * src0_gate->nb[1],
                (const float *) src1->data,
                nb11, nb12,
                (float *) dst->data,
                nb1, nb2,
                ggml_get_op_params_i32(dst, 0),
                fused_limit,
                matrix_row_counts,
                (const ggml_moe_stream_row_mapping *) matrix_rows,
                ids->ne[0]*ids->ne[1]);
            if (prompt_hybrid && !done) {
                GGML_ABORT("prompt hybrid GPU up/gate batch declined");
            }
            if (kimi_cpu_moe_profile) {
                const uint64_t elapsed_us = ggml_time_us() - kimi_cpu_moe_cuda_start;
                ggml_common_kimi_cpu_moe_profile.up_gate.cuda_batch_us += elapsed_us;
                kimi_cpu_moe_cuda_batch_this_us += elapsed_us;
                if (done) {
                    ggml_common_kimi_cpu_moe_profile.up_gate.cuda_batch_accepted++;
                } else {
                    ggml_common_kimi_cpu_moe_profile.up_gate.cuda_batch_declined++;
                }
            }
            if (done) {
                memset(matrix_row_counts, 0, n_as*sizeof(int64_t));
            }
        }

        if (!prompt_hybrid) {
            const uint64_t kimi_cpu_moe_post_cuda_barrier_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
            ggml_barrier(params->threadpool);
            if (kimi_cpu_moe_profile && ith == 0) {
                ggml_common_kimi_cpu_moe_profile.up_gate.post_cuda_barrier_us += ggml_time_us() - kimi_cpu_moe_post_cuda_barrier_start;
            }
        }
    }

    const size_t row_size = ggml_row_size(vec_dot_type, ne10);
    const enum ggml_unary_op op = (enum ggml_unary_op) ggml_get_op_params_i32(dst, 0);

    const bool moe_fallback_reason_profile = ggml_moe_fallback_reason_profile_enabled();
    const uint64_t moe_fallback_reason_start = (moe_fallback_reason_profile && ith == 0) ? ggml_time_us() : 0;
    const uint64_t kimi_cpu_moe_fallback_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
    const int64_t * fallback_row_counts = prompt_hybrid ? cpu_row_counts : matrix_row_counts;
    const struct mmid_row_mapping * fallback_matrix_rows =
        prompt_hybrid ? cpu_matrix_rows : matrix_rows;
    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = fallback_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        const char * src0_up_cur   = (const char *) src0_up->data   + cur_a * nb02;
        const char * src0_gate_cur = (const char *) src0_gate->data + cur_a * gate_nb02;
        const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;

        const int64_t nr0 = ne01;
        const int64_t nr1 = cne1;

        int chunk_size = 16;
        if (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

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

            ggml_compute_forward_moe_up_gate_common_one_chunk(
                dst, src0_up, src0_gate, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_up_cur, src0_gate_cur, fallback_matrix_rows, row_size, src1_cont, wdata, op, fused_limit);

            if (trace_cpu_chunk) {
                const double trace_t1_ms = ggml_moe_cpu_trace_now_ms();
                ggml_moe_cpu_chunk_trace_write(
                    "up_gate",
                    src0_up->name,
                    src0_up->type,
                    cur_a,
                    ith,
                    nth,
                    cne1,
                    ir0_start,
                    ir0_end,
                    ir1_start,
                    ir1_end,
                    (size_t) ne01 * nb01 + (size_t) src0_gate->ne[1] * src0_gate->nb[1],
                    trace_t1_ms - trace_t0_ms);
            }

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
    if (prompt_hybrid) {
        ggml_barrier(params->threadpool);
    }
    if (moe_fallback_reason_profile && ith == 0) {
        const uint64_t moe_fallback_reason_us = ggml_time_us() - moe_fallback_reason_start;
        int64_t fallback_rows = 0;
        for (int cur_a = 0; cur_a < n_as; ++cur_a) {
            fallback_rows += matrix_row_counts[cur_a];
        }
        const char * batch_reason_name = use_gpu_stream ? "eligible" : "upgate_batch_precondition_failed";
        const char * final_reason_name = use_gpu_stream ? "upgate_batch_declined_internal" : "upgate_batch_not_attempted";
        if (fallback_rows > 0) {
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0) {
                    continue;
                }
                const uint64_t expert_fallback_us =
                    (uint64_t) (((double) moe_fallback_reason_us * (double) cne1) / (double) fallback_rows);
                ggml_moe_fallback_reason_profile_record(
                        "up_gate",
                        src0_up->name,
                        src0_up->type,
                        ids->ne[1] > 1,
                        cur_a,
                        cne1,
                        (size_t) ne01 * nb01 + (size_t) src0_gate->ne[1] * src0_gate->nb[1],
                        expert_fallback_us,
                        batch_reason_name,
                        "not_applicable",
                        final_reason_name,
                        use_gpu_stream,
                        false,
                        false,
                        false);
                ggml_moe_fallback_source_probe_record(
                        "up_gate",
                        src0_up->name,
                        ids->ne[1] > 1,
                        cur_a,
                        cne1,
                        expert_fallback_us,
                        batch_reason_name,
                        "not_applicable",
                        final_reason_name,
                        src0_up,
                        src1,
                        ids,
                        dst);
            }
        }
    }
    if (kimi_cpu_moe_profile && ith == 0) {
        const uint64_t now_us = ggml_time_us();
        const uint64_t fallback_us = now_us - kimi_cpu_moe_fallback_start;
        const uint64_t total_us = now_us - kimi_cpu_moe_total_start;
        ggml_common_kimi_cpu_moe_profile.up_gate.fallback_us += fallback_us;
        ggml_common_kimi_cpu_moe_profile.up_gate.total_us += total_us;
        ggml_common_kimi_cpu_moe_profile.up_gate.calls++;
        ggml_common_kimi_cpu_moe_profile_record_phase(
                &ggml_common_kimi_cpu_moe_profile.up_gate,
                ids->ne[1] > 1,
                total_us,
                kimi_cpu_moe_cuda_batch_this_us,
                0,
                fallback_us);
    }
}
