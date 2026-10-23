// CPU MoE extension, included in the CPU translation unit.
static void ggml_moe_fallback_reason_profile_init(void) {
    ggml_moe_fallback_reason_profile.out = getenv("GGML_MOE_FALLBACK_REASON_PROFILE_OUT");
    ggml_moe_fallback_reason_profile.enabled =
        ggml_moe_fallback_reason_profile.out &&
        ggml_moe_fallback_reason_profile.out[0];
    if (ggml_moe_fallback_reason_profile.enabled) {
        atexit(ggml_moe_fallback_reason_profile_report);
    }
}

static bool ggml_moe_fallback_reason_profile_enabled(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, ggml_moe_fallback_reason_profile_init);
    return ggml_moe_fallback_reason_profile.enabled;
}

static bool ggml_moe_cpu_chunk_trace_enabled(void) {
    static atomic_int cached = -1;
    int enabled = atomic_load_explicit(&cached, memory_order_relaxed);
    if (enabled < 0) {
        const char * env = getenv("GGML_MOE_CPU_CHUNK_TRACE_OUT");
        enabled = env && env[0] ? 1 : 0;
        atomic_store_explicit(&cached, enabled, memory_order_relaxed);
    }
    return enabled != 0;
}

static const char * ggml_moe_tensor_role(const char * name) {
    if (name) {
        if (strstr(name, "ffn_up_exps")) {
            return "up";
        }
        if (strstr(name, "ffn_gate_exps")) {
            return "gate";
        }
        if (strstr(name, "ffn_down_exps")) {
            return "down";
        }
    }
    return "other";
}


static double ggml_moe_cpu_trace_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1000.0 + (double) ts.tv_nsec / 1000000.0;
}

static FILE * ggml_moe_cpu_chunk_trace_fp(void) {
    static FILE * fp = NULL;
    static int initialized = 0;
    static pthread_mutex_t init_mu = PTHREAD_MUTEX_INITIALIZER;

    pthread_mutex_lock(&init_mu);
    if (!initialized) {
        initialized = 1;
        const char * path = getenv("GGML_MOE_CPU_CHUNK_TRACE_OUT");
        if (path && path[0]) {
            fp = fopen(path, "w");
            if (fp) {
                setvbuf(fp, NULL, _IOLBF, 0);
                fprintf(fp,
                    "seq,role,tensor,type,expert,ith,nth,cne1,ir0_start,ir0_end,ir1_start,ir1_end,src0_bytes,ms\n");
            } else {
                fprintf(stderr, "[moe_cpu_trace] failed to open trace: %s\n", path);
            }
        }
    }
    pthread_mutex_unlock(&init_mu);

    return fp;
}

static void ggml_moe_cpu_chunk_trace_write(
    const char * role,
    const char * tensor,
    enum ggml_type type,
    int expert,
    int ith,
    int nth,
    int64_t cne1,
    int64_t ir0_start,
    int64_t ir0_end,
    int64_t ir1_start,
    int64_t ir1_end,
    size_t src0_bytes,
    double ms) {
    static atomic_int seq = 0;
    static int limit = -1;

    if (limit < 0) {
        const char * env = getenv("GGML_MOE_CPU_CHUNK_TRACE_LIMIT");
        limit = env && env[0] ? atoi(env) : 200000;
        if (limit <= 0) {
            limit = 200000;
        }
    }

    const int cur_seq = atomic_fetch_add_explicit(&seq, 1, memory_order_relaxed);
    if (cur_seq >= limit) {
        return;
    }

    FILE * fp = ggml_moe_cpu_chunk_trace_fp();
    if (!fp) {
        return;
    }

    flockfile(fp);
    fprintf(fp,
        "%d,%s,%s,%d,%d,%d,%d,%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%zu,%.3f\n",
        cur_seq,
        role ? role : "",
        tensor ? tensor : "",
        (int) type,
        expert,
        ith,
        nth,
        cne1,
        ir0_start,
        ir0_end,
        ir1_start,
        ir1_end,
        src0_bytes,
        ms);
    funlockfile(fp);
}

static bool ggml_moe_cpu_fallback_touch_profile_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_MOE_CPU_FALLBACK_TOUCH_PROFILE");
        enabled = env && env[0] && env[0] != '0' ? 1 : 0;
    }
    return enabled != 0;
}

static volatile uint8_t ggml_moe_cpu_touch_sink;

static uint64_t ggml_moe_cpu_touch_pages_us(const void * ptr, size_t size) {
#if defined(__linux__)
    if (!ptr || size == 0) {
        return 0;
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return 0;
    }

    const uintptr_t begin = (uintptr_t) ptr;
    const uintptr_t end = begin + size;
    const uintptr_t aligned_begin = begin & ~(uintptr_t) (page_size - 1);
    const uintptr_t aligned_end = (end + (uintptr_t) page_size - 1) & ~(uintptr_t) (page_size - 1);
    if (aligned_end <= aligned_begin) {
        return 0;
    }

    uint8_t acc = 0;
    const uint64_t t0 = ggml_time_us();
    for (uintptr_t p = aligned_begin; p < aligned_end; p += (uintptr_t) page_size) {
        acc ^= *(const volatile uint8_t *) p;
    }
    ggml_moe_cpu_touch_sink ^= acc;
    return ggml_time_us() - t0;
#else
    (void) ptr;
    (void) size;
    return 0;
#endif
}

static bool ggml_moe_cpu_willneed_enabled(void) {
    static atomic_int cached = -1;
    int enabled = atomic_load_explicit(&cached, memory_order_relaxed);
    if (enabled < 0) {
        const char * env = getenv("GGML_MOE_CPU_WILLNEED");
        enabled = env && env[0] && env[0] != '0' ? 1 : 0;
        atomic_store_explicit(&cached, enabled, memory_order_relaxed);
    }
    return enabled != 0;
}

static int ggml_moe_keep_topk_updown(void) {
    static int keep_topk = -1;
    if (keep_topk < 0) {
        const char * env = getenv("GGML_MOE_KEEP_TOPK_UPDOWN");
        keep_topk = env && env[0] ? atoi(env) : 0;
        if (keep_topk < 0) {
            keep_topk = 0;
        }
    }
    return keep_topk;
}

static bool ggml_moe_keep_topk_applies(const char * name) {
    if (!name) {
        return false;
    }
    if (strstr(name, "ffn_up_exps") || strstr(name, "ffn_down_exps")) {
        return true;
    }

    static int gate_enabled = -1;
    if (gate_enabled < 0) {
        const char * env = getenv("GGML_MOE_KEEP_TOPK_GATE");
        gate_enabled = env && env[0] && strcmp(env, "0") != 0 ? 1 : 0;
    }
    return gate_enabled && strstr(name, "ffn_gate_exps");
}

static int ggml_moe_tensor_layer(const char * name) {
    if (!name) {
        return -1;
    }
    const char * blk = strstr(name, "blk.");
    if (!blk) {
        return -1;
    }
    return atoi(blk + 4);
}

#define GGML_DS4_SPARSE_FUSED_MMVQ_MAX_LAYERS 128
#define GGML_DS4_SPARSE_FUSED_MMVQ_MAX_EXPERTS 512

struct ggml_ds4_sparse_fused_mmvq_membership_state {
    pthread_once_t once;
    bool enabled;
    bool profile_load_attempted;
    bool profile_loaded;
    const char * out;
    const char * profile;
    FILE * fp;
    pthread_mutex_t mutex;
    uint8_t hot[GGML_DS4_SPARSE_FUSED_MMVQ_MAX_LAYERS][GGML_DS4_SPARSE_FUSED_MMVQ_MAX_EXPERTS];
    uint64_t profile_pairs;
    uint64_t profile_duplicates;
    uint64_t profile_invalid;
    uint64_t records;
    uint64_t hit_records;
    uint64_t rows;
    uint64_t hit_rows;
    uint64_t prompt_rows;
    uint64_t prompt_hit_rows;
    uint64_t decode_rows;
    uint64_t decode_hit_rows;
    uint64_t source_bytes;
    uint64_t hit_source_bytes;
    uint64_t fallback_us;
    uint64_t hit_fallback_us;
    uint64_t touch_us;
    uint64_t hit_touch_us;
};

static struct ggml_ds4_sparse_fused_mmvq_membership_state ggml_ds4_sparse_fused_mmvq_membership = {
    .once = PTHREAD_ONCE_INIT,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
};

static void ggml_ds4_sparse_fused_mmvq_membership_add_profile_pair_locked(int layer, int expert) {
    if (layer < 0 || layer >= GGML_DS4_SPARSE_FUSED_MMVQ_MAX_LAYERS ||
            expert < 0 || expert >= GGML_DS4_SPARSE_FUSED_MMVQ_MAX_EXPERTS) {
        ggml_ds4_sparse_fused_mmvq_membership.profile_invalid++;
        return;
    }
    if (ggml_ds4_sparse_fused_mmvq_membership.hot[layer][expert]) {
        ggml_ds4_sparse_fused_mmvq_membership.profile_duplicates++;
        return;
    }
    ggml_ds4_sparse_fused_mmvq_membership.hot[layer][expert] = 1;
    ggml_ds4_sparse_fused_mmvq_membership.profile_pairs++;
}

static bool ggml_ds4_sparse_fused_mmvq_parse_json_int(const char * line, const char * key, int * out) {
    const char * p = strstr(line, key);
    if (!p) {
        return false;
    }
    p += strlen(key);
    while (*p && ((*p < '0' || *p > '9') && *p != '-')) {
        ++p;
    }
    if (!*p) {
        return false;
    }
    *out = atoi(p);
    return true;
}

static void ggml_ds4_sparse_fused_mmvq_membership_load_profile_locked(void) {
    if (ggml_ds4_sparse_fused_mmvq_membership.profile_load_attempted) {
        return;
    }
    ggml_ds4_sparse_fused_mmvq_membership.profile_load_attempted = true;

    const char * path = getenv("GGML_DS4_SPARSE_FUSED_MMVQ_PROFILE");
    if (!path || !path[0]) {
        path = getenv("GGML_DS4_SPARSE_FUSED_MMVQ_PROFILE_TSV");
    }
    if (!path || !path[0]) {
        path = getenv("DS4_SPARSE_FUSED_MMVQ_PROFILE");
    }
    ggml_ds4_sparse_fused_mmvq_membership.profile = path;
    if (!path || !path[0]) {
        fprintf(stderr, "[ds4_sparse_fused_mmvq_membership] profile env is unset\n");
        return;
    }

    FILE * f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[ds4_sparse_fused_mmvq_membership] failed to open profile: %s\n", path);
        return;
    }

    char line[2048];
    int json_layer = -1;
    int json_expert = -1;
    while (fgets(line, sizeof(line), f)) {
        int rank = 0;
        int layer = -1;
        int expert = -1;
        double hot_ms = 0.0;
        double up_hot_ms = 0.0;
        double down_hot_ms = 0.0;
        uint64_t up_calls = 0;
        uint64_t down_calls = 0;
        uint64_t fused_calls = 0;
        if (sscanf(line, "%d\t%d\t%d\t%lf\t%lf\t%lf\t%" SCNu64 "\t%" SCNu64 "\t%" SCNu64,
                    &rank, &layer, &expert, &hot_ms, &up_hot_ms, &down_hot_ms,
                    &up_calls, &down_calls, &fused_calls) >= 3) {
            ggml_ds4_sparse_fused_mmvq_membership_add_profile_pair_locked(layer, expert);
            continue;
        }

        if (strchr(line, '{')) {
            json_layer = -1;
            json_expert = -1;
        }
        (void) ggml_ds4_sparse_fused_mmvq_parse_json_int(line, "\"layer\"", &json_layer);
        (void) ggml_ds4_sparse_fused_mmvq_parse_json_int(line, "\"expert\"", &json_expert);
        if (strchr(line, '}') && json_layer >= 0 && json_expert >= 0) {
            ggml_ds4_sparse_fused_mmvq_membership_add_profile_pair_locked(json_layer, json_expert);
            json_layer = -1;
            json_expert = -1;
        }
    }
    fclose(f);

    ggml_ds4_sparse_fused_mmvq_membership.profile_loaded =
        ggml_ds4_sparse_fused_mmvq_membership.profile_pairs > 0;
    fprintf(stderr,
            "[ds4_sparse_fused_mmvq_membership] loaded profile=%s pairs=%" PRIu64
            " duplicates=%" PRIu64 " invalid=%" PRIu64 "\n",
            path,
            ggml_ds4_sparse_fused_mmvq_membership.profile_pairs,
            ggml_ds4_sparse_fused_mmvq_membership.profile_duplicates,
            ggml_ds4_sparse_fused_mmvq_membership.profile_invalid);
}

static void ggml_ds4_sparse_fused_mmvq_membership_report(void) {
    if (!ggml_ds4_sparse_fused_mmvq_membership.enabled) {
        return;
    }
    if (ggml_ds4_sparse_fused_mmvq_membership.fp) {
        fclose(ggml_ds4_sparse_fused_mmvq_membership.fp);
        ggml_ds4_sparse_fused_mmvq_membership.fp = NULL;
    }
    fprintf(stderr,
            "[ds4_sparse_fused_mmvq_membership] profile_loaded=%d pairs=%" PRIu64
            " records=%" PRIu64 " hit_records=%" PRIu64
            " rows=%" PRIu64 " hit_rows=%" PRIu64
            " prompt_rows=%" PRIu64 " prompt_hit_rows=%" PRIu64
            " decode_rows=%" PRIu64 " decode_hit_rows=%" PRIu64
            " source_bytes=%" PRIu64 " hit_source_bytes=%" PRIu64
            " fallback_us=%" PRIu64 " hit_fallback_us=%" PRIu64
            " touch_us=%" PRIu64 " hit_touch_us=%" PRIu64 "\n",
            ggml_ds4_sparse_fused_mmvq_membership.profile_loaded ? 1 : 0,
            ggml_ds4_sparse_fused_mmvq_membership.profile_pairs,
            ggml_ds4_sparse_fused_mmvq_membership.records,
            ggml_ds4_sparse_fused_mmvq_membership.hit_records,
            ggml_ds4_sparse_fused_mmvq_membership.rows,
            ggml_ds4_sparse_fused_mmvq_membership.hit_rows,
            ggml_ds4_sparse_fused_mmvq_membership.prompt_rows,
            ggml_ds4_sparse_fused_mmvq_membership.prompt_hit_rows,
            ggml_ds4_sparse_fused_mmvq_membership.decode_rows,
            ggml_ds4_sparse_fused_mmvq_membership.decode_hit_rows,
            ggml_ds4_sparse_fused_mmvq_membership.source_bytes,
            ggml_ds4_sparse_fused_mmvq_membership.hit_source_bytes,
            ggml_ds4_sparse_fused_mmvq_membership.fallback_us,
            ggml_ds4_sparse_fused_mmvq_membership.hit_fallback_us,
            ggml_ds4_sparse_fused_mmvq_membership.touch_us,
            ggml_ds4_sparse_fused_mmvq_membership.hit_touch_us);
}

static void ggml_ds4_sparse_fused_mmvq_membership_init(void) {
    ggml_ds4_sparse_fused_mmvq_membership.out =
        getenv("GGML_DS4_SPARSE_FUSED_MMVQ_MEMBERSHIP_OUT");
    ggml_ds4_sparse_fused_mmvq_membership.enabled =
        ggml_ds4_sparse_fused_mmvq_membership.out &&
        ggml_ds4_sparse_fused_mmvq_membership.out[0];
    if (ggml_ds4_sparse_fused_mmvq_membership.enabled) {
        atexit(ggml_ds4_sparse_fused_mmvq_membership_report);
    }
}

static bool ggml_ds4_sparse_fused_mmvq_membership_enabled(void) {
    pthread_once(&ggml_ds4_sparse_fused_mmvq_membership.once,
                 ggml_ds4_sparse_fused_mmvq_membership_init);
    return ggml_ds4_sparse_fused_mmvq_membership.enabled;
}

static FILE * ggml_ds4_sparse_fused_mmvq_membership_fp_locked(void) {
    if (ggml_ds4_sparse_fused_mmvq_membership.fp) {
        return ggml_ds4_sparse_fused_mmvq_membership.fp;
    }
    const char * path = ggml_ds4_sparse_fused_mmvq_membership.out;
    if (!path || !path[0]) {
        return NULL;
    }
    ggml_ds4_sparse_fused_mmvq_membership.fp = fopen(path, "w");
    if (!ggml_ds4_sparse_fused_mmvq_membership.fp) {
        fprintf(stderr, "[ds4_sparse_fused_mmvq_membership] failed to open output: %s\n", path);
        return NULL;
    }
    setvbuf(ggml_ds4_sparse_fused_mmvq_membership.fp, NULL, _IOLBF, 0);
    fprintf(ggml_ds4_sparse_fused_mmvq_membership.fp,
            "seq,phase,role,tensor,type,layer,expert,cne1,profile_hit,expert_bytes,source_bytes,"
            "fallback_us_est,touch_us,pack_mmap_source,profile_path,profile_pairs_loaded\n");
    return ggml_ds4_sparse_fused_mmvq_membership.fp;
}

static void ggml_ds4_sparse_fused_mmvq_membership_record(
        const char * name,
        int src0_type,
        bool prompt_phase,
        int expert_idx,
        int64_t count,
        size_t expert_bytes,
        uint64_t fallback_us,
        uint64_t touch_us,
        bool pack_mmap_source) {
    if (!ggml_ds4_sparse_fused_mmvq_membership_enabled() || count <= 0) {
        return;
    }

    const char * role = ggml_moe_tensor_role(name);
    if (strcmp(role, "up") != 0 && strcmp(role, "down") != 0) {
        return;
    }
    const int layer = ggml_moe_tensor_layer(name);
    if (layer < 0 || expert_idx < 0) {
        return;
    }

    pthread_mutex_lock(&ggml_ds4_sparse_fused_mmvq_membership.mutex);
    ggml_ds4_sparse_fused_mmvq_membership_load_profile_locked();
    const bool profile_hit =
        layer < GGML_DS4_SPARSE_FUSED_MMVQ_MAX_LAYERS &&
        expert_idx < GGML_DS4_SPARSE_FUSED_MMVQ_MAX_EXPERTS &&
        ggml_ds4_sparse_fused_mmvq_membership.hot[layer][expert_idx] != 0;

    const uint64_t rows = (uint64_t) count;
    const uint64_t source_bytes = (uint64_t) expert_bytes;

    const uint64_t seq = ggml_ds4_sparse_fused_mmvq_membership.records++;
    ggml_ds4_sparse_fused_mmvq_membership.rows += rows;
    ggml_ds4_sparse_fused_mmvq_membership.source_bytes += source_bytes;
    ggml_ds4_sparse_fused_mmvq_membership.fallback_us += fallback_us;
    ggml_ds4_sparse_fused_mmvq_membership.touch_us += touch_us;
    if (prompt_phase) {
        ggml_ds4_sparse_fused_mmvq_membership.prompt_rows += rows;
    } else {
        ggml_ds4_sparse_fused_mmvq_membership.decode_rows += rows;
    }
    if (profile_hit) {
        ggml_ds4_sparse_fused_mmvq_membership.hit_records++;
        ggml_ds4_sparse_fused_mmvq_membership.hit_rows += rows;
        ggml_ds4_sparse_fused_mmvq_membership.hit_source_bytes += source_bytes;
        ggml_ds4_sparse_fused_mmvq_membership.hit_fallback_us += fallback_us;
        ggml_ds4_sparse_fused_mmvq_membership.hit_touch_us += touch_us;
        if (prompt_phase) {
            ggml_ds4_sparse_fused_mmvq_membership.prompt_hit_rows += rows;
        } else {
            ggml_ds4_sparse_fused_mmvq_membership.decode_hit_rows += rows;
        }
    }

    FILE * fp = ggml_ds4_sparse_fused_mmvq_membership_fp_locked();
    if (fp) {
        fprintf(fp,
                "%" PRIu64 ",%s,%s,%s,%d,%d,%d,%" PRId64 ",%d,%zu,%" PRIu64
                ",%" PRIu64 ",%" PRIu64 ",%d,%s,%" PRIu64 "\n",
                seq,
                prompt_phase ? "prompt" : "decode",
                role,
                name ? name : "",
                src0_type,
                layer,
                expert_idx,
                count,
                profile_hit ? 1 : 0,
                expert_bytes,
                source_bytes,
                fallback_us,
                touch_us,
                pack_mmap_source ? 1 : 0,
                ggml_ds4_sparse_fused_mmvq_membership.profile ?
                    ggml_ds4_sparse_fused_mmvq_membership.profile : "",
                ggml_ds4_sparse_fused_mmvq_membership.profile_pairs);
    }
    pthread_mutex_unlock(&ggml_ds4_sparse_fused_mmvq_membership.mutex);
}

enum { GGML_MOE_KEEP_TOPK_MAX_SCHEDULE = 16 };

struct ggml_moe_keep_topk_schedule_entry {
    int start;
    int end;
    int keep_topk;
};

struct ggml_moe_keep_topk_schedule_state {
    int initialized;
    int count;
    struct ggml_moe_keep_topk_schedule_entry entries[GGML_MOE_KEEP_TOPK_MAX_SCHEDULE];
};

static void ggml_moe_keep_topk_schedule_init(
        struct ggml_moe_keep_topk_schedule_state * state,
        const char * env_name) {
    if (state->initialized) {
        return;
    }
    state->initialized = 1;
    const char * env = getenv(env_name);
    if (!env || !env[0]) {
        return;
    }

    const char * p = env;
    while (*p && state->count < GGML_MOE_KEEP_TOPK_MAX_SCHEDULE) {
        while (*p == ' ' || *p == '\t' || *p == ',') {
            ++p;
        }
        int start = -1;
        int end = -1;
        int keep = 0;
        int consumed = 0;
        if (sscanf(p, "%d-%d:%d%n", &start, &end, &keep, &consumed) == 3 ||
            sscanf(p, "%d:%d%n", &start, &keep, &consumed) == 2) {
            if (end < 0) {
                end = start;
            }
            if (start > end) {
                const int tmp = start;
                start = end;
                end = tmp;
            }
            if (keep < 0) {
                keep = 0;
            }
            state->entries[state->count++] = (struct ggml_moe_keep_topk_schedule_entry) {
                start,
                end,
                keep,
            };
            p += consumed;
            while (*p && *p != ',') {
                ++p;
            }
            continue;
        }
        break;
    }
}

static int ggml_moe_keep_topk_schedule_lookup(
        struct ggml_moe_keep_topk_schedule_state * state,
        const char * env_name,
        int layer) {
    ggml_moe_keep_topk_schedule_init(state, env_name);
    for (int i = 0; i < state->count; ++i) {
        if (layer >= state->entries[i].start && layer <= state->entries[i].end) {
            return state->entries[i].keep_topk;
        }
    }
    return -1;
}

static int ggml_moe_keep_topk_for_tensor(const char * name) {
    if (!ggml_moe_keep_topk_applies(name)) {
        return 0;
    }

    const int fallback_keep_topk = ggml_moe_keep_topk_updown();
    const int layer = ggml_moe_tensor_layer(name);

    const char * role = ggml_moe_tensor_role(name);
    static struct ggml_moe_keep_topk_schedule_state up_schedule;
    static struct ggml_moe_keep_topk_schedule_state gate_schedule;
    static struct ggml_moe_keep_topk_schedule_state down_schedule;
    if (strcmp(role, "up") == 0) {
        const int keep = ggml_moe_keep_topk_schedule_lookup(
                &up_schedule, "GGML_MOE_KEEP_TOPK_UP_LAYER_SCHEDULE", layer);
        if (keep >= 0) {
            return keep;
        }
    } else if (strcmp(role, "gate") == 0) {
        const int keep = ggml_moe_keep_topk_schedule_lookup(
                &gate_schedule, "GGML_MOE_KEEP_TOPK_GATE_LAYER_SCHEDULE", layer);
        if (keep >= 0) {
            return keep;
        }
    } else if (strcmp(role, "down") == 0) {
        const int keep = ggml_moe_keep_topk_schedule_lookup(
                &down_schedule, "GGML_MOE_KEEP_TOPK_DOWN_LAYER_SCHEDULE", layer);
        if (keep >= 0) {
            return keep;
        }
    }

    static struct ggml_moe_keep_topk_schedule_state schedule;
    const int scheduled_keep = ggml_moe_keep_topk_schedule_lookup(
            &schedule, "GGML_MOE_KEEP_TOPK_LAYER_SCHEDULE", layer);
    if (scheduled_keep >= 0) {
        return scheduled_keep;
    }

    static int initialized = 0;
    static int layer_start = -1;
    static int layer_end = -1;
    static int layer_keep_topk = 0;
    if (!initialized) {
        initialized = 1;
        const char * range = getenv("GGML_MOE_KEEP_TOPK_LAYER_RANGE");
        const char * value = getenv("GGML_MOE_KEEP_TOPK_LAYER_VALUE");
        if (range && range[0] && value && value[0]) {
            int start = -1;
            int end = -1;
            if (sscanf(range, "%d-%d", &start, &end) == 2) {
                if (start > end) {
                    const int tmp = start;
                    start = end;
                    end = tmp;
                }
                layer_start = start;
                layer_end = end;
                layer_keep_topk = atoi(value);
                if (layer_keep_topk < 0) {
                    layer_keep_topk = 0;
                }
            }
        }
    }

    if (layer_keep_topk > 0 && layer >= layer_start && layer <= layer_end) {
        return layer_keep_topk;
    }

    return fallback_keep_topk;
}

static void ggml_moe_cpu_willneed_pages(const void * ptr, size_t size) {
#if defined(__linux__)
    if (!ptr || size == 0) {
        return;
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return;
    }

    const uintptr_t begin = (uintptr_t) ptr;
    const uintptr_t end = begin + size;
    const uintptr_t aligned_begin = begin & ~(uintptr_t) (page_size - 1);
    const uintptr_t aligned_end = (end + (uintptr_t) page_size - 1) & ~(uintptr_t) (page_size - 1);
    if (aligned_end > aligned_begin) {
        (void) madvise((void *) aligned_begin, aligned_end - aligned_begin, MADV_WILLNEED);
    }
#else
    (void) ptr;
    (void) size;
#endif
}

static const int8_t ggml_moe_stream_mxfp4_values[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};
