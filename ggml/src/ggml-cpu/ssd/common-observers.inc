// CPU MoE extension, included in the CPU translation unit.
#define GGML_MOE_FALLBACK_NAME_LEN 96
#define GGML_MOE_FALLBACK_REASON_PROFILE_MAX 32768
#define GGML_MOE_FALLBACK_REASON_LEN 64

struct ggml_moe_fallback_reason_profile_entry {
    char tensor[GGML_MOE_FALLBACK_NAME_LEN];
    char role[16];
    char phase[8];
    char batch_reason[GGML_MOE_FALLBACK_REASON_LEN];
    char single_reason[GGML_MOE_FALLBACK_REASON_LEN];
    char final_reason[GGML_MOE_FALLBACK_REASON_LEN];
    int expert_idx;
    int src0_type;
    size_t expert_bytes;
    uint64_t rows;
    uint64_t calls;
    uint64_t fallback_us;
    uint64_t batch_attempts;
    uint64_t batch_accepts;
    uint64_t single_attempts;
    uint64_t single_accepts;
};

struct ggml_moe_fallback_reason_profile_state {
    bool enabled;
    const char * out;
    struct ggml_moe_fallback_reason_profile_entry entries[GGML_MOE_FALLBACK_REASON_PROFILE_MAX];
    int n_entries;
    uint64_t dropped;
};

static struct ggml_moe_fallback_reason_profile_state ggml_moe_fallback_reason_profile;

#define GGML_MOE_FALLBACK_SOURCE_PROBE_MAX 32768
#define GGML_MOE_FALLBACK_SOURCE_PROBE_NAME_LEN 96
#define GGML_MOE_FALLBACK_SOURCE_PROBE_BUFT_LEN 64

struct ggml_moe_fallback_source_probe_entry {
    char tensor[GGML_MOE_FALLBACK_SOURCE_PROBE_NAME_LEN];
    char role[16];
    char phase[8];
    char batch_reason[GGML_MOE_FALLBACK_REASON_LEN];
    char single_reason[GGML_MOE_FALLBACK_REASON_LEN];
    char final_reason[GGML_MOE_FALLBACK_REASON_LEN];
    char src0_buft[GGML_MOE_FALLBACK_SOURCE_PROBE_BUFT_LEN];
    char src1_buft[GGML_MOE_FALLBACK_SOURCE_PROBE_BUFT_LEN];
    char ids_buft[GGML_MOE_FALLBACK_SOURCE_PROBE_BUFT_LEN];
    char dst_buft[GGML_MOE_FALLBACK_SOURCE_PROBE_BUFT_LEN];
    int expert_idx;
    int src0_type;
    int src1_type;
    int ids_type;
    int dst_type;
    int64_t src0_ne[4];
    int64_t src1_ne[4];
    int64_t ids_ne[4];
    int64_t dst_ne[4];
    size_t src0_nbytes;
    size_t src1_nbytes;
    size_t ids_nbytes;
    size_t dst_nbytes;
    uint64_t rows;
    uint64_t calls;
    uint64_t fallback_us;
};

struct ggml_moe_fallback_source_probe_state {
    bool initialized;
    bool registered;
    bool enabled;
    const char * out;
    struct ggml_moe_fallback_source_probe_entry entries[GGML_MOE_FALLBACK_SOURCE_PROBE_MAX];
    int n_entries;
    uint64_t dropped;
};

static struct ggml_moe_fallback_source_probe_state ggml_moe_fallback_source_probe;


static bool ggml_moe_name_filter_matches_any(const char * filter, const char * name) {
    if (!filter || !filter[0] || !name) {
        return false;
    }

    const char * p = filter;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == ',' || *p == ':') {
            ++p;
        }
        const char * start = p;
        while (*p && *p != ',' && *p != ':') {
            ++p;
        }
        const char * end = p;
        while (end > start && (end[-1] == ' ' || end[-1] == '\t')) {
            --end;
        }
        const size_t len = (size_t) (end - start);
        if (len > 0) {
            for (const char * hit = name; *hit; ++hit) {
                if (strncmp(hit, start, len) == 0) {
                    return true;
                }
            }
        }
    }
    return false;
}

static bool ggml_moe_stream_one_gpu_only_filter_matches(const char * name) {
    return ggml_moe_name_filter_matches_any(getenv("GGML_MOE_STREAM_ONE_GPU_ONLY_FILTER"), name);
}

static void ggml_moe_fallback_reason_profile_report(void) {
    if (!ggml_moe_fallback_reason_profile.enabled ||
            !ggml_moe_fallback_reason_profile.out ||
            !ggml_moe_fallback_reason_profile.out[0]) {
        return;
    }

    FILE * f = fopen(ggml_moe_fallback_reason_profile.out, "w");
    if (!f) {
        fprintf(stderr,
                "[moe_fallback_reason_profile] open failed: %s\n",
                ggml_moe_fallback_reason_profile.out);
        return;
    }

    fprintf(f, "rank,role,tensor,phase,expert_idx,src0_type,expert_bytes,rows,calls,fallback_us,batch_reason,single_reason,final_reason,batch_attempts,batch_accepts,single_attempts,single_accepts\n");
    for (int i = 0; i < ggml_moe_fallback_reason_profile.n_entries; ++i) {
        const struct ggml_moe_fallback_reason_profile_entry * e =
            &ggml_moe_fallback_reason_profile.entries[i];
        fprintf(f,
                "%d,%s,%s,%s,%d,%d,%zu,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%s,%s,%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                i + 1,
                e->role,
                e->tensor,
                e->phase,
                e->expert_idx,
                e->src0_type,
                e->expert_bytes,
                e->rows,
                e->calls,
                e->fallback_us,
                e->batch_reason,
                e->single_reason,
                e->final_reason,
                e->batch_attempts,
                e->batch_accepts,
                e->single_attempts,
                e->single_accepts);
    }
    fclose(f);

    fprintf(stderr,
            "[moe_fallback_reason_profile] written: %s entries=%d dropped=%" PRIu64 "\n",
            ggml_moe_fallback_reason_profile.out,
            ggml_moe_fallback_reason_profile.n_entries,
            ggml_moe_fallback_reason_profile.dropped);
}

static bool ggml_moe_fallback_reason_profile_enabled(void);

static void ggml_moe_fallback_reason_profile_record(
        const char * role,
        const char * tensor,
        int src0_type,
        bool prompt_phase,
        int expert_idx,
        int64_t rows,
        size_t expert_bytes,
        uint64_t fallback_us,
        const char * batch_reason,
        const char * single_reason,
        const char * final_reason,
        bool batch_attempted,
        bool batch_accepted,
        bool single_attempted,
        bool single_accepted) {
    if (!ggml_moe_fallback_reason_profile_enabled() || rows <= 0) {
        return;
    }

    const char * safe_role = role ? role : "unknown";
    const char * safe_tensor = tensor ? tensor : "<unnamed>";
    const char * safe_phase = prompt_phase ? "prompt" : "decode";
    const char * safe_batch_reason = batch_reason ? batch_reason : "unknown";
    const char * safe_single_reason = single_reason ? single_reason : "unknown";
    const char * safe_final_reason = final_reason ? final_reason : "unknown";
    int idx = -1;

    for (int i = 0; i < ggml_moe_fallback_reason_profile.n_entries; ++i) {
        struct ggml_moe_fallback_reason_profile_entry * e =
            &ggml_moe_fallback_reason_profile.entries[i];
        if (e->expert_idx == expert_idx &&
                e->src0_type == src0_type &&
                e->expert_bytes == expert_bytes &&
                strcmp(e->role, safe_role) == 0 &&
                strcmp(e->phase, safe_phase) == 0 &&
                strncmp(e->tensor, safe_tensor, GGML_MOE_FALLBACK_NAME_LEN) == 0 &&
                strncmp(e->batch_reason, safe_batch_reason, GGML_MOE_FALLBACK_REASON_LEN) == 0 &&
                strncmp(e->single_reason, safe_single_reason, GGML_MOE_FALLBACK_REASON_LEN) == 0 &&
                strncmp(e->final_reason, safe_final_reason, GGML_MOE_FALLBACK_REASON_LEN) == 0) {
            idx = i;
            break;
        }
    }

    if (idx < 0) {
        if (ggml_moe_fallback_reason_profile.n_entries >= GGML_MOE_FALLBACK_REASON_PROFILE_MAX) {
            ggml_moe_fallback_reason_profile.dropped++;
            return;
        }
        idx = ggml_moe_fallback_reason_profile.n_entries++;
        struct ggml_moe_fallback_reason_profile_entry * e =
            &ggml_moe_fallback_reason_profile.entries[idx];
        snprintf(e->tensor, GGML_MOE_FALLBACK_NAME_LEN, "%s", safe_tensor);
        snprintf(e->role, sizeof(e->role), "%s", safe_role);
        snprintf(e->phase, sizeof(e->phase), "%s", safe_phase);
        snprintf(e->batch_reason, GGML_MOE_FALLBACK_REASON_LEN, "%s", safe_batch_reason);
        snprintf(e->single_reason, GGML_MOE_FALLBACK_REASON_LEN, "%s", safe_single_reason);
        snprintf(e->final_reason, GGML_MOE_FALLBACK_REASON_LEN, "%s", safe_final_reason);
        e->expert_idx = expert_idx;
        e->src0_type = src0_type;
        e->expert_bytes = expert_bytes;
    }

    struct ggml_moe_fallback_reason_profile_entry * e =
        &ggml_moe_fallback_reason_profile.entries[idx];
    e->rows += (uint64_t) rows;
    e->calls++;
    e->fallback_us += fallback_us;
    if (batch_attempted) {
        e->batch_attempts++;
    }
    if (batch_accepted) {
        e->batch_accepts++;
    }
    if (single_attempted) {
        e->single_attempts++;
    }
    if (single_accepted) {
        e->single_accepts++;
    }
}

static void ggml_moe_fallback_source_probe_report(void);

static const char * ggml_moe_fallback_source_probe_buft_name(const struct ggml_tensor * tensor) {
    if (tensor == NULL || tensor->buffer == NULL) {
        return "none";
    }
    const char * name = ggml_backend_buffer_name(tensor->buffer);
    return name ? name : "unknown";
}

static void ggml_moe_fallback_source_probe_copy_ne(int64_t dst_ne[4], const struct ggml_tensor * tensor) {
    for (int i = 0; i < 4; ++i) {
        dst_ne[i] = tensor ? tensor->ne[i] : -1;
    }
}

static bool ggml_moe_fallback_source_probe_enabled(void) {
    if (!ggml_moe_fallback_source_probe.initialized) {
        ggml_moe_fallback_source_probe.initialized = true;
        ggml_moe_fallback_source_probe.out = getenv("GGML_MOE_FALLBACK_SOURCE_PROBE_OUT");
        ggml_moe_fallback_source_probe.enabled =
            ggml_moe_fallback_source_probe.out &&
            ggml_moe_fallback_source_probe.out[0] &&
            ggml_moe_fallback_source_probe.out[0] != '0';
        if (ggml_moe_fallback_source_probe.enabled && !ggml_moe_fallback_source_probe.registered) {
            ggml_moe_fallback_source_probe.registered = true;
            atexit(ggml_moe_fallback_source_probe_report);
        }
    }
    return ggml_moe_fallback_source_probe.enabled;
}

static void ggml_moe_fallback_source_probe_report(void) {
    if (!ggml_moe_fallback_source_probe.enabled ||
            !ggml_moe_fallback_source_probe.out ||
            !ggml_moe_fallback_source_probe.out[0]) {
        return;
    }

    FILE * f = fopen(ggml_moe_fallback_source_probe.out, "w");
    if (!f) {
        fprintf(stderr, "[moe_fallback_source_probe] open failed: %s\n", ggml_moe_fallback_source_probe.out);
        return;
    }

    fprintf(f,
            "rank,role,tensor,phase,expert_idx,rows,calls,fallback_us,batch_reason,single_reason,final_reason,"
            "src0_buft,src1_buft,ids_buft,dst_buft,src0_type,src1_type,ids_type,dst_type,"
            "src0_nbytes,src1_nbytes,ids_nbytes,dst_nbytes,"
            "src0_ne0,src0_ne1,src0_ne2,src0_ne3,src1_ne0,src1_ne1,src1_ne2,src1_ne3,"
            "ids_ne0,ids_ne1,ids_ne2,ids_ne3,dst_ne0,dst_ne1,dst_ne2,dst_ne3\n");
    for (int i = 0; i < ggml_moe_fallback_source_probe.n_entries; ++i) {
        const struct ggml_moe_fallback_source_probe_entry * e = &ggml_moe_fallback_source_probe.entries[i];
        fprintf(f,
                "%d,%s,%s,%s,%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%s,%s,%s,"
                "%s,%s,%s,%s,%d,%d,%d,%d,%zu,%zu,%zu,%zu,"
                "%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ","
                "%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "\n",
                i + 1,
                e->role,
                e->tensor,
                e->phase,
                e->expert_idx,
                e->rows,
                e->calls,
                e->fallback_us,
                e->batch_reason,
                e->single_reason,
                e->final_reason,
                e->src0_buft,
                e->src1_buft,
                e->ids_buft,
                e->dst_buft,
                e->src0_type,
                e->src1_type,
                e->ids_type,
                e->dst_type,
                e->src0_nbytes,
                e->src1_nbytes,
                e->ids_nbytes,
                e->dst_nbytes,
                e->src0_ne[0], e->src0_ne[1], e->src0_ne[2], e->src0_ne[3],
                e->src1_ne[0], e->src1_ne[1], e->src1_ne[2], e->src1_ne[3],
                e->ids_ne[0], e->ids_ne[1], e->ids_ne[2], e->ids_ne[3],
                e->dst_ne[0], e->dst_ne[1], e->dst_ne[2], e->dst_ne[3]);
    }
    fclose(f);

    fprintf(stderr,
            "[moe_fallback_source_probe] written: %s entries=%d dropped=%" PRIu64 "\n",
            ggml_moe_fallback_source_probe.out,
            ggml_moe_fallback_source_probe.n_entries,
            ggml_moe_fallback_source_probe.dropped);
}

static void ggml_moe_fallback_source_probe_record(
        const char * role,
        const char * tensor_name,
        bool prompt_phase,
        int expert_idx,
        int64_t rows,
        uint64_t fallback_us,
        const char * batch_reason,
        const char * single_reason,
        const char * final_reason,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * ids,
        const struct ggml_tensor * dst) {
    if (!ggml_moe_fallback_source_probe_enabled() || rows <= 0) {
        return;
    }

    const char * safe_role = role ? role : "unknown";
    const char * safe_tensor = tensor_name ? tensor_name : "<unnamed>";
    const char * safe_phase = prompt_phase ? "prompt" : "decode";
    const char * safe_batch_reason = batch_reason ? batch_reason : "unknown";
    const char * safe_single_reason = single_reason ? single_reason : "unknown";
    const char * safe_final_reason = final_reason ? final_reason : "unknown";
    const char * src0_buft = ggml_moe_fallback_source_probe_buft_name(src0);
    const char * src1_buft = ggml_moe_fallback_source_probe_buft_name(src1);
    const char * ids_buft = ggml_moe_fallback_source_probe_buft_name(ids);
    const char * dst_buft = ggml_moe_fallback_source_probe_buft_name(dst);
    int idx = -1;

    for (int i = 0; i < ggml_moe_fallback_source_probe.n_entries; ++i) {
        struct ggml_moe_fallback_source_probe_entry * e = &ggml_moe_fallback_source_probe.entries[i];
        if (e->expert_idx == expert_idx &&
                strcmp(e->role, safe_role) == 0 &&
                strcmp(e->phase, safe_phase) == 0 &&
                strncmp(e->tensor, safe_tensor, GGML_MOE_FALLBACK_SOURCE_PROBE_NAME_LEN) == 0 &&
                strncmp(e->batch_reason, safe_batch_reason, GGML_MOE_FALLBACK_REASON_LEN) == 0 &&
                strncmp(e->single_reason, safe_single_reason, GGML_MOE_FALLBACK_REASON_LEN) == 0 &&
                strncmp(e->final_reason, safe_final_reason, GGML_MOE_FALLBACK_REASON_LEN) == 0 &&
                strncmp(e->src0_buft, src0_buft, GGML_MOE_FALLBACK_SOURCE_PROBE_BUFT_LEN) == 0 &&
                strncmp(e->src1_buft, src1_buft, GGML_MOE_FALLBACK_SOURCE_PROBE_BUFT_LEN) == 0 &&
                strncmp(e->ids_buft, ids_buft, GGML_MOE_FALLBACK_SOURCE_PROBE_BUFT_LEN) == 0 &&
                strncmp(e->dst_buft, dst_buft, GGML_MOE_FALLBACK_SOURCE_PROBE_BUFT_LEN) == 0) {
            idx = i;
            break;
        }
    }

    if (idx < 0) {
        if (ggml_moe_fallback_source_probe.n_entries >= GGML_MOE_FALLBACK_SOURCE_PROBE_MAX) {
            ggml_moe_fallback_source_probe.dropped++;
            return;
        }
        idx = ggml_moe_fallback_source_probe.n_entries++;
        struct ggml_moe_fallback_source_probe_entry * e = &ggml_moe_fallback_source_probe.entries[idx];
        snprintf(e->tensor, sizeof(e->tensor), "%s", safe_tensor);
        snprintf(e->role, sizeof(e->role), "%s", safe_role);
        snprintf(e->phase, sizeof(e->phase), "%s", safe_phase);
        snprintf(e->batch_reason, sizeof(e->batch_reason), "%s", safe_batch_reason);
        snprintf(e->single_reason, sizeof(e->single_reason), "%s", safe_single_reason);
        snprintf(e->final_reason, sizeof(e->final_reason), "%s", safe_final_reason);
        snprintf(e->src0_buft, sizeof(e->src0_buft), "%s", src0_buft);
        snprintf(e->src1_buft, sizeof(e->src1_buft), "%s", src1_buft);
        snprintf(e->ids_buft, sizeof(e->ids_buft), "%s", ids_buft);
        snprintf(e->dst_buft, sizeof(e->dst_buft), "%s", dst_buft);
        e->expert_idx = expert_idx;
        e->src0_type = src0 ? (int) src0->type : -1;
        e->src1_type = src1 ? (int) src1->type : -1;
        e->ids_type = ids ? (int) ids->type : -1;
        e->dst_type = dst ? (int) dst->type : -1;
        e->src0_nbytes = src0 ? ggml_nbytes(src0) : 0;
        e->src1_nbytes = src1 ? ggml_nbytes(src1) : 0;
        e->ids_nbytes = ids ? ggml_nbytes(ids) : 0;
        e->dst_nbytes = dst ? ggml_nbytes(dst) : 0;
        ggml_moe_fallback_source_probe_copy_ne(e->src0_ne, src0);
        ggml_moe_fallback_source_probe_copy_ne(e->src1_ne, src1);
        ggml_moe_fallback_source_probe_copy_ne(e->ids_ne, ids);
        ggml_moe_fallback_source_probe_copy_ne(e->dst_ne, dst);
    }

    struct ggml_moe_fallback_source_probe_entry * e = &ggml_moe_fallback_source_probe.entries[idx];
    e->rows += (uint64_t) rows;
    e->calls++;
    e->fallback_us += fallback_us;
}
