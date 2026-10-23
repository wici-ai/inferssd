// CPU MoE extension, included in the CPU translation unit.
struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};


static void ggml_compute_forward_mul_mat_id_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    ggml_vec_dot_t    const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float tmp[16];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                const int64_t _i12 = ir1; // logical row index for this expert

                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                const int id       = row_mapping.i1; // selected expert index

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2; // row index in src1

                const int64_t  i1 = id;  // selected expert index
                const int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                float * dst_col = (float *) ((char *) dst->data + (i1*nb1 + i2*nb2));

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*sizeof(float));
            }
        }
    }
}

static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {

    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}

static bool ggml_kimi_pipeline_v2_active(void) {
    const char * active = getenv("GGML_KIMI_PIPELINE_V2_ACTIVE");
    const char * mode = getenv("GGML_KIMI_PIPELINE_V2_MODE");
    return active && active[0] && active[0] != '0' && mode &&
        (strcmp(mode, "depth1") == 0 || strcmp(mode, "pipeline") == 0);
}

static uint64_t ggml_kimi_pipeline_m1_fnv1a(const void * data, size_t size) {
    const unsigned char * bytes = (const unsigned char *) data;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < size; ++i) {
        hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    }
    return hash;
}

static bool ggml_kimi_pipeline_m1_checkpoint_wants(const struct ggml_tensor * tensor) {
    if (!ggml_kimi_pipeline_v2_active() || !tensor || !tensor->data ||
            tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) {
        return false;
    }
    const char * path = getenv("GGML_KIMI_PIPELINE_M1_CHECKPOINT_OUT");
    if (!path || !path[0]) {
        return false;
    }
    return strstr(tensor->name, "ffn_moe_situ") != NULL ||
        strstr(tensor->name, "ffn_moe_weighted") != NULL ||
        strstr(tensor->name, "ffn_moe_out") != NULL;
}

static pthread_mutex_t ggml_kimi_pipeline_m1_checkpoint_mu = PTHREAD_MUTEX_INITIALIZER;
static atomic_uint_fast64_t ggml_kimi_pipeline_m1_checkpoint_seq = 0;
static bool ggml_kimi_pipeline_m1_checkpoint_header = false;

static void ggml_kimi_pipeline_m1_checkpoint_record(const struct ggml_tensor * tensor) {
    const char * path = getenv("GGML_KIMI_PIPELINE_M1_CHECKPOINT_OUT");
    if (!path || !path[0] || !tensor || !tensor->data) {
        return;
    }
    const char * run = getenv("GGML_KIMI_PIPELINE_M1_RUN");
    const char * mode = getenv("GGML_KIMI_PIPELINE_V2_MODE");
    const size_t bytes = ggml_nbytes(tensor);
    const uint64_t hash = ggml_kimi_pipeline_m1_fnv1a(tensor->data, bytes);
    const uint64_t seq = atomic_fetch_add_explicit(
        &ggml_kimi_pipeline_m1_checkpoint_seq, 1, memory_order_relaxed);
    pthread_mutex_lock(&ggml_kimi_pipeline_m1_checkpoint_mu);
    FILE * file = fopen(path, "a");
    if (file) {
        if (!ggml_kimi_pipeline_m1_checkpoint_header) {
            fprintf(file, "seq,run,mode,tensor,op,bytes,fnv\n");
            ggml_kimi_pipeline_m1_checkpoint_header = true;
        }
        fprintf(file, "%" PRIu64 ",%s,%s,%s,%s,%zu,%016" PRIx64 "\n",
                seq, run ? run : "", mode ? mode : "", tensor->name,
                ggml_op_name(tensor->op), bytes, hash);
        fclose(file);
    }
    pthread_mutex_unlock(&ggml_kimi_pipeline_m1_checkpoint_mu);
}

static atomic_bool ggml_kimi_a2_request_phase_active = false;
static atomic_uint_fast64_t ggml_kimi_full_coverage_request_seq = 0;
static atomic_uint_fast64_t ggml_kimi_full_coverage_expert_calls = 0;
static atomic_uint_fast64_t ggml_kimi_full_coverage_fallback_tensors = 0;
static atomic_uint_fast64_t ggml_kimi_full_coverage_fallback_rows = 0;
static pthread_mutex_t ggml_kimi_full_coverage_trace_mu = PTHREAD_MUTEX_INITIALIZER;
static bool ggml_kimi_full_coverage_trace_header = false;

static bool ggml_kimi_full_coverage_prefill_enabled(void) {
    const char * env = getenv("GGML_MOE_PREFILL_FULL_COVERAGE");
    return env && env[0] && env[0] != '0';
}

static void ggml_kimi_full_coverage_counters_reset(void) {
    atomic_store_explicit(&ggml_kimi_full_coverage_expert_calls, 0, memory_order_relaxed);
    atomic_store_explicit(&ggml_kimi_full_coverage_fallback_tensors, 0, memory_order_relaxed);
    atomic_store_explicit(&ggml_kimi_full_coverage_fallback_rows, 0, memory_order_relaxed);
}

static void ggml_kimi_full_coverage_report(void) {
    if (!ggml_kimi_full_coverage_prefill_enabled()) {
        return;
    }
    fprintf(stderr,
            "[kimi_full_coverage_prefill] request=%" PRIuFAST64
            " expert_calls=%" PRIuFAST64 " cpu_fallback_expert_tensors=%" PRIuFAST64
            " cpu_fallback_expert_rows=%" PRIuFAST64 "\n",
            atomic_load_explicit(&ggml_kimi_full_coverage_request_seq, memory_order_relaxed),
            atomic_load_explicit(&ggml_kimi_full_coverage_expert_calls, memory_order_relaxed),
            atomic_load_explicit(&ggml_kimi_full_coverage_fallback_tensors, memory_order_relaxed),
            atomic_load_explicit(&ggml_kimi_full_coverage_fallback_rows, memory_order_relaxed));
}

void ggml_cpu_kimi_a2_request_phase(int active);

void ggml_cpu_kimi_a2_request_phase(int active) {
    const bool enabled = active != 0;
    const bool previous = atomic_exchange_explicit(
        &ggml_kimi_a2_request_phase_active, enabled, memory_order_acq_rel);
    if (ggml_kimi_full_coverage_prefill_enabled()) {
        if (enabled && !previous) {
            atomic_fetch_add_explicit(
                &ggml_kimi_full_coverage_request_seq, 1, memory_order_relaxed);
            ggml_kimi_full_coverage_counters_reset();
        } else if (!enabled && previous) {
            ggml_kimi_full_coverage_report();
        }
    }
}

static bool ggml_cpu_kimi_a2_request_phase_is_active(void) {
    return atomic_load_explicit(
        &ggml_kimi_a2_request_phase_active, memory_order_acquire);
}

static void ggml_kimi_full_coverage_cpu_fallback_record(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * ids,
        const int64_t * matrix_row_counts,
        int64_t n_as) {
    if (!ggml_kimi_full_coverage_prefill_enabled() ||
            !ggml_cpu_kimi_a2_request_phase_is_active() ||
            !src0 || !ids || !matrix_row_counts || ids->ne[1] <= 1 || n_as <= 0) {
        return;
    }

    int64_t fallback_rows = 0;
    for (int64_t expert = 0; expert < n_as; ++expert) {
        if (matrix_row_counts[expert] > 0) {
            fallback_rows += matrix_row_counts[expert];
        }
    }
    atomic_fetch_add_explicit(
        &ggml_kimi_full_coverage_expert_calls, 1, memory_order_relaxed);
    if (fallback_rows <= 0) {
        return;
    }

    atomic_fetch_add_explicit(
        &ggml_kimi_full_coverage_fallback_tensors, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(
        &ggml_kimi_full_coverage_fallback_rows, (uint64_t) fallback_rows, memory_order_relaxed);
    fprintf(stderr,
            "[kimi_full_coverage_fallback] tensor=%s tokens=%" PRId64
            " cpu_fallback_expert_rows=%" PRId64 "\n",
            src0->name, ids->ne[1], fallback_rows);

    const char * path = getenv("GGML_MOE_PREFILL_FULL_COVERAGE_FALLBACK_OUT");
    if (!path || !path[0]) {
        return;
    }
    pthread_mutex_lock(&ggml_kimi_full_coverage_trace_mu);
    FILE * file = fopen(path, "a");
    if (file) {
        if (!ggml_kimi_full_coverage_trace_header) {
            fprintf(file, "request,tensor,tokens,cpu_fallback_expert_rows\n");
            ggml_kimi_full_coverage_trace_header = true;
        }
        fprintf(file, "%" PRIuFAST64 ",%s,%" PRId64 ",%" PRId64 "\n",
                atomic_load_explicit(
                    &ggml_kimi_full_coverage_request_seq, memory_order_relaxed),
                src0->name, ids->ne[1], fallback_rows);
        fclose(file);
    }
    pthread_mutex_unlock(&ggml_kimi_full_coverage_trace_mu);
}

#include "common-adapter.inc"

static void ggml_compute_forward_mul_mat_id_generic(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids = dst->src[2];
    // if(strstr(src0->name,"up")){
    //     printf("%d ",src0->ne[1]*src0->nb[1]);
    // }
    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;
    const bool ds4_sparse_fused_mmvq_membership = ggml_ds4_sparse_fused_mmvq_membership_enabled();

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
#if defined(__APPLE__)
    const bool mac_pack_source = ggml_mac_expert_pack_enabled(params->ssd_mac_cache) &&
        (n_as == 384 || n_as == 256) && (ids->ne[1] == 1 || ggml_mac_expert_pack_prefill_enabled(params->ssd_mac_cache)) &&
        strstr(src0->name, ".ffn_") && strstr(src0->name, "_exps.weight");
#endif


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

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

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
    if (ith == 0) {
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
    }

    // reset current_chunk
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    const char * gpu_stream_batch_reason = "decode_batch_only_cpu_fallback";
    const bool use_gpu_stream_batch = false;
    const bool gpu_stream_batch_done = false;
    const int64_t gpu_stream_single_attempts = 0;

    if (ggml_moe_cpu_willneed_enabled()) {
        //都不会走
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
        //会走
        memset(fallback_pack_mmap_ptrs, 0, (size_t) n_as * sizeof(void *));
        memset(fallback_touch_us, 0, n_as*sizeof(uint64_t));
        if (ggml_moe_cpu_fallback_touch_profile_enabled()) {
            //不会走
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


    const bool moe_fallback_reason_profile = ggml_moe_fallback_reason_profile_enabled();
    // const uint64_t kimi_cpu_moe_fallback_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
    const uint64_t moe_fallback_reason_start = (moe_fallback_reason_profile && ith == 0) ? ggml_time_us() : 0;
    const uint64_t ds4_sparse_fused_mmvq_fallback_start =
        (ds4_sparse_fused_mmvq_membership && ith == 0) ? ggml_time_us() : 0;
#if defined(__APPLE__)
    void * mac_read_batch = NULL;
    if (mac_pack_source && ith == 0) {
        mac_read_batch = ggml_mac_expert_pack_begin_batch(params->ssd_mac_cache, src0->name, matrix_row_counts, n_as, (size_t) ne01 * nb01, ids->ne[1] == 1);
    }
    uint64_t mac_compute_start = 0, mac_compute_us = 0, mac_barrier_start = 0, mac_barrier_us = 0;
#endif
    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) {
            continue;
        }
#if defined(__APPLE__)
        if (mac_pack_source) {
            if (ith == 0) {
                fallback_pack_mmap_ptrs[cur_a] =
                    ggml_mac_expert_pack_acquire_batch(params->ssd_mac_cache, mac_read_batch, src0->name, cur_a, (size_t) ne01 * nb01);
                mac_barrier_start = ggml_time_us();
            }
            ggml_barrier(params->threadpool);
            if (ith == 0) {
                mac_barrier_us = ggml_time_us() - mac_barrier_start;
                mac_compute_start = ggml_time_us();
            }
        }
#endif
        // printf("k=");
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
               //不会执行
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

#if defined(__APPLE__)
        if (mac_pack_source) {
            if (ith == 0) {
                mac_barrier_start = ggml_time_us();
                mac_compute_us = mac_barrier_start - mac_compute_start;
            }
            ggml_barrier(params->threadpool);
            if (ith == 0) {
                mac_barrier_us += ggml_time_us() - mac_barrier_start;
                ggml_mac_expert_pack_release(params->ssd_mac_cache, fallback_pack_mmap_ptrs[cur_a]);
                fallback_pack_mmap_ptrs[cur_a] = NULL;
                mac_barrier_start = ggml_time_us();
            }
            ggml_barrier(params->threadpool);
            if (ith == 0) {
                mac_barrier_us += ggml_time_us() - mac_barrier_start;
                ggml_mac_expert_pack_record_compute(params->ssd_mac_cache, mac_compute_us, mac_barrier_us);
            }
        }
#endif
    }
#if defined(__APPLE__)
    if (mac_pack_source && ith == 0) { ggml_mac_expert_pack_end_batch(params->ssd_mac_cache, mac_read_batch); }
#endif

    if (moe_fallback_reason_profile && ith == 0) {
        const uint64_t moe_fallback_reason_us = ggml_time_us() - moe_fallback_reason_start;
        int64_t fallback_rows = 0;
        for (int cur_a = 0; cur_a < n_as; ++cur_a) {
            fallback_rows += matrix_row_counts[cur_a];
        }
        const char * batch_reason_name = gpu_stream_batch_reason;
        const char * single_reason_name = "retired_gpu_bridge";
        const char * final_reason_name = "cpu_fallback";
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
                        gpu_stream_batch_done,
                        gpu_stream_single_attempts > 0,
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

}

static void ggml_compute_forward_mul_mat_id(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {
    if (ggml_common_model_type_is_kimi()) {
        ggml_compute_forward_mul_mat_id_common_kimi(params, dst);
        return;
    }
    ggml_compute_forward_mul_mat_id_generic(params, dst);
}
