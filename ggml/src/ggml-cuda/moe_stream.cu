#include "moe-adapter-dispatch.h"

#ifndef GGML_CUDA_MOE_STREAM
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
extern "C" {
typedef struct { int32_t i1; int32_t i2; } ggml_moe_row_mapping;
typedef struct {
    const void * role_ptr;
    uint64_t object_generation;
    uint64_t slot_generation;
    int32_t slot_id;
    int32_t role;
} ggml_moe_expert_descriptor;
void ggml_cuda_moe_stream_link_anchor(void) {}
bool ggml_cuda_moe_stream_available(void) { return false; }
int ggml_cuda_host_register(void *, size_t) { return 0; }
bool ggml_cuda_moe_stream_one(int, const char *, int64_t, const void *, int64_t, int64_t, size_t, const float *, size_t, size_t, int64_t, int64_t, const void *, size_t, float *, size_t, size_t, const ggml_moe_row_mapping *) { return false; }
void ggml_cuda_moe_stream_q80_probe(int, const char *, int64_t, const void *, int64_t, int64_t, size_t, const void *, size_t, int64_t, int64_t, const float *, size_t, size_t, const ggml_moe_row_mapping *) {}
void ggml_cuda_moe_stream_q80_write(int, const char *, int64_t, const void *, int64_t, int64_t, size_t, const void *, size_t, int64_t, int64_t, float *, size_t, size_t, const ggml_moe_row_mapping *) {}
bool ggml_cuda_moe_stream_q80_skip(int, const char *, int64_t, const void *, int64_t, int64_t, size_t, const void *, size_t, int64_t, int64_t, float *, size_t, size_t, const ggml_moe_row_mapping *) { return false; }
void ggml_cuda_moe_stream_q80_hot_batch_probe(int, const char *, int64_t, int64_t, int64_t, size_t, const void *, size_t, int64_t, const float *, size_t, size_t, const int64_t *, const ggml_moe_row_mapping *, int64_t, const float *, size_t, size_t) {}
bool ggml_cuda_moe_stream_one_cache_contains(const char *, const void *, size_t, int64_t) { return false; }
bool ggml_cuda_moe_stream_one_cache_ready(void) { return false; }
const void * ggml_cuda_moe_stream_one_cache_dev_ptr_or_insert(
    const char *, const void *, size_t, size_t, int64_t, cudaStream_t, int *) { return nullptr; }
bool ggml_cuda_moe_stream_one_cache_dev_ptrs_or_insert(
    const char *, const void *, size_t, size_t, const int *, const int32_t *, int, cudaStream_t,
    const void **, int *, ggml_moe_expert_descriptor *) { return false; }
void ggml_cuda_moe_stream_one_gate_up_prefetch(
    const char *, size_t, const int *, const int32_t *, int) {}
void ggml_cuda_moe_stream_one_cache_release_descriptors(
    const ggml_moe_expert_descriptor *, int, cudaStream_t) {}
void ggml_cuda_moe_stream_one_cache_release_descriptors_synchronized(
    const ggml_moe_expert_descriptor *, int) {}
const void * ggml_cuda_moe_stream_cache_dev_ptr(const char *, size_t, int) { return nullptr; }
bool ggml_cuda_moe_stream_mmvq_dev(int, const void *, int64_t, int64_t, size_t, const float *, void *, float *, cudaStream_t) { return false; }
bool ggml_cuda_moe_stream_mmvq_rows_dev(int, const void *, int64_t, int64_t, size_t, const float *, void *, const int32_t *, int64_t, float *, cudaStream_t) { return false; }
bool ggml_cuda_moe_stream_mmvq_batch_dev(int, const void *, int64_t, int64_t, const float *, void *, float *, const int32_t *, int64_t, int64_t, cudaStream_t) { return false; }
void ggml_cuda_moe_stream_sync(void) {}
}
#else
// Public compatibility entries for the common streaming implementation.
// Shape observation remains shared; the common transport is the sole adapter.
// Original streaming implementation author: Zili Meng <zilim@ieee.org>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

extern "C" {

typedef struct {
    int32_t i1;
    int32_t i2;
} ggml_moe_row_mapping;

typedef struct {
    const void * role_ptr;
    uint64_t object_generation;
    uint64_t slot_generation;
    int32_t slot_id;
    int32_t role;
} ggml_moe_expert_descriptor;

void ggml_cuda_moe_stream_link_anchor(void) {}

// Restored main/common adapter.  Its implementation is link-isolated in
// moe_stream_common.cu and has the same C ABI as the local path.
bool ggml_cuda_moe_common_stream_available(void);
const void * ggml_cuda_moe_common_stream_ds4_triplet_host_role_ptr(
    const char *, int64_t, size_t);
bool ggml_cuda_moe_common_stream_one(
    int, const char *, int64_t, const void *, int64_t, int64_t, size_t,
    const float *, size_t, size_t, int64_t, int64_t, const void *, size_t,
    float *, size_t, size_t, const ggml_moe_row_mapping *);
void ggml_cuda_moe_common_stream_q80_probe(
    int, const char *, int64_t, const void *, int64_t, int64_t, size_t,
    const void *, size_t, int64_t, int64_t, const float *, size_t, size_t,
    const ggml_moe_row_mapping *);
void ggml_cuda_moe_common_stream_q80_write(
    int, const char *, int64_t, const void *, int64_t, int64_t, size_t,
    const void *, size_t, int64_t, int64_t, float *, size_t, size_t,
    const ggml_moe_row_mapping *);
bool ggml_cuda_moe_common_stream_q80_skip(
    int, const char *, int64_t, const void *, int64_t, int64_t, size_t,
    const void *, size_t, int64_t, int64_t, float *, size_t, size_t,
    const ggml_moe_row_mapping *);
void ggml_cuda_moe_common_stream_q80_hot_batch_probe(
    int, const char *, int64_t, int64_t, int64_t, size_t, const void *,
    size_t, int64_t, const float *, size_t, size_t, const int64_t *,
    const ggml_moe_row_mapping *, int64_t, const float *, size_t, size_t);
bool ggml_cuda_moe_common_stream_one_cache_contains(
    const char *, const void *, size_t, int64_t);
bool ggml_cuda_moe_common_stream_one_cache_ready(void);
const void * ggml_cuda_moe_common_stream_one_cache_dev_ptr_or_insert(
    const char *, const void *, size_t, size_t, int64_t, cudaStream_t, int *);
bool ggml_cuda_moe_common_stream_one_cache_dev_ptrs_or_insert(
    const char *, const void *, size_t, size_t, const int *, const int32_t *,
    int, cudaStream_t, const void **, int *, ggml_moe_expert_descriptor *);
void ggml_cuda_moe_common_stream_one_gate_up_prefetch(
    const char *, size_t, const int *, const int32_t *, int);
void ggml_cuda_moe_common_stream_one_cache_release_descriptors(
    const ggml_moe_expert_descriptor *, int, cudaStream_t);
void ggml_cuda_moe_common_stream_one_cache_release_descriptors_synchronized(
    const ggml_moe_expert_descriptor *, int);
void ggml_cuda_moe_common_stream_sync(void);

} // extern "C"

extern "C" const void * ggml_cuda_moe_stream_ds4_triplet_host_role_ptr(
        const char * src0_name,
        int64_t expert_index,
        size_t role_nbytes) {

    return ggml_cuda_moe_common_stream_ds4_triplet_host_role_ptr(
        src0_name, expert_index, role_nbytes);
}

extern "C" void ggml_cuda_moe_stream_one_gate_up_prefetch(
        const char * src0_name,
        size_t src0_bytes,
        const int * expert_indices,
        const int32_t * route_ranks,
        int count) {

    ggml_cuda_moe_common_stream_one_gate_up_prefetch(
        src0_name, src0_bytes, expert_indices, route_ranks, count);
}

extern "C" bool ggml_cuda_moe_stream_one_cache_dev_ptrs_or_insert(
        const char * src0_name,
        const void * src0_base_data,
        size_t nb02,
        size_t src0_bytes,
        const int * expert_indices,
        const int32_t * route_ranks,
        int count,
        cudaStream_t stream,
        const void ** output_ptrs,
        int * cache_states,
        ggml_moe_expert_descriptor * descriptors) {

    return ggml_cuda_moe_common_stream_one_cache_dev_ptrs_or_insert(
        src0_name, src0_base_data, nb02, src0_bytes, expert_indices,
        route_ranks, count, stream, output_ptrs, cache_states, descriptors);
}

extern "C" void ggml_cuda_moe_stream_one_cache_release_descriptors(
        const ggml_moe_expert_descriptor * descriptors,
        int count,
        cudaStream_t stream) {

    ggml_cuda_moe_common_stream_one_cache_release_descriptors(
        descriptors, count, stream);
}

extern "C" void ggml_cuda_moe_stream_one_cache_release_descriptors_synchronized(
        const ggml_moe_expert_descriptor * descriptors,
        int count) {

    ggml_cuda_moe_common_stream_one_cache_release_descriptors_synchronized(
        descriptors, count);
}

extern "C" bool ggml_cuda_moe_stream_one_cache_contains(
    const char *src0_name,
    const void *src0_data,
    size_t src0_bytes,
    int64_t expert_index)
{

    return ggml_cuda_moe_common_stream_one_cache_contains(
        src0_name, src0_data, src0_bytes, expert_index);
}

extern "C" const void * ggml_cuda_moe_stream_one_cache_dev_ptr_or_insert(
    const char *src0_name,
    const void *src0_base_data,
    size_t nb02,
    size_t src0_bytes,
    int64_t expert_index,
    cudaStream_t st,
    int *cache_state)
{

    return ggml_cuda_moe_common_stream_one_cache_dev_ptr_or_insert(
        src0_name, src0_base_data, nb02, src0_bytes, expert_index, st,
        cache_state);
}

extern "C" bool ggml_cuda_moe_stream_one_cache_ready(void) {

    return ggml_cuda_moe_common_stream_one_cache_ready();
}

extern "C" void ggml_cuda_moe_stream_q80_probe(
    int src0_type_int,
    const char *src0_name,
    int64_t expert_index,
    const void *src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const void *src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_ne1,
    int64_t cne1,
    const float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const ggml_moe_row_mapping *rows) {

    ggml_cuda_moe_common_stream_q80_probe(
        src0_type_int, src0_name, expert_index, src0_data, ne01, ne00,
        nb01, src1_q8_0, src1_q8_0_row_size, src1_ne1, cne1, dst,
        dst_nb1, dst_nb2, rows);
}

extern "C" void ggml_cuda_moe_stream_q80_write(
    int src0_type_int,
    const char *src0_name,
    int64_t expert_index,
    const void *src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const void *src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_ne1,
    int64_t cne1,
    float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const ggml_moe_row_mapping *rows) {

    ggml_cuda_moe_common_stream_q80_write(
        src0_type_int, src0_name, expert_index, src0_data, ne01, ne00,
        nb01, src1_q8_0, src1_q8_0_row_size, src1_ne1, cne1, dst,
        dst_nb1, dst_nb2, rows);
}

extern "C" void ggml_cuda_moe_stream_q80_hot_batch_probe(
    int src0_type_int,
    const char *src0_name,
    int64_t n_as,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const void *src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_ne1,
    const float *src1_f32,
    size_t src1_nb1,
    size_t src1_nb2,
    const int64_t *matrix_row_counts,
    const ggml_moe_row_mapping *matrix_rows,
    int64_t rows_per_expert,
    const float *dst,
    size_t dst_nb1,
    size_t dst_nb2) {
    ggml_moe::adapter_select_model_shape(n_as, nb01, "q80-hot-batch");

    ggml_cuda_moe_common_stream_q80_hot_batch_probe(
        src0_type_int, src0_name, n_as, ne01, ne00, nb01, src1_q8_0,
        src1_q8_0_row_size, src1_ne1, src1_f32, src1_nb1, src1_nb2,
        matrix_row_counts, matrix_rows, rows_per_expert, dst, dst_nb1,
        dst_nb2);
}

extern "C" bool ggml_cuda_moe_stream_q80_skip(
    int src0_type_int,
    const char *src0_name,
    int64_t expert_index,
    const void *src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const void *src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_ne1,
    int64_t cne1,
    float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const ggml_moe_row_mapping *rows) {

    return ggml_cuda_moe_common_stream_q80_skip(
        src0_type_int, src0_name, expert_index, src0_data, ne01, ne00,
        nb01, src1_q8_0, src1_q8_0_row_size, src1_ne1, cne1, dst,
        dst_nb1, dst_nb2, rows);
}

extern "C" bool ggml_cuda_moe_stream_available(void) {
    // Availability is queried before the first routed tensor exposes the
    // model's expert shape.  Report the compiled path without initializing
    // either process-global adapter; the first shaped call selects one.
    if (!ggml_moe::adapter_is_selected()) {
        return true;
    }

    return ggml_cuda_moe_common_stream_available();
}

extern "C" int ggml_cuda_host_register(void *p, size_t n) {
    if (!p || n == 0) return 0;
    cudaError_t err = cudaHostRegister(p, n, cudaHostRegisterDefault);
    if (err == cudaSuccess) {
        std::fprintf(stderr, "[moe_stream] cudaHostRegister %.1f GiB OK\n",
                     n / (1024.0*1024.0*1024.0));
        return 1;
    }
    std::fprintf(stderr, "[moe_stream] cudaHostRegister %.1f GiB FAILED: %s\n",
                 n / (1024.0*1024.0*1024.0), cudaGetErrorString(err));
    return 0;
}

extern "C" bool ggml_cuda_moe_stream_one(
    int  src0_type_int,
    const char *src0_name,
    int64_t expert_index,
    const void *src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const float *src1_f32,
    size_t src1_nb1, size_t src1_nb2,
    int64_t src1_ne1,
    int64_t cne1,
    const void *src1_q8_1,
    size_t src1_padded_num_cols,
    float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const ggml_moe_row_mapping *rows)
{

    return ggml_cuda_moe_common_stream_one(
        src0_type_int, src0_name, expert_index, src0_data, ne01, ne00,
        nb01, src1_f32, src1_nb1, src1_nb2, src1_ne1, cne1,
        src1_q8_1, src1_padded_num_cols, dst, dst_nb1, dst_nb2, rows);
}

extern "C" void ggml_cuda_moe_stream_sync(void) {

    ggml_cuda_moe_common_stream_sync();
}

#endif // GGML_CUDA_MOE_STREAM
