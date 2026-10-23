#include "moe-common-adapter-symbols.h"
#include "moe-expert-descriptor.h"

#ifndef GGML_CUDA_MOE_STREAM
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
extern "C" {
typedef struct { int32_t i1; int32_t i2; } ggml_moe_row_mapping;
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
// Shared streaming entry points. Experimental compute and object owners were
// retired; ordinary host registration and the batch synchronization hook stay.
// Original streaming implementation author: Zili Meng <zilim@ieee.org>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace ggml_moe_common_adapter {
extern "C" {
typedef struct { int32_t i1; int32_t i2; } ggml_moe_row_mapping;
void ggml_cuda_moe_stream_link_anchor(void) {}
int ggml_cuda_moe_stream_batch_flush_gate_updown_cosubmit(void);
}

static bool gate_updown_cosubmit_enabled() {
    const char *env = std::getenv("GGML_MOE_GATE_UPDOWN_COSUBMIT");
    return env && env[0] && env[0] != '0';
}

extern "C" const void * ggml_cuda_moe_stream_ds4_triplet_host_role_ptr(
        const char * src0_name,
        int64_t expert_index,
        size_t role_nbytes) {
    // The experimental object/cache or probe path is retired.
    return nullptr;
}

extern "C" void ggml_cuda_moe_stream_one_gate_up_prefetch(
        const char * src0_name,
        size_t src0_bytes,
        const int * expert_indices,
        const int32_t * route_ranks,
        int count) {
    // The experimental object/cache or probe path is retired.
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
    // The experimental object/cache or probe path is retired.
    return false;
}

extern "C" void ggml_cuda_moe_stream_one_cache_release_descriptors(
        const ggml_moe_expert_descriptor * descriptors,
        int count,
        cudaStream_t stream) {
    // The experimental object/cache or probe path is retired.
}

extern "C" void ggml_cuda_moe_stream_one_cache_release_descriptors_synchronized(
        const ggml_moe_expert_descriptor * descriptors,
        int count) {
    // The experimental object/cache or probe path is retired.
}

extern "C" bool ggml_cuda_moe_stream_one_cache_contains(
    const char *src0_name,
    const void *src0_data,
    size_t src0_bytes,
    int64_t expert_index)
{
    // The experimental object/cache or probe path is retired.
    return false;
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
    if (cache_state) {
        *cache_state = 0;
    }
    // The experimental object/cache or probe path is retired.
    return nullptr;
}

extern "C" bool ggml_cuda_moe_stream_one_cache_ready(void) {
    // The experimental object/cache or probe path is retired.
    return false;
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
    // The experimental object/cache or probe path is retired.
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
    // The experimental object/cache or probe path is retired.
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
    // The experimental object/cache or probe path is retired.
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
    // The experimental object/cache or probe path is retired.
    return false;
}

extern "C" bool ggml_cuda_moe_stream_available(void) {
    // The opt-in runtime streaming compute path is retired.
    return false;
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
    // The opt-in runtime streaming compute path is retired.
    return false;
}

extern "C" void ggml_cuda_moe_stream_sync(void) {
    if (gate_updown_cosubmit_enabled()) {
        ggml_cuda_moe_stream_batch_flush_gate_updown_cosubmit();
    }
}

} // namespace ggml_moe_common_adapter
#endif // GGML_CUDA_MOE_STREAM
