// SPDX-License-Identifier: MIT
// Retired process-global exports retained only for the old forwarding wrapper.
// Production copy/dense/seed operations use gpu::transport_v1 with a context owner.
// These stubs own no source index, CUDA state, cache, environment or fallback.
#include "moe-common-adapter-symbols.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
namespace ggml_moe_common_adapter {
struct ggml_moe_row_mapping { int32_t i1, i2; };
struct ggml_moe_iq2_replay_result { float direct_up, direct_gate, direct_fused, repack_up, repack_gate, repack_fused; };
extern "C" void ggml_cuda_moe_stream_batch_link_anchor() {}
extern "C" void ggml_cuda_moe_ttft_trace_mark(const char *label) { (void)label; }
extern "C" void ggml_cuda_moe_ttft_trace_event(
        const char *op,
        const char *tensor_name,
        int marker_value,
        size_t expert_bytes,
        double elapsed_ms) { (void)op;(void)tensor_name;(void)marker_value;(void)expert_bytes;(void)elapsed_ms; }
extern "C" void ggml_cuda_moe_stream_batch_report_counters(void) {  }
extern "C" void ggml_cuda_moe_stream_batch_phase_report(const char *label) { (void)label; }
extern "C" bool ggml_cuda_moe_expert_pack_v2_lookup_debug(
        const char *tensor_name,
        int expert_idx,
        int *packed_type,
        int64_t *ne00,
        int64_t *ne01,
        size_t *nb01,
        size_t *nbytes) { (void)tensor_name;(void)expert_idx;(void)packed_type;(void)ne00;(void)ne01;(void)nb01;(void)nbytes;return false; }
extern "C" bool ggml_cuda_moe_expert_pack_v2_read_debug(
        const char *tensor_name,
        int expert_idx,
        void *dst,
        size_t dst_capacity,
        size_t *nread) { (void)tensor_name;(void)expert_idx;(void)dst;(void)dst_capacity;(void)nread;return false; }
extern "C" const void * ggml_cuda_moe_expert_pack_mmap_ptr(const char *tensor_name, int expert_idx, size_t nbytes) { (void)tensor_name;(void)expert_idx;(void)nbytes;return nullptr; }
extern "C" const void * ggml_cuda_moe_expert_pack_mmap_ptr_debug(
        const char * tensor_name,
        int expert_idx,
        size_t nbytes,
        const char ** reason,
        size_t * entry_nbytes,
        uint64_t * entry_offset) { (void)tensor_name;(void)expert_idx;(void)nbytes;(void)reason;(void)entry_nbytes;(void)entry_offset;return nullptr; }
extern "C" const void * ggml_cuda_moe_expert_pack_mmap_ptr_source_debug(
        const char * tensor_name,
        int expert_idx,
        size_t nbytes,
        const char ** reason,
        size_t * entry_nbytes,
        uint64_t * entry_offset,
        int * source_idx,
        const char ** source_path,
        int * source_is_gguf) { (void)tensor_name;(void)expert_idx;(void)nbytes;(void)reason;(void)entry_nbytes;(void)entry_offset;(void)source_idx;(void)source_path;(void)source_is_gguf;return nullptr; }
extern "C" bool ggml_cuda_moe_expert_pack_resolve_direct(
        const char * tensor_name,
        int expert_idx,
        size_t nbytes,
        int * fd_direct,
        uint64_t * offset,
        size_t * direct_nbytes) { (void)tensor_name;(void)expert_idx;(void)nbytes;(void)fd_direct;(void)offset;(void)direct_nbytes;return false; }
extern "C" bool ggml_cuda_moe_common_expert_pack_resolve_raw_direct(
        const char * tensor_name, int expert_idx, size_t nbytes,
        int * fd_direct, uint64_t * offset, size_t * logical_nbytes) { (void)tensor_name;(void)expert_idx;(void)nbytes;(void)fd_direct;(void)offset;(void)logical_nbytes;return false; }
extern "C" bool ggml_cuda_moe_expert_pack_read_to_host(
        const char *tensor_name,
        int expert_idx,
        size_t nbytes,
        void *dst,
        size_t dst_capacity,
        size_t *nread,
        int *source_is_gguf) { (void)tensor_name;(void)expert_idx;(void)nbytes;(void)dst;(void)dst_capacity;(void)nread;(void)source_is_gguf;return false; }
extern "C" bool ggml_cuda_moe_expert_pack_copy_ranges_to_device(
        const char *tensor_name,
        const int32_t *range_first,
        const int32_t *range_last,
        size_t range_count,
        int64_t n_expert,
        size_t expert_bytes,
        size_t padding,
        bool cache_bypass,
        void *device_dst,
        size_t device_capacity,
        void *stream_ptr,
        uint64_t *experts_copied,
        uint64_t *iouring_reads,
        uint64_t *iouring_bytes,
        uint64_t *cache_lookups,
        uint64_t *cache_hits) { (void)tensor_name;(void)range_first;(void)range_last;(void)range_count;(void)n_expert;(void)expert_bytes;(void)padding;(void)cache_bypass;(void)device_dst;(void)device_capacity;(void)stream_ptr;(void)experts_copied;(void)iouring_reads;(void)iouring_bytes;(void)cache_lookups;(void)cache_hits;return false; }
extern "C" bool ggml_cuda_moe_dense_has_tensor(const char *tensor_name, size_t nbytes, size_t *block_bytes) { (void)tensor_name;(void)nbytes;(void)block_bytes;return false; }
extern "C" bool ggml_cuda_moe_dense_stage_prepare(size_t *pinned_bytes) { (void)pinned_bytes;return false; }
extern "C" bool ggml_cuda_moe_dense_read_ahead(const char *tensor_name, size_t nbytes) { (void)tensor_name;(void)nbytes;return false; }
extern "C" bool ggml_cuda_moe_dense_copy_to_device(
        const char *tensor_name,
        void *device_dst,
        size_t nbytes,
        void *stream_ptr,
        bool *read_ahead_hit) { (void)tensor_name;(void)device_dst;(void)nbytes;(void)stream_ptr;(void)read_ahead_hit;return false; }
extern "C" bool ggml_cuda_moe_stream_handoff_upload(
        const float *host_ptr,
        int64_t ne01,
        int64_t dst_cols) { (void)host_ptr;(void)ne01;(void)dst_cols;return false; }
extern "C" bool ggml_cuda_moe_iq2_q8k_r8_selftest(void) { return false; }
extern "C" bool ggml_cuda_moe_iq2_xxs_q8k_selftest(void) { return false; }
extern "C" bool ggml_cuda_moe_iq2_prompt_replay(
        int src0_type_int,
        const void *up_expert,
        const void *gate_expert,
        int64_t ne01,
        int64_t ne00,
        size_t nb01,
        const float *src1_row_f32,
        int64_t col,
        int unary_op,
        float limit,
        ggml_moe_iq2_replay_result *out) { (void)src0_type_int;(void)up_expert;(void)gate_expert;(void)ne01;(void)ne00;(void)nb01;(void)src1_row_f32;(void)col;(void)unary_op;(void)limit;(void)out;return false; }
extern "C" bool ggml_cuda_moe_stream_stage_active_experts(
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes,
    const int64_t *matrix_row_counts,
    void *dst_staged,
    size_t dst_staged_bytes,
    cudaStream_t stream) { (void)src0_name;(void)src0_data;(void)n_as;(void)nb02;(void)expert_bytes;(void)matrix_row_counts;(void)dst_staged;(void)dst_staged_bytes;(void)stream;return false; }
extern "C" bool ggml_cuda_moe_stream_preload_tensor(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes) { (void)src0_type_int;(void)src0_name;(void)src0_data;(void)n_as;(void)nb02;(void)expert_bytes;return false; }
extern "C" bool ggml_cuda_moe_stream_preload_tensor_prompt(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes) { (void)src0_type_int;(void)src0_name;(void)src0_data;(void)n_as;(void)nb02;(void)expert_bytes;return false; }
extern "C" bool ggml_cuda_moe_stream_preload_tensor_async(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes) { (void)src0_type_int;(void)src0_name;(void)src0_data;(void)n_as;(void)nb02;(void)expert_bytes;return false; }
extern "C" bool ggml_cuda_moe_stream_preload_expert_async(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes,
    int expert_idx) { (void)src0_type_int;(void)src0_name;(void)src0_data;(void)n_as;(void)nb02;(void)expert_bytes;(void)expert_idx;return false; }
extern "C" bool ggml_cuda_moe_stream_preload_expert_from_pack_async(
    int src0_type_int,
    const char *src0_name,
    int64_t n_as,
    size_t expert_bytes,
    int expert_idx) { (void)src0_type_int;(void)src0_name;(void)n_as;(void)expert_bytes;(void)expert_idx;return false; }
extern "C" int ggml_cuda_moe_stream_batch_preload_active_from_pack(
    int src0_type_int,
    const char *src0_name,
    int64_t n_as,
    size_t expert_bytes,
    const int64_t *matrix_row_counts) { (void)src0_type_int;(void)src0_name;(void)n_as;(void)expert_bytes;(void)matrix_row_counts;return 0; }
extern "C" int ggml_cuda_moe_stream_q4_colibri_pilot_prefetch_from_gate(
    int src0_type_int,
    const char *src0_gate_name,
    int64_t n_as,
    size_t expert_bytes,
    const int64_t *matrix_row_counts) { (void)src0_type_int;(void)src0_gate_name;(void)n_as;(void)expert_bytes;(void)matrix_row_counts;return 0; }
extern "C" bool ggml_cuda_moe_stream_preload_synchronize(void) { return false; }
extern "C" bool ggml_cuda_moe_stream_cache_contains(
    const char *src0_name,
    size_t expert_bytes,
    int expert_idx) { (void)src0_name;(void)expert_bytes;(void)expert_idx;return false; }
extern "C" const void * ggml_cuda_moe_stream_cache_dev_ptr(
    const char *src0_name,
    size_t expert_bytes,
    int expert_idx) { (void)src0_name;(void)expert_bytes;(void)expert_idx;return nullptr; }
extern "C" bool ggml_cuda_moe_stream_register_tensor(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes) { (void)src0_type_int;(void)src0_name;(void)src0_data;(void)n_as;(void)nb02;(void)expert_bytes;return false; }
extern "C" int ggml_cuda_moe_stream_route_group_preload_updown_from_gate(
        const char *src0_gate_name,
        int64_t n_as,
        const int64_t *matrix_row_counts) { (void)src0_gate_name;(void)n_as;(void)matrix_row_counts;return 0; }
extern "C" int ggml_cuda_moe_stream_batch_preload_gate_updown(const char *gate_name, int expert_idx, size_t expert_bytes) { (void)gate_name;(void)expert_idx;(void)expert_bytes;return 0; }
extern "C" int ggml_cuda_moe_stream_batch_flush_gate_updown_cosubmit(void) { return 0; }
extern "C" bool ggml_cuda_moe_stream_batch(
    int  src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    size_t nb02,
    const float *src1_f32,
    int64_t src1_ne1,
    size_t src1_nb1, size_t src1_nb2,
    const void *src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_q8_0_ne1,
    float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const int64_t *matrix_row_counts,
    const ggml_moe_row_mapping *matrix_rows,
    int64_t rows_stride) { (void)src0_type_int;(void)src0_name;(void)src0_data;(void)n_as;(void)ne01;(void)ne00;(void)nb01;(void)nb02;(void)src1_f32;(void)src1_ne1;(void)src1_nb1;(void)src1_nb2;(void)src1_q8_0;(void)src1_q8_0_row_size;(void)src1_q8_0_ne1;(void)dst;(void)dst_nb1;(void)dst_nb2;(void)matrix_row_counts;(void)matrix_rows;(void)rows_stride;return false; }
} // namespace ggml_moe_common_adapter
