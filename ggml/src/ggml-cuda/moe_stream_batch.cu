
#include "moe-adapter-dispatch.h"

#ifndef GGML_CUDA_MOE_STREAM_BATCH
#include <cstdint>
#include <cstddef>
extern "C" {
typedef struct { int32_t i1; int32_t i2; } ggml_moe_row_mapping;
typedef struct { float direct_up; float direct_gate; float direct_fused; float repack_up; float repack_gate; float repack_fused; } ggml_moe_iq2_replay_result;
void ggml_cuda_moe_stream_batch_link_anchor(void) {}
void ggml_cuda_moe_ttft_trace_mark(const char *) {}
bool ggml_cuda_moe_iq2_prompt_replay(int, const void *, const void *, int64_t, int64_t, size_t, const float *, int64_t, int, float, ggml_moe_iq2_replay_result *) { return false; }
bool ggml_cuda_moe_stream_batch(int, const char *, const void *, int64_t, int64_t, int64_t, size_t, size_t, const float *, int64_t, size_t, size_t, const void *, size_t, int64_t, float *, size_t, size_t, const int64_t *, const ggml_moe_row_mapping *, int64_t) { return false; }
bool ggml_cuda_moe_stream_handoff_upload(const float *, int64_t, int64_t) { return false; }
int ggml_cuda_moe_stream_batch_preload_gate_updown(const char *, int, size_t) { return 0; }
int ggml_cuda_moe_stream_batch_flush_gate_updown_cosubmit(void) { return 0; }
int ggml_cuda_moe_stream_batch_preload_active_from_pack(int, const char *, int64_t, size_t, const int64_t *) { return 0; }
int ggml_cuda_moe_stream_q4_colibri_pilot_prefetch_from_gate(int, const char *, int64_t, size_t, const int64_t *) { return 0; }
int ggml_cuda_moe_stream_route_group_preload_updown_from_gate(const char *, int64_t, const int64_t *) { return 0; }
bool ggml_cuda_moe_stream_preload_tensor(int, const char *, const void *, int64_t, size_t, size_t) { return false; }
bool ggml_cuda_moe_stream_preload_tensor_prompt(int, const char *, const void *, int64_t, size_t, size_t) { return false; }
bool ggml_cuda_moe_stream_register_tensor(int, const char *, const void *, int64_t, size_t, size_t) { return false; }
bool ggml_cuda_moe_stream_cache_contains(const char *, size_t, int) { return false; }
const void * ggml_cuda_moe_stream_cache_dev_ptr(const char *, size_t, int) { return nullptr; }
void ggml_cuda_moe_stream_batch_phase_report(const char *) {}
bool ggml_cuda_moe_iq2_xxs_q8k_selftest(void) { return false; }
const void * ggml_cuda_moe_expert_pack_mmap_ptr(const char *, int, size_t) { return nullptr; }
const void * ggml_cuda_moe_expert_pack_mmap_ptr_debug(const char *, int, size_t, const char **, size_t *, uint64_t *) { return nullptr; }
const void * ggml_cuda_moe_expert_pack_mmap_ptr_source_debug(const char *, int, size_t, const char **, size_t *, uint64_t *, int *, const char **, int *) { return nullptr; }
bool ggml_cuda_moe_expert_pack_v2_lookup_debug(const char *, int, int *, int64_t *, int64_t *, size_t *, size_t *) { return false; }
bool ggml_cuda_moe_expert_pack_v2_read_debug(const char *, int, void *, size_t, size_t *) { return false; }
bool ggml_cuda_moe_expert_pack_read_to_host(const char *, int, size_t, void *, size_t, size_t *, int *) { return false; }
bool ggml_cuda_moe_expert_pack_copy_ranges_to_device(const char *, const int32_t *, const int32_t *, size_t, int64_t, size_t, size_t, bool, void *, size_t, void *, uint64_t *, uint64_t *, uint64_t *, uint64_t *, uint64_t *) { return false; }
bool ggml_cuda_moe_expert_pack_resolve_raw_direct_shape(const char *, int, size_t, int64_t, int *, uint64_t *, size_t *) { return false; }
bool ggml_cuda_moe_expert_pack_resolve_direct_shape(const char *, int, size_t, int64_t, int *, uint64_t *, size_t *) { return false; }
}
#else
// Supported streaming entries forward to the shared transport implementation.
// The retired K3 adapter owns no workers, caches or CUDA resources here.

#include <cstddef>
#include <cstdint>

extern "C" {
typedef struct {
    int32_t i1;
    int32_t i2;
} ggml_moe_row_mapping;

typedef struct {
    float direct_up;
    float direct_gate;
    float direct_fused;
    float repack_up;
    float repack_gate;
    float repack_fused;
} ggml_moe_iq2_replay_result;

// Link-isolated b654ca689 common adapter entry points.
void ggml_cuda_moe_common_ttft_trace_mark(const char *);
void ggml_cuda_moe_common_ttft_trace_event(
    const char *, const char *, int, size_t, double);
void ggml_cuda_moe_common_stream_batch_report_counters(void);
void ggml_cuda_moe_common_stream_batch_phase_report(const char *);
bool ggml_cuda_moe_common_expert_pack_v2_lookup_debug(
    const char *, int, int *, int64_t *, int64_t *, size_t *, size_t *);
bool ggml_cuda_moe_common_expert_pack_v2_read_debug(
    const char *, int, void *, size_t, size_t *);
const void * ggml_cuda_moe_common_expert_pack_mmap_ptr(
    const char *, int, size_t);
const void * ggml_cuda_moe_common_expert_pack_mmap_ptr_debug(
    const char *, int, size_t, const char **, size_t *, uint64_t *);
const void * ggml_cuda_moe_common_expert_pack_mmap_ptr_source_debug(
    const char *, int, size_t, const char **, size_t *, uint64_t *, int *,
    const char **, int *);
bool ggml_cuda_moe_common_expert_pack_read_to_host(
    const char *, int, size_t, void *, size_t, size_t *, int *);
bool ggml_cuda_moe_common_expert_pack_resolve_raw_direct(
    const char *, int, size_t, int *, uint64_t *, size_t *);
bool ggml_cuda_moe_common_expert_pack_resolve_direct(
    const char *, int, size_t, int *, uint64_t *, size_t *);
bool ggml_cuda_moe_common_expert_pack_copy_ranges_to_device(
    const char *, const int32_t *, const int32_t *, size_t, int64_t, size_t,
    size_t, bool, void *, size_t, void *, uint64_t *, uint64_t *, uint64_t *,
    uint64_t *, uint64_t *);
bool ggml_cuda_moe_common_stream_handoff_upload(
    const float *, int64_t, int64_t);
bool ggml_cuda_moe_common_iq2_q8k_r8_selftest(void);
bool ggml_cuda_moe_common_iq2_xxs_q8k_selftest(void);
bool ggml_cuda_moe_common_iq2_prompt_replay(
    int, const void *, const void *, int64_t, int64_t, size_t, const float *,
    int64_t, int, float, ggml_moe_iq2_replay_result *);
bool ggml_cuda_moe_common_stream_preload_tensor(
    int, const char *, const void *, int64_t, size_t, size_t);
bool ggml_cuda_moe_common_stream_preload_tensor_prompt(
    int, const char *, const void *, int64_t, size_t, size_t);
bool ggml_cuda_moe_common_stream_preload_tensor_async(
    int, const char *, const void *, int64_t, size_t, size_t);
bool ggml_cuda_moe_common_stream_preload_expert_async(
    int, const char *, const void *, int64_t, size_t, size_t, int);
bool ggml_cuda_moe_common_stream_preload_expert_from_pack_async(
    int, const char *, int64_t, size_t, int);
int ggml_cuda_moe_common_stream_batch_preload_active_from_pack(
    int, const char *, int64_t, size_t, const int64_t *);
int ggml_cuda_moe_common_stream_q4_colibri_pilot_prefetch_from_gate(
    int, const char *, int64_t, size_t, const int64_t *);
bool ggml_cuda_moe_common_stream_preload_synchronize(void);
bool ggml_cuda_moe_common_stream_cache_contains(const char *, size_t, int);
const void * ggml_cuda_moe_common_stream_cache_dev_ptr(
    const char *, size_t, int);
bool ggml_cuda_moe_common_stream_register_tensor(
    int, const char *, const void *, int64_t, size_t, size_t);
int ggml_cuda_moe_common_stream_route_group_preload_updown_from_gate(
    const char *, int64_t, const int64_t *);
int ggml_cuda_moe_common_stream_batch_preload_gate_updown(
    const char *, int, size_t);
int ggml_cuda_moe_common_stream_batch_flush_gate_updown_cosubmit(void);
bool ggml_cuda_moe_common_stream_batch(
    int, const char *, const void *, int64_t, int64_t, int64_t, size_t,
    size_t, const float *, int64_t, size_t, size_t, const void *, size_t,
    int64_t, float *, size_t, size_t, const int64_t *,
    const ggml_moe_row_mapping *, int64_t);

void ggml_cuda_moe_stream_batch_link_anchor(void) {}
}

extern "C" void ggml_cuda_moe_ttft_trace_mark(const char *label) {

    ggml_cuda_moe_common_ttft_trace_mark(label);
}

extern "C" void ggml_cuda_moe_ttft_trace_event(
        const char *op,
        const char *tensor_name,
        int marker_value,
        size_t expert_bytes,
        double elapsed_ms) {

    ggml_cuda_moe_common_ttft_trace_event(
        op, tensor_name, marker_value, expert_bytes, elapsed_ms);
}

extern "C" void ggml_cuda_moe_stream_batch_report_counters(void) {

    ggml_cuda_moe_common_stream_batch_report_counters();
}

extern "C" void ggml_cuda_moe_stream_batch_phase_report(const char *label) {

    ggml_cuda_moe_common_stream_batch_phase_report(label);
}

extern "C" bool ggml_cuda_moe_expert_pack_v2_lookup_debug(
        const char *tensor_name,
        int expert_idx,
        int *packed_type,
        int64_t *ne00,
        int64_t *ne01,
        size_t *nb01,
        size_t *nbytes) {

    return ggml_cuda_moe_common_expert_pack_v2_lookup_debug(
        tensor_name, expert_idx, packed_type, ne00, ne01, nb01, nbytes);
}

extern "C" bool ggml_cuda_moe_expert_pack_v2_read_debug(
        const char *tensor_name,
        int expert_idx,
        void *dst,
        size_t dst_capacity,
        size_t *nread) {

    return ggml_cuda_moe_common_expert_pack_v2_read_debug(
        tensor_name, expert_idx, dst, dst_capacity, nread);
}

extern "C" const void * ggml_cuda_moe_expert_pack_mmap_ptr(const char *tensor_name, int expert_idx, size_t nbytes) {

    return ggml_cuda_moe_common_expert_pack_mmap_ptr(
        tensor_name, expert_idx, nbytes);
}

extern "C" const void * ggml_cuda_moe_expert_pack_mmap_ptr_debug(
        const char * tensor_name,
        int expert_idx,
        size_t nbytes,
        const char ** reason,
        size_t * entry_nbytes,
        uint64_t * entry_offset) {

    return ggml_cuda_moe_common_expert_pack_mmap_ptr_debug(
        tensor_name, expert_idx, nbytes, reason, entry_nbytes,
        entry_offset);
}

extern "C" const void * ggml_cuda_moe_expert_pack_mmap_ptr_source_debug(
        const char * tensor_name,
        int expert_idx,
        size_t nbytes,
        const char ** reason,
        size_t * entry_nbytes,
        uint64_t * entry_offset,
        int * source_idx,
        const char ** source_path,
        int * source_is_gguf) {

    return ggml_cuda_moe_common_expert_pack_mmap_ptr_source_debug(
        tensor_name, expert_idx, nbytes, reason, entry_nbytes,
        entry_offset, source_idx, source_path, source_is_gguf);
}

extern "C" bool ggml_cuda_moe_expert_pack_resolve_raw_direct_shape(
        const char * tensor_name, int expert_idx, size_t nbytes, int64_t n_experts,
        int * fd_direct, uint64_t * offset, size_t * logical_nbytes) {
    if (!tensor_name || expert_idx < 0 || n_experts <= 0 || expert_idx >= n_experts || !nbytes) { return false; }
    ggml_moe::adapter_select_model_shape(n_experts, nbytes, "cpu-sink-raw-range");
    return ggml_moe::adapter_uses_common() && ggml_cuda_moe_common_expert_pack_resolve_raw_direct(
        tensor_name, expert_idx, nbytes, fd_direct, offset, logical_nbytes);
}

extern "C" bool ggml_cuda_moe_expert_pack_resolve_direct_shape(
        const char * tensor_name,
        int expert_idx,
        size_t nbytes,
        int64_t n_experts,
        int * fd_direct,
        uint64_t * offset,
        size_t * direct_nbytes) {
    if (fd_direct) *fd_direct = -1;
    if (offset) *offset = 0;
    if (direct_nbytes) *direct_nbytes = 0;
    if (!tensor_name || expert_idx < 0 || n_experts <= 0 || expert_idx >= n_experts ||
            nbytes == 0 || !direct_nbytes) {
        return false;
    }
    ggml_moe::adapter_select_model_shape(n_experts, nbytes, "cpu-sink");
    return ggml_moe::adapter_uses_common() &&
        ggml_cuda_moe_common_expert_pack_resolve_direct(
            tensor_name, expert_idx, nbytes, fd_direct, offset, direct_nbytes);
}

extern "C" bool ggml_cuda_moe_expert_pack_read_to_host(
        const char *tensor_name,
        int expert_idx,
        size_t nbytes,
        void *dst,
        size_t dst_capacity,
        size_t *nread,
        int *source_is_gguf) {

    return ggml_cuda_moe_common_expert_pack_read_to_host(
        tensor_name, expert_idx, nbytes, dst, dst_capacity, nread,
        source_is_gguf);
}

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
        uint64_t *cache_hits) {

    return ggml_cuda_moe_common_expert_pack_copy_ranges_to_device(
        tensor_name, range_first, range_last, range_count, n_expert,
        expert_bytes, padding, cache_bypass, device_dst, device_capacity,
        stream_ptr, experts_copied, iouring_reads, iouring_bytes,
        cache_lookups, cache_hits);
}

extern "C" bool ggml_cuda_moe_stream_handoff_upload(
        const float *host_ptr,
        int64_t ne01,
        int64_t dst_cols) {

    return ggml_cuda_moe_common_stream_handoff_upload(
        host_ptr, ne01, dst_cols);
}

extern "C" bool ggml_cuda_moe_iq2_q8k_r8_selftest(void) {

    return ggml_cuda_moe_common_iq2_q8k_r8_selftest();
}

extern "C" bool ggml_cuda_moe_iq2_xxs_q8k_selftest(void) {

    return ggml_cuda_moe_common_iq2_xxs_q8k_selftest();
}

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
        ggml_moe_iq2_replay_result *out) {

    return ggml_cuda_moe_common_iq2_prompt_replay(
        src0_type_int, up_expert, gate_expert, ne01, ne00, nb01,
        src1_row_f32, col, unary_op, limit, out);
}

extern "C" bool ggml_cuda_moe_stream_preload_tensor(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes) {
    ggml_moe::adapter_select_model_shape(n_as, expert_bytes, "preload-tensor");

    return ggml_cuda_moe_common_stream_preload_tensor(
        src0_type_int, src0_name, src0_data, n_as, nb02, expert_bytes);
}

extern "C" bool ggml_cuda_moe_stream_preload_tensor_prompt(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes) {
    ggml_moe::adapter_select_model_shape(n_as, expert_bytes, "preload-prompt");

    return ggml_cuda_moe_common_stream_preload_tensor_prompt(
        src0_type_int, src0_name, src0_data, n_as, nb02, expert_bytes);
}

extern "C" bool ggml_cuda_moe_stream_preload_tensor_async(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes) {
    ggml_moe::adapter_select_model_shape(n_as, expert_bytes, "preload-async");

    return ggml_cuda_moe_common_stream_preload_tensor_async(
        src0_type_int, src0_name, src0_data, n_as, nb02, expert_bytes);
}

extern "C" bool ggml_cuda_moe_stream_preload_expert_async(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes,
    int expert_idx) {
    ggml_moe::adapter_select_model_shape(n_as, expert_bytes, "preload-expert");

    return ggml_cuda_moe_common_stream_preload_expert_async(
        src0_type_int, src0_name, src0_data, n_as, nb02, expert_bytes,
        expert_idx);
}

extern "C" bool ggml_cuda_moe_stream_preload_expert_from_pack_async(
    int src0_type_int,
    const char *src0_name,
    int64_t n_as,
    size_t expert_bytes,
    int expert_idx) {
    ggml_moe::adapter_select_model_shape(n_as, expert_bytes, "preload-pack-expert");

    return ggml_cuda_moe_common_stream_preload_expert_from_pack_async(
        src0_type_int, src0_name, n_as, expert_bytes, expert_idx);
}

extern "C" int ggml_cuda_moe_stream_batch_preload_active_from_pack(
    int src0_type_int,
    const char *src0_name,
    int64_t n_as,
    size_t expert_bytes,
    const int64_t *matrix_row_counts) {
    ggml_moe::adapter_select_model_shape(n_as, expert_bytes, "preload-active");

    return ggml_cuda_moe_common_stream_batch_preload_active_from_pack(
        src0_type_int, src0_name, n_as, expert_bytes, matrix_row_counts);
}

extern "C" int ggml_cuda_moe_stream_q4_colibri_pilot_prefetch_from_gate(
    int src0_type_int,
    const char *src0_gate_name,
    int64_t n_as,
    size_t expert_bytes,
    const int64_t *matrix_row_counts) {
    ggml_moe::adapter_select_model_shape(n_as, expert_bytes, "pilot-prefetch");

    return ggml_cuda_moe_common_stream_q4_colibri_pilot_prefetch_from_gate(
        src0_type_int, src0_gate_name, n_as, expert_bytes,
        matrix_row_counts);
}

extern "C" bool ggml_cuda_moe_stream_preload_synchronize(void) {

    return ggml_cuda_moe_common_stream_preload_synchronize();
}

extern "C" bool ggml_cuda_moe_stream_cache_contains(
        const char *src0_name,
        size_t expert_bytes,
        int expert_idx) {

    return ggml_cuda_moe_common_stream_cache_contains(
        src0_name, expert_bytes, expert_idx);
}

extern "C" const void * ggml_cuda_moe_stream_cache_dev_ptr(
        const char *src0_name,
        size_t expert_bytes,
        int expert_idx) {

    return ggml_cuda_moe_common_stream_cache_dev_ptr(
        src0_name, expert_bytes, expert_idx);
}

extern "C" bool ggml_cuda_moe_stream_register_tensor(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes) {
    ggml_moe::adapter_select_model_shape(n_as, expert_bytes, "register-tensor");

    return ggml_cuda_moe_common_stream_register_tensor(
        src0_type_int, src0_name, src0_data, n_as, nb02, expert_bytes);
}

extern "C" int ggml_cuda_moe_stream_route_group_preload_updown_from_gate(
        const char *src0_gate_name,
        int64_t n_as,
        const int64_t *matrix_row_counts) {
    ggml_moe::adapter_select_model_shape(n_as, 0, "route-group-preload");

    return ggml_cuda_moe_common_stream_route_group_preload_updown_from_gate(
        src0_gate_name, n_as, matrix_row_counts);
}

extern "C" int ggml_cuda_moe_stream_batch_preload_gate_updown(const char *gate_name, int expert_idx, size_t expert_bytes) {

    return ggml_cuda_moe_common_stream_batch_preload_gate_updown(
        gate_name, expert_idx, expert_bytes);
}

extern "C" int ggml_cuda_moe_stream_batch_flush_gate_updown_cosubmit(void) {

    return ggml_cuda_moe_common_stream_batch_flush_gate_updown_cosubmit();
}

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
    int64_t rows_stride) {
    const size_t expert_bytes = (size_t)ne01 * nb01;
    ggml_moe::adapter_select_model_shape(n_as, expert_bytes, "stream-batch");

    return ggml_cuda_moe_common_stream_batch(
        src0_type_int, src0_name, src0_data, n_as, ne01, ne00, nb01,
        nb02, src1_f32, src1_ne1, src1_nb1, src1_nb2, src1_q8_0,
        src1_q8_0_row_size, src1_q8_0_ne1, dst, dst_nb1, dst_nb2,
        matrix_row_counts, matrix_rows, rows_stride);
}

#endif // GGML_CUDA_MOE_STREAM_BATCH
