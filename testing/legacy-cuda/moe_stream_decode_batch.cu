#include "moe-adapter-dispatch.h"

#ifndef GGML_CUDA_MOE_STREAM

#include <cstddef>
#include <cstdint>

extern "C" {
typedef struct { int32_t i1; int32_t i2; } ggml_moe_decode_row_mapping;

bool ggml_cuda_moe_stream_decode_batch(
    int, const char *, const void *, int64_t, int64_t, int64_t, size_t, size_t,
    const float *, int64_t, size_t, size_t, float *, size_t, size_t,
    int64_t *, const ggml_moe_decode_row_mapping *, int64_t) {
    return false;
}
}

#else

#include <cstddef>
#include <cstdint>

extern "C" {

typedef struct {
    int32_t i1;
    int32_t i2;
} ggml_moe_decode_row_mapping;

bool ggml_cuda_moe_common_stream_decode_batch(
    int, const char *, const void *, int64_t, int64_t, int64_t, size_t,
    size_t, const float *, int64_t, size_t, size_t, float *, size_t, size_t,
    int64_t *, const ggml_moe_decode_row_mapping *, int64_t);

}

extern "C" bool ggml_cuda_moe_stream_decode_batch(
    int src0_type_int,
    const char * src0_name,
    const void * src0_data,
    int64_t n_as,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    size_t nb02,
    const float * src1_f32,
    int64_t src1_ne1,
    size_t src1_nb1,
    size_t src1_nb2,
    const void * src1_q8_0,
    size_t src1_q8_0_row_size,
    float * dst,
    size_t dst_nb1,
    size_t dst_nb2,
    int64_t * matrix_row_counts,
    const ggml_moe_decode_row_mapping * rows,
    int64_t rows_stride) {
    const size_t expert_bytes = (size_t) ne01 * nb01;
    ggml_moe::adapter_select_model_shape(n_as, expert_bytes, "decode-batch");

    return ggml_cuda_moe_common_stream_decode_batch(
        src0_type_int, src0_name, src0_data, n_as, ne01, ne00, nb01,
        nb02, src1_f32, src1_ne1, src1_nb1, src1_nb2, dst, dst_nb1,
        dst_nb2, matrix_row_counts, rows, rows_stride);
}

#endif // GGML_CUDA_MOE_STREAM
