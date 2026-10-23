#include "moe-common-adapter-symbols.h"
#include "moe-expert-descriptor.h"
#include "moe-common-device-descriptor.h"

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
typedef struct { int32_t i1; int32_t i2; } ggml_moe_decode_row_mapping;

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
    // The opt-in runtime streaming compute path is retired.
    return false;
}
#endif // GGML_CUDA_MOE_STREAM
