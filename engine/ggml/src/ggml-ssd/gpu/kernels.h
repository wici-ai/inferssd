// SPDX-License-Identifier: MIT
#pragma once
#include "ggml.h"
#include <cstddef>
#include <cstdint>

namespace ssd { namespace gpu {
using vec_dot_t = void (*)(int, float * GGML_RESTRICT, size_t,
    const void * GGML_RESTRICT, size_t, const void * GGML_RESTRICT, size_t, int);
// CPU-owned immutable callbacks, copied by each tier. No CPU backend dependency
// is permitted in a GPU executor. Invalid/unsupported types return GGML_TYPE_COUNT.
struct cpu_kernels_v1 {
    uint32_t version;
    size_t size;
    ggml_from_float_t quantize_q8_0;
    vec_dot_t reference_mxfp4;
    int (*vec_dot_type)(int);
};
inline bool valid(const cpu_kernels_v1 * value) noexcept {
    try { return value && value->version == 1 && value->size == sizeof(*value) &&
        value->quantize_q8_0 && value->reference_mxfp4 && value->vec_dot_type &&
        value->vec_dot_type(GGML_TYPE_MXFP4) == GGML_TYPE_Q8_0; } catch (...) { return false; }
}
using get_cpu_kernels_v1 = const cpu_kernels_v1 * (*)();
}} // namespace ssd::gpu
