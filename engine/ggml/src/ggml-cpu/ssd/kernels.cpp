// SPDX-License-Identifier: Apache-2.0
#include "runtime.h"
#include "ggml-ssd/gpu/kernels.h"
#include "ggml-cpu.h"

namespace ssd { namespace cpu {
namespace {
int input_type(int type) {
    if (type < 0 || type >= GGML_TYPE_COUNT) { return GGML_TYPE_COUNT; }
    const auto * traits = ggml_get_type_traits_cpu(static_cast<ggml_type>(type));
    return traits && traits->vec_dot ? int(traits->vec_dot_type) : GGML_TYPE_COUNT;
}
}
const gpu::cpu_kernels_v1 * kernels() noexcept {
    static const gpu::cpu_kernels_v1 value = { 1, sizeof(gpu::cpu_kernels_v1),
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float,
        ggml_get_type_traits_cpu(GGML_TYPE_MXFP4)->vec_dot, input_type };
    return &value;
}
}} // namespace ssd::cpu
