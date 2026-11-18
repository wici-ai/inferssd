// Synthetic Q4_K transport geometry; included after vram-tier.cpp.
// Explicit owner metadata keeps the old 92-layer quota shape without the legacy parser.
#pragma once
#include "declared-layout.h"
#include <numeric>
// 92 routed layers (1..92), Q4_K throughout: 256 experts of 256 x 256.
static inline const declared_layout & lifecycle_layout() {
    static const declared_layout layout = [] {
        std::vector<int32_t> ids(92);
        std::iota(ids.begin(), ids.end(), 1);
        return declared_layout::uniform(256, 256, 256, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, ids);
    }();
    return layout;
}
static inline size_t lifecycle_role_bytes() {
    return (ggml_row_size(GGML_TYPE_Q4_K, 256) * size_t(256) + 4095) & ~size_t(4095);
}
// The tier is declared the layout before its first layer, as the CPU owner declares it once it made the tier.
static inline void lifecycle_declare(tier & target) {
    if (target.formats.empty()) { target.declare(lifecycle_layout().c); }
}
static inline void lifecycle_prepare(tier & target, const ssd_vram_request * requests, int count, int * slots) {
    lifecycle_declare(target);
    target.prepare_format(requests, count, slots);
}
static inline void lifecycle_launch(tier & target, const std::vector<float> & input) {
    const ssd_vram_activation activation = {SSD_VRAM_ACTIVATION_SITU, 4.0f, 25.0f, 0.0f};
    target.launch_activation(input.data(), input.size() * sizeof(float), &activation);
}
