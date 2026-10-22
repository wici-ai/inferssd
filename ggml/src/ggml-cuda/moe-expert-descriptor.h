#pragma once

#include <cstddef>
#include <cstdint>

typedef struct {
    const void *role_ptr;
    uint64_t object_generation;
    uint64_t slot_generation;
    int32_t slot_id;
    int32_t role;
} ggml_moe_expert_descriptor;

constexpr int32_t GGML_MOE_EXPERT_ROLE_MASK = 0xff;
constexpr int32_t GGML_MOE_EXPERT_LAYOUT_PLANAR_MXFP4 = 1 << 8;
constexpr int32_t GGML_MOE_EXPERT_LAYOUT_SCALE3_MXFP4 = 1 << 9;
constexpr int32_t GGML_MOE_EXPERT_LAYOUT_PLANAR_SCALE3_MXFP4 = 1 << 10;

static_assert(sizeof(ggml_moe_expert_descriptor) == 32,
        "MoE expert descriptor ABI must remain stable");
static_assert(offsetof(ggml_moe_expert_descriptor, role_ptr) == 0);
static_assert(offsetof(ggml_moe_expert_descriptor, object_generation) == 8);
static_assert(offsetof(ggml_moe_expert_descriptor, slot_generation) == 16);
static_assert(offsetof(ggml_moe_expert_descriptor, slot_id) == 24);
static_assert(offsetof(ggml_moe_expert_descriptor, role) == 28);
