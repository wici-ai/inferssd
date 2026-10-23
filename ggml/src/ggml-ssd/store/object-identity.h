#pragma once
#include <cstddef>
#include <cstdint>

namespace ggml_moe_common {

enum class role : uint8_t {
    gate = 0,
    up = 1,
    down = 2,
    count = 3,
};

constexpr uint8_t role_bit(role value) {
    return uint8_t{1} << static_cast<uint8_t>(value);
}

enum class role_slot_state : uint8_t {
    free,
    reserved,
    reading,
    h2d_ready,
    failed,
};

struct role_slot_marker {
    uintptr_t key = 0;
    uint64_t generation = 0;
    size_t bytes = 0;
};

struct object_key {
    int32_t source_idx = -1;
    int32_t layer_idx = -1;
    int32_t expert_idx = -1;
    uint64_t manifest_generation = 0;
    uint64_t offset = 0;
    size_t bytes = 0;
    // Zero means the whole pack uses manifest_generation (the Kimi layout).
    // Packs such as DeepSeek may additionally version each object independently.
    uint64_t object_generation = 0;
};

inline bool object_key_matches(const object_key &current, const object_key &expected) {
    return current.source_idx == expected.source_idx &&
        current.layer_idx == expected.layer_idx &&
        current.expert_idx == expected.expert_idx &&
        current.manifest_generation == expected.manifest_generation &&
        current.offset == expected.offset &&
        current.bytes == expected.bytes &&
        current.object_generation == expected.object_generation;
}

inline uint64_t object_key_effective_generation(const object_key &key) {
    return key.object_generation != 0 ? key.object_generation : key.manifest_generation;
}

inline bool role_slot_marker_matches(
        const role_slot_marker &current,
        const role_slot_marker &expected) {
    return current.key == expected.key &&
        current.generation == expected.generation &&
        current.bytes == expected.bytes;
}

inline bool role_slot_transition_allowed(role_slot_state from, role_slot_state to) {
    switch (from) {
        case role_slot_state::free:
            return to == role_slot_state::reserved;
        case role_slot_state::reserved:
            return to == role_slot_state::reading ||
                to == role_slot_state::failed ||
                to == role_slot_state::free;
        case role_slot_state::reading:
            return to == role_slot_state::h2d_ready ||
                to == role_slot_state::failed;
        case role_slot_state::h2d_ready:
        case role_slot_state::failed:
            return to == role_slot_state::free;
    }
    return false;
}

} // namespace ggml_moe_common
