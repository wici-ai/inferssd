#pragma once

#include "moe-object-engine.h"

using q4_colibri_slot_state = ggml_moe::role_slot_state;
using q4_colibri_slot_marker = ggml_moe::role_slot_marker;
using q4_colibri_object_marker = ggml_moe::object_key;

static inline bool q4_colibri_object_marker_matches(
        const q4_colibri_object_marker &current,
        const q4_colibri_object_marker &expected) {
    return ggml_moe::object_key_matches(current, expected);
}

static inline bool q4_colibri_slot_marker_matches(
        const q4_colibri_slot_marker &current,
        const q4_colibri_slot_marker &expected) {
    return ggml_moe::role_slot_marker_matches(current, expected);
}

static inline bool q4_colibri_slot_transition_allowed(
        q4_colibri_slot_state from,
        q4_colibri_slot_state to) {
    return ggml_moe::role_slot_transition_allowed(from, to);
}
