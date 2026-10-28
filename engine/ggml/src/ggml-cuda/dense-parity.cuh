// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <climits>
#include <cstdlib>
#include <cstring>

// The staged dense path already selects the CPU-compatible kernels by adding
// this suffix.  A separate, opt-in layer list extends the same numerical
// dialect to weights that moved from CUDA_Host to resident CUDA storage.
static inline bool ggml_cuda_dense_parity_is_staged(const char * name) {
    return name != nullptr && std::strstr(name, "#dense-prefetch") != nullptr;
}

static inline bool ggml_cuda_dense_parity_parse_uint(const char *& cursor, int & value) {
    if (cursor == nullptr || *cursor < '0' || *cursor > '9') {
        return false;
    }

    int parsed = 0;
    do {
        const int digit = *cursor - '0';
        if (parsed > (INT_MAX - digit) / 10) {
            return false;
        }
        parsed = parsed * 10 + digit;
        ++cursor;
    } while (*cursor >= '0' && *cursor <= '9');

    value = parsed;
    return true;
}

static inline void ggml_cuda_dense_parity_skip_space(const char *& cursor) {
    while (*cursor == ' ' || *cursor == '\t') {
        ++cursor;
    }
}

static inline bool ggml_cuda_dense_parity_layer_spec_contains(
        const char * spec,
        int layer) {
    if (spec == nullptr || spec[0] == '\0' || layer < 0) {
        return false;
    }

    const char * cursor = spec;
    bool selected = false;
    bool saw_range = false;
    while (true) {
        ggml_cuda_dense_parity_skip_space(cursor);
        if (*cursor == '\0') {
            return saw_range && selected;
        }

        int first = -1;
        if (!ggml_cuda_dense_parity_parse_uint(cursor, first)) {
            return false;
        }
        ggml_cuda_dense_parity_skip_space(cursor);

        int last = first;
        if (*cursor == '-') {
            ++cursor;
            ggml_cuda_dense_parity_skip_space(cursor);
            if (!ggml_cuda_dense_parity_parse_uint(cursor, last) || last < first) {
                return false;
            }
            ggml_cuda_dense_parity_skip_space(cursor);
        }

        saw_range = true;
        selected = selected || (layer >= first && layer <= last);
        if (*cursor == '\0') {
            return selected;
        }
        if (*cursor != ',') {
            return false;
        }
        ++cursor;
        ggml_cuda_dense_parity_skip_space(cursor);
        if (*cursor == '\0') {
            return false;
        }
    }
}

static inline int ggml_cuda_dense_parity_tensor_layer(const char * name) {
    if (name == nullptr || std::strncmp(name, "blk.", 4) != 0) {
        return -1;
    }

    const char * cursor = name + 4;
    int layer = -1;
    if (!ggml_cuda_dense_parity_parse_uint(cursor, layer) || *cursor != '.') {
        return -1;
    }
    return layer;
}

static inline bool ggml_cuda_dense_parity_resident_selected_for_spec(
        const char * spec,
        const char * name) {
    if (name == nullptr || ggml_cuda_dense_parity_is_staged(name) ||
            std::strstr(name, "_exps.") != nullptr) {
        return false;
    }
    return ggml_cuda_dense_parity_layer_spec_contains(
        spec, ggml_cuda_dense_parity_tensor_layer(name));
}

static inline const char * ggml_cuda_dense_parity_resident_spec() {
    return nullptr; // Retired ambient resident-layer probe; explicit oracle specs remain.
}

static inline bool ggml_cuda_dense_parity_resident_selected(const char * name) {
    return ggml_cuda_dense_parity_resident_selected_for_spec(
        ggml_cuda_dense_parity_resident_spec(), name);
}

static inline bool ggml_cuda_dense_parity_q4k_selected_for_spec(
        const char * spec,
        const char * name) {
    return ggml_cuda_dense_parity_is_staged(name) ||
        ggml_cuda_dense_parity_resident_selected_for_spec(spec, name);
}

static inline bool ggml_cuda_dense_parity_f16_selected_for_spec(
        const char * spec,
        const char * name) {
    return name != nullptr &&
        (std::strstr(name, ".ssm_f_b.weight#dense-prefetch") != nullptr ||
         (std::strstr(name, ".ssm_f_b.weight") != nullptr &&
          ggml_cuda_dense_parity_resident_selected_for_spec(spec, name)));
}

static inline bool ggml_cuda_dense_parity_f32_selected_for_spec(
        const char * spec,
        const char * name) {
    return name != nullptr &&
        (std::strstr(name, ".ffn_gate_inp.weight#dense-prefetch") != nullptr ||
         (std::strstr(name, ".ffn_gate_inp.weight") != nullptr &&
          ggml_cuda_dense_parity_resident_selected_for_spec(spec, name)));
}

static inline bool ggml_cuda_dense_parity_q4k_selected(const char * name) {
    return ggml_cuda_dense_parity_q4k_selected_for_spec(
        ggml_cuda_dense_parity_resident_spec(), name);
}

static inline bool ggml_cuda_dense_parity_f16_selected(const char * name) {
    return ggml_cuda_dense_parity_f16_selected_for_spec(
        ggml_cuda_dense_parity_resident_spec(), name);
}

static inline bool ggml_cuda_dense_parity_f32_selected(const char * name) {
    return ggml_cuda_dense_parity_f32_selected_for_spec(
        ggml_cuda_dense_parity_resident_spec(), name);
}
