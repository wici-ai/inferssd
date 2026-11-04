// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ggml.h"

// Initial warmup, unstable shapes and topology changes keep the direct path.
inline bool ggml_cuda_can_recapture(bool enabled, bool all_vmm, bool instance,
                                    bool replayed, bool nodes_changed) {
    return enabled && all_vmm && instance && replayed && !nodes_changed;
}

inline bool ggml_cuda_recapture_skips_kernel(ggml_op op, bool empty) {
    return empty || op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_TRANSPOSE ||
        op == GGML_OP_VIEW || op == GGML_OP_PERMUTE;
}

inline bool ggml_cuda_recapture_node_changed(ggml_op old_op, bool old_empty,
                                           ggml_op new_op, bool new_empty) {
    return old_op != new_op || ggml_cuda_recapture_skips_kernel(old_op, old_empty) !=
        ggml_cuda_recapture_skips_kernel(new_op, new_empty);
}
