// SPDX-License-Identifier: MIT
#pragma once

#include "prompt-cache.h"
#include "ggml.h"

static server_prompt_checkpoint server_get_checkpoint(llama_context * ctx, int id, int64_t n_tokens, llama_pos pos_min = -1, llama_pos pos_max = -1) {
    if (pos_min == -1) {
        pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), id);
    }
    if (pos_max == -1) {
        pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx), id);
    }

    const size_t checkpoint_size = llama_state_seq_get_size_ext(ctx, id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

    auto cur = server_prompt_checkpoint {
        /*.pos_min  = */ pos_min,
        /*.pos_max  = */ pos_max,
        /*.n_tokens = */ n_tokens,
        /*.data     = */ std::vector<uint8_t>(checkpoint_size),
    };

    const size_t n = llama_state_seq_get_data_ext(ctx, cur.data.data(), checkpoint_size, id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (n != checkpoint_size) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", checkpoint_size, n);
    }

    return cur;
}

// The partial state of the sequence after its first n_tokens tokens, taken after the batch that went past them by a
// memory that can save an earlier state (llama_state_seq_at_align). It holds the positions from pos_min to the one
// before n_tokens: a memory that cannot be rolled back resumes exactly after that position (pos_min = pos_max), a
// sliding-window cache holds the window that position attends to and is truncated after loading like an ordinary
// checkpoint.
static server_prompt_checkpoint server_get_checkpoint_at(llama_context * ctx, int id, int64_t n_tokens, llama_pos pos_min) {
    const size_t checkpoint_size = llama_state_seq_get_size_at(ctx, id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, n_tokens);

    auto cur = server_prompt_checkpoint {
        /*.pos_min  = */ pos_min,
        /*.pos_max  = */ (llama_pos) n_tokens - 1,
        /*.n_tokens = */ n_tokens,
        /*.data     = */ std::vector<uint8_t>(checkpoint_size),
    };

    const size_t n = llama_state_seq_get_data_at(ctx, cur.data.data(), checkpoint_size, id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, n_tokens);
    if (checkpoint_size == 0 || n != checkpoint_size) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", checkpoint_size, n);
    }

    return cur;
}

