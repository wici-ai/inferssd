#include "llama-kv-cache-iswa.h"

#include <limits>
#include <stdexcept>

// An earlier state of a sequence is the base half up to it and the sliding window before it, both still in the cache
// while the batch that went past it is the last one: the SWA half holds n_swa + n_ubatch cells and gives a cell to a
// new position only when the position after its last one no longer attends to it.  A restored sequence continues
// from its last position (the server evaluates that one again), so the window is the one that position attends to.
// K and V of a position must depend only on the tokens up to it; a window that reaches forward (symmetric,
// non-causal) does not give that, so such a cache has no earlier states.
uint32_t llama_kv_cache_iswa::state_write_at_align() const {
    return hparams.swa_type == LLAMA_SWA_TYPE_STANDARD || hparams.swa_type == LLAMA_SWA_TYPE_CHUNKED ? 1 : 0;
}

llama_pos llama_kv_cache_iswa::window_start(llama_pos n_tokens) const {
    llama_pos p0 = n_tokens - 1;
    while (p0 > 0 && !llama_hparams::is_masked_swa(hparams.n_swa, hparams.swa_type, p0 - 1, n_tokens - 1)) {
        --p0;
    }
    return p0;
}

bool llama_kv_cache_iswa::state_can_write_at(llama_seq_id seq_id, llama_pos n_tokens) const {
    if (seq_id < 0 || seq_id >= LLAMA_MAX_SEQ) {
        return false;
    }

    const llama_pos pos_max = seq_pos_max(seq_id);
    if (n_tokens == pos_max + 1) {
        return true;
    }
    if (state_write_at_align() == 0 || n_tokens <= 0 || n_tokens > pos_max) {
        return false;
    }

    // the window's cells must not have been given to later positions yet
    return kv_swa->seq_has_range(seq_id, window_start(n_tokens), n_tokens - 1);
}

void llama_kv_cache_iswa::state_write_at(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags, llama_pos n_tokens) const {
    if (!state_can_write_at(seq_id, n_tokens)) {
        throw std::runtime_error("the iSWA cache cannot save this earlier state of the sequence");
    }
    if (n_tokens == seq_pos_max(seq_id) + 1) {
        state_write(io, seq_id, flags);
        return;
    }

    // the layout of state_write(), which state_read() reads back
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        kv_base->state_write_range(io, seq_id, flags, std::numeric_limits<llama_pos>::min(), n_tokens - 1);
    }
    kv_swa->state_write_range(io, seq_id, flags, window_start(n_tokens), n_tokens - 1);
}

// the earlier state's window; the current state holds what the cache holds
llama_pos llama_kv_cache_iswa::state_pos_min_at(llama_seq_id seq_id, llama_pos n_tokens) const {
    return n_tokens == seq_pos_max(seq_id) + 1 ? seq_pos_min(seq_id) : window_start(n_tokens);
}

