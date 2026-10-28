// SPDX-License-Identifier: MIT
#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

enum common_reasoning_budget_state {
    REASONING_BUDGET_IDLE,         // waiting for start sequence
    REASONING_BUDGET_COUNTING,     // counting down tokens
    REASONING_BUDGET_FORCING,      // forcing budget message + end sequence
    REASONING_BUDGET_WAITING_UTF8, // budget exhausted, waiting for UTF-8 completion
    REASONING_BUDGET_DONE,         // passthrough, waiting for another start sequence
};

// Creates a reasoning budget sampler that limits token generation inside a
// reasoning block (e.g. between <think> and </think>).
//
// State machine: IDLE -> COUNTING -> WAITING_UTF8 -> FORCING -> DONE
// Each new start sequence in DONE re-arms COUNTING with the full budget.
//   IDLE:         passthrough, watching for start_tokens sequence
//   COUNTING:     counting down remaining tokens, watching for natural end_tokens
//   WAITING_UTF8: budget exhausted, allowing tokens to complete a UTF-8 sequence
//   FORCING:      forces forced_tokens token-by-token (all other logits -> -inf)
//   DONE:         passthrough, watching for the next start_tokens sequence
//
// Parameters:
//   vocab          - vocabulary (used for UTF-8 boundary detection; can be nullptr)
//   start_tokens   - token sequence that activates counting
//   end_tokens     - token sequence for natural deactivation
//   forced_tokens  - token sequence forced when budget expires
//   budget         - max tokens allowed in the reasoning block
//   prefill_tokens - tokens already present in the prompt (generation prompt);
//                    used to determine the initial state: COUNTING if they begin
//                    with start_tokens (but don't also end with end_tokens),
//                    IDLE otherwise. COUNTING with budget <= 0 is promoted to FORCING.
//
struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        const std::vector<llama_token> & prefill_tokens = {});

// Variant that takes an explicit initial state (used by tests and clone).
// COUNTING with budget <= 0 is promoted to FORCING.
struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        common_reasoning_budget_state    initial_state);

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl);

// Optional protocol evidence for the next accepted token's natural end span.
// Used for multi-token end tags, including alternative tokenizations. Ignored
// while FORCING: the existing forced sequence must finish unchanged.
void common_reasoning_budget_set_protocol_end(struct llama_sampler * smpl, bool in_progress, bool complete);

// Log attribution only; does not change the budget or sampler state.
void common_reasoning_budget_set_log_context(struct llama_sampler * smpl, int32_t slot, int64_t task);
