#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

// Clear any cached draft KV/recurrent state (e.g. when the target slot is cleared or replaced).
void common_speculative_clear(common_speculative * spec);

// Save/restore target-conditioned drafter state (e.g. DSpark KV cache) to a sidecar file
// alongside a disk prompt cache entry. Returns true on success, false if unsupported or failed.
bool common_speculative_save_state_file(
        common_speculative * spec,
        const char         * path,
        int64_t              n_tokens);

bool common_speculative_load_state_file(
        common_speculative * spec,
        const char         * path,
        int64_t              n_tokens_keep);

// Feed a successfully decoded target batch into target-conditioned drafters.
bool common_speculative_process(
        common_speculative * spec,
        const llama_batch & batch,
        llama_seq_id target_seq_id);

// Feed only the selected winning rows (indices into `batch`) into target-conditioned drafters,
// assigning contiguous positions starting at `pos0`.
bool common_speculative_process_rows(
        common_speculative * spec,
        const llama_batch  & batch,
        const int32_t      * keep_idxs,
        int32_t              n_keep,
        llama_pos            pos0,
        llama_seq_id         target_seq_id);

// Candidate speculation tree rooted at the target's anchor token (batch node 0).
// Draft tokens occupy batch nodes 1..K in non-decreasing depth order; parents[i] in [0..i].
struct common_speculative_tree {
    llama_tokens        tokens;
    std::vector<int8_t> parents;
    std::vector<float>  probs;

    bool   empty() const { return tokens.empty(); }
    size_t size()  const { return tokens.size(); }
    void   clear()       { tokens.clear(); parents.clear(); probs.clear(); }

    bool is_linear() const {
        for (size_t i = 0; i < parents.size(); ++i) {
            if (parents[i] != (int8_t) i) {
                return false;
            }
        }
        return true;
    }
};

// Whether this speculative drafter supports multi-branch candidate trees.
bool common_speculative_supports_tree(const common_speculative * spec);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last);

// Draft a throughput-optimal candidate tree rooted at `id_last`.
common_speculative_tree common_speculative_draft_tree(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

int32_t common_speculative_n_max(const common_speculative * spec, const common_params_speculative & params);
int32_t common_speculative_n_min(const common_speculative * spec, const common_params_speculative & params);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;
