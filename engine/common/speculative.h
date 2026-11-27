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

// Feed a successfully decoded target batch into target-conditioned drafters.
bool common_speculative_process(
        common_speculative * spec,
        const llama_batch & batch,
        llama_seq_id target_seq_id);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
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

// A step of the model's MTP draft head (draft-mtp) in the target's context: TOKEN at position POS of SEQ_ID, read with H,
// the hidden state of the position before it. On success H holds the hidden state the step left, which the next step
// reads, and BEST and P are the head's most likely next token and its probability; false when the step failed.
bool common_speculative_mtp_step(llama_context * ctx, llama_batch & batch, std::vector<float> & h, llama_token token,
        llama_pos pos, llama_seq_id seq_id, llama_token & best, double & p);

// The draft length that pays on this machine, for a draft head whose steps cost time (MTP). Each length 0 .. n_max
// keeps the tokens a step at that length gave and the time it took, draft steps included, as measured, and the
// length that gives the most tokens a second runs. A length runs in steps of its own: every length first for a
// burst, then the best one, and every probe_every steps the next other length for a burst again, so each stays
// measured and the choice follows the machine and the text. The step after a change of length is not measured:
// what a change costs once (another graph, other caches) is no measure of a length that keeps running. A burst
// measures its length anew, its steps replacing what the length had: a length probed again is compared by its latest
// burst with the recent steps of the best one, not by a mean that reaches back to its first burst on a cold cache.
struct common_speculative_adapt {
    static constexpr int    burst       = 4;
    static constexpr int    probe_every = 32;
    static constexpr double rate        = 0.1; // a new measurement's weight once a length has 1 / rate of them

    std::vector<double> t_len; // ms a step at length n took, index n; 0 while unmeasured
    std::vector<double> n_len; // tokens it gave
    std::vector<int>    c_len; // its measurements since its last burst began
    int    held    = -1;       // the length running, and its steps left before the next choice
    int    hold    = 0;
    int    n_warm  = 0;        // lengths run so far for their first burst
    int    n_probe = 0;
    int    n_best  = 0;        // steps at the best length since the last probe
    int    last    = -1;       // the length of the step before: a step at another length is not measured
    int    pending = -1;       // a length whose head proposed nothing: the step decodes without drafts after its draft
    double pending_ms = 0.0;

    // the draft length for a step that may draft up to n_max tokens
    int choose(int n_max);

    // a step decoded without drafts in ms: length 0's, or the length whose head proposed nothing (on_empty)
    void on_plain(double ms);
    // the head drafted at length n_len and proposed nothing, its steps taking draft_ms
    void on_empty(int n_len, double draft_ms);
    // a step at length n_len verified drafts: n_accepted of them accepted, ms with the draft steps
    void on_block(int n_len, int n_accepted, double ms);
    // forget the step before (a prompt came between): the next step is not measured
    void interrupt() { last = -1; pending = -1; }

    // the tokens a second length n gave, 0 while unmeasured
    double rate_of(int n) const;
    int    best(int n_max) const;
};

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;
