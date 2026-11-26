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

// The draft length that pays on this machine, for a draft head whose steps cost time (MTP). Each length 0 .. n_max
// keeps the tokens a step at that length gave and the time it took, draft steps included, as measured; the
// length that gave the most tokens a second is picked, 0 running no draft step. The first picks go round 0 ..
// n_max, and every probe_every-th pick takes the next length round again, so every length stays measured and
// the choice follows the machine and the text.
struct common_speculative_adapt {
    static constexpr int    warmup      = 2;    // rounds of 0 .. n_max before the measurements are used
    static constexpr int    probe_every = 16;
    static constexpr double rate        = 0.25; // the weight of a new measurement

    std::vector<double> t_len; // ms a step at length n took, index n; 0 while unmeasured
    std::vector<double> n_len; // tokens it gave
    int    n_picks    = 0;
    int    n_probe    = 0;
    int    pending    = -1;  // a length whose head proposed nothing: the step decodes without drafts after its draft
    double pending_ms = 0.0;

    // the draft length for a step that may draft up to n_max tokens
    int choose(int n_max);

    // a step decoded without drafts in ms: length 0's, or the length whose head proposed nothing (on_empty)
    void on_plain(double ms);
    // the head drafted at length n_len and proposed nothing, its steps taking draft_ms
    void on_empty(int n_len, double draft_ms);
    // a step at length n_len verified drafts: n_accepted of them accepted, ms with the draft steps
    void on_block(int n_len, int n_accepted, double ms);

    // the tokens a second length n gave, 0 while unmeasured
    double rate_of(int n) const;
};

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;
