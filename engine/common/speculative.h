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

// The draft length that pays on this machine, for a draft head whose steps cost time (MTP). It keeps the measured
// time of a decode of r rows (a token and the r - 1 drafts it verifies), of one draft step, and how often the
// target accepts the head's k-th draft token, and picks the length 0 .. n_max that decodes the most tokens a
// second; 0 runs no draft step. The first lengths it picks go round 0 .. n_max, and every probe_every-th pick
// takes the next one round again, so every estimate is measured and follows the machine and the text.
struct common_speculative_adapt {
    static constexpr int    warmup      = 2;     // rounds of 0 .. n_max before the estimates are used
    static constexpr int    probe_every = 16;
    static constexpr double rate        = 0.25;  // the weight of a new time measurement
    static constexpr double decay       = 0.98;  // of the acceptance counts, a block

    std::vector<double> t_rows;  // ms of a decode of r rows, index r; 0 while unmeasured
    double              t_step = 0.0;
    std::vector<double> acc;     // decayed counts: the k-th draft token (index k - 1) accepted
    std::vector<double> seen;    //                 and its block having drafted k - 1 tokens before it
    int n_picks = 0;
    int n_probe = 0;

    // the draft length for a step that may draft up to n_max tokens
    int choose(int n_max);

    // a step without drafts took ms; a step of n_proposed drafts took verify_ms past its draft_ms of n_steps steps
    // (n_proposed 0: the head ran n_steps and proposed nothing, then the step decoded without drafts)
    void on_plain(double ms);
    void on_block(int n_proposed, int n_accepted, int n_steps, double verify_ms, double draft_ms);

    // the tokens a second a draft length gives on the current estimates, 0 when its decode is unmeasured
    double rate_of(int n) const;
};

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;
