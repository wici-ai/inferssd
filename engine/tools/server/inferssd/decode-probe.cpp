// SPDX-License-Identifier: Apache-2.0
// What a decode step costs on this machine apart from its routed experts (--ssd-decode-probe), for the framework to
// decide before a launch whether drafting can pay. A draft, even one accepted whole, saves only a step's shared work:
// each token it gives still runs its own routed experts, which the framework counts itself. So the probe loads the
// model as the server does to serve it, every routed expert in the CPU sink (no GPU expert tier, no read-ahead), and
// takes the sink's time out of the wall time of target decodes of 1 .. n_max + 1 rows, as a verification decodes
// them, and of the MTP head's draft steps. The context is the probe's alone: its positions only go forward.
#include "decode-probe.h"
#include "runtime.h"

#include "../server-context.h"

#include "common.h"
#include "inferssd/options.h"
#include "llama-ssd.h"
#include "llama.h"
#include "speculative.h"
#include "../src/llama-ext.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace {
using json = nlohmann::ordered_json;

// English prose, for routes and caches as a request's text takes them
const char * const TEXT =
    "The river had been rising for three days before anyone in the village thought to move the boats. By the time the "
    "old ferryman walked down to the landing, the water had covered the lowest steps and was pulling at the ropes that "
    "held the small fleet in place. He stood there for a while, watching the current carry branches and leaves past the "
    "stone wall, and then he went from house to house, knocking on doors and telling people that the time had come to "
    "carry what they could to the higher fields. Some of them laughed and said the river would fall again by morning, "
    "as it always had, but most of them listened, because he had lived beside the water longer than any of them and "
    "had never once been wrong about it.";

constexpr int N_PROMPT   = 32; // tokens of the prompt, decoded first as one batch
constexpr int N_WARM_UP  = 2;  // steps of each kind before the measured ones
constexpr int N_MEASURED = 8;

// a step's wall time and the CPU expert sink's part of it, in ms
struct step_time {
    double ms;
    double sink_ms;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

// to the microsecond, the clock's resolution
double rounded(double ms) { return std::round(ms * 1e3) / 1e3; }

// the medians over STEPS of the wall time, the sink's time and the time of the rest of the step (dense_ms)
void add_medians(json & out, const std::vector<step_time> & steps) {
    std::vector<double> ms, sink_ms, dense_ms;
    for (const step_time & step : steps) {
        ms.push_back(step.ms);
        sink_ms.push_back(step.sink_ms);
        dense_ms.push_back(step.ms - step.sink_ms);
    }
    out["ms"]       = rounded(median(ms));
    out["sink_ms"]  = rounded(median(sink_ms));
    out["dense_ms"] = rounded(median(dense_ms));
}

// N_WARM_UP + N_MEASURED runs of STEP, the context synchronized after each; the measured runs' times in STEPS
template <typename F>
bool time_steps(llama_context * ctx, F && step, std::vector<step_time> & steps) {
    steps.clear();
    for (int i = 0; i < N_WARM_UP + N_MEASURED; ++i) {
        const double sink_us = llama_ssd_sink_time_us(ctx);
        const int64_t t_start = ggml_time_us();
        if (!step()) {
            return false;
        }
        llama_synchronize(ctx);
        const int64_t t_end = ggml_time_us();
        if (i >= N_WARM_UP) {
            steps.push_back({ (t_end - t_start) / 1e3, (llama_ssd_sink_time_us(ctx) - sink_us) / 1e3 });
        }
    }
    return true;
}

int fail(const std::string & why) {
    std::fprintf(stderr, "error: %s\n", why.c_str());
    return 1;
}

struct batch_guard {
    llama_batch batch;
    ~batch_guard() { llama_batch_free(batch); }
};
} // namespace

int server_decode_probe(server_context & ctx_server, common_params & params) {
    // Every routed expert in the CPU sink, whose time each step's is less: no GPU expert tier, no read-ahead. Set after
    // the launch's own settings, so they win, and before the context starts the SSD runtime.
    if (params.ssd_enabled) {
        params.ssd_overrides.emplace_back("vram.budget", "0");
        params.ssd_overrides.emplace_back("prefetch.depth", "0");
        try {
            common_ssd_prepare(params);
        } catch (const std::exception & e) {
            return fail(e.what());
        }
    }
    if (!ctx_server.load_model(params)) {
        return fail("the model did not load");
    }
    llama_context * ctx = ctx_server.get_llama_context();
    const llama_model * model = llama_get_model(ctx);
    const bool mtp = llama_model_has_mtp(model);

    // the longest draft the speculative decoding configured verifies (0 without one), as the context's rollback
    // planes count it: a verification decodes the last token and up to that many drafts
    const int n_max = (int) common_speculative_draft_bound(params.speculative);
    int n_positions = N_PROMPT + (mtp ? N_WARM_UP + N_MEASURED : 0);
    for (int rows = 1; rows <= n_max + 1; ++rows) {
        n_positions += (N_WARM_UP + N_MEASURED) * rows;
    }
    if (n_positions > (int) llama_n_ctx_seq(ctx)) {
        return fail("the probe decodes " + std::to_string(n_positions) + " positions, a sequence of the context holds " +
            std::to_string(llama_n_ctx_seq(ctx)));
    }

    const llama_tokens text = common_tokenize(llama_model_get_vocab(model), TEXT, /* add_special */ true);
    if (text.size() < 2) {
        return fail("the model's vocabulary gives the probe's text no tokens");
    }
    // the text's tokens, then its tokens after the first (a BOS) again, as long as positions are asked for
    const auto token_at = [&text](size_t i) { return i < text.size() ? text[i] : text[1 + (i - 1) % (text.size() - 1)]; };

    batch_guard owned = { llama_batch_init(std::max(N_PROMPT, n_max + 1), 0, 1) };
    llama_batch & batch = owned.batch;
    llama_pos pos = 0;

    // a request's prompt, as the server decodes one: its prefill phase, then the decode phase for the steps
    llama_memory_clear(llama_get_memory(ctx), true);
    if (!server_moe_workspace_transition(ctx, "request_begin", -1, N_PROMPT, N_PROMPT)) {
        return fail("the prompt's prefill phase did not start");
    }
    common_batch_clear(batch);
    for (int i = 0; i < N_PROMPT; ++i, ++pos) {
        common_batch_add(batch, token_at(pos), pos, { 0 }, i + 1 == N_PROMPT);
    }
    if (llama_decode(ctx, batch) != 0) {
        return fail("the prompt did not decode");
    }
    if (!server_moe_phase_report(ctx, "prefill_done", -1)) {
        return fail("the decode phase did not start after the prompt");
    }

    std::vector<step_time> steps;
    json rows = json::array();
    for (int n_rows = 1; n_rows <= n_max + 1; ++n_rows) {
        // a verification's decode: the next n_rows tokens, logits for each of them
        const auto decode = [&]() {
            common_batch_clear(batch);
            for (int i = 0; i < n_rows; ++i, ++pos) {
                common_batch_add(batch, token_at(pos), pos, { 0 }, true);
            }
            return llama_decode(ctx, batch) == 0;
        };
        if (!time_steps(ctx, decode, steps)) {
            return fail("a decode of " + std::to_string(n_rows) + " rows failed");
        }
        json row = json::object();
        row["rows"] = n_rows;
        add_medians(row, steps);
        rows.push_back(std::move(row));
    }

    json draft = nullptr;
    if (mtp) {
        // the head's steps as draft-mtp drafts: the first reads the hidden state of the target's last row and takes
        // the text's next token, as a draft does the token sampled from that row; each one after it reads the hidden
        // state the step before left and takes that step's token
        const float * h_last = llama_get_embeddings_nextn_ith(ctx, n_max);
        if (h_last == nullptr) {
            return fail("the target's decodes keep no hidden state for the MTP head");
        }
        std::vector<float> h(h_last, h_last + llama_model_n_embd_h(model));
        llama_token token = token_at(pos);
        const auto draft_step = [&]() {
            llama_token best;
            double p;
            if (!common_speculative_mtp_step(ctx, batch, h, token, pos, 0, best, p)) {
                return false;
            }
            token = best;
            ++pos;
            return true;
        };
        if (!time_steps(ctx, draft_step, steps)) {
            return fail("an MTP draft step failed");
        }
        draft = json::object();
        add_medians(draft, steps);
    }

    json out = json::object();
    out["version"] = 1;
    out["steps"]   = N_MEASURED;
    out["rows"]    = std::move(rows);
    out["draft"]   = std::move(draft);
    const std::string line = out.dump();
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    return std::fflush(stdout) == 0 ? 0 : 1;
}
