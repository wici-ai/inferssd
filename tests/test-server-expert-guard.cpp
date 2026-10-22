#include "server-expert-guard.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <vector>

using guard = server_expert_guard;
using token = guard::token_type;

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

static void accept(guard & observer, std::initializer_list<token> tokens) {
    for (const token value : tokens) {
        observer.accept_generated(value);
    }
}

static void test_default_and_n_zero_are_off() {
    guard observer;
    require(!observer.active() && !observer.start_active() && !observer.opener_active() && !observer.control_active() &&
            observer.remaining() == 0 && observer.start_remaining() == 0 && observer.control_remaining() == 0 &&
            observer.control_triggers() == 0, "default guard must be off");
    observer.reset(0, {});
    observer.prime_current_prefix({ 11, 12, 13 });
    accept(observer, { 11, 12, 13, 11, 12, 13 });
    observer.commit_forward(true);
    require(observer.state() == guard::phase::done && !observer.opened(), "N=0 must never arm");
}

static void test_n_one_protects_first_answer_prediction() {
    guard observer;
    observer.reset(1, { 11, 12, 13 });
    observer.accept_generated(11);
    require(!observer.active(), "first marker token must not activate guard");
    observer.commit_forward(true);
    observer.accept_generated(12);
    require(!observer.active(), "partial marker must not activate guard");
    observer.commit_forward(true);
    observer.accept_generated(13);
    require(observer.active() && observer.remaining() == 1,
            "guard must arm before final opener token's forward predicts answer #1");
    observer.commit_forward(true);
    require(!observer.active() && observer.remaining() == 0, "N=1 must finish after opener's forward");
    observer.accept_generated(21); // answer #1 sampled from that guarded forward
    require(!observer.active(), "answer #1 forward predicts answer #2 and must be aggressive for N=1");
}

static void test_n_eight_counts_successful_predictions() {
    guard observer;
    observer.reset(8, { 11, 12, 13 });
    accept(observer, { 11, 12, 13 });
    for (std::uint32_t i = 0; i < 8; ++i) {
        require(observer.active() && observer.remaining() == 8 - i, "each of N predictions must be protected");
        observer.commit_forward(true);
        observer.accept_generated(static_cast<token>(30 + i));
    }
    require(observer.state() == guard::phase::done && observer.remaining() == 0, "window must end after N forwards");
    observer.commit_forward(true);
    require(observer.remaining() == 0, "extra commit after completion must not underflow");
}

static void test_failed_forward_preserves_window() {
    guard observer;
    observer.reset(2, { 1, 2 });
    accept(observer, { 1, 2 });
    observer.commit_forward(false);
    observer.commit_forward(false);
    require(observer.active() && observer.remaining() == 2, "failed forward/retry must not consume prediction");
    observer.commit_forward(true);
    observer.commit_forward(false);
    require(observer.active() && observer.remaining() == 1, "failure after success must preserve remaining count");
    observer.commit_forward(true);
    require(!observer.active(), "second successful forward must finish N=2");
}

static void test_split_and_noise_tokens() {
    guard observer;
    observer.reset(2, { 11, 12, 13 });
    accept(observer, { 7, 11, 99, 12, 13 });
    require(!observer.active(), "discontinuous marker must not arm");
    accept(observer, { 11 });
    require(!observer.active() && observer.matched_prefix() == 1, "split marker part 1 must retain one prefix token");
    accept(observer, { 12 });
    require(!observer.active() && observer.matched_prefix() == 2, "split marker part 2 must retain two prefix tokens");
    accept(observer, { 13 });
    require(observer.active() && observer.remaining() == 2 && observer.matched_prefix() == 0, "complete split marker must arm and clear matcher prefix");
}

static void test_overlapping_prefixes() {
    guard observer;
    observer.reset(1, { 1, 2, 1, 2, 3 });
    accept(observer, { 1, 2, 1, 2, 1, 2 });
    require(!observer.active(), "overlapping marker prefix must not arm early");
    observer.accept_generated(3);
    require(observer.active(), "KMP fallback must keep a viable overlapping prefix");
}

static void test_repeated_token_marker() {
    guard observer;
    observer.reset(1, { 4, 4, 4, 5 });
    accept(observer, { 4, 4, 4, 4, 4 });
    require(!observer.active(), "repeated marker-prefix tokens must not count as full opener");
    observer.accept_generated(5);
    require(observer.active(), "repeated-prefix fallback must match final token");
}

static void test_repeated_opener_cannot_restart_active_or_done_window() {
    guard observer;
    observer.reset(2, { 1, 2 });
    accept(observer, { 1, 2 });
    observer.commit_forward(true);
    accept(observer, { 1, 2, 1, 2 });
    require(observer.remaining() == 1, "repeated opener while active must not reset N");
    observer.commit_forward(true);
    accept(observer, { 1, 2 });
    require(!observer.active() && observer.opened(), "opener after window completion must not rearm");
}

static void test_partial_current_prefix_combines_with_generated_opener() {
    guard observer;
    observer.reset(1, { 11, 12, 13 });
    observer.prime_current_prefix({ 90, 91, 11, 12 });
    require(!observer.active(), "partial current generation prefix must wait for generated completion");
    observer.accept_generated(13);
    require(observer.active(), "generated final opener token must complete prompt prefix");
    observer.commit_forward(true);
    require(!observer.active(), "prompt/generated split still guards exactly first answer prediction");
}

static void test_fully_primed_opener_guards_first_prefill_sample() {
    guard observer;
    observer.reset(1, { 11, 12, 13 });
    observer.prime_current_prefix({ 90, 11, 12, 13 });
    require(observer.active() && observer.remaining() == 1, "full current opener must arm before final prompt-row forward");
    // Parent forwards the final opener row separately, then samples answer #1.
    observer.commit_forward(true);
    observer.accept_generated(20);
    require(!observer.active(), "first sample after prefill already consumed N=1 protected prediction");
}

static void test_chunked_current_prefix_priming() {
    guard observer;
    observer.reset(2, { 11, 12, 13 });
    observer.prime_current_prefix({ 90, 11 });
    observer.prime_current_prefix({ 12 });
    require(!observer.active(), "chunked prefix must preserve only a partial match");
    observer.prime_current_prefix({ 13 });
    require(observer.active() && observer.remaining() == 2, "chunked prefix must complete one opener");
}

static void test_priming_after_generation_is_rejected() {
    guard observer;
    observer.reset(2, { 1, 2 });
    observer.accept_generated(1);
    bool rejected = false;
    try {
        observer.prime_current_prefix({ 2 });
    } catch (const std::logic_error &) {
        rejected = true;
    }
    require(rejected && !observer.active(), "late prompt replay must not alter current generated marker state");
    observer.accept_generated(2);
    require(observer.active(), "rejected late priming must preserve normal generated completion");
}

static void test_history_is_excluded_at_reset_boundary() {
    guard observer;
    observer.reset(2, { 11, 12, 13 });
    accept(observer, { 11, 12, 13 }); // previous assistant response
    observer.commit_forward(true);
    observer.finish();
    observer.reset(2, { 11, 12, 13 });
    // Caller supplies only the current assistant prefix. Historical openers are
    // deliberately not passed: the pure observer cannot infer message roles.
    observer.prime_current_prefix({ 80, 81 });
    require(!observer.active() && !observer.opened(), "prior assistant opener must not survive request reset");
    accept(observer, { 11, 12, 13 });
    require(observer.active() && observer.remaining() == 2, "current assistant opener must get fresh budget");
}

static void test_natural_and_forced_suffix_observation_are_identical() {
    guard natural;
    guard forced;
    natural.reset(3, { 11, 12, 13 });
    forced.reset(3, { 11, 12, 13 });
    for (const token value : { 7, 8, 9, 11, 12, 13 }) {
        // Source (sampled naturally vs reasoning-budget-forced) is intentionally
        // absent from this token-ID API: both use the same accepted token path.
        natural.accept_generated(value);
        forced.accept_generated(value);
        require(natural.active() == forced.active() && natural.remaining() == forced.remaining(),
                "forced and natural exact opener must share timing");
    }
    require(natural.active() && forced.remaining() == 3, "either suffix source must arm once");
}

static void test_eos_finishes_without_extra_forward() {
    guard observer;
    observer.reset(8, { 1, 2 });
    accept(observer, { 1, 2 });
    observer.commit_forward(true); // predicts EOS as the first response token
    observer.finish();            // caller observes EOS; no EOS forward occurs
    require(!observer.active() && observer.remaining() == 0, "EOS must cancel outstanding predictions");
    accept(observer, { 1, 2 });
    observer.commit_forward(true);
    require(!observer.active(), "tokens after finish cannot rearm completed request");
}

static void test_finish_before_opener_and_reset_discard_partial_match() {
    guard observer;
    observer.reset(2, { 1, 2, 3 });
    accept(observer, { 1, 2 });
    observer.finish();
    observer.accept_generated(3);
    require(!observer.opened(), "finish before complete marker must not activate");
    observer.reset(2, { 1, 2, 3 });
    observer.accept_generated(3);
    require(!observer.active(), "reset must discard prior partial marker");
    accept(observer, { 1, 2, 3 });
    require(observer.active(), "fresh complete marker must work after reset");
}

static void test_single_token_marker() {
    guard observer;
    observer.reset(1, { 42 });
    observer.prime_current_prefix({ 7, 8 });
    observer.accept_generated(42);
    require(observer.active(), "single-token opener must arm before its forward");
    observer.commit_forward(true);
    observer.accept_generated(42);
    require(!observer.active(), "repeated single-token opener must not rearm");
}

static void test_invalid_marker_fails_closed() {
    guard observer;
    observer.reset(2, { 1 });
    observer.accept_generated(1);
    for (const std::vector<token> & marker : { std::vector<token>{}, std::vector<token>{ 1, -1 } }) {
        bool rejected = false;
        try {
            observer.reset(2, marker, 4);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        require(rejected && !observer.active() && observer.remaining() == 0 && observer.start_remaining() == 0,
                "invalid positive-N marker must throw after discarding prior guard state");
    }
}

static void test_start_one_protects_first_sample() {
    guard observer;
    observer.reset(0, {}, 1);
    observer.prime_current_prefix({ 90, 91 });
    require(observer.active() && observer.start_active() && !observer.opener_active() &&
            observer.start_remaining() == 1 && observer.remaining() == 0,
            "start N=1 must arm before any generated token or opener");
    // Earlier prompt rows are deliberately not committed. This is the final
    // singleton prompt row, whose logits predict generated token #1.
    observer.commit_forward(true);
    observer.accept_generated(21);
    require(!observer.active() && observer.start_remaining() == 0,
            "generated token #1 forward predicts #2 and must be primary for start N=1");
    observer.commit_forward(true);
    require(observer.start_remaining() == 0, "completed start budget must not underflow");
}

static void test_start_only_ignores_absent_or_malformed_opener() {
    guard observer;
    observer.reset(0, {}, 3);
    // Arbitrary raw text/control-tail token IDs never form a response opener.
    for (const token value : { 81, 82, 83 }) {
        require(observer.active() && !observer.opener_active(), "start-only guard must not require an opener");
        observer.commit_forward(true);
        observer.accept_generated(value);
    }
    accept(observer, { 81, 81, 82, 83 });
    require(!observer.active() && !observer.opened() && observer.state() == guard::phase::done,
            "raw structural tail must not rearm a start-only guard");
}

static void test_start_and_generated_opener_overlap_as_union() {
    guard observer;
    observer.reset(3, { 11, 12 }, 4);
    observer.commit_forward(true); // first sample: 11
    observer.accept_generated(11);
    observer.commit_forward(true); // second sample: 12
    observer.accept_generated(12);
    require(observer.start_remaining() == 2 && observer.remaining() == 3 && observer.opened(),
            "opener must be observed while start window is active");
    observer.commit_forward(true); // first answer prediction, overlap
    require(observer.start_remaining() == 1 && observer.remaining() == 2,
            "one overlap forward must decrement both windows exactly once");
    observer.accept_generated(21);
    observer.commit_forward(true); // second answer prediction, overlap
    require(!observer.start_active() && observer.opener_active() && observer.remaining() == 1,
            "opener tail must outlive the start window");
    observer.accept_generated(22);
    observer.commit_forward(true); // fifth total guarded prediction, not seven
    require(!observer.active(), "overlap must use the union of windows, not sum their lengths");
    accept(observer, { 11, 12 });
    require(!observer.active(), "repeated opener must not rearm either completed window");
}

static void test_opener_completion_does_not_end_longer_start_window() {
    guard observer;
    observer.reset(1, { 11, 12 }, 4);
    observer.prime_current_prefix({ 90, 11, 12 });
    observer.commit_forward(true);
    require(observer.start_active() && !observer.opener_active() && observer.state() == guard::phase::done &&
            observer.start_remaining() == 3, "opener phase done must not hide the remaining start window");
    accept(observer, { 11, 12 });
    require(observer.remaining() == 0 && observer.start_remaining() == 3, "repeated opener must not extend either window");
    for (int i = 0; i < 3; ++i) {
        observer.commit_forward(true);
    }
    require(!observer.active(), "longer start window must finish at its original budget");
}

static void test_delayed_opener_arms_after_start_expires() {
    guard observer;
    observer.reset(2, { 11, 12 }, 1);
    observer.commit_forward(true);
    observer.accept_generated(80);
    require(!observer.active() && observer.state() == guard::phase::wait_open,
            "start expiry must preserve the pending opener observer");
    observer.commit_forward(true); // primary prediction in the gap
    observer.accept_generated(11);
    observer.commit_forward(true); // primary prediction of the final opener token
    observer.accept_generated(12);
    require(observer.opener_active() && !observer.start_active() && observer.remaining() == 2,
            "delayed opener must retain its independent full budget");
    observer.commit_forward(true);
    observer.commit_forward(true);
    require(!observer.active(), "delayed opener must consume exactly its own two predictions");
}

static void test_primed_windows_share_first_prediction() {
    guard observer;
    observer.reset(3, { 11, 12 }, 1);
    observer.prime_current_prefix({ 90, 11 });
    observer.prime_current_prefix({ 12 });
    require(observer.start_remaining() == 1 && observer.remaining() == 3,
            "priming must not consume either window");
    observer.commit_forward(true); // final prompt row predicts answer #1
    require(observer.start_remaining() == 0 && observer.remaining() == 2,
            "first primed prediction must consume both windows");
    observer.commit_forward(true);
    observer.commit_forward(true);
    require(!observer.active(), "primed windows must cost max(start, opener), not their sum");
}

static void test_start_overlap_failure_reset_and_finish() {
    guard observer;
    observer.reset(2, { 11 }, 3);
    observer.prime_current_prefix({ 11 });
    observer.commit_forward(false);
    observer.commit_forward(false);
    require(observer.start_remaining() == 3 && observer.remaining() == 2, "failed overlap must preserve both budgets");
    observer.commit_forward(true);
    observer.commit_forward(false);
    require(observer.start_remaining() == 2 && observer.remaining() == 1, "retry after success must not consume budgets");
    observer.finish(); // EOS, cancellation, or error: no terminal-token forward
    observer.accept_generated(11);
    observer.commit_forward(true);
    require(!observer.active() && observer.start_remaining() == 0 && observer.remaining() == 0,
            "finish must cancel both windows without permitting rearm");
    observer.reset(2, { 11, 12 }, 1);
    observer.prime_current_prefix({ 11 });
    observer.reset(2, { 11, 12 }); // legacy call must clear the prior start window
    observer.accept_generated(12);
    require(!observer.active() && !observer.opened() && observer.start_remaining() == 0,
            "reset must discard both start budget and partial opener history");
    observer.reset(0, {}, 2);
    require(observer.active() && observer.start_remaining() == 2, "fresh start-only request must receive a new budget");
    observer.finish(); // stop before first prediction
    require(!observer.active(), "finish before generation must cancel the start window");
}

static void test_control_zero_is_off() {
    guard observer;
    observer.reset(0, {}, 0, 0, -1, -1); // IDs need not be supplied when disabled
    accept(observer, { 11, 12, 11, 12 });
    observer.commit_forward(true);
    require(!observer.active() && observer.control_triggers() == 0, "M=0 must not observe or activate control markers");
    observer.reset(0, {}, 0, 0, 11, 12);
    accept(observer, { 11, 12 });
    require(!observer.control_active() && observer.control_triggers() == 0, "supplied IDs must not enable M=0");
}

static void test_control_one_protects_prediction_after_marker() {
    guard observer;
    observer.reset(0, {}, 0, 1, 11, 12);
    require(!observer.active(), "control-only guard must not protect the prediction which emits the first marker");
    observer.commit_forward(true); // primary logits emit open
    observer.accept_generated(11);
    require(observer.active() && observer.control_remaining() == 1 && observer.control_triggers() == 1,
            "accepted control marker must arm before its own forward");
    observer.commit_forward(true); // open's forward predicts the following token
    observer.accept_generated(20);
    require(!observer.active(), "M=1 must protect exactly the next prediction after open");
    observer.accept_generated(12);
    require(observer.control_remaining() == 1 && observer.control_triggers() == 2, "close must independently trigger");
    observer.commit_forward(true);
    observer.commit_forward(true);
    require(!observer.active() && observer.control_remaining() == 0, "completed control window must not underflow");
}

static void test_prompt_controls_do_not_trigger() {
    guard observer;
    observer.reset(1, { 11, 21, 22 }, 0, 3, 11, 12);
    observer.prime_current_prefix({ 12, 30, 22, 11 });
    observer.prime_current_prefix({ 21, 22 });
    require(observer.opener_active() && !observer.control_active() && observer.control_triggers() == 0,
            "primed opener may arm once but its prompt open/close markers must not arm control window");
    observer.commit_forward(true);
    observer.accept_generated(20);
    require(!observer.active(), "prompt control markers must not leave a deferred control window");
}

static void test_repeated_and_alternating_controls_rearm() {
    guard observer;
    observer.reset(0, {}, 0, 3, 11, 12);
    for (const token value : { 11, 11, 12, 11 }) {
        observer.accept_generated(value);
        require(observer.control_remaining() == 3, "every repeated/alternating marker must reset budget to M");
        observer.commit_forward(true);
        require(observer.control_remaining() == 2, "trigger forward must consume exactly one prediction");
    }
    require(observer.control_triggers() == 4, "each accepted marker must be counted once");
    observer.commit_forward(true);
    observer.commit_forward(true);
    require(!observer.active(), "repeated markers must reset, not add budgets");
    observer.accept_generated(12);
    require(observer.control_remaining() == 3 && observer.control_triggers() == 5,
            "later close must retrigger after all earlier windows expire");
}

static void test_malformed_json_control_tail_schedule() {
    guard observer;
    observer.reset(0, {}, 0, 3, 11, 12);
    // Synthetic token IDs for {"ok":true}<|close|>response<|sep|>
    // <|close|>message<|sep|><|open|>message garbage. The literal JSON/control
    // spellings are not a parser: only actual configured marker IDs trigger.
    const token tail[] = { 70, 71, 72, 12, 21, 22, 12, 23, 22, 11, 23, 24, 25 };
    const bool guarded[] = { false, false, false, true, true, true, true, true, true, true, true, true, false };
    unsigned predictions = 0;
    for (std::size_t i = 0; i < sizeof(tail)/sizeof(tail[0]); ++i) {
        observer.accept_generated(tail[i]);
        require(observer.active() == guarded[i], "malformed structural tail must follow the exact marker-based schedule");
        predictions += observer.active();
        observer.commit_forward(true);
    }
    require(predictions == 9 && observer.control_triggers() == 3 && !observer.active(),
            "three separated synthetic control markers must protect nine predictions");
}

static void test_three_windows_overlap_as_union() {
    guard observer;
    observer.reset(2, { 11, 21, 22 }, 4, 3, 11, 12);
    observer.commit_forward(true); // first sample, start-only
    observer.accept_generated(11);
    observer.commit_forward(true); // start + control
    observer.accept_generated(21);
    observer.commit_forward(true); // start + control
    observer.accept_generated(22);
    require(observer.start_remaining() == 1 && observer.remaining() == 2 && observer.control_remaining() == 1,
            "opener must be observed while start and control windows are active");
    observer.commit_forward(true); // all three, first answer prediction
    require(!observer.start_active() && !observer.control_active() && observer.remaining() == 1,
            "one forward in triple overlap must consume one unit from every active budget");
    observer.accept_generated(30);
    observer.commit_forward(true); // opener-only, five union predictions total
    require(!observer.active(), "three windows must protect their union, not sum their budgets");
}

static void test_control_retrigger_does_not_rearm_opener() {
    guard observer;
    observer.reset(1, { 11, 21, 22 }, 0, 2, 11, 12);
    accept(observer, { 11, 21, 22 });
    observer.commit_forward(true);
    observer.commit_forward(true);
    require(observer.opened() && !observer.active(), "initial opener/control windows must complete");
    accept(observer, { 11, 21, 22 });
    require(observer.remaining() == 0 && observer.state() == guard::phase::done &&
            observer.control_remaining() == 2 && observer.control_triggers() == 2,
            "repeated opener must retrigger only its generated control marker window");
}

static void test_control_retry_finish_and_reset() {
    guard observer;
    observer.reset(2, { 11 }, 4, 3, 11, 12);
    observer.accept_generated(11);
    observer.commit_forward(false);
    observer.commit_forward(false);
    require(observer.start_remaining() == 4 && observer.remaining() == 2 && observer.control_remaining() == 3 &&
            observer.control_triggers() == 1, "failed forwards must neither consume budgets nor replay marker acceptance");
    observer.commit_forward(true);
    observer.commit_forward(false);
    require(observer.start_remaining() == 3 && observer.remaining() == 1 && observer.control_remaining() == 2,
            "retry after success must preserve each window's remaining count");
    observer.finish(); // EOS/cancellation/error, without a terminal-token forward
    accept(observer, { 11, 12, 11 });
    observer.commit_forward(true);
    require(!observer.active() && observer.control_remaining() == 0 && observer.control_triggers() == 1,
            "finish must prevent subsequent control markers from resurrecting any window");
    observer.reset(0, {}, 0, 2, 11, 12);
    require(observer.control_triggers() == 0, "reset must discard previous request trigger count");
    observer.accept_generated(12);
    require(observer.control_remaining() == 2 && observer.control_triggers() == 1, "fresh request must accept new triggers");
    observer.reset(0, {}); // old call signature disables the new control window
    observer.accept_generated(12);
    require(!observer.active() && observer.control_triggers() == 0, "legacy reset must discard prior control policy");
}

static void test_invalid_control_ids_fail_closed() {
    guard observer;
    for (const auto & ids : { std::vector<token>{ -1, 12 }, std::vector<token>{ 11, -1 }, std::vector<token>{ 11, 11 } }) {
        observer.reset(1, { 11 }, 2, 3, 11, 12);
        observer.accept_generated(11);
        bool rejected = false;
        try { observer.reset(1, { 11 }, 2, 3, ids[0], ids[1]); }
        catch (const std::invalid_argument &) { rejected = true; }
        accept(observer, { 11, 12 });
        require(rejected && !observer.active() && observer.control_triggers() == 0,
                "invalid control IDs must discard all previous state and prevent reactivation");
    }
}

static void test_control_works_after_start_expires_or_finish_before_trigger() {
    guard observer;
    observer.reset(0, {}, 1, 2, 11, 12);
    observer.commit_forward(true);
    observer.accept_generated(30);
    require(!observer.active(), "expired start window must leave an unguarded gap before a control marker");
    observer.accept_generated(11);
    require(observer.control_active() && !observer.start_active(), "control must remain armed for future triggers after start expires");
    observer.finish();
    observer.reset(0, {}, 0, 2, 11, 12);
    observer.finish(); // request ended before any marker was generated
    observer.accept_generated(11);
    require(!observer.active() && observer.control_triggers() == 0, "finish before first marker must also close control observation");
}

int main() {
    test_default_and_n_zero_are_off();
    test_n_one_protects_first_answer_prediction();
    test_n_eight_counts_successful_predictions();
    test_failed_forward_preserves_window();
    test_split_and_noise_tokens();
    test_overlapping_prefixes();
    test_repeated_token_marker();
    test_repeated_opener_cannot_restart_active_or_done_window();
    test_partial_current_prefix_combines_with_generated_opener();
    test_fully_primed_opener_guards_first_prefill_sample();
    test_chunked_current_prefix_priming();
    test_priming_after_generation_is_rejected();
    test_history_is_excluded_at_reset_boundary();
    test_natural_and_forced_suffix_observation_are_identical();
    test_eos_finishes_without_extra_forward();
    test_finish_before_opener_and_reset_discard_partial_match();
    test_single_token_marker();
    test_invalid_marker_fails_closed();
    test_start_one_protects_first_sample();
    test_start_only_ignores_absent_or_malformed_opener();
    test_start_and_generated_opener_overlap_as_union();
    test_opener_completion_does_not_end_longer_start_window();
    test_delayed_opener_arms_after_start_expires();
    test_primed_windows_share_first_prediction();
    test_start_overlap_failure_reset_and_finish();
    test_control_zero_is_off();
    test_control_one_protects_prediction_after_marker();
    test_prompt_controls_do_not_trigger();
    test_repeated_and_alternating_controls_rearm();
    test_malformed_json_control_tail_schedule();
    test_three_windows_overlap_as_union();
    test_control_retrigger_does_not_rearm_opener();
    test_control_retry_finish_and_reset();
    test_invalid_control_ids_fail_closed();
    test_control_works_after_start_expires_or_finish_before_trigger();
    std::puts("server expert guard: 35 CPU observer tests passed (start, opener, control, overlap, retry, reset, EOS)");
    return 0;
}
