#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Validate the current generation-prompt boundary before priming an observer.
// Token IDs must come from actual tokenization of these strings; this helper
// checks the exact tail binding, while the caller owns that tokenizer evidence.
inline void server_expert_guard_verify_prefix(const std::string & generation_prefix,
        const std::vector<std::int32_t> & prefix_tokens, const std::vector<std::int32_t> & prompt_tokens) {
    const std::string assistant = "<|open|>message role=\"assistant\"<|sep|>";
    const std::string marker = "<|open|>response<|sep|>";
    const auto opened = generation_prefix.find(marker);
    if (generation_prefix.compare(0, assistant.size(), assistant) != 0 ||
            generation_prefix.find("<|end_of_msg|>") != std::string::npos ||
            generation_prefix.find("<|close|>message") != std::string::npos ||
            (opened != std::string::npos && opened + marker.size() != generation_prefix.size()) ||
            prefix_tokens.empty() || prefix_tokens.size() > prompt_tokens.size() ||
            !std::equal(prefix_tokens.rbegin(), prefix_tokens.rend(), prompt_tokens.rbegin())) {
        throw std::invalid_argument("expert guard requires a verified current K3 assistant generation prefix at the prompt tail; assistant prefill/ambiguous prefixes are unsupported");
    }
}

// Prediction windows for generation start, the response opener, and generated
// control markers. The windows overlap; only control markers rearm their own
// window. This observer has no
// tokenizer, role parser, sampler, model state, routing policy or forward call.
// The caller owns the boundary between historical prompt and current assistant.
class server_expert_guard {
public:
    using token_type = std::int32_t;

    // This phase describes only the opener window; the others are separate.
    enum class phase {
        wait_open,
        guard,
        done,
    };

    // The start window is active immediately. The caller must exclude earlier
    // prefill and begin committing at the singleton final prompt-row forward,
    // whose logits predict generated token #1. Priming consumes no budget.
    void reset(std::uint32_t n, const std::vector<token_type> & marker, std::uint32_t start_n = 0,
            std::uint32_t control_n = 0, token_type control_open = -1, token_type control_close = -1) {
        finish();
        marker_.clear();
        prefix_.clear();
        budget_ = n;
        opened_ = false;
        generated_started_ = false;
        control_triggers_ = 0;
        if (control_n > 0 && (control_open < 0 || control_close < 0 || control_open == control_close)) {
            throw std::invalid_argument("expert control guard requires distinct nonnegative open/close token IDs");
        }
        if (n > 0) {
            if (marker.empty()) {
                throw std::invalid_argument("expert guard requires a nonempty response-opener token sequence");
            }
            for (const token_type token : marker) {
                if (token < 0) {
                    throw std::invalid_argument("expert guard opener contains an invalid token ID");
                }
            }
            marker_ = marker;
            prefix_.resize(marker_.size(), 0);
            // KMP failure links preserve a possible overlapping marker prefix.
            for (std::size_t i = 1, matched = 0; i < marker_.size(); ++i) {
                while (matched > 0 && marker_[i] != marker_[matched]) {
                    matched = prefix_[matched - 1];
                }
                if (marker_[i] == marker_[matched]) {
                    ++matched;
                }
                prefix_[i] = matched;
            }
        }
        start_remaining_ = start_n;
        control_budget_ = control_n;
        control_open_ = control_open;
        control_close_ = control_close;
        running_ = true;
        phase_ = n > 0 ? phase::wait_open : phase::done;
    }

    // Prime only the verified CURRENT assistant generation-prompt suffix, never
    // historical turns or the full chat prompt. The caller must schedule these
    // tokens' forwards separately: when this completes an opener, query active()
    // before the final opener token's forward, whose logits predict answer #1.
    // No forward is implied by priming; chunked priming before generation is OK.
    // Prompt tokens never trigger the generated-control window.
    void prime_current_prefix(const std::vector<token_type> & tokens) {
        if (generated_started_) {
            throw std::logic_error("expert guard prompt priming must precede generated tokens");
        }
        for (const token_type token : tokens) {
            observe(token);
        }
    }

    // Call after accepting a generated token, before that token's next forward.
    // The final opener token arms the window immediately; repeated openers never
    // restart or extend it. A sampled EOS must call finish() instead of forwarding.
    // Each generated open/close marker rearms the control window before its own
    // forward, protecting the next M predictions, regardless of opener state.
    void accept_generated(token_type token) {
        if (!running_) { return; }
        generated_started_ = true;
        if (control_budget_ > 0 && (token == control_open_ || token == control_close_)) {
            control_remaining_ = control_budget_;
            ++control_triggers_;
        }
        observe(token);
    }

    bool active() const {
        return start_active() || opener_active() || control_active();
    }

    bool start_active() const {
        return start_remaining_ > 0;
    }

    bool opener_active() const {
        return phase_ == phase::guard;
    }

    bool control_active() const {
        return control_remaining_ > 0;
    }

    std::uint32_t control_remaining() const {
        return control_remaining_;
    }

    std::uint64_t control_triggers() const {
        return control_triggers_;
    }

    std::uint32_t start_remaining() const {
        return start_remaining_;
    }

    // Opener budget only; use active() for the union of all windows.
    std::uint32_t remaining() const {
        return remaining_;
    }

    phase state() const {
        return phase_;
    }

    std::size_t matched_prefix() const {
        return matched_;
    }

    bool opened() const {
        return opened_;
    }

    // Commit exactly once after a successfully executed forward which was
    // scheduled with active()==true. Failures/retries consume no prediction.
    // A successful overlapping prediction consumes one unit from each window.
    // For a prompt-primed opener this includes its final prompt-row forward, not
    // the subsequent sampled answer token's forward (the N=1 boundary).
    void commit_forward(bool success) {
        if (!success) {
            return;
        }
        if (start_active()) {
            --start_remaining_;
        }
        if (control_active()) {
            --control_remaining_;
        }
        if (opener_active()) {
            --remaining_;
            if (remaining_ == 0) {
                phase_ = phase::done;
            }
        }
    }

    // Stop on EOS/cancellation/error, without scheduling a spurious last forward.
    void finish() {
        phase_ = phase::done;
        remaining_ = 0;
        start_remaining_ = 0;
        control_remaining_ = 0;
        running_ = false;
        matched_ = 0;
    }

private:
    void observe(token_type token) {
        if (phase_ != phase::wait_open) {
            return;
        }
        while (matched_ > 0 && token != marker_[matched_]) {
            matched_ = prefix_[matched_ - 1];
        }
        if (token == marker_[matched_]) {
            ++matched_;
        }
        if (matched_ == marker_.size()) {
            opened_ = true;
            remaining_ = budget_;
            matched_ = 0;
            phase_ = phase::guard;
        }
    }

    phase phase_ = phase::done;
    std::uint32_t budget_ = 0;
    std::uint32_t remaining_ = 0;
    std::uint32_t start_remaining_ = 0;
    std::uint32_t control_budget_ = 0;
    std::uint32_t control_remaining_ = 0;
    std::uint64_t control_triggers_ = 0;
    token_type control_open_ = -1;
    token_type control_close_ = -1;
    std::size_t matched_ = 0;
    bool opened_ = false;
    bool generated_started_ = false;
    bool running_ = false;
    std::vector<token_type> marker_;
    std::vector<std::size_t> prefix_;
};
