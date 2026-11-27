// SPDX-License-Identifier: Apache-2.0
// The draft length common_speculative_adapt picks from the costs it measures.
#include "speculative.h"

#include <algorithm>
#include <cstdio>
#include <vector>
#include <cstdlib>

static void require(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

// a machine: a decode of r rows takes t1 + (r - 1) * row_ms, a draft step step_ms, and a change of length switch_ms
// once; a block accepts its first `ok` draft tokens; a head that is never sure enough (`propose` false) drafts nothing.
// The length it settles on: the one it runs most in its last 200 steps.
static int settle(double t1, double row_ms, double step_ms, int ok, int n_max, bool propose = true,
        double switch_ms = 0.0) {
    common_speculative_adapt a;
    std::vector<int> runs(n_max + 1, 0);
    int prev = -1;
    for (int i = 0; i < 600; ++i) {
        const int n = a.choose(n_max);
        const double extra = n != prev ? switch_ms : 0.0;
        prev = n;
        if (n == 0) {
            a.on_plain(t1 + extra);
        } else if (!propose) {
            a.on_empty(n, step_ms);
            a.on_plain(t1 + extra);
        } else {
            const int accepted = ok < n ? ok : n;
            a.on_block(n, accepted, t1 + n * row_ms + n * step_ms + extra);
        }
        if (i >= 400) {
            runs[n]++;
        }
    }
    return int(std::max_element(runs.begin(), runs.end()) - runs.begin());
}

int main() {
    // every length runs a burst first
    {
        common_speculative_adapt a;
        for (int i = 0; i < 4 * common_speculative_adapt::burst; ++i) {
            require(a.choose(3) == i / common_speculative_adapt::burst, "warm-up order");
        }
    }
    // a change of length that costs a lot once does not make the length that keeps running look slow
    require(settle(100.0, 100.0, 5.0, 3, 3, true, 400.0) == 0, "switching cost, rows dear");
    require(settle(100.0, 10.0, 2.0, 3, 3, true, 400.0) == 3, "switching cost, rows cheap");
    // a row costs little and every draft is accepted: the longest draft
    require(settle(100.0, 10.0, 2.0, 3, 3) == 3, "cheap rows, all accepted");
    // a row costs as much as a token (a decode bound by the drive): no draft
    require(settle(100.0, 100.0, 5.0, 3, 3) == 0, "rows as dear as tokens");
    // rows are cheap but only the first draft token is ever accepted: one
    require(settle(100.0, 20.0, 2.0, 1, 3) == 1, "only the first accepted");
    // no draft is ever accepted: no draft
    require(settle(100.0, 10.0, 2.0, 0, 3) == 0, "nothing accepted");
    // the head never proposes: its steps only cost time
    require(settle(100.0, 10.0, 2.0, 3, 3, false) == 0, "never proposed");
    // a length slow in its first burst (a cold expert cache) and fast later: probed again, it is compared by that
    // burst, not by a mean that reaches back to the cold start, and wins. Length 0 takes 200 ms a token in its warm-up
    // and 60 ms after; length 1 gives 2 tokens in 130 ms, 65 ms a token.
    {
        common_speculative_adapt a;
        const int burst = common_speculative_adapt::burst;
        // both warm-up bursts, the best length until the probe, and the probe's burst of length 0
        for (int i = 0; i < 2 * burst + common_speculative_adapt::probe_every + burst; ++i) {
            if (a.choose(1) == 0) {
                a.on_plain(i < burst ? 200.0 : 60.0);
            } else {
                a.on_block(1, 1, 130.0);
            }
        }
        require(a.choose(1) == 0, "a length probed again after a cold start");
    }
    // nothing to draft
    {
        common_speculative_adapt a;
        require(a.choose(0) == 0, "n_max 0");
    }
    std::printf("ok\n");
    return 0;
}
