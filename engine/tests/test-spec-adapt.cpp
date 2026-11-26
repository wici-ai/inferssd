// The draft length common_speculative_adapt picks from the costs it measures.
#include "speculative.h"

#include <cstdio>
#include <cstdlib>

static void require(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

// a machine: a decode of r rows takes t1 + (r - 1) * row_ms, a draft step step_ms; the k-th draft token is accepted
// with probability accept when every one before it was (deterministically: every block accepts its first `ok`)
static int settle(double t1, double row_ms, double step_ms, int ok, int n_max) {
    common_speculative_adapt a;
    int last = -1;
    for (int i = 0; i < 400; ++i) {
        const int n = a.choose(n_max);
        if (n == 0) {
            a.on_plain(t1);
        } else {
            const int accepted = ok < n ? ok : n;
            a.on_block(n, accepted, n, t1 + n * row_ms, n * step_ms);
        }
        if (i >= 300 && i % common_speculative_adapt::probe_every != 0) {
            last = n;
        }
    }
    return last;
}

int main() {
    // warm-up goes round every length, twice
    {
        common_speculative_adapt a;
        for (int i = 0; i < 2 * 4; ++i) {
            require(a.choose(3) == i % 4, "warm-up order");
        }
    }
    // a row costs little and every draft is accepted: the longest draft
    require(settle(100.0, 10.0, 2.0, 3, 3) == 3, "cheap rows, all accepted");
    // a row costs as much as a token (a decode bound by the drive): no draft
    require(settle(100.0, 100.0, 5.0, 3, 3) == 0, "rows as dear as tokens");
    // rows are cheap but only the first draft token is ever accepted: one
    require(settle(100.0, 20.0, 2.0, 1, 3) == 1, "only the first accepted");
    // no draft is ever accepted: no draft
    require(settle(100.0, 10.0, 2.0, 0, 3) == 0, "nothing accepted");
    // nothing to draft
    {
        common_speculative_adapt a;
        require(a.choose(0) == 0, "n_max 0");
    }
    std::printf("ok\n");
    return 0;
}
