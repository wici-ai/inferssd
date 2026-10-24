#include "../../ggml/src/ggml-ssd/io/submit-window.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#undef NDEBUG
#include <cassert>

// Fake I/O completes in reverse order; fake H2D keeps reading its staging
// buffer until finish(). This exercises the production submit controller
// without requiring liburing, CUDA, or a model.
struct fake_io {
    struct event {
        char kind;
        size_t count;
    };

    size_t jobs;
    size_t depth;
    size_t max_submit = SIZE_MAX;
    size_t ready_after_submits = 0;
    size_t submit_calls = 0;
    size_t drain_calls = 0;
    size_t bytes = 0;
    std::vector<size_t> prepared;
    std::vector<size_t> submitted;
    std::vector<size_t> completions;
    std::vector<size_t> h2d;
    std::vector<size_t> remaining_args;
    std::vector<event> events;
    std::vector<bool> occupied;
    std::vector<std::vector<uint8_t>> source;
    std::vector<std::vector<uint8_t>> host;
    std::vector<std::vector<uint8_t>> device;

    fake_io(size_t job_count, size_t io_depth) :
        jobs(job_count), depth(io_depth), occupied(depth), source(jobs), host(depth), device(jobs) {
        for (size_t i = 0; i < jobs; ++i) {
            source[i].resize(64 + i * 3);
            for (size_t j = 0; j < source[i].size(); ++j) {
                source[i][j] = static_cast<uint8_t>((i * 17 + j * 31) % 251);
            }
        }
    }

    bool prepare(size_t job) {
        assert(job < jobs && job < depth);
        // A harvested CQE has not released the host buffer used by H2D.
        assert(!occupied[job]);
        occupied[job] = true;
        prepared.push_back(job);
        events.push_back({'P', job});
        return true;
    }

    int submit(size_t remaining) {
        ++submit_calls;
        remaining_args.push_back(remaining);
        assert(remaining == prepared.size());
        const size_t count = std::min(remaining, max_submit);
        assert(count > 0);
        events.push_back({'S', count});
        for (size_t i = 0; i < count; ++i) {
            const size_t job = prepared[i];
            host[job] = source[job];
            submitted.push_back(job);
            completions.push_back(job);
        }
        prepared.erase(prepared.begin(), prepared.begin() + count);
        return static_cast<int>(count);
    }

    bool drain() {
        ++drain_calls;
        if (submit_calls < ready_after_submits) {
            return true;
        }
        while (!completions.empty()) {
            const size_t job = completions.back();
            completions.pop_back();
            assert(std::find(h2d.begin(), h2d.end(), job) == h2d.end());
            h2d.push_back(job);
            bytes += source[job].size();
            events.push_back({'H', job});
        }
        return true;
    }

    bool run(size_t chunk) {
        return ggml_moe_common::object_io_submit_initial(
            jobs, depth, chunk,
            [&](size_t job) { return prepare(job); },
            [&](size_t remaining) {
                return ggml_moe_common::object_io_submit_chunk_result(submit(remaining), remaining, false);
            },
            [&] { return drain(); });
    }

    void finish() {
        assert(drain());
        const size_t limit = std::min(jobs, depth);
        assert(submitted.size() == limit && h2d.size() == limit);
        size_t expected_bytes = 0;
        for (size_t i = 0; i < limit; ++i) {
            assert(submitted[i] == i);
            expected_bytes += source[i].size();
        }
        for (const size_t job : h2d) {
            device[job] = host[job];
            assert(device[job] == source[job]);
            occupied[job] = false;
        }
        assert(bytes == expected_bytes);
    }
};

static void test_early_h2d_and_payload() {
    fake_io pipelined(16, 16);
    assert(pipelined.run(4));
    assert(pipelined.submit_calls == 4 && pipelined.drain_calls == 3);
    assert(pipelined.completions.size() == 4 && pipelined.h2d.size() == 12);
    size_t first_h2d = SIZE_MAX;
    size_t last_submit = 0;
    for (size_t i = 0; i < pipelined.events.size(); ++i) {
        if (pipelined.events[i].kind == 'H' && first_h2d == SIZE_MAX) {
            first_h2d = i;
        }
        if (pipelined.events[i].kind == 'S') {
            last_submit = i;
        }
    }
    assert(first_h2d < last_submit);
    pipelined.finish();
    fake_io baseline(16, 16);
    assert(baseline.run(0));
    assert(baseline.submit_calls == 1 && baseline.h2d.empty());
    baseline.finish();
    assert(pipelined.submitted == baseline.submitted);
    assert(pipelined.bytes == baseline.bytes && pipelined.device == baseline.device);
}

static void test_initial_slot_budget() {
    fake_io io(17, 4);
    assert(io.run(1));
    // Draining the first three reads must not enlarge the initial budget or
    // reuse their host buffers while the delayed H2D operations still own them.
    assert(io.submitted.size() == 4 && io.h2d.size() == 3);
    assert(io.completions.size() == 1 && io.drain_calls == 3);
    assert(std::count(io.occupied.begin(), io.occupied.end(), true) == 4);
    io.finish();
    for (size_t i = 4; i < io.jobs; ++i) {
        assert(io.device[i].empty());
    }
}

static void test_empty_completions() {
    fake_io io(8, 8);
    io.ready_after_submits = 4;
    assert(io.run(2));
    assert(io.submit_calls == 4 && io.drain_calls == 3);
    assert(io.h2d.empty() && io.completions.size() == 8);
    io.finish();
}

static void test_chunk_boundaries() {
    const size_t chunks[] = {0, 1, 2, 3, 4, 99};
    const size_t calls[] = {1, 4, 2, 2, 1, 1};
    for (size_t i = 0; i < sizeof(chunks) / sizeof(chunks[0]); ++i) {
        fake_io io(7, 4);
        assert(io.run(chunks[i]));
        assert(io.submit_calls == calls[i] && io.drain_calls == calls[i] - 1);
        assert(!io.completions.empty());
        io.finish();
    }
    fake_io one(1, 8);
    assert(one.run(1) && one.drain_calls == 0 && one.completions.size() == 1);
    one.finish();
    fake_io empty(0, 0);
    assert(empty.run(0) && empty.events.empty() && empty.drain_calls == 0);
    fake_io no_depth(1, 0);
    assert(!no_depth.run(1) && no_depth.events.empty() && no_depth.drain_calls == 0);
}

static void test_adaptive_chunks() {
    using ggml_moe_common::object_io_adaptive_submit_chunk;
    struct test_case {
        size_t jobs;
        size_t depth;
        size_t chunk;
        std::vector<size_t> submissions;
    };
    const test_case cases[] = {
        { 0,  0,  0, {}},
        { 1, 16,  1, {1}},
        { 4, 16,  4, {4}},
        { 5, 16,  4, {4, 1}},
        { 6, 16,  4, {4, 2}},
        { 8, 16,  4, {4, 4}},
        { 9, 16,  2, {2, 2, 2, 2, 1}},
        {15, 16,  2, {2, 2, 2, 2, 2, 2, 2, 1}},
        {16, 16,  2, {2, 2, 2, 2, 2, 2, 2, 2}},
        {17, 17,  2, {2, 2, 2, 2, 2, 2, 2, 2, 1}},
        {19, 19,  2, {2, 2, 2, 2, 2, 2, 2, 2, 2, 1}},
        {31, 31,  2, {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1}},
        // Choose the default from all jobs; depth only limits the initial
        // budget. It must not increase when early CQEs are harvested.
        {17, 16,  2, {2, 2, 2, 2, 2, 2, 2, 2}},
        {17,  4,  2, {2, 2}},
        {32, 10,  2, {2, 2, 2, 2, 2}},
        {64, 24,  2, {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}},
    };
    for (const test_case & test : cases) {
        fake_io io(test.jobs, test.depth);
        assert(object_io_adaptive_submit_chunk(test.jobs) == test.chunk);
        assert(io.run(object_io_adaptive_submit_chunk(test.jobs)));
        assert(io.remaining_args == test.submissions);
        assert(io.submit_calls == test.submissions.size());
        const size_t drains = test.submissions.empty() ? 0 : test.submissions.size() - 1;
        const size_t tail = test.submissions.empty() ? 0 : test.submissions.back();
        assert(io.drain_calls == drains && io.completions.size() == tail);
        assert(io.h2d.size() == std::min(test.jobs, test.depth) - tail);
        io.finish();
    }
    assert(object_io_adaptive_submit_chunk(SIZE_MAX) == 2);
    // Explicit overrides bypass the default, including a legacy whole batch.
    const size_t overrides[] = {0, 2, 4, 8};
    const size_t override_calls[] = {1, 5, 3, 2};
    for (size_t i = 0; i < sizeof(overrides) / sizeof(overrides[0]); ++i) {
        fake_io io(9, 9);
        assert(io.run(overrides[i]) && io.submit_calls == override_calls[i]);
        io.finish();
    }
}

static void test_submit_policy_selection() {
    using ggml_moe_common::object_io_select_submit_chunk;
    // Missing/empty CHUNK resolves to SIZE_MAX in the production parser.
    // Missing/empty/0 ADAPTIVE resolves to false. With no override, default
    // submission must retain the original whole initial batch at every size.
    const size_t jobs[] = {0, 1, 4, 5, 8, 9, 16, 17};
    for (size_t count : jobs) {
        fake_io io(count, 16);
        const size_t chunk = object_io_select_submit_chunk(count, SIZE_MAX, false);
        assert(chunk == 0 && io.run(chunk));
        assert(io.submit_calls == (count ? 1 : 0));
        assert(io.drain_calls == 0 && io.h2d.empty());
        assert(io.submitted.size() == std::min(count, size_t(16)));
        io.finish();
    }
    // ADAPTIVE=1 selects the measured policy only without a CHUNK override.
    const size_t adaptive_jobs[] = {1, 4, 5, 8, 9, 16};
    const size_t adaptive_chunks[] = {1, 4, 4, 4, 2, 2};
    const size_t adaptive_submits[] = {1, 1, 2, 2, 5, 8};
    for (size_t i = 0; i < sizeof(adaptive_jobs) / sizeof(adaptive_jobs[0]); ++i) {
        fake_io io(adaptive_jobs[i], 16);
        const size_t chunk = object_io_select_submit_chunk(adaptive_jobs[i], SIZE_MAX, true);
        assert(chunk == adaptive_chunks[i] && io.run(chunk));
        assert(io.submit_calls == adaptive_submits[i]);
        io.finish();
    }
    // Explicit CHUNK=0 and the full supported positive range take priority
    // over either adaptive setting, including values above the job count.
    const size_t overrides[] = {0, 1, 2, 4, 8, 256};
    const size_t submissions[] = {1, 9, 5, 3, 2, 1};
    for (bool adaptive : {false, true}) {
        for (size_t i = 0; i < sizeof(overrides) / sizeof(overrides[0]); ++i) {
            fake_io io(9, 9);
            const size_t chunk = object_io_select_submit_chunk(9, overrides[i], adaptive);
            assert(chunk == overrides[i] && io.run(chunk));
            assert(io.submit_calls == submissions[i]);
            io.finish();
        }
    }
}

static void test_short_submits() {
    fake_io io(5, 5);
    io.max_submit = 2;
    assert(io.run(3));
    assert((io.remaining_args == std::vector<size_t>{3, 1, 2}));
    assert(io.drain_calls == 1 && io.h2d.size() == 3 && io.completions.size() == 2);
    io.finish();
}

static void test_sqpoll_submit_counts() {
    using ggml_moe_common::object_io_submit_chunk_result;
    using ggml_moe_common::object_io_submit_initial;
    for (const int raw : {-5, 0, 1, 4, 7}) {
        assert(object_io_submit_chunk_result(raw, 4, true) == (raw < 0 ? raw : 4));
        assert(object_io_submit_chunk_result(raw, 4, false) == raw);
    }

    for (const bool return_zero : {false, true}) {
        fake_io io(7, 7);
        assert(object_io_submit_initial(io.jobs, io.depth, 2,
            [&](size_t job) { return io.prepare(job); },
            [&](size_t remaining) {
                assert(io.submit(remaining) == static_cast<int>(remaining));
                // SQPOLL has published this chunk, but reports either all
                // chunks still awaiting its worker or an already empty SQ.
                const int raw = return_zero ? 0 : static_cast<int>(io.submitted.size());
                return object_io_submit_chunk_result(raw, remaining, true);
            },
            [&] { return io.drain(); }));
        assert(io.submit_calls == 4 && io.drain_calls == 3);
        assert((io.remaining_args == std::vector<size_t>{2, 2, 2, 1}));
        assert(io.submitted.size() == 7 && io.completions.size() == 1);
        io.finish();
    }

    fake_io failed(7, 7);
    size_t attempts = 0;
    assert(!object_io_submit_initial(failed.jobs, failed.depth, 2,
        [&](size_t job) { return failed.prepare(job); },
        [&](size_t remaining) {
            if (++attempts == 2) {
                return object_io_submit_chunk_result(-5, remaining, true);
            }
            return object_io_submit_chunk_result(failed.submit(remaining), remaining, true);
        },
        [&] { return failed.drain(); }));
    assert(attempts == 2 && failed.submit_calls == 1 && failed.drain_calls == 1);
    assert(failed.submitted.size() == 2 && failed.prepared.size() == 2);
}

static void test_failures() {
    using ggml_moe_common::object_io_submit_initial;
    size_t prepares = 0;
    size_t submits = 0;
    size_t drains = 0;
    assert(!object_io_submit_initial(8, 8, 3,
        [&](size_t job) { ++prepares; return job != 2; },
        [&](size_t remaining) { ++submits; return static_cast<int>(remaining); },
        [&] { ++drains; return true; }));
    assert(prepares == 3 && submits == 0 && drains == 0);

    // Negative, zero, and impossible submit counts must all stop immediately.
    for (const int result : {-5, 0, 4}) {
        prepares = submits = drains = 0;
        assert(!object_io_submit_initial(8, 8, 3,
            [&](size_t) { ++prepares; return true; },
            [&](size_t remaining) {
                ++submits;
                return ggml_moe_common::object_io_submit_chunk_result(result, remaining, false);
            },
            [&] { ++drains; return true; }));
        assert(prepares == 3 && submits == 1 && drains == 0);
    }

    prepares = submits = drains = 0;
    assert(!object_io_submit_initial(8, 8, 3,
        [&](size_t) { ++prepares; return true; },
        [&](size_t remaining) {
            ++submits;
            assert(remaining == (submits == 1 ? 3 : 2));
            return submits == 1 ? 1 : -5;
        },
        [&] { ++drains; return true; }));
    assert(prepares == 3 && submits == 2 && drains == 0);

    // The caller reports invalid/short CQEs or a failed enqueue through drain.
    prepares = submits = drains = 0;
    assert(!object_io_submit_initial(8, 8, 3,
        [&](size_t) { ++prepares; return true; },
        [&](size_t remaining) { ++submits; return static_cast<int>(remaining); },
        [&] { ++drains; return false; }));
    assert(prepares == 3 && submits == 1 && drains == 1);
}

int main() {
    test_early_h2d_and_payload();
    test_initial_slot_budget();
    test_empty_completions();
    test_chunk_boundaries();
    test_adaptive_chunks();
    test_submit_policy_selection();
    test_short_submits();
    test_sqpoll_submit_counts();
    test_failures();
    std::puts("common I/O initial submit: 9 CPU tests passed");
    return 0;
}
