// The MXFP4 grouped-prefill cost model alone (ggml/src/ggml-cpu/kimi-prefill-cost.h): the per-expert fit, the coupling
// of the drive with the lanes, and the choice of the experts that move to the GPU.
#include "kimi-prefill-cost.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int failures = 0;
static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}
static bool near(double a, double b, double rel) { return std::fabs(a - b) <= rel * std::max(std::fabs(a), std::fabs(b)); }

static void test_fit() {
    // us = 300 + 250 * routes with +-100 us deterministic noise, routes 1..400: the fit recovers both terms.
    kimi_prefill_fit s;
    for (int routes = 1; routes <= 400; ++routes) { s.add(routes, 300.0 + 250.0 * routes + 100.0 * std::sin(routes * 12.9898)); }
    double fixed = -1, slope = -1;
    kimi_prefill_fit_line(s, 204.0, 87.0, fixed, slope);
    check(near(slope, 250.0, 0.002) && std::fabs(fixed - 300.0) < 30.0, "fit recovers fixed and per-route terms");
    std::printf("fit: fixed=%.1f slope=%.2f (300, 250)\n", fixed, slope);

    // One route count only (a window of 1-token passes): solvable, the line passes through the data there, and keeps
    // the prior's shape: both terms scaled by the data over the prior there.
    kimi_prefill_fit one;
    for (int i = 0; i < 240; ++i) { one.add(1.0, 1000.0 + (i % 7)); }
    kimi_prefill_fit_line(one, 204.0, 87.0, fixed, slope);
    const double scale = (one.t / one.n) / (204.0 + 87.0);
    check(std::isfinite(fixed) && std::isfinite(slope) && fixed >= 0 && slope >= 0, "one route count stays solvable");
    check(near(fixed + slope, one.t / one.n, 1e-3), "one route count: the line passes through the data");
    check(near(slope, 87.0 * scale, 1e-3) && near(fixed, 204.0 * scale, 1e-3), "one route count: the prior's shape");
    std::printf("fit one route count: fixed=%.1f slope=%.2f\n", fixed, slope);

    // The drive's line per layer, read time against MB: 40 warm-up layers of ~190 MB and big layers of ~4.6 GB, a 12 ms
    // queue fill and drain per layer at 7.2 GB/s. The ratio of the sums would give 6.7 GB/s; the line keeps the rate.
    kimi_prefill_fit drive;
    for (int i = 0; i < 40; ++i) { const double mb = 150.0 + 2.0 * i; drive.add(mb, 12000.0 + mb * 1e3 / 7.2); }
    for (int i = 0; i < 10; ++i) { const double mb = 4400.0 + 40.0 * i; drive.add(mb, 12000.0 + mb * 1e3 / 7.2); }
    kimi_prefill_fit_line(drive, 0.0, 1.0, fixed, slope);
    check(near(1e3 / slope, 7.2, 5e-3) && near(fixed, 12000.0, 0.05), "the drive's line separates its fixed part");
    std::printf("fit drive: fixed=%.0f us rate=%.3f GB/s (12000, 7.2)\n", fixed, 1e3 / slope);

    // A negative intercept is clamped: the line through the origin.
    kimi_prefill_fit neg;
    for (int routes = 100; routes <= 200; ++routes) { neg.add(routes, 10.0 * routes - 500.0); }
    kimi_prefill_fit_line(neg, 0.0, 10.0, fixed, slope);
    check(fixed == 0.0 && slope > 0.0 && slope < 10.0, "negative intercept clamps to a line through the origin");

    // No samples: the terms are left as they are.
    fixed = 7; slope = 9;
    kimi_prefill_fit_line(kimi_prefill_fit(), 204.0, 87.0, fixed, slope);
    check(fixed == 7 && slope == 9, "no samples keep the terms");
}

static void test_coupled() {
    check(kimi_prefill_coupled(1000.0, 0.0, 8) == 1000.0, "coupled with an idle side is the busy side");
    check(near(kimi_prefill_coupled(1000.0, 1000.0, 8), 1125.0, 1e-12), "equal sides: max * (n + 1) / n");
    check(near(kimi_prefill_coupled(1000.0, 1000.0 * (1 - 1e-10), 8), 1125.0, 1e-6), "continuous at equal sides");
    check(kimi_prefill_coupled(1000.0, 300.0, 8) == kimi_prefill_coupled(300.0, 1000.0, 8), "symmetric");
    check(near(kimi_prefill_coupled(1000.0, 100.0, 8), 1000.0, 1e-7), "a far shorter side costs nothing");
    double previous = 0;
    bool monotone = true;
    for (int i = 0; i <= 300; ++i) {
        const double t = kimi_prefill_coupled(1000.0, 10.0 * i, 8);
        monotone &= t >= previous && t >= std::max(1000.0, 10.0 * i);
        previous = t;
    }
    check(monotone, "coupled rises with the shorter side and stays above the max");
    // The measured share on 233 (8-read ring): 0.136 with the lanes at the drive time, 0.002 with them at half of it.
    std::printf("coupled n=8: equal %.4f, half %.4f, 0.95 %.4f\n", kimi_prefill_coupled(1, 1, 8) - 1,
        kimi_prefill_coupled(1, 0.5, 8) - 1, kimi_prefill_coupled(1, 0.95, 8) - 1);
}

// A layer of 330 cold experts like DeepSeek's at 4104 tokens on 233: routes 1..~500, CPU 520 + 297 us per route,
// GPU 60 + 25 us per route plus an 18.8 MB upload at 5.4 GB/s, an 8-read ring.
static std::vector<kimi_prefill_move> moves_for(kimi_prefill_layer & layer, int experts) {
    std::vector<int> routes;
    for (int i = 0; i < experts; ++i) { routes.push_back(1 + int(500.0 * std::pow(double(experts - i) / experts, 3.0))); }
    std::vector<kimi_prefill_move> moves;
    layer = {};
    for (int r : routes) {
        const double cpu = 520.0 + 297.0 * r;
        layer.cpu += cpu;
        moves.push_back({ cpu, 60.0 + 25.0 * r + 18.8e6 / 5400.0, (18.8e6 + 4400.0 * r) / 5400.0 });
    }
    layer.ring = 8;
    layer.gpu_overhead = 4104 * 7.0;
    layer.fill = 18.8e6 / 6800.0;
    return moves;
}

static void test_select() {
    kimi_prefill_layer layer;
    auto moves = moves_for(layer, 330);
    const double cpu_all = layer.cpu;

    // CPU-bound (a fast drive): the lanes end balanced, within one move of each other.
    layer.drive = 0.2 * cpu_all;
    double t = 0;
    size_t k = kimi_prefill_best_prefix(layer, moves.data(), moves.size(), &t);
    double cpu = layer.cpu, gpu = 0;
    for (size_t i = 0; i < k; ++i) { cpu -= moves[i].cpu; gpu += moves[i].gpu; }
    check(k > 0 && k < moves.size() && std::fabs(cpu - gpu) < 2 * moves[k - 1].gpu + moves[k - 1].cpu + 2 * layer.fill,
        "a CPU-bound layer balances the lanes");
    std::printf("select cpu-bound: %zu of %zu, cpu %.0f gpu %.0f us\n", k, moves.size(), cpu, gpu);

    // A drive far longer than the CPU lane (a short batch): nothing moves, also when resident experts have already
    // paid the GPU overhead, where the lanes still gain a little: less than one read. A two-read ring couples more
    // (a share of r^2 instead of r^8), and nothing moves there once the drive is a thousand times the lanes.
    layer.drive = 20.0 * cpu_all;
    k = kimi_prefill_best_prefix(layer, moves.data(), moves.size(), &t);
    check(k == 0, "a drive-bound layer keeps its experts on the CPU");
    kimi_prefill_layer resident = layer;
    resident.gpu_work = true;
    check(kimi_prefill_best_prefix(resident, moves.data(), moves.size()) == 0,
        "a drive-bound layer with resident GPU work keeps its experts on the CPU");
    resident.ring = 2; resident.drive = 1000.0 * cpu_all; resident.fill = resident.drive / moves.size();
    check(kimi_prefill_best_prefix(resident, moves.data(), moves.size()) == 0,
        "a drive-bound layer with a two-read ring keeps its experts on the CPU");

    // No discontinuity: as the drive grows through the CPU lane's time, the number moved and the predicted time
    // change gradually.
    size_t previous_k = moves.size();
    double previous_t = 0;
    bool gradual = true;
    for (int i = 0; i <= 400; ++i) {
        layer.drive = cpu_all * (0.2 + 1.6 * i / 400.0);
        k = kimi_prefill_best_prefix(layer, moves.data(), moves.size(), &t);
        if (i) {
            gradual &= t >= previous_t - 1e-6 && t - previous_t <= 0.01 * cpu_all;
            gradual &= previous_k >= k ? previous_k - k <= moves.size() / 20 : k - previous_k <= moves.size() / 20;
        }
        previous_k = k; previous_t = t;
    }
    check(gradual, "the selection changes gradually with the drive time");
    layer.drive = 1.0001 * cpu_all;
    const size_t above = kimi_prefill_best_prefix(layer, moves.data(), moves.size());
    layer.drive = 0.9999 * cpu_all;
    const size_t below = kimi_prefill_best_prefix(layer, moves.data(), moves.size());
    check(above > 0 && below > 0 && (above > below ? above - below : below - above) <= 2,
        "the same choice just above and just below the CPU lane's time");
    std::printf("select at the drive: %zu (drive just above the CPU lane) %zu (just below)\n", above, below);

    // The GPU overhead decides a layer with little to gain.
    kimi_prefill_layer small = {};
    small.cpu = 6 * 1000.0; small.drive = 6 * 2800.0; small.fill = 2800.0; small.ring = 8; small.gpu_overhead = 7.0;
    std::vector<kimi_prefill_move> few(6, kimi_prefill_move{ 1000.0, 3600.0, 3500.0 });
    check(kimi_prefill_best_prefix(small, few.data(), few.size()) == 0, "a 1-token layer stays on the CPU");
}

// The streamed choice before it went through kimi_place_best_prefix (dev 8251825), as the reference of test_same_choice.
static size_t reference_best_prefix(const kimi_prefill_layer & l, const kimi_prefill_move * moves, size_t count,
        double * time) {
    double cpu = l.cpu, gpu = l.gpu, link = l.link;
    double lowest = kimi_prefill_time(l, cpu, gpu, link, l.gpu_work);
    for (size_t i = 0; i < count; ++i) {
        cpu -= moves[i].cpu; gpu += moves[i].gpu; link += moves[i].link;
        lowest = std::min(lowest, kimi_prefill_time(l, cpu, gpu, link, true));
    }
    cpu = l.cpu; gpu = l.gpu; link = l.link;
    double t = kimi_prefill_time(l, cpu, gpu, link, l.gpu_work);
    size_t chosen = 0;
    while (t > lowest + l.fill && chosen < count) {
        cpu -= moves[chosen].cpu; gpu += moves[chosen].gpu; link += moves[chosen].link;
        t = kimi_prefill_time(l, cpu, gpu, link, true);
        ++chosen;
    }
    *time = t;
    return chosen;
}

// A streamed layer chooses exactly what M8 chose: the same count and the same time, bit for bit, on random layers.
static void test_same_choice() {
    uint64_t state = 12345;
    auto unit = [&] { state = state * 6364136223846793005ull + 1442695040888963407ull; return double(state >> 11) / double(1ull << 53); };
    size_t same = 0, total = 0, moved = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        kimi_prefill_layer layer;
        auto moves = moves_for(layer, 1 + int(unit() * 400));
        layer.drive = layer.cpu * (0.05 + 2.0 * unit());
        layer.gpu = unit() < 0.5 ? 0.0 : layer.cpu * 0.3 * unit();
        layer.gpu_work = layer.gpu > 0.0;
        layer.ring = 1 + int(unit() * 16);
        layer.fill = unit() < 0.2 ? 0.0 : layer.fill * unit() * 2.0;
        double t_old = 0, t_new = 0;
        const size_t k_old = reference_best_prefix(layer, moves.data(), moves.size(), &t_old);
        const size_t k_new = kimi_prefill_best_prefix(layer, moves.data(), moves.size(), &t_new);
        same += k_old == k_new && t_old == t_new;
        moved += k_new > 0;
        ++total;
    }
    check(same == total, "the streamed choice is M8's, bit for bit");
    std::printf("same choice: %zu of %zu random streamed layers (%zu move experts)\n", same, total, moved);
}

// The in-ring list schedule and its choice, each value derived from kimi_place_ring_time's rules (us).
static void test_ring() {
    // 125's decode terms: a call 150 + 20 an expert from device memory, a page-locked copy 355, the CPU 470 an expert
    // (320 of it up/gate). Three residents, two RAM hits, no read.
    //   no move:  GPU 150 + 3 * 20 = 210; CPU 470 + 470 = 940;              T = 940
    //   one:      copy 0..355; host call max(210, 355) + 150 + 20 = 525; CPU 470; T = 525
    //   two:      copies 0..355..710; call max(210, 710) + 150 + 40 = 900; CPU 0;  T = 900
    std::vector<kimi_place_expert> e(5);
    for (int i = 0; i < 3; ++i) { e[i].resident = true; e[i].gpu = 20; }
    for (int i = 3; i < 5; ++i) { e[i].cpu = 470; e[i].early = 320; e[i].gpu = 20; e[i].link = 355; }
    kimi_place_lanes lanes = {};
    lanes.resident_call = 150; lanes.host_call = 150;
    uint8_t none[5] = {}, one[5] = { 0, 0, 0, 1, 0 }, two[5] = { 0, 0, 0, 1, 1 };
    check(kimi_place_ring_time(lanes, e.data(), 5, none) == 940, "125: no move 940");
    check(kimi_place_ring_time(lanes, e.data(), 5, one) == 525, "125: one move 525");
    check(kimi_place_ring_time(lanes, e.data(), 5, two) == 900, "125: two moves 900");
    const size_t candidates[2] = { 3, 4 };
    double t = 0;
    check(kimi_place_ring_best_prefix(lanes, e.data(), 5, candidates, 2, &t) == 1 && t == 525, "125: one moves");
    // The link busy with an admission until 200: the copy starts there, 200 + 355 = 555, the call ends at 725.
    lanes.link_free = 200;
    check(kimi_place_ring_time(lanes, e.data(), 5, one) == 725, "125: a busy link delays the copy");
    lanes.link_free = 0;
    // The CPU 1.5x slower while copies share the DRAM: one move, CPU 705 > the GPU's 525: T = 705, still below 940.
    // At 2.5x one move leaves the CPU 1175, above 940; two leave it nothing (900): both move. Without a move the CPU
    // keeps its own time.
    lanes.cpu_moved = 1.5;
    check(kimi_place_ring_time(lanes, e.data(), 5, one) == 705 && kimi_place_ring_time(lanes, e.data(), 5, none) == 940,
        "125: the CPU slower beside a copy");
    lanes.cpu_moved = 2.5;
    check(kimi_place_ring_best_prefix(lanes, e.data(), 5, candidates, 2, &t) == 2 && t == 900,
        "125: a CPU much slower beside copies: both move");
    lanes.cpu_moved = 1;

    // 233's terms: the copy 2700 (6.9 GB/s), the CPU 900. One move: call max(210, 2700) + 170 = 2870 > 1800.
    for (int i = 3; i < 5; ++i) { e[i].cpu = 900; e[i].early = 600; e[i].link = 2700; }
    check(kimi_place_ring_time(lanes, e.data(), 5, none) == 1800 && kimi_place_ring_time(lanes, e.data(), 5, one) == 2870,
        "233: 1800 and 2870");
    check(kimi_place_ring_best_prefix(lanes, e.data(), 5, candidates, 2, &t) == 0 && t == 1800, "233: nothing moves");

    // Unified memory: in place (no copy), 370 an expert in the host call; five residents at 190; the CPU 1200.
    //   no move: GPU 150 + 950 = 1100; CPU 2400 -> 2400. one: GPU 1100 + 150 + 370 = 1620; CPU 1200 -> 1620.
    //   two: GPU 1100 + 150 + 740 = 1990; CPU 0 -> 1990.
    std::vector<kimi_place_expert> u(7);
    for (int i = 0; i < 5; ++i) { u[i].resident = true; u[i].gpu = 190; }
    for (int i = 5; i < 7; ++i) { u[i].cpu = 1200; u[i].early = 800; u[i].gpu = 370; }
    const size_t uc[2] = { 5, 6 };
    uint8_t u1[7] = { 0, 0, 0, 0, 0, 1, 0 }, u2[7] = { 0, 0, 0, 0, 0, 1, 1 };
    check(kimi_place_ring_time(lanes, u.data(), 7, u1) == 1620 && kimi_place_ring_time(lanes, u.data(), 7, u2) == 1990,
        "unified: 1620 and 1990");
    check(kimi_place_ring_best_prefix(lanes, u.data(), 7, uc, 2, &t) == 1 && t == 1620, "unified: one moves");

    // Misses with the early up/gate: one RAM hit and two misses, gates landing at 1000 and 1500, downs at 2000 and
    // 2500 (role-major), residents as above (210). CPU: hit 0..320..470; miss 1 up/gate 1000..1320; miss 2 up/gate
    // 1500..1820; miss 1 down 2000..2150; miss 2 down 2500..2650. T = 2650. Moving the hit leaves the misses' times:
    // T = 2650: no gain, and a tie keeps the fewest moves.
    std::vector<kimi_place_expert> m(6);
    for (int i = 0; i < 3; ++i) { m[i].resident = true; m[i].gpu = 20; }
    m[3].cpu = 470; m[3].early = 320; m[3].gpu = 20; m[3].link = 355;
    m[4].cpu = 470; m[4].early = 320; m[4].early_ready = 1000; m[4].ready = 2000; m[4].gpu = 20; m[4].link = 355;
    m[5].cpu = 470; m[5].early = 320; m[5].early_ready = 1500; m[5].ready = 2500; m[5].gpu = 20; m[5].link = 355;
    lanes.reads_end = 2500;
    uint8_t m0[6] = {}, m1[6] = { 0, 0, 0, 1, 0, 0 };
    double cpu_end = 0, gpu_end = 0;
    check(kimi_place_ring_time(lanes, m.data(), 6, m0, &cpu_end, &gpu_end) == 2650 && cpu_end == 2650 && gpu_end == 210,
        "misses: 2650, the CPU's end");
    check(kimi_place_ring_time(lanes, m.data(), 6, m1) == 2650, "misses: moving the hit gains nothing");
    const size_t mc[3] = { 3, 4, 5 };
    check(kimi_place_ring_best_prefix(lanes, m.data(), 6, mc, 3, &t) == 0 && t == 2650, "misses: nothing moves");
    // A miss moved: its copy starts once it has landed (2000), 2355; the call ends at 2525. CPU: hit 470, miss 2
    // 1500..1820, 2500..2650. Still 2650.
    uint8_t m2[6] = { 0, 0, 0, 0, 1, 0 };
    check(kimi_place_ring_time(lanes, m.data(), 6, m2, &cpu_end, &gpu_end) == 2650 && gpu_end == 2525, "misses: a moved miss");

    // Grouped prefill, 2 tokens: six misses of 18.8 MB at 12.3 GB/s (1528 us each, 9170 in all), the CPU 380 an expert,
    // a transient 2800 (its upload in the worker) + 33. Every read lands before the CPU is busy: T = 9170 + 380 with no
    // move; a move adds the GPU's 2833 after its landing. Nothing moves.
    std::vector<kimi_place_expert> p(6);
    for (int i = 0; i < 6; ++i) { p[i].cpu = 380; p[i].ready = 1528.0 * (i + 1); p[i].gpu = 2833; }
    kimi_place_lanes pl = {};
    pl.reads_end = 9170;
    const size_t pc[6] = { 0, 1, 2, 3, 4, 5 };
    check(kimi_place_ring_best_prefix(pl, p.data(), 6, pc, 6, &t) == 0 && t == 1528.0 * 6 + 380, "2-token prefill: nothing moves");
    std::printf("ring: 125 one move 525 of 940; 233 none; unified one; misses none; 2-token prefill none\n");
}

// kimi_place_ring_time's CPU lane as first written (18bc4ec: a scan of every CPU expert at each pick), as the
// reference of test_ring_same: the heap schedule makes the same picks in the same order, so its end is the same, bit
// for bit.
static double reference_cpu_end(const kimi_place_expert * e, size_t n, const uint8_t * on_gpu) {
    std::vector<size_t> cpu_experts;
    for (size_t i = 0; i < n; ++i) { if (!e[i].resident && !(on_gpu && on_gpu[i]) && e[i].cpu > 0.0) { cpu_experts.push_back(i); } }
    std::vector<uint8_t> stage(cpu_experts.size(), 0);
    double cpu = 0.0;
    for (size_t left = cpu_experts.size(); left;) {
        size_t pick = cpu_experts.size();
        for (size_t j = 0; j < cpu_experts.size() && pick == cpu_experts.size(); ++j) {
            const auto & x = e[cpu_experts[j]];
            if (stage[j] == 0 && x.early > 0.0 && x.early_ready <= cpu) { pick = j; }
        }
        for (size_t j = 0; j < cpu_experts.size() && pick == cpu_experts.size(); ++j) {
            const auto & x = e[cpu_experts[j]];
            if ((stage[j] == 1 || (stage[j] == 0 && !(x.early > 0.0))) && x.ready <= cpu) { pick = j; }
        }
        if (pick == cpu_experts.size()) {
            double next = INFINITY;
            for (size_t j = 0; j < cpu_experts.size(); ++j) {
                const auto & x = e[cpu_experts[j]];
                if (stage[j] == 0 && x.early > 0.0) { next = std::min(next, x.early_ready); }
                else if (stage[j] < 2) { next = std::min(next, x.ready); }
            }
            cpu = next;
            continue;
        }
        const auto & x = e[cpu_experts[pick]];
        if (stage[pick] == 0 && x.early > 0.0) { cpu += std::min(x.early, x.cpu); stage[pick] = 1; continue; }
        cpu += stage[pick] == 1 ? std::max(0.0, x.cpu - x.early) : x.cpu;
        stage[pick] = 2;
        --left;
    }
    return cpu;
}

// Random layers, decode-like (early parts, misses landing in issue order) and prefill-like (no early part, up to 256
// experts): the CPU's end is the reference's, bit for bit, with and without moves.
static void test_ring_same() {
    uint64_t state = 777;
    auto unit = [&] { state = state * 6364136223846793005ull + 1442695040888963407ull; return double(state >> 11) / double(1ull << 53); };
    size_t same = 0, total = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const bool decode = trial % 2 == 0;
        const size_t n = decode ? 1 + size_t(unit() * 10) : 1 + size_t(unit() * 256);
        std::vector<kimi_place_expert> e(n);
        std::vector<uint8_t> on(n, 0);
        double landed = 0.0;
        for (size_t i = 0; i < n; ++i) {
            auto & x = e[i];
            if (unit() < 0.3) { x.resident = true; x.gpu = 20; continue; }
            x.cpu = 300 + 400 * unit();
            if (decode && unit() < 0.8) { x.early = x.cpu * (0.5 + 0.3 * unit()); }
            if (unit() < 0.5) {
                landed += 500 + 3000 * unit();
                x.ready = landed; x.early_ready = x.early > 0.0 ? landed * (0.4 + 0.5 * unit()) : 0.0;
            }
            // equal landing times, as several prefill misses of one read
            if (!decode && i && unit() < 0.2) { x.ready = e[i - 1].ready; }
            on[i] = unit() < 0.2;
        }
        double end = 0, end_moved = 0;
        kimi_place_ring_time(kimi_place_lanes{}, e.data(), n, nullptr, &end);
        kimi_place_ring_time(kimi_place_lanes{}, e.data(), n, on.data(), &end_moved);
        same += end == reference_cpu_end(e.data(), n, nullptr) && end_moved == reference_cpu_end(e.data(), n, on.data());
        ++total;
    }
    check(same == total, "the heap schedule's CPU end is the scan's, bit for bit");
    std::printf("ring same: %zu of %zu random layers\n", same, total);
}

// Layers without early parts (grouped prefill, up to 256 experts, few reads): kimi_place_ring_best_prefix folds the CPU
// lane by landing; the same choice as the list schedule (kimi_place_ring_time) for every prefix, and the same time
// within rounding (a different order of the same sums).
static void test_ring_prefix_fold() {
    uint64_t state = 4242;
    auto unit = [&] { state = state * 6364136223846793005ull + 1442695040888963407ull; return double(state >> 11) / double(1ull << 53); };
    size_t same = 0, total = 0, moved = 0;
    for (int trial = 0; trial < 1000; ++trial) {
        const size_t n = 2 + size_t(unit() * 255);
        std::vector<kimi_place_expert> e(n);
        std::vector<size_t> c;
        double landed = 0.0;
        for (size_t i = 0; i < n; ++i) {
            auto & x = e[i];
            if (unit() < 0.3) { x.resident = true; x.gpu = 20 + 200 * unit(); continue; }
            x.cpu = 200 + 2000 * unit(); x.gpu = 30 + 400 * unit();
            if (unit() < 0.05) { landed += 1500 * unit(); x.ready = landed; x.gpu += 2800; }
            if (unit() < 0.8) { c.push_back(i); }
        }
        kimi_place_lanes l = {};
        l.resident_call = 25; l.host_call = 25; l.gpu_overhead = 100 * unit(); l.reads_end = landed;
        std::vector<uint8_t> on(n, 0);
        const size_t k_ref = kimi_place_best_prefix(c.size(), 0.0, [&](size_t k) {
            std::fill(on.begin(), on.end(), 0);
            for (size_t j = 0; j < k; ++j) { on[c[j]] = 1; }
            return kimi_place_ring_time(l, e.data(), n, on.data());
        });
        double t_ref = 0, t = 0;
        std::fill(on.begin(), on.end(), 0);
        for (size_t j = 0; j < k_ref; ++j) { on[c[j]] = 1; }
        t_ref = kimi_place_ring_time(l, e.data(), n, on.data());
        const size_t k = kimi_place_ring_best_prefix(l, e.data(), n, c.data(), c.size(), &t);
        same += k == k_ref && near(t, t_ref, 1e-12);
        moved += k > 0;
        ++total;
    }
    check(same == total, "the folded CPU lane chooses as the list schedule");
    std::printf("ring fold: %zu of %zu random prefill layers (%zu move experts)\n", same, total, moved);
}

int main() {
    test_fit();
    test_coupled();
    test_select();
    test_same_choice();
    test_ring();
    test_ring_same();
    test_ring_prefix_fold();
    if (failures) { std::fprintf(stderr, "%d failures\n", failures); return 1; }
    std::printf("KIMI_PREFILL_COST_OK\n");
    return 0;
}
