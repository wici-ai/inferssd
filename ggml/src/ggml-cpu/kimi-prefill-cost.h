#pragma once
// The placement of a layer's routed experts, for any number of tokens: the per-expert fits, the layer's predicted time
// and the choice of the experts that move to the GPU. One evaluator for grouped prefill (kimi-cpu-sink-vram.inc,
// select_prefill_transient) and decode (kimi-cpu-sink.cpp, the decode plan). Kept apart from the sink so that it is
// tested alone (tests/test-kimi-prefill-cost.cpp).
//
// A layer's time has two forms, chosen by the physics of its reads:
//   - its reads stream through the ring (more reads than ring slots): the drive and the lanes hand the reads over
//     through the ring, kimi_prefill_time (the coupled form);
//   - every read is in flight at once (reads <= ring slots; decode always): each expert's bytes land at a known time,
//     and the lanes are a list schedule over those times, kimi_place_ring_time.
// The choice is the same for both: of the moves taken in their order, the fewest whose predicted time is within the
// form's resolution of the lowest, kimi_place_best_prefix. The coupled form does not resolve below one read (the
// drive hands the lanes their work one expert at a time); the list schedule places every read's landing, so it keeps
// no margin. On the Spark's decode traces one read kept 0.1 of 33 profitable moves a token (wf-unify/calc/
// spark_plan_replay2.py): a layer's gain is often below one read, and the token is the sum of 40 layers.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

// Sums for a least-squares line y = fixed + slope * x: samples, x, x^2, y, x*y.
struct kimi_prefill_fit {
    double n = 0, x = 0, xx = 0, t = 0, xt = 0;
    void add(double xv, double yv) { n += 1; x += xv; xx += xv * xv; t += yv; xt += xv * yv; }
    void add(const kimi_prefill_fit & o) { n += o.n; x += o.x; xx += o.xx; t += o.t; xt += o.xt; }
};

// The least-squares line through the samples, pulled by one pseudo-sample at their mean x toward the prior line scaled
// to the samples' mean: a window of one x value stays solvable and keeps the prior's shape on this machine's level, and
// against the hundreds of experts of a layer the pseudo-sample does not move the fit. Both terms stay >= 0. Without
// samples the terms are left as they are.
inline void kimi_prefill_fit_line(const kimi_prefill_fit & s, double prior_fixed, double prior_slope,
        double & fixed, double & slope) {
    if (s.n < 1.0) { return; }
    const double mean = s.x / s.n;
    const double prior_mean = prior_fixed + prior_slope * mean;
    if (prior_mean > 0.0 && s.t > 0.0) {
        const double scale = s.t / s.n / prior_mean;
        prior_fixed *= scale; prior_slope *= scale;
    }
    const double a = s.n + 1.0, b = s.x, d = s.xx + mean * mean;
    const double y0 = s.t + prior_fixed, y1 = s.xt + mean * mean * prior_slope;
    const double det = a * d - b * b;
    double f = det > 0 ? (y0 * d - b * y1) / det : -1.0, r = det > 0 ? (a * y1 - b * y0) / det : -1.0;
    if (f < 0) { f = 0; r = y1 / d; }
    else if (r < 0) { r = 0; f = y0 / a; }
    fixed = f; slope = r;
}

// The time a producer busy for a and a consumer busy for b take together when they hand the work over through a ring
// of n buffers: the throughput of a two-station tandem queue with n places between the stations (exponential
// service), as a time. It is max(a, b) when one is far shorter, rises smoothly as they approach, and is
// max * (n + 1) / n when they are equal: each station then often waits for the other, a full ring stopping the
// producer and an empty one the consumer.
inline double kimi_prefill_coupled(double a, double b, double n) {
    const double high = std::max(a, b), low = std::max(0.0, std::min(a, b));
    if (high <= 0.0) { return 0.0; }
    n = std::max(1.0, n);
    const double r = low / high;
    if (r > 1.0 - 1e-9) { return high * (n + 1.0) / n; }
    return high * (1.0 - std::pow(r, n + 1.0)) / (1.0 - std::pow(r, n));
}

// One grouped-prefill layer before any expert moves to the GPU (us).
struct kimi_prefill_layer {
    double cpu = 0, gpu = 0, link = 0; // the lanes' busy times: the CPU, the GPU worker, the link
    double drive = 0, fill = 0;        // the drive's time for the layer's reads, and for the first of them
    double ring = 1;                   // the reads the ring holds between the drive and the lanes
    double gpu_overhead = 0;           // the GPU work's time outside its experts, paid once the layer has GPU work
    bool gpu_work = false;             // resident experts give the layer GPU work before any move
};
// What moving one expert to the GPU takes off the CPU lane and adds to the GPU lane and the link (us).
struct kimi_prefill_move { double cpu, gpu, link; };

// The predicted layer time: the first read, the drive coupled with the slower lane, and the GPU overhead.
inline double kimi_prefill_time(const kimi_prefill_layer & l, double cpu, double gpu, double link, bool gpu_work) {
    return l.fill + kimi_prefill_coupled(l.drive, std::max(std::max(cpu, gpu), link), l.ring) +
        (gpu_work ? l.gpu_overhead : 0.0);
}

// How many of the moves, taken in their order, to make: the fewest whose predicted time is within `fill` of the lowest
// (the time model's resolution). time(k) is the predicted layer time with the first k moves made, for k = 0..count.
// *chosen receives the chosen time.
// Each time(k) is evaluated once.
template <class Time>
inline size_t kimi_place_best_prefix(size_t count, double fill, Time && time, double * chosen = nullptr) {
    std::vector<double> t(count + 1);
    for (size_t k = 0; k <= count; ++k) { t[k] = time(k); }
    const double lowest = *std::min_element(t.begin(), t.end());
    size_t k = 0;
    while (t[k] > lowest + fill && k < count) { ++k; }
    if (chosen) { *chosen = t[k]; }
    return k;
}

// The streamed form's choice: the moves' lane terms accumulate in their order. Its resolution is one read (l.fill): a
// gain shorter than one expert's read does not shorten the layer; it would only move work to the GPU for nothing.
inline size_t kimi_prefill_best_prefix(const kimi_prefill_layer & l, const kimi_prefill_move * moves, size_t count,
        double * time = nullptr) {
    std::vector<double> cpu(count + 1), gpu(count + 1), link(count + 1);
    cpu[0] = l.cpu; gpu[0] = l.gpu; link[0] = l.link;
    for (size_t i = 0; i < count; ++i) {
        cpu[i + 1] = cpu[i] - moves[i].cpu; gpu[i + 1] = gpu[i] + moves[i].gpu; link[i + 1] = link[i] + moves[i].link;
    }
    return kimi_place_best_prefix(count, l.fill, [&](size_t k) {
        return kimi_prefill_time(l, cpu[k], gpu[k], link[k], k ? true : l.gpu_work);
    }, time);
}

// One expert of a layer whose reads are all in flight at once (us from the layer's start).
struct kimi_place_expert {
    double cpu = 0;          // its CPU time
    double early = 0;        // the part of cpu the CPU may start at early_ready (decode's up/gate; 0: none)
    double early_ready = 0;  // when the bytes of that part are in host memory
    double ready = 0;        // when all its bytes are in host memory (0: host-ready)
    double gpu = 0;          // its share of a GPU call
    double link = 0;         // its copy into device memory before that call (0: none, or the GPU reads it in place)
    bool resident = false;   // in the tier: in the residents' call; never moves
};
// The rest of such a layer (us).
struct kimi_place_lanes {
    double resident_call = 0; // a GPU call's own time, for the residents' call
    double host_call = 0;     // the same for a call of moved experts
    double gpu_overhead = 0;  // once when the layer has GPU work (grouped prefill's zeroing and output addition)
    double reads_end = 0;     // the last read's landing (0: no reads)
    double link_free = 0;     // when the link can take this layer's first copy
    double cpu_moved = 1;     // the CPU's time per expert, relative, in a layer with moved experts: their copies (or
                              // the GPU's reads in place) share the DRAM with the CPU (decode measures it; 1: not)
};

// The GPU lane's end of such a layer (and the link's, which feeds it), see kimi_place_ring_time.
inline double kimi_place_gpu_end(const kimi_place_lanes & l, const kimi_place_expert * e, size_t n, const uint8_t * on_gpu) {
    double gpu = 0.0, link = l.link_free;
    bool gpu_work = false;
    double resident = 0.0;
    size_t residents = 0;
    for (size_t i = 0; i < n; ++i) { if (e[i].resident) { resident += e[i].gpu; ++residents; } }
    if (residents) { gpu = l.resident_call + resident; gpu_work = true; }
    std::vector<size_t> moved;
    for (size_t i = 0; i < n; ++i) { if (!e[i].resident && on_gpu && on_gpu[i]) { moved.push_back(i); } }
    std::stable_sort(moved.begin(), moved.end(), [&](size_t a, size_t b) { return e[a].ready < e[b].ready; });
    std::vector<double> copied(moved.size());
    for (size_t j = 0; j < moved.size(); ++j) {
        const auto & x = e[moved[j]];
        link = x.link > 0.0 ? std::max(link, x.ready) + x.link : link;
        copied[j] = x.link > 0.0 ? link : x.ready;
    }
    for (size_t j = 0; j < moved.size();) {
        const double start = std::max(gpu, e[moved[j]].ready);
        double call = l.host_call, done = 0.0;
        size_t k = j;
        while (k < moved.size() && e[moved[k]].ready <= start) { call += e[moved[k]].gpu; done = std::max(done, copied[k]); ++k; }
        gpu = std::max(start, done) + call;
        gpu_work = true;
        j = k;
    }
    if (gpu_work) { gpu += l.gpu_overhead; }
    return gpu;
}

// The layer's time when its reads are all in flight: a list schedule of the lanes over the experts' landing times.
//   CPU: one processor; it takes a ready early part first, else a ready rest (an expert's rest after its early part),
//        lowest index first, as the decode dispatcher does (up/gate work wins).
//   link: the moved experts' copies, one at a time, each once its bytes are in host memory and the link is free.
//   GPU: the residents' call from the start, then the moved experts in calls: a call takes the moved experts whose
//        bytes are in host memory when the GPU is free (the tier's worker takes every launched job), and starts once
//        their copies are done.
//   T = max(CPU end, GPU end, last landing). With moved experts the CPU's times are scaled by cpu_moved.
// on_gpu[i] nonzero moves expert i (ignored for residents). cpu_end and gpu_end, when given, receive the lanes' ends.
inline double kimi_place_ring_time(const kimi_place_lanes & l, const kimi_place_expert * e, size_t n,
        const uint8_t * on_gpu, double * cpu_end = nullptr, double * gpu_end = nullptr) {
    const double gpu = kimi_place_gpu_end(l, e, n, on_gpu);
    bool moved = false;
    for (size_t i = 0; i < n && !moved; ++i) { moved = !e[i].resident && on_gpu && on_gpu[i]; }
    const double scale = moved ? l.cpu_moved : 1.0;
    // CPU: the list schedule's picks in its order, O(m log m) for m CPU experts. An early part is released at its
    // early_ready, a whole expert without one at its ready, a rest once its early part is done and its ready has come;
    // at each pick the CPU takes the released early part of the lowest index, else the released rest of the lowest
    // index, else waits for the next release.
    struct release { double at; size_t j; };
    const auto later = [](const release & a, const release & b) { return a.at > b.at || (a.at == b.at && a.j > b.j); };
    std::vector<release> early_wait, rest_wait;         // min-heaps by release time
    std::vector<size_t> early_ready, rest_ready;        // min-heaps by index
    const auto higher = [](size_t a, size_t b) { return a > b; };
    size_t left = 0;
    for (size_t i = 0; i < n; ++i) {
        const auto & x = e[i];
        if (x.resident || (on_gpu && on_gpu[i]) || !(x.cpu > 0.0)) { continue; }
        ++left;
        if (x.early > 0.0) { early_wait.push_back({ x.early_ready, i }); } else { rest_wait.push_back({ x.ready, i }); }
    }
    std::make_heap(early_wait.begin(), early_wait.end(), later);
    std::make_heap(rest_wait.begin(), rest_wait.end(), later);
    double cpu = 0.0;
    auto take = [&](std::vector<release> & wait, std::vector<size_t> & ready) {
        while (!wait.empty() && wait.front().at <= cpu) {
            ready.push_back(wait.front().j); std::push_heap(ready.begin(), ready.end(), higher);
            std::pop_heap(wait.begin(), wait.end(), later); wait.pop_back();
        }
    };
    while (left) {
        take(early_wait, early_ready);
        take(rest_wait, rest_ready);
        if (!early_ready.empty()) {
            std::pop_heap(early_ready.begin(), early_ready.end(), higher);
            const size_t j = early_ready.back(); early_ready.pop_back();
            cpu += std::min(e[j].early, e[j].cpu) * scale;
            rest_wait.push_back({ e[j].ready, j }); std::push_heap(rest_wait.begin(), rest_wait.end(), later);
            continue;
        }
        if (!rest_ready.empty()) {
            std::pop_heap(rest_ready.begin(), rest_ready.end(), higher);
            const size_t j = rest_ready.back(); rest_ready.pop_back();
            cpu += (e[j].early > 0.0 ? std::max(0.0, e[j].cpu - e[j].early) : e[j].cpu) * scale;
            --left;
            continue;
        }
        double next = INFINITY;
        if (!early_wait.empty()) { next = std::min(next, early_wait.front().at); }
        if (!rest_wait.empty()) { next = std::min(next, rest_wait.front().at); }
        cpu = next;
    }
    if (cpu_end) { *cpu_end = cpu; }
    if (gpu_end) { *gpu_end = gpu; }
    return std::max(std::max(cpu, gpu), l.reads_end);
}

// The in-ring choice: the candidates move in their order (indices into e), the first k of them for time(k); the fewest
// with the lowest predicted time (no margin: the schedule places every landing). A layer without early parts (grouped
// prefill) has the CPU's end of any order that never idles while work is released: its busy periods follow the
// work's arrivals. So its CPU lane is a fold over the experts by landing, sorted once for every prefix: O(n) a prefix
// instead of the list schedule's O(n log n), for layers of up to 256 experts.
inline size_t kimi_place_ring_best_prefix(const kimi_place_lanes & l, const kimi_place_expert * e, size_t n,
        const size_t * candidates, size_t count, double * time = nullptr) {
    std::vector<uint8_t> on_gpu(n, 0);
    size_t set = 0;
    bool early = false;
    std::vector<size_t> order;
    for (size_t i = 0; i < n; ++i) {
        if (e[i].resident || !(e[i].cpu > 0.0)) { continue; }
        early = early || e[i].early > 0.0;
        order.push_back(i);
    }
    if (!early) { std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return e[a].ready < e[b].ready; }); }
    return kimi_place_best_prefix(count, 0.0, [&](size_t k) {
        if (k < set) { std::fill(on_gpu.begin(), on_gpu.end(), 0); set = 0; }
        while (set < k) { on_gpu[candidates[set++]] = 1; }
        if (early) { return kimi_place_ring_time(l, e, n, on_gpu.data()); }
        double cpu = 0.0;
        const double scale = k ? l.cpu_moved : 1.0;
        for (size_t i : order) { if (!on_gpu[i]) { cpu = std::max(cpu, e[i].ready) + e[i].cpu * scale; } }
        return std::max(std::max(cpu, kimi_place_gpu_end(l, e, n, on_gpu.data())), l.reads_end);
    }, time);
}
