// SPDX-License-Identifier: Apache-2.0
// A decode step of several rows against the GPU tier (ssd_vram_format_api prepare_rows, launch_rows), with a stand-in
// provider the test injects: its tier holds fixed experts of each layer, admits none, and gives each route to one a row
// computed from the expert, the route's input row and its weight alone, as an engine call a row does. The step as one
// run takes one prepare of the union of its rows' experts (each once, the first row's first, the heat counting those),
// one launch and one finish; its rows' outputs are bit for bit those of each row's own run (a prepare, launch and finish
// a row), the CPU's experts and the RAM cache's included, also with a per-expert post norm (the tier's routes come back
// unweighted). Like the other cache cases it needs a finite memory envelope.
#define main original_sink_test_main
#include "test-ssd-cpu-sink.cpp"
#undef main
#include "ggml-ssd/gpu/provider.h"
#include <set>

namespace {
std::set<std::pair<int32_t, int32_t>> tier_holds; // (layer, expert)
int tier_context = 0, tier_latent = 0;
// what the tier was asked for since the last reset
struct tier_calls {
    int prepare = 0, prepare_rows = 0, launch = 0, launch_rows = 0, finish = 0;
    std::vector<int32_t> union_experts; // the last prepare_rows' requests
    int counted = -1;                    // and its first row's
} calls;
// the prepared layer: its requests, which the tier holds, and the launched rows' inputs and routes
std::vector<ssd_vram_request> prepared;
std::vector<int> prepared_slot;
std::vector<float> launched_input;
size_t launched_stride = 0; // floats; 0 for a one-row launch
std::vector<int32_t> launched_request, launched_row;
std::vector<float> launched_weight;
bool launched = false, launched_rows = false;

int prepare_slots(const ssd_vram_request * r, int count, int * slots) {
    prepared.assign(r, r + count);
    prepared_slot.assign(size_t(count), -1);
    for (int i = 0; i < count; ++i) {
        slots[i] = tier_holds.count({ r[i].layer, r[i].expert }) ? i : -1;
        prepared_slot[size_t(i)] = slots[i];
    }
    launched = launched_rows = false;
    return 0;
}
int tier_prepare(void *, const ssd_vram_request * r, int count, int * slots, char *, size_t) {
    ++calls.prepare;
    return prepare_slots(r, count, slots);
}
int tier_prepare_rows(void *, const ssd_vram_request * r, int count, int counted, int * slots, char *, size_t) {
    ++calls.prepare_rows;
    calls.union_experts.clear();
    for (int i = 0; i < count; ++i) { calls.union_experts.push_back(r[i].expert); }
    calls.counted = counted;
    return prepare_slots(r, count, slots);
}
int tier_launch(void *, const void * input, size_t bytes, const ssd_vram_activation *, char *, size_t) {
    ++calls.launch;
    const auto * x = static_cast<const float *>(input);
    launched_input.assign(x, x + bytes / sizeof(float));
    launched = true;
    return 0;
}
int tier_launch_rows(void *, const void * input, size_t stride, const int32_t * request, const int32_t * row,
        const float * weight, int routes, const ssd_vram_activation *, char * e, size_t n) {
    ++calls.launch_rows;
    launched_stride = stride / sizeof(float);
    int rows = 0;
    for (int k = 0; k < routes; ++k) { rows = std::max(rows, row[k] + 1); }
    const auto * x = static_cast<const float *>(input);
    launched_input.assign(x, x + size_t(rows) * launched_stride);
    launched_request.assign(request, request + routes);
    launched_row.assign(row, row + routes);
    launched_weight.assign(weight, weight + routes);
    for (int k = 1; k < routes; ++k) {
        if (row[k] < row[k - 1]) { std::snprintf(e, n, "routes out of row order"); return 1; }
    }
    launched_rows = true;
    return 0;
}
// A route's row: a function of its expert, its input row and its weight only, the weight's multiply rounded alone.
void route_row(int32_t expert, const float * x, float weight, float * out) {
    for (int c = 0; c < tier_latent; ++c) {
        const float value = x[c] * (.01f * float(expert + 1)) + .001f * float(c % 7);
        volatile float scaled = value * weight;
        out[c] = scaled;
    }
}
int tier_finish(void *, float * contributions, size_t stride, char *, size_t) {
    ++calls.finish;
    if (launched_rows) {
        for (size_t k = 0; k < launched_request.size(); ++k) {
            const int request = launched_request[k];
            if (prepared_slot[size_t(request)] < 0) { continue; }
            route_row(prepared[size_t(request)].expert, launched_input.data() + size_t(launched_row[k]) * launched_stride,
                launched_weight[k], contributions + k * stride);
        }
    } else if (launched) {
        for (size_t j = 0; j < prepared.size(); ++j) {
            if (prepared_slot[j] < 0) { continue; }
            route_row(prepared[j].expert, launched_input.data(), prepared[j].weight, contributions + j * stride);
        }
    }
    launched = launched_rows = false;
    return 0;
}
int tier_declare(void *, const ssd_expert_layout * layout, char *, size_t) { tier_latent = layout->latent; return 0; }
void tier_held(void *, int32_t layer, const int32_t * experts, int count, uint8_t * out) {
    for (int i = 0; i < count; ++i) { out[i] = tier_holds.count({ layer, experts[i] }) != 0; }
}
int refuse(void *, char * e, size_t n) { std::snprintf(e, n, "not served by this tier"); return 1; }

// A stand-in GPU provider, as test-ssd-prefetch-tier.cpp's, whose format extension serves the experts it holds.
const ssd::gpu::provider_v1 * serving_provider() {
    static const ssd_vram_tier_api api = { 3, sizeof(ssd_vram_tier_api),
        [](uint64_t, char *, size_t) -> void * { return &tier_context; }, [](void *) {}, tier_declare,
        [](void *, uint64_t, uint64_t, int, char *, size_t) { return 0; },
        [](void *, const ssd_vram_request *, const void *, const void *, const void *, int, char *, size_t) { return 0; },
        tier_finish, [](void *, char *, size_t) { return 0; } };
    static const ssd_vram_residency_api residency = { 3, sizeof(ssd_vram_residency_api), tier_held };
    static const ssd_vram_format_api format = { 5, sizeof(ssd_vram_format_api), tier_prepare, tier_launch,
        tier_prepare_rows, tier_launch_rows };
    static const ssd_vram_prefill_api prefill = { 12, sizeof(ssd_vram_prefill_api),
        [](void *, const ssd_vram_request *, int count, int * slots, char *, size_t) {
            std::fill(slots, slots + count, -1); return 0; },
        [](void *) { return 0; },
        [](void *, const void *, size_t, const int32_t *, const float *, int, int, const ssd_vram_prefill_transient *,
           int, const ssd_vram_prefill_launch *, char * e, size_t n) { return refuse(nullptr, e, n); },
        [](void *, float *, size_t, ssd_vram_prefill_stats *, char * e, size_t n) { return refuse(nullptr, e, n); },
        refuse, [](void *) { return 0; },
        [](void *, const ssd_vram_request *, const void *, const void *, const void *, int * admitted, char * e, size_t n) {
            *admitted = 0; return refuse(nullptr, e, n); },
        [](void *, ssd_vram_prefill_lane *) {},
        [](void *, const int32_t *, int, char *, size_t) { return 0; },
        [](void *, int32_t, const void *, const void *, const void *, char *, size_t) { return 0; },
        [](void *, int32_t, char *, size_t) { return 0; } };
    static const ssd_vram_elastic_api elastic = { 8, sizeof(ssd_vram_elastic_api),
        [](void *, int, char *, size_t) { return 0; }, [](void *, int, char *, size_t) { return 0; },
        [](void *, char *, size_t) { return 0; },
        [](void *, int * slots, char *, size_t) { *slots = 0; return 0; },
        [](void *, const ssd_vram_request *, const void *, const void *, const void *, int * copied, char *, size_t) {
            *copied = 0; return 0; },
        [](void *, int, char *, size_t) { return 0; } };
    static const ssd_vram_deferred_api deferred = { 3, sizeof(ssd_vram_deferred_api),
        [](void *, const ssd_vram_request *, const void *, const void *, const void *, int * queued, char * e, size_t n) {
            *queued = 0; return refuse(nullptr, e, n); },
        [](void *, int32_t, ssd_vram_request *, int32_t *, int, int * count, char *, size_t) { *count = 0; return 0; },
        [](void *, char *, size_t) { return 0; } };
    static const ssd_vram_host_api host = { 3, sizeof(ssd_vram_host_api), [](void *) { return 0; },
        [](void *, const ssd_vram_host_job *, int, int * accepted, char *, size_t) { *accepted = 0; return 0; },
        [](void *, double * resident, double * hosted) { *resident = *hosted = 0; },
        [](void *, const void *, size_t, char *, size_t) { return 0; }, [](void *, const void *, char *, size_t) { return 0; },
        [](void *, ssd_vram_host_costs * costs) { *costs = {}; } };
    static const ssd_vram_policy_api policy = { 5, sizeof(ssd_vram_policy_api), [](void *, double, double) {},
        [](void *, uint64_t) {}, [](void *, uint64_t) {}, [](void *, uint64_t) { return 0; },
        [](void *) { return uint64_t(0); }, [](void *, const int *, int, char *, size_t) { return 0; },
        [](void *, int (*)(void *, int32_t, int32_t), void *) {} };
    static const ssd::gpu::settings_api settings = {};
    static const ssd::gpu::testing::api testing = { 3, sizeof(ssd::gpu::testing::api),
        [](uint64_t, const ssd::gpu::testing::tier_options *, char *, size_t) noexcept -> void * {
            return &tier_context; } };
    static const ssd::gpu::transport_v2 transport = {};
    static const ssd::gpu::provider_v1 provider = { 2, sizeof(ssd::gpu::provider_v1), &api, &residency, &format,
        &prefill, &elastic, &deferred, &host, &policy, &settings, &testing, &transport,
        [](int) { return false; }, [](ggml_backend_t) { return 0; },
        [](ggml_backend_t, const ssd::runtime_services *, char *, size_t) noexcept { return true; },
        [](ggml_backend_t) -> void * { return nullptr; } };
    return &provider;
}

// The distinct experts of a step's live slots in their first route's order, and how many of them its first row that
// names any has.
std::pair<std::vector<int32_t>, int> step_union(const rows_step & s) {
    std::vector<int32_t> experts;
    int first = 0;
    for (size_t i = 0; i < s.ids.size(); ++i) {
        if (s.ids[i] >= 0 && std::find(experts.begin(), experts.end(), s.ids[i]) == experts.end()) { experts.push_back(s.ids[i]); }
        if ((i + 1) % size_t(s.nslots) == 0 && !first) { first = int(experts.size()); }
    }
    return { experts, first };
}
}

int main(int argc, char ** argv) {
    fixture_inputs inputs; try {
    int threads = 4, depth = 2;
    if (argc == 5 && !std::strcmp(argv[1], "--threads") && !std::strcmp(argv[3], "--ring-experts")) {
        threads = std::stoi(argv[2]); depth = std::stoi(argv[4]);
    } else { require(argc == 1, "usage: test-ssd-rows-tier [--threads N --ring-experts N]"); }
    test_resources::cpu_participants(threads - 1, 1);
    test_resources::io_ring();
    test_resources::finite_envelope();
    for (const auto & kv : { std::pair<const char *, const char *>{ "ready_first", "1" }, { "early_up_gate", "1" },
            { "role_major", "1" }, { "nonblocking_down_sched", "1" }, { "dynamic_rows", "1" },
            { "dynamic_chunk_rows", "64" }, { "cache_policy", "lru" }, { "cache_allocation", "uniform" },
            { "read_fixed", "0" }, { "profile", "0" }, { "vram_budget", "1" }, { "vram_async", "0" },
            { "vram_deferred", "0" }, { "vram_prefill", "0" } }) {
        require(inputs.set(kv.first, kv.second) == 0, "set rows tier test environment");
    }
    ggml_cpu_init();
    ssd::runtime_services services;
    services.gpu = serving_provider();
    inputs.set_runtime_services(services);
    fixture f; make_synthetic(f);
    const uint64_t unit = triplet_bytes(f);
    tier_holds = { { 1, 1 }, { 1, 4 }, { 1, 7 }, { 2, 10 }, { 2, 13 } };
    auto rows = make_sink(inputs, f, threads, depth), own = make_sink(inputs, f, threads, depth);
    for (auto * s : { rows.get(), own.get() }) { configure_cache(s, 12 * unit, { 1, 2 }, 6); }
    std::vector<float> norm(size_t(f.latent));
    for (int i = 0; i < f.latent; ++i) { norm[size_t(i)] = .1f + float(i * 17 % 29) / 13.0f; }
    const ssd_cpu_sink_activation situ = { SSD_CPU_SINK_ACTIVATION_SITU, 4.0f, 25.0f, 0.0f, nullptr, 0.0f };
    const ssd_cpu_sink_activation normed = { SSD_CPU_SINK_ACTIVATION_SWIGLU, 0.0f, 0.0f, 0.0f, norm.data(), 1e-6f };
    // tier experts in and across rows beside the CPU's, repeats, empty slots, a row the tier serves alone, an empty row
    const std::vector<std::vector<rows_step>> tokens = {
        { make_rows(1, 4, { 1, 2, -1, 4,  4, 1, 3, -1,  7, 7, -1, -1,  2, 3, 4, 7 }),
          make_rows(2, 3, { 10, 11, 13,  -1, -1, -1,  11, 10, 12,  13, 13, 10 }) },
        { make_rows(1, 4, { 5, 1, 6, -1,  6, 5, 4, 7,  1, -1, 7, 6,  7, 6, 5, 1 }),
          make_rows(2, 3, { -1, -1, -1,  12, 14, 10,  14, 12, 13,  11, 10, 14 }) },
    };
    uint64_t token = 1;
    for (const auto * activation : { &situ, &normed }) {
        for (const auto & steps : tokens) {
            for (auto * s : { rows.get(), own.get() }) { begin_cache(s, 51, token); }
            for (const auto & step : steps) {
                const std::string what = std::string(activation == &situ ? "tier_situ" : "tier_post_norm") + "_token" +
                    std::to_string(token) + "_layer" + std::to_string(step.layer);
                const auto input = rows_input(f, step.rows(), int(token) * 5 + step.layer);
                ssd_cpu_sink_stats one = {}, each = {};
                calls = {};
                const auto a = run_rows_step(inputs, f, rows.get(), step, input, true, *activation, one);
                const auto once = calls;
                calls = {};
                const auto b = run_rows_step(inputs, f, own.get(), step, input, false, *activation, each);
                const auto per_row = calls;
                require(same_bits(a, b), what + ": the rows' run differs from their own runs");
                require(once.prepare_rows == 1 && once.launch_rows == 1 && once.finish == 1 && once.prepare == 0 &&
                        once.launch == 0, what + ": the rows' run did not take one prepare, launch and finish");
                require(per_row.prepare == step.rows() && per_row.launch == step.rows() && per_row.finish == step.rows() &&
                        per_row.prepare_rows == 0 && per_row.launch_rows == 0, what + ": a row's run took the rows' calls");
                const auto expected = step_union(step);
                require(once.union_experts == expected.first && once.counted == expected.second,
                    what + ": the tier was not given the union of the rows' experts, the first row's first");
                same_cache(rows.get(), own.get(), what);
                require(one.experts == step.experts() && one.cache_hits == each.cache_hits &&
                        one.cache_admitted == each.cache_admitted && one.read_requests <= each.read_requests,
                    what + ": the rows' run took its CPU experts otherwise than their own runs");
                std::printf("case=%s rows=%d experts=%zu counted=%d reads=%llu/%llu status=PASS\n", what.c_str(),
                    step.rows(), step.experts(), once.counted, (unsigned long long) one.read_requests,
                    (unsigned long long) each.read_requests);
            }
            ++token;
        }
    }
    std::printf("ROWS_TIER_OK threads=%d ring_experts=%d\n", threads, depth);
    return 0;
} catch (const std::exception & e) { return test_resources::report("ROWS_TIER", e); } }
