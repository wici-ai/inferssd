// The next-layer prefetch against the GPU tier's residency (kimi_vram_residency_api held): a hinted expert the tier
// holds is not novel and not read, and a read whose expert the tier came to hold before its run is not needed by
// that run. The tier here is a stand-in this executable exports: it holds the experts the test names and serves
// none (every expert is the CPU's), so the outputs are the CPU's and only the prefetch's decisions change. Like the
// other cache cases it needs a finite cgroup memory limit.
#define main original_sink_test_main
#include "test-kimi-cpu-sink.cpp"
#undef main
#include "kimi-vram-tier-api.h"
#include <set>

namespace {
std::set<std::pair<int32_t, int32_t>> tier_holds; // (layer, expert)
int tier_context = 0;
void * tier_create(uint64_t, char *, size_t) { return &tier_context; }
void tier_destroy(void *) {}
int tier_begin(void *, uint64_t, uint64_t, int, char *, size_t) { return 0; }
int tier_prepare(void *, const kimi_vram_request *, int count, int, int, int * slots, char *, size_t) {
    std::fill(slots, slots + count, -1);
    return 0;
}
int tier_launch(void *, const void *, size_t, float, float, char *, size_t) { return 0; }
int tier_admit(void *, const kimi_vram_request *, const void *, const void *, const void *, int, char *, size_t) { return 0; }
int tier_finish(void *, float *, size_t, char *, size_t) { return 0; }
int tier_cancel(void *, char *, size_t) { return 0; }
int tier_no_lease(void *, uint64_t, const kimi_vram_request *, int, kimi_vram_ready_lease *, int *, char * e, size_t n) {
    std::snprintf(e, n, "no ready leases in this test"); return 1;
}
int tier_no_lease_check(void *, const kimi_vram_ready_lease *, char * e, size_t n) {
    std::snprintf(e, n, "no ready leases in this test"); return 1;
}
void tier_held(void *, int32_t layer, const int32_t * experts, int count, uint8_t * out) {
    for (int i = 0; i < count; ++i) { out[i] = tier_holds.count({ layer, experts[i] }) != 0; }
}
}

extern "C" const kimi_vram_tier_api * kimi_vram_tier_get_api() {
    static const kimi_vram_tier_api api = { 2, sizeof(kimi_vram_tier_api), tier_create, tier_destroy, tier_begin,
        tier_prepare, tier_launch, tier_admit, tier_finish, tier_cancel };
    return &api;
}
extern "C" const kimi_vram_format_api * kimi_vram_format_get_api() {
    static const kimi_vram_format_api api = { 1, sizeof(kimi_vram_format_api),
        [](void *, const kimi_vram_request *, int count, const kimi_vram_expert_format *, int * slots, char *, size_t) {
            std::fill(slots, slots + count, -1); return 0;
        },
        [](void *, const void *, size_t, const kimi_vram_activation *, char *, size_t) { return 0; } };
    return &api;
}
extern "C" const ssd::gpu::testing::api * ggml_ssd_tier_testing_get_api() {
    static const ssd::gpu::testing::api api = { 1, sizeof(ssd::gpu::testing::api),
        [](uint64_t, const ssd::gpu::testing::tier_options *, char *, size_t) noexcept -> void * { return &tier_context; } };
    return &api;
}
extern "C" const kimi_vram_residency_api * kimi_vram_residency_get_api() {
    static const kimi_vram_residency_api api = { 2, sizeof(kimi_vram_residency_api), tier_no_lease,
        tier_no_lease_check, tier_no_lease_check, tier_held };
    return &api;
}

// Token 1 makes six slots of each layer resident (layer 2: 10..15). At token 2 the tier holds expert 6 of layer 2,
// and layer 1's run gets the hint 5, 6, 7, 12 (12 is cached). At depth 2: 6 is held (skip_vram, not novel), so only
// 5 is read; the ranks' novelty is 1,0,1,0. Before layer 2's run the tier comes to hold 5 and gives 6 back: layer 2
// selects 5, 6, 7 and 12. The read of 5 is not one it needs (vram_late, never activated); 6 was skipped and is lost
// (skip_vram_lost), read on demand like 7. The read of 5 has finished by then (the test waits), so the RAM cache
// publishes it and the run takes 5 from there, and no read ran past the window (no spill).
int main(int argc, char ** argv) {
    fixture_inputs inputs; try {
    int threads = 4;
    if (argc == 3 && !std::strcmp(argv[1], "--threads")) { threads = std::stoi(argv[2]); }
    else { require(argc == 1, "usage: test-kimi-prefetch-tier [--threads N]"); }
    inputs.set("split_cv", "1");
    test_resources::finite_cgroup();
    for (const auto & kv : { std::pair<const char *, const char *>{ "ready_first", "1" },
            { "early_up_gate", "1" },
            { "role_major", "1" }, { "nonblocking_down_sched", "1" },
            { "cache_policy", "lru" }, { "cache_allocation", "uniform" },
            { "read_fixed", "0" }, { "profile", "0" },
            { "prefetch_depth", "2" }, { "vram_budget", "1" },
            { "vram_exclusive", "0" }, { "vram_async", "0" },
            { "vram_deferred", "0" }, { "vram_prefill", "0" } }) {
        require(inputs.set(kv.first, kv.second) == 0, "set prefetch tier test environment");
    }
    ggml_cpu_init();
    fixture f; make_synthetic(f);
    const uint64_t unit = triplet_bytes(f);
    std::vector<float> input(f.latent);
    for (size_t i = 0; i < input.size(); ++i) { input[i] = float(int(i % 19) - 9) * .021f; }
    stderr_capture log;
    {
        auto sink = make_sink(inputs, f, threads, 2);
        configure_cache(sink.get(), 12 * unit, { 1, 2 }, 6);
        begin_cache(sink.get(), 1, 1);
        cache_case(inputs, "tier_warm_layer1", f, sink.get(), 1, input, { 1, 2 }, { .4f, .6f }, 0, 2);
        cache_case(inputs, "tier_warm_layer2", f, sink.get(), 2, input, { 10, 11, 12, 13, 14, 15 },
            { .1f, .1f, .2f, .2f, .2f, .2f }, 0, 6);
        begin_cache(sink.get(), 1, 2);
        tier_holds = { { 2, 6 } };
        const int32_t hint[] = { 5, 6, 7, 12 };
        kimi_cpu_sink_prefetch_hint(sink.get(), 2, hint, 4);
        cache_case(inputs, "tier_hinted_layer1", f, sink.get(), 1, input, { 1, 2 }, { .4f, .6f }, 2, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(200)); // the read of 5 completes
        tier_holds = { { 2, 5 } };
        cache_case(inputs, "tier_predicted_layer2", f, sink.get(), 2, input, { 5, 6, 7, 12 }, { .2f, .2f, .2f, .4f }, 2, -1);
    }
    std::istringstream lines(log.read());
    std::string line, report;
    while (std::getline(lines, line)) {
        if (line.rfind("[kimi_cpu_sink_prefetch] reason=destroy ", 0) == 0) { report = line; }
    }
    require(!report.empty(), "no prefetch report at destroy");
    auto field = [&](const std::string & key) {
        const size_t at = report.find(" " + key + "=");
        require(at != std::string::npos, "prefetch report lacks " + key);
        const size_t begin = at + key.size() + 2;
        return report.substr(begin, report.find(' ', begin) - begin);
    };
    require(field("skip_vram") == "1" && field("issued") == "1" && field("vram_late") == "1" &&
            field("skip_vram_lost") == "1" && field("activated") == "0",
            "the tier's residency did not decide the reads: " + report);
    require(field("predicted_r") == "1,1,1,1" && field("novel_r") == "1,0,1,0" &&
            field("novel_routed_r") == "1,0,1,0" && field("issued_r") == "1,0,0,0" && field("used_r") == "0,0,0,0",
            "the tier's residency did not decide the novelty: " + report);
    require(field("started_r") == "1,0,0,0" && field("spills") == "0" && field("spill_us") == "0" &&
            field("spill_us_r") == "0,0,0,0", "a read inside its window was counted as a spill: " + report);
    std::printf("PREFETCH_TIER_OK %s\n", report.c_str());
    return 0;
} catch (const std::exception & e) { return test_resources::report("PREFETCH_TIER", e); } }
