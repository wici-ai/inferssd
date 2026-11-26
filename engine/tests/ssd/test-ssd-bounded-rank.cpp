// SPDX-License-Identifier: Apache-2.0
#include "ggml-ssd/place/bounded-rank.h"
#include "ggml-ssd/host/expert-cache.h"
#include "ggml-cuda/ssd/vram-slot-table.h"
#include "ggml-cpu.h"
#include "llama.h"
#include "../../src/ssd/cpu-runtime.h"
#include "../../src/llama-graph.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

static void check(bool ok, const char * why) {
    if (!ok) { std::fprintf(stderr, "bounded rank: %s\n", why); std::exit(1); }
}

static void selection() {
    // Exhaust all readiness masks, K and J on a nontrivial ranked permutation.
    const int32_t rank[] = {5, 2, 7, 0, 6, 1, 4, 3};
    float scores[8];
    for (int r = 0; r < 8; ++r) { scores[rank[r]] = float(8 - r); }
    uint64_t cases = 0;
    for (int m = 1; m <= 8; ++m) {
        for (int mask = 0; mask < (1 << m); ++mask) {
            uint8_t ready[8];
            for (int r = 0; r < m; ++r) { ready[r] = (mask >> r) & 1; }
            for (int k = 1; k <= m; ++k) {
                for (int j = 0; j <= k; ++j) {
                    int32_t out[8];
                    // Independent lexicographic reference: protected ranks,
                    // then ready/unready partition in ranking order.
                    std::vector<int> refs(m);
                    std::iota(refs.begin(), refs.end(), 0);
                    std::stable_sort(refs.begin(), refs.end(), [&](int a, int b) {
                        const int ca = a < j ? 0 : ready[a] ? 1 : 2;
                        const int cb = b < j ? 0 : ready[b] ? 1 : 2;
                        return ca < cb;
                    });
                    int swaps = 0;
                    for (int i = 0; i < k; ++i) { swaps += refs[i] >= k; }
                    if (!swaps) { std::iota(refs.begin(), refs.end(), 0); }
                    int actual = ssd::place::bounded_rank_fill(rank, scores, ready, m, k, j, true, out);
                    check(actual == swaps, "substitution count");
                    for (int i = 0; i < k; ++i) { check(out[i] == rank[refs[i]], "reference selection"); }
                    ssd::place::bounded_rank_fill(rank, scores, ready, m, k, j, false, out);
                    check(std::equal(out, out + k, rank), "disabled identity");
                    ++cases;
                }
            }
        }
    }
    // Resident candidates in excluded groups must never be used.
    scores[rank[3]] = -INFINITY;
    scores[rank[4]] = -INFINITY;
    uint8_t ready[] = {0, 0, 0, 1, 1, 1, 1, 1};
    int32_t out[3];
    ssd::place::bounded_rank_fill(rank, scores, ready, 8, 3, 1, true, out);
    check(out[0] == 5 && out[1] == 1 && out[2] == 4, "group mask boundary");
    // Fewer eligible entries than K falls back to original graph IDs.
    for (auto & score : scores) { score = -INFINITY; }
    ssd::place::bounded_rank_fill(rank, scores, ready, 8, 3, 1, true, out);
    check(std::equal(out, out + 3, rank), "degenerate fallback");
    std::printf("selection: %llu exhaustive cases, masks and fallback passed\n", (unsigned long long) cases);
}

static void snapshots() {
    vram_slot_table table(2, {0}, false, 0, false, 2, 0, 0, 8);
    ssd_vram_request key = {}; key.layer = 0; key.expert = 3;
    auto & e = table.slots[0];
    table.change(e, [&](vram_slot_table::entry & x) { x.key = key; x.keyed = true; x.uploading = true; });
    check(table.holds(0, 3) && !table.route_ready(0, 3), "deferred upload excluded");
    table.change(e, [](vram_slot_table::entry & x) { x.uploading = false; x.valid = true; });
    const auto clock = table.clock, step = table.step;
    const auto counts = table.layers[0].counts;
    check(table.route_ready(0, 3) && !table.route_ready(0, 4) && !table.route_ready(-1, 3), "published GPU readiness");
    check(table.clock == clock && table.step == step && table.layers[0].counts == counts, "nontraining GPU query");
    // A keyed published payload outside the current decode allocation remains
    // visible to grouped prefill, but must not bias decode's expert selection.
    const auto old_begin = table.layers[0].begin, old_count = table.layers[0].count;
    table.layers[0].begin = 1; table.layers[0].count = 1;
    check(table.ready(0, 3) && !table.route_ready(0, 3), "phase-specific readiness was conflated");
    table.layers[0].begin = old_begin; table.layers[0].count = old_count;
    table.change(e, [](vram_slot_table::entry & x) { x.valid = false; x.seed = true; x.uploading = true; });
    check(!table.route_ready(0, 3), "unpublished seed excluded");

    ssd::host::expert_cache_options config; config.require_shared = false;
    ssd::host::expert_cache cache(config);
    cache.layers.push_back({}); cache.layers[0].id = 4; cache.layers[0].count = 1;
    cache.entries.resize(1);
    auto & h = cache.entries[0]; h.id = 3; h.enabled = h.valid = h.resident = true;
    // Metadata-only fixture: no payload access or fabricated page allocation.
    char byte = 0; cache.arena = &byte; cache.active = true;
    check(cache.ready(4, 3) && !cache.ready(0, 3), "published host readiness");
    h.inflight = true; check(cache.holds(4, 3) && !cache.ready(4, 3), "inflight host read excluded"); h.inflight = false;
    h.reserved = true; check(!cache.ready(4, 3), "host reservation excluded"); h.reserved = false;
    h.lent = true; check(!cache.ready(4, 3), "lent host slot excluded"); h.lent = false;
    cache.active = false; check(!cache.ready(4, 3), "prefill inactive");
    cache.arena = nullptr;
    std::puts("snapshot: ready-only host and GPU metadata passed");
}

static void numeric_weights(int only_threads) {
    // Non-owning participants must return before even reading userdata or dst.
    llama_ssd_bounded_rank_compute(nullptr, 1, 2, nullptr);
    llama_ssd_bounded_rank_compute(nullptr, 9, 10, nullptr);
    ggml_context * ctx = ggml_init({4u << 20, nullptr, false});
    check(ctx, "graph context");
    // Model-sized fixture: 384 experts, K=6, with a J=1/M=16 window.
    // Exercise real top-M ranking as well as unbiased weight gather.
    constexpr int experts = 384, m = 16, k = 6;
    int32_t rank[experts];
    float scores[experts];
    for (int r = 0; r < experts; ++r) {
        rank[r] = (r * 17 + 5) % experts;
        scores[rank[r]] = float(experts - r);
    }
    const int expected_ranks[k] = {0, 1, 7, 8, 9, 10};
    auto configured = ssd::settings::parse("", "");
    check(bool(configured), "default context settings");
    ssd::runtime_services services = {};
    ssd::store::source_view source;
    source.resolve = [](void *, const char *, int, size_t bytes, int64_t,
            int * fd, uint64_t * offset, size_t * logical) noexcept {
        *fd = 0; *offset = 0; *logical = bytes; return true;
    };
    // Only metadata is needed by the selector: actual MXFP4 geometry is legal.
    ggml_tensor * expert = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, 32, 32, experts);
    llama_ssd_cpu_sink_runtime runtime(1, {{0, {expert, expert, expert}}},
        configured.value, services, source, true);
    ssd::cpu::api_v2 api = {};
    int snapshot_calls = 0;
    api.route_ready = [](ssd_cpu_sink * sink, int32_t layer, const int32_t *, int count, uint8_t * out) {
        ++*reinterpret_cast<int *>(sink);
        check(layer == 0 && count == 16, "real selector snapshot handoff");
        const uint8_t ready[] = {0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0};
        std::copy(ready, ready + count, out);
    };
    runtime.cpu_api = &api;
    // The callback never dereferences the sink: the mock snapshot owns no bytes.
    runtime.sink = reinterpret_cast<ssd_cpu_sink *>(&snapshot_calls);
    llama_ssd_cpu_sink_node node = {}; node.runtime = &runtime; node.layer = 0; node.route_ready.resize(m);
    ggml_tensor * selection = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, experts);
    std::copy(scores, scores + experts, static_cast<float *>(selection->data));
    // A parallel add keeps the graph pool active for every custom-op worker,
    // even though the selector requests n_tasks=1. This matches mixed decode.
    ggml_tensor * zeros = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, experts);
    ggml_set_f32(zeros, 0.0f);
    selection = ggml_add(ctx, selection, zeros);
    ggml_tensor * ranking = ggml_cont(ctx, ggml_argsort_top_k(ctx, selection, m));
    ggml_tensor * args[] = {ranking, selection};
    ggml_tensor * ids = ggml_custom_4d(ctx, GGML_TYPE_I32, k, 1, 1, 1, args, 2,
        llama_ssd_bounded_rank_compute, 1, &node);
    ggml_tensor * probs = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, experts, 1);
    float gates[experts];
    for (int id = 0; id < experts; ++id) { gates[id] = 1e-10f * (id + 1); }
    std::copy(gates, gates + experts, static_cast<float *>(probs->data));
    ggml_tensor * weights = ggml_reshape_2d(ctx, ggml_get_rows(ctx, probs, ids), k, 1);
    weights = ggml_scale(ctx, ggml_div(ctx, weights, llm_moe_weights_denominator(ctx, ggml_sum_rows(ctx, weights), LLM_ARCH_DEEPSEEK41)), 1.5f);
    ggml_cgraph * gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, weights);
    int executions = 0;
    for (int threads : {1, 2, 10}) {
        if (only_threads && threads != only_threads) { continue; }
        for (int reuse = 0; reuse < 3; ++reuse) {
            check(ggml_graph_compute_with_ctx(ctx, gf, threads) == GGML_STATUS_SUCCESS,
                "mixed parallel selector graph compute/reuse");
            ++executions;
            check(snapshot_calls == executions, "one snapshot per selector execution");
            check(runtime.bounded_rank_receipt && runtime.route_slots == uint64_t(executions * k) &&
                runtime.route_swaps == uint64_t(executions * 4), "one selection and telemetry update per execution");
            const float * result = static_cast<float *>(weights->data);
            double mass = 1e-20; // DeepSeek41 additive denominator, including tiny gates
            for (int r : expected_ranks) { mass += gates[rank[r]]; }
            float sum = 0;
            for (int i = 0; i < k; ++i) {
                const int id = static_cast<int32_t *>(ids->data)[i];
                check(id == rank[expected_ranks[i]], "K=6/M=16 reference identities");
                const double ref = 1.5 * double(gates[id]) / mass;
                sum += result[i];
                check(std::abs(result[i] - ref) < 2e-7, "unbiased F32 gather, renormalization and scaling");
            }
            check(std::abs(sum - 1.5f) < 3e-7, "normalized mass");
        }
        std::printf("numeric: mixed graph threads=%d, 3 executions match reference IDs/weights and single-worker telemetry\n", threads);
    }
    runtime.sink = nullptr; // metadata-only mock has no executor to destroy
    llama_ssd_cpu_sink_runtime disabled(2, {{0, {expert, expert, expert}}},
        configured.value, services, source);
    check(!disabled.bounded_rank, "other architectures default off");
    llama_ssd_bounded_rank_receipt(disabled);
    llama_ssd_bounded_rank_receipt(disabled);
    ggml_free(ctx);
    std::puts("numeric: real ggml gather/normalization matches independent double reference");
}

int main(int argc, char ** argv) {
    int only_threads = 0;
    if (argc != 1) {
        check(argc == 3 && std::string(argv[1]) == "--threads", "usage: test-ssd-bounded-rank [--threads 1|2|10]");
        only_threads = std::atoi(argv[2]);
        check(only_threads == 1 || only_threads == 2 || only_threads == 10, "unsupported test thread count");
    }
    selection(); snapshots(); numeric_weights(only_threads); return 0;
}
