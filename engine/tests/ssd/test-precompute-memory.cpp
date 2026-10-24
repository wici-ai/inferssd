// SPDX-License-Identifier: MIT
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-model.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>

static std::atomic<long> fail_after{-1};
static std::atomic<size_t> allocations{0};
void * operator new(size_t bytes) {
    ++allocations;
    const auto remaining = fail_after.load();
    if (remaining == 0) { throw std::bad_alloc(); }
    if (remaining > 0) { --fail_after; }
    if (auto * result = std::malloc(bytes ? bytes : 1)) { return result; }
    throw std::bad_alloc();
}
void operator delete(void * value) noexcept { std::free(value); }
void * operator new[](size_t bytes) { return ::operator new(bytes); }
void operator delete[](void * value) noexcept { ::operator delete(value); }
#if defined(__cpp_sized_deallocation)
void operator delete(void * value, size_t) noexcept { ::operator delete(value); }
void operator delete[](void * value, size_t) noexcept { ::operator delete(value); }
#endif
static void require(bool condition, const char * message) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); std::abort(); }
}
static llama_ubatch batch(std::vector<llama_pos> positions, llama_seq_id seq = 0) {
    llama_ubatch result{};
    result.data = std::make_shared<llama_ubatch::data_t>();
    auto & data = *result.data;
    data.pos = std::move(positions);
    const auto n = data.pos.size();
    data.n_seq_id.assign(n, 1);
    data.seq_id_data.assign(n, seq);
    data.seq_id.resize(n);
    data.token.resize(n);
    data.seq_id_unq = {seq};
    for (size_t i = 0; i < n; ++i) { data.seq_id[i] = &data.seq_id_data[i]; data.token[i] = 100 + data.pos[i]; }
    result.b_equal_seqs = 1;
    result.n_tokens = result.n_seq_tokens = n;
    result.n_seqs = result.n_seqs_unq = result.n_pos = 1;
    result.pos = data.pos.data(); result.token = data.token.data();
    result.n_seq_id = data.n_seq_id.data(); result.seq_id = data.seq_id.data();
    result.seq_id_unq = data.seq_id_unq.data();
    return result;
}
static llama_kv_cache::slot_info slot(std::vector<uint32_t> indices, int stream = 0) {
    return {uint32_t(stream), uint32_t(stream), {stream}, {std::move(indices)}};
}
static void require_prefix(const llama_kv_cache & kv) {
    require(kv.seq_pos_min(0) == 0 && kv.seq_pos_max(0) == 10, "prefix metadata changed");
    auto query = batch({11});
    std::vector<llama_token> previous;
    kv.get_prev_tokens(query, 3, previous);
    require(previous == std::vector<llama_token>({108, 109, 110}), "Engram token metadata changed");
}
static void cells_partial_insert_oom() {
    llama_kv_cells cells;
    cells.resize(4); cells.pos_set(0, 1); cells.seq_add(0, 0); cells.seq_add(0, 1);
    cells.ext_set(0, {2, 3, 42});
    for (bool fail_sequence : {false, true}) {
        auto saved = cells.save_metadata({0, 1, 0});
        cells.rm(0);
        bool refused = false;
        if (fail_sequence) { cells.pos_set(0, 2); }
        fail_after = 0;
        try { if (fail_sequence) { cells.seq_add(0, 0); } else { cells.pos_set(0, 2); } }
        catch (const std::bad_alloc &) { refused = true; }
        const auto before = allocations.load();
        const bool restored = cells.restore_metadata(saved);
        const auto after = allocations.load();
        fail_after = -1;
        require(refused && restored && before == after, "partial insert undo allocated or failed");
        require(cells.get_used() == 1 && cells.seq_pos_max(0) == 1 && cells.seq_pos_max(1) == 1,
                "shared sequence indexes not restored");
        require(cells.seq_pos_tok_le(0, 2) == 42, "cell ext token not restored");
    }
}
int main() {
    cells_partial_insert_oom();
    llama_model model(llama_model_default_params());
    model.hparams = {};
    model.arch = LLM_ARCH_LLAMA;
    model.hparams.n_swa = 4;
    model.hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    model.hparams.n_embd_head_k_full = 8;
    model.hparams.n_embd_head_v_full = 8;
    model.hparams.indexer_head_size = 4;
    llama_kv_cache kv(model, GGML_TYPE_F32, GGML_TYPE_F32, false, false, true, 32, 2, 1,
                      4, LLAMA_SWA_TYPE_STANDARD, {}, {});
    auto prefix = batch({0,1,2,3,4,5,6,7,8,9,10});
    kv.apply_ubatch(slot({0,1,2,3,4,5,6,7,8,9,10}), prefix);
    kv.seq_cp(0, 1, 0, 4); // shared rows outside the overwritten slot
    auto overwrite = batch({11});
    const auto overwrite_slot = slot({6});
    auto saved = kv.save_ubatch_metadata(overwrite_slot);
    require(saved->streams.size() == 1 && saved->streams[0].second.rows.size() == 7,
            "purge outside written slot was not captured");
    kv.apply_ubatch(overwrite_slot, overwrite);
    require(kv.seq_pos_min(0) == 7 && kv.seq_pos_max(0) == 11, "actual overwrite did not purge");
    fail_after = 0;
    const auto before = allocations.load();
    const bool restored = kv.restore_ubatch_metadata(*saved);
    const auto after = allocations.load();
    fail_after = -1;
    require(restored && before == after, "real KV undo allocated");
    require_prefix(kv);
    require(kv.seq_pos_min(1) == 0 && kv.seq_pos_max(1) == 3, "shared purged sequence changed");

    // Exercise every capture allocation plus the first two actual index insert
    // failures. The failing batch overwrites an existing prefix position.
    llama_kv_cache_context measure(&kv, {overwrite_slot}, {overwrite});
    const auto start = allocations.load();
    require(measure.save_precompute(), "capture unavailable");
    const auto capture_allocations = allocations.load() - start;
    measure.discard_precompute();
    for (size_t budget = 0; budget <= capture_allocations + 3; ++budget) {
        llama_kv_cache_context context(&kv, {overwrite_slot}, {overwrite});
        fail_after = budget;
        try { (void) context.apply_precompute(); } catch (const std::bad_alloc &) {}
        fail_after = 0;
        const auto a = allocations.load();
        const bool undone = context.cancel_precompute();
        const auto b = allocations.load();
        fail_after = -1;
        require(undone && a == b, "capture/apply OOM cancellation was not allocation-free");
        require_prefix(kv);
    }
    // A two-stream write may purge different ranges in each stream.
    llama_kv_cache split(model, GGML_TYPE_F32, GGML_TYPE_F32, false, false, false, 32, 2, 1,
                         4, LLAMA_SWA_TYPE_STANDARD, {}, {});
    split.apply_ubatch(slot({0,1,2,3,4,5,6,7,8,9,10}, 0), prefix);
    split.apply_ubatch(slot({0,1,2,3,4,5,6,7,8,9,10}, 1), batch({0,1,2,3,4,5,6,7,8,9,10}, 1));
    auto dual = batch({11,11});
    dual.data->seq_id_data[1] = 1;
    const llama_kv_cache::slot_info dual_slot{0, 1, {0,1}, {{6},{4}}};
    auto dual_saved = split.save_ubatch_metadata(dual_slot);
    split.apply_ubatch(dual_slot, dual);
    fail_after = 0;
    const auto dual_before = allocations.load();
    const bool dual_restored = split.restore_ubatch_metadata(*dual_saved);
    const auto dual_after = allocations.load();
    fail_after = -1;
    require(dual_restored && dual_before == dual_after && split.seq_pos_min(0) == 0 &&
            split.seq_pos_max(0) == 10 && split.seq_pos_min(1) == 0 && split.seq_pos_max(1) == 10,
            "multi-stream purge restoration failed");

    // Force prepare's own speculative write to overwrite a non-oldest SWA
    // cell, purging other cells outside its selected slot. Every allocation
    // failure must restore both the original slot and all purged positions.
    llama_kv_cache planning(model, GGML_TYPE_F32, GGML_TYPE_F32, false, false, true, 32, 2, 1,
                            4, LLAMA_SWA_TYPE_STANDARD, {}, {});
    std::vector<llama_pos> reordered{7,0,1,2,3,4,5,6};
    std::vector<uint32_t> all_indices;
    for (int i = 8; i < 32; ++i) { reordered.push_back(i); }
    for (uint32_t i = 0; i < 32; ++i) { all_indices.push_back(i); }
    planning.apply_ubatch(slot(all_indices), batch(reordered));
    const std::vector<llama_ubatch> wrap_plan{batch({32}), batch({33})};
    const auto plan_start = allocations.load();
    require(planning.prepare(wrap_plan).size() == 2, "wrap prepare failed");
    const auto plan_allocations = allocations.load() - plan_start;
    for (size_t budget = 0; budget <= plan_allocations; ++budget) {
        fail_after = budget;
        try { (void) planning.prepare(wrap_plan); } catch (const std::bad_alloc &) {}
        fail_after = -1;
        require(planning.seq_pos_min(0) == 0 && planning.seq_pos_max(0) == 31,
                "prepare purge/exception changed prefix");
        std::vector<llama_token> tokens;
        planning.get_prev_tokens(wrap_plan[0], 32, tokens);
        for (int i = 0; i < 32; ++i) { require(tokens[i] == 100+i, "prepare purge lost token metadata"); }
    }
    // A submitted ubatch has no metadata-only restore path.
    llama_kv_cache_context submitted(&kv, {slot({11})}, {overwrite});
    require(submitted.apply_precompute(), "submit setup");
    submitted.submit_precompute();
    require(!submitted.cancel_precompute() && kv.seq_pos_max(0) == 11, "submitted work was metadata-rolled back");

    // Real compound DSV4 raw context: base and direct SWA publish together;
    // even a late snapshot OOM must leave both prefixes unchanged.
    llama_kv_cache_iswa raw(model, GGML_TYPE_F32, GGML_TYPE_F32, false, false, true,
                            false, 32, 2, 4, 1, {}, {});
    for (auto * cache : {raw.get_base(), raw.get_swa()}) {
        cache->apply_ubatch(slot({0,1}), batch({0,1}));
    }
    const auto append = batch({2});
    const auto append_slot = slot({2});
    for (size_t budget = 0; budget < 28; ++budget) {
        llama_kv_cache_dsv4_raw_context context(&raw, {append_slot}, {append_slot}, {append_slot}, {append}, {append});
        fail_after = budget;
        try { (void) context.apply_precompute(); } catch (const std::bad_alloc &) {}
        fail_after = 0;
        const auto a = allocations.load();
        const bool undone = context.cancel_precompute();
        const auto b = allocations.load();
        fail_after = -1;
        require(undone && a == b, "compound raw undo allocated or failed");
        require(raw.get_base()->seq_pos_max(0) == 1 && raw.get_swa()->seq_pos_max(0) == 1,
                "compound raw snapshot/apply OOM changed prefix");
    }
    // prepare's speculative apply also restores overwritten/purged metadata
    // on exceptions, rather than leaving its simulation published.
    const std::vector<llama_ubatch> planned = {append, batch({3})};
    for (size_t budget = 0; budget < 65; ++budget) {
        fail_after = budget;
        try { (void) raw.get_swa()->prepare(planned); } catch (const std::bad_alloc &) {}
        fail_after = -1;
        require(raw.get_swa()->seq_pos_max(0) == 1, "prepare exception leaked simulated metadata");
    }
    std::puts("PRECOMPUTE_METADATA_PASS: purge/ext/shared indexes, OOM capture/partial inserts, compound raw, prepare exception, submit boundary");
}
