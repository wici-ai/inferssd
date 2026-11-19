// SPDX-License-Identifier: Apache-2.0
#include "llama-context.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <new>
#include <numeric>

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
static std::vector<llama_pos> span(llama_pos first, llama_pos count) {
    std::vector<llama_pos> result(count);
    std::iota(result.begin(), result.end(), first);
    return result;
}
static std::vector<uint32_t> cells_of(const std::vector<llama_pos> & positions) {
    return std::vector<uint32_t>(positions.begin(), positions.end());
}
static bool same_recurrent(const llama_memory_recurrent & mem, const llama_memory_recurrent::metadata_undo & saved) {
    if (mem.head != saved.head || mem.used != saved.used || mem.n != saved.n || mem.rs_z != saved.rs_z ||
            mem.cells.size() != saved.cells.size()) { return false; }
    for (size_t i = 0; i < mem.cells.size(); ++i) {
        const auto & a = mem.cells[i];
        const auto & b = saved.cells[i];
        if (a.pos != b.pos || a.src != b.src || a.src0 != b.src0 || a.tail != b.tail || a.seq_id != b.seq_id) {
            return false;
        }
    }
    return true;
}
static void require_named(bool condition, const char * name, const char * message) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s: %s\n", name, message); std::abort(); }
}
// A prompt ubatch whose graph gets no memory is undone and split again (prefill_refit). Each allocation the
// capture and the apply make fails in turn, then none fails: the undo must succeed, allocate nothing, and leave
// the caches as they were.
static void undo_every_failure(const char * name, const std::function<std::unique_ptr<llama_memory_context_i>()> & make,
                               const std::function<bool()> & unchanged) {
    size_t total = 0;
    {
        auto context = make();
        const auto start = allocations.load();
        require_named(context->apply_precompute(), name, "apply failed");
        total = allocations.load() - start;
        require_named(context->cancel_precompute() && unchanged(), name, "undo after a full apply failed");
    }
    for (size_t budget = 0; budget <= total + 1; ++budget) {
        auto context = make();
        fail_after = (long) budget;
        try { (void) context->apply_precompute(); } catch (const std::bad_alloc &) {}
        fail_after = 0;
        const auto a = allocations.load();
        const bool undone = context->cancel_precompute();
        const auto b = allocations.load();
        fail_after = -1;
        require_named(undone && a == b, name, "undo allocated or failed");
        require_named(unchanged(), name, "caches changed after the undo");
    }
}
// Qwen3.6 and the other hybrid models: a prompt from position 0 (nothing reused) and one after a kept prefix.
static void recurrent_and_hybrid_undo(llama_model & model) {
    for (const bool prefix : {false, true}) {
        const auto first  = prefix ? span(0, 11) : std::vector<llama_pos>();
        const auto prompt = prefix ? span(11, 10) : span(0, 10);
        const llama_pos last = prefix ? 10 : -1;

        llama_memory_recurrent recr(model, GGML_TYPE_F32, GGML_TYPE_F32, false, 1, 1, {});
        if (prefix) { require(recr.find_slot(batch(first)), "recurrent prefix"); }
        const auto recr_saved = recr.save_metadata();
        undo_every_failure("recurrent",
            [&] { return std::make_unique<llama_memory_recurrent_context>(&recr, std::vector<llama_ubatch>{batch(prompt)}); },
            [&] { return same_recurrent(recr, *recr_saved) && recr.seq_pos_max(0) == last; });

        llama_memory_hybrid hybrid(model, GGML_TYPE_F32, GGML_TYPE_F32, false, 32, 1, 0, LLAMA_SWA_TYPE_NONE,
                                   GGML_TYPE_F32, GGML_TYPE_F32, 1, 1, false, true);
        if (prefix) {
            hybrid.get_mem_attn()->apply_ubatch(slot(cells_of(first)), batch(first));
            require(hybrid.get_mem_recr()->find_slot(batch(first)), "hybrid recurrent prefix");
        }
        const auto hybrid_saved = hybrid.get_mem_recr()->save_metadata();
        undo_every_failure("hybrid",
            [&] { return std::make_unique<llama_memory_hybrid_context>(&hybrid,
                      llama_kv_cache::slot_info_vec_t{slot(cells_of(prompt))}, std::vector<llama_ubatch>{batch(prompt)}); },
            [&] { return same_recurrent(*hybrid.get_mem_recr(), *hybrid_saved) &&
                      hybrid.get_mem_attn()->seq_pos_max(0) == last && hybrid.get_mem_recr()->seq_pos_max(0) == last; });

        llama_memory_hybrid_iswa iswa(model, GGML_TYPE_F32, GGML_TYPE_F32, false, true, 32, 16, 1,
                                      GGML_TYPE_F32, GGML_TYPE_F32, 1, 1, false, true);
        auto * base = iswa.get_mem_attn()->get_base();
        auto * swa  = iswa.get_mem_attn()->get_swa();
        if (prefix) {
            base->apply_ubatch(slot(cells_of(first)), batch(first));
            swa->apply_ubatch(slot(cells_of(first)), batch(first));
            require(iswa.get_mem_recr()->find_slot(batch(first)), "hybrid iswa recurrent prefix");
        }
        const auto iswa_saved = iswa.get_mem_recr()->save_metadata();
        const llama_pos swa_min = swa->seq_pos_min(0);
        undo_every_failure("hybrid iswa",
            [&] { return std::make_unique<llama_memory_hybrid_iswa_context>(&iswa,
                      llama_kv_cache::slot_info_vec_t{slot(cells_of(prompt))},
                      llama_kv_cache::slot_info_vec_t{slot(cells_of(prompt))}, std::vector<llama_ubatch>{batch(prompt)}); },
            [&] { return same_recurrent(*iswa.get_mem_recr(), *iswa_saved) && base->seq_pos_max(0) == last &&
                      swa->seq_pos_max(0) == last && swa->seq_pos_min(0) == swa_min; });

        // qwen4exp: the indexer cache and the pool layout too
        model.hparams.indexer_kpool = 2;
        llama_memory_hybrid_idx idx(model, GGML_TYPE_F32, GGML_TYPE_F32, false, 32, 1, 0, LLAMA_SWA_TYPE_NONE,
                                    GGML_TYPE_F32, GGML_TYPE_F32, 1, 1, false, true, nullptr, nullptr,
                                    [](int32_t) { return true; });
        model.hparams.indexer_kpool = 0;
        require(idx.get_mem_idx() != nullptr && idx.get_kpool() == 2, "indexer cache setup");
        if (prefix) {
            idx.get_mem_attn()->apply_ubatch(slot(cells_of(first)), batch(first));
            idx.get_mem_idx()->apply_ubatch(slot(cells_of(first)), batch(first));
            require(idx.get_mem_recr()->find_slot(batch(first)), "hybrid idx recurrent prefix");
            (void) idx.kpool_layout_update();
        }
        const auto idx_saved = idx.get_mem_recr()->save_metadata();
        const auto make_idx = [&] { return std::make_unique<llama_memory_hybrid_idx_context>(&idx,
            llama_kv_cache::slot_info_vec_t{slot(cells_of(prompt))},
            llama_kv_cache::slot_info_vec_t{slot(cells_of(prompt))}, std::vector<llama_ubatch>{batch(prompt)}); };
        undo_every_failure("hybrid idx", make_idx,
            [&] { return same_recurrent(*idx.get_mem_recr(), *idx_saved) && idx.get_mem_attn()->seq_pos_max(0) == last &&
                      idx.get_mem_idx()->seq_pos_max(0) == last; });
        // the dropped layout is rebuilt from the restored cells when the prompt is applied again
        auto applied = make_idx();
        require(applied->apply_precompute(), "hybrid idx apply after an undo");
        applied->submit_precompute();
        require(idx.get_mem_idx()->seq_pos_max(0) == prompt.back(), "hybrid idx apply after an undo did not publish");
    }
}

// The refit's ubatch search, whatever probe it ends on: the graph left reserved is the one of the size it returns.
// 48040 tokens with 25856 fitting ends on a refused probe (26112), as DeepSeek-V4.1 did on a 48 GB Mac, so 25856 is
// reserved again; 35418 with 18432 fitting ends on 18432 itself. A refused second reserve selects nothing.
static void refit_search_holds_selected() {
    struct search_case { uint32_t target, fits; bool again; uint32_t expect; const char * name; };
    for (const auto & c : {search_case{48040, 25856, true, 25856, "ends refused"},
                           search_case{35418, 18432, true, 18432, "ends accepted"},
                           search_case{48040, 25856, false, 0, "second reserve refused"}}) {
        uint32_t held = 0, last = 0;
        bool reserved_fits = false;
        const uint32_t got = llama_prefill_ubatch_search(c.target, 1, [&](uint32_t n) {
            bool ok = n <= c.fits && !(n == c.fits && reserved_fits && !c.again);
            reserved_fits = reserved_fits || (ok && n == c.fits);
            held = ok ? n : 0;
            last = n;
            return ok;
        });
        require_named(got == c.expect, c.name, "search selected another size");
        require_named(got == 0 || (held == got && last == got), c.name, "the graph held is not the selected size's");
    }
}

int main() {
    cells_partial_insert_oom();
    refit_search_holds_selected();
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
    recurrent_and_hybrid_undo(model);
    std::puts("PRECOMPUTE_METADATA_PASS: purge/ext/shared indexes, OOM capture/partial inserts, compound raw, prepare exception, submit boundary, recurrent and hybrid, refit search");
}
