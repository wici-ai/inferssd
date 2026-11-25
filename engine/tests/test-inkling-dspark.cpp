#include "chat.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "llama.h"
#include "../src/llama-ext.h"
#include "../src/llama-arch.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

static bool verify_pytorch_reference(
        llama_context * ctx_tgt,
        llama_model   * model_dft,
        const char    * ref_bin_path) {
    std::ifstream f(ref_bin_path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "FAIL: cannot open reference file %s\n", ref_bin_path);
        return false;
    }

    int32_t L = 0, n_taps = 0, anchor_id = 0;
    f.read(reinterpret_cast<char *>(&L), sizeof(int32_t));
    f.read(reinterpret_cast<char *>(&n_taps), sizeof(int32_t));
    f.read(reinterpret_cast<char *>(&anchor_id), sizeof(int32_t));

    const int32_t n_embd     = llama_model_n_embd(model_dft);
    const int32_t block_size = (int32_t) llama_model_dspark_block_size(model_dft);
    const int32_t n_draft    = block_size - 1;
    const int32_t n_embd_enc = n_taps * n_embd;

    std::vector<float>   raw_taps((size_t) L * n_embd_enc);
    std::vector<int32_t> expected_ids(n_draft);
    std::vector<float>   expected_max_logits(n_draft);

    f.read(reinterpret_cast<char *>(raw_taps.data()), raw_taps.size() * sizeof(float));
    f.read(reinterpret_cast<char *>(expected_ids.data()), expected_ids.size() * sizeof(int32_t));
    f.read(reinterpret_cast<char *>(expected_max_logits.data()), expected_max_logits.size() * sizeof(float));

    if (!llama_model_inkling2_dspark_attach_target(model_dft, llama_get_model(ctx_tgt))) {
        std::fprintf(stderr, "FAIL: llama_model_inkling2_dspark_attach_target failed\n");
        return false;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 256;
    cparams.n_batch         = 256;
    cparams.n_ubatch        = 256;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.ctx_other       = ctx_tgt;

    llama_context * ctx_dft = llama_init_from_model(model_dft, cparams);
    if (!ctx_dft) {
        std::fprintf(stderr, "FAIL: failed to init ctx_dft\n");
        return false;
    }
    llama_set_embeddings_nextn(ctx_dft, true, false);
    llama_set_causal_attn(ctx_dft, false);

    // 1. Inject L context rows
    llama_batch b_inj = llama_batch_init(L, n_embd_enc, 1);
    b_inj.n_tokens = L;
    std::memcpy(b_inj.embd, raw_taps.data(), raw_taps.size() * sizeof(float));
    for (int32_t i = 0; i < L; ++i) {
        b_inj.pos[i]       = i;
        b_inj.n_seq_id[i]  = 1;
        b_inj.seq_id[i][0] = 0;
        b_inj.logits[i]    = false;
    }
    if (llama_decode(ctx_dft, b_inj) != 0) {
        std::fprintf(stderr, "FAIL: context injection decode failed\n");
        llama_batch_free(b_inj);
        llama_free(ctx_dft);
        return false;
    }

    const float * f_rnn_all = llama_get_embeddings_nextn(ctx_dft);
    const float * f_rnn_last = f_rnn_all + (size_t) (L - 1) * n_embd;
    std::printf("Engine f_rnn[:4] = [%.6f, %.6f, %.6f, %.6f]\n",
            f_rnn_last[0], f_rnn_last[1], f_rnn_last[2], f_rnn_last[3]);

    llama_model_inkling2_dspark_set_f_rnn(model_dft, f_rnn_last);

    // 2. Run 8-token draft block with logits enabled so we check both ids and max_logits
    llama_batch b_dft = llama_batch_init(block_size, 0, 1);
    for (int32_t i = 0; i < block_size; ++i) {
        common_batch_add(b_dft, i == 0 ? anchor_id : 0, L + i, { 0 }, true);
    }
    if (llama_decode(ctx_dft, b_dft) != 0) {
        std::fprintf(stderr, "FAIL: draft block decode failed\n");
        llama_batch_free(b_dft);
        llama_batch_free(b_inj);
        llama_free(ctx_dft);
        return false;
    }
    llama_synchronize(ctx_dft);

    std::vector<int32_t> got_ids(n_draft, -1);
    llama_model_inkling2_dspark_get_draft_ids(model_dft, got_ids.data(), (uint32_t) n_draft);

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
    bool ok = true;
    for (int32_t s = 0; s < n_draft; ++s) {
        const float * logits_s = llama_get_logits_ith(ctx_dft, 1 + s);
        const float max_l = *std::max_element(logits_s, logits_s + n_vocab);
        const float diff  = std::fabs(max_l - expected_max_logits[s]);
        std::printf("  step %d: got_id=%6d (exp=%6d), max_logit=%.4f (exp=%.4f, diff=%.4f)\n",
                s, got_ids[s], expected_ids[s], max_l, expected_max_logits[s], diff);
        if (got_ids[s] != expected_ids[s] || diff > 0.25f) {
            ok = false;
        }
    }

    llama_batch_free(b_dft);
    llama_batch_free(b_inj);
    llama_free(ctx_dft);
    return ok;
}

int main(int argc, char ** argv) {
    llama_backend_init();

    // Unit check that always runs in ctest without external model files
    if (llm_arch_from_string("inkling2_dspark") != LLM_ARCH_INKLING2_DSPARK) {
        std::fprintf(stderr, "FAIL: inkling2_dspark arch registration broken\n");
        return 1;
    }

    if (argc < 3) {
        std::printf("PASS: inkling2_dspark arch registered (skipping full-model test; usage: %s <tgt.gguf> <dft.gguf> [ref.bin])\n", argv[0]);
        llama_backend_free();
        return 0;
    }

    const char * tgt_path = argv[1];
    const char * dft_path = argv[2];
    const char * ref_path = argc >= 4 ? argv[3] : nullptr;

    llama_model_params mparams_tgt = llama_model_default_params();
    mparams_tgt.n_gpu_layers = 99;
    mparams_tgt.use_mmap     = false;

    llama_model * model_tgt = llama_model_load_from_file(tgt_path, mparams_tgt);
    if (!model_tgt) {
        std::fprintf(stderr, "FAIL: could not load target model %s\n", tgt_path);
        return 1;
    }

    llama_model_params mparams_dft = llama_model_default_params();
    mparams_dft.n_gpu_layers = 99;
    mparams_dft.use_mmap     = false;

    llama_model * model_dft = llama_model_load_from_file(dft_path, mparams_dft);
    if (!model_dft) {
        std::fprintf(stderr, "FAIL: could not load draft model %s\n", dft_path);
        llama_model_free(model_tgt);
        return 1;
    }

    llama_context_params cparams_tgt = llama_context_default_params();
    cparams_tgt.n_ctx           = 2048;
    cparams_tgt.n_batch         = 512;
    cparams_tgt.n_ubatch        = 512;
    cparams_tgt.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams_tgt.no_perf         = false;

    llama_context * ctx_tgt = llama_init_from_model(model_tgt, cparams_tgt);
    if (!ctx_tgt) {
        std::fprintf(stderr, "FAIL: could not create target context\n");
        llama_model_free(model_dft);
        llama_model_free(model_tgt);
        return 1;
    }

    if (ref_path) {
        if (!verify_pytorch_reference(ctx_tgt, model_dft, ref_path)) {
            std::fprintf(stderr, "FAIL: PyTorch DSpark reference check failed\n");
            llama_free(ctx_tgt);
            llama_model_free(model_dft);
            llama_model_free(model_tgt);
            return 1;
        }
        std::printf("PASS: PyTorch DSpark reference parity verified!\n");
    }

    // End-to-end speculative decoding benchmark & greedy parity test
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    auto tmpls = common_chat_templates_init(model_tgt, "");
    common_chat_templates_inputs chat_in;
    common_chat_msg user_msg;
    user_msg.role    = "user";
    user_msg.content = "Write a concise Python function `merge_sorted(a, b)` that merges two sorted lists in O(n+m) time, followed by a brief complexity analysis.";
    chat_in.messages.push_back(user_msg);
    chat_in.add_generation_prompt = true;
    chat_in.use_jinja             = true;
    chat_in.reasoning_format      = COMMON_REASONING_FORMAT_AUTO;
    chat_in.chat_template_kwargs["reasoning_effort"] = "\"low\"";
    common_chat_params chat_params = common_chat_templates_apply(tmpls.get(), chat_in);
    llama_tokens prompt = common_tokenize(vocab, chat_params.prompt, false, true);
    std::printf("Formatted Inkling2 prompt (%zu tokens):\n%s\n", prompt.size(), chat_params.prompt.c_str());
    const int n_gen = 256;

    // 1. Speculative run with Inkling2 DSpark (tree speculation)
    common_params_speculative spec_params;
    spec_params.type                = COMMON_SPECULATIVE_TYPE_DRAFT;
    spec_params.draft.mparams.path  = dft_path;
    spec_params.draft.model         = model_dft;
    spec_params.draft.n_max         = 15;
    spec_params.draft.n_min         = 0;
    spec_params.draft.p_min         = 0.0f;
    spec_params.draft.cparams       = cparams_tgt;

    common_speculative * spec = common_speculative_init(spec_params, ctx_tgt);
    if (!spec) {
        std::fprintf(stderr, "FAIL: common_speculative_init returned null\n");
        return 1;
    }

    auto argmax_at = [&](int32_t i_batch) -> llama_token {
        const float * logits = llama_get_logits_ith(ctx_tgt, i_batch);
        const int32_t nv = llama_vocab_n_tokens(vocab);
        return (llama_token) std::distance(logits, std::max_element(logits, logits + nv));
    };

    llama_batch batch = llama_batch_init(512, 0, 1);

    // Warmup speculative cycles and all target verify batch sizes (1..8) to compile all Metal pipelines before timing
    {
        llama_memory_clear(llama_get_memory(ctx_tgt), false);
        common_batch_clear(batch);
        for (size_t i = 0; i < prompt.size(); ++i) {
            common_batch_add(batch, prompt[i], (llama_pos) i, { 0 }, i + 1 == prompt.size());
        }
        llama_decode(ctx_tgt, batch);
        common_speculative_process(spec, batch, 0);
        llama_tokens w_hist = prompt;
        llama_token w_last = argmax_at((int32_t) prompt.size() - 1);
        for (int w = 0; w < 2; ++w) {
            llama_tokens d = common_speculative_draft(spec, spec_params, w_hist, w_last);
            common_batch_clear(batch);
            const llama_pos p0 = (llama_pos) w_hist.size();
            common_batch_add(batch, w_last, p0, { 0 }, true);
            for (size_t i = 0; i < d.size(); ++i) {
                common_batch_add(batch, d[i], p0 + 1 + (llama_pos) i, { 0 }, true);
            }
            llama_decode(ctx_tgt, batch);
            common_speculative_process(spec, batch, 0);
            w_hist.push_back(w_last);
            w_last = argmax_at(0);
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, (llama_pos) w_hist.size(), -1);
        }
        for (int bs = 1; bs <= 12; ++bs) {
            common_batch_clear(batch);
            const llama_pos p0 = (llama_pos) w_hist.size();
            for (int i = 0; i < bs; ++i) {
                common_batch_add(batch, w_last, p0 + i, { 0 }, true);
            }
            llama_decode(ctx_tgt, batch);
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, p0, -1);
        }
    }

    // Timed speculative run
    llama_memory_clear(llama_get_memory(ctx_tgt), false);
    common_batch_clear(batch);
    for (size_t i = 0; i < prompt.size(); ++i) {
        common_batch_add(batch, prompt[i], (llama_pos) i, { 0 }, i + 1 == prompt.size());
    }
    if (llama_decode(ctx_tgt, batch) != 0 || !common_speculative_process(spec, batch, 0)) {
        std::fprintf(stderr, "FAIL: speculative prompt decode/process failed\n");
        return 1;
    }

    llama_tokens hist = prompt;
    llama_token id_last = argmax_at((int32_t) prompt.size() - 1);
    llama_tokens spec_out;
    spec_out.push_back(id_last);
    common_speculative_begin(spec, hist);

    int total_proposed = 0;
    int total_accepted = 0;
    int n_cycles = 0;
    int64_t us_draft = 0, us_tgt = 0, us_proc = 0;

    const int64_t t_spec_start = ggml_time_us();
    while ((int) spec_out.size() < n_gen && !llama_vocab_is_eog(vocab, id_last)) {
        const int64_t t0 = ggml_time_us();
        common_speculative_tree tree = common_speculative_draft_tree(spec, spec_params, hist, id_last);
        const int64_t t1 = ggml_time_us();
        us_draft += (t1 - t0);

        const int rem = n_gen - (int) spec_out.size();
        if ((int) tree.size() > rem - 1) {
            const size_t keep = (size_t) std::max(0, rem - 1);
            tree.tokens.resize(keep);
            tree.parents.resize(keep);
            tree.probs.resize(keep);
        }

        const bool is_tree = !tree.empty() && !tree.is_linear();
        common_batch_clear(batch);
        const llama_pos p0 = (llama_pos) hist.size();
        if (!is_tree) {
            common_batch_add(batch, id_last, p0, { 0 }, true);
            for (size_t i = 0; i < tree.size(); ++i) {
                common_batch_add(batch, tree.tokens[i], p0 + 1 + (llama_pos) i, { 0 }, true);
            }
        } else {
            int32_t depths[16] = {};
            int8_t  parents_full[16];
            parents_full[0] = -1;
            common_batch_add(batch, id_last, p0, { 0 }, true);
            for (size_t i = 0; i < tree.size(); ++i) {
                const int32_t p = tree.parents[i];
                depths[i + 1] = depths[p] + 1;
                parents_full[i + 1] = tree.parents[i];
                common_batch_add(batch, tree.tokens[i], p0 + depths[i + 1], { 0 }, true);
            }
            llama_set_tree_parents(ctx_tgt, parents_full, (int32_t) tree.size() + 1);
        }

        const int64_t t2 = ggml_time_us();
        if (llama_decode(ctx_tgt, batch) != 0) {
            std::fprintf(stderr, "FAIL: speculative verify decode failed\n");
            return 1;
        }
        const int64_t t3 = ggml_time_us();
        us_tgt += (t3 - t2);

        std::vector<int32_t> keep_idxs;
        llama_tokens acc_toks;
        int32_t cur_node = 0;
        keep_idxs.push_back(0);
        llama_token next_tok = LLAMA_TOKEN_NULL;
        while (true) {
            const llama_token pred = argmax_at(cur_node);
            int32_t match_node = -1;
            for (size_t i = 0; i < tree.size(); ++i) {
                if (tree.parents[i] == cur_node && tree.tokens[i] == pred) {
                    match_node = (int32_t) (i + 1);
                    break;
                }
            }
            if (match_node >= 0) {
                acc_toks.push_back(pred);
                keep_idxs.push_back(match_node);
                cur_node = match_node;
            } else {
                next_tok = pred;
                break;
            }
        }

        const int64_t t3b = ggml_time_us();
        if (is_tree) {
            if (!common_speculative_process_rows(spec, batch, keep_idxs.data(), (int32_t) keep_idxs.size(), p0, 0)) {
                std::fprintf(stderr, "FAIL: speculative process_rows failed\n");
                return 1;
            }
            if (!llama_tree_accept_path(ctx_tgt, 0, p0, keep_idxs.data(), (int32_t) keep_idxs.size())) {
                std::fprintf(stderr, "FAIL: llama_tree_accept_path failed\n");
                return 1;
            }
        } else {
            if (!common_speculative_process(spec, batch, 0)) {
                std::fprintf(stderr, "FAIL: speculative process failed\n");
                return 1;
            }
        }
        const int64_t t4 = ggml_time_us();
        us_proc += (t4 - t3b);

        const int n_acc = (int) acc_toks.size();
        total_proposed += (int) tree.size();
        total_accepted += n_acc;
        n_cycles++;
        common_speculative_accept(spec, (uint16_t) n_acc);

        hist.push_back(id_last);
        for (int i = 0; i < n_acc; ++i) {
            hist.push_back(acc_toks[i]);
            spec_out.push_back(acc_toks[i]);
            if (llama_vocab_is_eog(vocab, acc_toks[i])) {
                break;
            }
        }
        if (!llama_vocab_is_eog(vocab, spec_out.back())) {
            spec_out.push_back(next_tok);
        }
        id_last = next_tok;

        if (!is_tree) {
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, (llama_pos) hist.size(), -1);
        }
    }
    const double spec_ms   = (ggml_time_us() - t_spec_start) / 1000.0;
    const int    spec_toks = (int) spec_out.size() - 1;
    const double spec_tps  = 1000.0 * spec_toks / spec_ms;

    std::printf("\n=== Speculative Inkling2 + DSpark (warmed) ===\n");
    std::printf("Generated %d tokens in %.2f ms -> %.2f tok/s (cycles=%d, accepted=%d/%d = %.1f%%, mean_len=%.2f)\n",
            spec_toks, spec_ms, spec_tps, n_cycles, total_accepted, total_proposed,
            total_proposed > 0 ? 100.0 * total_accepted / total_proposed : 0.0,
            n_cycles > 0 ? (double) spec_toks / n_cycles : 0.0);
    std::printf("Per-cycle breakdown: tgt_verify=%.2f ms, dft_process=%.2f ms, dft_draft=%.2f ms (total=%.2f ms/cycle)\n",
            us_tgt / 1000.0 / std::max(1, n_cycles),
            us_proc / 1000.0 / std::max(1, n_cycles),
            us_draft / 1000.0 / std::max(1, n_cycles),
            spec_ms / std::max(1, n_cycles));
    std::printf("Output preview:\n%s\n", common_detokenize(ctx_tgt, spec_out, true).c_str());

    common_speculative_free(spec);

    // 2. Greedy parity check against standalone target decode (no speculation)
    for (uint32_t k = 0; k < llama_model_target_layer_ids_n(model_dft); ++k) {
        llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) llama_model_target_layer_ids(model_dft)[k], false);
    }
    llama_memory_clear(llama_get_memory(ctx_tgt), false);
    common_batch_clear(batch);
    for (size_t i = 0; i < prompt.size(); ++i) {
        common_batch_add(batch, prompt[i], (llama_pos) i, { 0 }, i + 1 == prompt.size());
    }
    if (llama_decode(ctx_tgt, batch) != 0) {
        std::fprintf(stderr, "FAIL: baseline prompt decode failed\n");
        return 1;
    }
    llama_tokens ref_out;
    llama_token cur_tok = argmax_at((int32_t) prompt.size() - 1);
    ref_out.push_back(cur_tok);
    llama_pos cur_pos = (llama_pos) prompt.size();

    // Warmup 1 single-token decode
    common_batch_clear(batch);
    common_batch_add(batch, cur_tok, cur_pos, { 0 }, true);
    llama_decode(ctx_tgt, batch);
    llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, cur_pos, -1);

    const int64_t t_base_start = ggml_time_us();
    while (ref_out.size() < spec_out.size()) {
        common_batch_clear(batch);
        common_batch_add(batch, cur_tok, cur_pos++, { 0 }, true);
        if (llama_decode(ctx_tgt, batch) != 0) {
            std::fprintf(stderr, "FAIL: baseline token decode failed\n");
            return 1;
        }
        cur_tok = argmax_at(0);
        ref_out.push_back(cur_tok);
    }
    const double base_ms  = (ggml_time_us() - t_base_start) / 1000.0;
    const double base_tps = 1000.0 * (ref_out.size() - 1) / base_ms;
    std::printf("\n=== Baseline Inkling2 (no speculation, warmed) ===\n");
    std::printf("Generated %zu tokens in %.2f ms -> %.2f tok/s (speedup = %.2fx)\n",
            ref_out.size() - 1, base_ms, base_tps, spec_tps / base_tps);

    int mismatch_at = -1;
    for (size_t i = 0; i < spec_out.size(); ++i) {
        if (spec_out[i] != ref_out[i]) {
            mismatch_at = (int) i;
            std::fprintf(stderr, "MISMATCH at token %zu: spec=%d ('%s') vs ref=%d ('%s')\n",
                    i, spec_out[i], common_token_to_piece(vocab, spec_out[i]).c_str(),
                    ref_out[i], common_token_to_piece(vocab, ref_out[i]).c_str());
            break;
        }
    }
    if (mismatch_at >= 0) {
        std::fprintf(stderr, "FAIL: speculative output diverged from greedy baseline at index %d\n", mismatch_at);
        llama_batch_free(batch);
        llama_free(ctx_tgt);
        llama_model_free(model_dft);
        llama_model_free(model_tgt);
        llama_backend_free();
        return 1;
    }
    std::printf("PASS: 100%% token-for-token greedy parity (%zu tokens) verified!\n", spec_out.size());

    llama_batch_free(batch);
    llama_free(ctx_tgt);
    llama_model_free(model_dft);
    llama_model_free(model_tgt);
    llama_backend_free();
    return 0;
}
