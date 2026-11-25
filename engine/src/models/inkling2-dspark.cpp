#include "models.h"

#include "llama-context.h"
#include "llama-kv-cache-iswa.h"

// Inkling 2 Mini DSpark draft model (`mtp.safetensors`):
//   * embd batch  -> mix target taps per layer via DFlare RMSNorm, project + NeoX-RoPE K_ctx/V_ctx into KV cache,
//                    and output f_rnn (row `n_layer` of fuse_alpha + markov_dflare_norm) via t_h_nextn.
//   * token batch -> [anchor, mask x (block_size - 1)] non-causal 3-layer transformer over context + block KV,
//                    followed by base logits + 7-step folded rnn_hidden Markov recurrence on GPU.
llm_build_inkling2_dspark::llm_build_inkling2_dspark(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params) {
    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_taps           = (int64_t) model.target_layer_ids.size();
    const int64_t rank             = hparams.dspark_markov_rank;
    const int64_t block_size       = hparams.dspark_block_size;
    const int64_t n_vocab_unpadded = hparams.n_vocab_unpadded;
    const float   kq_scale         = 1.0f / sqrtf(float(n_embd_head));

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_v());

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = (!ubatch.embd && n_outputs > 0 && n_outputs < n_tokens) ? build_inp_out_ids() : nullptr;
    llm_graph_input_attn_kv_iswa * inp_attn = build_attn_inp_kv_iswa();

    auto apply_rope_pos = [&](ggml_tensor * cur, ggml_tensor * pos_t) {
        return ggml_rope_ext(ctx0, cur, pos_t, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
    };
    auto apply_rope = [&](ggml_tensor * cur) {
        return apply_rope_pos(cur, inp_pos);
    };

    // Mode 1: Context KV injection + f_rnn computation from target feature taps
    if (ubatch.embd) {
        const int64_t n_embd_inp = hparams.n_embd_inp();
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);
        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
        ggml_set_input(inp->embd);
        ggml_tensor * inp_target = inp->embd;
        res->add_input(std::move(inp));

        std::vector<ggml_tensor *> tap_views(n_taps);
        for (int64_t k = 0; k < n_taps; ++k) {
            tap_views[k] = ggml_view_2d(ctx0, inp_target, n_embd, n_tokens,
                    inp_target->nb[1], (size_t) k * n_embd * sizeof(float));
        }

        auto mix_row = [&](int64_t row, ggml_tensor * norm_w) {
            ggml_tensor * mixed = ggml_scale(ctx0, tap_views[0], model.dspark_fuse_alpha[row][0]);
            for (int64_t k = 1; k < n_taps; ++k) {
                mixed = ggml_add(ctx0, mixed, ggml_scale(ctx0, tap_views[k], model.dspark_fuse_alpha[row][k]));
            }
            return build_norm(mixed, norm_w, nullptr, LLM_NORM_RMS, row < n_layer ? (int) row : -1);
        };

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            ggml_tensor * ctx_h = mix_row(il, layer.dspark_dflare_norm);

            ggml_tensor * Kcur = nullptr;
            ggml_tensor * Vcur = nullptr;
            if (layer.wkv_ctx && (!loras || loras->empty())) {
                ggml_tensor * KVcur = build_lora_mm(layer.wkv_ctx, ctx_h);
                const int64_t nk = layer.wk_ctx->ne[1];
                Kcur = ggml_view_3d(ctx0, KVcur, n_embd_head, n_head_kv, n_tokens, (size_t) n_embd_head * sizeof(float), KVcur->nb[1], 0);
                Vcur = ggml_view_3d(ctx0, KVcur, n_embd_head, n_head_kv, n_tokens, (size_t) n_embd_head * sizeof(float), KVcur->nb[1], (size_t) nk * sizeof(float));
            } else {
                Kcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.wk_ctx, ctx_h), n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.wv_ctx, ctx_h), n_embd_head, n_head_kv, n_tokens);
            }

            Kcur = build_norm(Kcur, layer.attn_k_ctx_norm, nullptr, LLM_NORM_RMS, il);
            Kcur = apply_rope(Kcur);

            const bool    is_swa = hparams.is_swa(il);
            const auto  * kv     = is_swa ? inp_attn->mctx->get_swa()       : inp_attn->mctx->get_base();
            ggml_tensor * k_idxs = is_swa ? inp_attn->get_k_idxs_swa()      : inp_attn->get_k_idxs();
            ggml_tensor * v_idxs = is_swa ? inp_attn->get_v_idxs_swa()      : inp_attn->get_v_idxs();

            ggml_build_forward_expand(gf, kv->cpy_k(ctx0, Kcur, k_idxs, il));
            ggml_build_forward_expand(gf, kv->cpy_v(ctx0, Vcur, v_idxs, il));
        }

        ggml_tensor * f_rnn = mix_row(n_layer, model.dspark_markov_dflare_norm);
        res->t_embd    = f_rnn;
        res->t_h_nextn = f_rnn;
        ggml_build_forward_expand(gf, f_rnn);
        return;
    }

    // Mode 2: (Optional fused tap KV injection + f_rnn) + draft block [anchor, mask x (n_blk - 1)] + folded Markov RNN head
    GGML_ASSERT(cparams.ctx_other != nullptr && "Inkling2 DSpark draft requires ctx_other (target context)");
    const llama_model * model_tgt = llama_get_model(cparams.ctx_other);
    GGML_ASSERT(model_tgt && model_tgt->tok_embd && model_tgt->tok_norm && model_tgt->output);

    const int64_t n_pend = (params.dspark_pending_n > 0 && params.dspark_pending_n < n_tokens &&
                            model.dspark_pending_taps != nullptr &&
                            inp_attn->get_k_idxs()->ne[0] == n_tokens &&
                            inp_attn->get_v_idxs()->ne[0] % n_tokens == 0)
            ? (int64_t) params.dspark_pending_n
            : 0;
    const int64_t n_blk = n_tokens - n_pend;
    ggml_tensor * pos_blk = (n_pend > 0)
            ? ggml_view_1d(ctx0, inp_pos, n_blk, (size_t) n_pend * inp_pos->nb[0])
            : inp_pos;

    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_tokens);
    res->t_inp_tokens = inp_tokens;
    {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
        inp->tokens = inp_tokens;
        res->add_input(std::move(inp));
    }

    ggml_tensor * anchor_tok = ggml_view_1d(ctx0, inp_tokens, 1, (size_t) n_pend * inp_tokens->nb[0]);
    // Expand CPU token embedding lookup BEFORE any GPU injection ops so the scheduler runs CPU get_rows first
    // and keeps all fused tap-injection + draft-block ops in a single Metal command buffer (zero mid-graph sync).
    ggml_tensor * anchor_embd_raw = ggml_get_rows(ctx0, model_tgt->tok_embd, anchor_tok);
    ggml_build_forward_expand(gf, anchor_embd_raw);

    ggml_tensor * f_rnn_fused = nullptr;
    if (n_pend > 0) {
        const int64_t n_embd_inp = hparams.n_embd_inp();
        ggml_tensor * pos_pend = ggml_view_1d(ctx0, inp_pos, n_pend, 0);
        ggml_tensor * inp_taps = ggml_view_2d(ctx0, model.dspark_pending_taps, n_embd_inp, n_pend,
                model.dspark_pending_taps->nb[1], 0);

        std::vector<ggml_tensor *> tap_views(n_taps);
        for (int64_t k = 0; k < n_taps; ++k) {
            tap_views[k] = ggml_view_2d(ctx0, inp_taps, n_embd, n_pend,
                    inp_taps->nb[1], (size_t) k * n_embd * sizeof(float));
        }

        auto mix_row_p = [&](int64_t row, ggml_tensor * norm_w) {
            ggml_tensor * mixed = ggml_scale(ctx0, tap_views[0], model.dspark_fuse_alpha[row][0]);
            for (int64_t k = 1; k < n_taps; ++k) {
                mixed = ggml_add(ctx0, mixed, ggml_scale(ctx0, tap_views[k], model.dspark_fuse_alpha[row][k]));
            }
            return build_norm(mixed, norm_w, nullptr, LLM_NORM_RMS, row < n_layer ? (int) row : -1);
        };

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];
            ggml_tensor * ctx_h = mix_row_p(il, layer.dspark_dflare_norm);

            ggml_tensor * Kcur = nullptr;
            ggml_tensor * Vcur = nullptr;
            if (layer.wkv_ctx && (!loras || loras->empty())) {
                ggml_tensor * KVcur = build_lora_mm(layer.wkv_ctx, ctx_h);
                const int64_t nk = layer.wk_ctx->ne[1];
                Kcur = ggml_view_3d(ctx0, KVcur, n_embd_head, n_head_kv, n_pend, (size_t) n_embd_head * sizeof(float), KVcur->nb[1], 0);
                Vcur = ggml_view_3d(ctx0, KVcur, n_embd_head, n_head_kv, n_pend, (size_t) n_embd_head * sizeof(float), KVcur->nb[1], (size_t) nk * sizeof(float));
            } else {
                Kcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.wk_ctx, ctx_h), n_embd_head, n_head_kv, n_pend);
                Vcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.wv_ctx, ctx_h), n_embd_head, n_head_kv, n_pend);
            }

            Kcur = build_norm(Kcur, layer.attn_k_ctx_norm, nullptr, LLM_NORM_RMS, il);
            Kcur = apply_rope_pos(Kcur, pos_pend);

            const bool    is_swa   = hparams.is_swa(il);
            const auto  * kv       = is_swa ? inp_attn->mctx->get_swa()  : inp_attn->mctx->get_base();
            ggml_tensor * k_all    = is_swa ? inp_attn->get_k_idxs_swa() : inp_attn->get_k_idxs();
            ggml_tensor * v_all    = is_swa ? inp_attn->get_v_idxs_swa() : inp_attn->get_v_idxs();
            const int64_t v_step   = v_all->ne[0] / n_tokens;
            ggml_tensor * k_idxs_p = ggml_view_1d(ctx0, k_all, n_pend, 0);
            ggml_tensor * v_idxs_p = ggml_view_1d(ctx0, v_all, n_pend * v_step, 0);

            ggml_build_forward_expand(gf, kv->cpy_k(ctx0, Kcur, k_idxs_p, il));
            ggml_build_forward_expand(gf, kv->cpy_v(ctx0, Vcur, v_idxs_p, il));
        }

        // Compute f_rnn for the last pending tap (row n_pend - 1) on GPU and mirror into dspark_f_rnn_buf
        const size_t last_off = (size_t) (n_pend - 1) * inp_taps->nb[1];
        ggml_tensor * mixed_last = ggml_scale(ctx0,
                ggml_view_2d(ctx0, inp_taps, n_embd, 1, inp_taps->nb[1], last_off),
                model.dspark_fuse_alpha[n_layer][0]);
        for (int64_t k = 1; k < n_taps; ++k) {
            mixed_last = ggml_add(ctx0, mixed_last, ggml_scale(ctx0,
                    ggml_view_2d(ctx0, inp_taps, n_embd, 1, inp_taps->nb[1], last_off + (size_t) k * n_embd * sizeof(float)),
                    model.dspark_fuse_alpha[n_layer][k]));
        }
        f_rnn_fused = build_norm(mixed_last, model.dspark_markov_dflare_norm, nullptr, LLM_NORM_RMS, -1);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_reshape_1d(ctx0, f_rnn_fused, n_embd), model.dspark_f_rnn_buf));
    }

    // Slot 0 of the draft block uses target's normalized token embedding; remaining slots use shared mask_embed.
    ggml_tensor * cur = nullptr;
    if (n_blk == 1) {
        cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, anchor_embd_raw, model_tgt->hparams.f_norm_rms_eps), model_tgt->tok_norm);
    } else {
        ggml_tensor * anchor_embd = ggml_mul(ctx0, ggml_rms_norm(ctx0, anchor_embd_raw, model_tgt->hparams.f_norm_rms_eps), model_tgt->tok_norm);

        ggml_tensor * mask_f32 = model.dspark_mask_embd->type == GGML_TYPE_F32
                ? model.dspark_mask_embd
                : ggml_cast(ctx0, model.dspark_mask_embd, GGML_TYPE_F32);
        ggml_tensor * masks = ggml_repeat_4d(ctx0,
                ggml_reshape_2d(ctx0, mask_f32, n_embd, 1),
                n_embd, n_blk - 1, 1, 1);
        cur = ggml_concat(ctx0, anchor_embd, masks, 1);
    }

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = cur;
        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);

        ggml_tensor * Qcur = nullptr;
        ggml_tensor * Kcur = nullptr;
        ggml_tensor * Vcur = nullptr;
        if (layer.wqkv && (!loras || loras->empty())) {
            ggml_tensor * QKVcur = build_lora_mm(layer.wqkv, cur);
            const int64_t nq = layer.wq->ne[1];
            const int64_t nk = layer.wk->ne[1];
            Qcur = ggml_view_3d(ctx0, QKVcur, n_embd_head, n_head,    n_blk, (size_t) n_embd_head * sizeof(float), QKVcur->nb[1], 0);
            Kcur = ggml_view_3d(ctx0, QKVcur, n_embd_head, n_head_kv, n_blk, (size_t) n_embd_head * sizeof(float), QKVcur->nb[1], (size_t) nq * sizeof(float));
            Vcur = ggml_view_3d(ctx0, QKVcur, n_embd_head, n_head_kv, n_blk, (size_t) n_embd_head * sizeof(float), QKVcur->nb[1], (size_t) (nq + nk) * sizeof(float));
        } else {
            Qcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.wq, cur), n_embd_head, n_head,    n_blk);
            Kcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.wk, cur), n_embd_head, n_head_kv, n_blk);
            Vcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.wv, cur), n_embd_head, n_head_kv, n_blk);
        }

        Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);

        Qcur = apply_rope_pos(Qcur, pos_blk);
        Kcur = apply_rope_pos(Kcur, pos_blk);

        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);

        const bool   is_swa   = hparams.is_swa(il);
        const auto * mctx_cur = is_swa ? inp_attn->mctx->get_swa() : inp_attn->mctx->get_base();
        ggml_tensor * k_all   = is_swa ? inp_attn->get_k_idxs_swa() : inp_attn->get_k_idxs();
        ggml_tensor * v_all   = is_swa ? inp_attn->get_v_idxs_swa() : inp_attn->get_v_idxs();
        const int64_t v_step  = v_all->ne[0] / n_tokens;
        ggml_tensor * k_idx_b = (n_pend > 0) ? ggml_view_1d(ctx0, k_all, n_blk,          (size_t) n_pend * k_all->nb[0]) : k_all;
        ggml_tensor * v_idx_b = (n_pend > 0) ? ggml_view_1d(ctx0, v_all, n_blk * v_step, (size_t) (n_pend * v_step) * v_all->nb[0]) : v_all;

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, k_idx_b, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur, v_idx_b, il));

        ggml_tensor * kq_mask = is_swa ? inp_attn->get_kq_mask_swa() : inp_attn->get_kq_mask();
        const int64_t max_block = std::max<int64_t>(block_size, 16);
        if (is_swa && n_blk > 1 && n_blk <= max_block) {
            // All queries in the draft block share the anchor's window [anchor_pos - sliding_window .. block_end]
            ggml_tensor * row_a = ggml_view_4d(ctx0, kq_mask,
                    kq_mask->ne[0], 1, kq_mask->ne[2], kq_mask->ne[3],
                    kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3],
                    (size_t) n_pend * kq_mask->nb[1]);
            kq_mask = ggml_repeat_4d(ctx0, row_a, kq_mask->ne[0], n_blk, kq_mask->ne[2], kq_mask->ne[3]);
        } else if (n_pend > 0) {
            kq_mask = ggml_view_4d(ctx0, kq_mask,
                    kq_mask->ne[0], n_blk, kq_mask->ne[2], kq_mask->ne[3],
                    kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3],
                    (size_t) n_pend * kq_mask->nb[1]);
        }

        ggml_tensor * attn_out = build_attn_mha(
                Qcur, mctx_cur->get_k(ctx0, il), mctx_cur->get_v(ctx0, il),
                nullptr, kq_mask, nullptr, nullptr, kq_scale, il);
        attn_out = build_lora_mm(layer.wo, attn_out);

        residual = ggml_add(ctx0, residual, attn_out);

        cur = build_norm(residual, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        if (layer.ffn_gate_up && (!loras || loras->empty())) {
            ggml_tensor * gu = build_lora_mm(layer.ffn_gate_up, cur);
            const int64_t n_ff = layer.ffn_gate->ne[1];
            ggml_tensor * gate = ggml_view_2d(ctx0, gu, n_ff, n_blk, gu->nb[1], 0);
            ggml_tensor * up   = ggml_view_2d(ctx0, gu, n_ff, n_blk, gu->nb[1], (size_t) n_ff * sizeof(float));
            cur = ggml_swiglu_split(ctx0, gate, up);
            cur = build_lora_mm(layer.ffn_down, cur);
        } else {
            cur = build_ffn(cur,
                    layer.ffn_up,   nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        }
        cur = ggml_add(ctx0, residual, cur);
    }

    ggml_tensor * head_hidden = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    res->t_embd = head_hidden;

    ggml_tensor * lm_head_v = ggml_view_2d(ctx0, model_tgt->output,
            n_embd, n_vocab_unpadded, model_tgt->output->nb[1], 0);

    // Reserve / non-block fallback (e.g. sched_reserve with bs=1 or bs=512)
    const int64_t max_block = std::max<int64_t>(block_size, 16);
    if (n_blk < 2 || n_blk > max_block || model.dspark_fold_w == nullptr) {
        ggml_tensor * logits = ggml_mul_mat(ctx0, lm_head_v, head_hidden);
        ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
        if (hparams.f_final_logit_softcapping != 0.0f) {
            const float cap = hparams.f_final_logit_softcapping;
            logits = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, logits, 1.0f / cap)), cap);
        }
        res->t_logits = logits;
        ggml_build_forward_expand(gf, logits);
        return;
    }

    // Active block drafting: slots 1..n_draft predict tokens d_1..d_{n_draft}
    const int64_t n_draft = n_blk - 1;
    ggml_tensor * sample_hidden = ggml_view_2d(ctx0, head_hidden, n_embd, n_draft, head_hidden->nb[1], head_hidden->nb[1]);

    const bool want_logits = n_outputs > 0;

    // with a draft vocabulary the head reads only the active populated rows (padded to 1024), and the drafted ids are
    // slots; callers requesting full logits keep the full head
    const bool use_vocab = model.dspark_vocab_lm != nullptr && !want_logits;
    model.dspark_vocab_drafted = use_vocab;
    const int64_t n_cand = use_vocab
            ? std::min<int64_t>(model.dspark_vocab_lm->ne[1],
                                std::max<int64_t>(4096, GGML_PAD((int64_t) model.dspark_vocab_ids.size(), 1024)))
            : n_vocab_unpadded;
    ggml_tensor * head_w = use_vocab
            ? ggml_view_2d(ctx0, model.dspark_vocab_lm,     n_embd, n_cand, model.dspark_vocab_lm->nb[1],     0)
            : lm_head_v;
    ggml_tensor * fold_w = use_vocab
            ? ggml_view_2d(ctx0, model.dspark_vocab_fold_w, rank,   n_cand, model.dspark_vocab_fold_w->nb[1], 0)
            : model.dspark_fold_w;
    ggml_tensor * fold_b = use_vocab
            ? ggml_view_1d(ctx0, model.dspark_vocab_fold_b, n_cand, 0)
            : model.dspark_fold_b;

    ggml_tensor * base_logits = ggml_mul_mat(ctx0, head_w, sample_hidden); // [n_cand, n_draft]
    ggml_mul_mat_set_prec(base_logits, GGML_PREC_F32);

    // Precompute rnn_proj context term [3*rank, n_draft] for all draft steps at once
    ggml_tensor * f_rnn_src = f_rnn_fused ? f_rnn_fused : ggml_reshape_2d(ctx0, model.dspark_f_rnn_buf, n_embd, 1);
    ggml_tensor * f_rnn_rep = ggml_repeat_4d(ctx0, f_rnn_src, n_embd, n_draft, 1, 1);
    ggml_tensor * ctx_in  = ggml_concat(ctx0, sample_hidden, f_rnn_rep, 0); // [2*n_embd, n_draft]
    ggml_tensor * rnn_ctx = ggml_mul_mat(ctx0, model.dspark_rnn_w_ctx, ctx_in);
    ggml_mul_mat_set_prec(rnn_ctx, GGML_PREC_F32);
    rnn_ctx = ggml_add(ctx0, rnn_ctx, model.dspark_rnn_proj_b); // [3*rank, n_draft]

    ggml_build_forward_expand(gf, base_logits);
    ggml_build_forward_expand(gf, rnn_ctx);

    ggml_tensor * state = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, rank, 1), 0.0f);
    ggml_tensor * prev  = anchor_tok;

    ggml_tensor * logits_cat = nullptr;
    if (want_logits) {
        // Slot 0 (anchor) placeholder column so batch index `1 + s` maps to `logits_cat` col `1 + s`
        logits_cat = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_vocab_unpadded, 1), 0.0f);
    }

    const float cap = hparams.f_final_logit_softcapping;

    std::vector<ggml_tensor *> ctx_s_arr((size_t) n_draft, nullptr);
    std::vector<ggml_tensor *> base_s_arr((size_t) n_draft, nullptr);
    ggml_tensor * state_after[5] = { nullptr, nullptr, nullptr, nullptr, nullptr };

    auto build_markov_step = [&](int64_t s_pos, ggml_tensor * st_in, ggml_tensor * prev_in,
                                 int64_t dst_slot, bool prev_is_vocab_slot, bool want_next_state,
                                 ggml_tensor ** out_cpy, ggml_tensor ** out_col) -> ggml_tensor * {
        ggml_tensor * ctx_s = ctx_s_arr[(size_t) s_pos];
        if (!ctx_s) {
            ctx_s = ggml_view_2d(ctx0, rnn_ctx, 3 * rank, 1, rnn_ctx->nb[1], (size_t) s_pos * rnn_ctx->nb[1]);
            ggml_build_forward_expand(gf, ctx_s);
            ctx_s_arr[(size_t) s_pos] = ctx_s;
        }
        ggml_tensor * base_s = base_s_arr[(size_t) s_pos];
        if (!base_s) {
            base_s = ggml_view_2d(ctx0, base_logits, n_cand, 1,
                    base_logits->nb[1], (size_t) s_pos * base_logits->nb[1]);
            base_s_arr[(size_t) s_pos] = base_s;
        }

        ggml_tensor * w1_prev = ggml_get_rows(ctx0, prev_is_vocab_slot ? model.dspark_vocab_w1 : model.dspark_markov_w1, prev_in); // [rank, 1]
        ggml_tensor * step_in = ggml_concat(ctx0, st_in, w1_prev, 0); // [2*rank, 1]
        ggml_tensor * step_mm = ggml_mul_mat(ctx0, model.dspark_rnn_w_step, step_in);
        ggml_mul_mat_set_prec(step_mm, GGML_PREC_F32);

        ggml_tensor * raw = ggml_add(ctx0, ctx_s, step_mm); // [3*rank, 1]

        auto chunk = [&](int64_t k) {
            return ggml_view_2d(ctx0, raw, rank, 1, raw->nb[1], (size_t) k * rank * sizeof(float));
        };

        ggml_tensor * latent = ggml_tanh(ctx0, chunk(2));
        ggml_build_forward_expand(gf, latent);

        ggml_tensor * st_next = st_in;
        if (want_next_state) {
            ggml_tensor * gate = ggml_sigmoid(ctx0, chunk(0));
            ggml_tensor * cand = ggml_tanh(ctx0, chunk(1));
            // state = gate * state + (1 - gate) * cand = cand + gate * (state - cand)
            st_next = ggml_add(ctx0, cand, ggml_mul(ctx0, gate, ggml_sub(ctx0, st_in, cand)));
            ggml_build_forward_expand(gf, st_next);
        }

        ggml_tensor * bias = ggml_mul_mat(ctx0, fold_w, latent); // [n_cand, 1]
        ggml_mul_mat_set_prec(bias, GGML_PREC_F32);
        bias = ggml_add(ctx0, bias, fold_b);

        ggml_tensor * col = ggml_add(ctx0, bias, base_s);
        // cap * tanh(x / cap) is strictly monotonic increasing for cap > 0, so argmax is invariant when logits are not materialized
        if (want_logits && cap != 0.0f) {
            col = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, col, 1.0f / cap)), cap);
        }

        ggml_tensor * tok_s = ggml_argmax(ctx0, col); // [1] I32
        ggml_tensor * dst_s = ggml_view_1d(ctx0, model.dspark_draft_ids_buf, 1, (size_t) dst_slot * sizeof(int32_t));
        ggml_tensor * cpy_s = ggml_cpy(ctx0, tok_s, dst_s);
        ggml_build_forward_expand(gf, cpy_s);

        if (out_cpy) { *out_cpy = cpy_s; }
        if (out_col) { *out_col = col; }
        return st_next;
    };

    for (int64_t s = 0; s < n_draft; ++s) {
        ggml_tensor * cpy_s = nullptr;
        ggml_tensor * col_s = nullptr;
        state = build_markov_step(s, state, prev, s, use_vocab && s > 0, s + 1 < n_draft, &cpy_s, &col_s);
        if (s < 5) {
            state_after[s] = state;
        }
        prev = cpy_s;

        if (want_logits) {
            logits_cat = ggml_concat(ctx0, logits_cat, col_s, 1);
        }
    }

    // High-value candidate tree rollouts (GPU-gated on branch cum_p >= alt_min; pruned to top-utility branches):
    //   Slots  0..14: Trunk d1..d15 (n_draft in 7..15)
    //   Slots 15..19: Alt-A d1..d5  (rank 2 at step 0, state_after[0]) -> slot 16 also writes top2 leaf -> slot 39
    //   Slot  21:     Alt-B d1      (rank 3 at step 0 leaf, written free at s=0)
    //   Slots 25..28: Alt-C d2..d5  (rank 2 at step 1, state_after[1])
    //   Slots 30..32: Alt-D d3..d5  (rank 2 at step 2, state_after[2])
    //   Slots 34..35: Alt-E d4..d5  (rank 2 at step 3, state_after[3])
    //   Slots 37,39,42: Free 1-token leaf alternatives (written during trunk/Alt-A)
    if (!want_logits && n_draft >= 7 && n_draft <= 15 && ggml_nelements(model.dspark_draft_ids_buf) == 152) {
        auto slot_view = [&](int64_t slot) {
            return ggml_view_1d(ctx0, model.dspark_draft_ids_buf, 1, (size_t) slot * sizeof(int32_t));
        };
        ggml_tensor * cpy_tmp = nullptr;

        // Alt-A: from top2(s=0) at slot 15 -> slots 16, 17, 18, 19
        ggml_tensor * st_a2 = build_markov_step(1, state_after[0], slot_view(15), 16, use_vocab, true,  &cpy_tmp, nullptr);
        ggml_tensor * st_a3 = build_markov_step(2, st_a2,          cpy_tmp,       17, use_vocab, true,  &cpy_tmp, nullptr);
        ggml_tensor * st_a4 = build_markov_step(3, st_a3,          cpy_tmp,       18, use_vocab, true,  &cpy_tmp, nullptr);
                              build_markov_step(4, st_a4,          cpy_tmp,       19, use_vocab, false, &cpy_tmp, nullptr);

        // Alt-C: from top2(s=1) at slot 25 -> slots 26, 27, 28
        ggml_tensor * st_c3 = build_markov_step(2, state_after[1], slot_view(25), 26, use_vocab, true,  &cpy_tmp, nullptr);
        ggml_tensor * st_c4 = build_markov_step(3, st_c3,          cpy_tmp,       27, use_vocab, true,  &cpy_tmp, nullptr);
                              build_markov_step(4, st_c4,          cpy_tmp,       28, use_vocab, false, &cpy_tmp, nullptr);

        // Alt-D: from top2(s=2) at slot 30 -> slots 31, 32
        ggml_tensor * st_d4 = build_markov_step(3, state_after[2], slot_view(30), 31, use_vocab, true,  &cpy_tmp, nullptr);
                              build_markov_step(4, st_d4,          cpy_tmp,       32, use_vocab, false, &cpy_tmp, nullptr);

        // Alt-E: from top2(s=3) at slot 34 -> slot 35
                              build_markov_step(4, state_after[3], slot_view(34), 35, use_vocab, false, &cpy_tmp, nullptr);
    }

    if (want_logits) {
        if (inp_out_ids) {
            logits_cat = ggml_get_rows(ctx0, logits_cat, inp_out_ids);
        }
        res->t_logits = logits_cat;
        ggml_build_forward_expand(gf, logits_cat);
    }
}
