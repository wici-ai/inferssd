#include "models.h"

#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

static ggml_tensor * dflash_mul_mat_aux(
        ggml_context * ctx, ggml_tensor * cur, ggml_tensor * rot) {
    const auto n = rot->ne[0];
    ggml_tensor * res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur) / n);
    res = ggml_mul_mat(ctx, rot, res);
    return ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
}

llm_build_dflash::llm_build_dflash(
        const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            build_encoder(model);
            break;
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            build_decoder(model);
            break;
        default:
            GGML_ABORT("invalid DFlash graph type");
    }
}

void llm_build_dflash::build_encoder(const llama_model & model) {
    auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());
    inp->embd = ggml_new_tensor_2d(
        ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * cur = inp->embd;
    cb(cur, "inp_embd", -1);
    res->add_input(std::move(inp));

    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);
    cur = build_norm(cur, model.output_norm_enc, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "enc_norm_out", -1);

    res->t_h_nextn = cur;
    ggml_build_forward_expand(gf, cur);
}

static void build_dspark_markov_head(
        llm_graph_context & graph, const llama_model & model, ggml_tensor * tokens) {
    ggml_context * ctx0 = graph.ctx0;
    auto * res = graph.res;

    GGML_ASSERT(model.dspark_markov_w1 && model.dspark_markov_w2 && model.dspark_conf_proj);

    ggml_tensor * base = res->t_logits;
    const int64_t n_vocab = base->ne[0];
    const int64_t n_tok   = base->ne[1];

    const auto it = model.gguf_kv.find("dflash.block_size");
    GGML_ASSERT(it != model.gguf_kv.end() && "DSpark requires dflash.block_size");
    const int64_t block_size = std::stoi(it->second);
    GGML_ASSERT(block_size > 0);

    const int64_t n_blocks = graph.ubatch.n_seqs_unq;
    GGML_ASSERT(n_blocks > 0 && n_tok % n_blocks == 0);
    const int64_t block_drafts = n_tok / n_blocks;
    if (block_drafts > block_size) {
        return;
    }

    const size_t token_stride = (size_t) block_drafts * tokens->nb[0];
    const size_t base_stride  = (size_t) block_drafts * base->nb[1];
    ggml_tensor * prev = ggml_view_2d(ctx0, tokens, 1, n_blocks, token_stride, 0);
    prev = ggml_cont_1d(ctx0, prev, n_blocks);

    ggml_tensor * conf_inp = res->t_embd;
    ggml_tensor * cat = nullptr;
    ggml_tensor * cat_conf = nullptr;

    for (int64_t i = 0; i < block_drafts; ++i) {
        ggml_tensor * w1_prev = ggml_get_rows(ctx0, model.dspark_markov_w1, prev);
        ggml_tensor * bias = ggml_mul_mat(ctx0, model.dspark_markov_w2, w1_prev);
        ggml_tensor * base_i = ggml_view_2d(
            ctx0, base, n_vocab, n_blocks, base_stride, i * base->nb[1]);
        ggml_tensor * col = ggml_add(ctx0, base_i, bias);
        cat = cat ? ggml_concat(ctx0, cat, col, 1) : col;

        ggml_tensor * conf_i = ggml_view_2d(
            ctx0, conf_inp, conf_inp->ne[0], n_blocks,
            (size_t) block_drafts * conf_inp->nb[1], i * conf_inp->nb[1]);
        ggml_tensor * feat = ggml_concat(ctx0, ggml_cont(ctx0, conf_i), w1_prev, 0);
        ggml_tensor * conf = ggml_mul_mat(ctx0, model.dspark_conf_proj, feat);
        if (model.dspark_conf_proj_b) {
            conf = ggml_add(ctx0, conf, model.dspark_conf_proj_b);
        }
        conf = ggml_sigmoid(ctx0, conf);
        cat_conf = cat_conf ? ggml_concat(ctx0, cat_conf, conf, 1) : conf;

        if (i + 1 < block_drafts) {
            prev = ggml_argmax(ctx0, col);
        }
    }

    ggml_tensor * out = ggml_reshape_3d(ctx0, cat, n_vocab, n_blocks, block_drafts);
    out = ggml_cont(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3));
    out = ggml_reshape_2d(ctx0, out, n_vocab, n_tok);

    ggml_tensor * conf = ggml_reshape_3d(ctx0, cat_conf, 1, n_blocks, block_drafts);
    conf = ggml_cont(ctx0, ggml_permute(ctx0, conf, 0, 2, 1, 3));
    conf = ggml_reshape_2d(ctx0, conf, 1, n_tok);
    conf = ggml_repeat(ctx0, conf, res->t_embd);

    res->t_h_nextn = conf;
    res->t_logits  = out;
    ggml_build_forward_expand(graph.gf, conf);
    ggml_build_forward_expand(graph.gf, out);
}

void llm_build_dflash::build_decoder(const llama_model & model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * inp_pos = build_inp_pos();
    const bool use_iswa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;

    llm_graph_input_attn_kv      * inp_attn      = nullptr;
    llm_graph_input_attn_kv_iswa * inp_attn_iswa = nullptr;
    if (use_iswa) {
        inp_attn_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->embd);
        ggml_tensor * inp_g = inp->embd;
        res->add_input(std::move(inp));

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];
            ggml_tensor * k = build_lora_mm(layer.wk, inp_g);
            ggml_tensor * v = build_lora_mm(layer.wv, inp_g);
            k = ggml_reshape_3d(ctx0, k, n_embd_head, n_head_kv, n_tokens);
            v = ggml_reshape_3d(ctx0, v, n_embd_head, n_head_kv, n_tokens);
            k = build_norm(k, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
            k = ggml_rope_ext(
                ctx0, k, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

            if (use_iswa) {
                const bool is_swa = hparams.is_swa(il);
                const auto * kv = is_swa ? inp_attn_iswa->mctx->get_swa() : inp_attn_iswa->mctx->get_base();
                ggml_tensor * k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa() : inp_attn_iswa->get_k_idxs();
                ggml_tensor * v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa() : inp_attn_iswa->get_v_idxs();
                ggml_tensor * k_rot = is_swa ? inp_attn_iswa->self_k_rot_swa : inp_attn_iswa->self_k_rot;
                ggml_tensor * v_rot = is_swa ? inp_attn_iswa->self_v_rot_swa : inp_attn_iswa->self_v_rot;
                if (k_rot) {
                    k = dflash_mul_mat_aux(ctx0, k, k_rot);
                }
                if (v_rot) {
                    v = dflash_mul_mat_aux(ctx0, v, v_rot);
                }
                ggml_build_forward_expand(gf, kv->cpy_k(ctx0, k, k_idxs, il));
                ggml_build_forward_expand(gf, kv->cpy_v(ctx0, v, v_idxs, il));
            } else {
                if (inp_attn->self_k_rot) {
                    k = dflash_mul_mat_aux(ctx0, k, inp_attn->self_k_rot);
                }
                if (inp_attn->self_v_rot) {
                    v = dflash_mul_mat_aux(ctx0, v, inp_attn->self_v_rot);
                }
                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, k, inp_attn->get_k_idxs(), il));
                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, v, inp_attn->get_v_idxs(), il));
            }
        }

        res->t_embd = inp_g;
        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    GGML_ASSERT(cparams.ctx_other != nullptr);
    const llama_model * target = llama_get_model(cparams.ctx_other);
    GGML_ASSERT(target && target->tok_embd && target->output);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    ggml_tensor * inp_tokens = inp->tokens;
    ggml_tensor * inp_l = ggml_get_rows(ctx0, target->tok_embd, inp_tokens);
    res->add_input(std::move(inp));

    const float kq_scale = 1.0f / sqrtf((float) n_embd_head);
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * norm = build_norm(inp_l, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        ggml_tensor * q = build_lora_mm(layer.wq, norm);
        ggml_tensor * k = build_lora_mm(layer.wk, norm);
        ggml_tensor * v = build_lora_mm(layer.wv, norm);
        q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head,    n_tokens);
        k = ggml_reshape_3d(ctx0, k, n_embd_head, n_head_kv, n_tokens);
        v = ggml_reshape_3d(ctx0, v, n_embd_head, n_head_kv, n_tokens);
        q = build_norm(q, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        k = build_norm(k, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        k = ggml_rope_ext(ctx0, k, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

        ggml_tensor * cur = use_iswa ?
            build_attn(inp_attn_iswa, layer.wo, nullptr, nullptr, q, k, v,
                nullptr, nullptr, nullptr, kq_scale, il) :
            build_attn(inp_attn, layer.wo, nullptr, nullptr, q, k, v,
                nullptr, nullptr, nullptr, kq_scale, il);
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inp_l);
        cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cur = build_ffn(cur,
            layer.ffn_up, nullptr, nullptr,
            layer.ffn_gate, nullptr, nullptr,
            layer.ffn_down, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        inp_l = ggml_add(ctx0, cur, ffn_inp);
    }

    ggml_tensor * hidden = build_norm(inp_l, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    res->t_embd = hidden;
    res->t_logits = build_lora_mm(target->output, hidden);
    ggml_build_forward_expand(gf, res->t_logits);

    if (model.dspark_markov_w1) {
        build_dspark_markov_head(*this, model, inp_tokens);
    }
}
