#include "models.h"

#include "../llama-memory-hybrid-iswa.h"
#include "../llama-memory-recurrent.h"
#include "../llama-kv-cache-iswa.h"

// scale each row down to an L2 norm of at most threshold, after the optional per-channel scale:
// x*min(1, t/|x|) = t*x/max(|x|, t), which is the L2 norm with eps = t
// returns x/max(|x|, t): the caller applies the factor t, or folds it into a later scale
ggml_tensor * llm_build_inkling2::clip(ggml_tensor * cur, ggml_tensor * scale, float threshold) {
    if (scale) {
        cur = ggml_mul(ctx0, cur, scale);
    }

    return ggml_l2_norm(ctx0, cur, threshold);
}

// the RMS norm with eps 1e-6 of the rows of x, computed once per tensor
ggml_tensor * llm_build_inkling2::unit_rms(ggml_tensor * x) {
    auto it = rms_cache.find(x);
    if (it != rms_cache.end()) {
        return it->second;
    }

    return rms_cache[x] = ggml_rms_norm(ctx0, x, 1e-6f);
}

// attention over the depth: mix the earlier block outputs and cur with learned queries
// the RMS norm is per state and the mix does not depend on the order of the states, so the normalized blocks
// are concatenated once and shared by every mix that sees them
ggml_tensor * llm_build_inkling2::mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query) {
    ggml_tensor * cur_norm = unit_rms(cur);
    if (blocks.empty() && query->ne[1] == 1) {
        return ggml_reshape_3d(ctx0, cur_norm, n_embd, n_tokens, 1);
    }

    ggml_tensor * bank = nullptr;
    if (!blocks.empty()) {
        size_t n_known = blocks.size();
        while (n_known > 0 && !mix_banks.count(std::vector<ggml_tensor *>(blocks.begin(), blocks.begin() + n_known))) {
            --n_known;
        }
        bank = n_known ? mix_banks[std::vector<ggml_tensor *>(blocks.begin(), blocks.begin() + n_known)] : nullptr;
        for (size_t i = n_known; i < blocks.size(); ++i) {
            ggml_tensor * normed = ggml_reshape_3d(ctx0, unit_rms(blocks[i]), n_embd, 1, n_tokens);
            bank = bank ? ggml_concat(ctx0, bank, normed, 1) : normed;
            mix_banks[std::vector<ggml_tensor *>(blocks.begin(), blocks.begin() + i + 1)] = bank;
        }
    }

    return ggml_inkling_mix(ctx0, cur_norm, query, bank);
}

// causal short convolution with a residual; dynamic adds input-dependent taps per head
// the last n_conv - 1 inputs of each sequence live in the recurrent state of slot il at channel offset `offset`
// returns scale * (cur + conv(cur, weight, dynamic)): the caller passes scale = 1 when folding conv_scale into a following norm
ggml_tensor * llm_build_inkling2::convolve(
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * dynamic,
        ggml_tensor * history,
        int64_t offset,
        float scale) {
    const int64_t n_conv     = hparams.n_shortconv_l_cache;
    const int64_t n_channels = cur->ne[0];
    const int64_t n_seqs     = ubatch.n_seqs;

    ggml_tensor * past = ggml_view_3d(ctx0, history, n_conv - 1, n_channels, n_seqs, history->nb[1], history->nb[2], offset*history->nb[1]);

    ggml_tensor * res = ggml_inkling_sconv(ctx0, cur, past, weight, dynamic, scale);
    if (tree_parents_packed != 0) {
        res->op_params[1] = (int32_t) (tree_parents_packed & 0xffffffffu);
        res->op_params[2] = (int32_t) (tree_parents_packed >> 32);
    }
    if (tree_inp) {
        tree_inp->sconv_nodes.push_back(res);
    }
    return res;
}

float llm_build_inkling2::conv_scale(bool dynamic) const {
    return 1.0f/sqrtf((dynamic ? 2 : 1)*hparams.n_shortconv_l_cache + 1);
}

// mix a cross-pass message into a looped layer input: (1 - forget)*cur + gain*message, both gated on cur and message
// message_norm is the RMS norm of message
ggml_tensor * llm_build_inkling2::control(ggml_tensor * cur, ggml_tensor * message, ggml_tensor * message_norm,
        const llama_layer & layer, int ctrl) {
    ggml_tensor * gate = ggml_concat(ctx0, unit_rms(cur), message_norm, 0);
    gate = loras->empty() && ctrl == LOOP_CTRL_FFN && layer.loop_ctrl_down_ffn
        ? ggml_mul_mat(ctx0, layer.loop_ctrl_down_ffn, gate)
        : build_lora_mm(layer.loop_ctrl_down[ctrl], gate);

    if (loras->empty() && ctrl == LOOP_CTRL_FFN && layer.loop_ctrl_up_ffn) {
        ggml_tensor * gf_raw = ggml_mul_mat(ctx0, layer.loop_ctrl_up_ffn, gate);
        return ggml_inkling_ctrl_blend(ctx0, cur, message, gf_raw);
    }

    ggml_tensor * gain   = ggml_tanh(ctx0, build_lora_mm(layer.loop_ctrl_gain[ctrl],   gate));
    ggml_tensor * forget = ggml_tanh(ctx0, build_lora_mm(layer.loop_ctrl_forget[ctrl], gate));

    return ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, forget, cur)), ggml_mul(ctx0, gain, message));
}

// attention of cache slot il with a per-head bias gathered by the distance from the query to each cell
// rel_bias: [extent + 1, n_head, n_tokens], added to the KQ mask after the KQ scale
ggml_tensor * llm_build_inkling2::attention(
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * rel_bias,
        ggml_tensor * sinks,
        float kq_scale,
        int il) {
    const bool is_swa = hparams.is_swa(il);

    auto * inp_attn = inp->get_attn();

    ggml_build_forward_expand(gf, q_cur);

    const auto * mctx_cur = is_swa ? inp_attn->mctx->get_swa() : inp_attn->mctx->get_base();

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, is_swa ? inp_attn->get_k_idxs_swa() : inp_attn->get_k_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, is_swa ? inp_attn->get_v_idxs_swa() : inp_attn->get_v_idxs(), il));

    ggml_tensor * mask    = is_swa ? inp_attn->get_kq_mask_swa() : inp_attn->get_kq_mask(); // [n_kv, n_tps, 1, n_stream]
    ggml_tensor * rel_pos = is_swa ? inp->self_rel_pos_swa       : inp->self_rel_pos;        // [n_kv, n_tokens]

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q_cur, k, v, nullptr, mask, sinks, nullptr, 0, kq_scale, il, rel_bias, rel_pos);
    cb(cur, "kqv_out", il);

    return cur;
}

ggml_tensor * llm_build_inkling2::build_layer(int il, int ip, ggml_tensor * cur, std::vector<ggml_tensor *> & blocks,
        ggml_tensor * messages, ggml_tensor * messages_norm) {
    const auto & layer = model.layers[il];

    const bool    is_swa       = hparams.is_swa(il);
    const int64_t n_head       = hparams.n_head(il);
    const int64_t n_head_kv    = hparams.n_head_kv(il);
    const int64_t n_embd_head  = hparams.n_embd_head_k(il);
    const int64_t n_embd_k_gqa = n_embd_head*n_head_kv;

    // the message of controller ctrl, [n_embd, n_tokens]
    auto message = [&](ggml_tensor * msgs, int ctrl) {
        return ggml_view_2d(ctx0, msgs, n_embd, n_tokens, msgs->nb[1], (size_t) ctrl * msgs->nb[2]);
    };

    ggml_tensor * state   = recr->mctx->get_r_l(il);
    ggml_tensor * history = nullptr;
    if (ubatch.n_seqs == 1 && recr->mctx->get_n_rs() == 1 && recr->mctx->get_rs_z() < 0 &&
        recr->mctx->s_copy(0) == (int32_t) recr->mctx->get_head()) {
        // Zero-copy single-sequence decode view: skips 48x GET_ROWS f32 (19968x1x1); sconv_state updates in-place safely.
        const size_t row_size = (size_t) hparams.n_embd_r() * ggml_element_size(state);
        const size_t offs     = (size_t) recr->mctx->get_head() * row_size;
        history = ggml_view_2d(ctx0, state, hparams.n_embd_r(), 1, row_size, offs);
    } else {
        history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
    }
    history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);

    ggml_tensor * inpSA = cur;

    // attention inputs: SWA layers mix one, full-attention layers one each for Q, K and V
    ggml_tensor * inputs_3  = nullptr;
    ggml_tensor * inputs[3] = {cur, cur, cur};
    if (layer.attn_res_query) {
        ggml_tensor * mixed = mix(blocks, cur, layer.attn_res_query);
        if (mixed->ne[2] == 1) {
            inputs[0] = inputs[1] = inputs[2] = mixed;
        } else {
            inputs_3 = mixed;
            for (int j = 0; j < 3; ++j) {
                inputs[j] = ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[1], (size_t) j * mixed->nb[2]);
            }
        }
    }

    const bool fast_qkv_ctrl = messages && loras->empty() && layer.loop_ctrl_down_qkv && layer.loop_ctrl_up_qkv &&
                               (is_swa || layer.attn_qkv_inp_norm);
    if (fast_qkv_ctrl) {
        ggml_tensor * msgs_qkv      = ggml_view_3d(ctx0, messages,      n_embd, n_tokens, 3, messages->nb[1],      messages->nb[2],      0);
        ggml_tensor * msgs_norm_qkv = ggml_view_3d(ctx0, messages_norm, n_embd, n_tokens, 3, messages_norm->nb[1], messages_norm->nb[2], 0);
        // Expand msgs_norm_qkv before cur_rms_qkv so REPEAT -> CONCAT -> MUL_MAT stays contiguous for Metal fusion
        ggml_build_forward_expand(gf, msgs_norm_qkv);
        ggml_tensor * cur_qkv       = inputs_3 ? inputs_3 : ggml_reshape_3d(ctx0, inputs[0], n_embd, n_tokens, 1);
        ggml_tensor * cur_rms_qkv   = inputs_3
                ? ggml_rms_norm(ctx0, inputs_3, 1e-6f)
                : ggml_repeat(ctx0, ggml_reshape_3d(ctx0, unit_rms(inputs[0]), n_embd, n_tokens, 1), msgs_norm_qkv);
        ggml_tensor * gate_qkv      = ggml_concat(ctx0, cur_rms_qkv, msgs_norm_qkv, 0);
        gate_qkv                    = ggml_mul_mat(ctx0, layer.loop_ctrl_down_qkv, gate_qkv);
        ggml_tensor * gf_qkv        = ggml_mul_mat(ctx0, layer.loop_ctrl_up_qkv,   gate_qkv);
        inputs_3                    = ggml_inkling_ctrl_blend(ctx0, cur_qkv, msgs_qkv, gf_qkv);

        if (is_swa) {
            inputs_3 = build_norm(inputs_3, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        } else {
            inputs_3 = ggml_mul(ctx0, ggml_rms_norm(ctx0, inputs_3, hparams.f_norm_rms_eps), layer.attn_qkv_inp_norm);
        }
        for (int j = 0; j < 3; ++j) {
            inputs[j] = ggml_view_2d(ctx0, inputs_3, n_embd, n_tokens, inputs_3->nb[1], (size_t) j * inputs_3->nb[2]);
        }
    } else {
        if (messages) {
            // the messages make the Q, K and V inputs of a SWA layer differ
            for (int j = 0; j < 3; ++j) {
                inputs[j] = control(inputs[j], message(messages, LOOP_CTRL_Q + j), message(messages_norm, LOOP_CTRL_Q + j),
                        layer, LOOP_CTRL_Q + j);
            }
            inputs_3 = nullptr;
        }

        if (is_swa && messages) {
            for (int j = 0; j < 3; ++j) {
                inputs[j] = build_norm(inputs[j], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
            }
        } else if (is_swa) {
            inputs[0] = build_norm(inputs[0], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
            inputs[1] = inputs[2] = inputs[0];
        } else if (inputs_3 && layer.attn_qkv_inp_norm) {
            inputs_3 = ggml_mul(ctx0, ggml_rms_norm(ctx0, inputs_3, hparams.f_norm_rms_eps), layer.attn_qkv_inp_norm);
            for (int j = 0; j < 3; ++j) {
                inputs[j] = ggml_view_2d(ctx0, inputs_3, n_embd, n_tokens, inputs_3->nb[1], (size_t) j * inputs_3->nb[2]);
            }
        } else {
            inputs[0] = build_norm(inputs[0], layer.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
            inputs[1] = build_norm(inputs[1], layer.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
            inputs[2] = build_norm(inputs[2], layer.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
            inputs_3  = nullptr;
        }
    }

    ggml_tensor * sconv_k_in   = nullptr;
    ggml_tensor * sconv_v_in   = nullptr;
    ggml_tensor * sconv_wo_in  = nullptr;
    ggml_tensor * sconv_ffn_in = nullptr;

    // self-attention
    {
        ggml_tensor * Qcur   = nullptr;
        ggml_tensor * k_dyn  = nullptr;
        ggml_tensor * v_dyn  = nullptr;
        ggml_tensor * Rcur   = nullptr;
        ggml_tensor * sinks3 = nullptr;
        ggml_tensor * qgate  = nullptr;

        ggml_tensor * kk_mm  = nullptr;
        ggml_tensor * vv_mm  = nullptr;
        ggml_tensor * kv_mm  = nullptr;
        ggml_tensor * grs_mm = nullptr;

        if (loras->empty() && is_swa && inputs[0] == inputs[1] && inputs[1] == inputs[2] && layer.attn_qkv_grs) {
            grs_mm = ggml_mul_mat(ctx0, layer.attn_qkv_grs, inputs[0]);
            const int64_t n_q = n_embd_head * n_head;
            const int64_t n_r = hparams.n_relative * n_head;
            const int64_t n_grs = 2 * n_q + n_r + n_head;
            Qcur       = ggml_view_3d(ctx0, grs_mm, n_embd_head, n_head, n_tokens,
                                      (size_t) n_embd_head * sizeof(float), grs_mm->nb[1], 0);
            qgate      = ggml_view_2d(ctx0, grs_mm, n_q, n_tokens, grs_mm->nb[1], (size_t) n_q * sizeof(float));
            Rcur       = ggml_view_3d(ctx0, grs_mm, hparams.n_relative, n_head, n_tokens,
                                      (size_t) hparams.n_relative * sizeof(float), grs_mm->nb[1],
                                      (size_t) (2 * n_q) * sizeof(float));
            sinks3     = ggml_view_3d(ctx0, grs_mm, 1, n_head, n_tokens,
                                      sizeof(float), grs_mm->nb[1],
                                      (size_t) (2 * n_q + n_r) * sizeof(float));
            sconv_k_in = ggml_view_2d(ctx0, grs_mm, n_embd_k_gqa, n_tokens, grs_mm->nb[1],
                                      (size_t) n_grs * sizeof(float));
            sconv_v_in = ggml_view_2d(ctx0, grs_mm, n_embd_v_gqa, n_tokens, grs_mm->nb[1],
                                      (size_t) (n_grs + n_embd_k_gqa) * sizeof(float));
        } else {
            if (loras->empty() && !is_swa && inputs_3 && layer.attn_kv_kvdyn) {
                ggml_tensor * inp_kv_3d = ggml_view_3d(ctx0, inputs_3, n_embd, n_tokens, 2,
                        inputs_3->nb[1], inputs_3->nb[2], inputs_3->nb[2]);
                kv_mm      = ggml_mul_mat(ctx0, layer.attn_kv_kvdyn, inp_kv_3d);
                sconv_k_in = ggml_view_2d(ctx0, kv_mm, n_embd_k_gqa, n_tokens, kv_mm->nb[1], 0);
                k_dyn      = ggml_view_2d(ctx0, kv_mm, layer.attn_k_conv_proj->ne[1], n_tokens, kv_mm->nb[1], (size_t) n_embd_k_gqa * sizeof(float));
                sconv_v_in = ggml_view_2d(ctx0, kv_mm, n_embd_k_gqa, n_tokens, kv_mm->nb[1], kv_mm->nb[2]);
                v_dyn      = ggml_view_2d(ctx0, kv_mm, layer.attn_v_conv_proj->ne[1], n_tokens, kv_mm->nb[1], kv_mm->nb[2] + (size_t) n_embd_k_gqa * sizeof(float));
            } else if (loras->empty() && !is_swa && layer.attn_k_kdyn && layer.attn_v_vdyn) {
                kk_mm      = ggml_mul_mat(ctx0, layer.attn_k_kdyn, inputs[1]);
                vv_mm      = ggml_mul_mat(ctx0, layer.attn_v_vdyn, inputs[2]);
                sconv_k_in = ggml_view_2d(ctx0, kk_mm, n_embd_k_gqa, n_tokens, kk_mm->nb[1], 0);
                k_dyn      = ggml_view_2d(ctx0, kk_mm, layer.attn_k_conv_proj->ne[1], n_tokens, kk_mm->nb[1], (size_t) n_embd_k_gqa * sizeof(float));
                sconv_v_in = ggml_view_2d(ctx0, vv_mm, n_embd_k_gqa, n_tokens, vv_mm->nb[1], 0);
                v_dyn      = ggml_view_2d(ctx0, vv_mm, layer.attn_v_conv_proj->ne[1], n_tokens, vv_mm->nb[1], (size_t) n_embd_k_gqa * sizeof(float));
            } else if (loras->empty() && is_swa && inputs_3 && layer.attn_qkv_grs && n_embd_k_gqa == n_embd_v_gqa) {
                const int64_t n_q   = n_embd_head * n_head;
                const int64_t n_r   = hparams.n_relative * n_head;
                const int64_t n_grs = 2 * n_q + n_r + n_head;
                ggml_tensor * w_kv_3d = ggml_view_3d(ctx0, layer.attn_qkv_grs, n_embd, n_embd_k_gqa, 2,
                        layer.attn_qkv_grs->nb[1], (size_t) n_embd_k_gqa * layer.attn_qkv_grs->nb[1],
                        (size_t) n_grs * layer.attn_qkv_grs->nb[1]);
                ggml_tensor * inp_kv_3d = ggml_view_3d(ctx0, inputs_3, n_embd, n_tokens, 2,
                        inputs_3->nb[1], inputs_3->nb[2], inputs_3->nb[2]);
                kv_mm      = ggml_mul_mat(ctx0, w_kv_3d, inp_kv_3d);
                sconv_k_in = ggml_view_2d(ctx0, kv_mm, n_embd_k_gqa, n_tokens, kv_mm->nb[1], 0);
                sconv_v_in = ggml_view_2d(ctx0, kv_mm, n_embd_k_gqa, n_tokens, kv_mm->nb[1], kv_mm->nb[2]);
            } else if (loras->empty() && is_swa && inputs[1] == inputs[2] && layer.attn_kv) {
                kv_mm      = ggml_mul_mat(ctx0, layer.attn_kv, inputs[1]);
                sconv_k_in = ggml_view_2d(ctx0, kv_mm, n_embd_k_gqa, n_tokens, kv_mm->nb[1], 0);
                sconv_v_in = ggml_view_2d(ctx0, kv_mm, n_embd_k_gqa, n_tokens, kv_mm->nb[1], (size_t) n_embd_k_gqa * sizeof(float));
            } else {
                sconv_k_in = build_lora_mm(layer.wk, inputs[1]);
                sconv_v_in = build_lora_mm(layer.wv, inputs[2]);
                k_dyn      = !is_swa ? build_lora_mm(layer.attn_k_conv_proj, inputs[1]) : nullptr;
                v_dyn      = !is_swa ? build_lora_mm(layer.attn_v_conv_proj, inputs[2]) : nullptr;
            }

            ggml_tensor * w_grs = layer.attn_gate_rel_sink;
            if (!w_grs && layer.attn_qkv_grs) {
                const int64_t n_q = n_embd_head * n_head;
                const int64_t n_r = hparams.n_relative * n_head;
                w_grs = ggml_view_2d(ctx0, layer.attn_qkv_grs, n_embd, 2 * n_q + n_r + n_head, layer.attn_qkv_grs->nb[1], 0);
            }
            if (loras->empty() && w_grs) {
                grs_mm = ggml_mul_mat(ctx0, w_grs, inputs[0]);
                const int64_t n_q = n_embd_head * n_head;
                const int64_t n_r = hparams.n_relative * n_head;
                Qcur   = ggml_view_3d(ctx0, grs_mm, n_embd_head, n_head, n_tokens,
                                      (size_t) n_embd_head * sizeof(float), grs_mm->nb[1], 0);
                qgate  = ggml_view_2d(ctx0, grs_mm, n_q, n_tokens, grs_mm->nb[1], (size_t) n_q * sizeof(float));
                Rcur   = ggml_view_3d(ctx0, grs_mm, hparams.n_relative, n_head, n_tokens,
                                      (size_t) hparams.n_relative * sizeof(float), grs_mm->nb[1],
                                      (size_t) (2 * n_q) * sizeof(float));
                sinks3 = ggml_view_3d(ctx0, grs_mm, 1, n_head, n_tokens,
                                      sizeof(float), grs_mm->nb[1],
                                      (size_t) (2 * n_q + n_r) * sizeof(float));
            } else {
                Qcur   = ggml_reshape_3d(ctx0, build_lora_mm(layer.wq, inputs[0]), n_embd_head, n_head, n_tokens);
                Rcur   = ggml_reshape_3d(ctx0, build_lora_mm(layer.attn_rel,       inputs[0]), hparams.n_relative, n_head, n_tokens);
                sinks3 = ggml_reshape_3d(ctx0, build_lora_mm(layer.attn_sink_proj, inputs[0]), 1,                  n_head, n_tokens);
                qgate  = build_lora_mm(layer.wqkv_gate, inputs[0]);
            }
        }

        // the projections read the same inputs: expanding them before any of their consumers puts them in one group of
        // concurrent kernels, so the GPU overlaps their tails instead of draining after each
        for (ggml_tensor * t : { Qcur, kk_mm, vv_mm, kv_mm, grs_mm, sconv_k_in, sconv_v_in, k_dyn, v_dyn, Rcur, sinks3, qgate }) {
            if (t) {
                ggml_build_forward_expand(gf, t);
            }
        }

        ggml_tensor * Kcur = convolve(sconv_k_in, layer.attn_k_conv, k_dyn, history, 0);
        ggml_tensor * Vcur = convolve(sconv_v_in, layer.attn_v_conv, v_dyn, history, n_embd_k_gqa);

        // the clip factors of q and k go into the KQ scale; the convolution scale s of k and v goes into their norms:
        // l2_norm(s*y, t) = l2_norm(y, t/s) and rms_norm(s*y, eps) = rms_norm(y, eps/s^2)
        const float qk_clip = hparams.f_qk_clip;
        const float s_kv    = conv_scale(!is_swa);

        Qcur = clip(Qcur, layer.attn_q_clip, qk_clip);
        Kcur = ggml_l2_norm(ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), qk_clip/s_kv);
        Vcur = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens), hparams.f_norm_rms_eps/(s_kv*s_kv));
        Rcur = clip(Rcur, layer.attn_rel_clip, hparams.f_relative_clip);

        if (!is_swa) {
            Qcur = ggml_mul(ctx0, Qcur, tau);
            Rcur = ggml_mul(ctx0, Rcur, tau_rel);
        } else {
            Rcur = ggml_scale(ctx0, Rcur, hparams.f_relative_clip);
        }

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        cb(Rcur, "Rcur", il);

        // project r on the bias table first, then gather by distance: only the gathered bias is n_kv x n_tokens x n_head
        // the padded table of a physical layer is pre-cast and zero-padded at load time when available
        ggml_tensor * rel_table = layer.attn_rel_bias_pad;
        if (!rel_table) {
            ggml_tensor *& rt = rel_tables[layer.attn_rel_bias];
            if (!rt) {
                rt = layer.attn_rel_bias;
                if (rt->type != GGML_TYPE_F32) {
                    rt = ggml_cast(ctx0, rt, GGML_TYPE_F32);
                }
                rt = ggml_pad(ctx0, rt, 0, 1, 0, 0);
            }
            rel_table = rt;
        }

        // the sink logit t is per query: exp(s - t)/(1 + sum(exp(s - t))) = exp(s)/(exp(t) + sum(exp(s))), with a zero sink
        ggml_tensor * table = ggml_mul_mat(ctx0, rel_table, Rcur);
        table = ggml_sub(ctx0, table, sinks3);

        if (!zero_sinks || zero_sinks->ne[0] != n_head) {
            zero_sinks = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);
        }

        cur = attention(Qcur, Kcur, Vcur, table, zero_sinks, qk_clip*qk_clip/sqrtf(n_embd_head), il);

        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, qgate));
        sconv_wo_in = build_lora_mm(layer.wo, cur);
        cur = convolve(sconv_wo_in, layer.attn_post_conv, nullptr, history, 2*n_embd_k_gqa, conv_scale(false));
        cb(cur, "attn_out", il);
    }

    // the first layer of each block starts a new residual stream, the earlier one joins the mixed blocks
    if (ip % hparams.attn_res_block_size == 0) {
        blocks.push_back(inpSA);
    } else {
        cur = ggml_add(ctx0, inpSA, cur);
    }

    ggml_tensor * ffn_inp = cur;
    cb(ffn_inp, "ffn_inp", il);

    // MoE
    cur = mix(blocks, cur, layer.ffn_res_query);
    if (messages) {
        cur = control(cur, message(messages, LOOP_CTRL_FFN), message(messages_norm, LOOP_CTRL_FFN), layer, LOOP_CTRL_FFN);
    }
    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

    inkling_ssd_layer = size_t(ip) < model.inkling_ssd_experts.size() ? &model.inkling_ssd_experts[ip] : nullptr;
    sconv_ffn_in = build_moe_ffn(cur,
            loras->empty() && layer.ffn_gate_inp_bf16 ? layer.ffn_gate_inp_bf16 : layer.ffn_gate_inp,
            nullptr,
            nullptr,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            hparams.n_expert,
            hparams.n_expert_used,
            LLM_FFN_SITU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
            il,
            nullptr,
            layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            nullptr,
            layer.ffn_down_exps_s);

    cur = convolve(sconv_ffn_in, layer.ffn_post_conv, nullptr, history, 2*n_embd_k_gqa + n_embd, conv_scale(false));
    cb(cur, "ffn_out", il);

    cur = ggml_add(ctx0, ffn_inp, cur);
    cb(cur, "l_out", il);

    // update all 4 shortconv channel groups of this layer visit in a single kernel at the end of the layer
    {
        const size_t row_size = (size_t) hparams.n_embd_r()*ggml_element_size(state);
        const size_t offs     = (size_t) recr->mctx->get_head()*row_size;

        ggml_tensor * cell   = ggml_view_2d(ctx0, state, hparams.n_embd_r(), ubatch.n_seqs, row_size, offs);
        ggml_tensor * r_hist = recr->mctx->get_r_hist_l(il);
        ggml_tensor * hist_v = r_hist
            ? ggml_view_3d(ctx0, r_hist, hparams.n_sconv_channels, r_hist->ne[1], ubatch.n_seqs,
                           r_hist->nb[1], r_hist->nb[2], (size_t) recr->mctx->get_head()*r_hist->nb[2])
            : nullptr;

        ggml_tensor * st = ggml_inkling_sconv_state_4_hist(
            ctx0, cell, history, sconv_k_in, sconv_v_in, sconv_wo_in, sconv_ffn_in,
            hist_v, hist_v ? inp_pos : nullptr);
        if (tree_parents_packed != 0) {
            st->op_params[0] = 1;
        }
        if (tree_inp) {
            tree_inp->sconv_state_nodes.push_back(st);
        }
        // Expand cur (which depends on all 4 convolve ops reading history) BEFORE st updates state in-place!
        ggml_build_forward_expand(gf, cur);
        ggml_build_forward_expand(gf, st);
    }

    return cur;
}

llm_build_inkling2::llm_build_inkling2(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params), model(model), tree_parents_packed(params.tree_parents_packed) {
    inp      = build_inp_mem_hybrid_iswa_rel(hparams.n_relative_extent, hparams.n_swa);
    recr     = inp->get_recr();
    tree_inp = static_cast<llm_graph_input_inkling_tree *>(
            res->add_input(std::make_unique<llm_graph_input_inkling_tree>(tree_parents_packed)));

    inp_pos                   = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // log-scaled positions: 1 + alpha*log(max(1, (pos + 1)/floor)), multiplies q and r of the full-attention layers
    tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);
    tau_rel = ggml_scale(ctx0, tau, hparams.f_relative_clip);

    ggml_tensor * cur = build_inp_embd(model.tok_embd);

    // do not normalize raw embeddings input (i.e. encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }

    std::vector<ggml_tensor *> blocks;

    auto record_tap = [&](int visit, ggml_tensor * t) {
        res->t_layer_inp[visit] = t;
        if (visit < (int) cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[visit]) {
            ggml_build_forward_expand(gf, t);
        }
    };

    if (hparams.n_layer_phys == 0) {
        for (int il = 0; il < n_layer; ++il) {
            record_tap(il, cur);
            cur = build_layer(il, il, cur, blocks, nullptr, nullptr);
        }
    } else {
        const int start  = hparams.loop_start;
        const int end    = hparams.loop_end;
        const int n_phys = hparams.n_layer_phys;
        int visit = 0;

        for (int il = 0; il < start; ++il) {
            record_tap(visit++, cur);
            cur = build_layer(il, il, cur, blocks, nullptr, nullptr);
        }

        // the message queries of every core layer, [n_embd, LOOP_CTRL_COUNT*(end - start)]: the messages of a pass
        // depend only on the previous pass, so one mix computes them for all its layers
        ggml_tensor * msg_query = model.loop_msg_query_all;
        if (!msg_query) {
            for (int ip = start; ip < end; ++ip) {
                ggml_tensor * q = model.layers[ip].loop_msg_query;
                msg_query = msg_query ? ggml_concat(ctx0, msg_query, q, 1) : q;
            }
        }

        // every pass starts from the prelude output
        ggml_tensor * prelude = cur;
        const auto prelude_blocks = blocks;
        for (uint32_t pass = 0; pass < hparams.loop_count; ++pass) {
            // the bank of the previous pass: its blocks after the first core block, and its output
            std::vector<ggml_tensor *> bank;
            ggml_tensor * bank_out = nullptr;
            if (pass > 0) {
                bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
                bank_out = cur;
                cur      = prelude;
                blocks   = prelude_blocks;
            }
            ggml_tensor * msgs      = nullptr;
            ggml_tensor * msgs_norm = nullptr;
            if (pass > 0) {
                msgs      = mix(bank, bank_out, msg_query); // [n_embd, n_tokens, LOOP_CTRL_COUNT*(end - start)]
                msgs_norm = ggml_rms_norm(ctx0, msgs, 1e-6f);
            }
            for (int ip = start; ip < end; ++ip) {
                const int il = pass == 0 ? ip : n_phys + (pass - 1)*(end - start) + (ip - start);
                record_tap(visit++, cur);
                if (pass == 0) {
                    cur = build_layer(il, ip, cur, blocks, nullptr, nullptr);
                    continue;
                }
                const size_t offs      = (size_t) (ip - start)*LOOP_CTRL_COUNT*msgs->nb[2];
                const size_t offs_norm = (size_t) (ip - start)*LOOP_CTRL_COUNT*msgs_norm->nb[2];
                cur = build_layer(il, ip, cur, blocks,
                        ggml_view_3d(ctx0, msgs,      n_embd, n_tokens, LOOP_CTRL_COUNT, msgs->nb[1],      msgs->nb[2],      offs),
                        ggml_view_3d(ctx0, msgs_norm, n_embd, n_tokens, LOOP_CTRL_COUNT, msgs_norm->nb[1], msgs_norm->nb[2], offs_norm));
            }
        }

        for (int il = end; il < n_phys; ++il) {
            record_tap(visit++, cur);
            cur = build_layer(il, il, cur, blocks, nullptr, nullptr);
        }
        GGML_ASSERT(visit == n_layer);
    }

    cur = mix(blocks, cur, model.output_res_query);
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    record_tap(n_layer, cur);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    // the logits feed a tanh softcap, so accumulate them in F32 on backends that would use F16
    // Slice padded vocabulary rows before the projection when no LoRA is attached to avoid a post-matmul copy.
    const bool pre_slice_vocab = loras->empty() && hparams.n_vocab_unpadded && hparams.n_vocab_unpadded < model.output->ne[1];
    ggml_tensor * lm_head_w = pre_slice_vocab
        ? ggml_view_2d(ctx0, model.output, model.output->ne[0], hparams.n_vocab_unpadded, model.output->nb[1], 0)
        : model.output;
    cur = pre_slice_vocab ? ggml_mul_mat(ctx0, lm_head_w, cur) : build_lora_mm(model.output, cur);
    ggml_tensor * logits_mm = cur;
    while (logits_mm->op == GGML_OP_ADD) { // LoRA adapters add their terms on top of the base product
        logits_mm = logits_mm->src[0];
    }
    ggml_mul_mat_set_prec(logits_mm, GGML_PREC_F32);

    if (hparams.f_final_logit_softcapping != 0.0f) {
        const float cap = hparams.f_final_logit_softcapping;

        cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    }

    // drop the padded vocabulary (fallback if LoRA prevented pre-slicing)
    if (!pre_slice_vocab && hparams.n_vocab_unpadded && hparams.n_vocab_unpadded < cur->ne[0]) {
        cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
