#include "models.h"
#include "llama-impl.h"
#include "llama-memory-recurrent.h"

#include <atomic>
#include <cinttypes>

//
// Kimi-K3 text model.
//
// Hybrid KDA (linear) + MLA (full) attention, as in Kimi-Linear-48B, plus five
// things that architecture does not have:
//
//   1. cross-layer residual attention  (attn_res_block_size)
//   2. latent MoE                      (routed experts run at n_expert_latent)
//   3. situ activation                 (replaces SwiGLU everywhere)
//   4. MLA output gate                 (sigmoid gate before o_proj)
//   5. full-rank KDA gate              (single ssm_g instead of ssm_g_a/ssm_g_b)
//

//
// situ activation:
//   situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)
// linear_beta <= 0 disables the transform on the up branch.
//
static ggml_tensor * kimi_k3_situ(ggml_context * ctx0, ggml_tensor * gate, ggml_tensor * up,
                                  float beta, float linear_beta) {
    ggml_tensor * a = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, gate, 1.0f/beta)), beta);
    a = ggml_mul(ctx0, a, ggml_sigmoid(ctx0, gate));

    if (linear_beta > 0.0f) {
        up = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, up, 1.0f/linear_beta)), linear_beta);
    }
    return ggml_mul(ctx0, a, up);
}

//
// cross-layer residual attention
//

void llm_build_kimi_k3::res_push(ggml_tensor * cur, int64_t n_embd, int64_t n_tokens) {
    ckpts.push_back(ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens));
}

ggml_tensor * llm_build_kimi_k3::res_stack(int64_t n_embd, int64_t n_tokens) {
    GGML_UNUSED(n_embd);
    GGML_UNUSED(n_tokens);
    if (stack_cache_n == (int) ckpts.size()) {
        return stack_cache;   // set unchanged since the last mix
    }
    ggml_tensor * acc = ckpts[0];
    for (size_t i = 1; i < ckpts.size(); ++i) {
        acc = ggml_concat(ctx0, acc, ckpts[i], 1);
    }
    stack_cache   = acc;
    stack_cache_n = (int) ckpts.size();
    return acc;
}

ggml_tensor * llm_build_kimi_k3::res_mix(ggml_tensor * cur, ggml_tensor * score_w,
                                         int64_t n_embd, int64_t n_tokens, int il) {
    const int n_ckpt = (int) ckpts.size();
    if (n_ckpt == 0) {
        return cur; // layer 0: nothing banked yet
    }

    const float eps = hparams.f_norm_rms_eps;

    ggml_tensor * src = res_stack(n_embd, n_tokens);   // [n_embd, n_ckpt, n_tokens]

    // Scores for the banked checkpoints. One rms_norm covers all of them because
    // ne0 is n_embd. NOTE: the scores use the *normalized* values, but the weighted
    // sum below uses the *raw* ones - mirroring _apply_attn_res.
    ggml_tensor * sc_src = ggml_rms_norm(ctx0, src, eps);
    sc_src = ggml_mul(ctx0, sc_src, score_w);
    sc_src = ggml_sum_rows(ctx0, sc_src);                          // [1, n_ckpt, n_tokens]
    sc_src = ggml_reshape_2d(ctx0, sc_src, n_ckpt, n_tokens);

    // The current residual stream is scored separately and kept out of the stack,
    // so the stack stays append-only.
    ggml_tensor * sc_cur = ggml_rms_norm(ctx0, cur, eps);
    sc_cur = ggml_mul(ctx0, sc_cur, score_w);
    sc_cur = ggml_sum_rows(ctx0, sc_cur);                          // [1, n_tokens]

    ggml_tensor * scores = ggml_concat(ctx0, sc_src, sc_cur, 0);   // [n_ckpt+1, n_tokens]
    ggml_tensor * probs  = ggml_soft_max(ctx0, scores);            // over ne0 = n_ckpt+1
    cb(probs, "res_probs", il);

    // Split the convex combination. The SSD branch predates the fused
    // ggml_dsv4_hc_pre op, so use the equivalent batched matrix-vector product:
    // [n_ckpt, n_embd, n_tokens] x [n_ckpt, 1, n_tokens].
    ggml_tensor * p_src = ggml_cont(ctx0, ggml_view_2d(ctx0, probs, n_ckpt, n_tokens, probs->nb[1], 0));
    ggml_tensor * p_cur = ggml_cont(ctx0, ggml_view_2d(ctx0, probs, 1, n_tokens, probs->nb[1],
                                                       probs->nb[0] * n_ckpt));

    ggml_tensor * src_t = ggml_cont(ctx0, ggml_permute(ctx0, src, 1, 0, 2, 3));
    p_src = ggml_reshape_3d(ctx0, p_src, n_ckpt, 1, n_tokens);
    ggml_tensor * out = ggml_mul_mat(ctx0, src_t, p_src);
    out = ggml_reshape_2d(ctx0, out, n_embd, n_tokens);
    out = ggml_add(ctx0, out, ggml_mul(ctx0, cur, p_cur));

    return out;
}


// ---- experiment: decode-time routed-expert layer skipping ----------------------------------------------
#include <sys/stat.h>
#include <fstream>
#include <mutex>
#include <sstream>

static std::vector<bool> llama_k3_parse_layer_set(const std::string & text, int n_layer) {
    std::vector<bool> set(n_layer, false);
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const size_t a = item.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        const size_t b = item.find_last_not_of(" \t\r\n");
        item = item.substr(a, b - a + 1);
        long lo, hi;
        const size_t dash = item.find('-');
        if (dash == std::string::npos) { lo = hi = std::atol(item.c_str()); }
        else { lo = std::atol(item.substr(0, dash).c_str()); hi = std::atol(item.substr(dash + 1).c_str()); }
        for (long l = std::max(0L, lo); l <= hi && l < n_layer; ++l) set[l] = true;
    }
    return set;
}

static std::string llama_k3_layer_set_text(const std::vector<bool> & set) {
    std::string out; int n = 0;
    for (size_t i = 0; i < set.size(); ++i) if (set[i]) { if (!out.empty()) out += ","; out += std::to_string(i); ++n; }
    return out.empty() ? "(none)" : out + " [n=" + std::to_string(n) + "]";
}

static bool llama_k3_decode_skip_moe_layer(int il, int n_layer) {
    static std::mutex mu;
    static std::vector<bool> set;
    static std::string file;
    static long long stamp = -1;
    static bool init = false;
    std::lock_guard<std::mutex> lock(mu);
    if (!init) {
        init = true;
        set.assign(n_layer, false);
        const char * env = std::getenv("LLAMA_K3_DECODE_SKIP_MOE_LAYERS");
        const char * f   = std::getenv("LLAMA_K3_DECODE_SKIP_MOE_LAYERS_FILE");
        if (f && f[0]) file = f;
        if (env && env[0]) {
            set = llama_k3_parse_layer_set(env, n_layer);
            LLAMA_LOG_INFO("[k3_skip_moe] env layers=%s\n", llama_k3_layer_set_text(set).c_str());
        }
    }
    if (!file.empty()) {
        struct stat st;
        long long cur = 0;
        if (stat(file.c_str(), &st) == 0) {
#if defined(__APPLE__)
            cur = (long long) st.st_mtimespec.tv_sec * 1000000000LL + st.st_mtimespec.tv_nsec + (long long) st.st_size;
#elif defined(_WIN32)
            cur = (long long) st.st_mtime * 1000000000LL + (long long) st.st_size;
#else
            cur = (long long) st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec + (long long) st.st_size;
#endif
        }
        if (cur != stamp) {
            stamp = cur;
            std::string text;
            if (cur != 0) { std::ifstream in(file); std::stringstream buf; buf << in.rdbuf(); text = buf.str(); }
            set = llama_k3_parse_layer_set(text, n_layer);
            LLAMA_LOG_INFO("[k3_skip_moe] file=%s layers=%s\n", file.c_str(), llama_k3_layer_set_text(set).c_str());
        }
    }
    return il >= 0 && il < (int) set.size() && set[il];
}
// ----------------------------------------------------------------------------------------------------------

llm_build_kimi_k3::llm_build_kimi_k3(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params),
    model(model),
    decode_expert_policy(params.k3_expert_policy),
    decode_expert_table(params.k3_expert_table) {

    if (!decode_expert_policy.valid()) {
        GGML_ABORT("invalid K3 decode expert policy: %s",
            llama_k3_decode_expert_policy_error_name(decode_expert_policy.error));
    }
    if (decode_expert_policy.mode == llama_k3_decode_expert_mode::mass &&
            !hparams.expert_weights_norm) {
        GGML_ABORT("K3 cumulative-mass expert truncation requires normalized routing weights");
    }
    if (decode_expert_policy.enabled()) {
        static std::atomic<bool> reported{false};
        if (!reported.exchange(true, std::memory_order_relaxed)) {
            LLAMA_LOG_INFO(
                "[k3_decode_experts] mode=%s model_top_k=%u hard_top_k=%" PRId64
                " mass=%.6f min_k=%" PRId64 " max_k=%" PRId64 " layers=%" PRId64 "-%" PRId64
                " decode_only=1 small_batch_max=%d prefill_top_k=%u\n",
                llama_k3_decode_expert_mode_name(decode_expert_policy.mode),
                hparams.n_expert_used,
                decode_expert_policy.top_k,
                decode_expert_policy.mass,
                decode_expert_policy.min_k,
                decode_expert_policy.max_k,
                decode_expert_policy.layer_start,
                decode_expert_policy.layer_end,
                decode_expert_policy.small_batch ? 8 : 1,
                hparams.n_expert_used);
        }
    }

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "inp_embd", -1);

    // K3 MLA is nope-only, so there is no position input

    auto * inp_kv      = !hparams.is_mla() ? build_inp_mem_hybrid()   : nullptr;
    auto * inp_k       =  hparams.is_mla() ? build_inp_mem_hybrid_k() : nullptr;
    auto * inp_rs      =  hparams.is_mla() ? inp_k->get_recr() : inp_kv->get_recr();
    auto * inp_attn_kv = !hparams.is_mla() ? inp_kv->get_attn() : nullptr;
    auto * inp_attn_k  =  hparams.is_mla() ? inp_k->get_attn()  : nullptr;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t n_head_kda = hparams.n_head();
    const int64_t head_dim   = hparams.n_embd_head_kda;
    const int64_t d_conv     = hparams.ssm_d_conv;
    const int64_t d_inner    = n_head_kda * head_dim;
    const int64_t n_seqs     = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    const int64_t n_embd_head_k_mla   = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla   = hparams.n_embd_head_v_mla();
    const int64_t kv_lora_rank        = hparams.n_lora_kv;
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    const float   kq_scale_mla        = 1.0f / sqrtf((float) n_embd_head_k_mla);

    const uint32_t res_bs        = hparams.attn_res_block_size;
    const bool     use_attn_res  = res_bs > 0;
    const int64_t  n_embd_latent = hparams.n_expert_latent > 0 ? hparams.n_expert_latent : n_embd;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        // DFlash target layer ids are stored with the Hugging Face hidden-state
        // +1 convention, so layer input il is the output of target layer il - 1.
        res->t_layer_inp[il] = inpL;

        // `prefix_sum` is the residual stream. On checkpoint layers it is banked
        // into res_stack and restarts from the attention output alone.
        ggml_tensor * prefix_sum = inpL;

        cur = use_attn_res ? res_mix(prefix_sum, layer.attn_res_score, n_embd, n_tokens, il)
                           : prefix_sum;

        bool banked = false;
        if (use_attn_res && (uint32_t) il % res_bs == 0) {
            res_push(prefix_sum, n_embd, n_tokens);  // banks the RAW layer input, not `cur`
            banked = true;
        }

        cur = build_norm(cur, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recurrent(il)) {
            cur = build_kda_layer(cur, layer, inp_rs, d_conv, head_dim, n_head_kda,
                                  d_inner, n_seq_tokens, n_seqs, il);
        } else {
            cur = build_mla_layer(cur, layer, inp_attn_k, inp_attn_kv,
                                  n_embd_head_k_mla, n_embd_head_v_mla, kv_lora_rank,
                                  n_embd_head_qk_rope, n_embd_head_qk_nope, kq_scale_mla, il);
        }

        prefix_sum = banked ? cur : ggml_add(ctx0, prefix_sum, cur);
        cb(prefix_sum, "prefix_sum_attn", il);

        cur = use_attn_res ? res_mix(prefix_sum, layer.ffn_res_score, n_embd, n_tokens, il)
                           : prefix_sum;

        cur = build_norm(cur, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            ggml_tensor * g = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
            ggml_tensor * u = ggml_mul_mat(ctx0, layer.ffn_up,   cur);
            cur = kimi_k3_situ(ctx0, g, u, hparams.situ_beta, hparams.situ_linear_beta);
            cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);
            cb(cur, "ffn_out", il);
        } else {
            cur = build_latent_moe(cur, layer, n_embd_latent, il);
        }

        prefix_sum = ggml_add(ctx0, prefix_sum, cur);
        prefix_sum = build_cvec(prefix_sum, il);
        cb(prefix_sum, "l_out", il);

        inpL = prefix_sum;
    }

    res->t_layer_inp[n_layer] = inpL;

    cur = inpL;

    // final mix, then narrow to the output tokens
    if (use_attn_res) {
        cur = res_mix(cur, model.output_res_score, n_embd, n_tokens, -1);
    }
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

//
// KDA layer
//

// Causal conv1d over one of Q/K/V. `qkv` selects which third of the conv state to use.
// Polled rather than blocking is not a concern here; this mirrors kimi-linear's helper.
static ggml_tensor * kimi_k3_conv1d(ggml_cgraph * gf, ggml_context * ctx0,
                                    ggml_tensor * conv_states_all, ggml_tensor * conv_state_all,
                                    int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
                                    int64_t d_conv, int64_t head_dim, int64_t n_head,
                                    int64_t n_seq_tokens, int64_t n_seqs, int64_t n_tokens, int64_t kv_head) {
    const int64_t d_inner         = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total  = 3 * conv_state_size;

    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1)   * ggml_element_size(conv_state_all),
        n_embd_r_total * ggml_element_size(conv_state_all),
        qkv * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);
    ggml_tensor * x_3d   = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);
    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    ggml_tensor * last_conv_x = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
        conv_x->nb[1], conv_x->nb[2], n_seq_tokens * conv_x->nb[0]);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, last_conv_x,
            ggml_view_3d(ctx0, conv_states_all, d_conv - 1, d_inner, n_seqs,
                (d_conv - 1)   * ggml_element_size(conv_states_all),
                n_embd_r_total * ggml_element_size(conv_states_all),
                (kv_head * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_all))));

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);
    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
    Xcur = ggml_silu(ctx0, Xcur);

    return ggml_reshape_4d(ctx0, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}

ggml_tensor * llm_build_kimi_k3::build_kda_layer(
        ggml_tensor * cur, const llama_layer & layer, llm_graph_input_rs * inp_rs,
        int64_t d_conv, int64_t head_dim, int64_t n_head_kda,
        int64_t d_inner, int64_t n_seq_tokens, int64_t n_seqs, int il) {

    const auto * mctx_cur = inp_rs->mctx;
    const auto   kv_head  = mctx_cur->get_head();

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * conv_state_all  = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);

    ggml_tensor * Qcur = kimi_k3_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head);
    ggml_tensor * Kcur = kimi_k3_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head);
    ggml_tensor * Vcur = kimi_k3_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head);
    cb(Qcur, "kda_q_conv", il);
    cb(Kcur, "kda_k_conv", il);
    cb(Vcur, "kda_v_conv", il);

    // The decay gate has two forms, selected by linear_attn_config.gate_lower_bound
    // (fla/ops/kda/gate.py). `lower_bound` is NOT a clamp - when set it swaps the
    // activation entirely:
    //
    //   unset (kimi-linear):  g = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)
    //   set   (K3, -5.0):     g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))
    //
    // ssm_a holds -exp(A_log) (folded at conversion time), so exp(A_log) == -ssm_a.
    ggml_tensor * f_a = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
    ggml_tensor * g1  = ggml_mul_mat(ctx0, layer.ssm_f_b, f_a);
    g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);

    ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head_kda, 1);

    if (hparams.kda_gate_lower_bound > -INFINITY) {
        g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head_kda, n_tokens);
        g1 = ggml_mul(ctx0, g1, A);                                    // -exp(A_log) * (...)
        g1 = ggml_sigmoid(ctx0, ggml_scale(ctx0, g1, -1.0f));
        g1 = ggml_scale(ctx0, g1, hparams.kda_gate_lower_bound);
    } else {
        g1 = ggml_softplus(ctx0, g1);
        g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head_kda, n_tokens);
        g1 = ggml_mul(ctx0, g1, A);
    }
    cb(g1, "kda_g1", il);

    g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head_kda, n_seq_tokens, n_seqs);

    ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
    beta = ggml_reshape_4d(ctx0, beta, 1, n_head_kda, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "kda_beta", il);

    ggml_tensor * cur_3d = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

    ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
    ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head_kda, n_seqs);

    const float eps = hparams.f_norm_rms_eps;
    Qcur = ggml_l2_norm(ctx0, Qcur, eps);
    Kcur = ggml_l2_norm(ctx0, Kcur, eps);

    auto attn_out = build_delta_net(Qcur, Kcur, Vcur, g1, beta, state, il);

    ggml_tensor * output    = ggml_cont(ctx0, attn_out.first);
    cb(output, "kda_scan_out", il);
    ggml_tensor * new_state = attn_out.second;

    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, new_state,
            ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                         kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

    // K3: single full-rank gate (kimi-linear factors this as g_b(g_a(x)))
    ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur_3d, cur_3d->ne[0], n_seq_tokens * n_seqs);
    ggml_tensor * g2     = ggml_mul_mat(ctx0, layer.ssm_g, cur_2d);
    g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head_kda, n_seq_tokens * n_seqs);

    ggml_tensor * o      = ggml_reshape_3d(ctx0, output, head_dim, n_head_kda, n_seq_tokens * n_seqs);
    ggml_tensor * normed = build_norm(o, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
    cb(g2, "kda_g2", il);
    cb(normed, "kda_normed", il);
    ggml_tensor * gated  = ggml_mul(ctx0, normed, ggml_sigmoid(ctx0, g2));

    gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
    cur   = ggml_mul_mat(ctx0, layer.wo, gated);
    cb(cur, "kda_out", il);

    return cur;
}

//
// MLA layer (nope-only, with K3's sigmoid output gate)
//

ggml_tensor * llm_build_kimi_k3::build_mla_layer(
        ggml_tensor * cur, const llama_layer & layer,
        llm_graph_input_attn_k * inp_attn_k, llm_graph_input_attn_kv * inp_attn_kv,
        int64_t n_embd_head_k_mla, int64_t n_embd_head_v_mla, int64_t kv_lora_rank,
        int64_t n_embd_head_qk_rope, int64_t n_embd_head_qk_nope, float kq_scale, int il) {

    ggml_tensor * inp_gate = cur; // the output gate reads the *normed* layer input

    ggml_tensor * Qcur;
    if (layer.wq_a) {
        Qcur = ggml_mul_mat(ctx0, layer.wq_a, cur);
        Qcur = build_norm(Qcur, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        Qcur = ggml_mul_mat(ctx0, layer.wq_b, Qcur);
    } else {
        Qcur = ggml_mul_mat(ctx0, layer.wq, cur);
    }

    ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);

    ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

    // no RoPE: mla_use_nope is asserted at conversion time
    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);

    ggml_tensor * out;
    if (layer.wk_b && layer.wv_b) {
        ggml_tensor * q_nope = ggml_view_3d(ctx0, Qcur, n_embd_head_qk_nope, n_head, n_tokens,
            ggml_row_size(Qcur->type, n_embd_head_k_mla),
            ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head, 0);
        ggml_tensor * q_pe = ggml_view_3d(ctx0, Qcur, n_embd_head_qk_rope, n_head, n_tokens,
            ggml_row_size(Qcur->type, n_embd_head_k_mla),
            ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head,
            ggml_row_size(Qcur->type, n_embd_head_qk_nope));

        q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
        ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
        q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);

        ggml_tensor * Q = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
        ggml_tensor * kv_cmpr_3d = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
        ggml_tensor * K = ggml_concat(ctx0, kv_cmpr_3d, k_pe, 0);
        ggml_tensor * V = kv_cmpr_3d;

        // wo == NULL: the output projection is applied after the gate below
        out = build_attn(inp_attn_k, nullptr, NULL, nullptr, Q, K, V, nullptr, nullptr, layer.wv_b, kq_scale, il);
    } else {
        ggml_tensor * Q = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k_mla, n_head, n_tokens);
        ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_b, kv_cmpr);
        const int64_t kv_per_head = n_embd_head_qk_nope + n_embd_head_v_mla;

        ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
            ggml_row_size(kv->type, kv_per_head), ggml_row_size(kv->type, kv_per_head * n_head), 0);
        ggml_tensor * V = ggml_cont(ctx0, ggml_view_3d(ctx0, kv, n_embd_head_v_mla, n_head, n_tokens,
            ggml_row_size(kv->type, kv_per_head), ggml_row_size(kv->type, kv_per_head * n_head),
            ggml_row_size(kv->type, n_embd_head_qk_nope)));

        ggml_tensor * k_pe_t = ggml_new_tensor_3d(ctx0, k_pe->type, n_embd_head_qk_rope, n_head, n_tokens);
        ggml_tensor * K = ggml_concat(ctx0, ggml_repeat(ctx0, k_pe, k_pe_t), k_nope, 0);

        out = build_attn(inp_attn_kv, nullptr, NULL, nullptr, Q, K, V, nullptr, nullptr, nullptr, kq_scale, il);
    }

    // K3: attn_output *= sigmoid(g_proj(x)), then o_proj
    if (layer.wqkv_gate) {
        ggml_tensor * g = ggml_sigmoid(ctx0, ggml_mul_mat(ctx0, layer.wqkv_gate, inp_gate));
        out = ggml_mul(ctx0, out, g);
        cb(out, "mla_gated", il);
    }

    out = ggml_mul_mat(ctx0, layer.wo, out);
    cb(out, "mla_out", il);

    return out;
}

//
// latent MoE: down-project, run the routed experts in the latent space, norm, up-project;
// shared experts stay at n_embd and read the un-projected input.
//

ggml_tensor * llm_build_kimi_k3::build_latent_moe(
        ggml_tensor * cur, const llama_layer & layer, int64_t n_embd_latent, int il) {

    ggml_tensor * identity = cur;

    // experiment: drop the routed-expert branch on decode-sized batches for selected layers
    const bool skip_routed = n_tokens <= (decode_expert_policy.small_batch ? 8 : 1) &&
        llama_k3_decode_skip_moe_layer(il, (int) hparams.n_layer);
    ggml_tensor * moe_out = nullptr;
    if (!skip_routed) {
    ggml_tensor * routed_in = layer.ffn_routed_down
        ? ggml_mul_mat(ctx0, layer.ffn_routed_down, cur)
        : cur;

    // The router scores the FULL-WIDTH input while the experts consume the latent
    // one, so the logits are computed here and handed to build_moe_ffn via
    // `probs_in` (which substitutes for the gate_inp matmul).
    ggml_tensor * logits = ggml_mul_mat(ctx0, layer.ffn_gate_inp, identity);
    cb(logits, "ffn_moe_logits", il);

    const auto expert_args = llama_k3_expert_graph_args(decode_expert_policy, decode_expert_table, n_tokens, il);

    moe_out = build_moe_ffn(routed_in,
        nullptr, // gate_inp unused: logits supplied below
        layer.ffn_up_exps,
        layer.ffn_gate_exps,
        layer.ffn_down_exps,
        layer.ffn_exp_probs_b,
        hparams.n_expert,
        hparams.n_expert_used,
        LLM_FFN_SITU, hparams.expert_weights_norm,
        hparams.expert_weights_scale,
        (llama_expert_gating_func_type) hparams.expert_gating_func,
        il,
        logits,
        /* gate_up_exps */ nullptr,
        /* up_exps_s    */ nullptr,
        /* gate_exps_s  */ nullptr,
        /* down_exps_s  */ nullptr,
        expert_args.hard_top_k,
        expert_args.mass_threshold,
        expert_args.mass_min);
    cb(moe_out, "ffn_moe_out", il);

    if (layer.ffn_routed_norm) {
        moe_out = build_norm(moe_out, layer.ffn_routed_norm, NULL, LLM_NORM_RMS, il);
    }
    if (layer.ffn_routed_up) {
        moe_out = ggml_mul_mat(ctx0, layer.ffn_routed_up, moe_out);
    }
    } // !skip_routed
    GGML_UNUSED(n_embd_latent);

    if (layer.ffn_gate_shexp) {
        ggml_tensor * g = ggml_mul_mat(ctx0, layer.ffn_gate_shexp, identity);
        ggml_tensor * u = ggml_mul_mat(ctx0, layer.ffn_up_shexp,   identity);
        ggml_tensor * sh = kimi_k3_situ(ctx0, g, u, hparams.situ_beta, hparams.situ_linear_beta);
        sh = ggml_mul_mat(ctx0, layer.ffn_down_shexp, sh);
        cb(sh, "ffn_shexp", il);
        moe_out = moe_out ? ggml_add(ctx0, moe_out, sh) : sh;
    }
    if (moe_out == nullptr) {
        moe_out = ggml_scale(ctx0, identity, 0.0f); // skipped layer without shared experts: zero contribution
    }

    cb(moe_out, "ffn_out", il);
    return moe_out;
}
