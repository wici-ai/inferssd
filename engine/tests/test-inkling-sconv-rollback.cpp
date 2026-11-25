#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool test_sconv_hist_backend(ggml_backend_t backend) {
    constexpr int64_t l_cache = 3;
    constexpr int64_t n_conv  = 4;
    constexpr int64_t c0 = 8, c1 = 8, c2 = 16, c3 = 16;
    constexpr int64_t n_ch = c0 + c1 + c2 + c3;
    constexpr int64_t hist_w = 32;

    std::printf("  testing ggml_inkling_sconv_state_4_hist on backend %s ...\n", ggml_backend_name(backend));

    // Generate deterministic channel inputs for positions 0..39
    constexpr int T_TOTAL = 40;
    std::vector<float> all_in(T_TOTAL * n_ch);
    for (int t = 0; t < T_TOTAL; ++t) {
        for (int c = 0; c < n_ch; ++c) {
            all_in[t * n_ch + c] = 0.1f * std::sin(0.37f * (t + 1) + 0.19f * c) + 0.05f * (t % 7 - 3);
        }
    }
    std::vector<float> weight(n_conv * n_ch);
    for (int i = 0; i < n_conv * n_ch; ++i) {
        weight[i] = 0.25f * std::cos(0.41f * i);
    }

    auto eval_step = [&](std::vector<float> & state_vec,
                         std::vector<float> & hist_vec,
                         int pos_start,
                         int n_tok,
                         std::vector<float> & out_last) {
        const size_t ctx_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead() + 65536;
        ggml_init_params params = { ctx_size, nullptr, true };
        ggml_context * ctx = ggml_init(params);

        ggml_tensor * t_cell = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, l_cache * n_ch, 1);
        ggml_tensor * t_past = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, l_cache, n_ch, 1);
        ggml_tensor * t_hist = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ch, hist_w, 1);
        ggml_tensor * t_pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tok);
        ggml_tensor * t_w    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_conv, n_ch);
        ggml_tensor * t_cur  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ch, n_tok);
        ggml_tensor * t_g0   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c0, n_tok);
        ggml_tensor * t_g1   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c1, n_tok);
        ggml_tensor * t_g2   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c2, n_tok);
        ggml_tensor * t_g3   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c3, n_tok);

        ggml_tensor * t_out  = ggml_inkling_sconv(ctx, t_cur, t_past, t_w, nullptr, 0.5f);
        ggml_tensor * t_upd  = ggml_inkling_sconv_state_4_hist(ctx, t_cell, t_past, t_g0, t_g1, t_g2, t_g3, t_hist, t_pos);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, t_out);
        ggml_build_forward_expand(gf, t_upd);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(t_cell, state_vec.data(), 0, state_vec.size() * sizeof(float));
        ggml_backend_tensor_set(t_past, state_vec.data(), 0, state_vec.size() * sizeof(float));
        ggml_backend_tensor_set(t_hist, hist_vec.data(),  0, hist_vec.size()  * sizeof(float));
        ggml_backend_tensor_set(t_w,    weight.data(),    0, weight.size()    * sizeof(float));

        std::vector<int32_t> pos_v(n_tok);
        std::vector<float> cur_v(n_tok * n_ch);
        std::vector<float> v0(n_tok * c0), v1(n_tok * c1), v2(n_tok * c2), v3(n_tok * c3);
        for (int t = 0; t < n_tok; ++t) {
            pos_v[t] = pos_start + t;
            const float * src = all_in.data() + (pos_start + t) * n_ch;
            std::memcpy(cur_v.data() + t * n_ch, src, n_ch * sizeof(float));
            std::memcpy(v0.data() + t * c0, src,                   c0 * sizeof(float));
            std::memcpy(v1.data() + t * c1, src + c0,              c1 * sizeof(float));
            std::memcpy(v2.data() + t * c2, src + c0 + c1,         c2 * sizeof(float));
            std::memcpy(v3.data() + t * c3, src + c0 + c1 + c2,    c3 * sizeof(float));
        }
        ggml_backend_tensor_set(t_pos, pos_v.data(), 0, pos_v.size() * sizeof(int32_t));
        ggml_backend_tensor_set(t_cur, cur_v.data(), 0, cur_v.size() * sizeof(float));
        ggml_backend_tensor_set(t_g0,  v0.data(),    0, v0.size()    * sizeof(float));
        ggml_backend_tensor_set(t_g1,  v1.data(),    0, v1.size()    * sizeof(float));
        ggml_backend_tensor_set(t_g2,  v2.data(),    0, v2.size()    * sizeof(float));
        ggml_backend_tensor_set(t_g3,  v3.data(),    0, v3.size()    * sizeof(float));

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "graph_compute failed\n");
            std::exit(1);
        }
        ggml_backend_synchronize(backend);

        std::vector<float> out_all(n_tok * n_ch);
        ggml_backend_tensor_get(t_out,  out_all.data(),   0, out_all.size()   * sizeof(float));
        ggml_backend_tensor_get(t_cell, state_vec.data(), 0, state_vec.size() * sizeof(float));
        ggml_backend_tensor_get(t_hist, hist_vec.data(),  0, hist_vec.size()  * sizeof(float));

        out_last.assign(out_all.end() - n_ch, out_all.end());

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    };

    auto rollback_state = [&](std::vector<float> & state_vec, const std::vector<float> & hist_vec, int p0) {
        for (int c = 0; c < n_ch; ++c) {
            for (int l = 0; l < l_cache; ++l) {
                const int tp = p0 - (int) l_cache + l;
                state_vec[c * l_cache + l] = tp < 0 ? 0.0f : hist_vec[(tp % hist_w) * n_ch + c];
            }
        }
    };

    // Compute ground-truth sequential 1-by-1 outputs and states for all positions 0..39
    std::vector<std::vector<float>> ref_out(T_TOTAL);
    std::vector<float> ref_state(l_cache * n_ch, 0.0f);
    std::vector<float> ref_hist(hist_w * n_ch, 0.0f);
    for (int t = 0; t < T_TOTAL; ++t) {
        eval_step(ref_state, ref_hist, t, 1, ref_out[t]);
    }

    // 1) Near-zero rollback: decode [0, 1], rollback p0=1, decode [1]
    {
        std::vector<float> st(l_cache * n_ch, 0.0f), hi(hist_w * n_ch, 0.0f), out;
        eval_step(st, hi, 0, 2, out);
        rollback_state(st, hi, 1);
        eval_step(st, hi, 1, 1, out);
        for (int c = 0; c < n_ch; ++c) {
            if (std::fabs(out[c] - ref_out[1][c]) > 1e-6f) {
                std::fprintf(stderr, "near-zero rollback mismatch at c=%d: %g vs %g\n", c, out[c], ref_out[1][c]);
                return false;
            }
        }
    }

    // 2) Mid-batch rollback across k = 1..7 both in normal window (pos 3..10) and across ring wrap-around (pos 28..35)
    for (int base_pos : { 3, 28 }) {
        for (int k = 1; k <= 7; ++k) {
            std::vector<float> st(l_cache * n_ch, 0.0f), hi(hist_w * n_ch, 0.0f), out;
            eval_step(st, hi, 0, base_pos, out);     // prompt prefill [0 .. base_pos-1]
            eval_step(st, hi, base_pos, 8, out);     // 8-token verify batch [base_pos .. base_pos+7]
            rollback_state(st, hi, base_pos + k);    // roll back rejected tail [base_pos+k .. +inf)
            eval_step(st, hi, base_pos + k, 1, out); // next step at base_pos + k
            for (int c = 0; c < n_ch; ++c) {
                const float diff = std::fabs(out[c] - ref_out[base_pos + k][c]);
                if (diff > 1e-6f) {
                    std::fprintf(stderr, "rollback mismatch at base=%d k=%d c=%d: %g vs %g (diff=%g)\n",
                                 base_pos, k, c, out[c], ref_out[base_pos + k][c], diff);
                    return false;
                }
            }
        }
    }

    return true;
}

static bool test_full_model_rollback(const char * model_path) {
    std::printf("  testing full Inkling 2 Mini model rollback on %s ...\n", model_path);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    mparams.use_mmap     = false;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model %s\n", model_path);
        return false;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = 2048;
    cparams.n_batch      = 512;
    cparams.n_ubatch     = 512;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return false;
    }

    const common_context_seq_rm_type rm_type = common_context_can_seq_rm(ctx);
    std::printf("  common_context_can_seq_rm = %d (expected %d = COMMON_CONTEXT_SEQ_RM_TYPE_PART)\n",
                (int) rm_type, (int) COMMON_CONTEXT_SEQ_RM_TYPE_PART);
    if (rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_PART) {
        llama_free(ctx);
        llama_model_free(model);
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // 12 tokens: 4 prompt tokens + 8 draft verification tokens
    std::vector<llama_token> toks = { 100, 205, 310, 415, 520, 625, 730, 835, 940, 1045, 1150, 1255 };
    llama_memory_t mem = llama_get_memory(ctx);

    for (int k : { 1, 4, 7 }) {
        // Ground-truth sequential decode of toks[0 .. 4+k]
        llama_memory_clear(mem, true);
        if (llama_decode(ctx, llama_batch_get_one(toks.data(), 4)) != 0) {
            return false;
        }
        for (int i = 4; i <= 4 + k; ++i) {
            if (llama_decode(ctx, llama_batch_get_one(&toks[i], 1)) != 0) {
                return false;
            }
        }
        const float * logits_ref_ptr = llama_get_logits_ith(ctx, -1);
        std::vector<float> logits_ref(logits_ref_ptr, logits_ref_ptr + n_vocab);

        // Speculative batch decode of toks[4..11] (8 tokens), mid-batch seq_rm(0, 4+k, -1), then decode toks[4+k]
        llama_memory_clear(mem, true);
        if (llama_decode(ctx, llama_batch_get_one(toks.data(), 4)) != 0) {
            return false;
        }
        if (llama_decode(ctx, llama_batch_get_one(&toks[4], 8)) != 0) {
            return false;
        }
        const float * sync_ptr = llama_get_logits_ith(ctx, -1);
        (void) sync_ptr;

        if (!llama_memory_seq_can_rollback(mem, 0, 4 + k, 1024)) {
            std::fprintf(stderr, "llama_memory_seq_can_rollback(0, %d, 1024) returned false!\n", 4 + k);
            return false;
        }
        if (!llama_memory_seq_rm(mem, 0, 4 + k, -1)) {
            std::fprintf(stderr, "llama_memory_seq_rm(0, %d, -1) returned false!\n", 4 + k);
            return false;
        }
        if (llama_decode(ctx, llama_batch_get_one(&toks[4 + k], 1)) != 0) {
            return false;
        }
        const float * logits_rb = llama_get_logits_ith(ctx, -1);

        float max_diff = 0.0f;
        llama_token argmax_ref = 0, argmax_rb = 0;
        for (int32_t v = 0; v < n_vocab; ++v) {
            max_diff = std::max(max_diff, std::fabs(logits_ref[v] - logits_rb[v]));
            if (logits_ref[v] > logits_ref[argmax_ref]) argmax_ref = v;
            if (logits_rb[v]  > logits_rb[argmax_rb])   argmax_rb  = v;
        }
        std::printf("  k=%d: max_logit_diff = %.6g, argmax_ref = %d, argmax_rb = %d\n",
                    k, max_diff, (int) argmax_ref, (int) argmax_rb);
        // Note: toks[4..4+k-1] were evaluated with batch kernels (n=8) in the speculative run vs n=1 MV kernels in ref,
        // so minor BF16/FP32 accumulation rounding (~0.1-0.4 out of [-30, 30] across 48 visits) is expected while argmax matches.
        if (argmax_ref != argmax_rb || max_diff > 1.0f) {
            std::fprintf(stderr, "full model rollback mismatch at k=%d!\n", k);
            llama_free(ctx);
            llama_model_free(model);
            return false;
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    return true;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU ||
            ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU  ||
            ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (!backend) {
                continue;
            }
            const bool ok = test_sconv_hist_backend(backend);
            ggml_backend_free(backend);
            if (!ok) {
                return 1;
            }
        }
    }

    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0) {
            if (!test_full_model_rollback(argv[i + 1])) {
                return 1;
            }
        }
    }

    std::printf("PASS: test-inkling-sconv-rollback\n");
    return 0;
}
