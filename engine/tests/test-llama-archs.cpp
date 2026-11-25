#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "llama-ssd.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-model.h"
#include "../src/llama-model-loader.h"
#include "../src/ssd/dense-homes.h"
#include "../src/llama-model-saver.h"
#include "../src/models/deepseek41-engram.h"

#include <algorithm>
#include <functional>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

// random values from the tensor's name, the same for every tensor of that name
static void set_tensor_data_named(struct ggml_tensor * tensor, const std::string & name, size_t seed) {
    std::hash<std::string> hasher;
    std::mt19937 gen(hasher(name) + seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    set_tensor_data_named(tensor, tensor->name, *(const size_t *) userdata);
}

static void usage(char ** argv) {
    printf("Usage: %s [-a/--arch arch] [-s/--seed seed] [-v/--verbose] [--kv-grow] [--nextn] [--engram-layouts] [--dense-ssd]\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

// The DeepSeek-V4.1 fixture's Engram: two layers of 3-grams with 2 heads each, 4 buckets per layer, each
// layer's table laid out from row 0 (the engram.* layout)
static const std::array<uint32_t, 2> DSV41_ENGRAM_LAYERS     = { 1, 3 };
static const uint32_t                DSV41_ENGRAM_HEADS      = 2;
static const uint32_t                DSV41_ENGRAM_KEY_LENGTH = 64;
static const uint32_t                DSV41_ENGRAM_NGRAM      = 3;
static const std::vector<uint64_t>   DSV41_ENGRAM_MULTIPLIERS = { 76632096046245ull, 4839876093313ull, 35959672319349ull,
                                                                  67716810739261ull, 51510806800915ull, 30921347202721ull };
static const std::vector<uint64_t>   DSV41_ENGRAM_PRIMES     = { 101, 103, 107, 109,   113, 127, 131, 137 };
static const std::vector<uint64_t>   DSV41_ENGRAM_OFFSETS    = {   0, 101, 204, 311,     0, 113, 240, 371 };
// the tokens the compressed token map marks -1; the pad token is 2, compressed 2
static const std::array<llama_token, 2> DSV41_ENGRAM_DEAD_TOKENS = { 5, 70 };

// n_nextn NextN blocks follow the model's blocks in the file, which counts them in block_count and
// sizes its per-block metadata for them too.
static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe, const uint32_t n_ctx = 128, const uint32_t n_nextn = 0) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_GEMMA4) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 192;
        n_layer = 5; // need at least 5 for swa_pattern (every 5th is full_attention)
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_DEEPSEEK41) {
        // window-only, ratio-2 and ratio-1 layers; 8 heads to fill the 8 output groups
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 6;
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_MISTRAL4) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_INKLING2 || arch == LLM_ARCH_INKLING2_LOOP) {
        n_layer = 4;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    }

    const uint32_t n_embd_head = n_embd / n_head;
    const uint32_t n_block     = n_layer + n_nextn;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_block);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    if (n_nextn > 0) {
        ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS, n_nextn);
    }

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_block);
        for (uint32_t il = 0; il < n_block; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_block);
        for (uint32_t il = 0; il < n_block; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, arch == LLM_ARCH_DEEPSEEK41 ? uint32_t(1) : n_head);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK41) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head/2);
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_MISTRAL4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            arch == LLM_ARCH_DEEPSEEK41 ? uint32_t(64) : uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    } else if (arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_MUSE_SPARK ||
            arch == LLM_ARCH_INKLING2 || arch == LLM_ARCH_INKLING2_LOOP) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_block);
        for (uint32_t il = 0; il < n_block; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    if (arch == LLM_ARCH_INKLING2 || arch == LLM_ARCH_INKLING2_LOOP) {
        ms.add_kv(LLM_KV_ATTENTION_QK_CLIP,         10.0f);
        ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,       uint32_t(1));
        ms.add_kv(LLM_KV_ATTENTION_RELATIVE_LENGTH, uint32_t(16));
        ms.add_kv(LLM_KV_ATTENTION_RELATIVE_EXTENT, uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_RELATIVE_CLIP,   10.0f);
        ms.add_kv(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, uint32_t(16));
        ms.add_kv(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, 0.1f);
        ms.add_kv(LLM_KV_VOCAB_SIZE_UNPADDED,       n_vocab);
        ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,        10.0f);
        ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 10.0f);
        ms.add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING,   30.0f);
        if (arch == LLM_ARCH_INKLING2_LOOP) {
            ms.add_kv(LLM_KV_LOOP_START,     uint32_t(1));
            ms.add_kv(LLM_KV_LOOP_END,       uint32_t(3));
            ms.add_kv(LLM_KV_LOOP_COUNT,     uint32_t(2));
            ms.add_kv(LLM_KV_LOOP_GATE_RANK, uint32_t(16));
        }
    }

    if (arch == LLM_ARCH_MUSE_SPARK) {
        ms.add_kv(LLM_KV_POST_NORM_EPS,           1e-5f);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,      10000.0f);
        ms.add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING, 30.0f);
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE,         n_embd/2);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,    1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,     true);
    }

    if (arch == LLM_ARCH_QWEN4EXP) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_block, 4));
    }

    if (arch == LLM_ARCH_DEEPSEEK41) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           uint32_t(32));
        std::vector<uint32_t> compress_ratios = {0, 0, 2, 2, 1, 1};
        compress_ratios.resize(n_block, compress_ratios.back());
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,            compress_ratios);
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                     uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                     10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                  true);

        std::vector<int32_t> token_map(n_vocab);
        for (uint32_t i = 0; i < n_vocab; ++i) {
            token_map[i] = (int32_t) (i % 61);
        }
        // tokens no n-gram reaches across
        for (const llama_token dead : DSV41_ENGRAM_DEAD_TOKENS) {
            token_map[dead] = -1;
        }
        const LLM_KV kv(arch);
        ms.add_kv(LLM_KV_ENGRAM_LAYER_IDS,     std::vector<uint32_t>(DSV41_ENGRAM_LAYERS.begin(), DSV41_ENGRAM_LAYERS.end()));
        ms.add_kv(LLM_KV_ENGRAM_HEAD_COUNT,    DSV41_ENGRAM_HEADS);
        ms.add_kv(LLM_KV_ENGRAM_KEY_LENGTH,    DSV41_ENGRAM_KEY_LENGTH);
        ms.add_kv(LLM_KV_ENGRAM_MAX_NGRAM_SIZE, DSV41_ENGRAM_NGRAM);
        ms.add_kv(LLM_KV_ENGRAM_PAD_ID,        uint32_t(2));
        gguf_set_arr_data(ms.gguf_ctx, kv(LLM_KV_ENGRAM_MULTIPLIERS).c_str(), GGUF_TYPE_UINT64, DSV41_ENGRAM_MULTIPLIERS.data(), DSV41_ENGRAM_MULTIPLIERS.size());
        gguf_set_arr_data(ms.gguf_ctx, kv(LLM_KV_ENGRAM_PRIMES).c_str(),      GGUF_TYPE_UINT64, DSV41_ENGRAM_PRIMES.data(),      DSV41_ENGRAM_PRIMES.size());
        gguf_set_arr_data(ms.gguf_ctx, kv(LLM_KV_ENGRAM_OFFSETS).c_str(),     GGUF_TYPE_UINT64, DSV41_ENGRAM_OFFSETS.data(),     DSV41_ENGRAM_OFFSETS.size());
        gguf_set_arr_data(ms.gguf_ctx, kv(LLM_KV_ENGRAM_TOKEN_MAP).c_str(),   GGUF_TYPE_INT32,  token_map.data(),   token_map.size());
    }

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(1));
    // qwen4exp ropes indexer keys with the main rotary width, so its head can't be < n_rot
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,
              arch == LLM_ARCH_QWEN4EXP ? n_embd_head : uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,      uint32_t(8));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "no_vocab");
    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         arch == LLM_ARCH_DEEPSEEK41 ? uint32_t(4) : uint32_t(2)); // sqrtsoftplus : sigmoid
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,       1.0f);
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         (arch == LLM_ARCH_INKLING2 || arch == LLM_ARCH_INKLING2_LOOP) ? uint32_t(4) : uint32_t(3));

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static llama_context_params get_ctx_params(bool encode) {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    if (!encode) {
        ctx_params.n_ubatch = 64;
    }
    return ctx_params;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;

    const llama_context_params ctx_params = get_ctx_params(encode);

    size_t tmp = seed;
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tmp, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, tokens[pos], pos, {0}, true);
    }
    batch.n_tokens = n_tokens;
    if (encode) {
        if (llama_encode(lctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    llama_batch_free(batch);
    return ret;
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_MUSE_SPARK:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK41:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_INKLING2:
        case LLM_ARCH_INKLING2_LOOP:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GEMMA4) {
        return false; // FIXME @ngxson
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    if (arch == LLM_ARCH_INKLING2_DSPARK) {
        return false; // Drafter architecture requires target hidden taps; tested by test-inkling-dspark.
    }

    // FIXME some models are segfaulting with WebGPU:
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_KIMI_LINEAR) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    return true;
}

static int save_models(const llm_arch target_arch, const size_t seed, const ggml_log_level log_level, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_model_saver_supports_arch(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {});
            const std::string path = dir + "/" + llm_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: Saving %s model (%s) to %s...\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model_and_ctx.first.get(), path.c_str());
        }
    }
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return 0;
}

static int test_backends(const llm_arch target_arch, const size_t seed, const ggml_log_level log_level) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode) {}
    };

    std::vector<device_config> dev_configs;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, ggml_backend_dev_description(dev), LLAMA_SPLIT_MODE_LAYER);

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
    }

    bool all_ok = true;
    common_log_flush(common_log_main());
    printf("|%16s|%30s|%6s|%15s|%9s|\n", "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip");
    printf("|----------------|------------------------------|------|---------------|---------|\n");
    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float> logits_cpu;
            for (device_config & dc : dev_configs) {
                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};
                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
#if defined(GGML_USE_WEBGPU)
                skip = true; // FIXME
#endif // GGML_USE_WEBGPU
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llm_arch_supports_sm_tensor(arch)) {
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, dc.devs, dc.split_mode, encode);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            all_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }
                    }

                    // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                    //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                    // tmpfile() can be null on Windows without administrator privileges.
                    FILE * file = llama_model_saver_supports_arch(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR ? tmpfile() : nullptr;
                    if (file != nullptr) {
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        llama_model_saver ms = llama_model_saver(model_and_ctx_dev.first.get());
                        ms.add_kv_from_model();
                        ms.add_tensors_from_model();
                        ms.save(file);
                        rewind(file);

                        {
                            auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, dc.devs, dc.split_mode, encode);
                            const std::vector<float> logits_roundtrip = get_logits(
                                model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                            status_roundtrip = "\033[1;32mOK\033[0m";
                            GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                            for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                                if (logits_roundtrip[i] != logits_dev[i]) {
                                    all_ok = false;
                                    status_roundtrip = "\033[1;31mFAIL\033[0m";
                                    break;
                                }
                            }
                        }
                        // the loaded model does not own the stream, and an open stream per row exhausts the
                        // descriptor limit, after which every later round trip is silently skipped
                        fclose(file);
                    }
                }

                printf("|%16s|%30s|%6s|%15s %10s|%20s|\n", llm_arch_name(arch), dc.label.c_str(),
                    config_name.c_str(), status_nmse.c_str(), nmse_str, status_roundtrip.c_str());
            }
        }
    }
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return all_ok ? 0 : 1;
}

// loads the GGUF file PATH on the CPU as a server does: mapped, with the routed experts deferred
static std::pair<llama_model_ptr, llama_context_ptr> get_file_model_and_ctx(const std::string & path, bool encode) {
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    ggml_backend_dev_t no_devices[] = { nullptr };
    model_params.devices = no_devices;
    model_params.use_mmap = true;
    model_params.defer_experts = true;

    llama_model_ptr model(llama_model_load_from_file(path.c_str(), model_params));
    if (!model) {
        throw std::runtime_error("failed to load llama model from " + path);
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), get_ctx_params(encode)));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

// Writes the GGUF file PATH: the metadata KV and every tensor of MODEL. With NEXTN it also holds the
// tensors of one NextN block after the model's blocks, as KV must then count: each tensor of the model's
// last block once more, under the next block's index and zero, and a tensor only a NextN block has.
static void save_model_file(const llama_model * model, const gguf_context * kv, const std::string & path, bool nextn) {
    gguf_context_ptr out(gguf_init_empty());
    gguf_set_kv(out.get(), kv);
    for (const auto & named : model->tensors_by_name) {
        if (gguf_find_tensor(out.get(), named.first.c_str()) < 0) {
            gguf_add_tensor(out.get(), named.second);
        }
    }
    ggml_context_ptr ctx; // holds the NextN tensors until the file is written
    if (nextn) {
        const int32_t     n_layer = llama_model_n_layer(model);
        const std::string last    = "blk." + std::to_string(n_layer - 1) + ".";
        const std::string next    = "blk." + std::to_string(n_layer) + ".";
        std::vector<const ggml_tensor *> block;
        size_t bytes = 16 * sizeof(float);
        for (const auto & named : model->tensors_by_name) {
            if (named.first.rfind(last, 0) == 0 && std::find(block.begin(), block.end(), named.second) == block.end()) {
                block.push_back(named.second);
                bytes += ggml_nbytes(named.second);
            }
        }
        GGML_ASSERT(!block.empty());
        const ggml_init_params params = { bytes + (block.size() + 1) * (ggml_tensor_overhead() + GGML_MEM_ALIGN), nullptr, false };
        ctx.reset(ggml_init(params));
        std::vector<ggml_tensor *> head;
        for (const ggml_tensor * t : block) {
            ggml_tensor * copy = ggml_new_tensor(ctx.get(), t->type, ggml_n_dims(t), t->ne);
            ggml_set_name(copy, (next + (ggml_get_name(t) + last.size())).c_str());
            head.push_back(copy);
        }
        ggml_tensor * eh_proj = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 16);
        ggml_set_name(eh_proj, (next + "nextn.eh_proj.weight").c_str());
        head.push_back(eh_proj);
        for (ggml_tensor * t : head) {
            if (gguf_find_tensor(out.get(), ggml_get_name(t)) < 0) {
                memset(t->data, 0, ggml_nbytes(t));
                gguf_add_tensor(out.get(), t);
            }
        }
    }
    if (!gguf_write_to_file(out.get(), path.c_str(), false)) {
        throw std::runtime_error("failed to write " + path);
    }
}

// A file's NextN blocks are not part of the model. A model whose metadata counts one after its blocks
// has the same blocks and logits, bit for bit, as the model without it; loaded from a file that also
// holds the block's tensors, it has the same blocks, size and logits as loaded from the file without
// them. A file of NextN blocks alone does not load.
static int test_nextn(const llm_arch target_arch, const size_t seed, const ggml_log_level log_level) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);
    const std::string ok   = "\033[1;32mOK\033[0m";
    const std::string fail = "\033[1;31mFAIL\033[0m";

    bool all_ok = true;
    common_log_flush(common_log_main());
    printf("|%16s|%6s|%10s|%10s|\n", "Model arch.", "Config", "Metadata", "File");
    printf("|----------------|------|----------|----------|\n");
    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN || !arch_supported(arch)) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_INKLING2_LOOP) {
            continue; // inkling2_loop explicitly rejects nextn_predict_layers
        }
#if defined(GGML_USE_WEBGPU)
        continue; // FIXME
#endif // GGML_USE_WEBGPU

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string stem = std::string("test-llama-archs-nextn-") + llm_arch_name(arch) + (moe ? "-moe" : "-dense");
            const std::string path_trunk = stem + ".gguf";
            const std::string path_nextn = stem + "-nextn.gguf";
            bool meta_ok = false;
            bool file_ok = false;
            std::string error;
            try {
                gguf_context_ptr trunk      = get_gguf_ctx(arch, moe);
                gguf_context_ptr with_nextn = get_gguf_ctx(arch, moe, 128, 1);
                auto base = get_model_and_ctx(trunk.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                const int32_t n_layer = llama_model_n_layer(base.first.get());
                {
                    auto meta = get_model_and_ctx(with_nextn.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                    meta_ok = llama_model_n_layer(meta.first.get()) == n_layer &&
                        get_logits(meta.first.get(), meta.second.get(), tokens, encode) ==
                        get_logits(base.first.get(), base.second.get(), tokens, encode);
                }
                save_model_file(base.first.get(), trunk.get(), path_trunk, false);
                save_model_file(base.first.get(), with_nextn.get(), path_nextn, true);
                auto from_trunk = get_file_model_and_ctx(path_trunk, encode);
                auto from_nextn = get_file_model_and_ctx(path_nextn, encode);
                file_ok = llama_model_n_layer(from_nextn.first.get()) == n_layer &&
                    llama_model_size(from_nextn.first.get()) == llama_model_size(from_trunk.first.get()) &&
                    get_logits(from_nextn.first.get(), from_nextn.second.get(), tokens, encode) ==
                    get_logits(from_trunk.first.get(), from_trunk.second.get(), tokens, encode);
            } catch (const std::exception & e) {
                error = e.what();
            }
            std::remove(path_trunk.c_str());
            std::remove(path_nextn.c_str());
            all_ok = all_ok && meta_ok && file_ok;
            printf("|%16s|%6s|%21s|%21s| %s\n", llm_arch_name(arch), moe ? "MoE" : "Dense",
                (meta_ok ? ok : fail).c_str(), (file_ok ? ok : fail).c_str(), error.c_str());
        }
    }

    if (target_arch == LLM_ARCH_UNKNOWN || target_arch == LLM_ARCH_LLAMA) {
        gguf_context_ptr heads_only = get_gguf_ctx(LLM_ARCH_LLAMA, false, 128, 1);
        const LLM_KV kv(LLM_ARCH_LLAMA);
        const uint32_t n_block = gguf_get_val_u32(heads_only.get(), gguf_find_key(heads_only.get(), kv(LLM_KV_BLOCK_COUNT).c_str()));
        gguf_set_val_u32(heads_only.get(), kv(LLM_KV_NEXTN_PREDICT_LAYERS).c_str(), n_block);
        llama_model_params model_params = llama_model_default_params();
        model_params.progress_callback = silent_model_load_progress;
        ggml_backend_dev_t no_devices[] = { nullptr };
        model_params.devices = no_devices;
        size_t tmp = seed;
        const llama_model_ptr model(llama_model_init_from_user(heads_only.get(), set_tensor_data, &tmp, model_params));
        const bool refused = model == nullptr;
        printf("NextN blocks only: %s\n", (refused ? ok : fail).c_str());
        all_ok = all_ok && refused;
    }

    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return all_ok ? 0 : 1;
}

// decodes tokens [p0, p1) of sequence 0 in batches of n_batch, the last token's logits returned
static std::vector<float> decode_range(llama_context * lctx, const std::vector<llama_token> & tokens, uint32_t p0, uint32_t p1) {
    const uint32_t n_batch = llama_n_batch(lctx);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<float> ret;
    for (uint32_t p = p0; p < p1; p += n_batch) {
        const uint32_t n = std::min(n_batch, p1 - p);
        batch.n_tokens = 0;
        for (uint32_t i = 0; i < n; i++) {
            common_batch_add(batch, tokens[p + i], p + i, {0}, i == n - 1);
        }
        if (llama_decode(lctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to decode batch");
        }
    }
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(lctx)));
    const float * logits = llama_get_logits_ith(lctx, -1);
    ret.assign(logits, logits + n_vocab);
    llama_batch_free(batch);
    return ret;
}

static std::vector<uint8_t> seq_state(llama_context * lctx) {
    std::vector<uint8_t> ret(llama_state_seq_get_size(lctx, 0));
    if (llama_state_seq_get_data(lctx, ret.data(), ret.size(), 0) != ret.size()) {
        throw std::runtime_error("failed to save the sequence state");
    }
    return ret;
}

// A KV cache allocates its cells as they are used, 8192 at a time (llama_kv_cache::n_grow_default). A context that
// grows its cache past them while decoding gives the logits of one whose cache holds every cell from the start (two
// sequences in streams of their own); the cells it held keep what they held as it grows; and a sequence's state
// restored into a fresh context, which grows its cache to take it, gives the logits it gives restored into a full one.
static int test_kv_grow(const llm_arch target_arch, const size_t seed, const ggml_log_level log_level) {
    struct user_data_t {
        ggml_log_callback callback;
        void * user_data;
        ggml_log_level min_level;
        int grows = 0; // the cache's growths and trims it logged (llama_kv_cache::grow_cells, trim_cells)
        int trims = 0;
    };
    user_data_t ud;
    llama_log_get(&ud.callback, &ud.user_data);
    ud.min_level = log_level;
    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        user_data_t * ud = (user_data_t *) user_data;
        ud->grows += strncmp(text, "grow_cells:", 11) == 0;
        ud->trims += strncmp(text, "trim_cells:", 11) == 0;
        ud->callback(level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG, text, ud->user_data);
    }, &ud);

    const uint32_t n_ctx = 16384;
    const uint32_t n0    = 8000; // fits the first 8192 cells
    const uint32_t n1    = 9000; // does not

    // the CPU, and every GPU with the model and its cache on it
    std::vector<ggml_backend_dev_t> devices = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) };
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            devices.push_back(dev);
        }
    }

    int failures = 0;
    for (ggml_backend_dev_t device : devices)
    for (const llm_arch arch : { LLM_ARCH_LLAMA, LLM_ARCH_QWEN35MOE, LLM_ARCH_DEEPSEEK41, LLM_ARCH_QWEN4EXP }) {
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe_mandatory(arch), n_ctx);

        llama_model_params model_params = llama_model_default_params();
        model_params.progress_callback = silent_model_load_progress;
        std::vector<ggml_backend_dev_t> devs = { device, nullptr };
        model_params.devices = devs.data();
        size_t tmp = seed;
        llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tmp, model_params));
        if (!model) {
            throw std::runtime_error("failed to create llama model");
        }
        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        const std::vector<llama_token> tokens = get_tokens(n1 + 1, n_vocab, seed);

        const auto make_ctx = [&](uint32_t n_seq_max) {
            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx           = n_ctx*n_seq_max;
            ctx_params.n_batch         = 1000;
            ctx_params.n_ubatch        = 250;
            ctx_params.n_seq_max       = n_seq_max;
            ctx_params.kv_unified      = false;
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            ctx_params.n_threads       = 4;
            ctx_params.n_threads_batch = 4;
            llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
            if (!lctx) {
                throw std::runtime_error("failed to create llama context");
            }
            return lctx;
        };

        const std::string label = std::string(llm_arch_name(arch)) + " on " + ggml_backend_dev_name(device);
        const char * name = label.c_str();
        bool ok = true;

        // grows from 8192 cells to 16384 between n0 and n1, in place; a device without growable buffers (the CPU on
        // Windows, Vulkan without sparse residency buffers) allocates every cell at once
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        auto * growable_buft = (ggml_backend_buffer_type_t (*)(ggml_backend_dev_t))
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_buffer_type_growable");
        const bool grows = growable_buft && growable_buft(device);
        llama_context_ptr grown = make_ctx(1);
        decode_range(grown.get(), tokens, 0, n0);
        const std::vector<uint8_t> state_n0 = seq_state(grown.get());
        ud.grows = 0;
        decode_range(grown.get(), tokens, n0, n1);
        LOG_INF("%s: %s: grew: %s (%s)\n", __func__, name, ud.grows > 0 ? "yes" : "no", grows ? "growable" : "not growable");
        ok = ok && (ud.grows > 0) == grows;
        const std::vector<uint8_t> state_n1 = seq_state(grown.get());
        const std::vector<float> logits_grown = decode_range(grown.get(), tokens, n1, n1 + 1);

        // every cell from the start: a sequence in a stream of its own
        llama_context_ptr full = make_ctx(2);
        decode_range(full.get(), tokens, 0, n1);
        const std::vector<uint8_t> state_full = seq_state(full.get());
        const std::vector<float> logits_full = decode_range(full.get(), tokens, n1, n1 + 1);
        LOG_INF("%s: %s: grown vs full cache: %s (nmse %.3g)\n", __func__, name,
                logits_grown == logits_full ? "identical" : "DIFFERENT", nmse(logits_grown, logits_full));
        ok = ok && logits_grown == logits_full;

        // the sequence's state restored into a cache of 8192 cells, and the full cache's into one like it (a restored
        // DeepSeek-V4.1 sequence does not give the logits of the one saved, so the two restores compare)
        std::vector<float> logits_restored[2];
        for (uint32_t i = 0; i < 2; i++) {
            llama_context_ptr restored = make_ctx(i + 1);
            const std::vector<uint8_t> & state = i == 0 ? state_n1 : state_full;
            if (llama_state_seq_set_data(restored.get(), state.data(), state.size(), 0) != state.size()) {
                throw std::runtime_error("failed to restore the sequence state");
            }
            logits_restored[i] = decode_range(restored.get(), tokens, n1, n1 + 1);
        }
        LOG_INF("%s: %s: restored into a grown vs a full cache: %s (nmse %.3g)\n", __func__, name,
                logits_restored[0] == logits_restored[1] ? "identical" : "DIFFERENT",
                nmse(logits_restored[0], logits_restored[1]));
        ok = ok && logits_restored[0] == logits_restored[1];

        // the first n0 cells after the growth, where the cache can drop the later ones; then a request of one token
        // gives back the cells past 8192, and the cache decodes as one that never grew
        if (llama_memory_seq_rm(llama_get_memory(grown.get()), 0, n0, -1)) {
            const bool same = seq_state(grown.get()) == state_n0;
            LOG_INF("%s: %s: first %u cells after growth: %s\n", __func__, name, n0, same ? "identical" : "DIFFERENT");
            ok = ok && same;

            ud.trims = 0;
            if (!llama_ssd_prefill_phase(grown.get(), true, n0 + 1, 1)) {
                throw std::runtime_error("failed to begin a request");
            }
            LOG_INF("%s: %s: trimmed: %s\n", __func__, name, ud.trims > 0 ? "yes" : "no");
            ok = ok && (ud.trims > 0) == grows;
            const std::vector<float> logits_trimmed = decode_range(grown.get(), tokens, n0, n0 + 1);
            llama_context_ptr small = make_ctx(1);
            decode_range(small.get(), tokens, 0, n0);
            const bool same_trimmed = logits_trimmed == decode_range(small.get(), tokens, n0, n0 + 1);
            LOG_INF("%s: %s: trimmed vs never grown: %s\n", __func__, name, same_trimmed ? "identical" : "DIFFERENT");
            ok = ok && same_trimmed;
        } else {
            // a memory that cannot drop part of a sequence (a recurrent state) starts over: a request of n0 tokens
            // gives back the cells past 8192, and the cache decodes them as one that never grew
            llama_memory_clear(llama_get_memory(grown.get()), true);
            ud.trims = 0;
            if (!llama_ssd_prefill_phase(grown.get(), true, n0, n0)) {
                throw std::runtime_error("failed to begin a request");
            }
            LOG_INF("%s: %s: trimmed: %s\n", __func__, name, ud.trims > 0 ? "yes" : "no");
            ok = ok && (ud.trims > 0) == grows;
            decode_range(grown.get(), tokens, 0, n0);
            const std::vector<float> logits_trimmed = decode_range(grown.get(), tokens, n0, n0 + 1);
            llama_context_ptr small = make_ctx(1);
            decode_range(small.get(), tokens, 0, n0);
            const bool same_trimmed = logits_trimmed == decode_range(small.get(), tokens, n0, n0 + 1);
            LOG_INF("%s: %s: trimmed vs never grown: %s\n", __func__, name, same_trimmed ? "identical" : "DIFFERENT");
            ok = ok && same_trimmed;
        }

        printf("%s: %s: %s\n", __func__, name, ok ? "OK" : "FAIL");
        failures += ok ? 0 : 1;
    }

    llama_log_set(ud.callback, ud.user_data);
    return failures == 0 ? 0 : 1;
}

// A model without routed experts in an SSD context (settings, no expert source) computes as it does without one: its
// prompt and its next tokens give the same logits, the CPU's work memory taken from the context's host budget, the
// prompt's graph holding the one output the request asks for.
static int test_dense_ssd(const size_t seed, const ggml_log_level log_level) {
    struct user_data_t {
        ggml_log_callback callback;
        void * user_data;
        ggml_log_level min_level;
    };
    user_data_t ud;
    llama_log_get(&ud.callback, &ud.user_data);
    ud.min_level = log_level;
    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        user_data_t * ud = (user_data_t *) user_data;
        ud->callback(level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG, text, ud->user_data);
    }, &ud);

    const uint32_t n_ctx    = 1024;
    const uint32_t n_prompt = 600; // more than one batch
    const uint32_t n_next   = 4;

    char error[512] = {};
    std::unique_ptr<llama_ssd_options, decltype(&llama_ssd_options_free)> options(
        llama_ssd_options_create(nullptr, nullptr, 0, nullptr, error, sizeof(error)), llama_ssd_options_free);
    if (!options) {
        throw std::runtime_error(std::string("failed to create SSD options: ") + error);
    }

    // every GPU with the model and its cache on it; the CPU where there is none (with a GPU the SSD context runs the
    // layers on it, wherever the model keeps their weights)
    std::vector<ggml_backend_dev_t> devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            devices.push_back(dev);
        }
    }
    if (devices.empty()) {
        devices.push_back(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    }

    int failures = 0;
    for (ggml_backend_dev_t device : devices) {
        gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_LLAMA, false, n_ctx);
        // F16 layer matrices: the CPU converts their activations first, in the work memory it takes from the
        // context's host budget (F32 ones need none)
        const LLM_KV kv(LLM_ARCH_LLAMA);
        const uint32_t n_layer = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), kv(LLM_KV_BLOCK_COUNT).c_str()));
        for (uint32_t il = 0; il < n_layer; il++) {
            for (const char * name : { "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_down", "ffn_up" }) {
                ggml_tensor t;
                memset(&t, 0, sizeof(ggml_tensor));
                t.type = GGML_TYPE_F16;
                ggml_format_name(&t, "blk.%" PRIu32 ".%s.weight", il, name);
                gguf_add_tensor(gguf_ctx.get(), &t);
            }
        }
        const bool on_cpu = ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU;
        // the model, its layers' matrices homed in buft when there is one, or homed for the SSD context to place
        const auto load = [&](ggml_backend_buffer_type_t buft, const char * pattern, bool ssd_homes = false) {
            llama_model_params model_params = llama_model_default_params();
            model_params.progress_callback = silent_model_load_progress;
            model_params.ssd_homes = ssd_homes;
            // the CPU as a CPU-only server has it: no device list, the context's own CPU backend running the graph
            std::vector<ggml_backend_dev_t> devs = { device, nullptr };
            model_params.devices = on_cpu ? nullptr : devs.data();
            model_params.n_gpu_layers = on_cpu ? 0 : 999;
            const llama_model_tensor_buft_override homes[] = { { pattern, buft }, { nullptr, nullptr } };
            model_params.tensor_buft_overrides = buft ? homes : nullptr;
            size_t tmp = seed;
            llama_model_ptr loaded(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tmp, model_params));
            if (!loaded) {
                throw std::runtime_error("failed to create llama model");
            }
            return loaded;
        };
        llama_model_ptr model = load(nullptr, nullptr);
        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        const std::vector<llama_token> tokens = get_tokens(n_prompt + n_next, n_vocab, seed);

        // a request as the server makes it: its prompt between the phase's begin and the prefill's end, then its
        // next tokens one at a time; placed: after the context is made, before its first batch
        const auto run = [&](const llama_model * m, const llama_ssd_options * ssd,
                             const std::function<void()> & placed = {}) {
            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx           = n_ctx;
            ctx_params.n_batch         = 256;
            ctx_params.n_ubatch        = 256;
            ctx_params.n_seq_max       = 1;
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            ctx_params.n_threads       = 4;
            ctx_params.n_threads_batch = 4;
            ctx_params.ssd_options     = ssd;
            llama_context_ptr lctx(llama_init_from_model(const_cast<llama_model *>(m), ctx_params));
            if (!lctx) {
                throw std::runtime_error("failed to create llama context");
            }
            if (placed) {
                placed();
            }
            if (!llama_ssd_prefill_phase(lctx.get(), true, n_prompt, n_prompt)) {
                throw std::runtime_error("failed to begin a request");
            }
            std::vector<float> logits = decode_range(lctx.get(), tokens, 0, n_prompt);
            if (!llama_ssd_prefill_phase(lctx.get(), false, 0, 0)) {
                throw std::runtime_error("failed to end the prompt");
            }
            for (uint32_t p = n_prompt; p < n_prompt + n_next; ++p) {
                const std::vector<float> next = decode_range(lctx.get(), tokens, p, p + 1);
                logits.insert(logits.end(), next.begin(), next.end());
            }
            return logits;
        };
        const std::vector<float> plain = run(model.get(), nullptr);
        const auto check = [&](const char * what, const std::vector<float> & got) {
            const bool ok = plain == got;
            printf("%s: llama on %s: %s vs no SSD context: %s (nmse %.3g)\n", __func__, ggml_backend_dev_name(device),
                    what, ok ? "identical" : "DIFFERENT", nmse(plain, got));
            failures += ok ? 0 : 1;
        };
        check("SSD context", run(model.get(), options.get()));
        if (!on_cpu) {
            // the second layer's matrices in host memory, plain and pinned: the GPU runs their products, their
            // bytes streamed in for each graph, and gives the logits it gives with them in its memory
            const char * layer = "blk\\.1\\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_down|ffn_up)\\.weight";
            check("matrices streamed from host memory",
                    run(load(ggml_backend_cpu_buffer_type(), layer).get(), options.get()));
            ggml_backend_buffer_type_t pinned = ggml_backend_dev_host_buffer_type(device);
            if (pinned) {
                check("matrices streamed from pinned memory", run(load(pinned, layer).get(), options.get()));
            }
            // the layers' matrices homed for the SSD context: the GPU holds every layer its memory has room for (all
            // of them here), and streams the ones it gives back
            llama_model_ptr homed = load(nullptr, nullptr, true);
            llama_dense_homes * homes = homed->dense_homes();
            if (llama_ssd_device_streams(device)) {
                if (!homes || homes->held_bytes() != 0 || homes->streamed_bytes() == 0) {
                    throw std::runtime_error("the layers' matrices were not homed in host memory");
                }
                check("matrices homed and held", run(homed.get(), options.get(), [&] {
                    if (homes->streamed_bytes() != 0) {
                        throw std::runtime_error("the GPU did not hold every layer");
                    }
                }));
                check("one layer given back", run(homed.get(), options.get(), [&] { homes->give_back(1); }));
                check("every layer given back", run(homed.get(), options.get(), [&] {
                    homes->give_back(SIZE_MAX);
                    if (homes->held_bytes() != 0) {
                        throw std::runtime_error("a layer was not given back");
                    }
                }));
            } else if (homes) {
                throw std::runtime_error("layers homed for a GPU that does not stream");
            }
        }
    }

    llama_log_set(ud.callback, ud.user_data);
    return failures == 0 ? 0 : 1;
}

// unsloth's DeepSeek-V4.1 GGUFs lay the Engram out another way: ple.* keys, every layer's table in one
// per_layer_token_embd, stored as MXFP8, whose head offsets run on across the layers, and for each layer
// the product of engram_q and engram_k as engram_gate. The same weights in either layout give the same logits.
struct dsv41_engram_layouts_data {
    size_t seed;
    std::vector<uint8_t> table; // MXFP8 rows of every Engram layer, the first layer's first
};

static const size_t DSV41_ENGRAM_ROW_BYTES = DSV41_ENGRAM_KEY_LENGTH / 32 * 33; // MXFP8: 33 bytes for 32 values

// rows of Engram layer e's table: its last bucket's end
static uint64_t dsv41_engram_rows(size_t e) {
    const size_t n_cols = (DSV41_ENGRAM_NGRAM - 1) * DSV41_ENGRAM_HEADS;
    uint64_t rows = 0;
    for (size_t b = 0; b < n_cols; ++b) {
        rows = std::max(rows, DSV41_ENGRAM_OFFSETS[e*n_cols + b] + DSV41_ENGRAM_PRIMES[e*n_cols + b]);
    }
    return rows;
}

static void set_engram_layouts_data(struct ggml_tensor * tensor, void * userdata) {
    const auto & ud = *(const dsv41_engram_layouts_data *) userdata;
    const std::string name = tensor->name;
    if (name == "per_layer_token_embd.weight") {
        GGML_ASSERT(tensor->type == GGML_TYPE_MXFP8 && ggml_nbytes(tensor) == ud.table.size());
        ggml_backend_tensor_set(tensor, ud.table.data(), 0, ud.table.size());
        return;
    }
    uint64_t first_row = 0;
    for (size_t e = 0; e < DSV41_ENGRAM_LAYERS.size(); first_row += dsv41_engram_rows(e++)) {
        if (name == "blk." + std::to_string(DSV41_ENGRAM_LAYERS[e]) + ".engram_embd.weight") {
            // the layer's rows of the MXFP8 table, decoded
            std::vector<float> values(ggml_nelements(tensor));
            ggml_get_type_traits(GGML_TYPE_MXFP8)->to_float(
                ud.table.data() + first_row*DSV41_ENGRAM_ROW_BYTES, values.data(), (int64_t) values.size());
            ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
            return;
        }
    }
    const std::string k = ".engram_k.weight", gate = ".engram_gate.";
    if (name.size() > k.size() && name.compare(name.size() - k.size(), k.size(), k) == 0) {
        // q * 1 is q: the per-layer model's q is the one-table model's gate
        const std::vector<float> ones(ggml_nelements(tensor), 1.0f);
        ggml_backend_tensor_set(tensor, ones.data(), 0, ggml_nbytes(tensor));
        return;
    }
    std::string source = name;
    const size_t at = source.find(gate);
    if (at != std::string::npos) {
        source.replace(at, gate.size(), ".engram_q.");
    }
    set_tensor_data_named(tensor, source, ud.seed);
}

// The model's files as llama_model_load_from_splits takes them (the model, then its --shard/--ple files), with the
// extra-shard rules of upstream llama.cpp 9581154a: a PLE-only file (one tensor, per_layer_token_embd.weight) replaces
// the model's table or stands for a numbered shard holding only the table, which is then skipped, present or missing,
// as long as the declared tensor count accounts for it; a given file named as a numbered shard replaces that shard
// wherever it lies; any other extra is appended; an extra's split numbering is ignored; a tensor found twice, but for
// the table an extra replaces, is refused. Every load that succeeds gives the logits of the model in one file.
static void test_engram_files(const llama_model * model, const gguf_context * kv,
                              llama_model_params model_params, llama_context_params ctx_params,
                              const std::vector<llama_token> & tokens, const std::vector<float> & logits) {
    namespace fs = std::filesystem;
    struct files_t {
        fs::path directory;
        files_t() {
            std::random_device entropy;
            do {
                directory = fs::temp_directory_path() / ("test-dsv41-ple-" + std::to_string(entropy()));
            } while (!fs::create_directory(directory));
        }
        ~files_t() {
            std::error_code ignored;
            fs::remove_all(directory, ignored);
        }
    } files;
    fs::create_directory(files.directory / "tables");
    fs::create_directory(files.directory / "elsewhere");

    std::vector<const ggml_tensor *> all, backbone;
    const ggml_tensor * ple = nullptr;
    for (const auto & named : model->tensors_by_name) {
        if (std::find(all.begin(), all.end(), named.second) != all.end()) {
            continue;
        }
        all.push_back(named.second);
        if (named.first == "per_layer_token_embd.weight") {
            ple = named.second;
        } else {
            backbone.push_back(named.second);
        }
    }
    GGML_ASSERT(ple && backbone.size() > 1);

    // NAME: TENSORS, the model's KV with METADATA, and the split keys with SPLIT_NO >= 0
    auto write = [&](const std::string & name, const std::vector<const ggml_tensor *> & tensors,
                     bool metadata, int split_no = -1, int split_count = 2, int tensor_count = -1) {
        gguf_context_ptr out(gguf_init_empty());
        if (metadata) {
            gguf_set_kv(out.get(), kv);
        }
        if (split_no >= 0) {
            gguf_set_val_u16(out.get(), "split.no", split_no);
            gguf_set_val_u16(out.get(), "split.count", split_count);
            gguf_set_val_i32(out.get(), "split.tensors.count", tensor_count < 0 ? (int) all.size() : tensor_count);
        }
        for (const ggml_tensor * tensor : tensors) {
            gguf_add_tensor(out.get(), tensor);
        }
        const std::string path = (files.directory / name).string();
        if (!gguf_write_to_file(out.get(), path.c_str(), false)) {
            throw std::runtime_error("failed to write " + path);
        }
        return path;
    };
    // PATH cut to its header and the first byte of its data: what it holds cannot be read
    auto truncate = [](const std::string & path) {
        gguf_init_params read_params = { true, nullptr };
        gguf_context_ptr meta(gguf_init_from_file(path.c_str(), read_params));
        GGML_ASSERT(meta);
        fs::resize_file(path, gguf_get_data_offset(meta.get()) + 1);
    };

    int accepted = 0, rejected = 0;
    // mode: 0 mmap, 1 no mmap, 2 mlock, 3 whole-tensor checks
    auto check = [&](const char * label, const std::vector<std::string> & paths, bool valid, int mode = 0,
                     ggml_type type = GGML_TYPE_MXFP8) {
        std::vector<const char *> c_paths;
        for (const auto & path : paths) {
            c_paths.push_back(path.c_str());
        }
        auto params = model_params;
        params.use_mmap = mode != 1;
        params.use_mlock = mode == 2;
        params.check_tensors = mode == 3;
        llama_model_ptr loaded(c_paths.size() == 1 ? llama_model_load_from_file(c_paths[0], params)
                                                   : llama_model_load_from_splits(c_paths.data(), c_paths.size(), params));
        if (bool(loaded) != valid) {
            throw std::runtime_error(std::string(label) + (valid ? ": load failed" : ": invalid input accepted"));
        }
        if (!valid) {
            ++rejected;
            return;
        }
        if (!loaded->per_layer_tok_embd || loaded->per_layer_tok_embd->type != type) {
            throw std::runtime_error(std::string(label) + ": the table is not the one expected");
        }
        llama_context_ptr loaded_ctx(llama_init_from_model(loaded.get(), ctx_params));
        if (!loaded_ctx || get_logits(loaded.get(), loaded_ctx.get(), tokens) != logits) {
            throw std::runtime_error(std::string(label) + ": logits changed in mode " + std::to_string(mode));
        }
        ++accepted;
    };

    const std::string full = write("embedded.gguf", all, true);
    const std::string base = write("backbone.gguf", backbone, true);
    // ISTA-DASLab's table keeps the numbering of the shard it stands for, and no architecture
    const std::string table = write("tables/arbitrary-name.gguf", {ple}, false, 5, 6);

    // a table of other bytes with the same values: an F32 decode, and one with rows past the last bucket's
    // room for the tables below: 1 + (1 and 7 rows) + 1/2 + 1 of the table's values
    const size_t table_bytes = ggml_nelements(ple)*sizeof(float);
    const ggml_init_params init = { 4*table_bytes + 8*ple->ne[0]*sizeof(float) + 8*ggml_tensor_overhead(), nullptr, false };
    ggml_context_ptr tensor_ctx(ggml_init(init));
    auto make_table = [&](int64_t width, int64_t rows) {
        ggml_tensor * tensor = ggml_new_tensor_2d(tensor_ctx.get(), GGML_TYPE_F32, width, rows);
        ggml_set_name(tensor, "per_layer_token_embd.weight");
        memset(tensor->data, 0, ggml_nbytes(tensor));
        return tensor;
    };
    ggml_tensor * f32 = make_table(ple->ne[0], ple->ne[1]);
    ggml_get_type_traits(ple->type)->to_float(ple->data, (float *) f32->data, ggml_nelements(ple));
    const std::string f32_table = write("tables/decoded.gguf", {f32}, false);
    ggml_tensor * padded = make_table(ple->ne[0], ple->ne[1] + 7);
    memcpy(padded->data, f32->data, ggml_nbytes(f32));
    const std::string padded_table = write("tables/padded.gguf", {padded}, false);

    for (int mode = 0; mode < 4; ++mode) {
        check("embedded", {full}, true, mode);
        check("single file plus its table", {base, table}, true, mode);
        check("table replaced by an extra", {full, f32_table}, true, mode, GGML_TYPE_F32);
    }
    check("table with rows past the last bucket", {base, padded_table}, true, 0, GGML_TYPE_F32);
    check("single file without its table", {base}, false);

    // a numbered model whose second shard holds only the table
    const std::string first = write("split-00001-of-00002.gguf", backbone, true, 0);
    const std::string second = write("split-00002-of-00002.gguf", {ple}, false, 1);
    check("complete numbered model", {first}, true);
    check("numbered shard given by its own name", {first, second}, true);
    check("numbered table replaced by an extra", {first, f32_table}, true, 0, GGML_TYPE_F32);
    truncate(second);
    check("numbered table shard unreadable", {first}, false);
    check("numbered table shard skipped for an extra", {first, table}, true);
    fs::remove(second);
    for (int mode = 0; mode < 4; ++mode) {
        check("numbered table shard missing", {first, table}, true, mode);
    }
    check("numbered table shard missing without an extra", {first}, false);
    write("split-00001-of-00002.gguf", backbone, true, 0, 2, (int) all.size() + 1);
    check("missing shard the tensor count does not account for", {first, table}, false);
    const std::string three = write("three-00001-of-00003.gguf", backbone, true, 0, 3);
    check("two missing shards", {three, table}, false);

    // a numbered model whose table sits in the middle shard, and whose last shard lies in another directory
    auto most = backbone;
    most.pop_back();
    const std::string middle = write("middle-00001-of-00003.gguf", most, true, 0, 3);
    const std::string last = write("elsewhere/middle-00003-of-00003.gguf", {backbone.back()}, false, 2, 3);
    check("shard elsewhere not given", {middle, table}, false);
    check("shard elsewhere given by its name", {middle, last, table}, true);
    check("shard elsewhere given by its name, without mmap", {middle, last, table}, true, 1);
    check("shard elsewhere given, table missing", {middle, last}, false);

    // other extras are appended to the model: a tensor the model lacks, but no tensor twice
    check("single file plus an extra with its last tensor", {write("most.gguf", most, true),
          write("extra.gguf", {backbone.back()}, false), table}, true);
    check("extra repeating a model tensor", {base, write("other.gguf", {backbone.front()}, false), table}, false);
    check("table and another tensor in one extra", {base, write("mixed.gguf", {ple, backbone.front()}, false)}, false);
    check("two tables", {base, table, f32_table}, false);
    check("one table given twice", {base, table, (files.directory / "tables" / "." / "arbitrary-name.gguf").string()},
          false);
    check("model given as its own extra", {full, (files.directory / "." / "embedded.gguf").string()}, false);
    check("missing extra", {base, (files.directory / "absent.gguf").string()}, false);

    check("table too narrow", {base, write("width.gguf", {make_table(ple->ne[0] / 2, ple->ne[1])}, false)}, false);
    check("table too short", {base, write("rows.gguf", {make_table(ple->ne[0], ple->ne[1] - 1)}, false)}, false);
    const std::string cut = write("tables/cut.gguf", {ple}, false);
    truncate(cut);
    check("table cut short", {base, cut}, false);
    printf("%s: %d identical-logit loads, %d invalid inputs refused\n", __func__, accepted, rejected);
}

// One Engram in each layout a GGUF stores: engram.*, ple.* (one table, offsets running on across the layers, the pad
// named by its token) and antirez's e4m3_e8m0_32_row264 (8 heads of 256 over 4-grams in two layers, U32 arrays, each
// layer's offsets from its prime sums). Each reader gives the same parameters, and the one hash the same rows: ple.*'s
// offset by the rows of the layers before.
static int test_engram_readers(const size_t seed) {
    const llm_arch arch = LLM_ARCH_DEEPSEEK41;
    const LLM_KV kv(arch);
    const uint32_t n_gram = 4, n_head = 8, key = 256, n_cols = (n_gram - 1)*n_head;
    const std::array<uint32_t, 2> layers = { 1, 3 };
    const uint32_t vocab = 128, compressed = 97, pad = 2;

    std::mt19937_64 gen(seed);
    std::vector<int32_t> token_map(vocab);
    for (auto & id : token_map) {
        id = int32_t(gen() % compressed);
    }
    uint32_t pad_token = vocab - 1; // a token other than the pad's compressed id stands for it
    token_map[pad_token] = (int32_t) pad;
    std::vector<uint64_t> mult(layers.size()*n_gram);
    for (auto & m : mult) {
        m = (gen() % (uint64_t(1) << 37)) | 1; // odd, and below INT64_MAX / compressed
    }
    std::vector<uint64_t> primes(layers.size()*n_cols), offsets(primes.size()), global(primes.size());
    std::vector<uint32_t> rows(layers.size());
    uint64_t base = 0;
    for (size_t e = 0; e < layers.size(); ++e) {
        uint64_t offset = 0;
        for (uint32_t c = 0; c < n_cols; ++c) {
            const size_t k = e*n_cols + c;
            primes[k]  = 1009 + gen() % 50000;
            offsets[k] = offset;
            global[k]  = base + offset;
            offset    += primes[k];
        }
        rows[e] = (uint32_t) offset;
        base   += offset;
    }

    auto layout = [&](int which) {
        gguf_context_ptr g = get_gguf_ctx(arch, moe_mandatory(arch));
        for (const llm_kv k : { LLM_KV_ENGRAM_LAYER_IDS, LLM_KV_ENGRAM_HEAD_COUNT, LLM_KV_ENGRAM_KEY_LENGTH,
                                LLM_KV_ENGRAM_MAX_NGRAM_SIZE, LLM_KV_ENGRAM_PAD_ID, LLM_KV_ENGRAM_MULTIPLIERS,
                                LLM_KV_ENGRAM_PRIMES, LLM_KV_ENGRAM_OFFSETS, LLM_KV_ENGRAM_TOKEN_MAP }) {
            GGML_ASSERT(gguf_remove_key(g.get(), kv(k).c_str()) >= 0);
        }
        gguf_context * c = g.get();
        const std::vector<uint32_t> ids(layers.begin(), layers.end());
        if (which == 1) {
            gguf_set_arr_data(c, kv(LLM_KV_PLE_LAYERS).c_str(), GGUF_TYPE_UINT32, ids.data(), ids.size());
            gguf_set_val_u32 (c, kv(LLM_KV_PLE_HEADS_PER_NGRAM).c_str(), n_head);
            gguf_set_val_u32 (c, kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER).c_str(), key);
            gguf_set_val_u32 (c, kv(LLM_KV_PLE_NGRAM_SIZE).c_str(), n_gram);
            gguf_set_val_u32 (c, kv(LLM_KV_ENGRAM_PAD_TOKEN_ID).c_str(), pad_token);
            gguf_set_arr_data(c, kv(LLM_KV_PLE_LAYER_MULTIPLIERS).c_str(), GGUF_TYPE_UINT64, mult.data(), mult.size());
            gguf_set_arr_data(c, kv(LLM_KV_PLE_HEAD_VOCAB_SIZES).c_str(), GGUF_TYPE_UINT64, primes.data(), primes.size());
            gguf_set_arr_data(c, kv(LLM_KV_PLE_HEAD_OFFSETS).c_str(), GGUF_TYPE_UINT64, global.data(), global.size());
            gguf_set_arr_data(c, kv(LLM_KV_ENGRAM_TOKEN_MAP).c_str(), GGUF_TYPE_INT32, token_map.data(), token_map.size());
            return g;
        }
        gguf_set_arr_data(c, kv(LLM_KV_ENGRAM_LAYER_IDS).c_str(), GGUF_TYPE_UINT32, ids.data(), ids.size());
        gguf_set_val_u32 (c, kv(LLM_KV_ENGRAM_PAD_ID).c_str(), pad);
        gguf_set_arr_data(c, kv(LLM_KV_ENGRAM_MULTIPLIERS).c_str(), GGUF_TYPE_UINT64, mult.data(), mult.size());
        if (which == 0) {
            gguf_set_val_u32 (c, kv(LLM_KV_ENGRAM_HEAD_COUNT).c_str(), n_head);
            gguf_set_val_u32 (c, kv(LLM_KV_ENGRAM_KEY_LENGTH).c_str(), key);
            gguf_set_val_u32 (c, kv(LLM_KV_ENGRAM_MAX_NGRAM_SIZE).c_str(), n_gram);
            gguf_set_arr_data(c, kv(LLM_KV_ENGRAM_PRIMES).c_str(), GGUF_TYPE_UINT64, primes.data(), primes.size());
            gguf_set_arr_data(c, kv(LLM_KV_ENGRAM_OFFSETS).c_str(), GGUF_TYPE_UINT64, offsets.data(), offsets.size());
            gguf_set_arr_data(c, kv(LLM_KV_ENGRAM_TOKEN_MAP).c_str(), GGUF_TYPE_INT32, token_map.data(), token_map.size());
            return g;
        }
        // antirez's files name the model's other hyperparameters after its config.json
        auto value = [&](llm_kv from, double fallback) -> double {
            const int64_t id = gguf_find_key(c, kv(from).c_str());
            if (id < 0) {
                GGML_ASSERT(!std::isnan(fallback));
                return fallback;
            }
            gguf_type type = gguf_get_kv_type(c, id);
            const void * data = gguf_get_val_data(c, id);
            if (type == GGUF_TYPE_ARRAY) {
                type = gguf_get_arr_type(c, id);
                data = gguf_get_arr_data(c, id);
            }
            switch (type) {
                case GGUF_TYPE_UINT32:  return *(const uint32_t *) data;
                case GGUF_TYPE_INT32:   return *(const int32_t *) data;
                case GGUF_TYPE_FLOAT32: return *(const float *) data;
                case GGUF_TYPE_BOOL:    return *(const int8_t *) data;
                default:                GGML_ABORT("unexpected type of %s", kv(from).c_str());
            }
        };
        const struct { const char * name; llm_kv from; gguf_type type; double fallback; } renamed[] = {
            { "max_position_embeddings", LLM_KV_CONTEXT_LENGTH, GGUF_TYPE_UINT32, NAN },
            { "hidden_size", LLM_KV_EMBEDDING_LENGTH, GGUF_TYPE_UINT32, NAN },
            { "num_hidden_layers", LLM_KV_BLOCK_COUNT, GGUF_TYPE_UINT32, NAN },
            { "n_routed_experts", LLM_KV_EXPERT_COUNT, GGUF_TYPE_UINT32, NAN },
            { "num_experts_per_tok", LLM_KV_EXPERT_USED_COUNT, GGUF_TYPE_UINT32, NAN },
            { "num_attention_heads", LLM_KV_ATTENTION_HEAD_COUNT, GGUF_TYPE_UINT32, NAN },
            { "num_key_value_heads", LLM_KV_ATTENTION_HEAD_COUNT_KV, GGUF_TYPE_UINT32, 1 },
            { "rope_scaling.original_max_position_embeddings", LLM_KV_CONTEXT_LENGTH, GGUF_TYPE_FLOAT32, NAN },
            { "rope_theta", LLM_KV_ROPE_FREQ_BASE, GGUF_TYPE_UINT32, 10000 },
            { "rope_scaling.factor", LLM_KV_ROPE_SCALING_FACTOR, GGUF_TYPE_FLOAT32, 1 },
            { "head_dim", LLM_KV_ATTENTION_KEY_LENGTH, GGUF_TYPE_UINT32, NAN },
            { "qk_rope_head_dim", LLM_KV_ROPE_DIMENSION_COUNT, GGUF_TYPE_UINT32, NAN },
            { "rms_norm_eps", LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, GGUF_TYPE_FLOAT32, NAN },
            { "q_lora_rank", LLM_KV_ATTENTION_Q_LORA_RANK, GGUF_TYPE_UINT32, NAN },
            { "sliding_window", LLM_KV_ATTENTION_SLIDING_WINDOW, GGUF_TYPE_UINT32, NAN },
            { "moe_intermediate_size", LLM_KV_EXPERT_FEED_FORWARD_LENGTH, GGUF_TYPE_UINT32, NAN },
            { "n_shared_experts", LLM_KV_EXPERT_SHARED_COUNT, GGUF_TYPE_UINT32, NAN },
            { "routed_scaling_factor", LLM_KV_EXPERT_WEIGHTS_SCALE, GGUF_TYPE_FLOAT32, 1 },
            { "norm_topk_prob", LLM_KV_EXPERT_WEIGHTS_NORM, GGUF_TYPE_BOOL, 1 },
            { "swiglu_limit", LLM_KV_SWIGLU_CLAMP_EXP, GGUF_TYPE_FLOAT32, NAN },
            { "index_n_heads", LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, GGUF_TYPE_UINT32, NAN },
            { "index_head_dim", LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, GGUF_TYPE_UINT32, NAN },
            { "index_topk", LLM_KV_ATTENTION_INDEXER_TOP_K, GGUF_TYPE_UINT32, NAN },
            { "o_groups", LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT, GGUF_TYPE_UINT32, NAN },
            { "o_lora_rank", LLM_KV_ATTENTION_OUTPUT_LORA_RANK, GGUF_TYPE_UINT32, NAN },
            { "hc_mult", LLM_KV_HYPER_CONNECTION_COUNT, GGUF_TYPE_UINT32, NAN },
            { "hc_sinkhorn_iters", LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, GGUF_TYPE_UINT32, NAN },
            { "hc_eps", LLM_KV_HYPER_CONNECTION_EPSILON, GGUF_TYPE_FLOAT32, NAN },
            { "compress_rope_theta", LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE, GGUF_TYPE_UINT32, NAN },
            { "rope_scaling.beta_fast", LLM_KV_ROPE_SCALING_YARN_BETA_FAST, GGUF_TYPE_FLOAT32, 32 },
            { "rope_scaling.beta_slow", LLM_KV_ROPE_SCALING_YARN_BETA_SLOW, GGUF_TYPE_FLOAT32, 1 },
        };
        for (const auto & r : renamed) {
            const std::string name = std::string("deepseek41.") + r.name;
            const double v = value(r.from, r.fallback);
            switch (r.type) {
                case GGUF_TYPE_UINT32:  gguf_set_val_u32 (c, name.c_str(), (uint32_t) v); break;
                case GGUF_TYPE_FLOAT32: gguf_set_val_f32 (c, name.c_str(), (float) v); break;
                default:                gguf_set_val_bool(c, name.c_str(), v != 0); break;
            }
        }
        const int64_t ratios = gguf_find_key(c, kv(LLM_KV_ATTENTION_COMPRESS_RATIOS).c_str());
        GGML_ASSERT(ratios >= 0);
        gguf_set_arr_data(c, "deepseek41.compress_ratios", gguf_get_arr_type(c, ratios), gguf_get_arr_data(c, ratios),
                          gguf_get_arr_n(c, ratios));
        gguf_set_val_str(c, "deepseek41.scoring_func", "sqrtsoftplus");

        const std::vector<uint32_t> primes32(primes.begin(), primes.end());
        const std::vector<uint32_t> map32(token_map.begin(), token_map.end());
        gguf_set_val_str (c, kv(LLM_KV_ENGRAM_ENCODING).c_str(), "e4m3_e8m0_32_row264");
        gguf_set_val_u32 (c, kv(LLM_KV_ENGRAM_COMPRESSED_VOCAB_SIZE).c_str(), compressed);
        gguf_set_arr_data(c, kv(LLM_KV_ENGRAM_ROWS).c_str(), GGUF_TYPE_UINT32, rows.data(), rows.size());
        gguf_set_arr_data(c, kv(LLM_KV_ENGRAM_PRIMES).c_str(), GGUF_TYPE_UINT32, primes32.data(), primes32.size());
        gguf_set_arr_data(c, kv(LLM_KV_ENGRAM_TOKEN_MAP).c_str(), GGUF_TYPE_UINT32, map32.data(), map32.size());
        return g;
    };

    // a prompt of two sequences' worth of positions; its predecessors are its own tokens
    const uint32_t n_tokens = 40;
    std::vector<llama_token> tokens(n_tokens);
    std::vector<llama_pos> pos(n_tokens);
    std::vector<llama_token> prev(n_tokens*(n_gram - 1));
    for (uint32_t i = 0; i < n_tokens; ++i) {
        tokens[i] = llama_token(gen() % vocab);
        pos[i]    = llama_pos(i < 25 ? i : i - 25);
        for (uint32_t s = 1; s < n_gram; ++s) {
            prev[i*(n_gram - 1) + (n_gram - 1 - s)] = pos[i] >= (llama_pos) s ? tokens[i - s] : LLAMA_TOKEN_NULL;
        }
    }

    const char * names[3] = { "engram.*", "ple.*", "e4m3_e8m0_32_row264" };
    std::vector<int32_t> found[3];
    int failures = 0;
    for (int which = 0; which < 3; ++which) {
        // the metadata read as a load reads it, up to the hyperparameters (as llama_ssd_model_facts does)
        gguf_context_ptr g = layout(which);
        std::vector<std::string> splits;
        llama_model_loader ml(g.get(), nullptr, nullptr, "", splits, nullptr, /* use_mmap */ false,
            /* use_direct_io */ false, /* check_tensors */ false, /* defer_experts */ false, /* no_alloc */ true,
            nullptr, nullptr);
        llama_model m(llama_model_default_params());
        try {
            m.load_arch(ml);
            m.load_hparams(ml);
        } catch (const std::exception & error) {
            printf("%s: the %s reader refused its metadata: %s\n", __func__, names[which], error.what());
            return 1;
        }
        const bool same = m.engram_n_layer == layers.size() && m.engram_pad_id == pad && m.engram_token_map == token_map &&
            m.engram_multipliers == mult && m.engram_primes == primes &&
            m.engram_offsets == (which == 1 ? global : offsets) &&
            m.hparams.engram_n_head == n_head && m.hparams.engram_key_length == key &&
            m.hparams.engram_max_ngram_size == n_gram;
        if (!same) {
            printf("%s: the %s reader's parameters differ\n", __func__, names[which]);
            ++failures;
        }
        for (uint32_t e = 0; e < m.engram_n_layer; ++e) {
            const llama_dsv41_engram_hash_params hp = {
                n_gram, n_head, m.engram_multipliers.data() + e*n_gram, m.engram_primes.data() + e*n_cols,
                m.engram_offsets.data() + e*n_cols, m.engram_token_map.data(), m.engram_token_map.size(), m.engram_pad_id,
            };
            std::vector<int32_t> out(n_tokens*n_cols);
            llama_dsv41_engram_hash(hp, n_tokens, tokens.data(), pos.data(), prev.data(), out.data());
            const uint64_t first = which == 1 ? global[e*n_cols] : 0; // ple.*: the rows of the layers before
            for (auto & row : out) {
                found[which].push_back(row - (int32_t) first);
            }
        }
    }
    for (int which = 1; which < 3; ++which) {
        if (found[which] != found[0]) {
            printf("%s: the %s rows differ from the engram.* rows\n", __func__, names[which]);
            ++failures;
        }
    }
    printf("%s: engram.*, ple.* and e4m3_e8m0_32_row264 readers: %s (%zu rows each)\n", __func__,
           failures ? "DIFFERENT" : "identical", found[0].size());
    return failures;
}

static int test_engram_layouts(const size_t seed, const ggml_log_level log_level) {
    const llm_arch arch = LLM_ARCH_DEEPSEEK41;
    const LLM_KV kv(arch);
    const size_t n_cols = (DSV41_ENGRAM_NGRAM - 1) * DSV41_ENGRAM_HEADS;

    dsv41_engram_layouts_data ud;
    ud.seed = seed;
    uint64_t n_rows = 0;
    for (size_t e = 0; e < DSV41_ENGRAM_LAYERS.size(); ++e) {
        n_rows += dsv41_engram_rows(e);
    }
    std::mt19937 gen(seed);
    ud.table.resize(n_rows*DSV41_ENGRAM_ROW_BYTES);
    for (size_t i = 0; i < ud.table.size(); ++i) {
        // E8M0 scales near 2^-6, E4M3 codes but the NaN ones
        ud.table[i] = i % 33 == 0 ? uint8_t(116 + gen() % 8) : uint8_t(gen() % 256);
        if (i % 33 != 0 && (ud.table[i] & 127) == 127) {
            ud.table[i] ^= 1;
        }
    }

    gguf_context_ptr per_layer = get_gguf_ctx(arch, moe_mandatory(arch));
    gguf_context_ptr one_table = get_gguf_ctx(arch, moe_mandatory(arch));
    for (const llm_kv key : { LLM_KV_ENGRAM_LAYER_IDS, LLM_KV_ENGRAM_HEAD_COUNT, LLM_KV_ENGRAM_KEY_LENGTH,
                              LLM_KV_ENGRAM_MAX_NGRAM_SIZE, LLM_KV_ENGRAM_PAD_ID, LLM_KV_ENGRAM_MULTIPLIERS,
                              LLM_KV_ENGRAM_PRIMES, LLM_KV_ENGRAM_OFFSETS }) {
        GGML_ASSERT(gguf_remove_key(one_table.get(), kv(key).c_str()) >= 0);
    }
    std::vector<uint64_t> offsets = DSV41_ENGRAM_OFFSETS;
    uint64_t base = 0;
    for (size_t e = 0; e < DSV41_ENGRAM_LAYERS.size(); base += dsv41_engram_rows(e++)) {
        for (size_t b = 0; b < n_cols; ++b) {
            offsets[e*n_cols + b] += base;
        }
    }
    gguf_context * ctx = one_table.get();
    gguf_set_arr_data(ctx, kv(LLM_KV_PLE_LAYERS).c_str(),            GGUF_TYPE_UINT32, DSV41_ENGRAM_LAYERS.data(), DSV41_ENGRAM_LAYERS.size());
    gguf_set_val_u32 (ctx, kv(LLM_KV_PLE_HEADS_PER_NGRAM).c_str(),        DSV41_ENGRAM_HEADS);
    gguf_set_val_u32 (ctx, kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER).c_str(), DSV41_ENGRAM_KEY_LENGTH);
    gguf_set_val_u32 (ctx, kv(LLM_KV_PLE_NGRAM_SIZE).c_str(),             DSV41_ENGRAM_NGRAM);
    // ple.* names the pad by a token whose compressed id is the engram.* pad id: 63 % 61 = 2
    gguf_set_val_u32 (ctx, kv(LLM_KV_ENGRAM_PAD_TOKEN_ID).c_str(),        63);
    gguf_set_arr_data(ctx, kv(LLM_KV_PLE_LAYER_MULTIPLIERS).c_str(), GGUF_TYPE_UINT64, DSV41_ENGRAM_MULTIPLIERS.data(), DSV41_ENGRAM_MULTIPLIERS.size());
    gguf_set_arr_data(ctx, kv(LLM_KV_PLE_HEAD_VOCAB_SIZES).c_str(),  GGUF_TYPE_UINT64, DSV41_ENGRAM_PRIMES.data(),      DSV41_ENGRAM_PRIMES.size());
    gguf_set_arr_data(ctx, kv(LLM_KV_PLE_HEAD_OFFSETS).c_str(),      GGUF_TYPE_UINT64, offsets.data(),                  offsets.size());
    {
        // the table's type: a model made from metadata alone takes each tensor's type from it
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type  = GGML_TYPE_MXFP8;
        t.ne[0] = DSV41_ENGRAM_KEY_LENGTH;
        t.ne[1] = (int64_t) n_rows;
        t.ne[2] = t.ne[3] = 1;
        t.nb[0] = ggml_type_size(t.type);
        t.nb[1] = DSV41_ENGRAM_ROW_BYTES;
        t.nb[2] = t.nb[3] = t.nb[1]*n_rows;
        ggml_set_name(&t, "per_layer_token_embd.weight");
        gguf_add_tensor(ctx, &t);
    }

    struct log_data_t {
        ggml_log_callback callback;
        void * user_data;
        ggml_log_level min_level;
    } ld;
    llama_log_get(&ld.callback, &ld.user_data);
    ld.min_level = log_level;
    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const auto * ld = (const log_data_t *) user_data;
        ld->callback(level >= ld->min_level ? level : GGML_LOG_LEVEL_DEBUG, text, ld->user_data);
    }, &ld);

    std::vector<ggml_backend_dev_t> devs = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
    std::vector<float> logits[2];
    gguf_context * layouts[2] = { per_layer.get(), one_table.get() };
    for (int i = 0; i < 2; ++i) {
        llama_model_params model_params = llama_model_default_params();
        model_params.progress_callback = silent_model_load_progress;
        model_params.devices = devs.data();
        llama_model_ptr model(llama_model_init_from_user(layouts[i], set_engram_layouts_data, &ud, model_params));
        if (!model) {
            throw std::runtime_error("failed to create llama model");
        }
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx           = 0;
        ctx_params.n_ubatch        = 64;
        ctx_params.n_threads       = 4;
        ctx_params.n_threads_batch = 4;
        llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
        if (!lctx) {
            throw std::runtime_error("failed to create llama context");
        }
        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        // the prompt crosses the tokens the map marks -1
        std::vector<llama_token> tokens = get_tokens(32, n_vocab, seed);
        tokens[4]  = DSV41_ENGRAM_DEAD_TOKENS[0];
        tokens[20] = DSV41_ENGRAM_DEAD_TOKENS[1];
        logits[i] = get_logits(model.get(), lctx.get(), tokens);
        if (i == 1) {
            test_engram_files(model.get(), one_table.get(), model_params, ctx_params, tokens, logits[i]);
        }
    }
    const int readers = test_engram_readers(seed);
    llama_log_set(ld.callback, ld.user_data);

    const bool ok = logits[0] == logits[1];
    printf("%s: engram.* vs ple.* (MXFP8 table) layout: %s (nmse %.3g)\n", __func__,
           ok ? "identical" : "DIFFERENT", nmse(logits[0], logits[1]));
    return ok && readers == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    // FIXME these tests are disabled in the CI for macOS-latest-cmake-arm64 because they are segfaulting
    common_init();
    std::random_device rd;

    llm_arch arch = LLM_ARCH_UNKNOWN;
    size_t seed = rd();
    ggml_log_level log_level = GGML_LOG_LEVEL_ERROR;
    std::string out;
    bool kv_grow = false;
    bool nextn = false;
    bool engram_layouts = false;
    bool dense_ssd = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--kv-grow") == 0) {
            kv_grow = true;
            continue;
        }
        if (strcmp(argv[i], "--nextn") == 0) {
            nextn = true;
            continue;
        }
        if (strcmp(argv[i], "--engram-layouts") == 0) {
            engram_layouts = true;
            continue;
        }
        if (strcmp(argv[i], "--dense-ssd") == 0) {
            dense_ssd = true;
            continue;
        }
        if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                arch = llm_arch_from_string(arch_name);
                if (arch == LLM_ARCH_UNKNOWN) {
                    LOG_ERR("%s: unkown LLM architecture: %s\n", __func__, arch_name.c_str());
                    return 1;
                }
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            log_level = GGML_LOG_LEVEL_INFO;
            continue;
        }
        if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        }
    }
    printf("%s: using seed %zu\n", __func__, seed);

    try {
        if (!out.empty()) {
            return save_models(arch, seed, log_level, out);
        }
        if (kv_grow) {
            return test_kv_grow(arch, seed, log_level);
        }
        if (nextn) {
            return test_nextn(arch, seed, log_level);
        }
        if (engram_layouts) {
            return test_engram_layouts(seed, log_level);
        }
        if (dense_ssd) {
            return test_dense_ssd(seed, log_level);
        }
        return test_backends(arch, seed, log_level);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
