#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <map>
#include <cinttypes>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::vector<enum common_speculative_type> common_speculative_types = {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT,
    COMMON_SPECULATIVE_TYPE_EAGLE3,
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE
};

const std::map<std::string, enum common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft",         COMMON_SPECULATIVE_TYPE_DRAFT},
    {"eagle3",        COMMON_SPECULATIVE_TYPE_EAGLE3},
    {"ngram_simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram_map_k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram_map_k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram_mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram_cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_WRN("%s: draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", __func__, vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        LOG_WRN("%s: draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        LOG_WRN("%s: draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_state
struct common_speculative_state {
    const enum common_speculative_type type;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_state(enum common_speculative_type type) : type(type) {}

    virtual ~common_speculative_state() = default;

    virtual void begin(const llama_tokens & prompt) = 0;

    virtual void clear() {}

    virtual bool process(const llama_batch & batch, llama_seq_id target_seq_id) {
        GGML_UNUSED(batch);
        GGML_UNUSED(target_seq_id);
        return true;
    }

    virtual bool supports_tree() const { return false; }

    virtual bool process_rows(
            const llama_batch & batch,
            const int32_t     * keep_idxs,
            int32_t             n_keep,
            llama_pos           pos0,
            llama_seq_id        target_seq_id) {
        GGML_UNUSED(batch);
        GGML_UNUSED(keep_idxs);
        GGML_UNUSED(n_keep);
        GGML_UNUSED(pos0);
        GGML_UNUSED(target_seq_id);
        return true;
    }

    virtual void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) = 0;

    virtual void draft_tree(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            common_speculative_tree & result) {
        result.clear();
        draft(params, prompt_tgt, id_last, result.tokens);
        result.parents.resize(result.tokens.size());
        for (size_t i = 0; i < result.tokens.size(); ++i) {
            result.parents[i] = (int8_t) i;
        }
        result.probs.assign(result.tokens.size(), 1.0f);
    }

    virtual void accept(uint16_t n_accepted) = 0;

    virtual bool save_state_file(const char * /*path*/, int64_t /*n_tokens*/) { return false; }
    virtual bool load_state_file(const char * /*path*/, int64_t /*n_tokens_keep*/) { return false; }

    virtual int32_t n_max(const common_params_speculative & params) const = 0;
    virtual int32_t n_min(const common_params_speculative & params) const = 0;
};

struct common_speculative_checkpoint {
    llama_pos pos_min  = 0;
    llama_pos pos_max  = 0;

    int64_t   n_tokens = 0;

    std::vector<uint8_t> data;

    size_t size() const {
        return data.size();
    }

    size_t ckpt_size   = 0;
};

struct common_speculative_state_draft : public common_speculative_state {
    llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    llama_context * ctx_dft;

    bool use_ckpt = false;
    struct common_speculative_checkpoint ckpt;

    common_sampler * smpl;

    llama_batch  batch;
    llama_tokens prompt_dft;

    bool vocab_cmpt = true; // whether retokenization is needed
    std::unordered_map<std::string, std::string> vocab_map;

    common_speculative_state_draft(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            const std::vector<std::pair<std::string, std::string>> & replacements,
            bool use_ckpt)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
        , use_ckpt(use_ckpt)
    {
        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
        smpl = nullptr;

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }
        {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        }

        vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_WRN("the target and draft vocabs are not compatible - tokens will be translated between the two\n");

            for (const auto & pair : replacements) {
                vocab_map[pair.first] = pair.second;
            }
        }
    }

    ~common_speculative_state_draft() override {
        llama_perf_context_print(ctx_dft);

        llama_free(ctx_dft);

        common_sampler_free(smpl);

        llama_batch_free(batch);
    }

    void begin(const llama_tokens & prompt) override {
        if (use_ckpt && ckpt.size() > 0) {
            // delete checkpoint
            LOG_DBG("%s: delete checkpoint, prompt.size=%zu, pos_min=%d, pos_max=%d, n_tokens=%" PRId64 ", size=%.3f MiB\n",
                    __func__, prompt.size(), ckpt.pos_min, ckpt.pos_max, ckpt.n_tokens, (float) ckpt.data.size() / 1024 / 1024);
            ckpt.pos_min   = 0;
            ckpt.pos_max   = 0;
            ckpt.n_tokens  = 0;
            ckpt.ckpt_size = 0;
            ckpt.data.clear();
        }
    }

    size_t draft_create_checkpoint(int n_tokens_prompt, int n_tokens_batch) {
        int slot_id = 0;
        const size_t checkpoint_size = llama_state_seq_get_size_ext(ctx_dft, slot_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

        ckpt.pos_min  = llama_memory_seq_pos_min(llama_get_memory(ctx_dft), slot_id);
        ckpt.pos_max  = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), slot_id);
        ckpt.n_tokens = n_tokens_prompt - n_tokens_batch;
        ckpt.data.resize(checkpoint_size);

        const size_t n = llama_state_seq_get_data_ext(ctx_dft, ckpt.data.data(), checkpoint_size, slot_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (n != checkpoint_size) {
            GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", checkpoint_size, n);
        }

        LOG_DBG("%s: pos_min = %d, pos_max = %d, size = %.3f MiB\n", __func__,
                ckpt.pos_min, ckpt.pos_max, (float) ckpt.data.size() / 1024 / 1024);
        return n;
    }

    size_t draft_restore_checkpoint(size_t ckpt_size_part_expected) {
        int slot_id = 0;
        LOG_DBG("%s: pos_min = %d, pos_max = %d\n", __func__, ckpt.pos_min, ckpt.pos_max);
        const size_t n = llama_state_seq_set_data_ext(ctx_dft, ckpt.data.data(), ckpt.size(), slot_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (n != ckpt_size_part_expected) {
            GGML_ABORT("%s: failed to restore context checkpoint (pos_min=%d, pos_max=%d, size=%zu, get_data_ext->%zu, set_data_ext->%zu",
                        __func__, ckpt.pos_min, ckpt.pos_max, ckpt.size(), ckpt_size_part_expected, n);
        }
        llama_memory_seq_rm(llama_get_memory(ctx_dft), slot_id, ckpt.pos_max + 1, -1);

        return n;
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        const auto & sparams = params.draft;

        auto * spec = this;

        auto & batch      = spec->batch;
        auto & ctx_tgt    = spec->ctx_tgt;
        auto & ctx_dft    = spec->ctx_dft;
        auto & smpl       = spec->smpl;
        auto & prompt_dft = spec->prompt_dft;

        auto * mem_dft = llama_get_memory(ctx_dft);

        int reuse_i = 0; // index of part to be reused in prompt_dft
        int reuse_n = 0; // length of part to be reused in prompt_dft

        const int n_ctx = llama_n_ctx(ctx_dft) - sparams.n_max;

        llama_tokens prompt_cnv;
        if (!spec->vocab_cmpt) {
            std::string text;

            text = common_detokenize(ctx_tgt, prompt_tgt, true);
            text = replace_to_dft(text);

            LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());

            prompt_cnv = common_tokenize(ctx_dft, text, false, true);

            // convert id_last to draft vocab. llama_detokenize is called directly to avoid an allocation
            const auto * model_tgt = llama_get_model(ctx_tgt);
            const auto * vocab_tgt = llama_model_get_vocab(model_tgt);

            int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
            GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last");

            text.resize(-n_chars);
            llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
            text = replace_to_dft(text);

            LOG_DBG("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
            id_last = common_tokenize(ctx_dft, text, false, true)[0];
        }

        const llama_tokens & prompt_cur = spec->vocab_cmpt ? prompt_tgt : prompt_cnv;

        const int i_start = std::max<int>(0, (int) prompt_cur.size() - n_ctx);

        // reuse as much as possible from the old draft context
        // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
        for (int i = 0; i < (int) prompt_dft.size(); ++i) {
            int cur = 0;
            while (i_start + cur < (int) prompt_cur.size() &&
                    i       + cur < (int) prompt_dft.size() &&
                    prompt_cur[i_start + cur] == prompt_dft[i + cur]) {
                cur++;
            }

            if ((cur >= 256 || n_ctx >= (int) prompt_cur.size()) && cur > reuse_n) {
                reuse_i = i;
                reuse_n = cur;
            }
        }

        LOG_DBG("%s: reuse_i = %d, reuse_n = %d, #prompt_dft = %zu, #prompt_cur = %zu\n",
                __func__, reuse_i, reuse_n, prompt_dft.size(), prompt_cur.size());
        if (use_ckpt && ckpt.ckpt_size == 0 && reuse_n > 0) {
            LOG_DBG("%s: no checkpoint available, no reuse, (reuse_i=%d, reuse_n=%d) -> (0, 0)\n",
                    __func__, reuse_i, reuse_n);
            reuse_i = 0;
            reuse_n = 0;
        }

        result.clear();
        result.reserve(sparams.n_max);

        bool needs_ckpt = use_ckpt && prompt_dft.size() > 0;
        if (reuse_n == 0 || (use_ckpt && reuse_i > 0)) {
            llama_memory_clear(mem_dft, false);
            prompt_dft.clear();
        } else {
            // this happens when a previous draft has been discarded (for example, due to being too small), but the
            // target model agreed with it. in this case, we simply pass back the previous results to save compute
            if (reuse_i + reuse_n < (int64_t) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
                for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                    result.push_back(prompt_dft[i]);

                    if (sparams.n_max <= (int) result.size()) {
                        break;
                    }
                }

                return;
            }

            bool do_restore = false;
            if (prompt_dft.size() > prompt_cur.size() && reuse_i + reuse_n < (int64_t) prompt_dft.size()) {
                // This can happen after a partial acceptance (speculative decoding with checkpoints)
                LOG_DBG("%s: #prompt_dft=%zu, #prompt_cur=%zu, shorten draft\n",
                        __func__, prompt_dft.size(), prompt_cur.size());
                prompt_dft.resize(prompt_cur.size());
                do_restore = true;
            }

            if (reuse_i > 0) {
                bool is_removed = llama_memory_seq_rm (mem_dft, 0, 0, reuse_i);
                if (!is_removed) {
                    LOG_ERR("%s: llama_memory_seq_rm failed, reuse_i=%d\n", __func__, reuse_i);
                }
                llama_memory_seq_add(mem_dft, 0, reuse_i, -1, -reuse_i);

                prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
            }

            if (reuse_n < (int) prompt_dft.size() || do_restore) {
                if (use_ckpt) {
                    if (ckpt.n_tokens > (int64_t) prompt_dft.size()) {
                        LOG_INF("%s: checkpoint is too large, prompt_tgt.size=%zu, ckpt.n_tokens=%" PRId64 ", reuse_n=%d, prompt_dft.size=%zu\n",
                                __func__, prompt_tgt.size(), ckpt.n_tokens, reuse_n, prompt_dft.size());
                    }
                    draft_restore_checkpoint(ckpt.ckpt_size);
                    reuse_n = ckpt.n_tokens;
                    prompt_dft.resize(reuse_n);
                    needs_ckpt = false;
                } else {
                    bool is_removed = llama_memory_seq_rm (mem_dft, 0, reuse_n, -1);
                    if (!is_removed) {
                        LOG_ERR("%s: llama_memory_seq_rm failed, reuse_n=%d, prompt_dft.size=%zu\n",
                                __func__, reuse_n, prompt_dft.size());
                    }
                    prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
                }
            }
        }

        if (needs_ckpt) {
            ckpt.ckpt_size = draft_create_checkpoint(prompt_dft.size(), batch.n_tokens);
        }

        // prepare a batch to evaluate any new tokens in the prompt
        common_batch_clear(batch);

        for (size_t i = i_start + reuse_n; i < prompt_cur.size(); ++i) {
            //LOG_DBG("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_cur[i]);
            common_batch_add(batch, prompt_cur[i], i - i_start, { 0 }, false);

            prompt_dft.push_back(prompt_cur[i]);
        }

        // we should rarely end-up here during normal decoding
        if (batch.n_tokens > 0) {
            //LOG_DBG("%s: draft prompt batch: %s\n", __func__, string_from(ctx, batch).c_str());

            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0 && ret != 1) {
                LOG_WRN("%s: llama_decode returned %d, prompt_cur.size=%zu\n",
                        __func__, ret, prompt_cur.size());
            }
        }

        const llama_pos n_past = prompt_dft.size();

        LOG_DBG("%s: n_past = %d\n", __func__, n_past);

        common_batch_clear(batch);
        common_batch_add  (batch, id_last, n_past, { 0 }, true);

        prompt_dft.push_back(id_last);

        LOG_DBG("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0 && ret != 1) {
            LOG_WRN("%s: llama_decode returned %d, prompt_cur.size=%zu, prompt_dft.size=%zu\n",
                    __func__, ret, prompt_cur.size(), prompt_dft.size());
        }

        common_sampler_reset(smpl);

        // sample n_draft tokens from the draft model
        for (int i = 0; i < sparams.n_max; ++i) {
            common_batch_clear(batch);

            common_sampler_sample(smpl, ctx_dft, 0, true);

            const auto * cur_p = common_sampler_get_candidates(smpl, true);

            for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                        k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
            }

            // add drafted token for each sequence
            const llama_token id = cur_p->data[0].id;

            common_sampler_accept(smpl, id, true);

            result.push_back(id);

            if (sparams.n_max <= (int) result.size()) {
                break;
            }

            // only collect very high-confidence draft tokens
            if (cur_p->data[0].p < sparams.p_min) {
                break;
            }

            common_batch_add(batch, id, n_past + i + 1, { 0 }, true);

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d, prompt_cur.size=%zu, prompt_dft.size=%zu\n",
                        __func__, i, ret, prompt_cur.size(), prompt_dft.size());
            }

            prompt_dft.push_back(id);
        }

        if (!spec->vocab_cmpt) {
            std::string detokenized = common_detokenize(ctx_dft, result, true);
            detokenized = replace_to_tgt(detokenized);
            LOG_DBG("draft->main detokenized string: '%s'\n", detokenized.c_str());
            result = common_tokenize(ctx_tgt, detokenized, false, true);
            if (result.size() > (size_t) sparams.n_max) {
                result.resize(sparams.n_max);
            }
        }

        if (result.size() < (size_t) sparams.n_min) {
            result.clear();
        }
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return params.draft.n_max;
    }

    int32_t n_min(const common_params_speculative & params) const override {
        return params.draft.n_min;
    }

    std::string replace_to_dft(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.first);
            while (pos != std::string::npos) {
                result.replace(pos, pair.first.length(), pair.second);
                pos = result.find(pair.first, pos + pair.second.length());
            }
        }

        return result;
    }

    std::string replace_to_tgt(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.second);
            while (pos != std::string::npos) {
                result.replace(pos, pair.second.length(), pair.first);
                pos = result.find(pair.second, pos + pair.first.length());
            }
        }

        return result;
    }
};

struct common_speculative_state_inkling2_dspark : public common_speculative_state {
    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;
    llama_model   * model_dft = nullptr;

    const int32_t * target_layer_ids   = nullptr;
    uint32_t        target_layer_ids_n = 0;
    int32_t         block_size         = 0;
    int32_t         n_embd             = 0;
    int32_t         n_embd_enc         = 0;

    llama_batch batch_inject {};
    llama_batch batch_draft  {};

    static constexpr size_t F_RNN_RING = 32;
    static constexpr int32_t MAX_PENDING_TAPS = 16;
    static constexpr size_t SUFFIX_BUCKETS = 1u << 16;

    std::vector<float>        f_rnn_ring;
    std::vector<llama_pos>    f_rnn_pos;
    std::vector<float>        pending_taps;
    llama_pos                 pending_pos0 = -1;
    int32_t                   pending_n    = 0;
    llama_pos                 gpu_f_rnn_pos = -1;
    std::vector<int32_t>      rows_scratch;
    std::vector<int32_t>      draft_ids_scratch;
    std::vector<float>        draft_probs_scratch;
    size_t                    n_vocab_synced = 0;

    std::vector<int32_t>      suf4_head;
    std::vector<int32_t>      suf4_prev;
    std::vector<int32_t>      suf3_head;
    std::vector<int32_t>      suf3_prev;
    size_t                    n_suffix_synced = 0;
    llama_token               suffix_first_tok = LLAMA_TOKEN_NULL;
    llama_token               suffix_last_tok  = LLAMA_TOKEN_NULL;
    bool                      last_was_suffix_bypass = false;
    bool                      last_was_mode_c        = false;
    int32_t                   suffix_bypass_penalty  = 0;
    int32_t                   last_verify_B          = 0;
    float                     ema_p0                 = 0.85f;
    int32_t                   last_n_accepted        = 0;
    int32_t                   last_dspark_accepted   = 0;
    int32_t                   last_trunk_max         = 7;
    int32_t                   last_trunk_tip         = 0;

    static inline uint32_t hash3(llama_token t0, llama_token t1, llama_token t2) {
        uint64_t h = (uint64_t) (uint32_t) t0 * 0x9E3779B97F4A7C15ULL;
        h = (h ^ ((uint64_t) (uint32_t) t1 + 0x9E3779B9ULL)) * 0xBF58476D1CE4E5B9ULL;
        h = (h ^ ((uint64_t) (uint32_t) t2 + 0x517CC1B7ULL)) * 0x94D049BB133111EBULL;
        return (uint32_t) ((h ^ (h >> 32)) & (SUFFIX_BUCKETS - 1));
    }

    static inline uint32_t hash4(llama_token t0, llama_token t1, llama_token t2, llama_token t3) {
        uint64_t h = (uint64_t) (uint32_t) t0 * 0x9E3779B97F4A7C15ULL;
        h = (h ^ ((uint64_t) (uint32_t) t1 + 0x9E3779B9ULL)) * 0xBF58476D1CE4E5B9ULL;
        h = (h ^ ((uint64_t) (uint32_t) t2 + 0x517CC1B7ULL)) * 0x94D049BB133111EBULL;
        h = (h ^ ((uint64_t) (uint32_t) t3 + 0x2545F491ULL)) * 0x9E3779B97F4A7C15ULL;
        return (uint32_t) ((h ^ (h >> 32)) & (SUFFIX_BUCKETS - 1));
    }

    void suffix_reset() {
        suf4_head.assign(SUFFIX_BUCKETS, -1);
        suf4_prev.clear();
        suf3_head.assign(SUFFIX_BUCKETS, -1);
        suf3_prev.clear();
        n_suffix_synced        = 0;
        suffix_first_tok       = LLAMA_TOKEN_NULL;
        suffix_last_tok        = LLAMA_TOKEN_NULL;
        last_was_suffix_bypass = false;
        suffix_bypass_penalty  = 0;
    }

    void suffix_sync(const llama_tokens & prompt_tgt) {
        if (prompt_tgt.empty()) {
            return;
        }
        if (suf4_head.empty() || prompt_tgt.size() < n_suffix_synced ||
            (n_suffix_synced > 0 && (prompt_tgt[0] != suffix_first_tok || prompt_tgt[n_suffix_synced - 1] != suffix_last_tok))) {
            suffix_reset();
        }
        const size_t n_old = n_suffix_synced;
        const size_t n_new = prompt_tgt.size();
        if (n_new > suf4_prev.size()) {
            suf4_prev.resize(n_new, -1);
            suf3_prev.resize(n_new, -1);
        }
        for (size_t i = std::max<size_t>(2, n_old); i < n_new; ++i) {
            const uint32_t b3 = hash3(prompt_tgt[i - 2], prompt_tgt[i - 1], prompt_tgt[i]);
            suf3_prev[i] = suf3_head[b3];
            suf3_head[b3] = (int32_t) i;
        }
        for (size_t i = std::max<size_t>(3, n_old); i < n_new; ++i) {
            const uint32_t b4 = hash4(prompt_tgt[i - 3], prompt_tgt[i - 2], prompt_tgt[i - 1], prompt_tgt[i]);
            suf4_prev[i] = suf4_head[b4];
            suf4_head[b4] = (int32_t) i;
        }
        n_suffix_synced  = n_new;
        suffix_first_tok = prompt_tgt.front();
        suffix_last_tok  = prompt_tgt.back();
    }

    // Match the trailing tokens in `tail[0..tail_len)` against the indexed `prompt_tgt`.
    // Returns {best_end_pos_in_prompt, best_match_len, n_matches_in_chain}. Prefers longest match, breaking ties by most recent position.
    struct SuffixMatch {
        int32_t pos = -1;
        int32_t len = 0;
        int32_t count = 0;
    };

    SuffixMatch suffix_match(
            const llama_tokens & prompt_tgt,
            const llama_token  * tail,
            int32_t              tail_len) const {
        SuffixMatch out;
        if (tail_len < 4 || prompt_tgt.size() < 5 || suf4_head.empty()) {
            return out;
        }
        const uint32_t b = hash4(tail[tail_len - 4], tail[tail_len - 3], tail[tail_len - 2], tail[tail_len - 1]);
        int32_t p = suf4_head[b];
        for (int hops = 0; p >= 3 && hops < 16; ++hops, p = suf4_prev[p]) {
            if ((size_t) (p + 1) >= prompt_tgt.size()) {
                continue;
            }
            if (prompt_tgt[p]     != tail[tail_len - 1] ||
                prompt_tgt[p - 1] != tail[tail_len - 2] ||
                prompt_tgt[p - 2] != tail[tail_len - 3] ||
                prompt_tgt[p - 3] != tail[tail_len - 4]) {
                continue;
            }
            out.count++;
            int32_t mlen = 4;
            const int32_t max_mlen = std::min<int32_t>(tail_len, p + 1);
            while (mlen < max_mlen && prompt_tgt[p - mlen] == tail[tail_len - 1 - mlen]) {
                mlen++;
            }
            if (mlen > out.len) {
                out.len = mlen;
                out.pos = p;
            }
        }
        return out;
    }

    SuffixMatch suffix_match3(
            const llama_tokens & prompt_tgt,
            const llama_token  * tail,
            int32_t              tail_len) const {
        SuffixMatch out;
        if (tail_len < 3 || prompt_tgt.size() < 4 || suf3_head.empty()) {
            return out;
        }
        const uint32_t b = hash3(tail[tail_len - 3], tail[tail_len - 2], tail[tail_len - 1]);
        int32_t p = suf3_head[b];
        for (int hops = 0; p >= 2 && hops < 8; ++hops, p = suf3_prev[p]) {
            if ((size_t) (p + 1) >= prompt_tgt.size()) {
                continue;
            }
            if (prompt_tgt[p]     != tail[tail_len - 1] ||
                prompt_tgt[p - 1] != tail[tail_len - 2] ||
                prompt_tgt[p - 2] != tail[tail_len - 3]) {
                continue;
            }
            out.count++;
            if (out.pos < 0) {
                out.pos = p;
                out.len = 3;
            }
            if (out.count > 2) {
                break;
            }
        }
        return out;
    }

    common_speculative_state_inkling2_dspark(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
        , model_dft(const_cast<llama_model *>(llama_get_model(ctx_dft)))
    {
        const llama_model * model_tgt = llama_get_model(ctx_tgt);
        if (!common_speculative_are_compatible(model_tgt, model_dft)) {
            throw std::runtime_error("Inkling2 DSpark draft vocab is incompatible with target model");
        }

        target_layer_ids   = llama_model_target_layer_ids(model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        block_size         = (int32_t) llama_model_dspark_block_size(model_dft);
        n_embd             = llama_model_n_embd(model_dft);
        n_embd_enc         = (int32_t) target_layer_ids_n * n_embd;

        GGML_ASSERT(target_layer_ids && target_layer_ids_n >= 2 && block_size >= 2 && n_embd > 0);

        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        llama_set_embeddings_nextn(ctx_dft, true, false);
        llama_set_causal_attn(ctx_dft, false);

        const int32_t max_draft_block = std::max<int32_t>(block_size, 16);
        const int32_t n_ubatch = std::max<int32_t>(max_draft_block + MAX_PENDING_TAPS, (int32_t) llama_n_ubatch(ctx_dft));
        batch_inject = llama_batch_init(n_ubatch, n_embd_enc, 1);
        batch_draft  = llama_batch_init(max_draft_block + MAX_PENDING_TAPS, 0, 1);

        // Draft over the top 38000 ASCII and 2000 other tokens the drafter favors, plus room for ~10000 dynamic tokens
        // from prompts/outputs; active-vocab slicing only reads the populated prefix (~40K rows = 20% of full vocab).
        constexpr uint32_t n_ascii = 38000;
        if (!llama_model_inkling2_dspark_init_draft_vocab(model_dft, n_ascii, 2000, n_ascii + 10000)) {
            LOG_WRN("%s: drafting over the whole vocabulary\n", __func__);
        }

        f_rnn_ring.assign(F_RNN_RING * (size_t) n_embd, 0.0f);
        f_rnn_pos.assign(F_RNN_RING, -1);
        pending_taps.assign((size_t) MAX_PENDING_TAPS * (size_t) n_embd_enc, 0.0f);
        pending_pos0  = -1;
        pending_n     = 0;
        gpu_f_rnn_pos = -1;
        draft_ids_scratch.resize((size_t) max_draft_block, 0);
        draft_probs_scratch.assign((size_t) max_draft_block, 0.0f);
        suffix_reset();

        LOG_INF("%s: initialized Inkling2 DSpark drafter (block_size = %d, taps = %u, n_embd = %d)\n",
                __func__, block_size, target_layer_ids_n, n_embd);
    }

    ~common_speculative_state_inkling2_dspark() override {
        llama_perf_context_print(ctx_dft);
        llama_free(ctx_dft);
        llama_batch_free(batch_inject);
        llama_batch_free(batch_draft);
    }

    void clear() override {
        if (ctx_dft) {
            llama_memory_clear(llama_get_memory(ctx_dft), false);
        }
        pending_n      = 0;
        pending_pos0   = -1;
        gpu_f_rnn_pos  = -1;
        n_vocab_synced = 0;
        std::fill(f_rnn_pos.begin(), f_rnn_pos.end(), -1);
        llama_model_inkling2_dspark_set_pending_taps(model_dft, nullptr, 0);
        suffix_reset();
    }

    static constexpr uint32_t DSPARK_STATE_MAGIC   = 0x4B505344; // "DSPK"
    static constexpr uint32_t DSPARK_STATE_VERSION = 1;

    bool save_state_file(const char * path, int64_t n_tokens) override {
        if (!ctx_dft || !path || n_tokens <= 0) {
            return false;
        }
        if (pending_n > 0 && !flush_pending()) {
            return false;
        }
        auto * mem_dft = llama_get_memory(ctx_dft);
        if (!mem_dft) {
            return false;
        }
        const llama_pos pos_min = llama_memory_seq_pos_min(mem_dft, 0);
        const llama_pos pos_max = llama_memory_seq_pos_max(mem_dft, 0);
        if (pos_min != 0 || pos_max + 1 < n_tokens) {
            return false;
        }

        const uint64_t n_tokens_saved = (uint64_t) (pos_max + 1);
        const uint64_t state_size     = (uint64_t) llama_state_seq_get_size_ext(ctx_dft, 0, 0);
        if (state_size == 0) {
            return false;
        }

        const std::string dst(path);
        const std::string tmp = dst + ".tmp";
        bool ok = false;
        if (FILE * f = std::fopen(tmp.c_str(), "wb")) {
            const uint32_t magic   = DSPARK_STATE_MAGIC;
            const uint32_t version = DSPARK_STATE_VERSION;
            ok = std::fwrite(&magic, sizeof(magic), 1, f) == 1 &&
                 std::fwrite(&version, sizeof(version), 1, f) == 1 &&
                 std::fwrite(&n_tokens_saved, sizeof(n_tokens_saved), 1, f) == 1 &&
                 std::fwrite(&state_size, sizeof(state_size), 1, f) == 1 &&
                 llama_state_seq_fwrite_at(ctx_dft, f, 0, 0, -1) == state_size &&
                 std::fflush(f) == 0;
            ok = std::fclose(f) == 0 && ok;
        }
        std::error_code ec;
        if (ok) {
            std::filesystem::rename(tmp, dst, ec);
            ok = !ec;
        }
        if (!ok) {
            std::filesystem::remove(tmp, ec);
        }
        return ok;
    }

    bool load_state_file(const char * path, int64_t n_tokens_keep) override {
        if (!ctx_dft || !path || n_tokens_keep <= 0) {
            return false;
        }
        FILE * f = std::fopen(path, "rb");
        if (!f) {
            return false;
        }
        uint32_t magic = 0, version = 0;
        uint64_t n_tokens_saved = 0, state_size = 0;
        bool ok = std::fread(&magic, sizeof(magic), 1, f) == 1 &&
                  std::fread(&version, sizeof(version), 1, f) == 1 &&
                  std::fread(&n_tokens_saved, sizeof(n_tokens_saved), 1, f) == 1 &&
                  std::fread(&state_size, sizeof(state_size), 1, f) == 1 &&
                  magic == DSPARK_STATE_MAGIC &&
                  version == DSPARK_STATE_VERSION &&
                  n_tokens_saved >= (uint64_t) n_tokens_keep &&
                  state_size > 0;
        if (!ok) {
            std::fclose(f);
            return false;
        }

        clear();
        const size_t nread = llama_state_seq_fread_ext(ctx_dft, f, 0, 0);
        std::fclose(f);
        if (nread != state_size) {
            clear();
            return false;
        }

        auto * mem_dft = llama_get_memory(ctx_dft);
        if (!mem_dft) {
            clear();
            return false;
        }
        if (n_tokens_saved > (uint64_t) n_tokens_keep) {
            if (!llama_memory_seq_rm(mem_dft, 0, (llama_pos) n_tokens_keep, -1)) {
                clear();
                return false;
            }
        }
        const llama_pos pos_min = llama_memory_seq_pos_min(mem_dft, 0);
        const llama_pos pos_max = llama_memory_seq_pos_max(mem_dft, 0);
        if (pos_min != 0 || pos_max != (llama_pos) (n_tokens_keep - 1)) {
            clear();
            return false;
        }
        return true;
    }

    void begin(const llama_tokens & prompt) override {
        last_verify_B   = 0;
        ema_p0          = 0.85f;
        last_n_accepted      = 0;
        last_dspark_accepted = 0;
        last_trunk_max       = block_size - 1;
        last_trunk_tip  = 0;
        n_vocab_synced  = 0;
        // Preserve buffered tail-of-prefill taps only if they end exactly at prompt.size()
        if (pending_n > 0 && pending_pos0 + pending_n != (llama_pos) prompt.size()) {
            pending_n    = 0;
            pending_pos0 = -1;
        }
        llama_model_inkling2_dspark_set_pending_taps(model_dft, nullptr, 0);
        if (!prompt.empty()) {
            llama_model_inkling2_dspark_add_draft_vocab(model_dft, prompt.data(), (uint32_t) prompt.size());
            n_vocab_synced = prompt.size();
        }
        suffix_reset();
        suffix_sync(prompt);
    }

    bool flush_pending() {
        if (pending_n <= 0) {
            return true;
        }
        auto * mem_dft = llama_get_memory(ctx_dft);
        if (pending_pos0 <= 0) {
            llama_memory_clear(mem_dft, false);
            std::fill(f_rnn_pos.begin(), f_rnn_pos.end(), -1);
            gpu_f_rnn_pos = -1;
        } else if (!llama_memory_seq_rm(mem_dft, 0, pending_pos0, -1)) {
            llama_memory_clear(mem_dft, false);
            std::fill(f_rnn_pos.begin(), f_rnn_pos.end(), -1);
            gpu_f_rnn_pos = -1;
        }

        const int32_t n = pending_n;
        std::memcpy(batch_inject.embd, pending_taps.data(), (size_t) n * (size_t) n_embd_enc * sizeof(float));
        batch_inject.n_tokens = n;
        for (int32_t i = 0; i < n; ++i) {
            batch_inject.pos[i]       = pending_pos0 + i;
            batch_inject.n_seq_id[i]  = 1;
            batch_inject.seq_id[i][0] = 0;
            batch_inject.logits[i]    = false;
        }
        llama_model_inkling2_dspark_set_pending_taps(model_dft, nullptr, 0);

        const int ret = llama_decode(ctx_dft, batch_inject);
        pending_n    = 0;
        pending_pos0 = -1;
        if (ret != 0) {
            LOG_ERR("%s: llama_decode(ctx_dft, flush_pending) failed with ret = %d (n = %d)\n", __func__, ret, n);
            return false;
        }

        const float * f_rnn_out = llama_get_embeddings_nextn(ctx_dft);
        if (f_rnn_out) {
            const int32_t keep_from = std::max<int32_t>(0, n - (int32_t) F_RNN_RING);
            for (int32_t i = keep_from; i < n; ++i) {
                const llama_pos p = batch_inject.pos[i];
                if (p >= 0) {
                    const size_t slot = (size_t) p % F_RNN_RING;
                    std::memcpy(f_rnn_ring.data() + slot * (size_t) n_embd,
                            f_rnn_out + (size_t) i * n_embd,
                            (size_t) n_embd * sizeof(float));
                    f_rnn_pos[slot] = p;
                }
            }
        }
        return true;
    }

    bool buffer_target_taps(const int32_t * src_rows, int32_t n_rows, llama_pos pos0) {
        if (pending_n > 0) {
            if (pos0 <= pending_pos0) {
                pending_n    = 0;
                pending_pos0 = -1;
            } else if (pos0 < pending_pos0 + pending_n) {
                pending_n = (int32_t) (pos0 - pending_pos0);
            } else if (pos0 > pending_pos0 + pending_n) {
                if (!flush_pending()) {
                    return false;
                }
            }
            if (pending_n > 0 && pending_n + n_rows > MAX_PENDING_TAPS) {
                if (!flush_pending()) {
                    return false;
                }
            }
        }
        if (pending_n == 0) {
            pending_pos0 = pos0;
        }
        for (uint32_t t = 0; t < target_layer_ids_n; ++t) {
            const float * layer_data = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[t]);
            if (!layer_data) {
                LOG_ERR("%s: target layer %d input was not extracted\n", __func__, target_layer_ids[t]);
                return false;
            }
            for (int32_t i = 0; i < n_rows; ++i) {
                const int32_t src_row = src_rows[i];
                float       * dst = pending_taps.data() + (size_t) (pending_n + i) * (size_t) n_embd_enc + (size_t) t * (size_t) n_embd;
                const float * src = layer_data + (size_t) src_row * (size_t) n_embd;
                std::memcpy(dst, src, (size_t) n_embd * sizeof(float));
            }
        }
        pending_n += n_rows;
        return true;
    }

    bool process(const llama_batch & batch_in, llama_seq_id target_seq_id) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }
        const bool has_tokens = batch_in.token != nullptr;
        const bool has_embd   = batch_in.embd  != nullptr;
        if (has_tokens == has_embd) {
            return true;
        }

        rows_scratch.clear();
        rows_scratch.reserve(batch_in.n_tokens);
        for (int32_t k = 0; k < batch_in.n_tokens; ++k) {
            const int32_t n_seq = batch_in.n_seq_id ? batch_in.n_seq_id[k] : 1;
            for (int32_t s = 0; s < n_seq; ++s) {
                const llama_seq_id sid = batch_in.seq_id ? batch_in.seq_id[k][s] : 0;
                if (sid == target_seq_id) {
                    rows_scratch.push_back(k);
                    break;
                }
            }
        }
        if (rows_scratch.empty()) {
            return true;
        }

        const int32_t n_rows = (int32_t) rows_scratch.size();
        const llama_pos p_first = batch_in.pos[rows_scratch.front()];
        const llama_pos p_last  = batch_in.pos[rows_scratch.back()];
        if (has_embd && n_rows > 1 && p_first == p_last) {
            return true;
        }

        // Fast path: buffer small contiguous decode taps to fuse into the next draft_tree GPU pass (0 GPU launch here).
        if (p_first > 0 && n_rows <= MAX_PENDING_TAPS) {
            bool contiguous = true;
            for (int32_t i = 0; i < n_rows; ++i) {
                if (batch_in.pos[rows_scratch[i]] != p_first + i) {
                    contiguous = false;
                    break;
                }
            }
            if (contiguous) {
                return buffer_target_taps(rows_scratch.data(), n_rows, p_first);
            }
        }

        if (pending_n > 0) {
            if (p_first <= pending_pos0) {
                pending_n    = 0;
                pending_pos0 = -1;
            } else {
                if (p_first < pending_pos0 + pending_n) {
                    pending_n = (int32_t) (p_first - pending_pos0);
                }
                if (!flush_pending()) {
                    return false;
                }
            }
        }

        auto * mem_dft = llama_get_memory(ctx_dft);
        if (p_first <= 0) {
            llama_memory_clear(mem_dft, false);
            std::fill(f_rnn_pos.begin(), f_rnn_pos.end(), -1);
            gpu_f_rnn_pos = -1;
        } else if (!llama_memory_seq_rm(mem_dft, 0, p_first, -1)) {
            llama_memory_clear(mem_dft, false);
            std::fill(f_rnn_pos.begin(), f_rnn_pos.end(), -1);
            gpu_f_rnn_pos = -1;
        }
        llama_model_inkling2_dspark_set_pending_taps(model_dft, nullptr, 0);

        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
            const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);

            for (uint32_t t = 0; t < target_layer_ids_n; ++t) {
                const float * layer_data = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[t]);
                if (!layer_data) {
                    LOG_ERR("%s: target layer %d input was not extracted\n", __func__, target_layer_ids[t]);
                    return false;
                }
                for (int32_t i = 0; i < n_chunk; ++i) {
                    const int32_t src_row = rows_scratch[offset + i];
                    float       * dst = batch_inject.embd + (size_t) i * n_embd_enc + (size_t) t * n_embd;
                    const float * src = layer_data + (size_t) src_row * n_embd;
                    std::memcpy(dst, src, (size_t) n_embd * sizeof(float));
                }
            }

            batch_inject.n_tokens = n_chunk;
            for (int32_t i = 0; i < n_chunk; ++i) {
                const int32_t src_row   = rows_scratch[offset + i];
                batch_inject.pos[i]      = batch_in.pos[src_row];
                batch_inject.n_seq_id[i] = 1;
                batch_inject.seq_id[i][0] = 0;
                batch_inject.logits[i]   = false;
            }

            const int ret = llama_decode(ctx_dft, batch_inject);
            if (ret != 0) {
                LOG_ERR("%s: llama_decode(ctx_dft, inject) failed with ret = %d (n_chunk = %d)\n", __func__, ret, n_chunk);
                return false;
            }

            const float * f_rnn_out = llama_get_embeddings_nextn(ctx_dft);
            if (f_rnn_out) {
                const int32_t keep_from = std::max<int32_t>(0, n_chunk - (int32_t) F_RNN_RING);
                for (int32_t i = keep_from; i < n_chunk; ++i) {
                    const llama_pos p = batch_inject.pos[i];
                    if (p >= 0) {
                        const size_t slot = (size_t) p % F_RNN_RING;
                        std::memcpy(f_rnn_ring.data() + slot * (size_t) n_embd,
                                f_rnn_out + (size_t) i * n_embd,
                                (size_t) n_embd * sizeof(float));
                        f_rnn_pos[slot] = p;
                    }
                }
            }
        }

        return true;
    }

    bool supports_tree() const override {
        return true;
    }

    bool process_rows(
            const llama_batch & batch_in,
            const int32_t     * keep_idxs,
            int32_t             n_keep,
            llama_pos           pos0,
            llama_seq_id        target_seq_id) override {
        GGML_UNUSED(batch_in);
        GGML_UNUSED(target_seq_id);
        if (!keep_idxs || n_keep <= 0) {
            return true;
        }

        if (pos0 > 0 && n_keep <= MAX_PENDING_TAPS) {
            return buffer_target_taps(keep_idxs, n_keep, pos0);
        }

        if (pending_n > 0) {
            if (pos0 <= pending_pos0) {
                pending_n    = 0;
                pending_pos0 = -1;
            } else {
                if (pos0 < pending_pos0 + pending_n) {
                    pending_n = (int32_t) (pos0 - pending_pos0);
                }
                if (!flush_pending()) {
                    return false;
                }
            }
        }

        auto * mem_dft = llama_get_memory(ctx_dft);
        if (pos0 <= 0) {
            llama_memory_clear(mem_dft, false);
            std::fill(f_rnn_pos.begin(), f_rnn_pos.end(), -1);
            gpu_f_rnn_pos = -1;
        } else if (!llama_memory_seq_rm(mem_dft, 0, pos0, -1)) {
            llama_memory_clear(mem_dft, false);
            std::fill(f_rnn_pos.begin(), f_rnn_pos.end(), -1);
            gpu_f_rnn_pos = -1;
        }
        llama_model_inkling2_dspark_set_pending_taps(model_dft, nullptr, 0);

        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t offset = 0; offset < n_keep; offset += n_ubatch) {
            const int32_t n_chunk = std::min(n_ubatch, n_keep - offset);

            for (uint32_t t = 0; t < target_layer_ids_n; ++t) {
                const float * layer_data = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[t]);
                if (!layer_data) {
                    LOG_ERR("%s: target layer %d input was not extracted\n", __func__, target_layer_ids[t]);
                    return false;
                }
                for (int32_t i = 0; i < n_chunk; ++i) {
                    const int32_t src_row = keep_idxs[offset + i];
                    float       * dst = batch_inject.embd + (size_t) i * n_embd_enc + (size_t) t * n_embd;
                    const float * src = layer_data + (size_t) src_row * n_embd;
                    std::memcpy(dst, src, (size_t) n_embd * sizeof(float));
                }
            }

            batch_inject.n_tokens = n_chunk;
            for (int32_t i = 0; i < n_chunk; ++i) {
                batch_inject.pos[i]       = pos0 + offset + i;
                batch_inject.n_seq_id[i]  = 1;
                batch_inject.seq_id[i][0] = 0;
                batch_inject.logits[i]    = false;
            }

            const int ret = llama_decode(ctx_dft, batch_inject);
            if (ret != 0) {
                LOG_ERR("%s: llama_decode(ctx_dft, inject_rows) failed with ret = %d (n_chunk = %d)\n", __func__, ret, n_chunk);
                return false;
            }

            const float * f_rnn_out = llama_get_embeddings_nextn(ctx_dft);
            if (f_rnn_out) {
                const int32_t keep_from = std::max<int32_t>(0, n_chunk - (int32_t) F_RNN_RING);
                for (int32_t i = keep_from; i < n_chunk; ++i) {
                    const llama_pos p = batch_inject.pos[i];
                    if (p >= 0) {
                        const size_t slot = (size_t) p % F_RNN_RING;
                        std::memcpy(f_rnn_ring.data() + slot * (size_t) n_embd,
                                f_rnn_out + (size_t) i * n_embd,
                                (size_t) n_embd * sizeof(float));
                        f_rnn_pos[slot] = p;
                    }
                }
            }
        }

        return true;
    }

    void draft_tree(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            common_speculative_tree & result) override {
        result.clear();
        if (!supports_tree()) {
            draft(params, prompt_tgt, id_last, result.tokens);
            result.parents.resize(result.tokens.size());
            for (size_t i = 0; i < result.tokens.size(); ++i) {
                result.parents[i] = (int8_t) i;
            }
            result.probs.assign(result.tokens.size(), 1.0f);
            return;
        }

        last_was_suffix_bypass = false;
        last_was_mode_c        = false;

        const int32_t eff_block       = (last_dspark_accepted >= 7 && ema_p0 >= 0.90f) ? 12 : block_size;
        const int32_t n_draft_max_ext = std::min<int32_t>(params.draft.n_max, 15);
        const int32_t n_draft_req     = std::min<int32_t>(params.draft.n_max, eff_block - 1);
        if (n_draft_req <= 0 || prompt_tgt.empty()) {
            return;
        }

        const llama_pos pos0     = (llama_pos) prompt_tgt.size();
        const llama_pos feat_pos = pos0 - 1;
        const size_t    slot     = (size_t) feat_pos % F_RNN_RING;

        suffix_sync(prompt_tgt);

        // Mode A: Pre-DSpark exact long suffix bypass (linear chain, 0 ms DSpark cost)
        // Only bypass DSpark on very reliable long verbatim matches; medium matches are grafted onto the tree (Mode D).
        SuffixMatch pre_match;
        SuffixMatch pre_match3;
        if (prompt_tgt.size() >= 8) {
            constexpr int32_t MAX_TAIL = 32;
            llama_token tail[MAX_TAIL];
            const int32_t keep_p = (int32_t) std::min<size_t>(MAX_TAIL - 1, prompt_tgt.size());
            std::memcpy(tail, prompt_tgt.data() + (prompt_tgt.size() - keep_p), (size_t) keep_p * sizeof(llama_token));
            tail[keep_p] = id_last;
            const int32_t tail_len = keep_p + 1;

            pre_match = suffix_match(prompt_tgt, tail, tail_len);
            if (pre_match.pos < 0 && pos0 >= 2048) {
                pre_match3 = suffix_match3(prompt_tgt, tail, tail_len);
            }
            const int32_t min_bypass = (pre_match.count == 1 ? 11 : 13) + suffix_bypass_penalty;
            if (pre_match.pos >= 0 && pre_match.len >= min_bypass) {
                const int32_t avail = (int32_t) prompt_tgt.size() - (pre_match.pos + 1);
                if (avail >= 4) {
                    const int32_t target_prop = (pre_match.len >= 13) ? 15 : 12;
                    const int32_t n_prop = std::min({ target_prop, avail, n_draft_max_ext });
                    if (n_prop >= params.draft.n_min) {
                        result.tokens.assign(prompt_tgt.begin() + pre_match.pos + 1,
                                             prompt_tgt.begin() + pre_match.pos + 1 + n_prop);
                        result.parents.resize(n_prop);
                        result.probs.assign(n_prop, 1.0f);
                        for (int32_t i = 0; i < n_prop; ++i) {
                            result.parents[i] = (int8_t) i;
                        }
                        last_was_suffix_bypass = true;
                        last_verify_B = 1 + n_prop;
                        return;
                    }
                }
            }
        }

        if (pending_n > 0) {
            if (pos0 <= pending_pos0) {
                pending_n    = 0;
                pending_pos0 = -1;
            } else if (pos0 < pending_pos0 + pending_n) {
                pending_n = (int32_t) (pos0 - pending_pos0);
            } else if (pos0 > pending_pos0 + pending_n) {
                if (!flush_pending()) {
                    return;
                }
            }
        }

        const int32_t cur_block = 1 + n_draft_req;
        const bool use_fused_inj = (pending_n > 0 && pending_pos0 + pending_n == pos0 &&
                                    pending_n + cur_block <= (int32_t) llama_n_ubatch(ctx_dft));

        auto * mem_dft = llama_get_memory(ctx_dft);
        if (use_fused_inj) {
            if (pending_pos0 <= 0) {
                llama_memory_clear(mem_dft, false);
                std::fill(f_rnn_pos.begin(), f_rnn_pos.end(), -1);
                gpu_f_rnn_pos = -1;
            } else if (!llama_memory_seq_rm(mem_dft, 0, pending_pos0, -1)) {
                llama_memory_clear(mem_dft, false);
                std::fill(f_rnn_pos.begin(), f_rnn_pos.end(), -1);
                gpu_f_rnn_pos = -1;
            }
        } else {
            if (pending_n > 0 && !flush_pending()) {
                return;
            }
            if (gpu_f_rnn_pos != feat_pos && f_rnn_pos[slot] != feat_pos) {
                return;
            }
            llama_memory_seq_rm(mem_dft, 0, pos0, -1);
        }

        if (prompt_tgt.size() < n_vocab_synced) {
            n_vocab_synced = 0;
        }
        if (prompt_tgt.size() > n_vocab_synced) {
            llama_model_inkling2_dspark_add_draft_vocab(
                    model_dft, prompt_tgt.data() + n_vocab_synced, (uint32_t) (prompt_tgt.size() - n_vocab_synced));
            n_vocab_synced = prompt_tgt.size();
        }
        llama_model_inkling2_dspark_add_draft_vocab(model_dft, &id_last, 1);

        constexpr float trunk_min = 0.10f;
        constexpr float alt_min   = 0.16f;
        if (use_fused_inj) {
            llama_model_inkling2_dspark_set_pending_taps(model_dft, pending_taps.data(), pending_n);
        } else {
            llama_model_inkling2_dspark_set_pending_taps(model_dft, nullptr, 0);
            if (gpu_f_rnn_pos != feat_pos) {
                llama_model_inkling2_dspark_set_f_rnn(model_dft, f_rnn_ring.data() + slot * (size_t) n_embd);
                gpu_f_rnn_pos = feat_pos;
            }
        }
        llama_model_inkling2_dspark_set_tree_params(model_dft, trunk_min, alt_min);

        common_batch_clear(batch_draft);
        if (use_fused_inj) {
            for (int32_t i = 0; i < pending_n; ++i) {
                common_batch_add(batch_draft, 0, pending_pos0 + i, { 0 }, false);
            }
        }
        for (int32_t i = 0; i < cur_block; ++i) {
            common_batch_add(batch_draft, i == 0 ? id_last : 0, pos0 + i, { 0 }, false);
        }

        const int ret = llama_decode(ctx_dft, batch_draft);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode(ctx_dft, draft_tree) returned %d\n", __func__, ret);
            llama_model_inkling2_dspark_set_pending_taps(model_dft, nullptr, 0);
            llama_memory_seq_rm(mem_dft, 0, use_fused_inj ? pending_pos0 : pos0, -1);
            return;
        }

        llama_synchronize(ctx_dft);

        int32_t cand_ids[44]   = {};
        float   cand_probs[44] = {};
        llama_model_inkling2_dspark_get_tree(model_dft, cand_ids, cand_probs, 44);
        llama_memory_seq_rm(mem_dft, 0, pos0, -1);
        if (use_fused_inj) {
            gpu_f_rnn_pos = feat_pos;
            pending_n     = 0;
            pending_pos0  = -1;
            llama_model_inkling2_dspark_set_pending_taps(model_dft, nullptr, 0);
        }

        if (cand_probs[0] > 0.0f) {
            ema_p0 = 0.70f * ema_p0 + 0.30f * cand_probs[0];
        }
        last_trunk_max = n_draft_req;

        // Fixed topology of the 44 GPU candidate slots (parent == -1 means anchor root)
        static constexpr int8_t CAND_PARENT[44] = {
            -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, //  0..14: Trunk d1..d15
            -1, 15, 16, 17, 18, 19,                                     // 15..20: Alt-A d1..d6 (rank 2 at step 0)
            -1, 21, 22, 23,                                             // 21..24: Alt-B d1..d4 (rank 3 at step 0)
             0, 25, 26, 27, 28,                                         // 25..29: Alt-C d2..d6 (rank 2 at step 1)
             1, 30, 31, 32,                                             // 30..33: Alt-D d3..d6 (rank 2 at step 2)
             2, 34, 35,                                                 // 34..36: Alt-E d4..d6 (rank 2 at step 3)
             3, 37,                                                     // 37..38: Alt-F d5..d6 (rank 2 at step 4)
            15, 39, 40,                                                 // 39..41: Alt-A2 d2..d4 (rank 2 at step 1 off Alt-A)
             0, 42,                                                     // 42..43: Alt-C2 d2..d3 (rank 3 at step 1)
        };
        static constexpr int8_t CAND_DEPTH[44] = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            1, 2, 3, 4, 5, 6,
            1, 2, 3, 4,
            2, 3, 4, 5, 6,
            3, 4, 5, 6,
            4, 5, 6,
            5, 6,
            2, 3, 4,
            2, 3,
        };

        // Fallback when probs are absent (e.g. non-Metal backend): return linear trunk
        if (cand_probs[0] <= 0.0f) {
            result.tokens.assign(cand_ids, cand_ids + n_draft_req);
            result.parents.resize(n_draft_req);
            result.probs.assign(n_draft_req, 1.0f);
            for (int32_t i = 0; i < n_draft_req; ++i) {
                result.parents[i] = (int8_t) i;
            }
            return;
        }

        // Compute empirically calibrated cumulative probability cum_p[c] for all 44 DSpark + up to 12 grafted Mode D suffix candidates.
        // Empirical calibration across short & 8K/16K/32K workloads shows DSpark is near-calibrated at high p (>= 0.90)
        // but increasingly overconfident at mid/low p (0.3-0.7), closely matching quadratic scaling p * (0.60 + 0.34*p).
        static constexpr int MAX_CAND = 56;
        constexpr float alt_root_cal = 0.88f;
        constexpr float alt_cont_cal = 0.72f;
        int32_t cand_tok[MAX_CAND]    = {};
        float   cand_p[MAX_CAND]      = {};
        int8_t  cand_parent[MAX_CAND] = {};
        int8_t  cand_depth[MAX_CAND]  = {};
        float   cum_p[MAX_CAND]       = {};
        bool    valid[MAX_CAND]       = {};
        int     n_cand_pool           = 44;

        for (int c = 0; c < 44; ++c) {
            cand_tok[c]    = cand_ids[c];
            cand_p[c]      = cand_probs[c];
            cand_parent[c] = CAND_PARENT[c];
            cand_depth[c]  = CAND_DEPTH[c];
            if (cand_probs[c] <= 0.0f || CAND_DEPTH[c] > n_draft_req) {
                continue;
            }
            const int p = CAND_PARENT[c];
            if (p >= 0 && !valid[p]) {
                continue;
            }
            // Deduplicate siblings under the same parent
            bool dup = false;
            for (int j = 0; j < c; ++j) {
                if (valid[j] && cand_parent[j] == p && cand_tok[j] == cand_tok[c]) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                continue;
            }
            const float rp = std::min(0.999f, cand_probs[c]);
            const float qcal = 0.60f + 0.34f * rp;
            const float sp = (c < 15)
                ? std::min(0.999f, rp * qcal)
                : ((p < 0 || p < 15) ? std::min(0.999f, rp * alt_root_cal)
                                     : std::min(0.999f, rp * qcal * alt_cont_cal));
            const float cp = (p < 0 ? 1.0f : cum_p[p]) * sp;
            cum_p[c] = cp;
            valid[c] = (cp > 1e-4f);
        }

        // Mode D: Graft medium-length prompt/history suffix continuation (off root, or off Trunk d1 if root missed)
        // as a candidate branch in the tree. Boosts matching DSpark tokens or verifies both paths!
        auto graft_suffix_chain = [&](const SuffixMatch & graft_m, bool is_m4, int start_parent, int start_depth, float base_cp) {
            const int32_t avail = (int32_t) prompt_tgt.size() - (graft_m.pos + 1);
            const int32_t max_d = std::min<int32_t>({ avail, n_draft_max_ext - start_depth, is_m4 ? 10 : 4 });
            const float p0_suf  = (graft_m.count == 1)
                ? (is_m4 ? std::min(0.86f, 0.50f + 0.05f * (float) graft_m.len) : 0.48f)
                : std::min(0.70f, 0.36f + 0.04f * (float) graft_m.len);
            float cur_cp = base_cp;
            int cur_parent = start_parent;
            for (int32_t d = 0; d < max_d; ++d) {
                const llama_token tok = prompt_tgt[graft_m.pos + 1 + d];
                const float step_p = (d == 0) ? p0_suf : (is_m4 ? 0.88f : 0.78f);
                cur_cp *= step_p;
                if (cur_parent >= 0) {
                    cur_cp = std::min(cur_cp, cum_p[cur_parent] * 0.999f);
                }
                int match_idx = -1;
                for (int j = 0; j < n_cand_pool; ++j) {
                    if (valid[j] && cand_parent[j] == cur_parent && cand_tok[j] == tok) {
                        match_idx = j;
                        break;
                    }
                }
                if (match_idx >= 0) {
                    if (cur_cp > cum_p[match_idx]) {
                        cum_p[match_idx] = cur_cp;
                    }
                    cur_parent = match_idx;
                } else if (n_cand_pool < MAX_CAND) {
                    const int c = n_cand_pool++;
                    cand_tok[c]    = tok;
                    cand_p[c]      = step_p;
                    cand_parent[c] = (int8_t) cur_parent;
                    cand_depth[c]  = (int8_t) (start_depth + d + 1);
                    cum_p[c]       = cur_cp;
                    valid[c]       = (cur_cp > 1e-4f);
                    cur_parent     = c;
                } else {
                    break;
                }
            }
        };

        const bool ok_m4 = (pre_match.pos >= 0 &&
            ((pre_match.count == 1 && pre_match.len >= 4) || (pre_match.count <= 3 && pre_match.len >= 6)));
        const bool ok_m3 = (!ok_m4 && pos0 >= 2048 && pre_match3.pos >= 0 && pre_match3.count == 1);
        if (ok_m4 || ok_m3) {
            graft_suffix_chain(ok_m4 ? pre_match : pre_match3, ok_m4, -1, 0, 1.0f);
        } else if (pos0 >= 1024 && valid[0] && cum_p[0] >= 0.45f && prompt_tgt.size() >= 8) {
            // Mode D1: try 4-gram suffix match starting after Trunk d1 (catches identifiers/spans beginning at d1)
            constexpr int32_t MAX_TAIL = 32;
            llama_token tail_d1[MAX_TAIL];
            const int32_t keep_p = (int32_t) std::min<size_t>(MAX_TAIL - 2, prompt_tgt.size());
            std::memcpy(tail_d1, prompt_tgt.data() + (prompt_tgt.size() - keep_p), (size_t) keep_p * sizeof(llama_token));
            tail_d1[keep_p]     = id_last;
            tail_d1[keep_p + 1] = cand_ids[0];
            const SuffixMatch m_d1 = suffix_match(prompt_tgt, tail_d1, keep_p + 2);
            if (m_d1.pos >= 0 && ((m_d1.count == 1 && m_d1.len >= 5) || (m_d1.count <= 3 && m_d1.len >= 7))) {
                graft_suffix_chain(m_d1, true, 0, 1, cum_p[0]);
            }
        }

        // Collect and sort valid candidates by descending cum_p (prefix-closed for every K!)
        int order[MAX_CAND];
        int n_valid = 0;
        for (int c = 0; c < n_cand_pool; ++c) {
            if (valid[c]) {
                order[n_valid++] = c;
            }
        }
        std::sort(order, order + n_valid, [&](int a, int b) {
            if (cum_p[a] != cum_p[b]) { return cum_p[a] > cum_p[b]; }
            if (cand_depth[a] != cand_depth[b]) { return cand_depth[a] < cand_depth[b]; }
            return a < b;
        });

        constexpr int32_t tree_kmax = 15; // B <= 16 (supports up to 14-step trunk + deep branches)
        const int32_t k_cap = std::min({ n_draft_max_ext, tree_kmax, n_valid });

        // Measured monotonic M4 Pro cycle-time model T_cycle(B, pos0) (target, DSpark and host overhead),
        // matching the MoE distinct-expert slope (~3.5-4.5 ms/tok across B=1..8) and the B=8 -> B=9 NR1=8->16 step.
        static constexpr float T_SHORT[17] = {
            32.8f, // unused
            32.8f, 36.3f, 40.8f, 45.3f, // B=1..4  (target + 3.8ms DSpark+host)
            50.3f, 54.8f, 58.3f, 61.3f, // B=5..8  (NR1=8, FA r24 sweet spot)
            67.8f, 71.1f, 74.1f, 76.3f, // B=9..12 (NR1=16 bk32 + Split-K)
            80.8f, 85.3f, 89.8f, 94.3f  // B=13..16 (stays on sorted mul_mv_id)
        };
        static constexpr float T_32K[17] = {
            38.1f, // unused
            38.1f, 42.6f, 48.1f, 53.6f, // B=1..4  (target + 5.1ms DSpark+host)
            58.6f, 63.1f, 66.6f, 69.7f, // B=5..8  (NR1=8, FA r24_nsg4 sweet spot)
            78.0f, 82.0f, 86.5f, 90.5f, // B=9..12 (NR1=16 bk32 + FA r40)
            96.0f, 101.5f, 107.5f, 113.5f // B=13..16
        };
        const float ctx_frac = std::min(1.25f, std::max(0.0f, ((float) pos0 - 64.0f) / 31936.0f));

        int   best_k    = 0;
        float best_rate = 1.0f / (T_SHORT[1] + ctx_frac * (T_32K[1] - T_SHORT[1]));
        float sum_e     = 1.0f;
        for (int k = 1; k <= k_cap; ++k) {
            sum_e += cum_p[order[k - 1]];
            const int   B    = 1 + k;
            const float tc   = T_SHORT[B] + ctx_frac * (T_32K[B] - T_SHORT[B]);
            const float rate = sum_e / tc;
            if (rate > best_rate) {
                best_rate = rate;
                best_k    = k;
            }
        }

        // Re-order the best_k chosen candidate slots in BFS order (depth ASC, slot ASC)
        int chosen[16];
        for (int i = 0; i < best_k; ++i) {
            chosen[i] = order[i];
        }
        std::sort(chosen, chosen + best_k, [&](int a, int b) {
            if (cand_depth[a] != cand_depth[b]) { return cand_depth[a] < cand_depth[b]; }
            return a < b;
        });

        int8_t slot_to_node[MAX_CAND];
        std::fill(slot_to_node, slot_to_node + MAX_CAND, (int8_t) -1);
        result.tokens.resize((size_t) best_k);
        result.parents.resize((size_t) best_k);
        result.probs.resize((size_t) best_k);
        for (int i = 0; i < best_k; ++i) {
            const int c = chosen[i];
            const int p = cand_parent[c];
            slot_to_node[c]   = (int8_t) (i + 1);
            result.tokens[i]  = cand_tok[c];
            result.probs[i]   = cand_p[c];
            result.parents[i] = (p < 0) ? (int8_t) 0 : slot_to_node[p];
        }

        // Find deepest selected trunk slot >= 4 (d5..d15) so Mode C fires whenever trunk enters a prompt span
        int32_t trunk_tip = -1;
        for (int32_t s = n_draft_req - 1; s >= 4; --s) {
            if (slot_to_node[s] > 0) {
                trunk_tip = s;
                break;
            }
        }

        // Mode C: Post-DSpark exact suffix extension when trunk reaches the block ceiling with solid confidence
        // (or deep trunk at long context). Attaches directly off the deepest trunk node.
        const bool allow_mode_c = (trunk_tip >= 4 && slot_to_node[trunk_tip] > 0 &&
            (int32_t) result.size() < n_draft_max_ext) &&
            ((pos0 >= 2048 && cum_p[trunk_tip] >= 0.45f) ||
             (trunk_tip == n_draft_req - 1 && cum_p[trunk_tip] >= 0.55f));
        if (allow_mode_c) {
            constexpr int32_t MAX_TAIL = 32;
            llama_token tail[MAX_TAIL];
            const int32_t n_trunk_used = trunk_tip + 1;
            const int32_t want_p = std::max<int32_t>(0, MAX_TAIL - 1 - n_trunk_used);
            const int32_t keep_p = (int32_t) std::min<size_t>((size_t) want_p, prompt_tgt.size());
            int32_t len = 0;
            if (keep_p > 0) {
                std::memcpy(tail, prompt_tgt.data() + (prompt_tgt.size() - keep_p), (size_t) keep_p * sizeof(llama_token));
                len += keep_p;
            }
            tail[len++] = id_last;
            for (int32_t s = 0; s < n_trunk_used && len < MAX_TAIL; ++s) {
                tail[len++] = cand_ids[s];
            }

            const SuffixMatch ext = suffix_match(prompt_tgt, tail, len);
            const bool ext_ok = (pos0 >= 2048)
                ? (ext.pos >= 0 && ((ext.count == 1 && ext.len >= 5) || ext.len >= 7))
                : (ext.pos >= 0 && ((ext.count == 1 && ext.len >= 6) || ext.len >= 8));
            if (ext_ok) {
                const int32_t max_total = std::min<int32_t>(n_draft_max_ext, pos0 >= 2048 ? 15 : 14);
                const int32_t ext_cap   = (ext.len >= 7) ? 7 : 4;
                const int32_t avail     = (int32_t) prompt_tgt.size() - (ext.pos + 1);
                const int32_t n_ext     = std::min({ ext_cap, avail, max_total - (int32_t) result.size() });
                if (n_ext > 0) {
                    const size_t old_n = result.size();
                    const int8_t tip_node = slot_to_node[trunk_tip];
                    for (int32_t e = 0; e < n_ext; ++e) {
                        result.tokens.push_back(prompt_tgt[ext.pos + 1 + e]);
                        result.parents.push_back(e == 0 ? tip_node : (int8_t) (old_n + e));
                        result.probs.push_back(1.0f);
                    }
                    last_was_mode_c = true;
                    last_trunk_tip  = trunk_tip + 1;
                }
            }
        }

        if ((int32_t) result.size() < params.draft.n_min) {
            result.clear();
        }
        last_verify_B = 1 + (int32_t) result.size();
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        result.clear();
        last_was_suffix_bypass = false;
        last_was_mode_c        = false;

        const int32_t eff_block       = (last_dspark_accepted >= 7 && ema_p0 >= 0.90f) ? 12 : block_size;
        const int32_t n_draft_max_ext = std::min<int32_t>(params.draft.n_max, 16);
        const int32_t n_draft_req     = std::min<int32_t>(params.draft.n_max, eff_block - 1);
        if (n_draft_req <= 0 || prompt_tgt.empty()) {
            return;
        }

        const llama_pos pos0     = (llama_pos) prompt_tgt.size();
        const llama_pos feat_pos = pos0 - 1;
        const size_t    slot     = (size_t) feat_pos % F_RNN_RING;

        // Keep the O(accepted) incremental suffix index in sync with prompt + accepted tokens.
        suffix_sync(prompt_tgt);

        // Mode A: Pre-DSpark exact long suffix bypass (0.01 ms CPU instead of ~5.5 ms DSpark GPU).
        // Only fires on unambiguous long spans (>= 10 tokens, or >= 8 tokens with a unique occurrence)
        // so common C++ idioms in large 16K-32K codebases never trigger false bypasses.
        if (prompt_tgt.size() >= 8) {
            constexpr int32_t MAX_TAIL = 32;
            llama_token tail[MAX_TAIL];
            const int32_t keep_p = (int32_t) std::min<size_t>(MAX_TAIL - 1, prompt_tgt.size());
            std::memcpy(tail, prompt_tgt.data() + (prompt_tgt.size() - keep_p), (size_t) keep_p * sizeof(llama_token));
            tail[keep_p] = id_last;
            const int32_t tail_len = keep_p + 1;

            const SuffixMatch m = suffix_match(prompt_tgt, tail, tail_len);
            const int32_t min_bypass = (m.count == 1 ? 8 : 10) + suffix_bypass_penalty;
            if (m.pos >= 0 && m.len >= min_bypass) {
                const int32_t avail = (int32_t) prompt_tgt.size() - (m.pos + 1);
                if (avail >= 4) {
                    const int32_t target_prop = (m.len >= 12) ? (pos0 >= 4096 ? 14 : 12) : 10;
                    const int32_t n_prop = std::min({ target_prop, avail, n_draft_max_ext });
                    if (n_prop >= params.draft.n_min) {
                        result.assign(prompt_tgt.begin() + m.pos + 1,
                                      prompt_tgt.begin() + m.pos + 1 + n_prop);
                        last_was_suffix_bypass = true;
                        return;
                    }
                }
            }
        }

        if (pending_n > 0) {
            if (pos0 <= pending_pos0) {
                pending_n    = 0;
                pending_pos0 = -1;
            } else {
                if (pos0 < pending_pos0 + pending_n) {
                    pending_n = (int32_t) (pos0 - pending_pos0);
                }
                if (!flush_pending()) {
                    return;
                }
            }
        }

        if (gpu_f_rnn_pos != feat_pos && f_rnn_pos[slot] != feat_pos) {
            LOG_DBG("%s: f_rnn not cached for pos %d (have %d), skipping 1 draft round\n",
                    __func__, (int) feat_pos, (int) f_rnn_pos[slot]);
            return;
        }

        auto * mem_dft = llama_get_memory(ctx_dft);
        llama_memory_seq_rm(mem_dft, 0, pos0, -1);

        // Incrementally add newly seen prompt/output tokens to the draft vocabulary in O(accepted) instead of O(L_ctx).
        if (prompt_tgt.size() < n_vocab_synced) {
            n_vocab_synced = 0;
        }
        if (prompt_tgt.size() > n_vocab_synced) {
            llama_model_inkling2_dspark_add_draft_vocab(
                    model_dft, prompt_tgt.data() + n_vocab_synced, (uint32_t) (prompt_tgt.size() - n_vocab_synced));
            n_vocab_synced = prompt_tgt.size();
        }
        llama_model_inkling2_dspark_add_draft_vocab(model_dft, &id_last, 1);

        // Context-adaptive stopping threshold: keeps high acceptance while avoiding empty-draft cycles when p_0 >= 0.35.
        const float default_cum_min = (pos0 >= 2048) ? 0.46f : 0.50f;
        const float default_p0_min  = 0.35f;
        const float cum_min     = params.draft.p_min > 0.0f ? params.draft.p_min : default_cum_min;
        const float p_min_first = params.draft.p_min > 0.0f ? params.draft.p_min : default_p0_min;
        llama_model_inkling2_dspark_set_pending_taps(model_dft, nullptr, 0);
        if (gpu_f_rnn_pos != feat_pos) {
            llama_model_inkling2_dspark_set_f_rnn(model_dft, f_rnn_ring.data() + slot * (size_t) n_embd);
            gpu_f_rnn_pos = feat_pos;
        }
        llama_model_inkling2_dspark_set_cum_min(model_dft, cum_min);

        const int32_t cur_block = 1 + n_draft_req;
        common_batch_clear(batch_draft);
        for (int32_t i = 0; i < cur_block; ++i) {
            common_batch_add(batch_draft, i == 0 ? id_last : 0, pos0 + i, { 0 }, false);
        }

        const int ret = llama_decode(ctx_dft, batch_draft);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode(ctx_dft, draft) returned %d\n", __func__, ret);
            llama_memory_seq_rm(mem_dft, 0, pos0, -1);
            return;
        }

        llama_synchronize(ctx_dft);
        std::fill(draft_probs_scratch.begin(), draft_probs_scratch.begin() + n_draft_req, 0.0f);
        llama_model_inkling2_dspark_get_draft_ids_and_probs(
                model_dft, draft_ids_scratch.data(), draft_probs_scratch.data(), (uint32_t) n_draft_req);

        // Remove temporary block KV entries [pos0 .. pos0 + block_size - 1]
        llama_memory_seq_rm(mem_dft, 0, pos0, -1);

        int32_t n_keep = n_draft_req;
        float final_cum_p = 1.0f;
        const bool have_probs = draft_probs_scratch[0] > 0.0f;
        if (have_probs) {
            float cum_p = 1.0f;
            n_keep = 0;
            for (int32_t s = 0; s < n_draft_req; ++s) {
                cum_p *= draft_probs_scratch[s];
                const float thold = (s == 0) ? p_min_first : cum_min;
                if (cum_p < thold) {
                    break;
                }
                n_keep = s + 1;
                final_cum_p = cum_p;
            }
        }

        result.assign(draft_ids_scratch.begin(), draft_ids_scratch.begin() + n_keep);

        // Mode C: Post-DSpark exact suffix extension when DSpark hits its 7-token ceiling with high confidence.
        if (n_keep == n_draft_req && n_keep < n_draft_max_ext && final_cum_p >= 0.65f) {
            constexpr int32_t MAX_TAIL = 32;
            llama_token tail[MAX_TAIL];
            const int32_t want_p = std::max<int32_t>(0, MAX_TAIL - 1 - n_keep);
            const int32_t keep_p = (int32_t) std::min<size_t>((size_t) want_p, prompt_tgt.size());
            int32_t len = 0;
            if (keep_p > 0) {
                std::memcpy(tail, prompt_tgt.data() + (prompt_tgt.size() - keep_p), (size_t) keep_p * sizeof(llama_token));
                len += keep_p;
            }
            tail[len++] = id_last;
            for (int32_t s = 0; s < n_keep && len < MAX_TAIL; ++s) {
                tail[len++] = draft_ids_scratch[s];
            }

            const SuffixMatch ext = suffix_match(prompt_tgt, tail, len);
            if (ext.pos >= 0 && ((ext.count == 1 && ext.len >= 6) || ext.len >= 8)) {
                const int32_t max_total = std::min<int32_t>(n_draft_max_ext, pos0 >= 4096 ? 13 : 11);
                const int32_t ext_cap   = (ext.len >= 7) ? 6 : 4;
                const int32_t avail     = (int32_t) prompt_tgt.size() - (ext.pos + 1);
                const int32_t n_ext     = std::min({ ext_cap, avail, max_total - n_keep });
                if (n_ext > 0) {
                    result.insert(result.end(),
                                  prompt_tgt.begin() + ext.pos + 1,
                                  prompt_tgt.begin() + ext.pos + 1 + n_ext);
                    last_was_mode_c = true;
                    last_trunk_tip  = n_keep;
                }
            }
        }

        if ((int32_t) result.size() < params.draft.n_min) {
            result.clear();
        }
    }

    void accept(uint16_t n_accepted) override {
        last_n_accepted = (int32_t) n_accepted;
        if (last_was_suffix_bypass) {
            last_dspark_accepted = 0;
            if (n_accepted < 2) {
                suffix_bypass_penalty = std::min<int32_t>(suffix_bypass_penalty + 2, 6);
            } else if (n_accepted >= 4 && suffix_bypass_penalty > 0) {
                suffix_bypass_penalty--;
            }
        } else {
            if (last_was_mode_c && last_trunk_tip > 0 && n_accepted > (uint16_t) last_trunk_tip) {
                last_dspark_accepted = 0;
            } else {
                last_dspark_accepted = (int32_t) n_accepted;
            }
            if (suffix_bypass_penalty > 0) {
                suffix_bypass_penalty--;
            }
        }
        last_was_suffix_bypass = false;
        last_was_mode_c        = false;
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return std::min<int32_t>(params.draft.n_max, 15);
    }

    int32_t n_min(const common_params_speculative & params) const override {
        return params.draft.n_min;
    }
};

struct common_speculative_state_eagle3 : public common_speculative_state {
    common_speculative_state_eagle3(enum common_speculative_type type) : common_speculative_state(type) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & draft_tokens) override {
        // TODO: implement
        GGML_UNUSED(params);
        GGML_UNUSED(prompt_tgt);
        GGML_UNUSED(id_last);
        GGML_UNUSED(draft_tokens);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return params.draft.n_max;
    }

    int32_t n_min(const common_params_speculative & params) const override {
        return params.draft.n_min;
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_state_ngram_simple : public common_speculative_state {
    common_ngram_simple_config config;

    common_speculative_state_ngram_simple(
            enum common_speculative_type type,
            common_ngram_simple_config config)
        : common_speculative_state(type), config(config) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {

        result = common_ngram_simple_draft(config, prompt_tgt, id_last);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & /*params*/) const override {
        return config.size_mgram;
    }

    int32_t n_min(const common_params_speculative & /*params*/) const override {
        return config.size_mgram;
    }
};

struct common_speculative_state_ngram_map_k : public common_speculative_state {
    // draft ngram map for speculative decoding without draft model
    common_ngram_map config;

    common_speculative_state_ngram_map_k(
            enum common_speculative_type type,
            common_ngram_map config)
        : common_speculative_state(type), config(std::move(config)) {}

    void begin(const llama_tokens & prompt) override {
        common_ngram_map_begin(config, prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        common_ngram_map_draft(config, prompt_tgt, id_last, result);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        common_ngram_map_accept(config, n_accepted);
    }

    int32_t n_max(const common_params_speculative & /*params*/) const override {
        return config.size_value;
    }

    int32_t n_min(const common_params_speculative & /*params*/) const override {
        return config.size_value;
    }
};

struct common_speculative_state_ngram_mod : public common_speculative_state {
    common_ngram_mod & mod;

    // the last position in the prompt that was added to the ngram container
    size_t i_last = 0;

    // length of the last drafted n‑gram (number of tokens returned by draft)
    size_t n_draft_last = 0;

    // consecutive accept rounds with low acceptance fraction (< 0.5)
    int n_low = 0;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    common_speculative_state_ngram_mod(enum common_speculative_type type, common_ngram_mod & mod)
        : common_speculative_state(type), mod(mod), verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));
    }

    void begin(const llama_tokens & prompt) override {
        i_last = 0;

        n_draft_last = 0;

        const size_t n = mod.get_n();

        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        const auto & sparams = params.ngram_mod;

        n_draft_last = 0;

        const size_t cur_len = prompt_tgt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (i_last + 32 < cur_len) {
            for (size_t i = i_last; i < cur_len - n; ++i) {
                mod.add(prompt_tgt.data() + i);
            }

            i_last = cur_len - n;
        }

        result.resize(n + sparams.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt_tgt[cur_len - n + 1 + i];
        }
        result[n - 1] = id_last;

        for (int i = 0; i < sparams.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < sparams.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n‑gram for later acceptance analysis
        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted) override {
        if (verbose) {
            LOG_INF("%s: accepted %d tokens from %zu drafted tokens\n", __func__, n_accepted, n_draft_last);
        }

        // compute acceptance fraction if we have a recorded draft length
        if (n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)n_draft_last;
            if (f_acc < 0.5) {
                n_low++;
                if (n_low >= 3) {
                    LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, n_low);

                    mod.reset();
                    n_low = 0;
                    i_last = 0;
                }
            } else {
                n_low = 0;
            }
        }
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return params.ngram_mod.n_max;
    }

    int32_t n_min(const common_params_speculative & params) const override {
        return params.ngram_mod.n_min;
    }
};

struct common_speculative_state_ngram_cache : public common_speculative_state {
    uint16_t n_draft;
    bool save_dynamic;
    bool save_static;

    common_ngram_cache ngram_cache_context;
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;

    size_t cache_size = 0; // number of tokens in n-gram cache

    common_speculative_state_ngram_cache(
            const enum common_speculative_type type,
            const std::string & path_static,
            const std::string & path_dynamic,
            uint16_t            n_draft,
            bool                save_dynamic,
            bool                save_static)
        : common_speculative_state(type)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        if (!path_static.empty()) {
            try {
                ngram_cache_static = common_ngram_cache_load(path_static);
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        if (cache_size < prompt_tgt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt_tgt.size() + 1 - cache_size);
            for (size_t j = cache_size; j < prompt_tgt.size(); ++j) {
                tokens_new.push_back(prompt_tgt[j]);
            }
            tokens_new.push_back(id_last); // add the last token

            // Update context ngram cache with new prompt_tgt:
            common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            cache_size = prompt_tgt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt_tgt.size() + 1);
        for (size_t j = 0; j < prompt_tgt.size(); ++j) {
            inp.push_back(prompt_tgt[j]);
        }
        inp.push_back(id_last);

        result.push_back(id_last);

        common_ngram_cache_draft(inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                ngram_cache_context,
                ngram_cache_dynamic,
                ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    void accept(uint16_t n_accepted) override {
        // TODO: noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & /*params*/) const override {
        return n_draft;
    }

    int32_t n_min(const common_params_speculative & /*params*/) const override {
        return 0;
    }
};

struct common_speculative {
    std::vector<std::unique_ptr<common_speculative_state>> impls; // list of implementations to use and their states

    common_speculative_state * curr_impl = nullptr; // current implementation in use (for stats)
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_state_ngram_cache create_state_ngram_cache(
        const std::string & path_static, const std::string & path_dynamic,
        const common_speculative_config & config) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_state_ngram_cache state(config.type, path_static, path_dynamic, n_draft, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str() {
    std::string result;
    for (size_t i = 0; i < common_speculative_types.size(); i++) {
        if (i > 0) {
            result += ", ";
        }
        result += common_speculative_type_to_str(common_speculative_types[i]);
    }
    return result;
}

std::string common_speculative_type_to_str(enum common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT:         return "draft";
        case COMMON_SPECULATIVE_TYPE_EAGLE3:        return "eagle3";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram_simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram_map_k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram_map_k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram_mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram_cache";
        default:                                    return "unknown";
    }
}

enum common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt) {
    llama_context * ctx_dft = nullptr;
    const bool is_inkling2_dspark = params.draft.model && llama_model_dspark_block_size(params.draft.model) > 0;
    if (params.draft.model) {
        if (is_inkling2_dspark) {
            // the target rolls a rejected tail back from a short history of its short-convolution states (32
            // positions, of which 3 must stay), so no draft source may propose more than it can roll back
            constexpr int32_t n_rollback_max = 24;
            params.ngram_mod.n_max       = std::min<int32_t>(params.ngram_mod.n_max, n_rollback_max);
            params.ngram_mod.n_min       = std::min<int32_t>(params.ngram_mod.n_min, params.ngram_mod.n_max);
            params.ngram_simple.size_m   = std::min<uint16_t>(params.ngram_simple.size_m,  n_rollback_max);
            params.ngram_map_k.size_m    = std::min<uint16_t>(params.ngram_map_k.size_m,   n_rollback_max);
            params.ngram_map_k4v.size_m  = std::min<uint16_t>(params.ngram_map_k4v.size_m, n_rollback_max);

            if (!llama_model_inkling2_dspark_attach_target(params.draft.model, llama_get_model(ctx_tgt))) {
                LOG_ERR("%s", "failed to attach target model to Inkling2 DSpark draft\n");
                return nullptr;
            }
            if (params.draft.cparams.n_ubatch == 0 || params.draft.cparams.n_ubatch > 512) {
                params.draft.cparams.n_ubatch = 512;
            }
            params.draft.cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            // Keep full SWA KV history in the 3-layer DSpark drafter (2 SWA layers = 8 KiB/tok, lazily
            // allocated 8K cells at a time) so rolling back to any target checkpoint preserves pos 0..n_past-1
            // across all 3 layers without evicting earlier SWA cells or losing the pos=0 attention sink.
            params.draft.cparams.swa_full = true;
        }
        params.draft.cparams.ctx_other   = ctx_tgt;
        params.draft.cparams.ssd_options = nullptr;
        ctx_dft = llama_init_from_model(params.draft.model, params.draft.cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s", "failed to create draft context\n");
            return nullptr;
        }
    }

    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        bool has_draft = !params.draft.mparams.path.empty();
        bool has_draft_eagle3 = false; // TODO PR-18039: if params.speculative.eagle3

        bool has_ngram_cache   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);
        bool has_ngram_simple  = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        bool has_ngram_map_k   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        bool has_ngram_map_k4v = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        bool has_ngram_mod     = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD);

        // In a more complex implementation we could use the same implementation but with different parameters.
        // This was initially used in PR-18471 but removed to simplify the code.
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            auto & sparams = params.ngram_mod;

            if (!sparams.obj) {
                sparams.obj = std::make_shared<common_ngram_mod>(sparams.n_match, 4*1024*1024);

                LOG_INF("%s: initialized ngram_mod with n_match=%d, size=%zu (%.3f MB)\n", __func__,
                        sparams.n_match, sparams.obj->size(), (float)(sparams.obj->size_bytes())/1024/1024);

                if (sparams.n_match < 16) {
                    LOG_WRN("%s: ngram_mod n_match=%d is too small - poor quality is possible, "
                            "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, sparams.n_match);
                }
            }

            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_draft) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_EAGLE3, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_state>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_DBG("%s: adding implementation %s\n", __func__, common_speculative_type_to_str(config.type).c_str());
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT: {
                if (is_inkling2_dspark) {
                    impls.push_back(std::make_unique<common_speculative_state_inkling2_dspark>(
                        config.type, ctx_tgt, ctx_dft));
                    break;
                }
                const bool use_ckpt = common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                impls.push_back(std::make_unique<common_speculative_state_draft>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .ctx_dft      = */ ctx_dft,
                    /* .replacements = */ params.draft.replacements,
                    /* .use_ckpt     = */ use_ckpt
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_state_eagle3>(config.type));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_state_ngram_simple>(
                    /* .type  = */ config.type,
                    /* .state = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(std::make_unique<common_speculative_state_ngram_map_k>(
                    (config.type),
                    get_common_ngram_map(config.type, config.params.ngram_map_k)
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                GGML_ASSERT(config.params.ngram_mod.obj);
                impls.push_back(std::make_unique<common_speculative_state_ngram_mod>(config.type, *config.params.ngram_mod.obj));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(params.ngram_cache.lookup_cache_static, params.ngram_cache.lookup_cache_dynamic, config);
                impls.push_back(std::make_unique<common_speculative_state_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .impls     = */ std::move(impls),
        /* .curr_impl = */ nullptr,
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(prompt);
        impl->n_call_begin++;
    }
}

void common_speculative_clear(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->clear();
    }
}

bool common_speculative_save_state_file(
        common_speculative * spec,
        const char         * path,
        int64_t              n_tokens) {
    if (spec == nullptr || path == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->save_state_file(path, n_tokens)) {
            return true;
        }
    }
    return false;
}

bool common_speculative_load_state_file(
        common_speculative * spec,
        const char         * path,
        int64_t              n_tokens_keep) {
    if (spec == nullptr || path == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->load_state_file(path, n_tokens_keep)) {
            return true;
        }
    }
    return false;
}

bool common_speculative_process(
        common_speculative * spec,
        const llama_batch & batch,
        llama_seq_id target_seq_id) {
    if (spec == nullptr) {
        return true;
    }

    bool result = true;
    for (auto & impl : spec->impls) {
        result = result && impl->process(batch, target_seq_id);
    }
    return result;
}

bool common_speculative_process_rows(
        common_speculative * spec,
        const llama_batch  & batch,
        const int32_t      * keep_idxs,
        int32_t              n_keep,
        llama_pos            pos0,
        llama_seq_id         target_seq_id) {
    if (spec == nullptr) {
        return true;
    }

    bool result = true;
    for (auto & impl : spec->impls) {
        result = result && impl->process_rows(batch, keep_idxs, n_keep, pos0, target_seq_id);
    }
    return result;
}

bool common_speculative_supports_tree(const common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }
    for (const auto & impl : spec->impls) {
        if (impl->supports_tree()) {
            return true;
        }
    }
    return false;
}

common_speculative_tree common_speculative_draft_tree(
        common_speculative * spec,
        const common_params_speculative & params,
        const llama_tokens & prompt_tgt,
        llama_token id_last) {
    common_speculative_tree result;

    spec->curr_impl = nullptr;

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft_tree(params, prompt_tgt, id_last, result);
            impl->n_call_draft++;
        }

        {
            const int n_min = impl->n_min(params);

            if (!result.empty() && (int) result.size() < n_min) {
                LOG_DBG("%s: ignoring small tree draft: %d < %d\n", __func__, (int) result.size(), n_min);
                result.clear();
            }
        }

        if (!result.empty()) {
            spec->curr_impl = impl.get();
            impl->n_gen_drafts++;
            impl->n_gen_tokens += result.size();
            break;
        }
    }

    return result;
}

llama_tokens common_speculative_draft(
        common_speculative * spec,
        const common_params_speculative & params,
        const llama_tokens & prompt_tgt, // specified in target model vocab
        llama_token id_last) {
    llama_tokens result;

    spec->curr_impl = nullptr; // reset current implementation

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(params, prompt_tgt, id_last, result);
            impl->n_call_draft++;
        }

        {
            const int n_min = impl->n_min(params);

            if (!result.empty() && (int) result.size() < n_min) {
                LOG_DBG("%s: ignoring small draft: %d < %d\n", __func__, (int) result.size(), n_min);
                result.clear();
            }
        }

        if (!result.empty()) {
            LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                    common_speculative_type_to_str(impl.get()->type).c_str(), prompt_tgt.size(),
                    impl.get()->n_call_draft, result.size());

            spec->curr_impl = impl.get(); // set current implementation for stats
            impl->n_gen_drafts++;
            impl->n_gen_tokens += result.size();

            break; // we have a draft, so break out of the loop and return it.
        }
    }

    return result;
}

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted) {
    if (n_accepted == 0) {
        return;
    }

    common_speculative_state * impl = spec->curr_impl;

    GGML_ASSERT(impl);

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(n_accepted);
        impl->n_call_accept++;
    }
}

int32_t common_speculative_n_max(const common_speculative * spec, const common_params_speculative & params) {
    if (spec == nullptr) {
        return 0;
    }

    int32_t n_max = 0;
    for (const auto & impl : spec->impls) {
        n_max = std::max(n_max, impl->n_max(params));
    }

    return n_max;
}

int32_t common_speculative_n_min(const common_speculative * spec, const common_params_speculative & params) {
    if (spec == nullptr) {
        return 0;
    }

    int32_t n_min = 0;
    for (const auto & impl : spec->impls) {
        n_min = std::max(n_min, impl->n_min(params));
    }

    return n_min;
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        LOG_INF("statistics %s: #calls(b,g,a) = %zu %zu %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());
    }
}
