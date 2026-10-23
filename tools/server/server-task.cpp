#include "server-task.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"

#include <algorithm>
#include <cinttypes>
#include <filesystem>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

using json = nlohmann::ordered_json;

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_parser_params.format)},
            {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
            {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
            {"generation_prompt",         chat_parser_params.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.type",          common_speculative_type_to_str(speculative.type)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_parser_params.format)},
        {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
        {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
        {"generation_prompt",         chat_parser_params.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.type",          common_speculative_type_to_str(speculative.type)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
common_chat_msg task_result_state::update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls,
        const common_chat_output_source * source) {
    generated_text += text_added;
    if (source && source->valid) {
        // Cumulative snapshot: chunks may combine tokens or end inside one.
        // Offsets always refer to generated_text, never to the last token.
        output_source = *source;
    }
    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\n", generated_text.c_str());
    common_chat_msg new_msg;

    try {
        new_msg = common_chat_parse(
            generated_text,
            is_partial,
            chat_parser_params, &output_source);
    } catch (const std::runtime_error & e) {
        // The PEG chat parser threw because the model output didn't match
        // the auto-generated grammar. Common cause: a reasoning model that
        // got stuck in <think>...</think> and emitted EOS without ever
        // closing </think>, so the optional reasoning rule's until(</think>)
        // never matched and the parser couldn't make sense of the rest.
        //
        // Rather than failing the entire request with a 500, fall back to a
        // best-effort split: if the prompt primed reasoning and we have
        // </think>, split there; otherwise treat everything as reasoning
        // (since the prompt told the model to reason). The caller still
        // gets a usable assistant message.
        new_msg = common_chat_msg();
        new_msg.role = "assistant";
        const bool primed_reasoning =
            chat_parser_params.reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK &&
            string_ends_with(chat_parser_params.generation_prompt, "<think>");
        const size_t close_pos = common_chat_find_control(generated_text, "</think>", chat_parser_params, output_source);
        if (primed_reasoning && close_pos != std::string::npos) {
            std::string reasoning = generated_text.substr(0, close_pos);
            if (common_chat_find_control(reasoning, "<think>", chat_parser_params, output_source) == 0) {
                reasoning.erase(0, std::string("<think>").size());
            }
            new_msg.reasoning_content = std::move(reasoning);
            new_msg.content = generated_text.substr(close_pos + std::string("</think>").size());
        } else if (primed_reasoning) {
            std::string text = generated_text;
            if (common_chat_find_control(text, "<think>", chat_parser_params, output_source) == 0) {
                text.erase(0, std::string("<think>").size());
            }
            new_msg.content = std::move(text);
        } else {
            new_msg.content = generated_text;
        }
    }


    if (is_partial &&
            chat_parser_params.reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK &&
            string_ends_with(chat_parser_params.generation_prompt, "<think>") &&
            common_chat_find_control(generated_text, "</think>", chat_parser_params, output_source) == std::string::npos) {
        std::string reasoning = generated_text;
        if (common_chat_find_control(reasoning, "<think>", chat_parser_params, output_source) == 0) {
            reasoning.erase(0, std::string("<think>").size());
        }

        new_msg.role = "assistant";
        new_msg.reasoning_content = std::move(reasoning);
    }

    // When generation has finished (not partial) and reasoning was primed but
    // never closed -- model hit EOS or length cutoff while still inside
    // <think>...</think> -- the parser may succeed by stuffing everything into
    // reasoning_content with empty content. The web UI then renders an empty
    // message bubble alongside a (possibly very long) Reasoning toggle, which
    // looks like the model said nothing. Promote that text into content so the
    // user sees what the model actually produced.
    if (!is_partial &&
            chat_parser_params.reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK &&
            string_ends_with(chat_parser_params.generation_prompt, "<think>") &&
            common_chat_find_control(generated_text, "</think>", chat_parser_params, output_source) == std::string::npos &&
            new_msg.content.empty() &&
            !new_msg.reasoning_content.empty()) {
        new_msg.content = std::move(new_msg.reasoning_content);
        new_msg.reasoning_content.clear();
    }

    if (!new_msg.empty()) {
        // First-content prefix strip (DeepSeek V4 + similar templates).
        // Until the decision is made, suppress emission of the diff so we
        // don't send bytes we'd later need to retract -- that breaks
        // common_chat_msg_diff's invariant that previous content is a
        // prefix of new content.
        if (!prefix_strip_decided) {
            const std::string & c = new_msg.content;
            // Buffer until we have either:
            //   - enough bytes to reliably classify (>= 32)
            //   - a triple-backtick within the first 200 bytes (means we
            //     can decide the fence-prefix strip now)
            //   - the final non-partial chunk (must commit something)
            //   - non-empty reasoning_content / tool_calls (these pieces
            //     always pass through; we're only buffering content bytes)
            const bool has_aux = !new_msg.reasoning_content.empty() || !new_msg.tool_calls.empty();
            bool can_decide = !is_partial || has_aux || c.size() >= 32;
            size_t fence_at = std::string::npos;
            const size_t scan_max = std::min<size_t>(c.size(), 200);
            for (size_t i = 0; i + 2 < scan_max; ++i) {
                if (c[i] == '`' && c[i+1] == '`' && c[i+2] == '`') {
                    fence_at = i;
                    can_decide = true;
                    break;
                }
            }

            if (!can_decide) {
                // Hold output for now. Don't commit chat_msg either, so
                // msg_prv_copy stays the original empty chat_msg on the
                // next call and the diff invariant is preserved.
                diffs.clear();
                return chat_msg;
            }

            // Compute the strip count once.
            int strip = 0;
            if (fence_at != std::string::npos && fence_at > 0) {
                bool has_newline = false;
                for (size_t i = 0; i < fence_at; ++i) {
                    if (c[i] == '\n') { has_newline = true; break; }
                }
                if (!has_newline) {
                    // Inline prefix glued to the fence. Drop everything
                    // before the fence so it renders as a real code block.
                    strip = (int) fence_at;
                }
            }
            if (strip == 0) {
                // Single sentence-boundary punctuation char with whitespace
                // before alphanumeric/markup/UTF-8 content.
                size_t i = 0;
                while (i < c.size() && (c[i] == ' ' || c[i] == '\t')) ++i;
                if (i < c.size() && (c[i] == '.' || c[i] == '?' || c[i] == '!' ||
                                     c[i] == ',' || c[i] == ':' || c[i] == ';')) {
                    size_t j = i + 1;
                    while (j < c.size() && (c[j] == ' ' || c[j] == '\t' ||
                                            c[j] == '\n' || c[j] == '\r')) ++j;
                    if (j < c.size()) {
                        unsigned char ch = (unsigned char) c[j];
                        bool real_content =
                            std::isalnum(ch) ||
                            ch == '*' || ch == '_' || ch == '`' ||
                            ch == '#' || ch == '"' || ch == '\'' ||
                            ch == '(' || ch == '[' || ch == '{' ||
                            ch >= 0x80;
                        if (real_content) {
                            strip = (int) j;
                        }
                    }
                }
            }

            prefix_strip_decided = true;
            prefix_strip_bytes = strip;
        }

        if (prefix_strip_bytes > 0) {
            const size_t n = std::min<size_t>((size_t) prefix_strip_bytes, new_msg.content.size());
            new_msg.content.erase(0, n);
        }

        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//
// server_task
//

task_params server_task::params_from_json_cmpl(
        const llama_vocab * vocab,
        const common_params & params_base,
        const int n_ctx_slot,
        const std::vector<llama_logit_bias> & logit_bias_eog,
        const json & data) {
    task_params params;

    // Sampling parameter defaults are loaded from the global server context (but individual requests can still them)
    task_params defaults;
    defaults.sampling      = params_base.sampling;
    defaults.speculative   = params_base.speculative;
    defaults.n_keep        = params_base.n_keep;
    defaults.n_predict     = params_base.n_predict;
    defaults.n_cache_reuse = params_base.n_cache_reuse;
    defaults.cache_prompt  = params_base.cache_prompt;
    defaults.antiprompt    = params_base.antiprompt;

    // enabling this will output extra debug information in the HTTP responses from the server
    params.verbose           = params_base.verbosity > 9;
    params.timings_per_token = json_value(data, "timings_per_token", false);

    params.stream           = json_value(data,       "stream",             false);
    auto stream_opt         = json_value(data,       "stream_options",     json::object());
    params.include_usage    = json_value(stream_opt, "include_usage",      false);
    params.cache_prompt     = json_value(data,       "cache_prompt",       defaults.cache_prompt);
    params.return_tokens    = json_value(data,       "return_tokens",      false);
    params.return_progress  = json_value(data,       "return_progress",    false);
    auto max_tokens         = json_value(data,       "max_tokens",         defaults.n_predict);
    params.n_predict        = json_value(data,       "n_predict",          json_value(data, "max_completion_tokens", max_tokens));
    params.n_indent         = json_value(data,       "n_indent",           defaults.n_indent);
    params.n_keep           = json_value(data,       "n_keep",             defaults.n_keep);
    params.n_discard        = json_value(data,       "n_discard",          defaults.n_discard);
    params.n_discard        = std::max(0, params.n_discard);
    params.n_cmpl           = json_value(data,       "n_cmpl",             json_value(data, "n", 1));
    params.n_cache_reuse    = json_value(data,       "n_cache_reuse",      defaults.n_cache_reuse);
    //params.t_max_prompt_ms  = json_value(data,       "t_max_prompt_ms",    defaults.t_max_prompt_ms); // TODO: implement
    params.t_max_predict_ms = json_value(data,       "t_max_predict_ms",   defaults.t_max_predict_ms);
    params.response_fields  = json_value(data,       "response_fields",    std::vector<std::string>());

    params.sampling.top_k              = json_value(data, "top_k",               defaults.sampling.top_k);
    params.sampling.top_p              = json_value(data, "top_p",               defaults.sampling.top_p);
    params.sampling.min_p              = json_value(data, "min_p",               defaults.sampling.min_p);
    params.sampling.top_n_sigma        = json_value(data, "top_n_sigma",         defaults.sampling.top_n_sigma);
    params.sampling.xtc_probability    = json_value(data, "xtc_probability",     defaults.sampling.xtc_probability);
    params.sampling.xtc_threshold      = json_value(data, "xtc_threshold",       defaults.sampling.xtc_threshold);
    params.sampling.typ_p              = json_value(data, "typical_p",           defaults.sampling.typ_p);
    params.sampling.temp               = json_value(data, "temperature",         defaults.sampling.temp);
    params.sampling.dynatemp_range     = json_value(data, "dynatemp_range",      defaults.sampling.dynatemp_range);
    params.sampling.dynatemp_exponent  = json_value(data, "dynatemp_exponent",   defaults.sampling.dynatemp_exponent);
    params.sampling.penalty_last_n     = json_value(data, "repeat_last_n",       defaults.sampling.penalty_last_n);
    params.sampling.penalty_repeat     = json_value(data, "repeat_penalty",      defaults.sampling.penalty_repeat);
    params.sampling.penalty_freq       = json_value(data, "frequency_penalty",   defaults.sampling.penalty_freq);
    params.sampling.penalty_present    = json_value(data, "presence_penalty",    defaults.sampling.penalty_present);
    params.sampling.dry_multiplier     = json_value(data, "dry_multiplier",      defaults.sampling.dry_multiplier);
    params.sampling.dry_base           = json_value(data, "dry_base",            defaults.sampling.dry_base);
    params.sampling.dry_allowed_length = json_value(data, "dry_allowed_length",  defaults.sampling.dry_allowed_length);
    params.sampling.dry_penalty_last_n = json_value(data, "dry_penalty_last_n",  defaults.sampling.dry_penalty_last_n);
    params.sampling.mirostat           = json_value(data, "mirostat",            defaults.sampling.mirostat);
    params.sampling.mirostat_tau       = json_value(data, "mirostat_tau",        defaults.sampling.mirostat_tau);
    params.sampling.mirostat_eta       = json_value(data, "mirostat_eta",        defaults.sampling.mirostat_eta);
    params.sampling.adaptive_target    = json_value(data, "adaptive_target",     defaults.sampling.adaptive_target);
    params.sampling.adaptive_decay     = json_value(data, "adaptive_decay",      defaults.sampling.adaptive_decay);
    params.sampling.seed               = json_value(data, "seed",                defaults.sampling.seed);
    params.sampling.n_probs            = json_value(data, "n_probs",             defaults.sampling.n_probs);
    params.sampling.min_keep           = json_value(data, "min_keep",            defaults.sampling.min_keep);
    params.sampling.backend_sampling   = json_value(data, "backend_sampling",    defaults.sampling.backend_sampling);
    params.post_sampling_probs         = json_value(data, "post_sampling_probs", defaults.post_sampling_probs);

    params.speculative = defaults.speculative;

    // TODO: for now, be able to adjust only the draft-model based speculative parameters
    params.speculative.draft.n_min = json_value(data, "speculative.n_min", defaults.speculative.draft.n_min);
    params.speculative.draft.n_max = json_value(data, "speculative.n_max", defaults.speculative.draft.n_max);
    params.speculative.draft.p_min = json_value(data, "speculative.p_min", defaults.speculative.draft.p_min);

    params.speculative.draft.n_min = std::min(params.speculative.draft.n_max, params.speculative.draft.n_min);
    params.speculative.draft.n_min = std::max(params.speculative.draft.n_min, 0);
    params.speculative.draft.n_max = std::max(params.speculative.draft.n_max, 0);

#if 0
    // for debugging and research purposes
    params.speculative.type = common_speculative_type_from_name(json_value(data, "speculative.type", common_speculative_type_to_str(defaults.speculative.type)));

    params.speculative.ngram_size_n     = json_value(data, "speculative.ngram_size_n", defaults.speculative.ngram_size_n);
    params.speculative.ngram_size_m     = json_value(data, "speculative.ngram_size_m", defaults.speculative.ngram_size_m);
    params.speculative.ngram_min_hits   = json_value(data, "speculative.ngram_m_hits", defaults.speculative.ngram_min_hits);

    params.speculative.ngram_size_n     = std::max(std::min(1, (int) params.speculative.ngram_size_n),     1024);
    params.speculative.ngram_size_m     = std::max(std::min(1, (int) params.speculative.ngram_size_m),     1024);
    params.speculative.ngram_min_hits   = std::max(std::min(1, (int) params.speculative.ngram_min_hits),   1024);
#endif

    // Use OpenAI API logprobs only if n_probs wasn't provided
    if (data.contains("logprobs") && params.sampling.n_probs == defaults.sampling.n_probs){
        params.sampling.n_probs = json_value(data, "logprobs", defaults.sampling.n_probs);
    }

    if (data.contains("lora")) {
        if (data.at("lora").is_array()) {
            params.lora = parse_lora_request(data.at("lora"));
        } else {
            throw std::runtime_error("Error: 'lora' must be an array of objects with 'id' and 'scale' fields");
        }
    } else {
        params.lora = {};
    }

    // TODO: add more sanity checks for the input parameters

    if (params.sampling.penalty_last_n < -1) {
        throw std::runtime_error("Error: repeat_last_n must be >= -1");
    }

    if (params.sampling.dry_penalty_last_n < -1) {
        throw std::runtime_error("Error: dry_penalty_last_n must be >= -1");
    }

    if (params.sampling.penalty_last_n == -1) {
        // note: should be the slot's context and not the full context, but it's ok
        params.sampling.penalty_last_n = n_ctx_slot;
    }

    if (params.sampling.dry_penalty_last_n == -1) {
        params.sampling.dry_penalty_last_n = n_ctx_slot;
    }

    if (params.sampling.dry_base < 1.0f) {
        params.sampling.dry_base = defaults.sampling.dry_base;
    }

    // sequence breakers for DRY
    {
        // Currently, this is not compatible with TextGen WebUI, Koboldcpp and SillyTavern format
        // Ref: https://github.com/oobabooga/text-generation-webui/blob/d1af7a41ade7bd3c3a463bfa640725edb818ebaf/extensions/openai/typing.py#L39

        if (data.contains("dry_sequence_breakers")) {
            params.sampling.dry_sequence_breakers = json_value(data, "dry_sequence_breakers", std::vector<std::string>());
            if (params.sampling.dry_sequence_breakers.empty()) {
                throw std::runtime_error("Error: dry_sequence_breakers must be a non-empty array of strings");
            }
        }
    }

    // process "json_schema" and "grammar"
    if (data.contains("json_schema") && !data.contains("grammar")) {
        try {
            auto schema                  = json_value(data, "json_schema", json::object());
            SRV_DBG("JSON schema: %s\n", schema.dump(2).c_str());
            std::string grammar_str      = json_schema_to_grammar(schema);
            SRV_DBG("Converted grammar: %s\n", grammar_str.c_str());
            params.sampling.grammar      = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, std::move(grammar_str)};
        } catch (const std::exception & e) {
            throw std::runtime_error(std::string("\"json_schema\": ") + e.what());
        }
    } else {
        params.sampling.grammar = defaults.sampling.grammar;

        std::string grammar_str = json_value(data, "grammar", std::string());
        if (!grammar_str.empty()) {
            // grammar_type key is set by the server when converting chat template grammars
            std::string grammar_type = json_value(data, "grammar_type", std::string());
            if (grammar_type == "tool_calls") {
                params.sampling.grammar = {COMMON_GRAMMAR_TYPE_TOOL_CALLS, std::move(grammar_str)};
            } else {
                // explicit grammar from the user (API field "grammar")
                params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, std::move(grammar_str)};
            }
            SRV_DBG("Grammar (%s): %s\n", grammar_type.c_str(), common_grammar_value(params.sampling.grammar).c_str());
        }
        params.sampling.grammar_lazy = json_value(data, "grammar_lazy", defaults.sampling.grammar_lazy);
        SRV_DBG("Grammar lazy: %s\n", params.sampling.grammar_lazy ? "true" : "false");
    }

    {
        auto it = data.find("chat_format");
        if (it != data.end()) {
            params.chat_parser_params.format = static_cast<common_chat_format>(it->get<int>());
            SRV_INF("Chat format: %s\n", common_chat_format_name(params.chat_parser_params.format));
        } else {
            params.chat_parser_params.format = defaults.chat_parser_params.format;
        }
        common_reasoning_format reasoning_format = params_base.reasoning_format;
        if (data.contains("reasoning_format")) {
            reasoning_format = common_reasoning_format_from_name(data.at("reasoning_format").get<std::string>());
        }
        params.chat_parser_params.reasoning_format = reasoning_format;
        params.chat_parser_params.reasoning_in_content = params.stream && (reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY);
        params.chat_parser_params.set_vocab(vocab);
        params.chat_parser_params.generation_prompt = json_value(data, "generation_prompt", std::string());
        params.sampling.generation_prompt = params.chat_parser_params.generation_prompt;
        SRV_DBG("Generation prompt: '%s'\n", params.chat_parser_params.generation_prompt.c_str());
        params.chat_parser_params.parse_tool_calls = json_value(data, "parse_tool_calls", false);
        params.tool_call_required = params.chat_parser_params.parse_tool_calls && json_value(data, "tool_call_required", false);
        if (data.contains("chat_parser")) {
            params.chat_parser_params.parser.load(data.at("chat_parser").get<std::string>());
        }
    }

    {
        const auto preserved_tokens = data.find("preserved_tokens");
        if (preserved_tokens != data.end()) {
            for (const auto & t : *preserved_tokens) {
                auto ids = common_tokenize(vocab, t.get<std::string>(), /* add_special= */ false, /* parse_special= */ true);
                if (ids.size() == 1) {
                    SRV_DBG("Preserved token: %d\n", ids[0]);
                    params.sampling.preserved_tokens.insert(ids[0]);
                } else {
                    // This may happen when using a tool call style meant for a model with special tokens to preserve on a model without said tokens.
                    SRV_DBG("Not preserved because more than 1 token: %s\n", t.get<std::string>().c_str());
                }
            }
        }
        const auto grammar_triggers = data.find("grammar_triggers");
        if (grammar_triggers != data.end()) {
            for (const auto & t : *grammar_triggers) {
                server_grammar_trigger ct(t);
                if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                    const auto & word = ct.value.value;
                    auto ids = common_tokenize(vocab, word, /* add_special= */ false, /* parse_special= */ true);
                    if (ids.size() == 1) {
                        auto token = ids[0];
                        if (std::find(params.sampling.preserved_tokens.begin(), params.sampling.preserved_tokens.end(), (llama_token) token) == params.sampling.preserved_tokens.end()) {
                            throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + word);
                        }
                        SRV_DBG("Grammar trigger token: %d (`%s`)\n", token, word.c_str());
                        common_grammar_trigger trigger;
                        trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                        trigger.value = word;
                        trigger.token = token;
                        params.sampling.grammar_triggers.push_back(std::move(trigger));
                    } else {
                        SRV_DBG("Grammar trigger word: `%s`\n", word.c_str());
                        params.sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
                    }
                } else {
                    if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN) {
                        SRV_DBG("Grammar trigger pattern: `%s`\n", ct.value.value.c_str());
                    } else if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL) {
                        SRV_DBG("Grammar trigger pattern full: `%s`\n", ct.value.value.c_str());
                    } else {
                        throw std::runtime_error("Unknown grammar trigger type");
                    }
                    params.sampling.grammar_triggers.emplace_back(std::move(ct.value));
                }
            }
        }
        if (params.sampling.grammar_lazy && params.sampling.grammar_triggers.empty()) {
            throw std::runtime_error("Error: no triggers set for lazy grammar!");
        }
    }

    // Parse reasoning budget sampler parameters
    {
        const int32_t budget = json_value(data, "reasoning_budget_tokens", (int32_t) -1);
        const auto start_tag = json_value(data, "reasoning_budget_start_tag", std::string());
        const auto end_tag   = json_value(data, "reasoning_budget_end_tag", std::string());
        const auto message   = json_value(data, "reasoning_budget_message", std::string());
        const auto suffix    = json_value(data, "reasoning_budget_suffix", std::string());
        params.sampling.reasoning_budget_tokens = budget;

        if (!start_tag.empty()) {
            params.sampling.reasoning_budget_start = common_tokenize(vocab, start_tag, false, true);
        }
        if (!end_tag.empty()) {
            params.sampling.reasoning_budget_end = common_tokenize(vocab, end_tag, false, true);
            params.sampling.reasoning_budget_forced = common_tokenize(vocab, message + end_tag + suffix, false, true);

            SRV_DBG("reasoning budget: tokens=%d, generation_prompt='%s', start=%zu toks, end=%zu toks, forced=%zu toks\n",
                budget, params.sampling.generation_prompt.c_str(),
                params.sampling.reasoning_budget_start.size(),
                params.sampling.reasoning_budget_end.size(),
                params.sampling.reasoning_budget_forced.size());
        }
    }

    {
        params.sampling.logit_bias.clear();

        const auto & logit_bias = data.find("logit_bias");
        if (logit_bias != data.end() && logit_bias->is_array()) {
            const int n_vocab = llama_vocab_n_tokens(vocab);
            for (const auto & el : *logit_bias) {
                // TODO: we may want to throw errors here, in case "el" is incorrect
                if (el.is_array() && el.size() == 2) {
                    float bias;
                    if (el[1].is_number()) {
                        bias = el[1].get<float>();
                    } else if (el[1].is_boolean() && !el[1].get<bool>()) {
                        bias = -INFINITY;
                    } else {
                        continue;
                    }

                    if (el[0].is_number_integer()) {
                        llama_token tok = el[0].get<llama_token>();
                        if (tok >= 0 && tok < n_vocab) {
                            params.sampling.logit_bias.push_back({tok, bias});
                        }
                    } else if (el[0].is_string()) {
                        auto toks = common_tokenize(vocab, el[0].get<std::string>(), false);
                        for (auto tok : toks) {
                            params.sampling.logit_bias.push_back({tok, bias});
                        }
                    }
                }
            }
        } else if (logit_bias != data.end() && logit_bias->is_object()) {
            const int n_vocab = llama_vocab_n_tokens(vocab);
            for (const auto & el : logit_bias->items()) {
                float bias;
                const auto & key = el.key();
                const auto & value = el.value();
                if (value.is_number()) {
                    bias = value.get<float>();
                } else if (value.is_boolean() && !value.get<bool>()) {
                    bias = -INFINITY;
                } else {
                    continue;
                }

                char *end;
                llama_token tok = strtol(key.c_str(), &end, 10);
                if (*end == 0) {
                    if (tok >= 0 && tok < n_vocab) {
                        params.sampling.logit_bias.push_back({tok, bias});
                    }
                } else {
                    auto toks = common_tokenize(vocab, key, false);
                    for (auto tok : toks) {
                        params.sampling.logit_bias.push_back({tok, bias});
                    }
                }
            }
        }

        params.sampling.ignore_eos = json_value(data, "ignore_eos", params_base.sampling.ignore_eos);
        if (params.sampling.ignore_eos) {
            params.sampling.logit_bias.insert(
                    params.sampling.logit_bias.end(),
                    logit_bias_eog.begin(), logit_bias_eog.end());
        }
    }

    {
        params.antiprompt.clear();

        const auto & stop = data.find("stop");
        if (stop != data.end() && stop->is_array()) {
            for (const auto & word : *stop) {
                if (!word.empty()) {
                    params.antiprompt.push_back(word);
                }
            }
        }
        // set reverse prompt from cli args if not set in the request
        if (params.antiprompt.empty()) {
            params.antiprompt = defaults.antiprompt;
        }
    }

    {
        const auto samplers = data.find("samplers");
        if (samplers != data.end()) {
            if (samplers->is_array()) {
                params.sampling.samplers = common_sampler_types_from_names(*samplers, false);
            } else if (samplers->is_string()){
                params.sampling.samplers = common_sampler_types_from_chars(samplers->get<std::string>());
            }
        } else {
            params.sampling.samplers = defaults.sampling.samplers;
        }
    }

    if (params.n_cmpl > params_base.n_parallel) {
        throw std::runtime_error("n_cmpl cannot be greater than the number of slots, please increase -np");
    }

    return params;
}

//
// result_timings
//

json result_timings::to_json() const {
    json base = {
        {"cache_n",                cache_n},

        {"prompt_n",               prompt_n},
        {"prompt_ms",              prompt_ms},
        {"prompt_per_token_ms",    prompt_per_token_ms},
        {"prompt_per_second",      prompt_per_second},

        {"predicted_n",            predicted_n},
        {"predicted_ms",           predicted_ms},
        {"predicted_per_token_ms", predicted_per_token_ms},
        {"predicted_per_second",   predicted_per_second},
    };

    if (draft_n > 0) {
        base["draft_n"] = draft_n;
        base["draft_n_accepted"] = draft_n_accepted;
    }

    return base;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // nlohmann::json converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             timings.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (timings.prompt_n >= 0) {
        deltas.back().push_back({"timings", timings.to_json()});
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

// A namespace tool is offered to the model as "namespace::name"; a Responses item names them apart.
static json oai_resp_function_call(json item, const std::string & name) {
    const size_t sep = name.find("::");
    if (sep == std::string::npos) {
        item["name"] = name;
    } else {
        item["namespace"] = name.substr(0, sep);
        item["name"]      = name.substr(sep + 2);
    }
    return item;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(oai_resp_function_call(json {
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "fc_" + tool_call.id},
        }, tool_call.name));
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    if (oaicompat_msg.reasoning_content != "") {
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const json output_item = oai_resp_function_call(json {
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "fc_" + tool_call.id},
        }, tool_call.name);
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    state.update_chat_msg(content, true, oaicompat_msg_diffs, false, &output_source);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (timings.prompt_n > 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (timings.prompt_n >= 0) {
            last_json.push_back({"timings", timings.to_json()});
        }
        if (is_progress) {
            last_json.push_back({"prompt_progress", progress.to_json()});
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (n_decoded == 1) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"item", oai_resp_function_call(json {
                        {"arguments", ""},
                        {"call_id",   "fc_" + diff.tool_call_delta.id},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }, diff.tool_call_delta.name)},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }
    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_metrics::to_json() {
    return json {
        { "idle",                            n_idle_slots },
        { "processing",                      n_processing_slots },
        { "deferred",                        n_tasks_deferred },
        { "t_start",                         t_start },

        { "n_prompt_tokens_processed_total", n_prompt_tokens_processed_total },
        { "t_tokens_generation_total",       t_tokens_generation_total },
        { "n_tokens_predicted_total",        n_tokens_predicted_total },
        { "t_prompt_processing_total",       t_prompt_processing_total },

        { "n_tokens_max",                    n_tokens_max },

        { "n_prompt_tokens_processed",       n_prompt_tokens_processed },
        { "t_prompt_processing",             t_prompt_processing },
        { "n_tokens_predicted",              n_tokens_predicted },
        { "t_tokens_generation",             t_tokens_generation },

        { "n_decode_total",                  n_decode_total },
        { "n_busy_slots_total",              n_busy_slots_total },

        { "slots",                           slots_data },
    };
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

//
// server_prompt_cache
//
static bool server_prompt_can_restore_prefix(
        const server_prompt        & prompt,
        int64_t                      n_tokens,
        common_context_seq_rm_type   seq_rm_type) {
    if (n_tokens < 0 || n_tokens > prompt.n_tokens()) {
        return false;
    }

    if (seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
        return true;
    }

    if (n_tokens == prompt.n_tokens()) {
        return true;
    }

    for (const auto & checkpoint : prompt.checkpoints) {
        if (!checkpoint.empty() && checkpoint.n_tokens == n_tokens) {
            return true;
        }
    }

    return false;
}

int64_t server_prompt_reuse_full(
        const server_prompt              & prompt,
        const server_tokens              & tokens_new,
        const server_prompt_checkpoint ** checkpoint) {
    *checkpoint = nullptr;

    const int64_t lcp   = prompt.tokens.get_common_prefix(tokens_new);
    const int64_t limit = std::min<int64_t>(lcp, (int64_t) tokens_new.size() - 1);

    if (prompt.n_tokens() <= limit) {
        return prompt.n_tokens();
    }

    int64_t best = -1;
    for (const auto & cp : prompt.checkpoints) {
        if (!cp.empty() && cp.n_tokens <= limit && cp.n_tokens > best) {
            best        = cp.n_tokens;
            *checkpoint = &cp;
        }
    }

    return best;
}

size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.n_tokens();
    }

    return res;
}

//
// on-disk layout of a cached prompt: header, checkpoint states, state
//

static constexpr uint32_t SERVER_PROMPT_FILE_MAGIC   = 0x4643504c; // "LPCF"
static constexpr uint32_t SERVER_PROMPT_FILE_VERSION = 1;

static bool server_prompt_fwrite(FILE * f, const void * src, size_t size) {
    return size == 0 || std::fwrite(src, 1, size, f) == size;
}

static bool server_prompt_fread(FILE * f, void * dst, size_t size) {
    return size == 0 || std::fread(dst, 1, size, f) == size;
}

// drop the file's pages from the page cache: a state is read once per restore, and its pages would otherwise take
// memory from the server's cgroup, where the expert caches need it
static void server_prompt_file_drop_cache(FILE * f) {
#if defined(__linux__)
    posix_fadvise(fileno(f), 0, 0, POSIX_FADV_DONTNEED);
#else
    (void) f;
#endif
}

static void server_prompt_file_touch(const std::string & path) {
    std::error_code ec;
    std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), ec);
}

// read the header of a cached prompt file into `prompt` (tokens, checkpoint and state locations); false if the
// file is not a prompt of this fingerprint
static bool server_prompt_read_header(FILE * f, const std::string & fingerprint, server_prompt & prompt) {
    uint32_t magic = 0, version = 0, n_fp = 0;
    if (!server_prompt_fread(f, &magic, sizeof(magic)) || magic != SERVER_PROMPT_FILE_MAGIC ||
        !server_prompt_fread(f, &version, sizeof(version)) || version != SERVER_PROMPT_FILE_VERSION ||
        !server_prompt_fread(f, &n_fp, sizeof(n_fp)) || n_fp > 4096) {
        return false;
    }
    std::string fp(n_fp, '\0');
    if (!server_prompt_fread(f, fp.data(), n_fp) || fp != fingerprint) {
        return false;
    }

    uint32_t n_tokens = 0;
    if (!server_prompt_fread(f, &n_tokens, sizeof(n_tokens)) || n_tokens > (1u << 26)) {
        return false;
    }
    llama_tokens tokens(n_tokens);
    if (!server_prompt_fread(f, tokens.data(), n_tokens*sizeof(llama_token))) {
        return false;
    }

    uint32_t n_checkpoints = 0;
    if (!server_prompt_fread(f, &n_checkpoints, sizeof(n_checkpoints)) || n_checkpoints > 4096) {
        return false;
    }
    std::list<server_prompt_checkpoint> checkpoints;
    for (uint32_t i = 0; i < n_checkpoints; ++i) {
        server_prompt_checkpoint cp {};
        uint64_t size = 0;
        if (!server_prompt_fread(f, &cp.pos_min, sizeof(cp.pos_min)) ||
            !server_prompt_fread(f, &cp.pos_max, sizeof(cp.pos_max)) ||
            !server_prompt_fread(f, &cp.n_tokens, sizeof(cp.n_tokens)) ||
            !server_prompt_fread(f, &size, sizeof(size))) {
            return false;
        }
        cp.file_size = size;
        checkpoints.push_back(std::move(cp));
    }

    uint64_t data_size = 0;
    if (!server_prompt_fread(f, &data_size, sizeof(data_size))) {
        return false;
    }

    size_t offset = std::ftell(f);
    for (auto & cp : checkpoints) {
        cp.file_offset = offset;
        offset += cp.file_size;
    }

    prompt.tokens           = server_tokens(tokens, false);
    prompt.checkpoints      = std::move(checkpoints);
    prompt.file_data_offset = offset;
    prompt.file_data_size   = data_size;

    return true;
}

static bool server_prompt_read_blob(const std::string & path, size_t offset, size_t size, std::vector<uint8_t> & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    out.resize(size);
    const bool ok = std::fseek(f, (long) offset, SEEK_SET) == 0 && server_prompt_fread(f, out.data(), size);
    server_prompt_file_drop_cache(f);
    std::fclose(f);
    return ok;
}

// the checkpoint's partial state, from memory or from the prompt's file
static bool server_prompt_checkpoint_data(const server_prompt & prompt, const server_prompt_checkpoint & cp, std::vector<uint8_t> & out) {
    if (!cp.data.empty()) {
        out = cp.data;
        return true;
    }
    return server_prompt_read_blob(prompt.file, cp.file_offset, cp.file_size, out);
}

// Restore a cached prompt into the slot's sequence: its whole state, then, to start at one of its checkpoints, the
// checkpoint's partial state over it. A partial state holds only what the memory cannot truncate (a sliding window,
// a recurrent or compressed state) and relies on the rest being this prompt's, not whatever the slot held before.
static bool server_prompt_restore(
        const server_prompt            & prompt,
        const server_prompt_checkpoint * checkpoint,
        llama_context                  * ctx,
        int32_t                          id_slot) {
    if (prompt.file.empty()) {
        if (llama_state_seq_set_data_ext(ctx, prompt.data.data(), prompt.data.size(), id_slot, 0) != prompt.data.size()) {
            return false;
        }
    } else {
        FILE * f = std::fopen(prompt.file.c_str(), "rb");
        if (!f) {
            return false;
        }
        bool ok = std::fseek(f, (long) prompt.file_data_offset, SEEK_SET) == 0 &&
            llama_state_seq_fread_ext(ctx, f, id_slot, 0) == prompt.file_data_size;
        server_prompt_file_drop_cache(f);
        std::fclose(f);
        if (!ok) {
            return false;
        }
    }

    if (checkpoint != nullptr) {
        // straight from the prompt's file when it is there, without a copy in memory
        if (!checkpoint->data.empty()) {
            return llama_state_seq_set_data_ext(ctx, checkpoint->data.data(), checkpoint->data.size(), id_slot,
                    LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == checkpoint->data.size();
        }
        FILE * f = std::fopen(prompt.file.c_str(), "rb");
        if (!f) {
            return false;
        }
        const bool ok = std::fseek(f, (long) checkpoint->file_offset, SEEK_SET) == 0 &&
            llama_state_seq_fread_ext(ctx, f, id_slot, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == checkpoint->file_size;
        server_prompt_file_drop_cache(f);
        std::fclose(f);
        return ok;
    }

    return true;
}

void server_prompt_cache::use_dir(const std::string & dir, const std::string & fingerprint) {
    this->dir         = dir;
    this->fingerprint = fingerprint;
    limit_tokens      = 0;

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        throw std::runtime_error("cannot create the prompt cache directory " + dir + ": " + ec.message());
    }

    std::vector<std::pair<std::filesystem::file_time_type, server_prompt>> found;
    for (const auto & entry : std::filesystem::directory_iterator(dir, ec)) {
        const std::string path = entry.path().string();
        if (entry.path().extension() == ".tmp") {
            std::filesystem::remove(entry.path(), ec);
            continue;
        }
        if (entry.path().extension() != ".prompt") {
            continue;
        }
        server_prompt prompt;
        bool ok = false;
        if (FILE * f = std::fopen(path.c_str(), "rb")) {
            ok = server_prompt_read_header(f, fingerprint, prompt);
            std::fclose(f);
        }
        if (!ok) {
            SRV_WRN(" - removing cached prompt %s: other model, build or format\n", path.c_str());
            std::filesystem::remove(entry.path(), ec);
            continue;
        }
        prompt.file = path;
        found.emplace_back(std::filesystem::last_write_time(entry.path(), ec), std::move(prompt));
    }
    std::sort(found.begin(), found.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    for (auto & item : found) {
        states.push_back(std::move(item.second));
    }

    SRV_WRN("prompt cache on disk in %s: %zu prompts found\n", dir.c_str(), states.size());
    update();
}

bool server_prompt_cache::contains(const server_prompt & prompt, common_context_seq_rm_type seq_rm_type) {
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp = it->tokens.get_common_prefix(prompt.tokens);
        if (lcp == (int) prompt.tokens.size() && server_prompt_can_restore_prefix(*it, prompt.n_tokens(), seq_rm_type)) {
            states.splice(states.end(), states, it);
            if (!states.back().file.empty()) {
                server_prompt_file_touch(states.back().file);
            }
            return true;
        }
    }
    return false;
}

void server_prompt_cache::remove(std::list<server_prompt>::iterator it) {
    if (!it->file.empty()) {
        std::error_code ec;
        std::filesystem::remove(it->file, ec);
    }
    states.erase(it);
}

void server_prompt_cache::save(
        const server_prompt        & prompt,
        bool                         current,
        llama_context              * ctx,
        int32_t                      id_slot,
        common_context_seq_rm_type   seq_rm_type) {
    const int32_t n_state = current ? -1 : prompt.n_tokens();

    if (contains(prompt, seq_rm_type)) {
        SRV_WRN("%s", " - prompt is already in the cache, skipping\n");
        return;
    }

    const size_t state_size = llama_state_seq_get_size_at(ctx, id_slot, 0, n_state);
    if (state_size == 0) {
        SRV_WRN(" - cannot save the state after %d tokens\n", prompt.n_tokens());
        return;
    }

    SRV_WRN(" - saving prompt with length %d, total state size = %.3f MiB\n", prompt.n_tokens(), state_size / (1024.0 * 1024.0));

    if (!on_disk()) {
        // remove any cached prompts that are fully contained in the current prompt
        for (auto it = states.begin(); it != states.end();) {
            const int len = it->tokens.get_common_prefix(prompt.tokens);

            if (len == (int) it->tokens.size() &&
                    server_prompt_can_restore_prefix(prompt, it->n_tokens(), seq_rm_type)) {
                SRV_WRN(" - removing obsolete cached prompt with length %d\n", len);

                it = states.erase(it);
            } else {
                ++it;
            }
        }

        std::vector<uint8_t> state_data;

        // check if we can allocate enough memory for the new state
        try {
            state_data.resize(state_size);
        } catch (const std::bad_alloc & e) {
            SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

            limit_size = std::max<size_t>(1, 0.4*size());

            SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

            update();

            return;
        }

        if (llama_state_seq_get_data_at(ctx, state_data.data(), state_size, id_slot, 0, n_state) != state_size) {
            SRV_WRN("%s", " - failed to save the state\n");
            return;
        }

        server_prompt entry;
        entry.tokens      = prompt.tokens.clone();
        entry.data        = std::move(state_data);
        entry.checkpoints = prompt.checkpoints;
        states.push_back(std::move(entry));

        return;
    }

    const llama_tokens & tokens = prompt.tokens.get_tokens();
    if (std::find(tokens.begin(), tokens.end(), LLAMA_TOKEN_NULL) != tokens.end()) {
        SRV_WRN("%s", " - prompts with media are not cached on disk\n");
        return;
    }

    uint64_t hash = 1469598103934665603ull; // FNV-1a over the tokens names the file
    for (llama_token t : tokens) {
        for (int b = 0; b < 4; ++b) {
            hash = (hash ^ ((uint32_t) t >> (8*b) & 0xff))*1099511628211ull;
        }
    }
    char name[32];
    std::snprintf(name, sizeof(name), "%016" PRIx64 ".prompt", hash);
    const std::string path = (std::filesystem::path(dir) / name).string();
    const std::string tmp  = path + ".tmp";

    server_prompt entry;
    entry.tokens = prompt.tokens.clone();
    entry.file   = path;

    bool ok = false;
    if (FILE * f = std::fopen(tmp.c_str(), "wb")) {
        const uint32_t magic = SERVER_PROMPT_FILE_MAGIC, version = SERVER_PROMPT_FILE_VERSION;
        const uint32_t n_fp = fingerprint.size(), n_tokens = tokens.size(), n_checkpoints = prompt.checkpoints.size();
        const uint64_t data_size = state_size;
        ok = server_prompt_fwrite(f, &magic, sizeof(magic)) && server_prompt_fwrite(f, &version, sizeof(version)) &&
             server_prompt_fwrite(f, &n_fp, sizeof(n_fp)) && server_prompt_fwrite(f, fingerprint.data(), n_fp) &&
             server_prompt_fwrite(f, &n_tokens, sizeof(n_tokens)) &&
             server_prompt_fwrite(f, tokens.data(), n_tokens*sizeof(llama_token)) &&
             server_prompt_fwrite(f, &n_checkpoints, sizeof(n_checkpoints));
        for (const auto & cp : prompt.checkpoints) {
            const uint64_t size = cp.size();
            ok = ok && server_prompt_fwrite(f, &cp.pos_min, sizeof(cp.pos_min)) &&
                 server_prompt_fwrite(f, &cp.pos_max, sizeof(cp.pos_max)) &&
                 server_prompt_fwrite(f, &cp.n_tokens, sizeof(cp.n_tokens)) &&
                 server_prompt_fwrite(f, &size, sizeof(size));
        }
        ok = ok && server_prompt_fwrite(f, &data_size, sizeof(data_size));

        size_t offset = ok ? std::ftell(f) : 0;
        for (const auto & cp : prompt.checkpoints) {
            // a checkpoint in memory is written as it is; one in another prompt's file is read back first
            std::vector<uint8_t> copy;
            const bool in_memory = !cp.data.empty();
            ok = ok && (in_memory || (llama_host_reserve(ctx, cp.size()) && server_prompt_checkpoint_data(prompt, cp, copy)));
            const std::vector<uint8_t> & data = in_memory ? cp.data : copy;
            ok = ok && server_prompt_fwrite(f, data.data(), data.size());
            // A copied-and-cleared vector retains its payload capacity. Disk
            // cache entries keep only metadata, not a second host checkpoint.
            server_prompt_checkpoint meta = {};
            meta.pos_min  = cp.pos_min;
            meta.pos_max  = cp.pos_max;
            meta.n_tokens = cp.n_tokens;
            meta.file_offset = offset;
            meta.file_size   = data.size();
            offset += data.size();
            entry.checkpoints.push_back(std::move(meta));
        }
        entry.file_data_offset = offset;
        entry.file_data_size   = state_size;

        ok = ok && llama_state_seq_fwrite_at(ctx, f, id_slot, 0, n_state) == state_size;
        ok = ok && std::fflush(f) == 0;
#if defined(__linux__)
        ok = ok && fdatasync(fileno(f)) == 0;
#endif
        server_prompt_file_drop_cache(f);
        ok = std::fclose(f) == 0 && ok;
    }
    std::error_code ec;
    if (ok) {
        std::filesystem::rename(tmp, path, ec);
        ok = !ec;
    }
    if (!ok) {
        SRV_WRN(" - failed to write cached prompt %s\n", path.c_str());
        std::filesystem::remove(tmp, ec);
        return;
    }

    states.push_back(std::move(entry));
}

bool server_prompt_cache::load(
        server_prompt              & prompt,
        const server_tokens        & tokens_new,
        int64_t                      n_reuse_slot,
        llama_context              * ctx,
        int32_t                      id_slot,
        common_context_seq_rm_type   seq_rm_type) {
    auto it_best = states.end();
    const server_prompt_checkpoint * checkpoint_best = nullptr;
    int64_t n_tokens_best = -1;

    if (seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
        // A state that cannot be rolled back is reused up to its end or one of its checkpoints. Take the cached
        // prompt that lets the slot skip the most of the new prompt, if that is more than its own state does.
        SRV_WRN(" - looking for a better prompt, the slot reuses %" PRId64 " of %zu tokens\n", n_reuse_slot, tokens_new.size());

        n_tokens_best = n_reuse_slot;
        for (auto it = states.begin(); it != states.end(); ++it) {
            const server_prompt_checkpoint * checkpoint_cur = nullptr;
            const int64_t n_tokens_cur = server_prompt_reuse_full(*it, tokens_new, &checkpoint_cur);
            if (n_tokens_cur > n_tokens_best) {
                it_best         = it;
                checkpoint_best = checkpoint_cur;
                n_tokens_best   = n_tokens_cur;
            }
        }
    } else {
        const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

        float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
        float sim_best    = float(lcp_best) / tokens_new.size();

        SRV_WRN(" - looking for better prompt, base f_keep = %.3f, sim = %.3f\n", f_keep_best, sim_best);

        // find the most similar cached prompt, that would also preserve the most context
        for (auto it = states.begin(); it != states.end(); ++it) {
            const int lcp_cur = it->tokens.get_common_prefix(tokens_new);

            const float f_keep_cur = float(lcp_cur) / it->tokens.size();
            const float sim_cur    = float(lcp_cur) / tokens_new.size();

            // don't trash large prompts
            if (f_keep_cur < 0.25f) {
                continue;
            }

            if (f_keep_best < f_keep_cur && sim_best < sim_cur) {
                f_keep_best = f_keep_cur;
                sim_best    = sim_cur;

                it_best       = it;
                n_tokens_best = lcp_cur;
            }
        }
    }

    if (it_best == states.end()) {
        return true;
    }

    SRV_WRN(" - found a better prompt%s: reuses %" PRId64 " tokens\n", checkpoint_best ? " checkpoint" : "", n_tokens_best);

    if (!server_prompt_restore(*it_best, checkpoint_best, ctx, id_slot)) {
        SRV_WRN("%s", "failed to restore the cached prompt\n");
        remove(it_best);
        return false;
    }

    // the slot takes over the cached prompt, cut back to the checkpoint it starts from; its checkpoints move to memory
    server_prompt loaded;
    loaded.tokens = it_best->tokens.clone();
    for (const auto & cp : it_best->checkpoints) {
        if (checkpoint_best != nullptr && cp.n_tokens > n_tokens_best) {
            continue;
        }
        // into memory as far as the memory limit allows (the RAM expert cache gives back room first)
        server_prompt_checkpoint cur = cp;
        if ((cp.data.empty() && !llama_host_reserve(ctx, cp.size())) || !server_prompt_checkpoint_data(*it_best, cp, cur.data)) {
            continue;
        }
        cur.file_offset = 0;
        cur.file_size   = 0;
        loaded.checkpoints.push_back(std::move(cur));
    }
    if (checkpoint_best != nullptr) {
        loaded.tokens.keep_first(n_tokens_best);
    }
    prompt = std::move(loaded);

    if (on_disk()) {
        // the file stays: other conversations may start from it too
        states.splice(states.end(), states, it_best);
        server_prompt_file_touch(states.back().file);
    } else {
        states.erase(it_best);
    }

    return true;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        // always keep at least one state, regardless of the limits
        while (states.size() > 1 && size() > limit_size) {
            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            remove(states.begin());
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (states.size() > 1 && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            remove(states.begin());
        }
    }

    SRV_WRN(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)%s\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur,
            on_disk() ? " on disk" : "");

    for (const auto & state : states) {
        SRV_WRN("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.n_tokens(), state.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}
