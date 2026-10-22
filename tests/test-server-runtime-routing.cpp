#include "server-common.h"
#include "server-expert-guard.h"
#include "chat.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}
template<class F> static void rejects(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::exception &) { rejected = true; }
    require(rejected, "invalid generation prefix was accepted");
}

static void test_template_kwargs_generation_prefix() {
    const std::string source =
        "{%- for m in messages -%}{{- m.role + ':' + m.content + '\\n' -}}{%- endfor -%}"
        "{%- if add_generation_prompt -%}assistant:"
        "{%- if thinking is undefined or thinking -%}<think>{%- else -%}<response>{%- endif -%}"
        "{%- endif -%}";
    const auto templates = common_chat_templates_init(nullptr, source);
    common_chat_templates_inputs inputs;
    common_chat_msg message;
    message.role = "user";
    message.content = "fixture";
    inputs.messages = {message};
    inputs.use_jinja = true;
    inputs.add_generation_prompt = true;
    inputs.force_pure_content = true; // Exercise the shared prefix diff without a model/parser heuristic.
    for (const bool thinking : {true, false}) {
        inputs.chat_template_kwargs["thinking"] = thinking ? "true" : "false";
        const auto rendered = common_chat_templates_apply(templates.get(), inputs);
        const std::string expected = thinking ? "assistant:<think>" : "assistant:<response>";
        require(rendered.generation_prompt == expected,
                "generation prefix diff must use the same template kwargs as the actual prompt render");
        require(rendered.prompt.size() >= expected.size()
                && rendered.prompt.compare(rendered.prompt.size() - expected.size(), expected.size(), expected) == 0,
                "kwargs-controlled generation prefix does not match rendered prompt tail");
    }
    for (const auto * value : {"true", "false"}) {
        inputs.chat_template_kwargs["thinking"] = "false";
        inputs.chat_template_kwargs["add_generation_prompt"] = value;
        const auto rendered = common_chat_templates_apply(templates.get(), inputs);
        require(rendered.generation_prompt == "assistant:<response>",
                "reserved template kwarg must not collapse the prefix comparison to identical renders");
        require(rendered.prompt == "user:fixture\nassistant:<response>",
                "prefix correction changed the actual prompt render");
    }
    inputs.chat_template_kwargs.clear();
    require(common_chat_templates_apply(templates.get(), inputs).generation_prompt == "assistant:<think>",
            "a previous request's thinking kwarg leaked into the template default");
}

static void test_verified_current_prefix() {
    const std::string assistant = "<|open|>message role=\"assistant\"<|sep|>";
    const std::string response = "<|open|>response<|sep|>";
    // Earlier history contains the same synthetic marker IDs. Only the verified
    // current suffix is passed to the observer by the production launch path.
    const llama_tokens prompt = {1, 2, 3, 88, 89, 4, 5};
    server_expert_guard_verify_prefix(assistant + "<|open|>think<|sep|>", {4, 5}, prompt);
    server_expert_guard_verify_prefix(assistant + response, {4, 5}, prompt);
    server_expert_guard_verify_prefix(assistant + "<|open|>response", {4, 5}, prompt);
    rejects([&] { server_expert_guard_verify_prefix("", {}, prompt); });
    rejects([&] { server_expert_guard_verify_prefix(response, {4, 5}, prompt); });
    rejects([&] { server_expert_guard_verify_prefix(assistant + response, {1, 2, 3}, prompt); });
    rejects([&] { server_expert_guard_verify_prefix(assistant + response + "prefilled answer", {4, 5}, prompt); });
    rejects([&] { server_expert_guard_verify_prefix(assistant + "<|end_of_msg|>" + response, {4, 5}, prompt); });
}
// Optional real-tokenizer probe: GGUF vocabulary and tracked template only.
// No context, weight tensors, model forward, network or service is created.
static void test_real_k3_vocab_guard(const char * model_path, const char * template_path) {
    std::ifstream template_file(template_path, std::ios::binary);
    require(template_file.good(), "cannot read --template-file");
    const std::string template_source((std::istreambuf_iterator<char>(template_file)),
                                      std::istreambuf_iterator<char>());
    require(!template_source.empty() && !template_file.bad(), "empty/unreadable K3 template");

    struct backend_scope {
        backend_scope() { llama_backend_init(); }
        ~backend_scope() { llama_backend_free(); }
    } backend;
    auto model_params = llama_model_default_params();
    model_params.vocab_only = true;
    model_params.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(model_path, model_params), llama_model_free);
    require(model != nullptr, "vocab-only model load failed");
    char architecture[64] = {};
    require(llama_model_meta_val_str(model.get(), "general.architecture", architecture, sizeof(architecture)) > 0
            && std::string(architecture) == "kimi-k3", "--vocab-model must contain a Kimi-K3 vocabulary");
    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    require(vocab != nullptr, "loaded model has no vocabulary");
    const auto tokenize = [&](const std::string & text) {
        return common_tokenize(vocab, text, false, true);
    };
    const std::string assistant = "<|open|>message role=\"assistant\"<|sep|>";
    const std::string response = "<|open|>response<|sep|>";
    const std::string think = "<|open|>think<|sep|>";
    const auto marker = tokenize(response);
    require(!marker.empty(), "canonical response opener tokenized to nothing");
    const auto control_open_tokens = tokenize("<|open|>");
    const auto control_close_tokens = tokenize("<|close|>");
    require(control_open_tokens.size() == 1 && control_close_tokens.size() == 1 &&
            control_open_tokens[0] != control_close_tokens[0] &&
            llama_vocab_is_control(vocab, control_open_tokens[0]) &&
            llama_vocab_is_control(vocab, control_close_tokens[0]),
            "K3 control markers must be distinct single control-token IDs");
    const llama_token control_open = control_open_tokens[0], control_close = control_close_tokens[0];
    const auto ends_with = [](const llama_tokens & whole, const llama_tokens & suffix) {
        return suffix.size() <= whole.size() && std::equal(suffix.rbegin(), suffix.rend(), whole.rbegin());
    };
    const auto answer = tokenize("Hello.");
    require(!answer.empty(), "answer fixture tokenized to nothing");
    json cases = json::array();
    json window_cases = json::array();
    json control_cases = json::array();

    // Schedule predictions, not accepted tokens: prediction 0 belongs to the
    // final prompt row. No preceding prefill row consumes the start window.
    // This is a token-ID fixture, not a model forward or a server execution.
    const auto replay_windows = [&](const char * name, const llama_tokens & prefix,
            const llama_tokens & generated, uint32_t opener_n, uint32_t start_n) {
        require(!generated.empty(), "empty predictive-window fixture");
        server_expert_guard guard;
        guard.reset(opener_n, marker, start_n);
        guard.prime_current_prefix(prefix);
        require(guard.start_remaining() == start_n, "prefix priming consumed start predictions");
        const bool primed = guard.opened();
        const auto found = std::search(generated.begin(), generated.end(), marker.begin(), marker.end());
        const size_t opener_begin = primed ? 0 : found == generated.end() ? generated.size() + 1
            : size_t(found - generated.begin()) + marker.size();
        uint32_t union_count = 0, start_count = 0, opener_count = 0, overlap_count = 0;
        for (size_t i = 0; i < generated.size(); ++i) {
            const bool start = i < start_n;
            const bool opener = opener_n > 0 && i >= opener_begin && i - opener_begin < opener_n;
            require(guard.active() == (start || opener), "real-token union selected the wrong predictive row");
            require(guard.start_remaining() == (start ? start_n - i : 0),
                    "real-token start window has the wrong remaining prediction count");
            require(guard.remaining() == (opener ? opener_n - (i - opener_begin) : 0),
                    "real-token opener window has the wrong remaining prediction count");
            if (i == 0 || i == opener_begin) {
                const auto start_before = guard.start_remaining();
                const auto opener_before = guard.remaining();
                guard.commit_forward(false);
                require(guard.start_remaining() == start_before && guard.remaining() == opener_before,
                        "failed prediction consumed a real-token guard window");
            }
            union_count += guard.active();
            start_count += start;
            opener_count += opener;
            overlap_count += start && opener;
            guard.commit_forward(true);
            if (llama_vocab_is_eog(vocab, generated[i])) {
                require(i + 1 == generated.size(), "fixture continued after an end-of-generation token");
                guard.finish();
                break;
            }
            guard.accept_generated(generated[i]);
        }
        require(union_count == start_count + opener_count - overlap_count,
                "overlapping windows double-counted a guarded prediction");
        guard.finish();
        require(!guard.active() && guard.start_remaining() == 0 && guard.remaining() == 0,
                "stop left a real-token window active");
        return json{{"name", name}, {"opener_tokens", opener_n}, {"start_tokens", start_n},
            {"generated_tokens", generated}, {"prompt_primed", primed},
            {"first_prediction_guarded", start_n > 0 || (primed && opener_n > 0)},
            {"predictions", union_count}, {"start_predictions", start_count},
            {"opener_predictions", opener_count}, {"overlap_predictions", overlap_count},
            {"failed_predictions_preserve_both_windows", true}, {"union_identity_passed", true}};
    };

    const auto replay_control = [&](const char * name, const llama_tokens & prefix,
            const llama_tokens & generated, uint32_t opener_n, uint32_t start_n, uint32_t control_n) {
        require(!generated.empty(), "empty control-window fixture");
        server_expert_guard guard;
        guard.reset(opener_n, marker, start_n, control_n, control_open, control_close);
        guard.prime_current_prefix(prefix);
        require(!guard.control_active() && guard.control_remaining() == 0 && guard.control_triggers() == 0,
                "historical/current prompt control tokens must not arm control protection");
        require(guard.start_remaining() == start_n, "prefix consumed the start budget");

        // Independent interval oracle: each generated control at index j owns
        // future prediction indices j+1 .. j+M. Their union models retriggering,
        // without copying the observer's mutable countdown implementation.
        std::vector<size_t> trigger_indices;
        std::vector<bool> expected_control(generated.size(), false);
        for (size_t j = 0; j < generated.size(); ++j) {
            require(!llama_vocab_is_eog(vocab, generated[j]) || j + 1 == generated.size(),
                    "control fixture continues after EOS");
            if (control_n > 0 && (generated[j] == control_open || generated[j] == control_close)) {
                trigger_indices.push_back(j);
                for (size_t i = j + 1; i < generated.size() && i - j <= control_n; ++i) {
                    expected_control[i] = true;
                }
            }
        }
        const bool primed = opener_n > 0 && ends_with(prefix, marker);
        const auto found = std::search(generated.begin(), generated.end(), marker.begin(), marker.end());
        const size_t opener_begin = primed ? 0 : found == generated.end() ? generated.size() + 1
            : size_t(found - generated.begin()) + marker.size();
        require(guard.opened() == primed, "control setup changed opener priming");
        uint64_t union_count = 0, start_count = 0, opener_count = 0, overlap_count = 0;
        uint64_t control_count = 0, control_only_count = 0, triple_count = 0;
        json predictions = json::array();
        std::vector<size_t> expected_control_predictions, observed_control_predictions;
        for (size_t i = 0; i < generated.size(); ++i) {
            const bool start = i < start_n;
            const bool opener = opener_n > 0 && i >= opener_begin && i - opener_begin < opener_n;
            const bool control = expected_control[i];
            uint32_t control_remaining = 0;
            uint64_t prior_triggers = 0;
            for (const size_t j : trigger_indices) {
                if (j >= i) { break; }
                ++prior_triggers;
                if (i - j <= control_n) { control_remaining = uint32_t(j + control_n + 1 - i); }
            }
            require(guard.start_active() == start && guard.opener_active() == opener &&
                    guard.control_active() == control && guard.active() == (start || opener || control),
                    "three-window guard selected the wrong real-token prediction");
            require(guard.control_remaining() == control_remaining && guard.control_triggers() == prior_triggers,
                    "control countdown/retrigger count disagrees with the independent interval oracle");
            const auto start_before = guard.start_remaining(), opener_before = guard.remaining();
            const json failed_before = {{"start_remaining", guard.start_remaining()},
                {"opener_remaining", guard.remaining()}, {"control_remaining", guard.control_remaining()},
                {"control_triggers", guard.control_triggers()}};
            guard.commit_forward(false);
            const json failed_after = {{"start_remaining", guard.start_remaining()},
                {"opener_remaining", guard.remaining()}, {"control_remaining", guard.control_remaining()},
                {"control_triggers", guard.control_triggers()}};
            require(guard.start_remaining() == start_before && guard.remaining() == opener_before &&
                    guard.control_remaining() == control_remaining && guard.control_triggers() == prior_triggers,
                    "failed forward consumed or retriggered a control/start/opener window");
            if (control) { expected_control_predictions.push_back(i); }
            if (guard.control_active()) { observed_control_predictions.push_back(i); }
            union_count += guard.active(); start_count += start; opener_count += opener;
            overlap_count += start && opener; control_count += control;
            control_only_count += control && !start && !opener; triple_count += start && opener && control;
            predictions.push_back({{"prediction_index", i}, {"token_id", generated[i]},
                {"start_active", start}, {"opener_active", opener}, {"control_active", control},
                {"guarded", start || opener || control}, {"control_remaining", control_remaining},
                {"start_remaining", start_before}, {"opener_remaining", opener_before},
                {"control_triggers_before_prediction", prior_triggers},
                {"failed_forward_before", failed_before}, {"failed_forward_after", failed_after},
                {"failed_forward_preserves_windows", true}});
            guard.commit_forward(true);
            require(guard.control_remaining() == (control_remaining > 0 ? control_remaining - 1 : 0),
                    "successful forward failed to consume exactly one control prediction");
            if (llama_vocab_is_eog(vocab, generated[i])) { guard.finish(); }
            else { guard.accept_generated(generated[i]); }
        }
        require(expected_control_predictions == observed_control_predictions &&
                union_count == start_count + opener_count - overlap_count + control_only_count,
                "three-way overlap double-counted successful predictions");
        const auto triggers = guard.control_triggers();
        require(triggers == trigger_indices.size(), "generated control tokens were missed or counted twice");
        guard.finish();
        guard.accept_generated(control_open);
        guard.accept_generated(control_close);
        require(!guard.active() && guard.control_remaining() == 0 && guard.control_triggers() == triggers,
                "finished control observer resurrected after EOS/stop");
        return json{{"name", name}, {"opener_tokens", opener_n}, {"start_tokens", start_n},
            {"control_tokens", control_n}, {"generated_tokens", generated}, {"prompt_primed", primed},
            {"first_prediction_guarded", start_n > 0 || primed}, {"predictions", union_count},
            {"start_predictions", start_count}, {"opener_predictions", opener_count},
            {"overlap_predictions", overlap_count}, {"control_predictions", control_count},
            {"control_only_predictions", control_only_count}, {"triple_predictions", triple_count},
            {"control_triggers", triggers}, {"generated_control_indices", trigger_indices},
            {"expected_control_prediction_indices", expected_control_predictions},
            {"observed_control_prediction_indices", observed_control_predictions},
            {"prediction_schedule", predictions}, {"prefix_controls_excluded", true},
            {"union_identity_passed", true}, {"finish_blocks_retrigger", true}};
    };

    for (const bool thinking : {true, false}) {
        server_chat_params options = {};
        options.use_jinja = true;
        options.enable_thinking = thinking;
        options.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        options.tmpls = common_chat_templates_init(model.get(), template_source);
        require(options.tmpls != nullptr, "K3 template initialization failed");
        json chat = {
            {"messages", json::array({
                {{"role", "user"}, {"content", "Hello."}},
                {{"role", "assistant"}, {"content", "Hello again."}, {"reasoning_content", ""}},
                {{"role", "user"}, {"content", "你好。"}}})},
            {"chat_template_kwargs", {{"thinking", thinking}}},
            {"stream", false}, {"add_generation_prompt", true}};
        std::vector<raw_buffer> files;
        const json native = oaicompat_chat_params_parse(chat, options, files);
        const std::string prompt = native.at("prompt").get<std::string>();
        const std::string prefix = native.at("generation_prompt").get<std::string>();
        const auto tail_start = prompt.rfind("<|open|>message");
        const json diagnostic = {{"test", "k3_vocab_response_guard_prefix_diagnostic"},
            {"thinking", thinking}, {"enable_thinking", options.enable_thinking},
            {"chat_template_kwargs", chat.at("chat_template_kwargs")},
            {"actual_generation_prompt", prefix},
            {"expected_generation_prompt", assistant + (thinking ? think : response)},
            {"actual_prompt_tail", prompt.substr(tail_start == std::string::npos ? 0 : tail_start)},
            {"actual_prompt_bytes", prompt.size()}, {"generation_prompt_bytes", prefix.size()}};
        std::fprintf(stderr, "%s\n", diagnostic.dump().c_str());
        require(prefix == assistant + (thinking ? think : response),
                "actual template did not produce the canonical current assistant prefix");
        const auto prompt_tokens = tokenize(prompt);
        const auto prefix_tokens = tokenize(prefix);
        require(!prefix_tokens.empty() && prompt_tokens.size() > prefix_tokens.size(), "missing historical prompt fixture");
        require(ends_with(prompt_tokens, prefix_tokens), "separately tokenized current prefix is not the actual prompt suffix");
        server_expert_guard_verify_prefix(prefix, prefix_tokens, prompt_tokens);
        rejects([&] { server_expert_guard_verify_prefix(prompt, prompt_tokens, prompt_tokens); });
        auto altered_prefix = prefix_tokens;
        altered_prefix.back() = LLAMA_TOKEN_NULL;
        rejects([&] { server_expert_guard_verify_prefix(prefix, altered_prefix, prompt_tokens); });
        const auto replay = [&](const char * name, const llama_tokens & generated,
                uint32_t opener_n, uint32_t start_n) {
            // Bind every window case to this actual template's current suffix,
            // including the thinking prefix preceding the reported raw tail.
            server_expert_guard_verify_prefix(prefix, prefix_tokens, prompt_tokens);
            auto row = replay_windows(name, prefix_tokens, generated, opener_n, start_n);
            row["generation_prefix"] = prefix;
            row["thinking"] = thinking;
            row["current_suffix_verified"] = true;
            return row;
        };
        const auto control_replay = [&](const char * name, const llama_tokens & generated,
                uint32_t opener_n, uint32_t start_n, uint32_t control_n) {
            server_expert_guard_verify_prefix(prefix, prefix_tokens, prompt_tokens);
            auto row = replay_control(name, prefix_tokens, generated, opener_n, start_n, control_n);
            row["generation_prefix"] = prefix;
            row["thinking"] = thinking;
            row["current_suffix_verified"] = true;
            return row;
        };

        const auto history_end = prompt_tokens.end() - prefix_tokens.size();
        require(std::search(prompt_tokens.begin(), history_end, marker.begin(), marker.end()) != history_end,
                "historical assistant fixture must contain a real response opener");
        server_expert_guard observer;
        observer.reset(1, marker);
        // Only the verified current prefix enters the observer; earlier empty
        // thinking/response blocks in actual tokenized history are excluded.
        observer.prime_current_prefix(prefix_tokens);
        if (thinking) {
            require(!observer.opened() && !observer.active(), "historical response opener armed the current thinking turn");
            const auto suffix = tokenize("Short thought.<|close|>think<|sep|>" + response);
            require(ends_with(suffix, marker), "generated/forced response opener is not a contiguous canonical token suffix");
            for (size_t i = 0; i < suffix.size(); ++i) {
                observer.accept_generated(suffix[i]);
                if (i + 1 < suffix.size()) {
                    require(!observer.active(), "real-token guard armed before the final opener token");
                }
            }
            require(observer.active() && observer.remaining() == 1,
                    "real generated opener must protect its forward predicting answer token #1");
        } else {
            require(ends_with(prefix_tokens, marker), "thinking=false prefix does not end in the same contiguous response marker");
            require(observer.active() && observer.remaining() == 1,
                    "thinking=false must guard the final prompt row before its first sample");
        }
        observer.commit_forward(false);
        require(observer.remaining() == 1, "failed real-token boundary forward consumed N=1");
        observer.commit_forward(true);
        observer.accept_generated(answer.front());
        require(!observer.active() && observer.remaining() == 0,
                "answer token #1 forward predicts token #2 and must be unguarded for N=1");
        for (const auto token : marker) { observer.accept_generated(token); }
        require(!observer.active(), "a repeated real response opener must not restart N=1");

        if (!thinking) {
            // Also prove that a real partial current prefix can be completed by
            // a generated separator, using the same suffix verification path.
            const std::string separator = "<|sep|>";
            const std::string partial_prefix = prefix.substr(0, prefix.size() - separator.size());
            const std::string partial_prompt = prompt.substr(0, prompt.size() - separator.size());
            const auto partial_tokens = tokenize(partial_prefix);
            server_expert_guard_verify_prefix(partial_prefix, partial_tokens, tokenize(partial_prompt));
            observer.reset(1, marker);
            observer.prime_current_prefix(partial_tokens);
            require(!observer.active() && observer.matched_prefix() > 0, "real partial opener did not retain a marker prefix");
            for (const auto token : tokenize(separator)) { observer.accept_generated(token); }
            require(observer.active() && observer.remaining() == 1, "real generated separator failed to complete partial prompt opener");
        }
        const std::string body = "One two three four five six seven eight nine ten.<|end_of_msg|>";
        if (thinking) {
            const std::string shopping_tail =
                "[\"牛奶\", \"面包\", \"香蕉\"]<|close|>response<|sep|>"
                "[\"牛奶\", \"面包\", \"香蕉\"]<|close|>response<|sep|>"
                "<|close|>message<|sep|><|end_of_msg|>";
            const auto shopping_tokens = tokenize(shopping_tail);
            require(std::search(shopping_tokens.begin(), shopping_tokens.end(), marker.begin(), marker.end()) == shopping_tokens.end(),
                    "reported shopping tail unexpectedly contains a canonical response opener");
            const std::vector<std::vector<uint32_t>> shopping_configs = {{8, 4, 3}, {0, 0, 3}, {8, 4, 0}};
            for (const auto & config : shopping_configs) {
                const char * name = config[2] == 0 ? "reported_shopping_tail_control_disabled"
                    : config[0] == 0 ? "reported_shopping_tail_control_only" : "reported_shopping_tail_control3";
                auto row = control_replay(name, shopping_tokens, config[0], config[1], config[2]);
                require(row.at("opener_predictions") == 0, "malformed shopping closers armed the response opener");
                require(row.at("control_triggers") == (config[2] ? 3 : 0), "shopping tail must contain exactly three generated closers");
                row["generated_text"] = shopping_tail;
                control_cases.push_back(std::move(row));
            }
            const std::string natural = "OK.<|close|>think<|sep|>" + response + body;
            const std::string forced = "<|close|>think<|sep|>" + response + body;
            control_cases.push_back(control_replay("natural_transition_control3", tokenize(natural), 8, 4, 3));
            control_cases.push_back(control_replay("forced_transition_control3", tokenize(forced), 8, 4, 3));
            const auto natural_tokens = tokenize(natural);
            const auto natural_opener = std::search(natural_tokens.begin(), natural_tokens.end(), marker.begin(), marker.end());
            require(natural_opener != natural_tokens.end(), "natural control fixture lost its response opener");
            const uint32_t long_start = uint32_t(natural_opener - natural_tokens.begin() + marker.size() + 2);
            auto triple = control_replay("overlapping_all_three_control3", natural_tokens, 8, long_start, 3);
            require(triple.at("triple_predictions").get<uint64_t>() > 0, "real K3 transition did not exercise triple overlap");
            control_cases.push_back(std::move(triple));
            auto retrigger = control_replay("repeated_controls_control3_only",
                tokenize("<|close|><|open|><|close|>OK.<|end_of_msg|>"), 0, 0, 3);
            require(retrigger.at("control_triggers") == 3 && !retrigger.at("first_prediction_guarded").get<bool>(),
                    "control-only must start inactive and rearm for each generated control");
            control_cases.push_back(std::move(retrigger));
            const auto control_eos = llama_vocab_eos(vocab);
            require(control_eos != LLAMA_TOKEN_NULL && llama_vocab_is_eog(vocab, control_eos), "control fixture has no real EOS");
            auto eos_after_control = control_replay("eos_after_control3", {control_close, control_eos}, 0, 0, 3);
            require(eos_after_control.at("predictions") == 1 && eos_after_control.at("control_predictions") == 1,
                    "EOS-producing prediction must consume one armed control position");
            control_cases.push_back(std::move(eos_after_control));
            control_cases.push_back(control_replay("eos_first_control3", {control_eos}, 0, 0, 3));
            const std::string raw_tail = "Got it.<|close|>raw<|close|>message<|sep|><|end_of_msg|>";
            const auto raw_tokens = tokenize(raw_tail);
            require(raw_tokens.size() >= 4 &&
                    std::search(raw_tokens.begin(), raw_tokens.end(), marker.begin(), marker.end()) == raw_tokens.end(),
                    "reported raw-tail fixture must have four tokens and no response opener");
            for (const uint32_t start_n : {0u, 2u, 4u}) {
                auto row = replay("reported_raw_tail_without_opener", raw_tokens, 8, start_n);
                require(row.at("predictions").get<uint32_t>() == start_n && row.at("opener_predictions") == 0,
                        "start guard failed to cover a real no-opener generation");
                row["generated_text"] = raw_tail;
                window_cases.push_back(std::move(row));
            }
            window_cases.push_back(replay("start_only_without_opener", raw_tokens, 0, 2));
            window_cases.push_back(replay("both_windows_disabled", raw_tokens, 0, 0));

            const auto transition = tokenize("OK.<|close|>think<|sep|>" + response + body);
            const auto opening = std::search(transition.begin(), transition.end(), marker.begin(), marker.end());
            require(opening != transition.end(), "short thinking transition lost its canonical response opener");
            const size_t begin = size_t(opening - transition.begin()) + marker.size();
            require(begin > 2 && transition.size() >= begin + 8,
                    "short thinking fixture must separate start2 and an eight-prediction opener window");
            auto delayed = replay("short_thinking_delayed_opener", transition, 8, 2);
            require(delayed.at("predictions") == 10 && delayed.at("overlap_predictions") == 0,
                    "delayed response opener did not retain its own full window after start expiry");
            window_cases.push_back(std::move(delayed));
            auto overlap = replay("short_thinking_overlapping_windows", transition, 4,
                    uint32_t(begin + 2));
            require(overlap.at("overlap_predictions") == 2 &&
                    overlap.at("predictions").get<size_t>() == begin + 4,
                    "overlapping short thinking windows must cover a union, not concatenated budgets");
            window_cases.push_back(std::move(overlap));

            const auto eos = llama_vocab_eos(vocab);
            require(eos != LLAMA_TOKEN_NULL && llama_vocab_is_eog(vocab, eos), "real K3 vocabulary has no EOS token");
            auto early_eos = replay("first_prediction_is_eos", {eos}, 8, 4);
            require(early_eos.at("predictions") == 1 && early_eos.at("start_predictions") == 1,
                    "the prediction producing EOS must consume exactly one start position");
            window_cases.push_back(std::move(early_eos));
        } else {
            const auto body_tokens = tokenize(body);
            control_cases.push_back(control_replay("prompt_primed_control3", body_tokens, 8, 4, 3));
            auto prefix_excluded = control_replay("prompt_control_only_excluded", body_tokens, 0, 0, 3);
            require(prefix_excluded.at("predictions") == 0 && prefix_excluded.at("control_triggers") == 0,
                    "current response prefix controls must not arm a control-only request");
            control_cases.push_back(std::move(prefix_excluded));
            require(body_tokens.size() >= 8, "prompt-primed fixture is too short");
            auto primed = replay("prompt_primed_start2_opener8", body_tokens, 8, 2);
            require(primed.at("predictions") == 8 && primed.at("overlap_predictions") == 2,
                    "prompt-primed first predictions must overlap without extending opener8");
            window_cases.push_back(std::move(primed));
            auto start_tail = replay("prompt_primed_start4_opener2", body_tokens, 2, 4);
            require(start_tail.at("predictions") == 4 && start_tail.at("overlap_predictions") == 2,
                    "start window must survive expiry of the shorter prompt-primed opener window");
            window_cases.push_back(std::move(start_tail));
        }
        cases.push_back({{"thinking", thinking}, {"generation_prefix", prefix},
                         {"prompt_tokens", prompt_tokens.size()}, {"prefix_tokens", prefix_tokens},
                         {"historical_marker_present", true}, {"current_suffix_verified", true},
                         {"n1_predictive_boundary_passed", true}});
    }
    require(window_cases.size() == 10, "real-vocabulary predictive-window case count changed");
    require(control_cases.size() == 11, "real-vocabulary control-window case count changed");
    const json receipt = {{"test", "k3_vocab_response_guard"}, {"status", "PASS"},
        {"model", model_path}, {"template_file", template_path}, {"vocab_only", true},
        {"n_gpu_layers", 0}, {"context_created", false}, {"model_forward_executed", false},
        {"tokenize_add_special", false}, {"tokenize_parse_special", true},
        {"marker", response}, {"marker_token_ids", marker}, {"cases", cases},
        {"start_opener_window_cases", window_cases}, {"start_opener_window_case_count", window_cases.size()},
        {"control_window_cases", control_cases}, {"control_window_case_count", control_cases.size()},
        {"control_open_token_id", control_open}, {"control_close_token_id", control_close},
        {"control_markers_single_control_typed", true},
        {"predictive_replay_model_forward", false}};
    std::puts("KIMI_K3_START_OPENER_VOCAB_OK cases=10 raw_no_opener=PASS first_prediction=PASS "
              "delayed=PASS overlap=PASS prompt_primed=PASS eos_first=PASS model_forward=NOT_RUN");
    std::puts("KIMI_K3_CONTROL_GUARD_VOCAB_OK cases=11 shopping_tail=PASS predictive_ids=PASS "
              "natural_forced=PASS triple_overlap=PASS retrigger=PASS prefix_excluded=PASS eos=PASS model_forward=NOT_RUN");
    std::puts(receipt.dump().c_str());
}

int main(int argc, char ** argv) {
    try {
        if (argc == 5 && std::strcmp(argv[1], "--vocab-model") == 0
                      && std::strcmp(argv[3], "--template-file") == 0) {
            test_real_k3_vocab_guard(argv[2], argv[4]);
            return 0;
        }
        require(argc == 1,
                "usage: test-server-runtime-routing [--vocab-model MODEL.gguf --template-file models/templates/Kimi-K3.jinja]");
        test_template_kwargs_generation_prefix();
        test_verified_current_prefix();
        std::puts("runtime routing server: 2 groups PASS (no service started)");
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
