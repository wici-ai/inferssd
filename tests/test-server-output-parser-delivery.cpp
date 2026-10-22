#include "server-task.h"
#include "llama-cpp.h"

#include <iostream>
#include <stdexcept>

static void require(bool value, const std::string & message) {
    if (!value) throw std::runtime_error(message);
}

struct generated_fixture {
    std::string text;
    llama_tokens ids;
    std::vector<size_t> token_ends;
    common_chat_output_source source;
};

static common_chat_output_source snapshot(const generated_fixture & fixture, size_t bytes) {
    common_chat_output_source result;
    result.valid = true;
    for (const auto & control : fixture.source.controls) {
        if (control.begin < bytes) result.controls.push_back(control);
    }
    return result;
}

static void check(const char * path) {
    auto params = llama_model_default_params();
    ggml_backend_dev_t no_devices[] = { nullptr };
    params.devices = no_devices;
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    params.use_mmap = false;
    llama_model_ptr model(llama_model_load_from_file(path, params));
    require(bool(model), "vocabulary load failed");
    const auto * vocab = llama_model_get_vocab(model.get());
    auto templates = common_chat_templates_init(model.get(), "");
    common_chat_templates_inputs input;
    common_chat_msg user;
    user.role = "user"; user.content = "Synthetic parser check.";
    input.messages.push_back(user);
    input.use_jinja = true; input.enable_thinking = true;
    input.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    input.chat_template_kwargs["thinking_effort"] = "\"low\"";
    const auto rendered = common_chat_templates_apply(templates.get(), input);
    common_chat_parser_params pp(rendered);
    pp.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    pp.parser.load(rendered.parser);
    pp.set_vocab(vocab);

    const std::string boundary = "<|close|>think<|sep|><|open|>response<|sep|>";
    const std::string ending = "<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>";
    struct testcase { std::string name, reasoning, content; };
    const std::vector<testcase> cases = {
        {"ordinary", "Synthetic reasoning.", "Synthetic response."},
        {"close-spelling", "Synthetic reasoning.", "Literal A <|close|> B."},
        {"response-delimiter", "Synthetic reasoning.", "Literal A <|close|>response<|sep|> B."},
        {"repeated-delimiter", "Synthetic reasoning.", "A <|close|>response<|sep|> B <|close|>response<|sep|> C."},
        {"reasoning-delimiter", "Reason A <|close|>think<|sep|> B.", "Synthetic response."},
        {"utf8-delimiter", "Synthetic reasoning 冰.", "冰 A <|close|>response<|sep|> μ B."},
    };
    size_t checked = 0, raw_checked = 0;
    for (const auto & item : cases) {
        generated_fixture fixture;
        fixture.source.valid = true;
        auto append = [&](const std::string & text, bool special) {
            const auto tokens = common_tokenize(vocab, text, false, special);
            std::string rebuilt;
            for (const auto id : tokens) {
                const auto piece = common_token_to_piece(vocab, id, true);
                const bool control = llama_vocab_is_control(vocab, id);
                require(special || !control, "fixture literal unexpectedly contains a CONTROL ID");
                if (control) fixture.source.controls.push_back({fixture.text.size(), fixture.text.size() + piece.size(), id});
                fixture.text += piece; rebuilt += piece;
                fixture.ids.push_back(id); fixture.token_ends.push_back(fixture.text.size());
            }
            require(rebuilt == text, "fixture text did not roundtrip");
        };
        append(item.reasoning, false); append(boundary, true);
        append(item.content, false); append(ending, true);
        require(fixture.source.controls.size() == 9, "expected Kimi protocol control IDs");

        // Result parsing must work without the optional externally returned IDs.
        for (const bool return_tokens : {false, true}) {
            for (int mode = 0; mode < 4; ++mode) {
                const std::string label = item.name + "/mode=" + std::to_string(mode) + "/tokens=" + std::to_string(return_tokens);
                task_result_state state(pp);
                std::string content, reasoning;
                auto accumulate = [&](const std::vector<common_chat_msg_diff> & diffs) {
                    for (const auto & d : diffs) { content += d.content_delta; reasoning += d.reasoning_content_delta; }
                };
                if (mode != 0) {
                    std::vector<size_t> boundaries;
                    if (mode == 3) {
                        // Includes cuts inside a real CONTROL spelling. The full
                        // token identity is known before its bytes are released.
                        for (size_t i = 1; i <= fixture.text.size(); ++i) boundaries.push_back(i);
                    } else {
                        const size_t group = mode == 1 ? 1 : 3;
                        for (size_t i = group; i < fixture.token_ends.size(); i += group) boundaries.push_back(fixture.token_ends[i - 1]);
                        boundaries.push_back(fixture.text.size());
                    }
                    size_t sent = 0;
                    for (const auto end : boundaries) {
                        if (validate_utf8(fixture.text.substr(0, end)) != end) continue;
                        server_task_result_cmpl_partial result{};
                        result.content = fixture.text.substr(sent, end - sent);
                        result.output_source = snapshot(fixture, end);
                        if (return_tokens) {
                            for (size_t i = 0; i < fixture.ids.size(); ++i) {
                                if (fixture.token_ends[i] > sent && fixture.token_ends[i] <= end) result.tokens.push_back(fixture.ids[i]);
                            }
                        }
                        result.update(state); accumulate(result.oaicompat_msg_diffs); sent = end;
                    }
                    require(sent == fixture.text.size(), label + ": incomplete UTF-8 send");
                }
                server_task_result_cmpl_final result{};
                result.stream = mode != 0;
                if (!result.stream) {
                    result.content = fixture.text;
                    if (return_tokens) result.tokens = fixture.ids;
                }
                result.output_source = fixture.source;
                result.stop = STOP_TYPE_EOS;
                result.update(state); accumulate(result.oaicompat_msg_diffs);
                require(result.oaicompat_msg.content == item.content && result.oaicompat_msg.reasoning_content == item.reasoning,
                        label + ": incorrect final channels");
                require(content == item.content && reasoning == item.reasoning, label + ": incorrect stream reconstruction");
                require(result.stop == STOP_TYPE_EOS, label + ": parser changed stop reason");
                ++checked;
            }
        }
        common_chat_parser_params raw;
        raw.set_vocab(vocab);
        const auto unparsed = common_chat_parse(fixture.text, false, raw, &fixture.source);
        require(unparsed.content == fixture.text && unparsed.reasoning_content.empty(), "raw content contract changed");
        ++raw_checked;
    }
    std::cout << "SERVER_OUTPUT_PARSER_DELIVERY_PASS cases=" << checked << " raw=" << raw_checked
              << " vocab_only=1 model_evaluations=0\n";
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    try { check(argv[1]); }
    catch (const std::exception & e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
