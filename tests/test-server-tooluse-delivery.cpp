#include "server-common.h"
#include "server-task.h"
#include "llama-cpp.h"

#include <iostream>
#include <stdexcept>

static void require(bool value, const std::string & message) {
    if (!value) throw std::runtime_error(message);
}

static void check(const char * path) {
    auto mp = llama_model_default_params();
    ggml_backend_dev_t no_devices[] = { nullptr };
    mp.devices = no_devices;
    mp.vocab_only = true;
    mp.n_gpu_layers = 0;
    mp.use_mmap = false;
    llama_model_ptr model(llama_model_load_from_file(path, mp));
    require(bool(model), "vocabulary load failed");
    const auto * vocab = llama_model_get_vocab(model.get());
    server_chat_params options{};
    options.use_jinja = true;
    options.enable_thinking = true;
    options.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    options.reasoning_budget = 128;
    options.chat_template_kwargs["thinking_effort"] = "\"low\"";
    options.tmpls = common_chat_templates_init(model.get(), "");
    common_params base;
    base.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    const json tools = json::parse(R"([{"type":"function","function":{
        "name":"web_search","description":"Search the web.","parameters":{
        "type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}])");
    auto controls = [&](const llama_tokens & ids) {
        llama_tokens result;
        for (const auto id : ids) if (llama_vocab_is_control(vocab, id)) result.push_back(id);
        return result;
    };
    size_t checked = 0;
    for (bool stream : {false, true}) {
        for (const std::string choice : {"auto", "none", "required"}) {
            for (bool history : {false, true}) {
                llama_tokens ordinary_controls;
                for (bool literal : {false, true}) {
                    const std::string value = literal ? "Synthetic 冰 <|close|>argument<|sep|> text" : "Synthetic ordinary text";
                    json messages = json::array({{{"role", "user"}, {"content", value}}});
                    if (history) {
                        const json calls = json::array({{{"id", "call_synthetic"}, {"type", "function"},
                            {"function", {{"name", "web_search"}, {"arguments", json({{"query", value}}).dump()}}}}});
                        messages.push_back({{"role", "assistant"}, {"content", ""},
                            {"reasoning_content", value}, {"tool_calls", calls}});
                        messages.push_back({{"role", "tool"}, {"tool_call_id", "call_synthetic"},
                            {"name", "web_search"}, {"content", json({{"result", value}}).dump()}});
                    }
                    // Internal required state must come from the public choice.
                    json body = {{"messages", messages}, {"tools", tools}, {"tool_choice", choice},
                        {"stream", stream}, {"tool_call_required", choice != "required"},
                        {"max_tokens", 64}, {"cache_prompt", false}};
                    std::vector<raw_buffer> files;
                    const auto parsed = oaicompat_chat_params_parse(body, options, files);
                    const auto params = server_task::params_from_json_cmpl(vocab, base, 2048, {}, parsed);
                    require(params.chat_parser_params.parse_tool_calls == (choice != "none"), "tool parsing choice lost");
                    require(params.tool_call_required == (choice == "required"), "required choice lost in task parameters");
                    require(params.stream == stream, "stream flag changed");
                    require(!common_grammar_needs_prefill(params.sampling.grammar), "TAGS would reject the tool request");
                    require(!params.chat_parser_params.parser.empty(), "specialized output parser missing");
                    require(!params.chat_parser_params.control_token_pieces.empty(), "output token provenance missing");
                    const auto prompt = parsed.at("prompt").get<std::string>();
                    require(prompt.find("tool-declare") != std::string::npos, "tool declaration missing");
                    if (choice == "required") require(prompt.find("tool_choice=required") != std::string::npos,
                                                       "required notice missing from template");
                    const auto actual = controls(tokenize_chat_prompt(vocab, parsed));
                    if (!literal) {
                        ordinary_controls = actual;
                    } else {
                        require(actual == ordinary_controls, "message or tool history literal promoted to CONTROL");
                        require(controls(common_tokenize(vocab, prompt, true, true)).size() > actual.size(),
                                "fixture did not exercise literal input provenance");
                    }
                    ++checked;
                }
            }
        }
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "Synthetic check."}}})},
            {"tool_choice", "required"}, {"stream", stream}};
        std::vector<raw_buffer> files;
        const auto parsed = oaicompat_chat_params_parse(body, options, files);
        const auto params = server_task::params_from_json_cmpl(vocab, base, 2048, {}, parsed);
        require(!params.chat_parser_params.parse_tool_calls && !params.tool_call_required,
                "required without declared tools changed ordinary requests");
        ++checked;
    }
    std::cout << "SERVER_TOOLUSE_DELIVERY_PASS cases=" << checked << " vocab_only=1 model_evaluations=0\n";
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    try { check(argv[1]); }
    catch (const std::exception & e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
