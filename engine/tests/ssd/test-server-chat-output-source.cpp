#include "server-task.h"
#include "llama-cpp.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

static void require(bool ok, const std::string & what) {
    if (!ok) throw std::runtime_error(what);
}
static std::string hex(const std::string & s) {
    static const char * digits = "0123456789abcdef";
    std::string out;
    for (unsigned char c : s) { out += digits[c >> 4]; out += digits[c & 15]; }
    return out;
}
static json result_json(const common_chat_msg & msg) {
    return {{"content", msg.content}, {"reasoning", msg.reasoning_content}};
}
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    auto params = llama_model_default_params();
    ggml_backend_dev_t no_devices[] = { nullptr };
    params.devices = no_devices;
    params.vocab_only = true; params.n_gpu_layers = 0; params.use_mmap = false;
    llama_model_ptr model(llama_model_load_from_file(argv[1], params));
    require(bool(model), "vocabulary load failed");
    const auto * vocab = llama_model_get_vocab(model.get());
    auto templates = common_chat_templates_init(model.get(), "");
    common_chat_templates_inputs in;
    common_chat_msg user; user.role = "user"; user.content = "Synthetic parser check.";
    in.messages.push_back(user); in.use_jinja = true; in.enable_thinking = true;
    in.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    in.chat_template_kwargs["thinking_effort"] = "\"low\"";
    const auto rendered = common_chat_templates_apply(templates.get(), in);
    common_chat_parser_params pp(rendered);
    pp.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    pp.parser.load(rendered.parser);
    pp.set_vocab(vocab);
    require(!pp.parser.empty(), "Kimi parser not constructed");
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
        {"all-marker-spellings", "A <|open|>response<|sep|> B <|close|>think<|sep|> C.", "A <|open|>think<|sep|> B <|close|>message<|sep|><|end_of_msg|> C."},
        {"literal-tail", "Synthetic reasoning.", "Literal A <|close|>response<|sep|>"},
        {"partial-literal-tail", "Synthetic reasoning.", "Literal A <|cl"},
    };
    json rows = json::array();
    for (const auto & item : cases) {
        llama_tokens ids;
        std::string text;
        common_chat_output_source source; source.valid = true;
        json token_rows = json::array();
        auto append = [&](const std::string & value, bool structural, const char * region) {
            const auto tokens = common_tokenize(vocab, value, false, structural);
            std::string rebuilt;
            for (auto id : tokens) {
                const auto piece = common_token_to_piece(vocab, id, true);
                const bool control = llama_vocab_is_control(vocab, id);
                require(structural || !control, "literal region contains a control token ID");
                token_rows.push_back({{"id", id}, {"control", control}, {"region", region},
                    {"offset", text.size()}, {"bytes", piece.size()}, {"piece_hex", hex(piece)}});
                if (control && !piece.empty()) source.controls.push_back({text.size(), text.size()+piece.size(), id});
                text += piece; rebuilt += piece; ids.push_back(id);
            }
            require(rebuilt == value, "synthetic token roundtrip changed bytes");
        };
        append(item.reasoning, false, "reasoning"); append(boundary, true, "boundary");
        append(item.content, false, "content"); append(ending, true, "ending");
        require(std::count_if(token_rows.begin(), token_rows.end(), [](const json & r) {
            return r.at("control").get<bool>();
        }) >= 8, "missing structural control token IDs");
        json row = {{"name", item.name}, {"expected", {{"content", item.content}, {"reasoning", item.reasoning}}},
                    {"text", text}, {"tokens", token_rows}};
        try {
            task_result_state state(pp);
            server_task_result_cmpl_final final{};
            final.content = text; final.tokens = {}; final.stream = false;
            final.output_source = source; // return_tokens=false must still parse correctly
            final.update(state);
            row["nonstream"] = result_json(final.oaicompat_msg);
            row["nonstream"]["correct"] = final.oaicompat_msg.content == item.content &&
                                          final.oaicompat_msg.reasoning_content == item.reasoning;
        } catch (const std::exception & e) { row["nonstream"] = {{"correct", false}, {"error", e.what()}}; }
        size_t token_index = 0;
        try {
            task_result_state state(pp);
            std::string pending, all, wire_content, wire_reasoning;
            auto add_diffs = [&](const std::vector<common_chat_msg_diff> & diffs) {
                for (const auto & diff : diffs) { wire_content += diff.content_delta; wire_reasoning += diff.reasoning_content_delta; }
            };
            for (; token_index < ids.size(); ++token_index) {
                const auto piece = common_token_to_piece(vocab, ids[token_index], true);
                all += piece; pending += piece;
                if (validate_utf8(all) != all.size()) continue; // Same UTF-8 send boundary as server.
                server_task_result_cmpl_partial partial{};
                partial.content = pending; partial.tokens = {};
                partial.output_source.valid = true;
                for (const auto & t : source.controls) if (t.begin < all.size()) partial.output_source.controls.push_back(t);
                partial.update(state); add_diffs(partial.oaicompat_msg_diffs); pending.clear();
            }
            server_task_result_cmpl_final final{};
            final.content = pending; final.stream = true; final.output_source = source;
            final.update(state); add_diffs(final.oaicompat_msg_diffs);
            row["stream"] = result_json(final.oaicompat_msg);
            row["stream"]["wire_content"] = wire_content; row["stream"]["wire_reasoning"] = wire_reasoning;
            row["stream"]["correct"] = final.oaicompat_msg.content == item.content && final.oaicompat_msg.reasoning_content == item.reasoning &&
                                       wire_content == item.content && wire_reasoning == item.reasoning;
        } catch (const std::exception & e) { row["stream"] = {{"correct", false}, {"error", e.what()}, {"token_index", token_index}}; }
        rows.push_back(row);
    }
    require(rows[0]["nonstream"]["correct"] == true && rows[0]["stream"]["correct"] == true, "ordinary baseline failed");
    for (const auto & row : rows) require(row["nonstream"]["correct"] == true && row["stream"]["correct"] == true, "output provenance correction failed: " + row["name"].get<std::string>());
    // Declared tools: generated CONTROL tokens delimit the tool section; the same
    // spellings produced as ordinary tokens stay inside argument values or content.
    json tool_rows = json::array();
    {
        common_chat_templates_inputs tool_in = in;
        tool_in.tools = {{"web_search", "Search the web.",
            R"({"type":"object","properties":{"query":{"type":"string"},"max_results":{"type":"integer"}},"required":["query"]})"}};
        tool_in.parallel_tool_calls = true;
        const auto tool_rendered = common_chat_templates_apply(templates.get(), tool_in);
        require(tool_rendered.prompt.find("tool-declare") != std::string::npos, "tool declaration missing from prompt");
        common_chat_parser_params tool_pp(tool_rendered);
        tool_pp.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        tool_pp.parser.load(tool_rendered.parser);
        tool_pp.set_vocab(vocab);
        tool_pp.parse_tool_calls = true;

        const std::string reasoning = "Need a search.";
        const std::string opened = "<|close|>think<|sep|><|open|>response<|sep|>";
        const std::string tools_open = "<|close|>response<|sep|><|open|>tools<|sep|>";
        const std::string call1 = "<|open|>call tool=\"web_search\" index=\"1\"<|sep|>";
        const std::string call2 = "<|open|>call tool=\"web_search\" index=\"2\"<|sep|>";
        const std::string query = "<|open|>argument key=\"query\" type=\"string\"<|sep|>";
        const std::string arg_end = "<|close|>argument<|sep|>";
        const std::string call_end = "<|close|>call<|sep|>";
        const std::string tail = "<|close|>tools<|sep|><|close|>message<|sep|><|end_of_msg|>";
        struct segment { std::string text; bool structural; };
        struct tool_case { std::string name; std::vector<segment> segments; std::string content; json calls; };
        auto call = [](const std::string & query_text) {
            return json{{"name", "web_search"}, {"arguments", {{"query", query_text}}}};
        };
        const std::string literal_value = "A <|close|>argument<|sep|> B <|close|>call<|sep|> C";
        const std::string json_value = "{\"query\":\"a \\\"b\\\" <|close|>json<|sep|> c\"}";
        const std::vector<tool_case> tool_cases = {
            {"tagged-string", {{opened + tools_open + call1 + query, true}, {"Kimi K3 发布", false},
                {arg_end + call_end + tail, true}}, "", json::array({call("Kimi K3 发布")})},
            {"tagged-literal-markers", {{opened + tools_open + call1 + query, true}, {literal_value, false},
                {arg_end + call_end + tail, true}}, "", json::array({call(literal_value)})},
            {"typed-number", {{opened + tools_open + call1 + query, true}, {"Kimi", false},
                {arg_end + "<|open|>argument key=\"max_results\" type=\"number\"<|sep|>", true}, {"3", false},
                {arg_end + call_end + tail, true}}, "",
                json::array({{{"name", "web_search"}, {"arguments", {{"query", "Kimi"}, {"max_results", 3}}}}})},
            {"json-block-literal", {{opened + tools_open + call1 + "<|open|>json type=\"object\"<|sep|>", true},
                {json_value, false}, {"<|close|>json<|sep|>" + call_end + tail, true}}, "",
                json::array({call("a \"b\" <|close|>json<|sep|> c")})},
            {"parallel-calls", {{opened + tools_open + call1 + query, true}, {"A", false},
                {arg_end + call_end + call2 + query, true}, {"B", false}, {arg_end + call_end + tail, true}}, "",
                json::array({call("A"), call("B")})},
            {"content-then-call", {{opened, true}, {"我先搜索一下。", false}, {tools_open + call1 + query, true},
                {"天气", false}, {arg_end + call_end + tail, true}}, "我先搜索一下。", json::array({call("天气")})},
            {"content-literal-tools-marker", {{opened, true}, {"字面 <|open|>tools<|sep|> 标记", false},
                {"<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>", true}},
                "字面 <|open|>tools<|sep|> 标记", json::array()},
        };
        auto calls_json = [](const common_chat_msg & msg) {
            json out = json::array();
            for (const auto & c : msg.tool_calls) {
                json args;
                try { args = json::parse(c.arguments); } catch (const std::exception &) { args = c.arguments; }
                out.push_back({{"name", c.name}, {"arguments", args}});
            }
            return out;
        };
        for (const auto & item : tool_cases) {
            std::string text;
            llama_tokens ids;
            common_chat_output_source source; source.valid = true;
            std::vector<segment> segments = {{reasoning, false}};
            segments.insert(segments.end(), item.segments.begin(), item.segments.end());
            for (const auto & seg : segments) {
                for (auto id : common_tokenize(vocab, seg.text, false, seg.structural)) {
                    const auto piece = common_token_to_piece(vocab, id, true);
                    const bool control = llama_vocab_is_control(vocab, id);
                    require(seg.structural || !control, "literal region contains a control token ID");
                    if (control && !piece.empty()) source.controls.push_back({text.size(), text.size() + piece.size(), id});
                    text += piece; ids.push_back(id);
                }
            }
            auto correct = [&](const common_chat_msg & msg) {
                return msg.content == item.content && msg.reasoning_content == reasoning && calls_json(msg) == item.calls;
            };
            json row = {{"name", item.name}, {"text", text}, {"expected_content", item.content}, {"expected_calls", item.calls}};
            try {
                task_result_state state(tool_pp);
                server_task_result_cmpl_final final{};
                final.content = text; final.stream = false; final.output_source = source;
                final.update(state);
                row["nonstream"] = {{"content", final.oaicompat_msg.content}, {"reasoning", final.oaicompat_msg.reasoning_content},
                                    {"calls", calls_json(final.oaicompat_msg)}, {"correct", correct(final.oaicompat_msg)}};
            } catch (const std::exception & e) { row["nonstream"] = {{"correct", false}, {"error", e.what()}}; }
            try {
                task_result_state state(tool_pp);
                std::string pending, all, wire_content;
                for (size_t i = 0; i < ids.size(); ++i) {
                    const auto piece = common_token_to_piece(vocab, ids[i], true);
                    all += piece; pending += piece;
                    if (validate_utf8(all) != all.size()) continue;
                    server_task_result_cmpl_partial partial{};
                    partial.content = pending;
                    partial.output_source.valid = true;
                    for (const auto & t : source.controls) if (t.begin < all.size()) partial.output_source.controls.push_back(t);
                    partial.update(state);
                    for (const auto & diff : partial.oaicompat_msg_diffs) wire_content += diff.content_delta;
                    pending.clear();
                }
                server_task_result_cmpl_final final{};
                final.content = pending; final.stream = true; final.output_source = source;
                final.update(state);
                for (const auto & diff : final.oaicompat_msg_diffs) wire_content += diff.content_delta;
                row["stream"] = {{"content", final.oaicompat_msg.content}, {"wire_content", wire_content},
                                 {"calls", calls_json(final.oaicompat_msg)},
                                 {"correct", correct(final.oaicompat_msg) && wire_content == item.content}};
            } catch (const std::exception & e) { row["stream"] = {{"correct", false}, {"error", e.what()}}; }
            tool_rows.push_back(row);
        }
        for (const auto & row : tool_rows) {
            require(row["nonstream"]["correct"] == true && row["stream"]["correct"] == true,
                    "tool section provenance failed: " + row.dump());
        }
    }
    std::cout << json({{"status", "OUTPUT_LITERAL_PARSER_SYNTHETIC_PASS"}, {"model_evaluations", 0}, {"vocab_only", true},
        {"generation_prompt", rendered.generation_prompt}, {"format", common_chat_format_name(rendered.format)}, {"cases", rows},
        {"tool_cases", tool_rows}}).dump(2) << '\n';
}
