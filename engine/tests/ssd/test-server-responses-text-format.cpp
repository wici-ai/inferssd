// SPDX-License-Identifier: Apache-2.0
// The structured output a Responses request asks for in text.format constrains the reply, as response_format does on
// Chat Completions. The ChatGPT desktop app titles a conversation with such a request: a JSON object of a title and a
// description, with the app's tools offered and tool_choice "auto" (its exec declares the app's tools, which the server
// offers in exec's place, inferssd/code-mode.h). Runs the server's request steps on a trimmed copy of that request
// with the repository's Gemma 4 and Qwen3.6 templates, and matches replies against the grammar the server samples
// with; no model is loaded.
#include "server-chat.h"
#include "server-common.h"
#include "chat.h"
#include "../../src/llama-grammar.h"
#include "../../src/unicode.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const std::string & message) {
    if (!ok) { throw std::runtime_error(message); }
}

static std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    require(in.good(), "cannot read " + path + " (the test runs from the engine folder)");
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

// text.format of the ChatGPT desktop app's title request (26.1120, model gpt-6-luna)
static const char * title_format = R"json({"type": "json_schema", "strict": true, "schema": {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
    "properties": {"title": {"type": "string", "minLength": 1, "maxLength": 60},
                   "description": {"type": "string", "minLength": 1}},
    "required": ["title", "description"], "additionalProperties": false}, "name": "codex_output_schema"})json";

// the app's title request, trimmed: its tools (here the function wait and the custom tool exec of its namespace
// functions, which declares the tool exec_command), the instructions, the prompt, and its fields
static json title_request(const json & text) {
    json request = json::parse(R"json({
        "model": "gpt-6-luna", "stream": true, "tool_choice": "auto", "parallel_tool_calls": false,
        "reasoning": {"effort": "low", "context": "all_turns"}, "store": false,
        "include": ["reasoning.encrypted_content"],
        "input": [
          {"type": "additional_tools", "role": "developer", "tools": [
            {"type": "namespace", "name": "functions", "description": "", "tools": [
              {"type": "custom", "name": "exec", "description": "Run JavaScript code to orchestrate/compose tool calls\n\n### `exec_command`\nRuns a command in a PTY, returning output or a session ID for ongoing interaction.\n\nexec tool declaration:\n```ts\ndeclare const tools: { exec_command(args: {\n  // Shell command to execute.\n  cmd: string;\n}): Promise<unknown>; };\n```",
               "format": {"type": "grammar", "syntax": "lark", "definition": "start: /[\\s\\S]+/"}},
              {"type": "function", "name": "wait", "description": "Waits on a yielded exec cell.", "strict": false,
               "parameters": {"type": "object", "properties": {"cell_id": {"type": "string"}},
                              "required": ["cell_id"], "additionalProperties": false}}]}]},
          {"type": "message", "role": "developer",
           "content": [{"type": "input_text", "text": "You are Codex, an agent based on GPT-6."}]},
          {"type": "message", "role": "user", "content": [{"type": "input_text",
           "text": "Generate a concise UI title (up to 36 characters) for this task.\nFill the structured title field with plain text.\n\nUser prompt:\nfix the login redirect"}]}]
    })json");
    if (!text.is_null()) {
        request["text"] = text;
    }
    return request;
}

// The format becomes response_format; everything else converts as before.
static void test_conversion() {
    const json format = json::parse(title_format);
    const json chat   =
        server_chat_convert_responses_to_chatcmpl(title_request({ { "verbosity", "low" }, { "format", format } }));
    json expected = { { "type", "json_schema" }, { "json_schema", format } };
    expected["json_schema"].erase("type");
    require(chat.at("response_format") == expected,
            "text.format becomes response_format: " + chat.dump().substr(0, 300));
    require(!chat.contains("text"), "the text field goes");
    std::vector<std::string> names;
    for (const json & tool : chat.at("tools")) {
        names.push_back(tool.at("function").at("name"));
    }
    require(names == std::vector<std::string>({ "functions::wait", "exec_command" }) &&
                chat.at("tool_choice") == "auto" && chat.contains(SERVER_CODE_MODE),
            "the tools stay offered, those exec declares in its place: " + json(names).dump());

    // a request without text.format converts as it did
    json verbose = server_chat_convert_responses_to_chatcmpl(title_request({ { "verbosity", "low" } }));
    require(verbose.at("text") == json{ { "verbosity", "low" } } && !verbose.contains("response_format"),
            "text without a format stays as it is");
    verbose.erase("text");
    require(verbose.dump() == server_chat_convert_responses_to_chatcmpl(title_request(json())).dump(),
            "and the rest as without text");
    require(server_chat_convert_responses_to_chatcmpl(title_request(json("low"))).at("text") == "low",
            "a text that is no object stays as it is");

    for (const char * type : { "json_object", "text" }) {
        const json other =
            server_chat_convert_responses_to_chatcmpl(title_request({ { "format", { { "type", type } } } }));
        require(other.at("response_format") == json{ { "type", type } }, std::string("the format ") + type);
    }
    for (const json & bad : { json{ { "format", { { "type", "yaml" } } } }, json{ { "format", "json_object" } } }) {
        bool threw = false;
        try {
            server_chat_convert_responses_to_chatcmpl(title_request(bad));
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "an unknown format is refused: " + bad.dump());
    }
}

// whether the grammar takes the text from its start to its end
static bool grammar_takes(const std::string & grammar_text, const std::string & text) {
    llama_grammar * grammar =
        llama_grammar_init_impl(nullptr, grammar_text.c_str(), "root", false, nullptr, 0, nullptr, 0);
    require(grammar != nullptr, "the grammar builds: " + grammar_text);
    bool ok = true;
    for (size_t offset = 0; ok && offset < text.size();) {
        const std::string piece = unicode_cpt_to_utf8(unicode_cpt_from_utf8(text, offset));
        try {
            llama_grammar_accept_token(*grammar, 0, piece);
        } catch (const std::runtime_error &) {
            ok = false;
        }
        ok = ok && !llama_grammar_get_stacks(grammar).empty();
    }
    bool done = false;
    for (const auto & stack : llama_grammar_get_stacks(grammar)) {
        done = done || stack.empty();
    }
    llama_grammar_free_impl(grammar);
    return ok && done;
}

// With the Gemma 4 and Qwen3.6 templates the request renders with its tools and samples with a grammar from the
// start that takes only a JSON object of the schema (Gemma 4 writes it in a ```json block); the reply parses to that
// object as content, with no tool call.
static void test_templates() {
    const json reply_object = { { "title", "Fix login redirect" },
                                { "description", "Fix the login page redirect loop" } };
    const std::string reply = reply_object.dump();
    struct template_case {
        const char *             path;
        std::vector<std::string> takes;
    };
    for (const template_case & t : {
             template_case{ "models/templates/google-gemma-4-31B-it.jinja",
                            { "```json\n" + reply + "\n```",
                              "<|channel>thought\nShort.<channel|>```json" + reply + "```" } },
             template_case{ "models/templates/unsloth-Qwen3.6-35B-A3B.jinja",
                            { "Short.\n</think>\n\n" + reply, "Short.\n</think>\n\n```json\n" + reply + "\n```" } },
         }) {
        server_chat_params opt{};
        opt.use_jinja         = true;
        opt.prefill_assistant = true;
        opt.reasoning_format  = COMMON_REASONING_FORMAT_DEEPSEEK;
        opt.tmpls             = common_chat_templates_init(nullptr, read_file(t.path));
        opt.reasoning         = common_chat_templates_reasoning(opt.tmpls.get());
        json body = server_chat_convert_responses_to_chatcmpl(
            title_request({ { "verbosity", "low" }, { "format", json::parse(title_format) } }));
        std::vector<raw_buffer> files;
        const json params = oaicompat_chat_params_parse(body, opt, files);
        const std::string where   = t.path;
        const std::string grammar = json_value(params, "grammar", std::string());
        require(!grammar.empty() && params.at("grammar_lazy") == false,
                where + ": the reply is sampled with a grammar from its start: " + params.dump().substr(0, 400));
        require(params.at("prompt").get<std::string>().find("functions::wait") != std::string::npos &&
                    params.at("prompt").get<std::string>().find("exec_command") != std::string::npos,
                where + ": the tools are in the prompt");
        // the sampler's grammar starts with the generation prompt, which the server feeds it first
        const std::string prompt_end = json_value(params, "generation_prompt", std::string());
        for (const std::string & text : t.takes) {
            require(grammar_takes(grammar, prompt_end + text), where + ": the grammar takes " + text + "\n" + grammar);
        }
        for (const std::string & text : {
                 std::string("**Title:** Fix login redirect\n\n**Description:** Fix the login page redirect loop"),
                 json{ { "title", "Fix login redirect" } }.dump(),
                 json{ { "title", "" }, { "description", "Fix it" } }.dump(),
                 json{ { "title", std::string(61, 'a') }, { "description", "Fix it" } }.dump(),
                 json{ { "title", "Fix" }, { "description", "Fix it" }, { "tags", "login" } }.dump(),
                 std::string("<tool_call>\n<function=functions::wait>\n<parameter=cell_id>\n7\n</parameter>\n"
                             "</function>\n</tool_call>"),
                 std::string("<|tool_call>call:functions::wait{cell_id:<|\"|>7<|\"|>}<tool_call|>"),
                 std::string("<tool_call>\n<function=exec_command>\n<parameter=cmd>\nls\n</parameter>\n</function>\n"
                             "</tool_call>"),
                 std::string("<|tool_call>call:exec_command{cmd:<|\"|>ls<|\"|>}<tool_call|>"),
             }) {
            require(!grammar_takes(grammar, prompt_end + text), where + ": the grammar refuses " + text);
        }

        common_chat_parser_params parser;
        parser.format            = static_cast<common_chat_format>(params.at("chat_format").get<int>());
        parser.reasoning_format  = COMMON_REASONING_FORMAT_DEEPSEEK;
        parser.generation_prompt = json_value(params, "generation_prompt", std::string());
        parser.parse_tool_calls  = json_value(params, "parse_tool_calls", false);
        parser.parser.load(params.at("chat_parser").get<std::string>());
        const common_chat_msg msg = common_chat_parse(t.takes[0], false, parser);
        require(json::parse(msg.content) == reply_object && msg.tool_calls.empty(),
                where + ": the reply is the object: " + msg.content);
    }
}

int main() {
    try {
        test_conversion();
        test_templates();
        std::printf("responses text format: ok\n");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    return 0;
}
