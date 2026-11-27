// SPDX-License-Identifier: Apache-2.0
// A custom tool of the Responses API takes free-form text instead of JSON arguments: the ChatGPT desktop app's exec
// (Codex's code mode) takes the JavaScript the model writes, and it is the app's only way to run a command. It is
// offered to the model as a function of one string argument, "input"; a call of it in the request
// (custom_tool_call, custom_tool_call_output) goes into the prompt as a call of that function does, and a call the
// model writes comes out as a custom_tool_call item, its input streamed as text. Runs the server's own request and
// output steps on a trimmed copy of the app's exec tool and on the repository's DeepSeek-V4.1 template; a vocabulary
// is loaded alone, no model.
#include "server-chat.h"
#include "server-common.h"
#include "server-task.h"
#include "chat.h"
#include "llama-cpp.h"

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

// The additional_tools item of a ChatGPT desktop app request (26.1120, a GPT model picked): its "functions"
// namespace holds the custom tool exec, here with the first part of its description and none of the tools it declares
// (with them it is Codex's code-mode exec, whose tools take its place: test-server-code-mode), and the function wait.
static const char * app_tools_item = R"json({"type": "additional_tools", "id": "at_1", "role": "developer", "tools": [
  {"type": "namespace", "name": "functions", "description": "", "tools": [
    {"type": "custom", "name": "exec", "description": "Run JavaScript code to orchestrate/compose tool calls\n- Evaluates the provided JavaScript code in a fresh V8 isolate as an async module.\n- All nested tools are available on the global `tools` object, for example `await tools.exec_command(...)`. Tool names are exposed as normalized JavaScript identifiers, for example `await tools.mcp__ologs__get_profile(...)`.\n- Nested tool methods take either a string or an object as their input argument.\n- Nested tools return either an object or a string, based on the description.\n- Runs raw JavaScript -- no Node, no file system, no network access, no console.\n- Accepts raw JavaScript source text, not JSON, quoted strings, or markdown code fences.\n- You may optionally start the tool input with a first-line pragma like `// @exec: {\"yield_time_ms\": 10000, \"max_output_tokens\": 1000}`.\n- `yield_time_ms` asks `exec` to yield early if the script is still running. Defaults to 30000 ms.\n- `max_output_tokens` sets the token budget for direct `exec` results. Defaults to 10000 tokens.\n- When the JS code is fully evaluated, the isolate's lifetime ends and unawaited promises are silently discarded.",
     "format": {"type": "grammar", "syntax": "lark", "definition": "\nstart: pragma_source | plain_source\npragma_source: PRAGMA_LINE NEWLINE SOURCE\nplain_source: SOURCE\n\nPRAGMA_LINE: /[ \\t]*\\/\\/ @exec:[^\\r\\n]*/\nNEWLINE: /\\r?\\n/\nSOURCE: /[\\s\\S]+/\n"}},
    {"type": "function", "name": "wait", "description": "Waits on a yielded `exec` cell and returns new output or completion.\n- Use `wait` only after `exec` returns `Script running with cell ID ...`.\n- `cell_id` identifies the running `exec` cell to resume.\n- `yield_time_ms` controls how long to wait for more output before yielding again. Defaults to 10000 ms.\n- `max_tokens` limits how much new output this wait call returns. Defaults to 10000 tokens.\n- `terminate: true` stops the running cell; false or omitted waits for output.\n- `wait` returns only the new output since the last yield, or the final completion or termination result for that cell.\n- If the cell is still running, `wait` may yield again with the same `cell_id`.\n- If the cell has already finished, `wait` returns the completed result and closes the cell.", "strict": false, "parameters": {"type": "object", "properties": {"cell_id": {"type": "string", "description": "Identifier of the running exec cell."}, "max_tokens": {"type": "number", "description": "Output token budget for this wait call. Defaults to 10000 tokens."}, "terminate": {"type": "boolean", "description": "True stops the running exec cell; false or omitted waits for output."}, "yield_time_ms": {"type": "number", "description": "Wait before yielding more output. Defaults to 10000 ms."}}, "required": ["cell_id"], "additionalProperties": false}}
  ]}
]})json";

// What the model writes as exec's input: JavaScript with quotes, a backslash, new lines and characters beyond ASCII,
// which the arguments hold JSON-escaped.
static const std::string script =
    "// @exec: {\"yield_time_ms\": 10000}\n"
    "const r = await tools.exec_command({cmd: \"ls -la \\\"My Files\\\"\", workdir: \"C:\\\\work\"});\n"
    "text(r.output + \"\\n\" + \"\xc3\xa9 \xe5\x86\xb0 \xf0\x9f\x99\x82\");\n";

static json user_message(const char * text) {
    return json{ { "type", "message" }, { "role", "user" },
                 { "content", json::array({ { { "type", "input_text" }, { "text", text } } }) } };
}

static json responses_request(json input, json extra = json::object()) {
    json request = { { "model", "gpt-6.2-sol" }, { "input", std::move(input) }, { "tool_choice", "auto" },
                     { "parallel_tool_calls", true } };
    for (const auto & [key, value] : extra.items()) {
        request[key] = value;
    }
    return request;
}

// The tool exec becomes the function functions::exec with its description and one string argument; the function
// wait next to it converts as it did without exec; and the request names exec as a custom tool.
static void test_tools() {
    const json converted = server_chat_convert_responses_to_chatcmpl(
        responses_request(json::array({ json::parse(app_tools_item), user_message("list my files") })));
    const json & tools = converted.at("tools");
    require(tools.size() == 2, "exec and wait must both be offered");
    const json exec = tools[0];
    const json expected_parameters = json::parse(R"({"type": "object", "properties": {"input": {"type": "string",
        "description": "The raw input text the tool's description asks for, not JSON-encoded."}},
        "required": ["input"], "additionalProperties": false})");
    require(exec.at("type") == "function" && exec.at("function").at("name") == "functions::exec",
            "exec must be offered as the function functions::exec");
    require(exec.at("function").at("description") ==
                json::parse(app_tools_item).at("tools")[0].at("tools")[0].at("description"),
            "exec must keep its description");
    require(exec.at("function").at("parameters").dump() == expected_parameters.dump(),
            "exec must take the one string argument input: " + exec.dump());
    require(!exec.at("function").contains("format") && exec.at("function").at("strict") == true,
            "exec's format has no chat-completions form");
    require(converted.at(SERVER_CUSTOM_TOOLS).dump() == R"(["functions::exec"])", "exec must be named as custom");

    json without_exec = json::parse(app_tools_item);
    without_exec["tools"][0]["tools"].erase(0);
    const json plain = server_chat_convert_responses_to_chatcmpl(
        responses_request(json::array({ without_exec, user_message("list my files") })));
    require(plain.at("tools").size() == 1 && plain.at("tools")[0].dump() == tools[1].dump(),
            "a function tool must convert as without a custom tool");
    require(!plain.contains(SERVER_CUSTOM_TOOLS), "a request without custom tools must name none");

    // a custom tool among the request's tools (Codex's apply_patch), declared once however often it comes
    const json custom_patch = json::parse(R"({"type": "custom", "name": "apply_patch",
        "description": "Use the `apply_patch` tool to edit files.", "format": {"type": "text"}})");
    const json top = server_chat_convert_responses_to_chatcmpl(responses_request(
        json::array({ json::parse(app_tools_item), user_message("hi"), json::parse(app_tools_item) }),
        { { "tools", json::array({ custom_patch, custom_patch }) } }));
    require(top.at("tools").size() == 3 && top.at("tools")[0].at("function").at("name") == "apply_patch" &&
                top.at("tools")[0].at("function").at("parameters").dump() == expected_parameters.dump(),
            "a custom tool in tools must be offered as a function of input");
    require(top.at(SERVER_CUSTOM_TOOLS).dump() == R"(["apply_patch","functions::exec"])",
            "each custom tool must be named once: " + top.at(SERVER_CUSTOM_TOOLS).dump());
}

// A custom_tool_call and its custom_tool_call_output go into the messages as a function_call with the input as its
// argument input, and its function_call_output, would.
static void test_calls() {
    const json earlier = json::array({
        json::parse(app_tools_item),
        user_message("list my files"),
        { { "type", "message" }, { "role", "assistant" },
          { "content", json::array({ { { "type", "output_text" }, { "text", "Listing them." } } }) } },
    });
    json custom = earlier;
    custom.push_back({ { "type", "custom_tool_call" }, { "id", "ctc_1" }, { "status", "completed" },
                       { "call_id", "call_1" }, { "namespace", "functions" }, { "name", "exec" }, { "input", script } });
    custom.push_back({ { "type", "custom_tool_call_output" }, { "call_id", "call_1" }, { "output", "a.txt\n" } });
    custom.push_back({ { "type", "custom_tool_call" }, { "call_id", "call_2" }, { "namespace", "functions" },
                       { "name", "exec" }, { "input", "text(1)" } });
    custom.push_back({ { "type", "custom_tool_call_output" }, { "call_id", "call_2" },
                       { "output", json::array({ { { "type", "input_text" }, { "text", "1" } } }) } });
    json function = earlier;
    function.push_back({ { "type", "function_call" }, { "id", "fc_1" }, { "status", "completed" },
                         { "call_id", "call_1" }, { "namespace", "functions" }, { "name", "exec" },
                         { "arguments", json{ { "input", script } }.dump() } });
    function.push_back({ { "type", "function_call_output" }, { "call_id", "call_1" }, { "output", "a.txt\n" } });
    function.push_back({ { "type", "function_call" }, { "call_id", "call_2" }, { "namespace", "functions" },
                         { "name", "exec" }, { "arguments", R"j({"input":"text(1)"})j" } });
    function.push_back({ { "type", "function_call_output" }, { "call_id", "call_2" },
                         { "output", json::array({ { { "type", "input_text" }, { "text", "1" } } }) } });

    const json messages = server_chat_convert_responses_to_chatcmpl(responses_request(custom)).at("messages");
    require(messages.dump() == server_chat_convert_responses_to_chatcmpl(responses_request(function)).at("messages").dump(),
            "custom tool calls must convert as function calls with the input as argument");
    require(messages.size() == 5, "expected user, assistant with a call, tool, assistant with a call, tool");
    const json & call = messages[1].at("tool_calls")[0];
    require(messages[1].at("content")[0].at("text") == "Listing them." && call.at("id") == "call_1" &&
                call.at("function").at("name") == "functions::exec" &&
                json::parse(call.at("function").at("arguments").get<std::string>()).at("input") == script,
            "the call must join the assistant's message with the input as its argument: " + messages[1].dump());
    require(messages[2].at("role") == "tool" && messages[2].at("tool_call_id") == "call_1" &&
                messages[2].at("content") == "a.txt\n", "the output must answer the call");
}

// The input written so far of arguments still being written: every prefix of the arguments, cut anywhere, gives a
// prefix of the input, and the whole arguments give it whole.
static void test_input_prefixes() {
    for (const bool ascii : { false, true }) {
        const std::string arguments = json{ { "input", script } }.dump(-1, ' ', ascii);
        std::string previous;
        for (size_t n = 0; n <= arguments.size(); ++n) {
            const std::string input = server_chat_custom_tool_input(arguments.substr(0, n), true);
            require(script.compare(0, input.size(), input) == 0 && input.size() >= previous.size(),
                    "a partial input must grow as a prefix of the input: " + arguments.substr(0, n));
            previous = input;
        }
        require(previous == script, "the whole arguments must give the whole input");
        require(server_chat_custom_tool_input(arguments, false) == script, "finished arguments must give the input");
    }
    require(server_chat_custom_tool_input(" { \"input\" : \"a\\u00e9", true) == "a\xc3\xa9", "spaces and escapes");
    require(server_chat_custom_tool_input("{\"input\":\"\\ud83d", true).empty(), "half a surrogate pair waits");
    require(server_chat_custom_tool_input("{\"cmd\":\"ls\"}", true).empty(), "other arguments give no input");
    require(server_chat_custom_tool_input("{\"cmd\":\"ls\"}", false) == "{\"cmd\":\"ls\"}" &&
                server_chat_custom_tool_input("ls -la", false) == "ls -la",
            "finished arguments without the input must stay as they are");
}

struct vocabulary {
    llama_model_ptr model;

    explicit vocabulary(const char * path) {
        auto mp = llama_model_default_params();
        static ggml_backend_dev_t no_devices[] = { nullptr };
        mp.devices      = no_devices;
        mp.vocab_only   = true;
        mp.n_gpu_layers = 0;
        mp.use_mmap     = false;
        model.reset(llama_model_load_from_file(path, mp));
        require(bool(model), std::string("cannot load the vocabulary ") + path);
    }
};

static const std::string dsml = "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c"; // "｜DSML｜"

// The model's reply: a call of exec with the script and a call of wait, as DeepSeek-V4.1 writes them.
static std::string reply() {
    return "\n\n<" + dsml + " calls>\n<" + dsml + " invoke name=\"functions::exec\">\n<" + dsml +
           " parameter name=\"input\" string=\"true\">" + script + "</" + dsml + " parameter>\n</" + dsml +
           " invoke>\n<" + dsml + " invoke name=\"functions::wait\">\n<" + dsml +
           " parameter name=\"cell_id\" string=\"true\">7</" + dsml + " parameter>\n</" + dsml + " invoke>\n</" +
           dsml + " calls>";
}

// The request of test_calls goes through the server's request steps with DeepSeek-V4.1's template: the prompt offers
// exec with its argument input and holds the earlier call as the template writes a call, and the task knows exec is
// custom. Returns the task's parameters, for the replies in test_output.
static task_params test_prompt(const llama_vocab * vocab, bool stream) {
    server_chat_params opt{};
    opt.use_jinja         = true;
    opt.prefill_assistant = true;
    opt.reasoning_format  = COMMON_REASONING_FORMAT_DEEPSEEK;
    opt.tmpls             = common_chat_templates_init(nullptr, read_file("models/templates/deepseek-ai-DeepSeek-V4.1.jinja"));
    opt.reasoning         = common_chat_templates_reasoning(opt.tmpls.get());

    json input = json::array({ json::parse(app_tools_item), user_message("list my files") });
    input.push_back({ { "type", "custom_tool_call" }, { "call_id", "call_1" }, { "namespace", "functions" },
                      { "name", "exec" }, { "input", script } });
    input.push_back({ { "type", "custom_tool_call_output" }, { "call_id", "call_1" }, { "output", "a.txt\n" } });
    json body = server_chat_convert_responses_to_chatcmpl(
        responses_request(input, { { "stream", stream }, { "reasoning", { { "effort", "none" } } } }));
    std::vector<raw_buffer> files;
    const json parsed = oaicompat_chat_params_parse(body, opt, files);
    const std::string prompt = parsed.at("prompt").get<std::string>();
    require(prompt.find(R"({"name": "functions::exec", "description": "Run JavaScript code)") != std::string::npos &&
                prompt.find(R"("parameters": {"type": "object", "properties": {"input": {"type": "string")") !=
                    std::string::npos,
            "the prompt must offer exec with its argument input");
    require(prompt.find("<" + dsml + " invoke name=\"functions::exec\">\n<" + dsml +
                        " parameter name=\"input\" string=\"true\">" + script + "</" + dsml + " parameter>") !=
                std::string::npos && prompt.find("a.txt") != std::string::npos,
            "the prompt must hold the earlier call with its input as written, and its output");
    require(parsed.at(SERVER_CUSTOM_TOOLS).dump() == R"(["functions::exec"])", "the request must name exec as custom");

    common_params base;
    base.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    json data = parsed;
    data["stream"] = stream;
    const task_params params = server_task::params_from_json_cmpl(vocab, base, 4096, {}, data);
    require(params.custom_tools == std::vector<std::string>{ "functions::exec" } &&
                params.chat_parser_params.parse_tool_calls,
            "the task must know exec is custom");
    return params;
}

static json final_result(task_result_state & state, bool stream, const std::string & content) {
    server_task_result_cmpl_final final;
    final.res_type              = TASK_RESPONSE_TYPE_OAI_RESP;
    final.stream                = stream;
    final.content               = content;
    final.stop                  = STOP_TYPE_EOS;
    final.n_decoded             = 64;
    final.n_prompt_tokens       = 128;
    final.n_prompt_tokens_cache = 0;
    final.oaicompat_model       = "gpt-6.2-sol";
    final.update(state);
    return final.to_json();
}

// The two calls of the output items: exec as a custom_tool_call with the script as input, wait as a function_call.
static void check_items(const json & output, const std::string & where) {
    require(output.size() == 2, where + ": expected two output items: " + output.dump());
    const json & exec = output[0];
    require(exec.at("type") == "custom_tool_call" && exec.at("status") == "completed" && exec.at("input") == script &&
                exec.at("namespace") == "functions" && exec.at("name") == "exec" && !exec.contains("arguments") &&
                exec.at("call_id").get<std::string>().rfind("fc_", 0) == 0,
            where + ": exec must be a custom_tool_call with the script: " + exec.dump());
    const json & wait = output[1];
    require(wait.at("type") == "function_call" && wait.at("arguments") == R"({"cell_id":"7"})" &&
                wait.at("namespace") == "functions" && wait.at("name") == "wait",
            where + ": wait must stay a function_call: " + wait.dump());
    require(wait.dump() == json({ { "type", "function_call" }, { "status", "completed" },
                                  { "arguments", R"({"cell_id":"7"})" }, { "call_id", wait.at("call_id") },
                                  { "namespace", "functions" }, { "name", "wait" } }).dump(),
            where + ": a function_call keeps its form");
}

// A reply that calls exec and wait, whole and streamed in pieces of several sizes.
static void test_output(const llama_vocab * vocab) {
    const std::string text = reply();
    {
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.params = test_prompt(vocab, false);
        task_result_state state = task.create_state();
        check_items(final_result(state, false, text).at("output"), "response");
    }

    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.params = test_prompt(vocab, true);
    for (const size_t piece : { size_t(1), size_t(5), size_t(64), text.size() }) {
        const std::string where = "stream in pieces of " + std::to_string(piece);
        task_result_state state = task.create_state();
        std::vector<json> events;
        int32_t n_decoded = 0;
        for (size_t pos = 0; pos < text.size(); pos += piece) {
            server_task_result_cmpl_partial partial;
            partial.res_type  = TASK_RESPONSE_TYPE_OAI_RESP;
            partial.content   = text.substr(pos, piece);
            partial.n_decoded = ++n_decoded;
            partial.update(state);
            for (const json & event : partial.to_json()) {
                events.push_back(event);
            }
        }
        const json done = final_result(state, true, "");

        std::string exec_id;
        std::string input;
        std::string wait_arguments;
        size_t added = 0;
        for (const json & event : events) {
            const json & data = event.at("data");
            require(event.at("event") == data.at("type"), where + ": an event names its type");
            if (data.at("type") == "response.output_item.added" && data.at("item").at("name") == "exec") {
                require(data.at("item").dump() == json({ { "input", "" }, { "call_id", data.at("item").at("call_id") },
                                                         { "type", "custom_tool_call" }, { "status", "in_progress" },
                                                         { "namespace", "functions" }, { "name", "exec" } }).dump(),
                        where + ": exec must open as a custom_tool_call: " + data.dump());
                exec_id = data.at("item").at("call_id");
                ++added;
            } else if (data.at("type") == "response.output_item.added") {
                require(data.at("item").at("type") == "function_call" && data.at("item").at("name") == "wait",
                        where + ": wait must open as a function_call");
                ++added;
            } else if (data.at("type") == "response.custom_tool_call_input.delta") {
                require(data.at("item_id") == exec_id && !data.at("delta").get<std::string>().empty(),
                        where + ": an input delta belongs to exec");
                input += data.at("delta").get<std::string>();
                require(script.compare(0, input.size(), input) == 0, where + ": the input must stream in order");
            } else if (data.at("type") == "response.function_call_arguments.delta") {
                require(!exec_id.empty() && data.at("item_id") != exec_id, where + ": exec has no argument deltas");
                wait_arguments += data.at("delta").get<std::string>();
            }
        }
        require(added == 2 && input == script, where + ": exec's input must stream as its text: " + input);
        require(wait_arguments == R"({"cell_id":"7"})", where + ": wait's arguments must stream as JSON");

        std::vector<json> items;
        for (const json & event : done) {
            if (event.at("event") == "response.output_item.done") {
                items.push_back(event.at("data").at("item"));
            }
        }
        check_items(items, where + ", output_item.done");
        require(items[0].at("call_id") == exec_id, where + ": the done item must be the opened one");
        check_items(done.back().at("data").at("response").at("output"), where + ", response.completed");

        // on the wire
        const std::string sse = format_oai_resp_sse(json(events));
        require(sse.find("event: response.custom_tool_call_input.delta\ndata: "
                         "{\"type\":\"response.custom_tool_call_input.delta\",\"delta\":") != std::string::npos,
                where + ": the input delta event on the wire");
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <vocabulary gguf>\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    int rc = 0;
    try {
        vocabulary vocab(argv[1]);
        test_tools();
        test_calls();
        test_input_prefixes();
        test_output(llama_model_get_vocab(vocab.model.get()));
        std::printf("responses custom tools: ok\n");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        rc = 1;
    }
    llama_backend_free();
    return rc;
}
