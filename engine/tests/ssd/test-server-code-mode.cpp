// SPDX-License-Identifier: Apache-2.0
// Codex's code mode (tools/server/inferssd/code-mode.h): the tools its exec declares in TypeScript are offered as
// functions, a call of one goes out as an exec call of a fixed script, and that exec call comes back as the call. Runs on
// the "functions" namespace of the ChatGPT desktop app 26.1120 as it sends it on macOS (50957) and on Windows (41243),
// on the tools Codex 0.172 sends itself with a model it does not run in code mode, and on the server's own request and
// output steps with the repository's DeepSeek-V4.1 template; a vocabulary is loaded alone, no model.
#include "server-chat.h"
#include "server-common.h"
#include "server-task.h"
#include "chat.h"
#include "llama-cpp.h"
#include "inferssd/code-mode.h"

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

static json fixture(const std::string & name) {
    return json::parse(read_file("tests/ssd/fixtures/" + name));
}

static const char * const systems[] = { "macos", "windows" };

static std::vector<server_code_mode_tool> declared(const char * system) {
    return server_code_mode_tools(fixture(std::string("chatgpt-functions-") + system + ".json").at("tools").at(0));
}

static const server_code_mode_tool & tool_named(const std::vector<server_code_mode_tool> & tools,
                                                const std::string & name) {
    for (const auto & tool : tools) {
        if (tool.name == name) {
            return tool;
        }
    }
    throw std::runtime_error("no tool " + name);
}

// as the conversion offers a tool that takes an object
static json function_of(const server_code_mode_tool & tool) {
    return { { "name", tool.name }, { "description", tool.description }, { "strict", false },
             { "parameters", tool.parameters } };
}

// Every "### `name`" section of exec's description is a tool, in their order.
static void test_reads_the_declarations() {
    for (const char * system : systems) {
        const json        exec        = fixture(std::string("chatgpt-functions-") + system + ".json").at("tools").at(0);
        const std::string description = exec.at("description").get<std::string>();
        std::vector<std::string> sections;
        for (size_t at = description.find("\n### `"); at != std::string::npos; at = description.find("\n### `", at + 1)) {
            sections.push_back(description.substr(at + 6, description.find('`', at + 6) - at - 6));
        }
        const auto tools = server_code_mode_tools(exec);
        require(sections.size() == (std::string(system) == "macos" ? 47u : 46u) && tools.size() == sections.size(),
                std::string(system) + ": a tool for each section: " + std::to_string(tools.size()) + " of " +
                    std::to_string(sections.size()));
        for (size_t k = 0; k < tools.size(); ++k) {
            require(tools[k].name == sections[k], "in their order: " + tools[k].name);
            require(tools[k].string_input == (tools[k].name == "apply_patch") &&
                        (tools[k].string_input ? tools[k].parameters.is_null()
                                               : tools[k].parameters.at("type") == "object"),
                    "apply_patch takes a string, the others an object: " + tools[k].name);
        }
        require(tool_named(tools, "exec_command").description ==
                    "Runs a command in a PTY, returning output or a session ID for ongoing interaction." ||
                    std::string(system) == "windows",
                "a tool's description is its section's text");
        require(tool_named(tools, "mcp__codex_app__archive_worktree").description.rfind("Archive a managed worktree", 0) ==
                    0,
                "the tools of a namespace's section are declared by their full names");
    }
}

// The parameters read from the declarations are the ones Codex 0.172 sends itself, on the same system, for the tools it
// offers as functions with a model it does not run in code mode, but for two things TypeScript does not tell: an
// integer is a number, and an object whose members are all optional lists them as required, none.
static void test_schemas_match_codex() {
    for (const char * system : systems) {
        const auto tools = declared(system);
        size_t compared = 0;
        for (json expected : fixture(std::string("codex-0.172-functions-") + system + ".json")) {
            const std::string name = expected.at("name");
            if (name == "request_user_input") {
                continue; // the app's own, in its namespace functions
            }
            expected.erase("type");
            if (name == "create_goal") {
                expected["parameters"]["properties"]["token_budget"]["type"] = "number";
            }
            if (name == "list_mcp_resources" || name == "list_mcp_resource_templates") {
                json parameters = json::object();
                for (const auto & [key, value] : expected.at("parameters").items()) {
                    if (key == "additionalProperties") {
                        parameters["required"] = json::array();
                    }
                    parameters[key] = value;
                }
                expected["parameters"] = parameters;
            }
            const json derived = function_of(tool_named(tools, name));
            require(derived.dump() == expected.dump(),
                    std::string(system) + ": " + name + " as Codex sends it:\n" + derived.dump() + "\n" + expected.dump());
            ++compared;
        }
        require(compared == 9, "nine tools compared");
    }
}

// a code-mode exec that declares one tool with this argument type
static json exec_declaring(const std::string & name, const std::string & argument) {
    return { { "type", "custom" }, { "name", "exec" },
             { "description", "Run JavaScript code to orchestrate/compose tool calls\n\n### `" + name + "`\nDoes it.\n\n"
                              "exec tool declaration:\n```ts\ndeclare const tools: { " + name + "(" + argument +
                              "): Promise<unknown>; };\n```" } };
}

static json parameters_of(const std::string & argument) {
    const auto tools = server_code_mode_tools(exec_declaring("t", argument));
    require(tools.size() == 1, "the declaration reads: " + argument);
    return tools[0].parameters;
}

// The TypeScript Codex writes from JSON Schema, read back.
static void test_types() {
    require(parameters_of("args: {\n  // The path.\n  // Absolute.\n  path: string;\n  depth?: number;\n  all?: boolean;\n}")
                    .dump() ==
                R"({"type":"object","properties":{"path":{"type":"string","description":"The path.\nAbsolute."},)"
                R"("depth":{"type":"number"},"all":{"type":"boolean"}},"required":["path"],"additionalProperties":false})",
            "comments, optional members, plain types");
    require(parameters_of(R"(args: { mode: "a" | "b"; tags?: Array<string>; ids: number[]; nothing: "x" | null; })")
                    .dump() ==
                R"({"type":"object","properties":{"mode":{"type":"string","enum":["a","b"]},"tags":{"type":"array",)"
                R"("items":{"type":"string"}},"ids":{"type":"array","items":{"type":"number"}},"nothing":{"type":)"
                R"(["string","null"],"enum":["x",null]}},"required":["mode","ids","nothing"],"additionalProperties":false})",
            "enums, arrays, a literal or null");
    require(parameters_of("args: {\n  // Where.\n  target: {\n  // The id.\n  id: string | null;\n};\n  meta?: Record<string, "
                          "unknown>;\n  extra: { [key: string]: number; };\n}")
                    .dump() ==
                R"({"type":"object","properties":{"target":{"type":"object","description":"Where.","properties":{"id":)"
                R"({"type":["string","null"],"description":"The id."}},"required":["id"],"additionalProperties":false},)"
                R"("meta":{"type":"object","additionalProperties":{}},"extra":{"type":"object","properties":{},)"
                R"("required":[],"additionalProperties":{"type":"number"}}},"required":["target","extra"],)"
                R"("additionalProperties":false})",
            "nested objects, records, index signatures");
    require(parameters_of(R"(args: { kind: "a"; x: string; } | { kind: "b"; y?: number; })").dump() ==
                R"({"type":"object","anyOf":[{"type":"object","properties":{"kind":{"type":"string","enum":["a"]},"x":)"
                R"({"type":"string"}},"required":["kind","x"],"additionalProperties":false},{"type":"object","properties":)"
                R"({"kind":{"type":"string","enum":["b"]},"y":{"type":"number"}},"required":["kind"],)"
                R"("additionalProperties":false}]})",
            "a union of shapes");
    require(parameters_of(R"(args: { kind: "a"; } | unknown)").dump() == R"({"type":"object"})",
            "a union with unknown is any object");
    require(parameters_of("args: {}").dump() ==
                R"({"type":"object","properties":{},"required":[],"additionalProperties":false})",
            "no members");
    const auto patch = server_code_mode_tools(exec_declaring("apply_patch", "input: string"));
    require(patch.size() == 1 && patch[0].string_input && patch[0].parameters.is_null(), "a string argument");

    // what does not read so leaves exec as it is
    require(server_code_mode_tools(exec_declaring("t", "args: { when: Date; }")).empty(), "an unknown type");
    require(server_code_mode_tools(exec_declaring("t", "args: { a: string")).empty(), "an open object");
    json other = exec_declaring("t", "args: {}");
    other["name"] = "run";
    require(server_code_mode_tools(other).empty(), "another custom tool");
    other = exec_declaring("t", "args: {}");
    other["type"] = "function";
    require(server_code_mode_tools(other).empty(), "a function named exec");
    other = exec_declaring("t", "args: {}");
    other["description"] = "Run JavaScript code.";
    require(server_code_mode_tools(other).empty(), "an exec that declares no tool");
}

// The script of a call, and the call of a script.
static void test_scripts() {
    const std::vector<server_code_mode_tool> tools = declared("macos");
    const std::string input = server_code_mode_input("exec_command", R"({ "cmd": "ls \"a b\"",  "workdir": "/tmp" })", false);
    require(input == "const r = await tools.exec_command({\"cmd\":\"ls \\\"a b\\\"\",\"workdir\":\"/tmp\"});\n"
                     "if (r !== null && typeof r === \"object\" && typeof r.output === \"string\") {\n"
                     "  const { output, ...rest } = r;\n"
                     "  text(JSON.stringify(rest));\n"
                     "  text(output);\n"
                     "} else {\n"
                     "  text(r);\n"
                     "}",
            "the script: " + input);
    std::string name;
    std::string arguments;
    require(server_code_mode_call(input, tools, name, arguments) && name == "exec_command" &&
                arguments == R"({"cmd":"ls \"a b\"","workdir":"/tmp"})",
            "the script is the call: " + name + " " + arguments);

    const std::string patch = "*** Begin Patch\n*** Add File: a.txt\n+\xc3\xa9\n*** End Patch\n";
    const std::string patch_input = server_code_mode_input("apply_patch", json{ { "input", patch } }.dump(), true);
    require(patch_input.rfind("const r = await tools.apply_patch(\"*** Begin Patch\\n", 0) == 0, "a string tool's script");
    require(server_code_mode_call(patch_input, tools, name, arguments) && name == "apply_patch" &&
                json::parse(arguments) == json{ { "input", patch } },
            "and its call: " + arguments);

    for (const std::string & other : {
             std::string("const r = await tools.exec_command({cmd: \"ls\"});\ntext(r.output);"),
             input.substr(0, input.size() - 1),
             server_code_mode_input("not_declared", R"({"cmd":"ls"})", false),
             server_code_mode_input("exec_command", R"(["ls"])", false),
             server_code_mode_input("apply_patch", R"({"cmd":"ls"})", false),
         }) {
        require(!server_code_mode_call(other, tools, name, arguments), "another script is not a call: " + other);
    }
    require(server_code_mode_input("exec_command", "not json", false).find("tools.exec_command(not json);") !=
                std::string::npos,
            "arguments that are not JSON go into the script as written");
}

static json user_message(const char * text) {
    return json{ { "type", "message" }, { "role", "user" },
                 { "content", json::array({ { { "type", "input_text" }, { "text", text } } }) } };
}

// a request of the ChatGPT desktop app on this system with these items after its tools
static json app_request(const char * system, const json & items, json extra = json::object()) {
    json input = json::array({ { { "type", "additional_tools" }, { "role", "developer" },
                                 { "tools", json::array({ fixture(std::string("chatgpt-functions-") + system + ".json") }) } } });
    for (const json & item : items) {
        input.push_back(item);
    }
    json request = { { "model", "gpt-6.2-sol" }, { "input", input }, { "tool_choice", "auto" },
                     { "parallel_tool_calls", true } };
    for (const auto & [key, value] : extra.items()) {
        request[key] = value;
    }
    return request;
}

static json exec_call(const std::string & id, const std::string & input) {
    return { { "type", "custom_tool_call" }, { "status", "completed" }, { "call_id", id }, { "namespace", "functions" },
             { "name", "exec" }, { "input", input } };
}

static json exec_output(const std::string & id, const std::string & text) {
    return { { "type", "custom_tool_call_output" }, { "call_id", id },
             { "output", json::array({ { { "type", "input_text" }, { "text", text } } }) } };
}

// The conversion offers the declared tools in exec's place and takes the exec calls of their scripts for their calls.
static void test_conversion() {
    const json chat = server_chat_convert_responses_to_chatcmpl(app_request("macos", json::array({ user_message("hi") })));
    std::vector<std::string> names;
    for (const json & tool : chat.at("tools")) {
        names.push_back(tool.at("function").at("name"));
    }
    require(names.size() == 4 + 47 && names[0] == "functions::wait" && names[3] == "functions::send_message_to_user_async" &&
                names[4] == "apply_patch" && names[6] == "exec_command" && names.back() == "mcp__codex_app__wait_threads",
            "the app's functions, then the declared tools, without exec: " + json(names).dump());
    require(chat.at("tools")[6].at("function").dump() == function_of(tool_named(declared("macos"), "exec_command")).dump(),
            "a declared tool as a function");
    const json patch = chat.at("tools")[4].at("function");
    require(patch.at("parameters").at("required") == json::array({ "input" }) && patch.at("strict") == true,
            "a string tool as a custom tool is: " + patch.dump());
    require(chat.at(SERVER_CODE_MODE).at("exec") == "functions::exec" && chat.at(SERVER_CODE_MODE).at("tools").size() == 47 &&
                chat.at(SERVER_CODE_MODE).at("tools").at("apply_patch") == true &&
                chat.at(SERVER_CODE_MODE).at("tools").at("exec_command") == false && !chat.contains(SERVER_CUSTOM_TOOLS),
            "the request names exec and its tools: " + chat.at(SERVER_CODE_MODE).dump());

    // history: the exec call of a declared tool's script is that tool's call; another script stays an exec call
    const std::string ls     = server_code_mode_input("exec_command", R"({"cmd":"ls"})", false);
    const std::string script = "const r = await tools.exec_command({cmd: \"pwd\"});\ntext(r.output);";
    const json history = json::array({ user_message("list my files"), exec_call("call_1", ls),
                                       exec_output("call_1", "{\"exit_code\":0}\na.txt\n"), exec_call("call_2", script),
                                       exec_output("call_2", "/work\n") });
    const json messages = server_chat_convert_responses_to_chatcmpl(app_request("macos", history)).at("messages");
    require(messages.size() == 5, "user, call, output, call, output: " + messages.dump());
    const json & call = messages[1].at("tool_calls")[0];
    require(call.at("id") == "call_1" && call.at("function").at("name") == "exec_command" &&
                call.at("function").at("arguments") == R"({"cmd":"ls"})" && messages[2].at("tool_call_id") == "call_1" &&
                messages[2].at("content")[0].at("text") == "{\"exit_code\":0}\na.txt\n",
            "the call of exec_command and its output: " + messages.dump());
    const json & other = messages[3].at("tool_calls")[0];
    require(other.at("function").at("name") == "functions::exec" &&
                json::parse(other.at("function").at("arguments").get<std::string>()) == json{ { "input", script } },
            "another script is exec's call: " + other.dump());

    // and renders as the model's call of exec_command would
    json function_history = history;
    function_history[1] = { { "type", "function_call" }, { "call_id", "call_1" }, { "name", "exec_command" },
                            { "arguments", R"({"cmd":"ls"})" } };
    function_history[2]["type"] = "function_call_output";
    require(server_chat_convert_responses_to_chatcmpl(app_request("macos", function_history)).at("messages").dump() ==
                messages.dump(),
            "as a function call of exec_command");

    // the declared tools come last, after the MCP servers' tools, with a point where those end
    server_chat_params opt{};
    opt.use_jinja          = true;
    opt.prefill_assistant  = true;
    opt.reasoning_format   = COMMON_REASONING_FORMAT_DEEPSEEK;
    opt.force_pure_content = true;
    opt.tmpls = common_chat_templates_init(/* model= */ nullptr,
            "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}"
            "{% if loop.first and tools %}{% for tool in tools %}<tool>{{ tool.function.name }}</tool>{% endfor %}{% endif %}"
            "<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}");
    json request = app_request("macos", json::array({ { { "type", "message" }, { "role", "developer" },
                                                         { "content", "You are Codex." } },
                                                       user_message("hi") }));
    request["input"][0]["tools"].push_back(json::parse(R"({"type": "namespace", "name": "mcp__docs",
        "description": "Docs.", "tools": [{"type": "function", "name": "search", "description": "Search.",
        "parameters": {"type": "object", "properties": {}}}]})"));
    json body = server_chat_convert_responses_to_chatcmpl(request);
    std::vector<raw_buffer> files;
    const json params = oaicompat_chat_params_parse(body, opt, files);
    const std::string prompt = params.at("prompt").get<std::string>();
    const size_t docs  = prompt.find("<tool>mcp__docs::search</tool>");
    const size_t patch_at = prompt.find("<tool>apply_patch</tool>");
    require(docs != std::string::npos && patch_at == docs + std::string("<tool>mcp__docs::search</tool>").size() &&
                prompt.find("<tool>functions::wait</tool>") < docs,
            "the declared tools after the MCP servers' tools: " + prompt.substr(0, 400));
    bool point = false;
    for (const json & p : json_value(params, "branch_points_chars", json::array())) {
        point = point || p.get<size_t>() == patch_at + 1;
    }
    require(point, "a point where the MCP servers' tools end: " + params.at("branch_points_chars").dump());
    require(params.at(SERVER_CODE_MODE) == body.at(SERVER_CODE_MODE), "the field is passed on");
}

// Requests without a code-mode exec convert as before: an exec that declares no tool is a custom tool.
static void test_other_requests() {
    json ns = fixture("chatgpt-functions-macos.json");
    std::string & description = ns["tools"][0]["description"].get_ref<std::string &>();
    description = description.substr(0, description.find("\n### `"));
    const json request = { { "model", "gpt-6.2-sol" },
                           { "input", json::array({ { { "type", "additional_tools" }, { "role", "developer" },
                                                      { "tools", json::array({ ns }) } },
                                                    user_message("hi"), exec_call("call_1", "text(1)") }) } };
    const json chat = server_chat_convert_responses_to_chatcmpl(request);
    require(chat.at("tools")[0].at("function").at("name") == "functions::exec" && !chat.contains(SERVER_CODE_MODE) &&
                chat.at(SERVER_CUSTOM_TOOLS) == json::array({ "functions::exec" }) &&
                chat.at("messages")[1].at("tool_calls")[0].at("function").at("name") == "functions::exec",
            "an exec without declarations stays a custom tool: " + chat.dump().substr(0, 400));
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

static std::string invoke(const std::string & name, const std::string & parameter, const std::string & value) {
    return "<" + dsml + " invoke name=\"" + name + "\">\n<" + dsml + " parameter name=\"" + parameter +
           "\" string=\"true\">" + value + "</" + dsml + " parameter>\n</" + dsml + " invoke>\n";
}

// The model's reply, as DeepSeek-V4.1 writes it: two calls of exec_command at once and a call of wait.
static std::string reply() {
    return "\n\n<" + dsml + " calls>\n" + invoke("exec_command", "cmd", "ls -la") + invoke("exec_command", "cmd", "pwd") +
           invoke("functions::wait", "cell_id", "7") + "</" + dsml + " calls>";
}

static const std::string ls_script  = server_code_mode_input("exec_command", R"({"cmd":"ls -la"})", false);
static const std::string pwd_script = server_code_mode_input("exec_command", R"({"cmd":"pwd"})", false);

static task_params task_of(const llama_vocab * vocab, bool stream) {
    server_chat_params opt{};
    opt.use_jinja         = true;
    opt.prefill_assistant = true;
    opt.reasoning_format  = COMMON_REASONING_FORMAT_DEEPSEEK;
    opt.tmpls             = common_chat_templates_init(nullptr, read_file("models/templates/deepseek-ai-DeepSeek-V4.1.jinja"));
    opt.reasoning         = common_chat_templates_reasoning(opt.tmpls.get());
    json body = server_chat_convert_responses_to_chatcmpl(app_request(
        "macos", json::array({ user_message("list my files") }), { { "stream", stream }, { "reasoning", { { "effort", "none" } } } }));
    std::vector<raw_buffer> files;
    json data = oaicompat_chat_params_parse(body, opt, files);
    data["stream"] = stream;
    common_params base;
    base.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    const task_params params = server_task::params_from_json_cmpl(vocab, base, 4096, {}, data);
    require(params.code_mode.exec == "functions::exec" && params.code_mode.has("exec_command") &&
                params.code_mode.tools.at("apply_patch"),
            "the task knows exec and its tools");
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

// the three items: an exec call for each call of exec_command, and wait as a function_call
static void check_items(const json & output, const std::string & where) {
    require(output.size() == 3, where + ": three items: " + output.dump());
    for (size_t k = 0; k < 2; ++k) {
        const json & exec = output[k];
        require(exec.at("type") == "custom_tool_call" && exec.at("status") == "completed" &&
                    exec.at("namespace") == "functions" && exec.at("name") == "exec" &&
                    exec.at("input") == (k == 0 ? ls_script : pwd_script) && !exec.contains("arguments"),
                where + ": a call of exec_command is an exec call of its script: " + exec.dump());
    }
    require(output[0].at("call_id") != output[1].at("call_id"), where + ": one exec call each");
    require(output[2].at("type") == "function_call" && output[2].at("name") == "wait" &&
                output[2].at("arguments") == R"({"cell_id":"7"})",
            where + ": wait stays a function_call: " + output[2].dump());
}

static void test_output(const llama_vocab * vocab) {
    const std::string text = reply();
    {
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.params = task_of(vocab, false);
        task_result_state state = task.create_state();
        check_items(final_result(state, false, text).at("output"), "response");
    }

    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.params = task_of(vocab, true);
    for (const size_t piece : { size_t(1), size_t(7), text.size() }) {
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
        std::vector<std::string> opened;
        for (const json & event : events) {
            const json & data = event.at("data");
            if (data.at("type") == "response.output_item.added" && data.at("item").at("type") == "custom_tool_call") {
                require(data.at("item").at("name") == "exec" && data.at("item").at("namespace") == "functions" &&
                            data.at("item").at("input") == "",
                        where + ": a call of exec_command opens as an exec call: " + data.dump());
                opened.push_back(data.at("item").at("call_id"));
            }
            require(data.at("type") != "response.custom_tool_call_input.delta", where + ": no input before the end");
            if (data.at("type") == "response.function_call_arguments.delta") {
                for (const std::string & id : opened) {
                    require(data.at("item_id") != id, where + ": an exec call has no argument deltas");
                }
            }
        }
        require(opened.size() == 2, where + ": two exec calls opened");

        const json done = final_result(state, true, "");
        std::vector<json> items;
        std::vector<std::string> inputs;
        for (const json & event : done) {
            const json & data = event.at("data");
            if (data.at("type") == "response.custom_tool_call_input.delta") {
                require(data.at("item_id") == opened[inputs.size()], where + ": the input belongs to its call");
                inputs.push_back(data.at("delta"));
            } else if (data.at("type") == "response.output_item.done") {
                items.push_back(data.at("item"));
                if (items.back().at("type") == "custom_tool_call") {
                    require(inputs.size() == items.size() && inputs.back() == items.back().at("input"),
                            where + ": the whole input before its item is done");
                }
            }
        }
        check_items(items, where + ", output_item.done");
        require(items[0].at("call_id") == opened[0] && items[1].at("call_id") == opened[1], where + ": the opened calls");
        check_items(done.back().at("data").at("response").at("output"), where + ", response.completed");
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
        test_reads_the_declarations();
        test_schemas_match_codex();
        test_types();
        test_scripts();
        test_conversion();
        test_other_requests();
        vocabulary vocab(argv[1]);
        test_output(llama_model_get_vocab(vocab.model.get()));
        std::printf("code mode: ok\n");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        rc = 1;
    }
    llama_backend_free();
    return rc;
}
