#include "server-chat.h"
#include "server-common.h"
#include "inferssd/app-versions.h"
#include "inferssd/code-mode.h"
#include "inferssd/identifiers.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <sstream>
#include <unordered_set>

// The text of a Responses tool output: a string as is, or the text parts of an array joined by newlines.
static std::string responses_output_text(const json & output) {
    if (output.is_string()) {
        return output.get<std::string>();
    }
    std::string text;
    for (const json & part : output) {
        const std::string type = json_value(part, "type", std::string());
        if (!text.empty()) {
            text += "\n";
        }
        text += type == "input_text" || type == "output_text" ? json_value(part, "text", std::string())
                                                              : "[Unsupported " + type + "]";
    }
    return text;
}

// The texts of the parts of a system or developer message that holds only text; empty parts are left out.
static std::optional<std::vector<std::string>> responses_system_parts(const json & message) {
    const std::string role = json_value(message, "role", std::string());
    if (role != "system" && role != "developer") {
        return std::nullopt;
    }
    const json content = json_value(message, "content", json());
    if (content.is_string()) {
        const std::string text = content.get<std::string>();
        return text.empty() ? std::vector<std::string>() : std::vector<std::string>{ text };
    }
    if (!content.is_array()) {
        return std::nullopt;
    }
    std::vector<std::string> parts;
    for (const json & part : content) {
        if (json_value(part, "type", std::string()) != "text") {
            return std::nullopt;
        }
        std::string part_text = json_value(part, "text", std::string());
        if (!part_text.empty()) {
            parts.push_back(std::move(part_text));
        }
    }
    return parts;
}

static std::string join_texts(const std::vector<std::string> & texts) {
    std::string joined;
    for (const auto & text : texts) {
        joined += (joined.empty() ? "" : "\n\n") + text;
    }
    return joined;
}

// Whether a part of Codex's leading developer messages depends on the user, the computer or the session rather than
// on the app: its display mode and the folders it may write (app-context), the skills and plugins installed and the
// account's (skills_instructions), the sandbox and approval settings and the commands approved
// (permissions instructions), the mode the user chose (collaboration_mode) and the tools the account defers (tools).
// The app's instructions before the first of them are the same for every user of an app version.
static bool responses_user_context(const std::string & text) {
    static const char * const tags[] = { "<app-context>", "<skills_instructions>", "<permissions instructions>",
                                         "<collaboration_mode>", "<tools>" };
    const size_t start = text.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return false;
    }
    for (const char * tag : tags) {
        if (text.compare(start, std::char_traits<char>::length(tag), tag) == 0) {
            return true;
        }
    }
    return false;
}

// Codex's sub-agent tool lists the OpenAI models a sub-agent may run on, with their reasoning efforts and service
// tiers. With this server the sub-agent runs on the same model whatever it names, and the list changes with the
// models Codex knows of (Codex 0.160 listed gpt-6.1-sol first, 0.172 gpt-6.2-sol), changing the tools every user of
// the app shares. It is replaced by a sentence that says so; the rest of the description stays.
static std::string responses_tool_without_model_list(const std::string & description) {
    static const std::string header = "Available model overrides (optional; inherited parent model is preferred):\n";
    const size_t begin = description.find(header);
    if (begin == std::string::npos) {
        return description;
    }
    // the list: one line "- `<model>`: ..." for each model
    size_t end = begin + header.size();
    while (description.compare(end, 3, "- `") == 0) {
        const size_t line_end = description.find('\n', end);
        end = line_end == std::string::npos ? description.size() : line_end + 1;
    }
    if (end == begin + header.size()) {
        return description;
    }
    std::string out = description.substr(0, begin);
    out += "Model overrides: none; the new agent runs on the parent's model.";
    if (description[end - 1] == '\n') {
        out += '\n';
    }
    out += description.substr(end);
    return out;
}

// A custom tool takes free-form text instead of JSON arguments (the ChatGPT app's exec, which runs the JavaScript the
// model writes in Codex's code mode; Codex's apply_patch). It is offered as a function of one string argument,
// "input"; its format (a grammar or plain text) has no chat-completions form. The Responses output turns a call of it
// back into a custom_tool_call (SERVER_CUSTOM_TOOLS).
static json responses_custom_tool_function(json tool) {
    tool.erase("type");
    tool.erase("format");
    tool["parameters"] = json {
        {"type", "object"},
        {"properties", json {
            {"input", json {
                {"type",        "string"},
                {"description", "The raw input text the tool's description asks for, not JSON-encoded."},
            }},
        }},
        {"required",             json::array({"input"})},
        {"additionalProperties", false},
    };
    return tool;
}

// The text of the JSON string that starts at `pos` (after its opening quote) as far as it is written: up to its
// closing quote, or up to an escape sequence that is not whole yet (a surrogate pair counts as one).
static std::string json_string_written(const std::string & s, size_t pos) {
    const auto hex4 = [&](size_t at, uint32_t & cp) {
        if (at + 4 > s.size()) {
            return false;
        }
        cp = 0;
        for (size_t i = at; i < at + 4; ++i) {
            const char c = s[i];
            const int  digit = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 :
                               c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
            if (digit < 0) {
                return false;
            }
            cp = (cp << 4) | (uint32_t) digit;
        }
        return true;
    };
    std::string out;
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] != '\\') {
            out += s[pos++];
            continue;
        }
        if (pos + 1 >= s.size()) {
            break;
        }
        const char escape = s[pos + 1];
        if (escape != 'u') {
            static const std::string from = "\"\\/bfnrt";
            static const std::string to   = "\"\\/\b\f\n\r\t";
            const size_t k = from.find(escape);
            if (k == std::string::npos) {
                break;
            }
            out += to[k];
            pos += 2;
            continue;
        }
        uint32_t cp  = 0;
        size_t   len = 6;
        if (!hex4(pos + 2, cp)) {
            break;
        }
        if (cp >= 0xD800 && cp <= 0xDBFF) {
            uint32_t low = 0;
            if (s.compare(pos + 6, 2, "\\u") != 0 || !hex4(pos + 8, low) || low < 0xDC00 || low > 0xDFFF) {
                break;
            }
            cp  = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
            len = 12;
        }
        if (cp < 0x80) {
            out += (char) cp;
        } else if (cp < 0x800) {
            out += (char) (0xC0 | (cp >> 6));
            out += (char) (0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += (char) (0xE0 | (cp >> 12));
            out += (char) (0x80 | ((cp >> 6) & 0x3F));
            out += (char) (0x80 | (cp & 0x3F));
        } else {
            out += (char) (0xF0 | (cp >> 18));
            out += (char) (0x80 | ((cp >> 12) & 0x3F));
            out += (char) (0x80 | ((cp >> 6) & 0x3F));
            out += (char) (0x80 | (cp & 0x3F));
        }
        pos += len;
    }
    return out;
}

std::string server_chat_custom_tool_input(const std::string & arguments, bool partial) {
    if (!partial) {
        try {
            const json args = json::parse(arguments);
            if (args.is_object() && args.contains("input") && args.at("input").is_string()) {
                return args.at("input").get<std::string>();
            }
        } catch (const std::exception &) {
            // not JSON: the arguments as they are
        }
        return arguments;
    }
    size_t pos = 0;
    const auto take = [&](const char * token) {
        while (pos < arguments.size() && std::isspace((unsigned char) arguments[pos])) {
            ++pos;
        }
        const size_t n = std::char_traits<char>::length(token);
        if (arguments.compare(pos, n, token) != 0) {
            return false;
        }
        pos += n;
        return true;
    };
    if (!take("{") || !take("\"input\"") || !take(":") || !take("\"")) {
        return std::string();
    }
    return json_string_written(arguments, pos);
}

// Takes out of the user's context the data that grows as the user works and returns it, to go at the context's end:
// the command prefixes the user approved (a list at the end of the permission instructions, one more each time the
// user allows a command for good) and the tools the account defers (a part of its own). The sections the ChatGPT
// desktop app adds to its app context when the user picks the detail mode Prose rather than Commands go there too,
// first: Non-technical UI and Writing blocks, which say how to talk to the user and refer to nothing around them. The
// rest of the context then stays the same from one conversation and one mode to the next, and the server keeps a
// state where the data starts. Only these move, each with the line that says what it is; the rest keeps its order.
static std::vector<std::string> responses_move_context_data(std::vector<std::string> & context_texts) {
    std::vector<std::string> data;
    static const char * const prose_sections[] = { "\n\n### Non-technical UI\n", "\n\n### Writing blocks\n" };
    for (auto & text : context_texts) {
        const size_t start = text.find_first_not_of(" \t\r\n");
        if (start == std::string::npos || text.compare(start, 13, "<app-context>") != 0) {
            continue;
        }
        for (const char * heading : prose_sections) {
            const size_t begin = text.find(heading);
            if (begin == std::string::npos) {
                continue;
            }
            // the section runs to the next heading after a blank line, to the line that closes the app context or to
            // the end of the text (Writing blocks comes after the closing line)
            size_t end = std::min(text.find("\n\n#", begin + 2), text.find("\n</app-context>", begin + 2));
            if (end == std::string::npos) {
                end = text.size();
            }
            data.push_back(text.substr(begin + 2, end - begin - 2));
            text.erase(begin, end - begin);
        }
    }
    static const std::string approved = "\n\n## Approved command prefixes\n";
    for (auto & text : context_texts) {
        const size_t start = text.find_first_not_of(" \t\r\n");
        if (start == std::string::npos || text.compare(start, 26, "<permissions instructions>") != 0) {
            continue;
        }
        const size_t begin = text.find(approved);
        if (begin == std::string::npos) {
            continue;
        }
        // the list runs to the writable roots, which Codex writes after it, or to the end of the instructions
        size_t end = text.find("\n The writable root", begin + approved.size());
        if (end == std::string::npos) {
            end = text.find("\n</permissions instructions>", begin + approved.size());
        }
        if (end == std::string::npos) {
            end = text.size();
        }
        data.push_back(text.substr(begin + 2, end - begin - 2));
        text.erase(begin, end - begin);
    }
    for (auto it = context_texts.begin(); it != context_texts.end();) {
        const size_t start = it->find_first_not_of(" \t\r\n");
        if (start != std::string::npos && it->compare(start, 33, "<tools>\nDeferred tool namespaces:") == 0) {
            data.push_back(*it);
            it = context_texts.erase(it);
        } else {
            ++it;
        }
    }
    return data;
}

// Codex sends its instructions and several developer messages at the start, another developer message at the start
// of a later turn, and now and then one inside a tool-call loop. Chat templates often take one system message at
// the start and raise on the others or drop them. As CC Switch's Codex proxy converts them for Chat Completions,
// with the parts that depend on the user after everything that does not:
// - the leading ones (the instructions and the first developer messages) become two system messages: the parts
//   before the first that depends on the user (responses_user_context), then that one and all after it. No part
//   moves: the app's later parts may refer to earlier ones (a mode that ends what earlier parts allowed), so the
//   parts keep the order sent and only the cut between the two messages is new. The tools follow the first message
//   in the templates that put them in the system turn after the text (Gemma 4), and come before both in those that
//   put them first (Qwen3.6), so the parts that depend on the user come after the app's instructions and the tools
//   either way, and the prompt of every user of the app starts with the same instructions and tools (see the shared
//   prefix in oaicompat_chat_params_parse). Templates that take more than one system message take two (Qwen3.6
//   drops a third);
// - a later one whose next message other than a system message is a user message becomes a user message where it
//   stands, so the request sent before stays a prefix of this one;
// - any other later one joins the user's context, so a tool-call loop stays one turn and the app's part is the same.
// Texts are joined by a blank line; an empty one is dropped. Returns whether the second system message is the user's
// context; *data_chars is then the length of the data at its end (responses_move_context_data).
static bool responses_collapse_system_messages(std::vector<json> & messages, size_t * data_chars) {
    std::vector<bool> followed_by_user(messages.size(), false);
    bool next_is_user = false;
    for (size_t i = messages.size(); i-- > 0;) {
        followed_by_user[i] = next_is_user;
        if (!responses_system_parts(messages[i])) {
            next_is_user = json_value(messages[i], "role", std::string()) == "user";
        }
    }

    std::vector<std::string> app_texts;
    std::vector<std::string> context_texts;
    std::vector<json> rest;
    bool seen_non_system = false;
    bool in_context      = false; // a leading part that depends on the user came: it and all after it are context
    for (size_t i = 0; i < messages.size(); i++) {
        const auto parts = responses_system_parts(messages[i]);
        if (!parts) {
            seen_non_system = true;
            rest.push_back(std::move(messages[i]));
            continue;
        }
        const std::string text = join_texts(*parts);
        if (text.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }
        if (!seen_non_system) {
            for (const auto & part : *parts) {
                if (part.find_first_not_of(" \t\r\n") != std::string::npos) {
                    in_context = in_context || responses_user_context(part);
                    (in_context ? context_texts : app_texts).push_back(part);
                }
            }
        } else if (followed_by_user[i]) {
            rest.push_back({ { "role", "user" }, { "content", text } });
        } else {
            context_texts.push_back(text);
        }
    }

    std::vector<json> out;
    if (!app_texts.empty()) {
        out.push_back({ { "role", "system" }, { "content", join_texts(app_texts) } });
    }
    *data_chars = 0;
    if (!context_texts.empty()) {
        const std::vector<std::string> data = responses_move_context_data(context_texts);
        std::string context = join_texts(context_texts);
        if (!data.empty()) {
            const std::string tail = join_texts(data);
            *data_chars = (context.empty() ? 0 : 2) + tail.size();
            context += (context.empty() ? "" : "\n\n") + tail;
        }
        out.push_back({ { "role", "system" }, { "content", context } });
    }
    const bool user_context = !app_texts.empty() && !context_texts.empty();
    out.insert(out.end(), std::make_move_iterator(rest.begin()), std::make_move_iterator(rest.end()));
    messages = std::move(out);
    return user_context;
}

// The code-mode exec of a Responses request (inferssd/code-mode.h) as the prompt would name it ("functions::exec" in
// its namespace), and the tools it declares, which the prompt offers in its place; none without such an exec. The
// first exec among the request's tools and those of its additional_tools items counts, with its description as the
// tools of Codex versions render it (inferssd/app-versions.h).
struct responses_code_mode {
    std::string                        exec;
    std::vector<server_code_mode_tool> tools;
};

static responses_code_mode responses_code_mode_of(const json & response_body) {
    std::vector<json> tools;
    if (response_body.contains("tools") && response_body.at("tools").is_array()) {
        tools.insert(tools.end(), response_body.at("tools").begin(), response_body.at("tools").end());
    }
    const json input = json_value(response_body, "input", json());
    if (input.is_array()) {
        for (const json & item : input) {
            if (item.is_object() && json_value(item, "type", std::string()) == "additional_tools" &&
                !item.contains("content")) {
                for (const json & tool : json_value(item, "tools", json::array())) {
                    tools.push_back(tool);
                }
            }
        }
    }
    for (json tool : tools) {
        server_unify_codex_tool(tool, server_identifier_env::current().windows);
        const std::string type = json_value(tool, "type", std::string());
        if (type == "namespace") {
            const std::string ns = json_value(tool, "name", std::string());
            for (const json & sub_tool : json_value(tool, "tools", json::array())) {
                if (json_value(sub_tool, "type", std::string()) == "custom" &&
                    json_value(sub_tool, "name", std::string()) == "exec" && !ns.empty()) {
                    return { ns + "::exec", server_code_mode_tools(sub_tool) };
                }
            }
        } else if (type == "custom" && json_value(tool, "name", std::string()) == "exec") {
            return { "exec", server_code_mode_tools(tool) };
        }
    }
    return {};
}

json server_chat_convert_responses_to_chatcmpl(const json & response_body) {
    if (!response_body.contains("input")) {
        throw std::invalid_argument("'input' is required");
    }
    if (!json_value(response_body, "previous_response_id", std::string{}).empty()) {
        throw std::invalid_argument("llama.cpp does not support 'previous_response_id'.");
    }

    const json input_value = response_body.at("input");
    json chatcmpl_body = response_body;
    chatcmpl_body.erase("input");
    const responses_code_mode code_mode = responses_code_mode_of(response_body);
    std::vector<json> chatcmpl_messages;
    std::vector<json> input_tools; // tools that additional_tools input items declare

    if (response_body.contains("instructions")) {
        chatcmpl_messages.push_back({
            {"role",    "system"},
            {"content", json_value(response_body, "instructions", std::string())},
        });
        chatcmpl_body.erase("instructions");
    }

    if (input_value.is_string()) {
        // #responses_create-input-text_input
        chatcmpl_messages.push_back({
            {"role",    "user"},
            {"content", input_value},
        });
    } else if (input_value.is_array()) {
        // #responses_create-input-input_item_list

        static auto exists_and_is_array = [](const json & j, const char * key) -> bool {
            return j.contains(key) && j.at(key).is_array();
        };
        static auto exists_and_is_string = [](const json & j, const char * key) -> bool {
            return j.contains(key) && j.at(key).is_string();
        };

        for (json item : input_value) {
            bool merge_prev = !chatcmpl_messages.empty() && chatcmpl_messages.back().value("role", "") == "assistant";

            if (exists_and_is_string(item, "content")) {
                // #responses_create-input-input_item_list-input_message-content-text_input
                // Only "Input message" contains item["content"]::string
                // After converting item["content"]::string to item["content"]::array,
                // we can treat "Input message" as sum of "Item-Input message" and "Item-Output message"
                item["content"] = json::array({
                    json {
                        {"text", item.at("content")},
                        {"type", "input_text"}
                    }
                });
            }

            if (exists_and_is_array(item, "content") &&
                exists_and_is_string(item, "role") &&
                (item.at("role") == "user" ||
                    item.at("role") == "system" ||
                    item.at("role") == "developer")
            ) {
                // #responses_create-input-input_item_list-item-input_message
                std::vector<json> chatcmpl_content;

                for (const json & input_item : item.at("content")) {
                    const std::string type = json_value(input_item, "type", std::string());

                    if (type == "input_text") {
                        if (!input_item.contains("text")) {
                            throw std::invalid_argument("'Input text' requires 'text'");
                        }
                        chatcmpl_content.push_back({
                            {"text", input_item.at("text")},
                            {"type", "text"},
                        });
                    } else if (type == "input_image") {
                        // While `detail` is marked as required,
                        // it has default value("auto") and can be omitted.

                        if (!input_item.contains("image_url")) {
                            throw std::invalid_argument("'image_url' is required");
                        }
                        chatcmpl_content.push_back({
                            {"image_url", json {
                                {"url", input_item.at("image_url")}
                            }},
                            {"type", "image_url"},
                        });
                    } else if (type == "input_file") {
                        throw std::invalid_argument("'input_file' is not supported by llamacpp at this moment");
                    } else {
                        throw std::invalid_argument("'type' must be one of 'input_text', 'input_image', or 'input_file'");
                    }
                }

                if (item.contains("type")) {
                    item.erase("type");
                }
                if (item.contains("status")) {
                    item.erase("status");
                }
                item["content"] = chatcmpl_content;

                chatcmpl_messages.push_back(item);
            } else if (exists_and_is_string(item, "role") &&
                item.at("role") == "assistant" &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "message"
            ) {
                // #responses_create-input-input_item_list-item-output_message
                auto chatcmpl_content = json::array();

                // Handle both string content and array content
                if (item.contains("content") && item.at("content").is_string()) {
                    // String content - convert to text content part
                    chatcmpl_content.push_back({
                        {"text", item.at("content")},
                        {"type", "text"},
                    });
                } else if (exists_and_is_array(item, "content")) {
                    // Array content - process each item
                    for (const auto & output_text : item.at("content")) {
                        const std::string type = json_value(output_text, "type", std::string());
                        if (type == "output_text" || type == "input_text") {
                            // Accept both output_text and input_text (string content gets converted to input_text)
                            if (!exists_and_is_string(output_text, "text")) {
                                throw std::invalid_argument("'Output text' requires 'text'");
                            }
                            chatcmpl_content.push_back({
                                {"text", output_text.at("text")},
                                {"type", "text"},
                            });
                        } else if (type == "refusal") {
                            if (!exists_and_is_string(output_text, "refusal")) {
                                throw std::invalid_argument("'Refusal' requires 'refusal'");
                            }
                            chatcmpl_content.push_back({
                                {"refusal", output_text.at("refusal")},
                                {"type", "refusal"},
                            });
                        } else {
                            throw std::invalid_argument("'type' must be one of 'output_text' or 'refusal'");
                        }
                    }
                }

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!exists_and_is_array(prev_msg, "content")) {
                        prev_msg["content"] = json::array();
                    }
                    auto & prev_content = prev_msg["content"];
                    prev_content.insert(prev_content.end(), chatcmpl_content.begin(), chatcmpl_content.end());
                } else {
                    item.erase("status");
                    item.erase("type");
                    item["content"] = chatcmpl_content;
                    chatcmpl_messages.push_back(item);
                }
            } else if (exists_and_is_string(item, "call_id") &&
                exists_and_is_string(item, "name") &&
                exists_and_is_string(item, "type") &&
                ((item.at("type") == "function_call" && exists_and_is_string(item, "arguments")) ||
                    (item.at("type") == "custom_tool_call" && exists_and_is_string(item, "input")))
            ) {
                // #responses_create-input-input_item_list-item-function_tool_call
                // A call of a namespace tool carries the namespace apart; the prompt names it "namespace::name".
                // A call of a custom tool carries its free-form input, which the model wrote as the argument
                // "input" of the function the tool is offered as (responses_custom_tool_function).
                std::string name = item.at("name");
                if (exists_and_is_string(item, "namespace") && !item.at("namespace").get<std::string>().empty()) {
                    name = item.at("namespace").get<std::string>() + "::" + name;
                }
                json arguments = item.at("type") == "custom_tool_call"
                    ? json(json { {"input", item.at("input")} }.dump())
                    : item.at("arguments");
                // A call of a code-mode exec that runs the script the server writes for a call of a tool exec
                // declares is that tool's call, as the model wrote it (server_code_mode_call).
                std::string called;
                std::string called_arguments;
                if (item.at("type") == "custom_tool_call" && name == code_mode.exec && !code_mode.tools.empty() &&
                    server_code_mode_call(item.at("input").get<std::string>(), code_mode.tools, called,
                                          called_arguments)) {
                    name      = called;
                    arguments = called_arguments;
                }
                json tool_call = {
                    {"function", json {
                        {"arguments", arguments},
                        {"name",      name},
                    }},
                    {"id",   item.at("call_id")},
                    {"type", "function"},
                };

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!exists_and_is_array(prev_msg, "tool_calls")) {
                        prev_msg["tool_calls"] = json::array();
                    }
                    prev_msg["tool_calls"].push_back(tool_call);
                } else {
                    chatcmpl_messages.push_back(json {
                        {"role",       "assistant"},
                        {"tool_calls", json::array({tool_call})}
                    });
                }
            } else if (exists_and_is_string(item, "call_id") && !item.at("call_id").get<std::string>().empty() &&
                (exists_and_is_string(item, "output") || exists_and_is_array(item, "output")) &&
                exists_and_is_string(item, "type") &&
                (item.at("type") == "function_call_output" || item.at("type") == "custom_tool_call_output")
            ) {
                // #responses_create-input-input_item_list-item-function_tool_call_output
                // The output of a custom tool takes the same form.
                if (item.at("output").is_string()) {
                    chatcmpl_messages.push_back(json {
                        {"content",      item.at("output")},
                        {"role",         "tool"},
                        {"tool_call_id", item.at("call_id")},
                    });
                } else {
                    json chatcmpl_outputs = item.at("output");
                    for (json & chatcmpl_output : chatcmpl_outputs) {
                        const std::string part_type = json_value(chatcmpl_output, "type", std::string());
                        if (part_type == "input_text" || part_type == "output_text") {
                            chatcmpl_output["type"] = "text";
                        } else {
                            // A tool message holds text only; an image a tool returns (Codex's view_image) is
                            // replaced by the placeholder DeepSeek-V4.1's encoding uses for unsupported blocks.
                            chatcmpl_output = json {
                                {"type", "text"},
                                {"text", "[Unsupported " + part_type + "]"},
                            };
                        }
                    }
                    chatcmpl_messages.push_back(json {
                        {"content",      chatcmpl_outputs},
                        {"role",         "tool"},
                        {"tool_call_id", item.at("call_id")},
                    });
                }
            } else if ((exists_and_is_string(item, "output") || exists_and_is_array(item, "output")) &&
                exists_and_is_string(item, "type") &&
                (item.at("type") == "function_call_output" || item.at("type") == "custom_tool_call_output")
            ) {
                // An output without a call_id answers no call: Codex adds context this way, such as the ChatGPT
                // app's snapshot of the open Page from the second turn on. A chat template takes a tool message
                // only after a tool call, so it goes in as user text under the marker CC Switch's Codex proxy uses.
                std::string name = json_value(item, "name", std::string());
                const std::string ns = json_value(item, "namespace", std::string());
                if (!ns.empty() && !name.empty()) {
                    name = ns + "::" + name;
                }
                const std::string text = "[tool_result without a matching tool_call" +
                    (name.empty() ? std::string() : ": " + name) + "]\n" + responses_output_text(item.at("output"));
                chatcmpl_messages.push_back(json {
                    {"role",    "user"},
                    {"content", json::array({ json {
                        {"text", text},
                        {"type", "text"},
                    }})},
                });
            } else if (exists_and_is_string(item, "type") && item.at("type") == "additional_tools") {
                // Codex sends the tools of some models (those it runs on Responses Lite) as this input item
                // instead of the tools field; they join the request's tools.
                for (const json & tool : json_value(item, "tools", json::array())) {
                    input_tools.push_back(tool);
                }
            } else if (exists_and_is_array(item, "summary") &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "reasoning") {
                // #responses_create-input-input_item_list-item-reasoning
                // Only the reasoning text can go back into the prompt; an item without it (a summary or
                // encrypted content only) has nothing to carry over.
                if (!exists_and_is_array(item, "content") || item.at("content").empty() ||
                    !exists_and_is_string(item.at("content")[0], "text")) {
                    continue;
                }

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    prev_msg["reasoning_content"] = item.at("content")[0].at("text");
                } else {
                    chatcmpl_messages.push_back(json {
                        {"role", "assistant"},
                        {"content", json::array()},
                        {"reasoning_content", item.at("content")[0].at("text")},
                    });
                }
            } else {
                throw std::invalid_argument("Cannot determine type of 'item'");
            }
        }
    } else {
        throw std::invalid_argument("'input' must be a string or array of objects");
    }

    size_t context_data_chars = 0;
    if (responses_collapse_system_messages(chatcmpl_messages, &context_data_chars)) {
        // read and removed by oaicompat_chat_params_parse: the shared prefix ends with the first system message
        chatcmpl_body[SERVER_USER_CONTEXT_SYSTEM_MESSAGE] = true;
        if (context_data_chars > 0) {
            chatcmpl_body[SERVER_USER_CONTEXT_DATA_CHARS] = context_data_chars;
        }
    }
    chatcmpl_body["messages"] = std::move(chatcmpl_messages);

    if (response_body.contains("tools") && !response_body.at("tools").is_array()) {
        throw std::invalid_argument("'tools' must be an array of objects");
    }
    if (response_body.contains("tools") || !input_tools.empty()) {
        std::vector<json> resp_tools;
        if (response_body.contains("tools")) {
            resp_tools.insert(resp_tools.end(), response_body.at("tools").begin(), response_body.at("tools").end());
        }
        resp_tools.insert(resp_tools.end(), input_tools.begin(), input_tools.end());
        // every request of Codex offers the MCP tools the apps of a computer add (see inferssd/app-versions.h)
        server_add_codex_common_tools(resp_tools, server_identifier_env::current().windows);

        // A tool keeps its first definition: Codex declares its additional_tools again in later turns.
        std::vector<json> chatcmpl_tools;
        std::unordered_set<std::string> tool_names;
        std::vector<std::string> custom_tools; // see SERVER_CUSTOM_TOOLS
        const auto add_tool = [&](json function, bool custom) {
            const std::string name = json_value(function, "name", std::string());
            if (!tool_names.insert(name).second) {
                return;
            }
            if (custom) {
                custom_tools.push_back(name);
            }
            if (function.contains("description") && function.at("description").is_string()) {
                function["description"] = responses_tool_without_model_list(function.at("description").get<std::string>());
            }
            if (!function.contains("strict")) {
                function["strict"] = true;
            }
            chatcmpl_tools.push_back(json {
                {"type",     "function"},
                {"function", function},
            });
        };
        for (json resp_tool : resp_tools) {
            // the tools of Codex versions render as one (see inferssd/app-versions.h)
            server_unify_codex_tool(resp_tool, server_identifier_env::current().windows);
            const std::string type = json_value(resp_tool, "type", std::string());
            if (type == "namespace") {
                // A namespace groups function and custom tools (Codex: sub-agents, one per MCP server; the
                // ChatGPT app: exec in "functions"). Offer each as "namespace::name", the qualified name
                // DeepSeek-V4.1's encoding uses, with the namespace's description ahead of its own; the Responses
                // output splits the name again.
                const std::string ns      = json_value(resp_tool, "name", std::string());
                const std::string ns_desc = json_value(resp_tool, "description", std::string());
                for (json sub_tool : json_value(resp_tool, "tools", json::array())) {
                    const std::string sub_type = json_value(sub_tool, "type", std::string());
                    if (ns.empty() || (sub_type != "function" && sub_type != "custom")) {
                        LOG_WRN("Responses request: ignoring a tool in namespace '%s'\n", ns.c_str());
                        continue;
                    }
                    if (sub_type == "custom" && !code_mode.tools.empty() &&
                        ns + "::" + json_value(sub_tool, "name", std::string()) == code_mode.exec) {
                        continue; // the tools it declares take its place, below
                    }
                    if (sub_type == "custom") {
                        sub_tool = responses_custom_tool_function(std::move(sub_tool));
                    }
                    sub_tool.erase("type");
                    sub_tool["name"] = ns + "::" + json_value(sub_tool, "name", std::string());
                    if (!ns_desc.empty()) {
                        sub_tool["description"] = ns_desc + "\n" + json_value(sub_tool, "description", std::string());
                    }
                    add_tool(sub_tool, sub_type == "custom");
                }
                continue;
            }
            if (type == "custom" && !code_mode.tools.empty() &&
                json_value(resp_tool, "name", std::string()) == code_mode.exec) {
                continue; // the tools it declares take its place, below
            }
            if (type == "custom") {
                add_tool(responses_custom_tool_function(std::move(resp_tool)), true);
                continue;
            }
            if (type != "function") {
                // Hosted tools (web_search, image_generation, ...) have no chat-completions form. Clients such as
                // Codex always send some; serve the request with its function and custom tools.
                LOG_WRN("Responses request: ignoring a tool of type '%s'\n", type.c_str());
                continue;
            }
            resp_tool.erase("type");
            add_tool(resp_tool, false);
        }
        // The tools a code-mode exec declares, each as a function: a string tool (apply_patch) as a custom tool is.
        // A tool of the same name the request offers otherwise keeps its own definition.
        json code_mode_tools = json::object();
        for (const server_code_mode_tool & tool : code_mode.tools) {
            if (tool_names.count(tool.name) > 0) {
                continue;
            }
            const json named = { { "name", tool.name }, { "description", tool.description } };
            json function    = named;
            if (tool.string_input) {
                function = responses_custom_tool_function(named);
            } else {
                function["strict"]     = false;
                function["parameters"] = tool.parameters;
            }
            add_tool(function, false);
            code_mode_tools[tool.name] = tool.string_input;
        }
        if (!code_mode_tools.empty()) {
            chatcmpl_body[SERVER_CODE_MODE] = { { "exec", code_mode.exec }, { "tools", code_mode_tools } };
        }
        if (!custom_tools.empty()) {
            chatcmpl_body[SERVER_CUSTOM_TOOLS] = custom_tools;
        }
        chatcmpl_body.erase("tools");
        if (!chatcmpl_tools.empty()) {
            chatcmpl_body["tools"] = chatcmpl_tools;
        } else {
            chatcmpl_body.erase("tool_choice");
            chatcmpl_body.erase("parallel_tool_calls");
        }
    }

    if (response_body.contains("max_output_tokens")) {
        chatcmpl_body.erase("max_output_tokens");
        chatcmpl_body["max_tokens"] = response_body["max_output_tokens"];
    }

    // reasoning.effort becomes the chat-completions field reasoning_effort, which switches thinking per request.
    const json reasoning = json_value(response_body, "reasoning", json::object());
    if (reasoning.is_object() && reasoning.contains("effort") && !reasoning.at("effort").is_null()) {
        chatcmpl_body["reasoning_effort"] = reasoning.at("effort");
    }
    chatcmpl_body.erase("reasoning");

    return chatcmpl_body;
}

// Claude Code puts system messages in the conversation too, such as its system reminders after the user's message.
// Chat templates often take one system message at the start and raise on the others or drop them. As CC Switch
// converts Claude requests for models that take no system message mid-conversation
// (demote_mid_conversation_system_messages): the leading ones stay, a later one becomes a user message where it
// stands, so the request stays append-only across turns, and one between an assistant's tool calls and their
// results moves right after the results. An empty one is dropped.
static void anthropic_demote_mid_conversation_system_messages(json & messages) {
    const auto is_role = [](const json & message, const char * role) {
        return json_value(message, "role", std::string()) == role;
    };
    size_t leading = 0;
    while (leading < messages.size() && is_role(messages[leading], "system")) {
        leading++;
    }
    bool later = false;
    for (size_t i = leading; i < messages.size() && !later; i++) {
        later = is_role(messages[i], "system");
    }
    if (!later) {
        return;
    }

    json out = json::array();
    for (size_t i = 0; i < leading; i++) {
        out.push_back(std::move(messages[i]));
    }
    std::vector<json> deferred;
    for (size_t i = leading; i < messages.size(); i++) {
        json & message = messages[i];
        if (is_role(message, "system")) {
            const json content = json_value(message, "content", json());
            const bool empty   = content.is_null() || (content.is_string() && content.get<std::string>().empty()) ||
                               (content.is_array() && content.empty());
            if (!empty) {
                message["role"] = "user";
                deferred.push_back(std::move(message));
            }
            continue;
        }
        if (!is_role(message, "tool")) {
            for (auto & d : deferred) {
                out.push_back(std::move(d));
            }
            deferred.clear();
        }
        out.push_back(std::move(message));
    }
    for (auto & d : deferred) {
        out.push_back(std::move(d));
    }
    messages = std::move(out);
}

// The text of a converted message that is Claude's environment block: it opens with "# Environment\n", inside a
// <system-reminder> in the Claude desktop app. Empty for any other message.
static std::string anthropic_environment_text(const json & message) {
    const json content = json_value(message, "content", json());
    std::string text;
    if (content.is_string()) {
        text = content.get<std::string>();
    } else if (content.is_array()) {
        for (const json & part : content) {
            if (json_value(part, "type", std::string()) != "text") {
                return std::string();
            }
            text += json_value(part, "text", std::string());
        }
    }
    static const std::string reminder = "<system-reminder>\n";
    const size_t heading = text.compare(0, reminder.size(), reminder) == 0 ? reminder.size() : 0;
    return text.compare(heading, 14, "# Environment\n") == 0 ? text : std::string();
}

// Claude Code and the Claude desktop app send the environment (the working directories, the system, the deferred
// tools, the agents, the skills, the date) as a system message right after the user's first words. Demoted where it
// stands (below), it followed the words, so a new conversation of the same setup shared only the system prompt and
// the tools. When it directly follows the conversation's first user message, its text goes into that message ahead
// of the user's words, after the system reminders that open the message, with a blank line after it; a later system
// message is demoted as before. Returns the number of characters of the message's text (its text parts in order)
// that come before the user's words, where the server keeps a state (SERVER_FIRST_USER_CONTEXT_CHARS); 0 when the
// request has no environment there.
static size_t anthropic_join_environment(json & messages) {
    const auto role_of = [](const json & message) { return json_value(message, "role", std::string()); };
    size_t first = 0;
    while (first < messages.size() && role_of(messages[first]) == "system") {
        first++;
    }
    if (first + 1 >= messages.size() || role_of(messages[first]) != "user" ||
        role_of(messages[first + 1]) != "system") {
        return 0;
    }
    const std::string environment = anthropic_environment_text(messages[first + 1]);
    if (environment.empty() || !messages[first].contains("content") ||
        (!messages[first].at("content").is_string() && !messages[first].at("content").is_array())) {
        return 0;
    }
    json & content = messages[first].at("content");
    const std::string text = environment + "\n\n";
    size_t chars = text.size();
    if (content.is_string()) {
        content = text + content.get<std::string>();
    } else {
        const auto is_text = [](const json & part) {
            return json_value(part, "type", std::string()) == "text" && part.contains("text") &&
                   part.at("text").is_string();
        };
        size_t words = 0;
        while (words < content.size() && is_text(content[words]) &&
               content[words].at("text").get<std::string>().rfind("<system-reminder>", 0) == 0) {
            chars += content[words].at("text").get<std::string>().size();
            words++;
        }
        if (words < content.size() && is_text(content[words])) {
            content[words]["text"] = text + content[words].at("text").get<std::string>();
        } else {
            content.insert(content.begin() + words, json{ { "type", "text" }, { "text", text } });
        }
    }
    messages.erase(messages.begin() + first + 1);
    return chars;
}

json server_chat_convert_anthropic_to_oai(const json & body) {
    json oai_body;

    // Convert system prompt
    json oai_messages = json::array();
    auto system_param = json_value(body, "system", json());
    // the variants of Claude Code's system prompt and tools render as one (see inferssd/app-versions.h)
    const bool claude_code = server_is_claude_code(system_param);
    if (claude_code) {
        server_unify_claude_code_system(system_param);
    }
    if (!system_param.is_null()) {
        std::string system_content;

        // Claude Code's billing line at the start goes in oaicompat_chat_params_parse (see inferssd/identifiers.h).
        if (system_param.is_string()) {
            system_content = system_param.get<std::string>();
        } else if (system_param.is_array()) {
            for (const auto & block : system_param) {
                if (json_value(block, "type", std::string()) == "text") {
                    auto system_text = json_value(block, "text", std::string());
                    system_content += system_text;
                }
            }
        }

        oai_messages.push_back({
            {"role", "system"},
            {"content", system_content}
        });
    }

    // Convert messages
    if (!body.contains("messages")) {
        throw std::runtime_error("'messages' is required");
    }
    const json & messages = body.at("messages");
    if (messages.is_array()) {
        for (const auto & msg : messages) {
            std::string role = json_value(msg, "role", std::string());

            if (!msg.contains("content")) {
                if (role == "assistant") {
                    continue;
                }
                oai_messages.push_back(msg);
                continue;
            }

            const json & content = msg.at("content");

            if (content.is_string()) {
                oai_messages.push_back(msg);
                continue;
            }

            if (!content.is_array()) {
                oai_messages.push_back(msg);
                continue;
            }

            json tool_calls = json::array();
            json converted_content = json::array();
            json tool_results = json::array();
            std::string reasoning_content;
            bool has_tool_calls = false;

            for (const auto & block : content) {
                std::string type = json_value(block, "type", std::string());

                if (type == "text") {
                    converted_content.push_back(block);
                } else if (type == "thinking") {
                    reasoning_content += json_value(block, "thinking", std::string());
                } else if (type == "image") {
                    json source = json_value(block, "source", json::object());
                    std::string source_type = json_value(source, "type", std::string());

                    if (source_type == "base64") {
                        std::string media_type = json_value(source, "media_type", std::string("image/jpeg"));
                        std::string data = json_value(source, "data", std::string());
                        std::ostringstream ss;
                        ss << "data:" << media_type << ";base64," << data;

                        converted_content.push_back({
                            {"type", "image_url"},
                            {"image_url", {
                                {"url", ss.str()}
                            }}
                        });
                    } else if (source_type == "url") {
                        std::string url = json_value(source, "url", std::string());
                        converted_content.push_back({
                            {"type", "image_url"},
                            {"image_url", {
                                {"url", url}
                            }}
                        });
                    }
                } else if (type == "tool_use") {
                    tool_calls.push_back({
                        {"id", json_value(block, "id", std::string())},
                        {"type", "function"},
                        {"function", {
                            {"name", json_value(block, "name", std::string())},
                            {"arguments", json_value(block, "input", json::object()).dump()}
                        }}
                    });
                    has_tool_calls = true;
                } else if (type == "tool_result") {
                    std::string tool_use_id = json_value(block, "tool_use_id", std::string());

                    auto result_content = json_value(block, "content", json());
                    std::string result_text;
                    if (result_content.is_string()) {
                        result_text = result_content.get<std::string>();
                    } else if (result_content.is_array()) {
                        for (const auto & c : result_content) {
                            if (json_value(c, "type", std::string()) == "text") {
                                result_text += json_value(c, "text", std::string());
                            }
                        }
                    }

                    tool_results.push_back({
                        {"role", "tool"},
                        {"tool_call_id", tool_use_id},
                        {"content", result_text}
                    });
                }
            }

            if (!converted_content.empty() || has_tool_calls || !reasoning_content.empty()) {
                json new_msg = {{"role", role}};
                if (!converted_content.empty()) {
                    new_msg["content"] = converted_content;
                } else if (has_tool_calls || !reasoning_content.empty()) {
                    new_msg["content"] = "";
                }
                if (!tool_calls.empty()) {
                    new_msg["tool_calls"] = tool_calls;
                }
                if (!reasoning_content.empty()) {
                    new_msg["reasoning_content"] = reasoning_content;
                }
                oai_messages.push_back(new_msg);
            }

            for (const auto & tool_msg : tool_results) {
                oai_messages.push_back(tool_msg);
            }
        }
    }

    const size_t first_user_context_chars = anthropic_join_environment(oai_messages);
    anthropic_demote_mid_conversation_system_messages(oai_messages);
    oai_body["messages"] = oai_messages;
    if (first_user_context_chars > 0) {
        // read and removed by oaicompat_chat_params_parse
        oai_body[SERVER_FIRST_USER_CONTEXT_CHARS] = first_user_context_chars;
    }

    // Convert tools
    if (body.contains("tools")) {
        const json & tools = body.at("tools");
        if (tools.is_array()) {
            json oai_tools = json::array();
            for (json tool : tools) {
                if (claude_code) {
                    server_unify_claude_code_tool(tool);
                }
                oai_tools.push_back({
                    {"type", "function"},
                    {"function", {
                        {"name", json_value(tool, "name", std::string())},
                        {"description", json_value(tool, "description", std::string())},
                        {"parameters", tool.contains("input_schema") ? tool.at("input_schema") : json::object()}
                    }}
                });
            }
            oai_body["tools"] = oai_tools;
        }
    }

    // Convert tool_choice
    if (body.contains("tool_choice")) {
        const json & tc = body.at("tool_choice");
        if (tc.is_object()) {
            std::string type = json_value(tc, "type", std::string());
            if (type == "auto") {
                oai_body["tool_choice"] = "auto";
            } else if (type == "any" || type == "tool") {
                oai_body["tool_choice"] = "required";
            }
        }
    }

    // Convert stop_sequences to stop
    if (body.contains("stop_sequences")) {
        oai_body["stop"] = body.at("stop_sequences");
    }

    // Handle max_tokens (required in Anthropic, but we're permissive)
    if (body.contains("max_tokens")) {
        oai_body["max_tokens"] = body.at("max_tokens");
    } else {
        oai_body["max_tokens"] = 4096;
    }

    // Pass through common params
    for (const auto & key : {"temperature", "top_p", "top_k", "stream", "chat_template_kwargs"}) {
        if (body.contains(key)) {
            oai_body[key] = body.at(key);
        }
    }

    // Anthropic's thinking becomes reasoning_effort, which is fitted to what the chat template offers, as on every
    // route: "enabled" or "adaptive" thinks at output_config.effort where it names one, else as the template does by
    // default ("on"); "disabled", or no thinking at all, is "none", as in Anthropic's API, where a request thinks only
    // when it asks to. Claude Code turns thinking off that way for a model it does not know, keeping
    // output_config.effort. A budget stays a budget.
    const json thinking = json_value(body, "thinking", json::object());
    const std::string thinking_type = thinking.is_object() ? json_value(thinking, "type", std::string()) : std::string();
    if (thinking_type == "enabled") {
        int budget_tokens = json_value(thinking, "budget_tokens", 10000);
        oai_body["thinking_budget_tokens"] = budget_tokens;
    }
    if (thinking_type.empty() || thinking_type == "disabled") {
        oai_body["reasoning_effort"] = "none";
    } else if (thinking_type == "enabled" || thinking_type == "adaptive") {
        const json output_config = json_value(body, "output_config", json::object());
        const std::string effort = output_config.is_object() ? json_value(output_config, "effort", std::string()) : std::string();
        oai_body["reasoning_effort"] = effort.empty() ? "on" : effort;
    }

    // Handle Anthropic-specific metadata param
    if (body.contains("metadata")) {
        json metadata = json_value(body, "metadata", json::object());
        std::string user_id = json_value(metadata, "user_id", std::string());
        if (!user_id.empty()) {
            oai_body["__metadata_user_id"] = user_id;
        }
    }

    return oai_body;
}

json server_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff) {
    json delta = json::object();
    if (!diff.reasoning_content_delta.empty()) {
        delta["reasoning_content"] = diff.reasoning_content_delta;
    }
    if (!diff.content_delta.empty()) {
        delta["content"] = diff.content_delta;
    }
    if (diff.tool_call_index != std::string::npos) {
        json tool_call;
        tool_call["index"] = diff.tool_call_index;
        if (!diff.tool_call_delta.id.empty()) {
            tool_call["id"]   = diff.tool_call_delta.id;
            tool_call["type"] = "function";
        }
        if (!diff.tool_call_delta.name.empty() || !diff.tool_call_delta.arguments.empty()) {
            json function = json::object();
            if (!diff.tool_call_delta.name.empty()) {
                function["name"] = diff.tool_call_delta.name;
            }
            if (!diff.tool_call_delta.arguments.empty()) {
                function["arguments"] = diff.tool_call_delta.arguments;
            }
            tool_call["function"] = function;
        }
        delta["tool_calls"] = json::array({ tool_call });
    }
    return delta;
}

json convert_transcriptions_to_chatcmpl(
        const json & inp_body,
        const common_chat_templates * tmpls,
        const std::map<std::string, uploaded_file> & in_files,
        std::vector<raw_buffer> & out_files) {
    // TODO @ngxson : this function may need to be improved in the future
    // handle input files
    out_files.clear();
    auto it = in_files.find("file");
    if (it != in_files.end()) {
        out_files.push_back(it->second.data);
    } else {
        throw std::invalid_argument("No input file found for transcription");
    }

    // handle input data
    std::string prompt          = json_value(inp_body, "prompt", std::string());
    std::string language        = json_value(inp_body, "language", std::string());
    std::string response_format = json_value(inp_body, "response_format", std::string("json"));
    if (response_format != "json") {
        throw std::invalid_argument("Only 'json' response_format is supported for transcription");
    }
    const common_chat_prompt_preset preset = common_chat_get_asr_prompt(tmpls);
    if (prompt.empty()) {
        prompt = preset.user;
    }
    if (!language.empty()) {
        prompt += string_format(" (language: %s)", language.c_str());
    }
    prompt += get_media_marker();

    json messages = json::array();
    if (!preset.system.empty()) {
        messages.push_back({{"role", "system"}, {"content", preset.system}});
    }
    messages.push_back({{"role", "user"}, {"content", prompt}});

    json chatcmpl_body = inp_body; // copy all fields
    chatcmpl_body["messages"] = messages;

    // because input from form-data, everything is string, we need to correct the types here
    std::string stream = json_value(inp_body, "stream", std::string("false"));
    chatcmpl_body["stream"] = stream == "true";

    if (inp_body.contains("max_tokens")) {
        std::string inp = inp_body["max_tokens"].get<std::string>();
        chatcmpl_body["max_tokens"] = std::stoul(inp);
    }

    if (inp_body.contains("temperature")) {
        std::string inp = inp_body["temperature"].get<std::string>();
        chatcmpl_body["temperature"] = std::stof(inp);
    }

    return chatcmpl_body;
}
