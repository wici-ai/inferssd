// SPDX-License-Identifier: Apache-2.0
#include "code-mode.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

// The TypeScript of a declaration's argument type, read as JSON Schema (see server_code_mode_tools). Throws on anything
// else.
struct ts_reader {
    const std::string &      s;
    size_t                   i = 0;
    std::vector<std::string> comments; // the comment lines read since the last member

    explicit ts_reader(const std::string & text) : s(text) {}

    [[noreturn]] void fail(const char * what) const {
        throw std::runtime_error(std::string(what) + " at " + std::to_string(i));
    }

    // past white space and comments, which are kept for the member they come before
    void skip() {
        while (i < s.size()) {
            if (std::isspace((unsigned char) s[i])) {
                ++i;
            } else if (s.compare(i, 2, "//") == 0) {
                size_t end = s.find('\n', i);
                end        = end == std::string::npos ? s.size() : end;
                size_t at  = i + 2;
                at += at < end && s[at] == ' ';
                comments.push_back(s.substr(at, end - at));
                i = end;
            } else {
                break;
            }
        }
    }

    bool peek(char c) {
        skip();
        return i < s.size() && s[i] == c;
    }

    bool eat(char c) {
        if (!peek(c)) {
            return false;
        }
        ++i;
        return true;
    }

    void expect(char c) {
        if (!eat(c)) {
            fail("expected a character");
        }
    }

    std::string identifier() {
        skip();
        const size_t begin = i;
        while (i < s.size() && (std::isalnum((unsigned char) s[i]) || s[i] == '_' || s[i] == '$')) {
            ++i;
        }
        if (i == begin) {
            fail("expected a name");
        }
        return s.substr(begin, i - begin);
    }

    // a string literal, as JSON reads it
    std::string string_literal() {
        skip();
        const size_t begin = i;
        for (++i; i < s.size() && s[i] != '"'; ++i) {
            i += s[i] == '\\';
        }
        if (i >= s.size()) {
            fail("unterminated string");
        }
        ++i;
        return json::parse(s.substr(begin, i - begin)).get<std::string>();
    }

    json type() {
        eat('|');
        std::vector<json> members{ postfix() };
        while (eat('|')) {
            members.push_back(postfix());
        }
        return merge(members);
    }

    json postfix() {
        json t = primary();
        while (peek('[') && s.compare(i, 2, "[]") == 0) {
            i += 2;
            t = json{ { "type", "array" }, { "items", t } };
        }
        return t;
    }

    json primary() {
        skip();
        if (eat('{')) {
            return object();
        }
        if (eat('(')) {
            json t = type();
            expect(')');
            return t;
        }
        if (peek('"')) {
            return json{ { "type", "string" }, { "enum", json::array({ string_literal() }) } };
        }
        if (i < s.size() && (std::isdigit((unsigned char) s[i]) || s[i] == '-')) {
            const size_t begin = i;
            for (++i; i < s.size() && (std::isdigit((unsigned char) s[i]) || s[i] == '.' || s[i] == 'e'); ++i) {
            }
            return json{ { "type", "number" }, { "enum", json::array({ json::parse(s.substr(begin, i - begin)) }) } };
        }
        const std::string name = identifier();
        if (name == "string" || name == "number" || name == "boolean" || name == "null") {
            return json{ { "type", name } };
        }
        if (name == "true" || name == "false") {
            return json{ { "type", "boolean" }, { "enum", json::array({ name == "true" }) } };
        }
        if (name == "unknown" || name == "any") {
            return json::object();
        }
        if (name == "Array") {
            expect('<');
            json items = type();
            expect('>');
            return json{ { "type", "array" }, { "items", items } };
        }
        if (name == "Record") {
            expect('<');
            type();
            expect(',');
            json values = type();
            expect('>');
            return json{ { "type", "object" }, { "additionalProperties", values } };
        }
        fail("unknown type");
    }

    // the members after '{' up to its '}'
    json object() {
        json properties = json::object();
        json required   = json::array();
        json additional = false;
        comments.clear();
        while (!eat('}')) {
            std::vector<std::string> description = std::move(comments);
            comments.clear();
            if (eat('[')) {
                identifier();
                expect(':');
                type();
                expect(']');
                expect(':');
                additional = type();
            } else {
                skip();
                const std::string name     = i < s.size() && s[i] == '"' ? string_literal() : identifier();
                const bool        optional = eat('?');
                expect(':');
                properties[name] = described(type(), description);
                if (!optional) {
                    required.push_back(name);
                }
            }
            if (!eat(';')) {
                eat(',');
            }
            comments.clear();
        }
        return json{ { "type", "object" },
                     { "properties", properties },
                     { "required", required },
                     { "additionalProperties", additional } };
    }

    // the schema with its description after its type, as Codex writes its tools' schemas
    static json described(const json & t, const std::vector<std::string> & lines) {
        if (lines.empty()) {
            return t;
        }
        std::string text;
        for (const auto & line : lines) {
            text += (text.empty() ? "" : "\n") + line;
        }
        json out = json::object();
        if (t.contains("type")) {
            out["type"] = t.at("type");
        }
        out["description"] = text;
        for (const auto & [key, value] : t.items()) {
            if (key != "type") {
                out[key] = value;
            }
        }
        return out;
    }

    // a union of schemas: any value with an unknown member, an enum of literals, a list of plain types, else anyOf
    static json merge(const std::vector<json> & members) {
        if (members.size() == 1) {
            return members[0];
        }
        bool literals = true;
        bool plain    = true;
        for (const json & m : members) {
            if (m.empty()) {
                return json::object();
            }
            const bool literal = m.size() == 2 && m.contains("enum") && m.at("enum").size() == 1;
            const bool null    = m.size() == 1 && m.contains("type") && m.at("type") == "null";
            literals           = literals && (literal || null);
            plain              = plain && m.size() == 1 && m.contains("type") && m.at("type").is_string();
        }
        if (!literals && !plain) {
            return json{ { "anyOf", members } };
        }
        json types  = json::array();
        json values = json::array();
        for (const json & m : members) {
            if (std::find(types.begin(), types.end(), m.at("type")) == types.end()) {
                types.push_back(m.at("type"));
            }
            values.push_back(m.contains("enum") ? m.at("enum")[0] : json());
        }
        json out = json{ { "type", types.size() == 1 ? types[0] : types } };
        if (literals) {
            out["enum"] = values;
        }
        return out;
    }
};

// constant, so that they are set before any other file's static data uses them
constexpr std::string_view declaration_head = "\n\nexec tool declaration:\n```ts\ndeclare const tools: { ";
constexpr std::string_view script_head      = "const r = await tools.";
constexpr std::string_view script_tail =
    ");\n"
    "if (r !== null && typeof r === \"object\" && typeof r.output === \"string\") {\n"
    "  const { output, ...rest } = r;\n"
    "  text(JSON.stringify(rest));\n"
    "  text(output);\n"
    "} else {\n"
    "  text(r);\n"
    "}";

bool is_identifier(const std::string & name) {
    if (name.empty() || std::isdigit((unsigned char) name[0])) {
        return false;
    }
    for (const char c : name) {
        if (!std::isalnum((unsigned char) c) && c != '_' && c != '$') {
            return false;
        }
    }
    return true;
}

// The tool of one "### `name`" section: its text, then its declaration.
bool read_section(const std::string & section, server_code_mode_tool & tool) {
    const size_t name_end = section.find("`\n");
    if (name_end == std::string::npos) {
        return false;
    }
    tool.name = section.substr(0, name_end);
    const size_t text_end = section.find(declaration_head, name_end);
    const std::string call = tool.name + "(";
    if (!is_identifier(tool.name) || text_end == std::string::npos ||
        section.compare(text_end + declaration_head.size(), call.size(), call) != 0) {
        return false;
    }
    tool.description = section.substr(name_end + 2, text_end - name_end - 2);
    ts_reader reader(section);
    reader.i = text_end + declaration_head.size() + call.size();
    try {
        reader.identifier();
        reader.expect(':');
        json t = reader.type();
        reader.expect(')');
        reader.expect(':');
        if (reader.identifier() != "Promise") {
            return false;
        }
        tool.string_input = t == json{ { "type", "string" } };
        if (tool.string_input) {
            tool.parameters = json();
        } else if (t.contains("type") && t.at("type") == "object") {
            tool.parameters = t;
        } else {
            // a union of shapes, or any value: an object of one of them, or any object
            json object = json{ { "type", "object" } };
            for (const auto & [key, value] : t.items()) {
                object[key] = value;
            }
            tool.parameters = object;
        }
    } catch (const std::exception &) {
        return false;
    }
    return true;
}

} // namespace

std::vector<server_code_mode_tool> server_code_mode_tools(const json & exec) {
    if (!exec.is_object() || !exec.contains("type") || exec.at("type") != "custom" || !exec.contains("name") ||
        exec.at("name") != "exec" || !exec.contains("description") || !exec.at("description").is_string()) {
        return {};
    }
    const std::string & description = exec.at("description").get_ref<const std::string &>();
    static const std::string heading = "\n### `";
    std::vector<server_code_mode_tool> tools;
    for (size_t at = description.find(heading); at != std::string::npos;) {
        const size_t begin = at + heading.size();
        const size_t next  = description.find(heading, begin);
        server_code_mode_tool tool;
        const size_t size = next == std::string::npos ? std::string::npos : next - begin;
        if (!read_section(description.substr(begin, size), tool)) {
            return {};
        }
        tools.push_back(std::move(tool));
        at = next;
    }
    return tools;
}

std::string server_code_mode_input(const std::string & name, const std::string & arguments, bool string_input) {
    std::string argument = arguments;
    try {
        const json parsed = json::parse(arguments);
        if (string_input) {
            argument = json(parsed.is_object() && parsed.contains("input") && parsed.at("input").is_string() ?
                                parsed.at("input").get<std::string>() : arguments).dump();
        } else {
            argument = parsed.dump();
        }
    } catch (const std::exception &) {
        // not JSON: as written, which exec then reports as an error
        if (string_input) {
            argument = json(arguments).dump();
        }
    }
    return std::string(script_head) + name + "(" + argument + std::string(script_tail);
}

bool server_code_mode_call(const std::string & input, const std::vector<server_code_mode_tool> & tools,
                           std::string & name, std::string & arguments) {
    if (input.size() < script_head.size() + script_tail.size() ||
        input.compare(0, script_head.size(), script_head) != 0 ||
        input.compare(input.size() - script_tail.size(), script_tail.size(), script_tail) != 0) {
        return false;
    }
    const std::string call  = input.substr(script_head.size(), input.size() - script_head.size() - script_tail.size());
    const size_t      paren = call.find('(');
    if (paren == std::string::npos) {
        return false;
    }
    const std::string called = call.substr(0, paren);
    const auto        tool   = std::find_if(tools.begin(), tools.end(),
                                            [&](const server_code_mode_tool & t) { return t.name == called; });
    if (tool == tools.end()) {
        return false;
    }
    try {
        const json argument = json::parse(call.substr(paren + 1));
        if (tool->string_input ? !argument.is_string() : !argument.is_object()) {
            return false;
        }
        const std::string args = tool->string_input ? json{ { "input", argument } }.dump() : argument.dump();
        if (server_code_mode_input(called, args, tool->string_input) != input) {
            return false;
        }
        name      = called;
        arguments = args;
        return true;
    } catch (const std::exception &) {
        return false;
    }
}
