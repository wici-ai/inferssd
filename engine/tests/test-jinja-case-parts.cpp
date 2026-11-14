// SPDX-License-Identifier: Apache-2.0
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct example {
    const char * name;
    const char * source;
    json variables;
    std::vector<jinja::string_part> expected;
};

static const std::vector<example> examples = {
    {"cap-concat", "{{ ('he' ~ word) | capitalize }}", {{"word", "LLO WORLD"}},
        {{false, "He"}, {true, "llo world"}}},
    {"cap-three", "{{ (prefix ~ 'LLO ' ~ suffix) | capitalize }}", {{"prefix", "he"}, {"suffix", "WORLD"}},
        {{true, "He"}, {false, "llo "}, {true, "world"}}},
    {"cap-space", "{{ (' ' ~ word) | capitalize }}", {{"word", "HELLO"}},
        {{false, " "}, {true, "hello"}}},
    {"cap-constant", "{{ ('he' ~ 'LLO') | capitalize }}", json::object(),
        {{false, "Hello"}}},
    {"cap-repeat", "{{ (('hE' ~ word) * 2) | capitalize }}", {{"word", "LLo"}},
        {{false, "He"}, {true, "llo"}, {false, "he"}, {true, "llo"}}},
    {"cap-macro", "{% macro label(x) %}he{{ x }}{% endmacro %}{{ label(word) | capitalize }}", {{"word", "LLO"}},
        {{false, "He"}, {true, "llo"}}},
    {"title-concat", "{{ ('he' ~ word) | title }}", {{"word", "LLO WORLD"}},
        {{false, "He"}, {true, "llo World"}}},
    {"title-three", "{{ (prefix ~ 'BAR baZ' ~ suffix) | title }}", {{"prefix", "foo"}, {"suffix", "QUX"}},
        {{true, "Foo"}, {false, "bar Baz"}, {true, "qux"}}},
    {"title-join", "{{ ['he', word, 'RE'] | join('') | title }}", {{"word", "LLO THE"}},
        {{false, "He"}, {true, "llo The"}, {false, "re"}}},
    {"title-newline", "{{ ('he' ~ word) | title }}", {{"word", "LLO\nWORLD"}},
        {{false, "He"}, {true, "llo\nWorld"}}},
    {"cap-empty-prefix", "{{ ('' ~ word) | capitalize }}", {{"word", "HELLO"}},
        {{true, "Hello"}}},
    {"cap-single", "{{ word | capitalize }}", {{"word", "hELLO WORLD"}},
        {{true, "Hello world"}}},
    {"title-boundary", "{{ ('hello ' ~ word) | title }}", {{"word", "WORLD"}},
        {{false, "Hello "}, {true, "World"}}},
    {"title-empty", "{{ ('' ~ word ~ '') | title }}", {{"word", ""}},
        {}},
    {"upper", "{{ ('he' ~ word) | upper }}", {{"word", "LLO WORLD"}},
        {{false, "HE"}, {true, "LLO WORLD"}}},
    {"lower", "{{ ('HE' ~ word) | lower }}", {{"word", "LLO WORLD"}},
        {{false, "he"}, {true, "llo world"}}},
};

static void require(bool condition, const char * message) {
    if (!condition) { fprintf(stderr, "%s\n", message); std::exit(1); }
}

static std::string text(const std::vector<jinja::string_part> & parts) {
    std::string result;
    for (const auto & part : parts) { result += part.val; }
    return result;
}

static std::vector<bool> origins(const std::vector<jinja::string_part> & parts) {
    std::vector<bool> result;
    for (const auto & part : parts) { result.insert(result.end(), part.val.size(), part.is_input); }
    return result;
}

int main(int argc, char ** argv) {
    require(argc == 2, "expected a scenario name");
    for (const auto & item : examples) {
        if (std::string(argv[1]) != item.name) { continue; }
        jinja::lexer lexer;
        const auto tokens = lexer.tokenize(item.source);
        auto program = jinja::parse_from_tokens(tokens);
        jinja::context context(item.source);
        jinja::global_from_json(context, item.variables, true);
        jinja::runtime runtime(context);
        const auto rendered = runtime.gather_string_parts(runtime.execute(program))->as_string();
        if (text(rendered.parts) != text(item.expected)) {
            fprintf(stderr, "expected: %s\nactual: %s\n", text(item.expected).c_str(), text(rendered.parts).c_str());
            return 1;
        }
        require(origins(rendered.parts) == origins(item.expected), "template/input byte origins changed");
        return 0;
    }
    require(false, "unknown scenario");
}
