// SPDX-License-Identifier: Apache-2.0
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static void require(bool condition, const char * message) {
    if (!condition) { fprintf(stderr, "%s\n", message); std::exit(1); }
}

struct fixture {
    std::string source;
    json variables = json::object();
    std::vector<jinja::string_part> expected;

    fixture(const std::string & mode, size_t count) {
        if (mode == "join" || mode == "gather") {
            source = mode == "join" ? "{{ values | join('|') }}"
                : "{% for item in values %}{{ item }}{% if not loop.last %}|{% endif %}{% endfor %}";
            variables["values"] = json::array();
            for (size_t i = 0; i < count; ++i) {
                if (i) { expected.push_back({false, "|"}); }
                const std::string value = "item-" + std::to_string(i) + "-\xe4\xb8\xad";
                variables["values"].push_back(value);
                expected.push_back({true, value});
            }
        } else if (mode == "repeat") {
            source = "{{ ('<' ~ word ~ '>') * count }}";
            variables = {{"word", "payload"}, {"count", count}};
            for (size_t i = 0; i < count; ++i) {
                expected.push_back({false, "<"});
                expected.push_back({true, "payload"});
                expected.push_back({false, ">"});
            }
        } else if (mode == "replace") {
            source = "{{ word | replace('X', '_') }}";
            std::string input, output;
            for (size_t i = 0; i < count; ++i) { input += "abX"; output += "ab_"; }
            variables = {{"word", input}};
            expected.push_back({true, output});
        } else if (mode == "slice") {
            source = "{{ word[::-1] }}";
            std::string input, output;
            for (size_t i = 0; i < count; ++i) { input += "abc"; output += "cba"; }
            variables = {{"word", input}};
            expected.push_back({true, output});
        } else if (mode == "concat") {
            source = "{{ 'prefix' ~ word ~ 'suffix' }}";
            variables = {{"word", "input"}};
            expected = {{false, "prefix"}, {true, "input"}, {false, "suffix"}};
        } else if (mode == "null-concat") {
            source = "{{ none ~ word }}";
            variables = {{"word", "input"}};
            expected = {{true, "input"}};
        } else { require(false, "unknown workload"); }
    }
};

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

static uint64_t checksum(const std::string & value, const std::vector<bool> & input) {
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < value.size(); ++i) {
        hash = (hash ^ static_cast<unsigned char>(value[i])) * 1099511628211ULL;
        hash = (hash ^ static_cast<unsigned char>(input[i])) * 1099511628211ULL;
    }
    return hash;
}

int main(int argc, char ** argv) {
    require(argc >= 2 && argc <= 4, "usage: workload [item-count [measured-renders]]");
    const std::string mode = argv[1];
    const size_t count = argc >= 3 ? std::stoul(argv[2]) : 16;
    const size_t repeats = argc >= 4 ? std::stoul(argv[3]) : 1;
    require(count <= 100000 && repeats >= 1 && repeats <= 100, "fixture size out of range");
    const fixture data(mode, count);
    const auto expected_text = text(data.expected);
    const auto expected_origins = origins(data.expected);
    jinja::lexer lexer;
    const auto tokens = lexer.tokenize(data.source);
    auto program = jinja::parse_from_tokens(tokens);
    jinja::context prepared(data.source);
    jinja::global_from_json(prepared, data.variables, true);
    for (size_t iteration = 0; iteration <= repeats; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        jinja::context context(prepared);
        jinja::runtime runtime(context);
        const auto result = runtime.gather_string_parts(runtime.execute(program))->as_string();
        const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const auto output = text(result.parts);
        const auto provenance = origins(result.parts);
        require(output == expected_text, "rendered bytes changed");
        require(provenance == expected_origins, "template/input byte origins changed");
        if (iteration) {
            printf("{\"workload\":\"%s\",\"count\":%zu,\"bytes\":%zu,\"checksum\":\"%llu\",\"milliseconds\":%.6f}\n",
                    mode.c_str(), count, output.size(), static_cast<unsigned long long>(checksum(output, provenance)), elapsed);
        }
    }
}
