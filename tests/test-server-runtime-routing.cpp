#include "server-common.h"
#include "server-expert-guard.h"
#include "chat.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}
template<class F> static void rejects(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::exception &) { rejected = true; }
    require(rejected, "invalid generation prefix was accepted");
}

static void test_template_kwargs_generation_prefix() {
    const std::string source =
        "{%- for m in messages -%}{{- m.role + ':' + m.content + '\\n' -}}{%- endfor -%}"
        "{%- if add_generation_prompt -%}assistant:"
        "{%- if thinking is undefined or thinking -%}<think>{%- else -%}<response>{%- endif -%}"
        "{%- endif -%}";
    const auto templates = common_chat_templates_init(nullptr, source);
    common_chat_templates_inputs inputs;
    common_chat_msg message;
    message.role = "user";
    message.content = "fixture";
    inputs.messages = {message};
    inputs.use_jinja = true;
    inputs.add_generation_prompt = true;
    inputs.force_pure_content = true; // Exercise the shared prefix diff without a model/parser heuristic.
    for (const bool thinking : {true, false}) {
        inputs.chat_template_kwargs["thinking"] = thinking ? "true" : "false";
        const auto rendered = common_chat_templates_apply(templates.get(), inputs);
        const std::string expected = thinking ? "assistant:<think>" : "assistant:<response>";
        require(rendered.generation_prompt == expected,
                "generation prefix diff must use the same template kwargs as the actual prompt render");
        require(rendered.prompt.size() >= expected.size()
                && rendered.prompt.compare(rendered.prompt.size() - expected.size(), expected.size(), expected) == 0,
                "kwargs-controlled generation prefix does not match rendered prompt tail");
    }
    for (const auto * value : {"true", "false"}) {
        inputs.chat_template_kwargs["thinking"] = "false";
        inputs.chat_template_kwargs["add_generation_prompt"] = value;
        const auto rendered = common_chat_templates_apply(templates.get(), inputs);
        require(rendered.generation_prompt == "assistant:<response>",
                "reserved template kwarg must not collapse the prefix comparison to identical renders");
        require(rendered.prompt == "user:fixture\nassistant:<response>",
                "prefix correction changed the actual prompt render");
    }
    inputs.chat_template_kwargs.clear();
    require(common_chat_templates_apply(templates.get(), inputs).generation_prompt == "assistant:<think>",
            "a previous request's thinking kwarg leaked into the template default");
}

static void test_verified_current_prefix() {
    const std::string assistant = "<|open|>message role=\"assistant\"<|sep|>";
    const std::string response = "<|open|>response<|sep|>";
    // Earlier history contains the same synthetic marker IDs. Only the verified
    // current suffix is passed to the observer by the production launch path.
    const llama_tokens prompt = {1, 2, 3, 88, 89, 4, 5};
    server_expert_guard_verify_prefix(assistant + "<|open|>think<|sep|>", {4, 5}, prompt);
    server_expert_guard_verify_prefix(assistant + response, {4, 5}, prompt);
    server_expert_guard_verify_prefix(assistant + "<|open|>response", {4, 5}, prompt);
    rejects([&] { server_expert_guard_verify_prefix("", {}, prompt); });
    rejects([&] { server_expert_guard_verify_prefix(response, {4, 5}, prompt); });
    rejects([&] { server_expert_guard_verify_prefix(assistant + response, {1, 2, 3}, prompt); });
    rejects([&] { server_expert_guard_verify_prefix(assistant + response + "prefilled answer", {4, 5}, prompt); });
    rejects([&] { server_expert_guard_verify_prefix(assistant + "<|end_of_msg|>" + response, {4, 5}, prompt); });
}
int main(int argc, char **) {
    try {
        require(argc == 1,
                "usage: test-server-runtime-routing");
        test_template_kwargs_generation_prefix();
        test_verified_current_prefix();
        std::puts("runtime routing server: 2 groups PASS (no service started)");
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
