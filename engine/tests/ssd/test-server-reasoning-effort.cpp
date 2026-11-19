// SPDX-License-Identifier: Apache-2.0
// reasoning_effort takes what the chat template offers, as /props reports it, and "on" where the model thinks, which
// thinks as a request without the field does. A client whose every choice sends a value, such as Codex, offers
// thinking on a template that only turns it off with "on". Runs the server's own request steps; no model is loaded.
#include "server-chat.h"
#include "server-common.h"
#include "chat.h"

#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const std::string & message) {
    if (!ok) { throw std::runtime_error(message); }
}

struct fixture {
    server_chat_params opt{};

    explicit fixture(const std::string & source) {
        opt.use_jinja          = true;
        opt.prefill_assistant  = true;
        opt.reasoning_format   = COMMON_REASONING_FORMAT_DEEPSEEK;
        opt.force_pure_content = true; // the prompt alone is under test, not a parser for the model's output
        opt.tmpls              = common_chat_templates_init(/* model= */ nullptr, source);
        opt.reasoning          = common_chat_templates_reasoning(opt.tmpls.get());
    }

    // The prompt of a chat-completions request with this reasoning_effort; nullptr sends none.
    std::string prompt(const char * effort) const {
        json body = json::parse(R"({"messages": [{"role": "user", "content": "hi"}]})");
        if (effort) { body["reasoning_effort"] = effort; }
        std::vector<raw_buffer> files;
        return oaicompat_chat_params_parse(body, opt, files).at("prompt").get<std::string>();
    }

    bool refused(const char * effort) const {
        try {
            prompt(effort);
        } catch (const std::invalid_argument &) {
            return true;
        }
        return false;
    }
};

// What the template offers, as /props reports it, for a failure message.
static std::string offer(const fixture & f) {
    if (!f.opt.reasoning) { return " (it does not think)"; }
    return std::string(" (can_turn_off=") + (f.opt.reasoning->can_turn_off ? "true" : "false") + " efforts=[" +
           string_join(f.opt.reasoning->efforts, ",") + "] default=" + f.opt.reasoning->default_effort + ")";
}

static const std::string turns =
    "{%- for m in messages -%}<|{{ m.role }}|>{{ m.content }}<|end|>{%- endfor -%}";
// The turns of a template that thinks: an earlier answer keeps its thinking, which is how the engine sees it thinks.
static const std::string thinking_turns =
    "{%- for m in messages -%}<|{{ m.role }}|>"
    "{%- if m.reasoning_content -%}<think>{{ m.reasoning_content }}</think>{%- endif -%}"
    "{{ m.content }}<|end|>{%- endfor -%}";

// Thinks unless enable_thinking turns it off, with no efforts, as Qwen3.6's and Gemma 4's templates do.
static void test_switch() {
    const fixture f(thinking_turns +
        "{%- if add_generation_prompt -%}<|assistant|>"
        "{%- if enable_thinking is defined and not enable_thinking -%}<think></think>{%- else -%}<think>{%- endif -%}"
        "{%- endif -%}");
    require(f.opt.reasoning && f.opt.reasoning->can_turn_off && f.opt.reasoning->efforts.empty(),
            "the switch fixture must offer a switch and no efforts");
    require(string_ends_with(f.prompt(nullptr), "<|assistant|><think>"), "no reasoning_effort must think");
    require(f.prompt("on") == f.prompt(nullptr), "\"on\" must render as no reasoning_effort does");
    require(string_ends_with(f.prompt("none"), "<|assistant|><think></think>"), "\"none\" must turn thinking off");
    require(f.refused("low"), "an effort the template does not offer must be refused");
}

// Always thinks and takes high or low, high when none is named, as Muse's template does with its own values.
static void test_efforts() {
    const fixture f(thinking_turns +
        "{%- if add_generation_prompt -%}<|assistant|>"
        "{%- if reasoning_effort is not defined or reasoning_effort == 'high' -%}high<think>"
        "{%- elif reasoning_effort == 'low' -%}low<think>{%- endif -%}"
        "{%- endif -%}");
    require(f.opt.reasoning && !f.opt.reasoning->can_turn_off && f.opt.reasoning->efforts.size() == 2 &&
                f.opt.reasoning->default_effort == "high",
            "the effort fixture must always think and take high and low, high by default" + offer(f));
    require(f.prompt("on") == f.prompt(nullptr), "\"on\" must render as no reasoning_effort does");
    require(string_ends_with(f.prompt("on"), "<|assistant|>high<think>"), "\"on\" must run as the template's default");
    require(string_ends_with(f.prompt("low"), "<|assistant|>low<think>"), "an offered effort must reach the template");
    require(f.refused("none"), "\"none\" must be refused where the template cannot turn thinking off");
}

// A template that does not think takes "none", and nothing else.
static void test_no_thinking() {
    const fixture f(turns + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}");
    require(!f.opt.reasoning, "the plain fixture must not think");
    require(f.prompt("none") == f.prompt(nullptr), "\"none\" must be taken by a model that does not think");
    require(f.refused("on"), "\"on\" must be refused by a model that does not think");
    require(f.refused("low"), "an effort must be refused by a model that does not think");
}

// A Responses request's reasoning.effort becomes the same field, so "on" works there too.
static void test_responses() {
    const fixture f(thinking_turns +
        "{%- if add_generation_prompt -%}<|assistant|>"
        "{%- if enable_thinking is defined and not enable_thinking -%}<think></think>{%- else -%}<think>{%- endif -%}"
        "{%- endif -%}");
    const auto prompt = [&](const char * effort) {
        json request = json::parse(R"({"model": "local", "input": "hi", "stream": true})");
        if (effort) { request["reasoning"] = {{"effort", effort}}; }
        json body = server_chat_convert_responses_to_chatcmpl(request);
        std::vector<raw_buffer> files;
        return oaicompat_chat_params_parse(body, f.opt, files).at("prompt").get<std::string>();
    };
    require(prompt("on") == prompt(nullptr), "reasoning.effort \"on\" must run as no effort does");
    require(string_ends_with(prompt("none"), "<|assistant|><think></think>"), "reasoning.effort \"none\" must turn thinking off");
}

int main(int argc, char **) {
    try {
        require(argc == 1, "usage: test-server-reasoning-effort");
        test_switch();
        test_efforts();
        test_no_thinking();
        test_responses();
        std::puts("reasoning effort: PASS (no model loaded)");
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
