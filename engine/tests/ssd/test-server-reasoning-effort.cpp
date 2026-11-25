// SPDX-License-Identifier: Apache-2.0
// reasoning_effort is fitted to what the chat template offers, as /props reports it, so each client's own words run
// on any model: a value the template takes is used as it is; "none" turns thinking off, or takes the least effort
// where the template cannot; an effort on the ladder minimal < low < medium < high < xhigh < max (ultra and
// persistent above it) takes the template's highest that does not exceed it, else its least; a template with no such
// efforts thinks as it does by default, as do "on" and any other word. The Responses and Anthropic routes reach the
// same field; an Anthropic request that names no thinking does not think. Runs the server's own request steps on
// fixtures and on the repository's templates; no model is loaded.
#include "server-chat.h"
#include "server-common.h"
#include "chat.h"

#include <cstdio>
#include <fstream>
#include <optional>
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

    std::string prompt_of(json body) const {
        std::vector<raw_buffer> files;
        return oaicompat_chat_params_parse(body, opt, files).at("prompt").get<std::string>();
    }

    // The prompt of a chat-completions request with this reasoning_effort; nullptr sends none.
    std::string prompt(const char * effort) const {
        json body = json::parse(R"({"messages": [{"role": "user", "content": "hi"}]})");
        if (effort) { body["reasoning_effort"] = effort; }
        return prompt_of(body);
    }

    // The prompt of a Responses request with this reasoning.effort; nullptr sends none.
    std::string responses_prompt(const char * effort) const {
        json request = json::parse(R"({"model": "local", "input": "hi", "stream": true})");
        if (effort) { request["reasoning"] = {{"effort", effort}}; }
        return prompt_of(server_chat_convert_responses_to_chatcmpl(request));
    }

    // The prompt of an Anthropic Messages request with this thinking type and output_config.effort; nullptr omits.
    std::string anthropic_prompt(const char * thinking_type, const char * effort) const {
        json request = json::parse(R"({"model": "local", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]})");
        if (thinking_type) { request["thinking"] = {{"type", thinking_type}}; }
        if (effort) { request["output_config"] = {{"effort", effort}}; }
        return prompt_of(server_chat_convert_anthropic_to_oai(request));
    }
};

// What the template offers, as /props reports it, for a failure message.
static std::string offer(const fixture & f) {
    if (!f.opt.reasoning) { return " (it does not think)"; }
    return std::string(" (can_turn_off=") + (f.opt.reasoning->can_turn_off ? "true" : "false") + " efforts=[" +
           string_join(f.opt.reasoning->efforts, ",") + "] default=" + f.opt.reasoning->default_effort + ")";
}

static bool offers(const fixture & f, bool can_turn_off, std::vector<std::string> efforts, const std::string & def) {
    if (!f.opt.reasoning || f.opt.reasoning->can_turn_off != can_turn_off || f.opt.reasoning->default_effort != def) {
        return false;
    }
    std::vector<std::string> got = f.opt.reasoning->efforts;
    std::sort(got.begin(), got.end());
    std::sort(efforts.begin(), efforts.end());
    return got == efforts;
}

static const std::string turns =
    "{%- for m in messages -%}<|{{ m.role }}|>{{ m.content }}<|end|>{%- endfor -%}";
// The turns of a template that thinks: an earlier answer keeps its thinking, which is how the engine sees it thinks.
static const std::string thinking_turns =
    "{%- for m in messages -%}<|{{ m.role }}|>"
    "{%- if m.reasoning_content -%}<think>{{ m.reasoning_content }}</think>{%- endif -%}"
    "{{ m.content }}<|end|>{%- endfor -%}";

// Thinks unless enable_thinking turns it off, with no efforts, as Qwen3.6's and Gemma 4's templates do.
static const std::string switch_template = thinking_turns +
    "{%- if add_generation_prompt -%}<|assistant|>"
    "{%- if enable_thinking is defined and not enable_thinking -%}<think></think>{%- else -%}<think>{%- endif -%}"
    "{%- endif -%}";

// Turns thinking off, or thinks at low, medium, high or max, high by default, as DeepSeek-V4.1's template does.
static const std::string graded_template = thinking_turns +
    "{%- if add_generation_prompt -%}<|assistant|>"
    "{%- if enable_thinking is defined and not enable_thinking -%}chat"
    "{%- else -%}{%- set e = reasoning_effort if reasoning_effort is defined else 'high' -%}"
    "{%- if e == 'low' -%}low{%- elif e == 'medium' -%}medium{%- elif e == 'high' -%}high{%- elif e == 'max' -%}max"
    "{%- else -%}{{ raise_exception('bad effort') }}{%- endif -%}<think>"
    "{%- endif -%}"
    "{%- endif -%}";

// Always thinks and takes high or low, high when none is named.
static const std::string always_template = thinking_turns +
    "{%- if add_generation_prompt -%}<|assistant|>"
    "{%- if reasoning_effort is not defined or reasoning_effort == 'high' -%}high<think>"
    "{%- elif reasoning_effort == 'low' -%}low<think>{%- endif -%}"
    "{%- endif -%}";

static void test_switch() {
    const fixture f(switch_template);
    require(offers(f, true, {}, ""), "the switch fixture must offer a switch and no efforts" + offer(f));
    const std::string thinks = f.prompt(nullptr);
    require(string_ends_with(thinks, "<|assistant|><think>"), "no reasoning_effort must think");
    require(string_ends_with(f.prompt("none"), "<|assistant|><think></think>"), "\"none\" must turn thinking off");
    for (const char * effort : {"on", "minimal", "low", "medium", "high", "xhigh", "max", "ultra", "persistent", "banana"}) {
        require(f.prompt(effort) == thinks, std::string("\"") + effort + "\" must think as the template does by default");
    }
}

static void test_graded() {
    const fixture f(graded_template);
    require(offers(f, true, {"low", "medium", "high", "max"}, "high"),
            "the graded fixture must turn off and take low, medium, high, max, high by default" + offer(f));
    const auto runs = [&](const char * effort, const std::string & tail) {
        require(string_ends_with(f.prompt(effort), "<|assistant|>" + tail),
                std::string("\"") + (effort ? effort : "(none sent)") + "\" must run as " + tail);
    };
    runs(nullptr, "high<think>");
    runs("on", "high<think>");
    runs("none", "chat");
    runs("minimal", "low<think>");
    runs("low", "low<think>");
    runs("medium", "medium<think>");
    runs("high", "high<think>");
    runs("xhigh", "high<think>");
    runs("max", "max<think>");
    runs("ultra", "max<think>");
    runs("persistent", "max<think>");
    runs("banana", "high<think>");
}

static void test_always() {
    const fixture f(always_template);
    require(offers(f, false, {"high", "low"}, "high"),
            "the always fixture must always think and take high and low, high by default" + offer(f));
    const auto runs = [&](const char * effort, const std::string & tail) {
        require(string_ends_with(f.prompt(effort), "<|assistant|>" + tail),
                std::string("\"") + effort + "\" must run as " + tail);
    };
    runs("on", "high<think>");
    runs("none", "low<think>");     // it cannot turn thinking off: its least
    runs("minimal", "low<think>");  // below its least: its least
    runs("medium", "low<think>");   // its highest not above medium
    runs("xhigh", "high<think>");
    runs("banana", "high<think>");
}

// A template that does not think runs without thinking whatever is asked.
static void test_no_thinking() {
    const fixture f(turns + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}");
    require(!f.opt.reasoning, "the plain fixture must not think");
    for (const char * effort : {"none", "on", "low", "xhigh"}) {
        require(f.prompt(effort) == f.prompt(nullptr), std::string("\"") + effort + "\" must run as no effort does");
    }
}

// A Responses request's reasoning.effort becomes the same field.
static void test_responses() {
    const fixture f(graded_template);
    require(f.responses_prompt("on") == f.responses_prompt(nullptr), "reasoning.effort \"on\" must run as no effort does");
    require(string_ends_with(f.responses_prompt("none"), "<|assistant|>chat"), "reasoning.effort \"none\" must turn thinking off");
    require(string_ends_with(f.responses_prompt("xhigh"), "<|assistant|>high<think>"), "reasoning.effort \"xhigh\" must run as high");
}

// Anthropic's thinking and output_config.effort reach the same field. As in Anthropic's API, a request thinks only
// when it asks to: no thinking is thinking off, whatever its output_config says, which is how Claude Code turns
// thinking off for a model it does not know.
static void test_anthropic() {
    const fixture f(graded_template);
    require(string_ends_with(f.anthropic_prompt(nullptr, nullptr), "<|assistant|>chat"), "no thinking must turn thinking off");
    require(string_ends_with(f.anthropic_prompt(nullptr, "high"), "<|assistant|>chat"), "no thinking at an effort must turn thinking off");
    require(string_ends_with(f.anthropic_prompt("disabled", "high"), "<|assistant|>chat"), "thinking disabled must turn thinking off");
    require(string_ends_with(f.anthropic_prompt("adaptive", nullptr), "<|assistant|>high<think>"), "adaptive thinking with no effort must run as the default");
    require(string_ends_with(f.anthropic_prompt("adaptive", "xhigh"), "<|assistant|>high<think>"), "adaptive at xhigh must run as high");
    require(string_ends_with(f.anthropic_prompt("enabled", "low"), "<|assistant|>low<think>"), "enabled at low must run as low");
    require(string_ends_with(f.anthropic_prompt("adaptive", "max"), "<|assistant|>max<think>"), "adaptive at max must run as max");
    // A template that cannot turn thinking off thinks at its least.
    const fixture always(always_template);
    require(string_ends_with(always.anthropic_prompt(nullptr, nullptr), "<|assistant|>low<think>"), "no thinking must take the least effort where thinking cannot be off");
}

// The repository's templates of the models InferSSD runs offer what the mapping assumes.
static void test_model_templates() {
    for (const char * path : {"models/templates/unsloth-Qwen3.6-35B-A3B.jinja", "models/templates/google-gemma-4-31B-it.jinja"}) {
        const fixture f(read_file(path));
        require(offers(f, true, {}, ""), std::string(path) + " must only turn thinking on and off" + offer(f));
        const std::string thinks = f.prompt(nullptr);
        require(f.prompt("none") != thinks, std::string(path) + ": \"none\" must turn thinking off");
        for (const char * effort : {"low", "medium", "xhigh", "max"}) {
            require(f.prompt(effort) == thinks, std::string(path) + ": \"" + effort + "\" must think as by default");
        }
    }
    {
        const fixture f(read_file("models/templates/deepseek-ai-DeepSeek-V4.1.jinja"));
        require(offers(f, true, {"low", "medium", "high", "max"}, "high"), "DeepSeek-V4.1" + offer(f));
        require(f.prompt("none") != f.prompt(nullptr), "DeepSeek-V4.1: \"none\" must turn thinking off");
        require(f.prompt("minimal") == f.prompt("low"), "DeepSeek-V4.1: minimal must run as low");
        require(f.prompt("xhigh") == f.prompt("high"), "DeepSeek-V4.1: xhigh must run as high");
        require(f.prompt("ultra") == f.prompt("max"), "DeepSeek-V4.1: ultra must run as max");
        require(f.prompt("on") == f.prompt(nullptr) && f.prompt("high") == f.prompt(nullptr), "DeepSeek-V4.1: high is its default");
    }
    {
        const fixture f(read_file("models/templates/muse-spark.jinja"));
        require(offers(f, false, {"low", "medium", "high", "xhigh"}, "xhigh"), "Muse Spark" + offer(f));
        require(f.prompt("none") == f.prompt("low"), "Muse Spark: \"none\" must run as low, its least");
        require(f.prompt("minimal") == f.prompt("low"), "Muse Spark: minimal must run as low");
        require(f.prompt("max") == f.prompt("xhigh"), "Muse Spark: max must run as xhigh");
        require(f.prompt("on") == f.prompt(nullptr) && f.prompt("xhigh") == f.prompt(nullptr), "Muse Spark: xhigh is its default");
    }
    {
        const fixture f(read_file("models/templates/thinkingmachines-Inkling-2-Mini.jinja"));
        require(offers(f, true, {"minimal", "low", "medium", "high", "xhigh"}, "high"), "Inkling-2-Mini" + offer(f));
        require(f.prompt("none") != f.prompt(nullptr), "Inkling-2-Mini: \"none\" must turn thinking off");
        require(f.prompt("none").find("Thinking effort level: 0<|end_message|>") != std::string::npos, "Inkling-2-Mini: \"none\" must emit effort 0");
        require(f.prompt("low").find("Thinking effort level: 0.2<|end_message|>") != std::string::npos, "Inkling-2-Mini: \"low\" must emit effort 0.2");
        require(f.prompt("max") == f.prompt("xhigh"), "Inkling-2-Mini: max must run as xhigh");
        require(f.prompt("on") == f.prompt(nullptr) && f.prompt("high") == f.prompt(nullptr), "Inkling-2-Mini: high is its default");
    }
}

int main(int argc, char **) {
    try {
        require(argc == 1, "usage: test-server-reasoning-effort");
        test_switch();
        test_graded();
        test_always();
        test_no_thinking();
        test_responses();
        test_anthropic();
        test_model_templates();
        std::puts("reasoning effort: PASS (no model loaded)");
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
