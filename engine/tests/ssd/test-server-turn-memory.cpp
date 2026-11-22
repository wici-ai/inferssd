// SPDX-License-Identifier: Apache-2.0
// The server renders the assistant turns it generated as it generated them, with the reasoning a client left out.
// Two generated turns can be alike in content and tool calls, as when a model reads the same file twice, and differ
// in reasoning: each message must get the reasoning of its own turn, or none when the turn cannot be told, and never
// another turn's. Runs the server's own request steps on a fixture template; no model is loaded.
#include "server-common.h"
#include "chat.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const std::string & message) {
    if (!ok) { throw std::runtime_error(message); }
}

// Every turn with its reasoning, and the names of its tool calls.
static const std::string turns_template =
    "{%- for m in messages -%}<|{{ m.role }}|>"
    "{%- if m.reasoning_content -%}<think>{{ m.reasoning_content }}</think>{%- endif -%}"
    "{{ m.content }}"
    "{%- if m.tool_calls -%}{%- for c in m.tool_calls -%}<call>{{ c.function.name }}</call>{%- endfor -%}{%- endif -%}"
    "<|end|>{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

struct fixture {
    server_chat_params opt{};

    fixture() {
        opt.use_jinja          = true;
        opt.prefill_assistant  = true;
        opt.reasoning_format   = COMMON_REASONING_FORMAT_DEEPSEEK;
        opt.force_pure_content = true; // the prompt alone is under test, not a parser for the model's output
        opt.tmpls              = common_chat_templates_init(/* model= */ nullptr, turns_template);
        opt.reasoning          = common_chat_templates_reasoning(opt.tmpls.get());
    }

    std::string prompt(const json & messages) const {
        std::vector<raw_buffer> files;
        json body = {{"messages", messages}};
        return oaicompat_chat_params_parse(body, opt, files).at("prompt").get<std::string>();
    }
};

static const char * READ_ARGUMENTS = "{\"path\":\"notes.txt\"}";

// A turn as the server generated it: reasoning, then a call that reads notes.txt.
static void generate_read(const std::string & reasoning) {
    common_chat_msg msg;
    msg.role              = "assistant";
    msg.reasoning_content = reasoning;
    msg.tool_calls.push_back({"read_file", READ_ARGUMENTS, "call"});
    server_turn_memory_record(msg);
}

// The same turn as a client sends it back: arguments re-encoded, reasoning as given (nullptr: left out).
static json sent_read(const char * reasoning) {
    json msg = {{"role", "assistant"}, {"content", ""},
                {"tool_calls", json::array({{{"id", "call"}, {"type", "function"},
                    {"function", {{"name", "read_file"}, {"arguments", "{ \"path\": \"notes.txt\" }"}}}}})}};
    if (reasoning) { msg["reasoning_content"] = reasoning; }
    return msg;
}

static json conversation(const std::vector<json> & reads) {
    json messages = json::array({{{"role", "user"}, {"content", "Read notes.txt."}}});
    for (const auto & read : reads) {
        messages.push_back(read);
        messages.push_back({{"role", "tool"}, {"tool_call_id", "call"}, {"content", "Ship on Friday."}});
    }
    return messages;
}

static size_t count(const std::string & text, const std::string & part) {
    size_t n = 0;
    for (size_t at = text.find(part); at != std::string::npos; at = text.find(part, at + part.size())) { n++; }
    return n;
}

int main() {
    try {
        const fixture f;
        const std::string first  = "<think>The user wants the notes.</think>";
        const std::string second = "<think>I should check the notes again.</think>";

        // one generated turn: a message without its reasoning gets it back
        generate_read("The user wants the notes.");
        std::string prompt = f.prompt(conversation({sent_read(nullptr)}));
        require(count(prompt, first) == 1, "a turn's reasoning was not restored:\n" + prompt);

        // the model reads the same file again, thinking something else
        generate_read("I should check the notes again.");

        // both turns come back without reasoning: each gets its own, in the order generated
        prompt = f.prompt(conversation({sent_read(nullptr), sent_read(nullptr)}));
        require(count(prompt, first) == 1 && count(prompt, second) == 1 && prompt.find(first) < prompt.find(second),
                "two like turns did not each get their own reasoning:\n" + prompt);

        // the client sends the reasoning back: it tells the turns apart, in any order and number
        prompt = f.prompt(conversation({sent_read("I should check the notes again."), sent_read(" The user wants the notes.\n")}));
        require(count(prompt, first) == 1 && count(prompt, second) == 1 && prompt.find(second) < prompt.find(first),
                "reasoning sent back did not choose its own turn:\n" + prompt);
        prompt = f.prompt(conversation({sent_read("The user wants the notes.")}));
        require(count(prompt, first) == 1 && count(prompt, second) == 0, "reasoning sent back was replaced:\n" + prompt);

        // one message without reasoning and two turns it could be: it is rendered as sent, not with either's reasoning
        prompt = f.prompt(conversation({sent_read(nullptr)}));
        require(count(prompt, "<think>") == 0, "a turn that cannot be told got another turn's reasoning:\n" + prompt);

        // reasoning the client wrote itself is kept
        prompt = f.prompt(conversation({sent_read("Something else."), sent_read(nullptr)}));
        require(count(prompt, "<think>Something else.</think>") == 1, "changed reasoning was not kept:\n" + prompt);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-server-turn-memory: %s\n", error.what());
        return 1;
    }
    std::printf("test-server-turn-memory: ok\n");
    return 0;
}
