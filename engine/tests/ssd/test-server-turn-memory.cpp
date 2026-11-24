// SPDX-License-Identifier: Apache-2.0
// The server renders the assistant turns it generated as it generated them, with the reasoning a client left out.
// Two generated turns can be alike in content and tool calls and differ in reasoning, as when a model reads the same
// file twice in a conversation, or the same file in two conversations: each message must get the reasoning of its own
// turn, found by the messages before it, and never another turn's. Runs the server's own request steps on a fixture
// template; no model is loaded.
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

    json params(const json & messages) const {
        std::vector<raw_buffer> files;
        json body = {{"messages", messages}};
        return oaicompat_chat_params_parse(body, opt, files);
    }

    std::string prompt(const json & messages) const {
        return params(messages).at("prompt").get<std::string>();
    }

    // The server answers `messages` with a call that reads `path`, thinking `reasoning`.
    void generate_read(const json & messages, const std::string & reasoning, const std::string & path = "notes.txt") const {
        common_chat_msg msg;
        msg.role              = "assistant";
        msg.reasoning_content = reasoning;
        msg.tool_calls.push_back({"read_file", "{\"path\":\"" + path + "\"}", "call"});
        server_turn_memory_record(msg, params(messages).at("turn_context").get<std::string>());
    }
};

// A read as a client sends it back: arguments re-encoded, reasoning as given (nullptr: left out).
static json sent_read(const char * reasoning, const std::string & path = "notes.txt") {
    json msg = {{"role", "assistant"}, {"content", ""},
                {"tool_calls", json::array({{{"id", "call"}, {"type", "function"},
                    {"function", {{"name", "read_file"}, {"arguments", "{ \"path\": \"" + path + "\" }"}}}}})}};
    if (reasoning) { msg["reasoning_content"] = reasoning; }
    return msg;
}

// A conversation that opens with `question` and has the reads, each answered by the tool.
static json conversation(const std::string & question, const std::vector<json> & reads) {
    json messages = json::array({{{"role", "user"}, {"content", question}}});
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

static std::string think(const std::string & reasoning) { return "<think>" + reasoning + "</think>"; }

int main() {
    try {
        const fixture f;
        const std::string a1 = "The user wants the notes.";
        const std::string a2 = "I should check the notes again.";
        const std::string b1 = "The other user wants the notes too.";
        const std::string A  = "Read notes.txt.";
        const std::string B  = "What do the notes say?";

        // one generated turn: a message without its reasoning gets it back
        f.generate_read(conversation(A, {}), a1);
        std::string prompt = f.prompt(conversation(A, {sent_read(nullptr)}));
        require(count(prompt, think(a1)) == 1, "a turn's reasoning was not restored:\n" + prompt);

        // the model reads the same file again, thinking something else
        f.generate_read(conversation(A, {sent_read(nullptr)}), a2);

        // another conversation reads the same file, thinking something else again
        f.generate_read(conversation(B, {}), b1);

        // each turn comes back without reasoning and gets its own
        prompt = f.prompt(conversation(A, {sent_read(nullptr), sent_read(nullptr)}));
        require(count(prompt, think(a1)) == 1 && count(prompt, think(a2)) == 1 && prompt.find(think(a1)) < prompt.find(think(a2)) &&
                count(prompt, think(b1)) == 0, "two like turns of a conversation did not each get their own reasoning:\n" + prompt);
        prompt = f.prompt(conversation(B, {sent_read(nullptr)}));
        require(count(prompt, think(b1)) == 1 && count(prompt, "<think>") == 1,
                "a turn like one of another conversation did not get its own reasoning:\n" + prompt);
        prompt = f.prompt(conversation(A, {sent_read(nullptr)}));
        require(count(prompt, think(a1)) == 1 && count(prompt, "<think>") == 1,
                "the first turn of a conversation got another turn's reasoning:\n" + prompt);

        // a conversation this server did not generate the turn for: rendered as sent, with nobody's reasoning
        prompt = f.prompt(conversation("Show me the notes.", {sent_read(nullptr)}));
        require(count(prompt, "<think>") == 0, "a turn generated elsewhere got another conversation's reasoning:\n" + prompt);

        // reasoning sent back unchanged (up to whitespace) is the generated one; reasoning the client wrote is kept,
        // and the turns after it are still found
        prompt = f.prompt(conversation(A, {sent_read(" The user wants the notes.\n"), sent_read(nullptr)}));
        require(count(prompt, think(a1)) == 1 && count(prompt, think(a2)) == 1, "reasoning sent back was replaced:\n" + prompt);
        prompt = f.prompt(conversation(A, {sent_read("Something else."), sent_read(nullptr)}));
        require(count(prompt, think("Something else.")) == 1 && count(prompt, think(a2)) == 1,
                "changed reasoning was not kept, or the turn after it was lost:\n" + prompt);

        // a turn after other messages than it was generated after is not that turn
        prompt = f.prompt(conversation(A, {sent_read(nullptr, "todo.md"), sent_read(nullptr)}));
        require(count(prompt, think(a2)) == 0, "a turn was found after other messages:\n" + prompt);

        // A conversation still going on keeps its turns however many others come: each request restores them.
        // Others' turns of 1 MiB of reasoning, 80 of them (more than the memory keeps), and conversation A between.
        const std::string big(1u << 20, 'x');
        for (int i = 0; i < 80; ++i) {
            f.generate_read(conversation("Question " + std::to_string(i), {}), big + std::to_string(i));
            prompt = f.prompt(conversation(A, {sent_read(nullptr), sent_read(nullptr)}));
            require(count(prompt, think(a1)) == 1 && count(prompt, think(a2)) == 1,
                    "a conversation going on lost its turns to others at " + std::to_string(i));
        }
        prompt = f.prompt(conversation("Question 0", {sent_read(nullptr)}));
        require(count(prompt, "<think>") == 0, "the memory kept more than its limit");
        prompt = f.prompt(conversation("Question 79", {sent_read(nullptr)}));
        require(count(prompt, big + "79") == 1, "the newest turn was not kept");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-server-turn-memory: %s\n", error.what());
        return 1;
    }
    std::printf("test-server-turn-memory: ok\n");
    return 0;
}
