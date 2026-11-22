// SPDX-License-Identifier: Apache-2.0
// An assistant message that has been sent renders to the same text whatever follows it, so the prompt cache, which
// matches a token prefix, keeps a finished turn when the next user message arrives. Renders a conversation up to a
// tool result, then the same conversation with an answer and a new user message after it, and requires the first to
// be a prefix of the second. Runs on the repository's templates; no model is loaded.
#include "chat.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

using json = nlohmann::ordered_json;

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

static bool contains(const std::string & text, const std::string & part) {
    return text.find(part) != std::string::npos;
}

static const char * TOOLS = R"([{"type": "function", "function": {"name": "list_files", "description": "List a folder.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}])";

// The turn in progress: a user message, the assistant's tool call, the tool's result.
static json open_turn(const char * reasoning) {
    json messages = json::parse(R"([
        {"role": "user", "content": "List the folder."},
        {"role": "assistant", "content": "Let me look.", "tool_calls": [{"id": "call_1", "type": "function",
            "function": {"name": "list_files", "arguments": "{\"path\": \".\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "a.txt\nb.txt"}
    ])");
    if (reasoning) { messages[1]["reasoning_content"] = reasoning; }
    return messages;
}

// The same turn finished, and the next user message.
static json next_turn(const char * reasoning) {
    json messages = open_turn(reasoning);
    messages.push_back({{"role", "assistant"}, {"content", "Two files: a.txt and b.txt."}});
    messages.push_back({{"role", "user"}, {"content", "Thanks. And now?"}});
    return messages;
}

struct fixture {
    common_chat_templates_ptr tmpls;

    explicit fixture(const std::string & path) : tmpls(common_chat_templates_init(/* model= */ nullptr, read_file(path))) {}

    std::string prompt(const json & messages, bool generation_prompt, bool thinking,
                       const std::map<std::string, std::string> & kwargs = {}) const {
        common_chat_templates_inputs inputs;
        inputs.messages              = common_chat_msgs_parse_oaicompat(messages);
        inputs.tools                 = common_chat_tools_parse_oaicompat(json::parse(TOOLS));
        inputs.add_generation_prompt = generation_prompt;
        inputs.enable_thinking       = thinking;
        inputs.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;
        inputs.chat_template_kwargs  = kwargs;
        if (!thinking) { inputs.chat_template_kwargs["enable_thinking"] = "false"; } // as the server passes it
        return common_chat_templates_apply(tmpls.get(), inputs).prompt;
    }
};

static void history_is_a_prefix(const std::string & path) {
    const fixture f(path);
    for (const bool thinking : {true, false}) {
        for (const char * reasoning : {(const char *) nullptr, "The user wants the folder listed."}) {
            const std::string what = path + (thinking ? ", thinking on" : ", thinking off") +
                                     (reasoning ? ", with reasoning_content" : ", no reasoning_content");
            const std::string open = f.prompt(open_turn(reasoning), /* generation_prompt= */ false, thinking);
            const std::string next = f.prompt(next_turn(reasoning), /* generation_prompt= */ true, thinking);
            require(contains(open, "Let me look.") && contains(open, "a.txt\nb.txt"), what + ": the turn is not rendered");
            require(next.compare(0, open.size(), open) == 0,
                    what + ": a new user message changed how the earlier turn renders\n--- before ---\n" + open +
                            "\n--- after ---\n" + next);
            // The reasoning a client sends back stays where it was written.
            require(!reasoning || contains(open, reasoning), what + ": reasoning_content is not rendered");
        }
    }
}

int main() {
    try {
        const std::string qwen36 = "models/templates/unsloth-Qwen3.6-35B-A3B.jinja";
        for (const char * path : { qwen36.c_str(), "models/templates/Qwen-Qwen3.8-Flash-Next.jinja",
                                   "models/templates/bartowski-Qwen3.8-Flash-Next.jinja",
                                   "models/templates/unsloth-Qwen3.8-Flash-Next.jinja" }) {
            history_is_a_prefix(path);
        }

        const fixture f(qwen36);
        // An assistant message with no reasoning carries the empty block it was generated after.
        const std::string open = f.prompt(open_turn(nullptr), false, true);
        require(contains(open, "<|im_start|>assistant\n<think>\n\n</think>\n\nLet me look.\n\n<tool_call>"),
                "Qwen3.6: the tool call's message does not open with an empty think block:\n" + open);
        // The generation prompt is the template's own.
        const std::string on  = f.prompt(open_turn(nullptr), true, true);
        const std::string off = f.prompt(open_turn(nullptr), true, false);
        require(on == open + "<|im_start|>assistant\n<think>\n", "Qwen3.6: the thinking generation prompt changed:\n" + on);
        require(off == open + "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                "Qwen3.6: the non-thinking generation prompt changed:\n" + off);
        // A request that names preserve_thinking keeps its say: false is the template's own rendering, which drops
        // the block from the messages before the last user message.
        const std::string dropped = f.prompt(next_turn("The user wants the folder listed."), true, true,
                                             {{"preserve_thinking", "false"}});
        require(contains(dropped, "<|im_start|>assistant\nLet me look.") && !contains(dropped, "The user wants"),
                "Qwen3.6: preserve_thinking=false did not drop the earlier turn's think block:\n" + dropped);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-chat-history-prefix: %s\n", error.what());
        return 1;
    }
    std::printf("test-chat-history-prefix: ok\n");
    return 0;
}
