// SPDX-License-Identifier: Apache-2.0
// The prompt cache keeps one state per conversation: a state saved for a conversation replaces the earlier states
// of the same conversation, also the ones it does not extend (the history was rendered or sent differently, a reply
// was generated again). Two prompts are states of one conversation when they share its opening: the system messages,
// the tools and the user messages before the first reply. Conversations that share only the system messages and the
// tools, and the shared prefix kept for all of them, are left alone. Runs on token lists and on the server's own
// chat request step; no model is loaded.
#include "server-chat.h"
#include "server-common.h"
#include "server-task.h"
#include "server-prompt-cache-fixture.h"
#include "chat.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const std::string & message) {
    if (!ok) { throw std::runtime_error(message); }
}

static const common_context_seq_rm_type FULL = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

// Tokens 1..4 are the system message and the tools; a conversation opens with two more.
static server_prompt make_prompt(std::vector<llama_token> tokens, int64_t n_conversation) {
    server_prompt prompt;
    prompt.tokens         = server_tokens(tokens, false);
    prompt.n_conversation = n_conversation;
    return prompt;
}

static bool cached(const server_prompt_cache & cache, const std::vector<llama_token> & tokens) {
    for (const auto & state : cache.states) {
        if (state.tokens.get_tokens() == tokens) {
            return true;
        }
    }
    return false;
}

static void test_supersedes() {
    const server_prompt state = make_prompt({ 1, 2, 3, 4, 5, 6, 20, 21, 22, 23 }, 6);

    require(server_prompt_supersedes(state, make_prompt({ 1, 2, 3, 4, 5, 6, 10, 11, 12 }, 0)),
            "a state replaces an earlier state of its conversation that it does not extend");
    require(server_prompt_supersedes(state, make_prompt({ 1, 2, 3, 4, 5, 6, 20, 21 }, 0)),
            "a state replaces an earlier state of its conversation that it extends");
    require(server_prompt_supersedes(state, make_prompt({ 1, 2, 3, 4, 5, 6, 10, 11, 12, 13 }, 0)),
            "a state replaces a state of its conversation of the same length");

    require(!server_prompt_supersedes(state, make_prompt({ 1, 2, 3, 4 }, 0)),
            "the shared prefix is not a state of the conversation");
    require(!server_prompt_supersedes(state, make_prompt({ 1, 2, 3, 4, 7, 8, 30, 31 }, 0)),
            "a conversation that shares only the system messages and tools is another conversation");
    require(!server_prompt_supersedes(state, make_prompt({ 1, 2, 3, 4, 5, 9, 30, 31 }, 0)),
            "a conversation that parts inside the opening user message is another conversation");
    require(!server_prompt_supersedes(state, make_prompt({ 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14 }, 0)),
            "a longer state of the conversation stays");
    require(!server_prompt_supersedes(make_prompt({ 1, 2, 3, 4, 5, 6, 20, 21, 22, 23 }, 0),
                                      make_prompt({ 1, 2, 3, 4, 5, 6, 10, 11, 12 }, 6)),
            "a state whose opening is unknown replaces nothing");
    require(!server_prompt_supersedes(make_prompt({ 1, 2, 3, 4, 5, 6 }, 6), make_prompt({ 1, 2, 3, 4, 5, 6 }, 0)),
            "a state that ends with its opening replaces nothing");
}

static void test_memory_cache() {
    server_prompt_cache cache(0, 0);

    const auto add = [&](std::vector<llama_token> tokens, int64_t n_conversation) {
        return server_prompt_cache_testing::alloc_metadata(cache, make_prompt(tokens, n_conversation), 8, FULL) != nullptr;
    };

    require(add({ 1, 2, 3, 4 }, 0), "the shared prefix is cached");
    require(add({ 1, 2, 3, 4, 5, 6, 10, 11, 12 }, 6), "the first conversation is cached");
    require(add({ 1, 2, 3, 4, 7, 8, 30, 31, 32 }, 6), "the second conversation is cached");
    require(cache.states.size() == 3, "two conversations behind one shared prefix do not replace each other");

    // the first conversation again, its history rendered differently after the opening
    require(add({ 1, 2, 3, 4, 5, 6, 20, 21, 22, 23 }, 6), "the newer state is cached");
    require(cache.states.size() == 3, "the newer state replaces the earlier one");
    require(!cached(cache, { 1, 2, 3, 4, 5, 6, 10, 11, 12 }), "the earlier state of the conversation is gone");
    require(cached(cache, { 1, 2, 3, 4 }), "the shared prefix stays");
    require(cached(cache, { 1, 2, 3, 4, 7, 8, 30, 31, 32 }), "the other conversation stays");

    // a state without a known opening (a raw completion) that runs along a conversation removes nothing
    require(add({ 1, 2, 3, 4, 7, 8, 40, 41, 42, 43 }, 0), "a prompt with no known opening is cached");
    require(cache.states.size() == 4, "a prompt with no known opening replaces nothing");
}

struct temp_dir {
    std::filesystem::path path;

    temp_dir() {
        path = std::filesystem::temp_directory_path() /
            ("inferssd-prompt-cache-test-" + std::to_string(std::random_device{}()));
        std::filesystem::create_directories(path);
    }

    ~temp_dir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }

    std::string file(const std::string & name) const {
        const std::string res = (path / name).string();
        std::ofstream(res, std::ios::binary) << "state";
        return res;
    }
};

static void test_disk_cache() {
    const temp_dir dir;
    server_prompt_cache cache(0, 0);

    const auto add = [&](const server_prompt & prompt, const std::string & file) {
        return server_prompt_cache_testing::add_file_metadata(cache, prompt, file, 5, FULL) != nullptr;
    };
    const auto exists = [](const std::string & file) { return std::filesystem::exists(file); };

    const std::string f_prefix = dir.file("prefix.prompt");
    const std::string f_a1     = dir.file("a1.prompt");
    const std::string f_a2     = dir.file("a2.prompt");
    const std::string f_a3     = dir.file("a3.prompt");
    const std::string f_a4     = dir.file("a4.prompt");
    const std::string f_b1     = dir.file("b1.prompt");
    const std::string f_b2     = dir.file("b2.prompt");

    require(add(make_prompt({ 1, 2, 3, 4 }, 0), f_prefix), "the shared prefix is cached");

    // a state with checkpoints in its file: they go with the file, and with nothing else
    server_prompt a1 = make_prompt({ 1, 2, 3, 4, 5, 6, 10, 11, 12 }, 6);
    server_prompt_checkpoint checkpoint = {};
    checkpoint.pos_min  = 3;
    checkpoint.pos_max  = 3;
    checkpoint.n_tokens = 4;
    checkpoint.data     = { 1, 2, 3, 4 };
    a1.checkpoints.push_back(checkpoint);
    require(add(a1, f_a1), "the first conversation is cached");

    server_prompt b1 = make_prompt({ 1, 2, 3, 4, 7, 8, 30, 31, 32 }, 6);
    b1.checkpoints.push_back(checkpoint);
    require(add(b1, f_b1), "the second conversation is cached");
    require(cache.states.size() == 3 && exists(f_a1) && exists(f_b1) && exists(f_prefix),
            "two conversations behind one shared prefix do not replace each other");

    // the first conversation again, parting from its earlier state after the opening
    require(add(make_prompt({ 1, 2, 3, 4, 5, 6, 20, 21, 22, 23 }, 6), f_a2), "the newer state is cached");
    require(cache.states.size() == 3, "the newer state replaces the earlier one");
    require(!exists(f_a1), "the earlier state's file is removed");
    require(exists(f_a2) && exists(f_b1) && exists(f_prefix), "the other files stay");
    require(cache.states.front().file == f_prefix && cache.states.back().file == f_a2, "the newest state is last");

    // the next turn extends it: the state it extends goes too, the shared prefix it also extends does not
    require(add(make_prompt({ 1, 2, 3, 4, 5, 6, 20, 21, 22, 23, 24, 25 }, 6), f_a3), "the extended state is cached");
    require(cache.states.size() == 3 && !exists(f_a2) && exists(f_a3) && exists(f_prefix),
            "a state replaces the state of its conversation that it extends");

    // a shorter state of the conversation (a short request beside it, a reply generated again) leaves the longer
    require(add(make_prompt({ 1, 2, 3, 4, 5, 6, 50, 51 }, 6), f_a4), "the shorter state is cached");
    require(cache.states.size() == 4 && exists(f_a3) && exists(f_a4), "a shorter state replaces nothing");

    // once the conversation outgrows both, both go
    const std::string f_a5 = dir.file("a5.prompt");
    require(add(make_prompt({ 1, 2, 3, 4, 5, 6, 50, 51, 52, 53, 54, 55, 56 }, 6), f_a5), "the longest state is cached");
    require(cache.states.size() == 3 && !exists(f_a3) && !exists(f_a4) && exists(f_a5),
            "the longest state replaces every earlier state of the conversation");

    // the other conversation moves on; the first one is untouched
    require(add(make_prompt({ 1, 2, 3, 4, 7, 8, 60, 61, 62, 63 }, 6), f_b2), "the second conversation is cached again");
    require(cache.states.size() == 3 && !exists(f_b1) && exists(f_b2) && exists(f_a5) && exists(f_prefix),
            "each conversation keeps its newest state");
}

// The chat request step reports where the shared prefix and the conversation's opening end in the prompt.
static void test_chat_request_reports_the_opening() {
    server_chat_params opt{};
    opt.use_jinja          = true;
    opt.prefill_assistant  = true;
    opt.reasoning_format   = COMMON_REASONING_FORMAT_DEEPSEEK;
    opt.force_pure_content = true;
    opt.tmpls              = common_chat_templates_init(/* model= */ nullptr,
            "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}");

    const auto parse = [&](const char * messages) {
        json body = json::parse(std::string(R"({"messages": )") + messages + "}");
        std::vector<raw_buffer> files;
        return oaicompat_chat_params_parse(body, opt, files);
    };
    const auto prefix = [](const json & params, const char * key) {
        return params.at("prompt").get<std::string>().substr(0, json_value(params, key, (size_t) 0));
    };

    const std::string system = "<|im_start|>system\nBe brief.<|im_end|>\n";
    const std::string first  = "<|im_start|>user\nWhat is two plus two?<|im_end|>\n";

    const json one_turn = parse(R"([{"role": "system", "content": "Be brief."},
                                    {"role": "user", "content": "What is two plus two?"}])");
    require(prefix(one_turn, "shared_prefix_chars") == system, "the shared prefix ends after the system message");
    require(prefix(one_turn, "conversation_prefix_chars") == system + first,
            "the opening ends after the first user message");

    const json later = parse(R"([{"role": "system", "content": "Be brief."},
                                 {"role": "user", "content": "What is two plus two?"},
                                 {"role": "assistant", "content": "Four."},
                                 {"role": "user", "content": "And three plus three?"}])");
    require(prefix(later, "conversation_prefix_chars") == system + first,
            "a later turn of the conversation has the same opening");

    const json two_users = parse(R"([{"role": "system", "content": "Be brief."},
                                     {"role": "user", "content": "The folder is /work."},
                                     {"role": "user", "content": "What is two plus two?"},
                                     {"role": "assistant", "content": "Four."},
                                     {"role": "user", "content": "And three plus three?"}])");
    require(prefix(two_users, "conversation_prefix_chars") ==
            system + "<|im_start|>user\nThe folder is /work.<|im_end|>\n" + first,
            "the opening holds every user message before the first reply");

    const json no_system = parse(R"([{"role": "user", "content": "What is two plus two?"}])");
    require(!no_system.contains("shared_prefix_chars") && !no_system.contains("conversation_prefix_chars"),
            "a chat without a system message reports neither");

    // A Responses request whose second system message holds the user's context (SERVER_USER_CONTEXT_SYSTEM_MESSAGE):
    // the shared prefix is the app's system message alone, the context prefix both, the opening still holds both.
    const auto parse_with_context = [&](const server_chat_params & params) {
        json body = json::parse(R"({"messages": [{"role": "system", "content": "Be brief.\nAnswer in English.\n"},
                                                 {"role": "system", "content": "The folder is /work."},
                                                 {"role": "user", "content": "What is two plus two?"}]})");
        body[SERVER_USER_CONTEXT_SYSTEM_MESSAGE] = true;
        std::vector<raw_buffer> files;
        return oaicompat_chat_params_parse(body, params, files);
    };
    const json separate = parse_with_context(opt);
    const std::string app     = "<|im_start|>system\nBe brief.\nAnswer in English.\n<|im_end|>\n";
    const std::string context = "<|im_start|>system\nThe folder is /work.<|im_end|>\n";
    require(prefix(separate, "shared_prefix_chars") == app, "the shared prefix ends after the app's system message");
    require(prefix(separate, "context_prefix_chars") == app + context, "the context prefix ends after the user's context");
    require(prefix(separate, "conversation_prefix_chars") == app + context + first, "the opening holds the user's context");
    require(!parse(R"([{"role": "system", "content": "Be brief."}, {"role": "system", "content": "And kind."},
                       {"role": "user", "content": "Hi."}])").contains("context_prefix_chars"),
            "leading system messages that the request does not mark have no context prefix");
    require(!separate.contains(SERVER_USER_CONTEXT_SYSTEM_MESSAGE), "the request field is not passed on");

    // a template that merges the leading system messages into one turn and trims their texts: the shared prefix ends
    // at the start of the app text's last line, where tokenizers start a token; the context prefix after the turn
    server_chat_params merged{};
    merged.use_jinja          = opt.use_jinja;
    merged.prefill_assistant  = opt.prefill_assistant;
    merged.reasoning_format   = opt.reasoning_format;
    merged.force_pure_content = opt.force_pure_content;
    merged.tmpls              = common_chat_templates_init(/* model= */ nullptr,
            "<|im_start|>system\n{% for message in messages %}{% if message.role == 'system' %}{% if not loop.first %}{{ '\\n' }}{% endif %}"
            "{{ message.content | trim }}{% endif %}{% endfor %}<|im_end|>\n{% for message in messages %}{% if message.role != 'system' %}"
            "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}");
    const json one_turn_system = parse_with_context(merged);
    require(prefix(one_turn_system, "shared_prefix_chars") == "<|im_start|>system\nBe brief.\n",
            "the shared prefix of a merged system turn ends at the start of the app text's last line");
    require(prefix(one_turn_system, "context_prefix_chars") ==
            "<|im_start|>system\nBe brief.\nAnswer in English.\nThe folder is /work.<|im_end|>\n",
            "the context prefix of a merged system turn ends with the turn");
}

// The environment block Claude Code and the Claude desktop app send as a system message after the user's first
// words, trimmed from what they send: it goes into the first user message ahead of the words, after the system
// reminders that open it, so a new conversation of the same setup shares everything up to the words, where the server
// keeps a state.
static void test_claude_environment_opens_the_conversation() {
    const std::string reminder = "<system-reminder>\nAttribution for git commits.\n</system-reminder>\n";
    const std::string code_env =
        "# Environment\nYou have been invoked in the following environment: \n"
        " - Primary working directory: /work/project\n - Plugins: /opt/.codex/plugins/cache/tools/run/26.1123.11515/bin\n"
        " - Platform: linux\n\nToday's date is 2026-11-25.";
    const std::string desktop_env =
        "<system-reminder>\n# Environment\nYou have been invoked in the following environment: \n"
        " - Primary working directory: /private/var/empty\n</system-reminder>\n\n"
        "<system-reminder>\nToday's date is 2026-11-24.\n</system-reminder>";
    const json system = json::array({{{"type", "text"}, {"text", "You are Claude Code, Anthropic's official CLI for Claude."}},
                                      {{"type", "text"}, {"text", "Plugins are in /opt/.codex/plugins/cache/tools/run/26.1123.11515/bin."}}});
    const auto text = [](const std::string & t) { return json{{"type", "text"}, {"text", t}}; };
    const auto code_messages = [&](const std::string & words) {
        return json::array({
            {{"role", "user"}, {"content", json::array({text(reminder), text(words)})}},
            {{"role", "system"}, {"content", json::array({{{"type", "text"}, {"text", code_env},
                                                           {"cache_control", {{"type", "ephemeral"}}}}})}},
        });
    };
    const auto convert = [&](const json & messages) {
        return server_chat_convert_anthropic_to_oai({{"system", system}, {"messages", messages}, {"max_tokens", 16}});
    };
    const auto context_chars = [](const json & chat) { return json_value(chat, SERVER_FIRST_USER_CONTEXT_CHARS, (size_t) 0); };

    // Claude Code: the block goes after the reminder, ahead of the words, with a blank line
    const json code = convert(code_messages("hi"));
    require(code.at("messages").size() == 2 &&
                code.at("messages").at(1) == json({{"role", "user"},
                                                   {"content", json::array({text(reminder), text(code_env + "\n\nhi")})}}),
            "the environment opens the first user message: " + code.at("messages").dump());
    require(context_chars(code) == reminder.size() + code_env.size() + 2, "the request says how long the context is");

    // the Claude desktop app: a string in a reminder, a message without reminders; the words keep their fields
    const json cache_control = {{"type", "ephemeral"}, {"ttl", "1h"}};
    const json desktop = convert(json::array({
        {{"role", "user"}, {"content", json::array({{{"type", "text"}, {"text", "hi"}, {"cache_control", cache_control}}})}},
        {{"role", "system"}, {"content", desktop_env}},
    }));
    require(desktop.at("messages").at(1).at("content") ==
                json::array({{{"type", "text"}, {"text", desktop_env + "\n\nhi"}, {"cache_control", cache_control}}}) &&
                context_chars(desktop) == desktop_env.size() + 2,
            "the desktop app's environment: " + desktop.dump());
    const json words_string = convert(json::array({{{"role", "user"}, {"content", "hi"}},
                                                   {{"role", "system"}, {"content", desktop_env}}}));
    require(words_string.at("messages").at(1).at("content") == desktop_env + "\n\nhi", "words sent as a string");

    // a later turn: the environment opens the conversation as in its first request; a later system message becomes a
    // user message where it stands, as before
    json later_messages = code_messages("hi");
    later_messages.push_back({{"role", "assistant"}, {"content", "Hello."}});
    later_messages.push_back({{"role", "user"}, {"content", "Thanks."}});
    later_messages.push_back({{"role", "system"}, {"content", "<system-reminder>\nLater.\n</system-reminder>"}});
    const json later = convert(later_messages);
    require(later.at("messages").at(1) == code.at("messages").at(1) && later.at("messages").size() == 5 &&
                later.at("messages").at(4) == json({{"role", "user"}, {"content", "<system-reminder>\nLater.\n</system-reminder>"}}),
            "a later turn: " + later.at("messages").dump());

    // requests without that shape convert as before: an environment after a reply, another system message there
    const json after_reply = convert(json::array({{{"role", "user"}, {"content", "hi"}},
                                                  {{"role", "assistant"}, {"content", "Hello."}},
                                                  {{"role", "system"}, {"content", desktop_env}}}));
    require(after_reply.at("messages").at(3) == json({{"role", "user"}, {"content", desktop_env}}) &&
                context_chars(after_reply) == 0,
            "an environment after a reply is demoted where it stands: " + after_reply.dump());
    const json other = convert(json::array({{{"role", "user"}, {"content", "hi"}},
                                            {{"role", "system"}, {"content", "<system-reminder>\nOther.\n</system-reminder>"}}}));
    require(other.at("messages") == json::array({{{"role", "system"}, {"content", system[0].at("text").get<std::string>() +
                                                                                   system[1].at("text").get<std::string>()}},
                                                 {{"role", "user"}, {"content", "hi"}},
                                                 {{"role", "user"}, {"content", "<system-reminder>\nOther.\n</system-reminder>"}}}) &&
                context_chars(other) == 0,
            "another system message after the words is demoted where it stands: " + other.dump());

    // rendered, two conversations of the same setup share the prompt up to where the words start, where the server
    // keeps the context prefix, though a plugin's version folder is shorter in the prompt (0.0.0) than as sent; also
    // with a template that trims each text part
    for (const char * tmpl : {
             "{% for message in messages %}<|im_start|>{{ message.role }}\n{% if message.content is string %}"
             "{{ message.content }}{% else %}{% for part in message.content %}{{ part.text }}{% endfor %}{% endif %}"
             "<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
             "{% for message in messages %}<|im_start|>{{ message.role }}\n{% if message.content is string %}"
             "{{ message.content | trim }}{% else %}{% for part in message.content %}{{ part.text | trim }}{% endfor %}"
             "{% endif %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}" }) {
        server_chat_params opt{};
        opt.use_jinja          = true;
        opt.prefill_assistant  = true;
        opt.reasoning_format   = COMMON_REASONING_FORMAT_DEEPSEEK;
        opt.tmpls = common_chat_templates_init(/* model= */ nullptr, tmpl);
        const auto parse = [&](json chat) {
            std::vector<raw_buffer> files;
            const json params = oaicompat_chat_params_parse(chat, opt, files);
            require(!params.contains(SERVER_FIRST_USER_CONTEXT_CHARS), "the request field is not passed on");
            return params;
        };
        const json a = parse(code);
        const json b = parse(convert(code_messages("What is in this folder?")));
        const std::string pa = a.at("prompt").get<std::string>();
        const std::string pb = b.at("prompt").get<std::string>();
        const size_t at = json_value(a, "context_prefix_chars", (size_t) 0);
        require(pa.find("/run/0.0.0/bin\n") != std::string::npos, "the stand-in: " + pa);
        require(at > json_value(a, "shared_prefix_chars", (size_t) 0) && at == json_value(b, "context_prefix_chars", (size_t) 0) &&
                    pa.compare(0, at, pb, 0, at) == 0,
                std::string("both conversations share the context prefix: ") + tmpl + "\n" + a.dump());
        require(pa.compare(at - 2, 4, "\n\nhi") == 0 && pb.compare(at, 4, "What") == 0,
                "the context prefix ends where the words start: " + pa.substr(at - 20));
        require(json_value(a, "conversation_prefix_chars", (size_t) 0) > at, "the opening holds the words");
        require(!parse(other).contains("context_prefix_chars"), "a request without the environment has no context prefix");
    }
}

// The layout of a request for reuse: the tools of MCP servers after the app's own tools, by server; a state kept
// where each group of tools ends and where the data at the end of the user's context starts, or would start.
static void test_chat_request_reports_branch_points() {
    server_chat_params opt{};
    opt.use_jinja          = true;
    opt.prefill_assistant  = true;
    opt.reasoning_format   = COMMON_REASONING_FORMAT_DEEPSEEK;
    opt.force_pure_content = true;
    // the first system message with the tools in one turn, as Gemma 4 renders them, then the other messages
    opt.tmpls = common_chat_templates_init(/* model= */ nullptr,
            "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}"
            "{% if loop.first and tools %}{% for tool in tools %}<tool>{{ tool.function.name }}</tool>{% endfor %}{% endif %}"
            "<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}");
    const auto tool = [](const char * name) {
        return json{{"type", "function"}, {"function", {{"name", name}, {"description", "d"},
                                                        {"parameters", {{"type", "object"}}}}}};
    };
    const auto parse = [&](json body) {
        std::vector<raw_buffer> files;
        return oaicompat_chat_params_parse(body, opt, files);
    };
    const auto points = [](const json & params) {
        std::vector<std::string> out;
        const std::string prompt = params.at("prompt").get<std::string>();
        for (const auto & p : json_value(params, "branch_points_chars", json::array())) {
            out.push_back(prompt.substr(0, p.get<size_t>()));
        }
        return out;
    };

    json body = {
        {"messages", json::array({{{"role", "system"}, {"content", "App."}},
                                  {{"role", "user"}, {"content", "Hi."}}})},
        {"tools", json::array({tool("mcp__web::fetch"), tool("read"), tool("mcp__docs__search"), tool("write"),
                               tool("mcp__web::open")})},
    };
    const json params = parse(body);
    const std::string system = "<|im_start|>system\nApp.";
    require(params.at("prompt").get<std::string>().rfind(system + "<tool>read</tool><tool>write</tool>"
                "<tool>mcp__docs__search</tool><tool>mcp__web::fetch</tool><tool>mcp__web::open</tool><|im_end|>\n", 0) == 0,
            "the app's tools come first, then each MCP server's in the order of their names: " +
                params.at("prompt").get<std::string>());
    const std::vector<std::string> tools_points = points(params);
    require(tools_points.size() == 3, "a point where each group of tools ends");
    // where the renders part, here one byte into the next tag; the server keeps the state of the whole tokens before
    require(tools_points[0] == system + "<tool>read</tool><tool>write</tool><", "after the app's tools: " + tools_points[0]);
    require(tools_points[1] == system + "<tool>read</tool><tool>write</tool><tool>mcp__docs__search</tool><",
            "after the first server's: " + tools_points[1]);
    require(tools_points[2] == system + "<tool>read</tool><tool>write</tool><tool>mcp__docs__search</tool>"
                "<tool>mcp__web::fetch</tool><tool>mcp__web::open</tool><",
            "after the last tool, where another would go: " + tools_points[2]);

    // the custom tools that the conversion of a Responses request names (SERVER_CUSTOM_TOOLS) come last, after the
    // MCP servers' tools, so a state is kept before them; the field stays in the request
    json with_custom = body;
    with_custom["tools"] = json::array({tool("functions::exec"), tool("read"), tool("mcp__web::fetch"), tool("write")});
    with_custom[SERVER_CUSTOM_TOOLS] = json::array({"functions::exec"});
    const json custom_params = parse(with_custom);
    const std::string own_and_web = system + "<tool>read</tool><tool>write</tool><tool>mcp__web::fetch</tool>";
    require(custom_params.at("prompt").get<std::string>().rfind(own_and_web + "<tool>functions::exec</tool><|im_end|>\n", 0) == 0,
            "the custom tools come after the MCP servers' tools: " + custom_params.at("prompt").get<std::string>());
    require(points(custom_params) == std::vector<std::string>({ system + "<tool>read</tool><tool>write</tool><",
                                                                own_and_web + "<", own_and_web + "<tool>functions::exec</tool><" }),
            "a point where the MCP servers' tools end, before the custom tools: " + json(points(custom_params)).dump());
    require(custom_params.at(SERVER_CUSTOM_TOOLS) == json::array({"functions::exec"}), "the field is passed on");
    with_custom.erase(SERVER_CUSTOM_TOOLS);
    require(parse(with_custom).at("prompt").get<std::string>().rfind(
                system + "<tool>functions::exec</tool><tool>read</tool><tool>write</tool><tool>mcp__web::fetch</tool>", 0) == 0,
            "without the field the tool is one of the app's own");

    // a Responses request's user context with the approved commands at its end
    const std::string data = "\n\n## Approved command prefixes\n- [\"git\", \"status\"]";
    json with_data = {
        {"messages", json::array({{{"role", "system"}, {"content", "App."}},
                                  {{"role", "system"}, {"content", "<permissions instructions>\nAsk.\n</permissions instructions>" + data}},
                                  {{"role", "user"}, {"content", "Hi."}}})},
    };
    with_data[SERVER_USER_CONTEXT_SYSTEM_MESSAGE] = true;
    with_data[SERVER_USER_CONTEXT_DATA_CHARS]     = data.size();
    const json data_params = parse(with_data);
    const std::string context = "<|im_start|>system\nApp.<|im_end|>\n<|im_start|>system\n<permissions instructions>\nAsk.\n</permissions instructions>";
    require(points(data_params) == std::vector<std::string>({ context }), "a point where the data starts");
    require(!data_params.contains(SERVER_USER_CONTEXT_DATA_CHARS), "the request field is not passed on");

    // data that names the folder of a plugin's version, which the model sees as 0.0.0 (inferssd/identifiers.h): the
    // data is shorter in the prompt than as sent, and the point is still where it starts
    const std::string plugin_data =
        "\n\n## Approved command prefixes\n- [\"/opt/.codex/plugins/cache/tools/run/26.1123.11515/bin/run\"]";
    json with_plugin = with_data;
    with_plugin["messages"][1]["content"] = "<permissions instructions>\nAsk.\n</permissions instructions>" + plugin_data;
    with_plugin[SERVER_USER_CONTEXT_DATA_CHARS] = plugin_data.size();
    const json plugin_params = parse(with_plugin);
    require(plugin_params.at("prompt").get<std::string>().find("/run/0.0.0/bin/run") != std::string::npos,
            "the stand-in is in the prompt: " + plugin_params.at("prompt").get<std::string>());
    require(points(plugin_params) == std::vector<std::string>({ context }),
            "the point is where the data starts: " + (points(plugin_params).empty() ? "" : points(plugin_params)[0]));

    json no_data = with_data;
    no_data["messages"][1]["content"] = "<permissions instructions>\nAsk.\n</permissions instructions>";
    no_data.erase(SERVER_USER_CONTEXT_DATA_CHARS);
    require(points(parse(no_data)) == std::vector<std::string>({ context }),
            "a point where the data would start, so that a later context with data finds it");
}

// The conversion of a Responses request moves the data the user adds to over time to the end of the user's context
// and replaces the sub-agent tool's model list.
static void test_responses_request_layout() {
    const std::string permissions = "<permissions instructions>\nApproval policy is `granular`.\n- `skill_approval`";
    const std::string approved    = "\n\n## Approved command prefixes\nThe following prefix rules have already been approved: - [\"git\", \"status\"]";
    const std::string roots       = "\n The writable roots are `/private/tmp`.\n</permissions instructions>";
    const std::string deferred    = "<tools>\nDeferred tool namespaces:\n- mcp__node_repl\n</tools>";
    const std::string mode        = "<collaboration_mode># Collaboration Mode: Default</collaboration_mode>";
    const std::string models      = "Available model overrides (optional; inherited parent model is preferred):\n"
                                    "- `gpt-6.2-sol`: Fast. Reasoning efforts: low (default), high.\n"
                                    "- `gpt-6.1-sol`: Previous. Reasoning efforts: low (default).\n";
    const json body = {
        {"model", "m"},
        {"input", json::array({
            {{"role", "developer"}, {"content", json::array({{{"type", "input_text"}, {"text", "You are Codex."}}})}},
            {{"role", "developer"}, {"content", json::array({{{"type", "input_text"}, {"text", permissions + approved + roots}},
                                                             {{"type", "input_text"}, {"text", deferred}},
                                                             {{"type", "input_text"}, {"text", mode}}})}},
            {{"role", "user"}, {"content", "hi"}},
        })},
        {"tools", json::array({{{"type", "namespace"}, {"name", "multi_agent_v1"}, {"description", "Sub-agents."},
                                {"tools", json::array({{{"type", "function"}, {"name", "spawn_agent"},
                                                        {"description", "Spawns an agent.\n\n" + models + "\nDo not spawn unless asked."},
                                                        {"parameters", {{"type", "object"}}}}})}}})},
    };
    const json chat = server_chat_convert_responses_to_chatcmpl(body);
    const std::string context = chat.at("messages").at(1).at("content").get<std::string>();
    const std::string moved   = approved.substr(2) + "\n\n" + deferred;
    require(context == permissions + roots + "\n\n" + mode + "\n\n" + moved,
            "the approved commands and the deferred tools go to the end of the context: " + context);
    require(json_value(chat, SERVER_USER_CONTEXT_DATA_CHARS, (size_t) 0) == moved.size() + 2,
            "the request says how long the data at the end is");
    require(chat.at("tools").at(0).at("function").at("description").get<std::string>() ==
                "Sub-agents.\nSpawns an agent.\n\nModel overrides: none; the new agent runs on the parent's model.\n"
                "\nDo not spawn unless asked.",
            "the model list becomes one sentence: " + chat.at("tools").at(0).at("function").at("description").get<std::string>());
}

// The ChatGPT desktop app's app context in its detail modes, trimmed from what the app sends: Prose adds the sections
// Non-technical UI and Writing blocks to what Commands sends. They go to the end of the user's context, so a request
// of either mode renders the same up to where the data starts, and the server keeps a state there.
static void test_chatgpt_detail_modes_share_the_context() {
    const std::string head =
        "<app-context>\n# Codex desktop context\n"
        "- You are running inside the Codex (desktop) app, which allows some additional features not available in the "
        "CLI alone:\n\n"
        "### Thread Coordination\n"
        "- Only use `create_thread` when the user explicitly asks to create a new thread.\n";
    const std::string non_technical =
        "### Non-technical UI\n"
        "- The user has requested a non-technical UI.\n"
        "- However, if the user asks for detail or it would help the user debug, you can still decide to dive into "
        "technical details.";
    const std::string comments =
        "### Inline Code Comments\n"
        "- Keep line ranges tight; end defaults to start.\n"
        "</app-context>";
    const std::string writing =
        "### Writing blocks\n\n"
        "- A writing block contains a finished, reusable writing artifact that the user can copy, edit, or use outside "
        "this conversation.\n"
        "- Use this syntax for non-email drafts:\n\n"
        ":::writing{variant=\"<variant>\" id=\"<id>\" title=\"<title>\"}\n<content>\n:::\n\n"
        "- If distinct tone or style choices would materially help the user, put two or three alternatives in one "
        "writing block and start every alternative with a line in this exact form:\n\n"
        "---tone <label>\n<alternative content>\n\n"
        "- Keep any explanation outside the writing block and do not mention this formatting contract to the user.";
    const std::string projectless =
        "### Projectless Chat\n"
        "Do not write directly in the home directory unless the user explicitly asks.";
    const std::string commands = head + "\n" + comments + "\n\n" + projectless;
    const std::string prose    = head + "\n" + non_technical + "\n\n" + comments + "\n\n" + writing + "\n\n" + projectless;
    const std::string permissions = "<permissions instructions>\nApproval policy is `never`.\n</permissions instructions>";
    const std::string deferred    = "<tools>\nDeferred tool namespaces:\n- mcp__node_repl\n</tools>";

    const auto convert = [&](const std::string & app_context, const std::string & last) {
        json parts = json::array({{{"type", "input_text"}, {"text", app_context}},
                                  {{"type", "input_text"}, {"text", permissions}}});
        if (!last.empty()) {
            parts.push_back({{"type", "input_text"}, {"text", last}});
        }
        return server_chat_convert_responses_to_chatcmpl({
            {"model", "m"},
            {"input", json::array({
                {{"role", "developer"}, {"content", json::array({{{"type", "input_text"}, {"text", "You are Codex."}}})}},
                {{"role", "developer"}, {"content", parts}},
                {{"role", "user"}, {"content", "hi"}},
            })},
        });
    };
    const auto context_of = [](const json & chat) { return chat.at("messages").at(1).at("content").get<std::string>(); };
    const auto data_of    = [](const json & chat) { return json_value(chat, SERVER_USER_CONTEXT_DATA_CHARS, (size_t) 0); };

    const json chat_commands = convert(commands, deferred);
    const json chat_prose    = convert(prose, deferred);
    const std::string rest   = commands + "\n\n" + permissions;
    require(context_of(chat_commands) == rest + "\n\n" + deferred,
            "a context without the sections stays as sent: " + context_of(chat_commands));
    const std::string moved = non_technical + "\n\n" + writing + "\n\n" + deferred;
    require(context_of(chat_prose) == rest + "\n\n" + moved,
            "the sections go to the end of the context, before the deferred tools: " + context_of(chat_prose));
    require(data_of(chat_prose) == moved.size() + 2, "the sections are data");

    // either section alone, and no other data
    const json only_non_technical = convert(head + "\n" + non_technical + "\n\n" + comments + "\n\n" + projectless, "");
    require(context_of(only_non_technical) == rest + "\n\n" + non_technical && data_of(only_non_technical) == non_technical.size() + 2,
            "Non-technical UI alone: " + context_of(only_non_technical));
    const json last_section = convert(head + "\n" + comments + "\n\n" + writing, "");
    require(context_of(last_section) == head + "\n" + comments + "\n\n" + permissions + "\n\n" + writing,
            "a section that ends the text: " + context_of(last_section));

    // the headings in another part stay
    const json elsewhere = convert(commands, "<skills_instructions>\n\n### Writing blocks\nA skill.\n</skills_instructions>");
    require(context_of(elsewhere) == rest + "\n\n<skills_instructions>\n\n### Writing blocks\nA skill.\n</skills_instructions>" &&
                data_of(elsewhere) == 0,
            "a heading outside the app context stays: " + context_of(elsewhere));

    // rendered, both modes share the prompt up to the state the server keeps where the data starts
    server_chat_params opt{};
    opt.use_jinja          = true;
    opt.prefill_assistant  = true;
    opt.reasoning_format   = COMMON_REASONING_FORMAT_DEEPSEEK;
    opt.force_pure_content = true;
    opt.tmpls = common_chat_templates_init(/* model= */ nullptr,
            "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}");
    const auto render = [&](json chat) {
        std::vector<raw_buffer> files;
        const json params = oaicompat_chat_params_parse(chat, opt, files);
        const std::string prompt = params.at("prompt").get<std::string>();
        const json points = json_value(params, "branch_points_chars", json::array());
        require(points.size() == 1, "a point where the data starts: " + params.dump());
        return std::make_pair(prompt, points.at(0).get<size_t>());
    };
    const auto [prompt_commands, at_commands] = render(chat_commands);
    const auto [prompt_prose, at_prose]       = render(chat_prose);
    require(at_commands == at_prose && prompt_commands.compare(0, at_commands, prompt_prose, 0, at_prose) == 0,
            "both modes render the same up to the data: " + prompt_commands.substr(0, at_commands) + " | " +
                prompt_prose.substr(0, at_prose));
    require(prompt_prose.compare(at_prose, 2 + non_technical.size(), "\n\n" + non_technical) == 0,
            "and part where the data starts: " + prompt_prose.substr(at_prose));
}

int main() {
    try {
        test_supersedes();
        test_memory_cache();
        test_disk_cache();
        test_chat_request_reports_the_opening();
        test_claude_environment_opens_the_conversation();
        test_chat_request_reports_branch_points();
        test_responses_request_layout();
        test_chatgpt_detail_modes_share_the_context();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }

    return 0;
}
