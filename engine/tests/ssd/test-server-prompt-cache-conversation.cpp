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
}

int main() {
    try {
        test_supersedes();
        test_memory_cache();
        test_disk_cache();
        test_chat_request_reports_the_opening();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }

    return 0;
}
