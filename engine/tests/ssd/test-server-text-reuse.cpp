// server_tokens_reuse_text: a prompt that continues, as text, the cached tokens it shares a prefix with takes their
// token ids for that text, up to the furthest offset where both end a token; tokens without text match by id.
#include "server-common.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "TEXT_REUSE_FAIL: %s\n", message);
        std::exit(1);
    }
}

// token id -> text; ids name their text so that a test reads as the strings it tokenizes. 1, 7 and 10 are control
// tokens, 14 has no text, 15-17 spell 7's text literally.
static const std::map<llama_token, std::string> vocab = {
    { 1, "<s>" }, { 2, "Hello" }, { 3, " wor" }, { 4, "ld" }, { 5, " world" }, { 6, "!" }, { 7, "<end>" },
    { 8, " wo" }, { 9, "rld" }, { 10, "<user>" }, { 11, " next" }, { 12, " w" }, { 13, "x" }, { 14, "" },
    { 15, "<" }, { 16, "end" }, { 17, ">" },
};
// as the server gives it: no text for a control token
static std::string piece(llama_token t) { return t == 1 || t == 7 || t == 10 ? std::string() : vocab.at(t); }

static void test_reply_tokenized_back_differently() {
    // generated " wor" "ld" "!", the history renders the same text back as " world" "!"
    const llama_tokens cached = { 1, 2, 3, 4, 6 };
    llama_tokens prompt = { 1, 2, 5, 6, 7, 10, 11 };
    const size_t n = server_tokens_reuse_text(cached, prompt, piece);
    require(n == 5, "the reply's cached ids were not taken over");
    require(prompt == llama_tokens({ 1, 2, 3, 4, 6, 7, 10, 11 }), "the prompt after the reply changed");
}

static void test_partial_text_stops_at_a_shared_boundary() {
    // cached " wo" "rld" "!", prompt " world" "x": the texts agree on " world", where both end a token
    const llama_tokens cached = { 1, 2, 8, 9, 6 };
    llama_tokens prompt = { 1, 2, 5, 13 };
    const size_t n = server_tokens_reuse_text(cached, prompt, piece);
    require(n == 4 && prompt == llama_tokens({ 1, 2, 8, 9, 13 }), "the shared text up to its last common boundary was not taken");
}

static void test_no_common_boundary() {
    // cached " wor" "ld", prompt " wo" "rld": the texts agree, but no offset inside ends a token of both
    // until the end, where they do
    const llama_tokens cached = { 1, 2, 3, 4 };
    llama_tokens prompt = { 1, 2, 8, 9, 6 };
    require(server_tokens_reuse_text(cached, prompt, piece) == 4 && prompt == llama_tokens({ 1, 2, 3, 4, 6 }),
            "the end of the cached text is a boundary of both");
    // cached " w" "x", prompt " wo": the texts part inside the prompt's token
    const llama_tokens cached2 = { 1, 2, 12, 13 };
    llama_tokens prompt2 = { 1, 2, 8, 9 };
    require(server_tokens_reuse_text(cached2, prompt2, piece) == 2 && prompt2 == llama_tokens({ 1, 2, 8, 9 }),
            "a prompt whose text parts from the cache was changed");
}

static void test_text_differs() {
    const llama_tokens cached = { 1, 2, 3, 4 };
    llama_tokens prompt = { 1, 2, 11, 6 };
    require(server_tokens_reuse_text(cached, prompt, piece) == 2 && prompt == llama_tokens({ 1, 2, 11, 6 }),
            "a different text took cached ids");
}

static void test_extension_and_short_prompt() {
    const llama_tokens cached = { 1, 2, 3 };
    llama_tokens prompt = { 1, 2, 3, 4, 6 };
    require(server_tokens_reuse_text(cached, prompt, piece) == 3 && prompt == llama_tokens({ 1, 2, 3, 4, 6 }),
            "a prompt that extends the cache was changed");
    // the prompt's text ends inside the cached text: its tokens up to there are replaced
    const llama_tokens cached2 = { 1, 2, 3, 4, 6 };
    llama_tokens prompt2 = { 1, 2, 5 };
    require(server_tokens_reuse_text(cached2, prompt2, piece) == 4 && prompt2 == llama_tokens({ 1, 2, 3, 4 }),
            "a prompt ending inside the cached text did not take its ids");
}

static void test_tokens_without_text() {
    // a cached token with no text at the shared boundary is not put into the prompt
    const llama_tokens cached = { 1, 2, 3, 4, 14, 6 };
    llama_tokens prompt = { 1, 2, 5, 13 };
    require(server_tokens_reuse_text(cached, prompt, piece) == 4 && prompt == llama_tokens({ 1, 2, 3, 4, 13 }),
            "a cached token without text was taken into the prompt");
    // nor inside a span the texts share: the match stops before it
    llama_tokens prompt2 = { 1, 2, 5, 6 };
    require(server_tokens_reuse_text(cached, prompt2, piece) == 4 && prompt2 == llama_tokens({ 1, 2, 3, 4, 6 }),
            "the match went past a cached token without text");
}

static void test_control_tokens() {
    // the reply's control token against its literal spelling in the prompt: the match stops before it
    const llama_tokens cached = { 1, 2, 3, 4, 7 };
    llama_tokens prompt = { 1, 2, 5, 15, 16, 17, 6 };
    require(server_tokens_reuse_text(cached, prompt, piece) == 4 && prompt == llama_tokens({ 1, 2, 3, 4, 15, 16, 17, 6 }),
            "a control token took the place of its literal spelling");
    // the literal spelling cached against the control token in the prompt
    const llama_tokens cached2 = { 1, 2, 3, 4, 15, 16, 17 };
    llama_tokens prompt2 = { 1, 2, 5, 7, 10 };
    require(server_tokens_reuse_text(cached2, prompt2, piece) == 4 && prompt2 == llama_tokens({ 1, 2, 3, 4, 7, 10 }),
            "a literal spelling took the place of a control token");
    // the same control tokens at shared boundaries match by id, and the text after them continues
    const llama_tokens cached3 = { 1, 2, 3, 4, 7, 10, 8, 9 };
    llama_tokens prompt3 = { 1, 2, 5, 7, 10, 8, 9, 6 };
    require(server_tokens_reuse_text(cached3, prompt3, piece) == 8 && prompt3 == llama_tokens({ 1, 2, 3, 4, 7, 10, 8, 9, 6 }),
            "equal control tokens did not match by id");
    // a control token must sit where both texts end a token
    const llama_tokens cached4 = { 1, 2, 12, 7 };
    llama_tokens prompt4 = { 1, 2, 8, 7 };
    require(server_tokens_reuse_text(cached4, prompt4, piece) == 2 && prompt4 == llama_tokens({ 1, 2, 8, 7 }),
            "a control token matched inside a token's text");
}

int main() {
    test_reply_tokenized_back_differently();
    test_partial_text_stops_at_a_shared_boundary();
    test_no_common_boundary();
    test_text_differs();
    test_extension_and_short_prompt();
    test_tokens_without_text();
    test_control_tokens();
    std::printf("TEXT_REUSE_PASS\n");
    return 0;
}
