// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <utility>
#include <vector>

// Identifiers of the user, the machine and the session in the context a coding agent starts every conversation with,
// and the fixed stand-ins the model sees in their place.
//
// A prompt is cached as a prefix: the state of an app's leading context (its system prompt and tools, 10k-50k tokens)
// is shared only up to the first byte that differs. Claude Code writes its memory directory into that context (the
// home directory and the project), the Claude desktop app the folders of the session, Codex and the ChatGPT desktop
// app the skill folders under the home directory, and the ChatGPT desktop app the conversation's folder. A new
// project, session or user would process all of the context after them again. The server gives the model stand-ins
// for these identifiers wherever they occur in a request, and puts the request's identifiers back in place of the
// stand-ins in what the model writes. The model only copies such strings into paths, so it does the same work, and
// the client sees its own paths. The user name in the stand-ins is infer: with inferssd, three tokens to Qwen3.6,
// Qwen3.6-35B-A3B left out the .claude after it in 3 of 4 copies of Claude Code's memory folder, with infer in none
// of 6.
//
// The identifiers are the home directory, the user's temporary directory, and the folders the apps name in their
// system and developer messages (see identifiers.cpp). A match must start and end at a name boundary. A request whose
// text the stand-ins cannot be taken back out of exactly (one that has a stand-in in it already, say) is left as it is.
struct server_identifier_map {
    // (identifier, stand-in), every spelling; empty: the request is left as it is
    std::vector<std::pair<std::string, std::string>> pairs;

    bool empty() const { return pairs.empty(); }

    // the text with stand-ins in place of the identifiers
    std::string to_stand_ins(const std::string & text) const;

    // the text with the identifiers in place of the stand-ins. With partial, the text is still growing: the tail that
    // may yet turn out to be a stand-in is left out, so that each result is a prefix of the next one.
    std::string to_identifiers(const std::string & text, bool partial) const;

    nlohmann::ordered_json to_json() const;
    static server_identifier_map from_json(const nlohmann::ordered_json & value);
};

// The identifiers of the machine the server runs on, and the stand-ins for them.
struct server_identifier_env {
    std::string home;             // the user's home directory, e.g. C:\Users\name
    std::string temp;             // the user's temporary directory if it is outside the home directory (macOS)
    std::string home_stand_in;    // a home directory of a user named infer: C:\Users\infer on Windows
    std::string temp_stand_in;    // /var/folders/00/infer/T
    bool        windows = false;  // paths are spelled with '\' as well as '/'

    static const server_identifier_env & current();
};

// Puts stand-ins in place of the identifiers in the messages and tools of a chat request (OpenAI chat completions
// form), and returns the map that takes them back; empty when the request has none or cannot be mapped exactly.
server_identifier_map server_identifiers_apply(nlohmann::ordered_json & body, const server_identifier_env & env);

// Codex names the project's folder in that context as well, as a writable root in its permission instructions
// (" The writable roots are `<cwd>`, `/private/tmp`."), after it says that the sandbox permits "editing files in `cwd`
// and `writable_roots`". The environment Codex sends in a user message, after the tools, names the folder:
// <cwd>...</cwd>. The root that is the cwd goes from the sentence, which reads as Codex writes it for the roots left,
// and the model learns which folder the cwd is from the environment. The project keeps its place among the other
// folders, which a stand-in for it would not: a folder next to the project, say, would seem to be somewhere else.
void server_drop_codex_cwd_root(nlohmann::ordered_json & messages);

// Claude Code, and the Claude desktop app's agent, start their system prompt with a line such as
// "x-anthropic-billing-header: cc_version=2.1.328.ce7; cc_entrypoint=sdk-cli;" (older versions add "cch=a5145;").
// Its values follow the conversation's first message or change with each request, so a prompt that keeps the line
// shares no cached state past it. The line is for Anthropic's billing, not for the model: it is removed from the
// system messages, as vLLM does. Text after it in the same message stays.
void server_strip_anthropic_billing_header(nlohmann::ordered_json & messages);

// The same for one text; returns whether the line was there.
bool server_strip_anthropic_billing_header(std::string & text);
