// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nlohmann/json.hpp>

// Texts in which versions or variants of an agent app differ while they tell the model the same, and the one text the
// server renders for all of them, so that their users share the cached prefix past them (as with the identifiers in
// identifiers.h). Each known text comes from requests that a named version of the app sent (the tables in
// app-versions.cpp); a text that differs from the known ones in any way is left as it is.

// Whether the system blocks of an Anthropic Messages request are Claude Code's: the second one is its identity line,
// "You are Claude Code, Anthropic's official CLI for Claude."
bool server_is_claude_code(const nlohmann::ordered_json & system);

// Claude Code 2.1.328 sends one of two variants of its system prompt, on macOS and on Windows alike: one has two
// passages that the other lacks, a bullet on the text inside <pasted_content> tags and a paragraph on keeping private
// what it learns about the user. A block after the identity line of a Claude Code request (server_is_claude_code) that
// lacks a passage gets it where the other variant has it: between the same two lines, when the block has those lines
// next to each other and the passage nowhere.
void server_unify_claude_code_system(nlohmann::ordered_json & system);
