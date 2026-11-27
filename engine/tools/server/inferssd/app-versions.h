// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nlohmann/json.hpp>

#include <vector>

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

// Claude Code's ToolSearch tool requires its parameters query and max_results in 2.1.328 and only query in 2.1.332.
// A Claude Code request's ToolSearch (an Anthropic tool) that requires both requires only query, as 2.1.332 sends it:
// 2.1.328 defines max_results as optional, with a default of 5, and takes a call without it as well.
void server_unify_claude_code_tool(nlohmann::ordered_json & tool);

// Codex's exec_command tool describes its parameter login one way in 0.160 to 0.170 (also in the ChatGPT desktop app)
// and another in 0.172, which says it differently on macOS and on Windows. The earlier description becomes the 0.172
// one of the system whose requests the server renders (server_identifier_env::current): Windows', or macOS' on the
// others. Codex 0.170 and 0.172 have three worktree tools in the namespace mcp__codex_tui that 0.160 lacks
// (create_worktree, get_worktree_creation_status and list_worktrees): a namespace without one gets it, as 0.172 defines
// it, after the tool it follows there. The ChatGPT desktop app's custom tool exec (namespace functions) lists the
// JavaScript API of each tool set the app offers it, and some of its requests lack the section of the Codex app's tools
// (mcp__codex_app): a description that ends with the section clock and lacks it gets it there, as the app sends it on
// the system whose requests the server renders. tool is a Responses tool.
void server_unify_codex_tool(nlohmann::ordered_json & tool, bool windows);

// Codex and the ChatGPT desktop app offer the tools of the MCP servers and plugins configured on the computer, which
// the apps installed there add: the ChatGPT desktop app adds mcp__node_repl to Codex's configuration, and its computer
// use plugin offers mcp__cua_repl, so the requests of two users of the same app version have them or not. A request
// of Codex (it has the function exec_command) that lacks one of them gets it at its end, and a request of a ChatGPT
// desktop app conversation (the namespace collaboration, send_message_to_user_async in the namespace functions, no
// exec_command) mcp__cua_repl, as the app versions named in app-versions.cpp define them on the system whose requests
// the server renders; the app lists mcp__node_repl among the deferred tools of its context instead, and the requests
// that title its conversations never offer either. A definition the request has stays as sent. The server orders the
// tools of MCP servers by server after the app's own (server_order_tools). The model may then call such a tool where
// the user's app does not offer it, and the app answers the call with an error. tools are Responses tools.
void server_add_codex_common_tools(std::vector<nlohmann::ordered_json> & tools, bool windows);
