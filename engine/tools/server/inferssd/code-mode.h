// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

// Codex's code mode. For the models it runs in code mode (every GPT model of Codex 0.172; the ChatGPT desktop app
// always picks one), Codex offers no tool of its own at the top level: it offers the custom tool exec, which runs the
// JavaScript the model writes, and declares its tools only inside exec's description, as TypeScript the scripts call
// them by:
//   ### `exec_command`
//   Runs a command in a PTY, returning output or a session ID for ongoing interaction.
//
//   exec tool declaration:
//   ```ts
//   declare const tools: { exec_command(args: {
//     // Shell command to execute.
//     cmd: string;
//     ...
//   }): Promise<{ ... }>; };
//   ```
// Open models are trained to call functions, not to write scripts that call them, and run poorly with exec alone. The
// server offers them each declared tool as a function instead, with the parameters Codex wrote the declaration from,
// and makes each call of one an exec call of a fixed script that calls the tool (server_code_mode_input). In a later
// request such an exec call is the function call again (server_code_mode_call), so the conversation renders as the
// model wrote it.

// A tool that a code-mode exec declares, as the prompt offers it in exec's place.
struct server_code_mode_tool {
    std::string            name;         // as the declaration names it, the name the model calls
    std::string            description;  // the text of its section
    bool                   string_input; // it takes a string (a free-form tool such as apply_patch), not an object
    nlohmann::ordered_json parameters;   // JSON Schema of its object argument; null for a string input
};

// The tools a custom tool named exec declares in its description: one for each "### `name`" section that holds the text
// and the declaration Codex writes. The declaration's argument type becomes JSON Schema: `// ` comment lines before a
// member are its description, `?` makes it optional, string, number and boolean are those types, null and unknown are
// null and any value, a union of literals is an enum, Array<T> and T[] are arrays, objects are objects whose other
// properties are not allowed (Record<K, V> and index signatures allow them), and any other union is anyOf. Empty when
// the tool is not such an exec or a section's declaration does not read that way: the request then offers exec itself.
std::vector<server_code_mode_tool> server_code_mode_tools(const nlohmann::ordered_json & exec);

// The script exec runs for a call of a declared tool with these arguments (JSON, as the model wrote them): it awaits
// tools.<name>(<the arguments, or the input of a string tool>) and returns the result as text, a command's output as it
// is with the result's other fields as JSON before it.
std::string server_code_mode_input(const std::string & name, const std::string & arguments, bool string_input);

// When an exec input is the script server_code_mode_input writes for a call of one of the tools: the call's name and
// its arguments (JSON).
bool server_code_mode_call(const std::string & input, const std::vector<server_code_mode_tool> & tools,
                           std::string & name, std::string & arguments);
