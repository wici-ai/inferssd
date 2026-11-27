// Chat conversion functions for server (Responses API, Anthropic API, OAI streaming diffs)

#pragma once

#include "chat.h"
#include "server-common.h"
#include "server-http.h"

#include <nlohmann/json_fwd.hpp>

using json = nlohmann::ordered_json;

// Convert OpenAI Responses API format to OpenAI Chat Completions API format
json server_chat_convert_responses_to_chatcmpl(const json & body);

// The input of a call to a custom tool of a Responses request, which the model writes as the string argument "input"
// (SERVER_CUSTOM_TOOLS). Of arguments still being written, the text of that string written so far, up to its last
// whole escape sequence; empty until the arguments open with it. Of finished arguments, the string, or the arguments
// as they are when they hold no such string.
std::string server_chat_custom_tool_input(const std::string & arguments, bool partial);

// Convert Anthropic Messages API format to OpenAI Chat Completions API format
json server_chat_convert_anthropic_to_oai(const json & body);

// convert OpenAI transcriptions API format to OpenAI Chat Completions API format
json convert_transcriptions_to_chatcmpl(
    const json & body,
    const common_chat_templates * tmpls,
    const std::map<std::string, uploaded_file> & in_files,
    std::vector<raw_buffer> & out_files);

json server_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff);
