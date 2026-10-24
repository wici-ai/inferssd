// SPDX-License-Identifier: MIT
#include "../chat.h"
#include "../common.h"
#include <algorithm>
#include <string_view>

void common_chat_parser_params::set_vocab(const llama_vocab * vocab) {
    control_token_pieces.clear();
    for (llama_token id = 0; id < llama_vocab_n_tokens(vocab); ++id) {
        if (llama_vocab_is_control(vocab, id)) {
            const auto piece = common_token_to_piece(vocab, id, true);
            if (!piece.empty()) {
                control_token_pieces.emplace(id, piece);
            }
        }
    }
}

bool common_chat_source_allows_literal(const common_chat_parser_params & params,
                                      const common_chat_output_source & source,
                                      size_t input_size, size_t pos, std::string_view literal,
                                      size_t trusted_prefix_bytes) {
    if (!source.valid) {
        return true; // legacy text-only callers have no token-origin information
    }
    for (const auto & token : params.control_token_pieces) {
        size_t off = literal.find(token.second);
        while (off != std::string_view::npos) {
            const size_t begin = pos + off;
            if (begin >= input_size) {
                break; // no input for this part of a partial marker yet
            }
            const size_t end = begin + token.second.size();
            if (begin < trusted_prefix_bytes) {
                if (end > trusted_prefix_bytes) {
                    return false;
                }
            } else {
                const size_t generated_begin = begin - trusted_prefix_bytes;
                const auto it = std::lower_bound(source.controls.begin(), source.controls.end(), generated_begin,
                    [](const common_chat_control_token & t, size_t value) { return t.begin < value; });
                if (it == source.controls.end() || it->begin != generated_begin ||
                        it->end - it->begin != token.second.size() || it->id != token.first) {
                    return false;
                }
            }
            off = literal.find(token.second, off + token.second.size());
        }
    }
    return true;
}

size_t common_chat_find_control(const std::string & text, const std::string & marker,
                               const common_chat_parser_params & params,
                               const common_chat_output_source & source) {
    size_t pos = text.find(marker);
    while (pos != std::string::npos &&
           !common_chat_source_allows_literal(params, source, text.size(), pos, marker)) {
        pos = text.find(marker, pos + 1);
    }
    return pos;
}

