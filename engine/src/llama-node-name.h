// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cctype>
#include <cstring>
#include <initializer_list>

// Whether a graph node's name is one the model gave it, and so names the same op in another graph of the model: the
// graph builder's callback names a node by its role and layer (`attn_norm-3`), and ggml derives a view's or a copy's
// name from it (`attn_q-3 (view)`). ggml names a node the model left unnamed by its position in the graph
// (`node_2036`, `leaf_17`), and a view of an unnamed tensor gets the suffix alone (` (view)`): in another graph such a
// name is another op.
inline bool llama_node_name_is_given(const char * name) {
    if (name == nullptr || name[0] == '\0' || name[0] == ' ') {
        return false;
    }
    for (const char * prefix : { "node_", "leaf_" }) {
        const size_t n = std::strlen(prefix);
        if (std::strncmp(name, prefix, n) != 0) {
            continue;
        }
        const char * p = name + n;
        if (!std::isdigit((unsigned char) *p)) {
            continue;
        }
        while (std::isdigit((unsigned char) *p)) {
            ++p;
        }
        if (*p == '\0' || *p == ' ') {
            return false;
        }
    }
    return true;
}
