// SPDX-License-Identifier: MIT
#pragma once

#include "../server-common.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <list>
#include <string>
#include <vector>

struct server_prompt_checkpoint {
    llama_pos pos_min;
    llama_pos pos_max;

    int64_t n_tokens;

    std::vector<uint8_t> data;

    // a checkpoint of a prompt the cache keeps on disk has its data in the prompt's file instead
    size_t file_offset = 0;
    size_t file_size   = 0;

    size_t size() const {
        return data.empty() ? file_size : data.size();
    }

    bool empty() const {
        return size() == 0;
    }

    void clear() {
        pos_min = 0;
        pos_max = 0;
        n_tokens = 0;
        data.clear();
        file_offset = 0;
        file_size   = 0;
    }
};

struct server_prompt {
    server_tokens tokens;

    std::vector<uint8_t> data;

    std::list<server_prompt_checkpoint> checkpoints;

    // a prompt the cache keeps on disk: its state and its checkpoints' data are in this file instead
    std::string file;
    size_t      file_data_offset = 0;
    size_t      file_data_size   = 0;

    // how many of the tokens every state of this prompt's conversation starts with: the prefix other conversations
    // share (the leading system messages and tools) and the user messages before the first reply; 0 if unknown
    // (not a chat prompt, no system message, or a prompt as the cache found it on disk)
    int64_t n_conversation = 0;

    // a prefix cache (server_prompt_cache::use_prefix_dir): made by another server for this model, read from its
    // file and never written, removed or counted against the cache's limits
    bool read_only = false;

    size_t size() const {
        size_t res = file.empty() ? data.size() : file_data_size;

        for (const auto & checkpoint : checkpoints) {
            res += checkpoint.size();
        }

        return res;
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            data,
            checkpoints,
            file,
            file_data_offset,
            file_data_size,
            n_conversation,
            read_only,
        };
    }
};

// For a state that cannot be rolled back: how many tokens of `tokens_new` a prompt lets a slot skip, restoring its
// whole state (when that is a prefix of tokens_new) or else its last checkpoint inside the common prefix, always
// leaving a token of tokens_new to process for the logits; -1 if neither fits. *checkpoint is the checkpoint to
// restore, nullptr for the whole state.
int64_t server_prompt_reuse_full(
        const server_prompt              & prompt,
        const server_tokens              & tokens_new,
        const server_prompt_checkpoint ** checkpoint);

// For a memory that is checkpointed only between batches: whether a prompt batch stops after n_done of the prompt's
// n_prompt tokens, so that a checkpoint can be taken there. It stops where a later prompt is likely to branch off: at
// the end of the shared prefix (n_shared, the leading system messages and tools; 0 for none) and 4 tokens before the
// end of the prompt (the generation prompt may be rendered differently in the next turn).
inline bool server_prompt_batch_stops(int64_t n_done, int64_t n_prompt, int64_t n_shared, int64_t n_batch) {
    const int64_t n_end = n_prompt - std::min<int64_t>(n_batch, 4);
    return n_done == n_end || (n_done == n_shared && n_done >= 64 && n_done < n_end);
}

// Whether `prompt` is a newer state of the conversation that the cached prompt `old` is a state of, and takes its
// place in the prompt cache. The two are states of one conversation when they share the conversation's opening
// (prompt.n_conversation tokens): the system messages, the tools and the user messages before the first reply. A
// conversation that only starts behind the same system messages and tools parts from the prompt before that, and
// the shared prefix kept for all of them is shorter than it. The newer state replaces only one that is not longer:
// a short request made beside a long conversation, with the same opening, leaves the long state in place.
inline bool server_prompt_supersedes(const server_prompt & prompt, const server_prompt & old) {
    const int64_t n = prompt.n_conversation;
    if (n <= 0 || n >= prompt.n_tokens() || old.n_tokens() > prompt.n_tokens()) {
        return false;
    }
    return (int64_t) old.tokens.get_common_prefix(prompt.tokens) >= n;
}

inline const server_prompt_checkpoint * server_prompt_find_checkpoint_before_pos(
        const server_prompt & prompt,
        llama_pos             p0) {
    for (auto it = prompt.checkpoints.rbegin(); it != prompt.checkpoints.rend(); ++it) {
        if (it->pos_max < p0 && it->n_tokens >= 0 && (size_t) it->n_tokens <= prompt.tokens.size()) {
            return &*it;
        }
    }

    return nullptr;
}

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens) {
        this->limit_size   = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;
    }

    // Keep the prompts in files under `dir` instead of in memory. They outlive the server: prompts saved by a server
    // with the same fingerprint (model file, build) are found again on start, other files are removed. Loading a
    // prompt leaves its file in place, so a shared prefix serves every conversation that starts with it.
    void use_dir(const std::string & dir, const std::string & fingerprint);

    bool on_disk() const {
        return !dir.empty();
    }

    // Also restore from the prefix caches in `dir`: cached prompts that another server made for this model and that
    // came with it (a download), each the state after the shared prefix of an app's requests: files of that server's
    // prompt cache on disk, whose fingerprint is `fingerprint` (the model, not where it is) and the build that saved
    // them, any build. The cache only reads them: it
    // neither removes them nor counts them against its limits, and it reads the directory again when the directory
    // changes, so a file put there while the server runs serves the next request.
    void use_prefix_dir(const std::string & dir, const std::string & fingerprint);

    std::list<server_prompt> states; // least recently used first

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit (memory only)
    size_t limit_tokens = 0;

    std::string dir;
    std::string fingerprint;

    std::string prefix_dir;
    std::string prefix_fingerprint;

    // of the cache's own states, not the prefix caches
    size_t size() const;

    size_t n_tokens() const;

    // Save the slot's state for `prompt`, with the prompt's checkpoints, unless the cache can already restore it.
    // The older states of the same conversation go (server_prompt_supersedes): a conversation whose history is
    // rendered or sent differently from one turn to the next leaves states that no later prompt extends.
    // The state is the current one (current = true) or the one the sequence had after prompt.tokens, which needs a
    // memory that can save an earlier state (llama_state_seq_can_save_at).
    void save(
            const server_prompt        & prompt,
            bool                         current,
            llama_context              * ctx,
            int32_t                      id_slot,
            common_context_seq_rm_type   seq_rm_type);

    // Load into the slot the cached state that lets it skip the most of `tokens_new`, if that is more than
    // `n_reuse_slot`, the number of tokens the slot can already reuse.
    bool load(
            server_prompt              & prompt,
            const server_tokens        & tokens_new,
            int64_t                      n_reuse_slot,
            llama_context              * ctx,
            int32_t                      id_slot,
            common_context_seq_rm_type   seq_rm_type);

    void update();

private:
    friend struct server_prompt_cache_testing; // Explicit metadata-only unit fixture; no production admission shortcut.
    void remove_obsolete(const server_prompt & prompt, common_context_seq_rm_type seq_rm_type);
    // removes the states `prompt` supersedes, except `saved`, the cache's own entry of it
    void remove_superseded(const server_prompt & prompt, const server_prompt * saved);
    bool contains(const server_prompt & prompt, common_context_seq_rm_type seq_rm_type);
    void remove(std::list<server_prompt>::iterator it);
    // reads the prefix cache directory again if it changed since the last read
    void scan_prefix_dir();
    int64_t prefix_dir_time = -1;
    // prefix caches that failed to restore here (another state layout): not taken again from the directory
    std::vector<std::string> prefix_unusable;
};

// The model's identity in a prefix cache's fingerprint: the size and the header (metadata and tensor list) of each
// file it loads from, the files of a split model included. Neither the path nor the time of a file is in it.
std::string server_model_files_identity(const std::vector<std::string> & paths);
