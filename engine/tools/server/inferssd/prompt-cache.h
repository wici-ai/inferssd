// SPDX-License-Identifier: MIT
#pragma once

#include "../server-common.h"
#include "common.h"
#include "llama.h"
#include "platform/file.h"

#include <algorithm>
#include <cstdint>
#include <list>
#include <optional>
#include <string>
#include <unordered_map>
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

    // Empty for the base model. No key means the slot's state cannot be shared.
    std::optional<std::string> adapter_key = std::string();
    std::optional<ssd::platform::file_identity> prefix_identity;

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
            adapter_key,
            prefix_identity,
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

// The places of a prompt where the server keeps a state for later prompts (oaicompat_chat_params_parse): the end of
// the shared prefix (the leading system messages and the tools, which every conversation with the app's setup starts
// with) and of the context prefix (with the user's context too), whose states go to the prompt cache, and the branch
// points inside them, where the prompt of another user or setup is likely to part from this one (where a group of
// tools ends, where the data in the user's context starts), which the slot keeps checkpoints at when the memory can
// save an earlier state after a batch, and the states kept in the prompt cache carry. 0 for none.
struct server_prompt_points {
    int64_t              n_shared  = 0;
    int64_t              n_context = 0;
    std::vector<int64_t> branch; // sorted, each inside the shared or the context prefix

    // every point, sorted
    std::vector<int64_t> all() const {
        std::vector<int64_t> out = branch;
        for (const int64_t n : { n_shared, n_context }) {
            if (n > 0) {
                out.push_back(n);
            }
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    // whether n is a point, rounded down to a multiple of align (the positions a memory can save an earlier state at)
    bool has(int64_t n, int64_t align = 1) const {
        const auto at = [&](int64_t p) { return p > 0 && p - p % std::max<int64_t>(1, align) == n; };
        return at(n_shared) || at(n_context) || std::any_of(branch.begin(), branch.end(), at);
    }

    // whether the state at n, rounded as above, goes to the prompt cache
    bool kept(int64_t n, int64_t align = 1) const {
        const auto at = [&](int64_t p) { return p > 0 && p - p % std::max<int64_t>(1, align) == n; };
        return at(n_shared) || at(n_context);
    }

    // The prompt's tokens [n_before, replaced_end) became others, delta more (server_tokens_reuse_text): a point after
    // them moves with them, one inside them is lost.
    void replaced(int64_t n_before, int64_t replaced_end, int64_t delta) {
        const auto move = [&](int64_t & n) {
            if (n >= replaced_end) {
                n += delta;
            } else if (n > n_before) {
                n = 0;
            }
        };
        move(n_shared);
        move(n_context);
        for (int64_t & n : branch) {
            move(n);
        }
        branch.erase(std::remove(branch.begin(), branch.end(), 0), branch.end());
    }
};

// Whether the state a slot holds goes to the prompt cache before a new prompt that shares n_common of its n_state
// tokens takes the slot, so that its conversation can go on from it later: when the new prompt keeps less than half of
// it (f_keep, upstream's rule), and for a state that cannot be rolled back whenever the new prompt parts from it. Such
// a state is kept only up to its end and its checkpoints, so the new prompt keeps none of its tail; another
// conversation behind the same system prompt and tools keeps more than half and would let it go.
inline bool server_prompt_save_before_switch(float f_keep, common_context_seq_rm_type seq_rm_type, int64_t n_common,
                                             int64_t n_state) {
    return f_keep < 0.5f || (seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL && n_common < n_state);
}

// For a memory that is checkpointed only between batches: whether a prompt batch stops after n_done of the prompt's
// n_prompt tokens, so that a checkpoint can be taken there. It stops at the end of the shared and of the context prefix
// and 4 tokens before the end of the prompt (the generation prompt may be rendered differently in the next turn). It
// does not stop at the branch points: each stop reads the model's weights once more, a cost every new prompt pays for
// a state that serves only some later ones; a memory that can save an earlier state keeps them after the batch.
inline bool server_prompt_batch_stops(int64_t n_done, int64_t n_prompt, const server_prompt_points & points,
                                      int64_t n_batch) {
    const int64_t n_end = n_prompt - std::min<int64_t>(n_batch, 4);
    return n_done == n_end || (n_done >= 64 && n_done < n_end && points.kept(n_done));
}

// For a memory that saves an earlier state after the batch that went past it (llama_state_seq_at_align): whether the
// batch ends after n_in_batch of its tokens, once it holds one of the prompt's points (past_point). Such a state is
// there only while the batch's last ubatch is the one that went past it: a sliding window gives the cells the state
// needs to the positions of a later ubatch. So a batch that holds a point ends with the ubatch that holds it.
inline bool server_prompt_batch_ends_for_point(bool past_point, int64_t n_in_batch, int64_t n_ubatch) {
    return past_point && n_ubatch > 0 && n_in_batch % n_ubatch == 0;
}

// The checkpoint that goes to make room by count for a new one after n_new tokens, when the slot keeps at most n_max
// checkpoints that are not at one of the prompt's points: the oldest of those, while they are n_max, and none when the
// new one is at a point. The checkpoints at points are as many as the points, and they serve the prompts that part
// there, so they are kept, within the memory limit like the others. end() when none goes.
inline std::list<server_prompt_checkpoint>::iterator server_prompt_checkpoint_over_count(
        std::list<server_prompt_checkpoint> & checkpoints, const server_prompt_points & points, int64_t align,
        int64_t n_new, int32_t n_max) {
    if (points.has(n_new, align)) {
        return checkpoints.end();
    }
    const auto ordinary = [&](const server_prompt_checkpoint & cp) { return !points.has(cp.n_tokens, align); };
    if (std::count_if(checkpoints.begin(), checkpoints.end(), ordinary) < n_max) {
        return checkpoints.end();
    }
    return std::find_if(checkpoints.begin(), checkpoints.end(), ordinary);
}

// Whether `prompt` is a newer state of the conversation that the cached prompt `old` is a state of, and takes its
// place in the prompt cache. The two are states of one conversation when they share the conversation's opening
// (prompt.n_conversation tokens): the system messages, the tools and the user messages before the first reply. A
// conversation that only starts behind the same system messages and tools parts from the prompt before that, and
// the shared prefix kept for all of them is shorter than it. The newer state replaces only one that is not longer:
// a short request made beside a long conversation, with the same opening, leaves the long state in place.
inline bool server_prompt_supersedes(const server_prompt & prompt, const server_prompt & old) {
    const int64_t n = prompt.n_conversation;
    if (prompt.adapter_key != old.adapter_key || n <= 0 || n >= prompt.n_tokens() || old.n_tokens() > prompt.n_tokens()) {
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
    // the cached prompt that lets the slot skip the most of tokens_new, if more than n_reuse_slot (end() if none),
    // with the checkpoint it starts from (nullptr for its whole state) and the tokens it lets the slot skip
    std::list<server_prompt>::iterator best_to_load(const server_prompt & prompt, const server_tokens & tokens_new,
                                                    int64_t n_reuse_slot, common_context_seq_rm_type seq_rm_type,
                                                    const server_prompt_checkpoint ** checkpoint, int64_t * n_tokens);
    enum class take_result { restored, stale_index, restore_failed };
    // A stale index has not changed the sequence. A failed restore may have overwritten it.
    take_result take(std::list<server_prompt>::iterator it_best, const server_prompt_checkpoint * checkpoint_best,
              int64_t n_tokens_best, server_prompt & prompt, llama_context * ctx, int32_t id_slot);
    void remove_obsolete(const server_prompt & prompt, common_context_seq_rm_type seq_rm_type);
    // removes the states `prompt` supersedes, except `saved`, the cache's own entry of it
    void remove_superseded(const server_prompt & prompt, const server_prompt * saved);
    bool contains(const server_prompt & prompt, common_context_seq_rm_type seq_rm_type);
    void remove(std::list<server_prompt>::iterator it);
    // reads the prefix cache directory again if it changed since the last read
    void scan_prefix_dir();
    int64_t prefix_dir_time = -1;
    // Present file versions that failed header validation or native restore. A replacement is tried again.
    std::unordered_map<std::string, ssd::platform::file_identity> prefix_unusable;
};

// The model's identity in a prefix cache's fingerprint: the size and the header (metadata and tensor list) of each
// file it loads from, the files of a split model included. Neither the path nor the time of a file is in it.
std::string server_model_files_identity(const std::vector<std::string> & paths);

std::string server_adapter_file_identity(const std::string & path);
std::optional<std::string> server_prompt_adapter_key(const std::vector<std::string> & identities,
        const std::vector<common_adapter_lora_info> & adapters);
