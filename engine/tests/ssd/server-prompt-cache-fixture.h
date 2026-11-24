#pragma once

#include "inferssd/prompt-cache.h"

#include <string>

// Metadata policy unit tests only: no context, serialization, host admission or
// first-touch claim. Real save/load remains covered by the native fixture.
struct server_prompt_cache_testing {
    static server_prompt * alloc_metadata(
            server_prompt_cache & cache,
            const server_prompt & prompt,
            size_t bytes,
            common_context_seq_rm_type seq_rm_type) {
        if (cache.on_disk() || cache.contains(prompt, seq_rm_type)) {
            return nullptr;
        }
        cache.remove_obsolete(prompt, seq_rm_type);
        cache.remove_superseded(prompt, nullptr);
        auto entry = prompt.clone();
        entry.data.resize(bytes);
        cache.states.push_back(std::move(entry));
        return &cache.states.back();
    }

    // The same for a cache on disk, as save() leaves it once `file` holds the prompt's state: the entry keeps the
    // tokens and the file, and the states it supersedes go with their files.
    static server_prompt * add_file_metadata(
            server_prompt_cache & cache,
            const server_prompt & prompt,
            const std::string & file,
            size_t bytes,
            common_context_seq_rm_type seq_rm_type) {
        if (cache.contains(prompt, seq_rm_type)) {
            return nullptr;
        }
        server_prompt entry;
        entry.tokens         = prompt.tokens.clone();
        entry.file           = file;
        entry.file_data_size = bytes;
        entry.n_conversation = prompt.n_conversation;
        for (const auto & cp : prompt.checkpoints) {
            server_prompt_checkpoint meta = {};
            meta.pos_min   = cp.pos_min;
            meta.pos_max   = cp.pos_max;
            meta.n_tokens  = cp.n_tokens;
            meta.file_size = cp.size();
            entry.checkpoints.push_back(std::move(meta));
        }
        cache.states.push_back(std::move(entry));
        cache.remove_superseded(prompt, &cache.states.back());
        return &cache.states.back();
    }

    // reads the prefix cache directory again if it changed, as load() does before it looks for a state
    static void scan_prefix_dir(server_prompt_cache & cache) {
        cache.scan_prefix_dir();
    }
};
