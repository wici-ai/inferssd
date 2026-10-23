#pragma once

#include "inferssd/prompt-cache.h"

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
        auto entry = prompt.clone();
        entry.data.resize(bytes);
        cache.states.push_back(std::move(entry));
        return &cache.states.back();
    }
};
