    std::sort(entries.begin(), entries.end(),
        [](const expert_pack_entry &a, const expert_pack_entry &b) {
            const int name_cmp = std::strcmp(a.tensor, b.tensor);
            if (name_cmp != 0) return name_cmp < 0;
            if (a.expert_idx != b.expert_idx) return a.expert_idx < b.expert_idx;
            if (a.nbytes != b.nbytes) return a.nbytes < b.nbytes;
            return a.source_idx < b.source_idx;
        });
    uint64_t replaced_duplicates = 0;
    if (replace_duplicates && !entries.empty()) {
        std::vector<expert_pack_entry> deduped;
        deduped.reserve(entries.size());
        for (const expert_pack_entry &entry : entries) {
            if (!deduped.empty() &&
                    expert_pack_entry_cmp(deduped.back(), entry.tensor, entry.expert_idx, entry.nbytes) == 0) {
                if (entry.source_idx >= deduped.back().source_idx) {
                    deduped.back() = entry;
                }
                ++replaced_duplicates;
            } else {
                deduped.push_back(entry);
            }
        }
        entries = std::move(deduped);
        if (replaced_duplicates > 0) {
            std::fprintf(stderr,
                "[moe_stream_batch] expert pack: replaced %lu duplicate keys with later pack sources\n",
                (unsigned long)replaced_duplicates);
        }
    }
    for (size_t i = 1; i < entries.size(); ++i) {
        if (expert_pack_entry_cmp(entries[i - 1], entries[i].tensor, entries[i].expert_idx, entries[i].nbytes) == 0) {
            std::fprintf(stderr, "[moe_stream_batch] expert pack: duplicate key across packs: %s expert=%d bytes=%lu\n",
                         entries[i].tensor, entries[i].expert_idx, (unsigned long)entries[i].nbytes);
            g_expert_pack.inited = true;
            return;
        }
    }

