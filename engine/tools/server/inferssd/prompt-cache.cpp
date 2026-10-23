#include "../server-task.h"

#include "build-info.h"
#include "../server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "../server-common.h"
#include "../server-ssd-reservation.h"

#include <algorithm>
#include <cinttypes>
#include <filesystem>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

using json = nlohmann::ordered_json;

//
// server_prompt_cache
//
static bool server_prompt_can_restore_prefix(
        const server_prompt        & prompt,
        int64_t                      n_tokens,
        common_context_seq_rm_type   seq_rm_type) {
    if (n_tokens < 0 || n_tokens > prompt.n_tokens()) {
        return false;
    }

    if (seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
        return true;
    }

    if (n_tokens == prompt.n_tokens()) {
        return true;
    }

    for (const auto & checkpoint : prompt.checkpoints) {
        if (!checkpoint.empty() && checkpoint.n_tokens == n_tokens) {
            return true;
        }
    }

    return false;
}

int64_t server_prompt_reuse_full(
        const server_prompt              & prompt,
        const server_tokens              & tokens_new,
        const server_prompt_checkpoint ** checkpoint) {
    *checkpoint = nullptr;

    const int64_t lcp   = prompt.tokens.get_common_prefix(tokens_new);
    const int64_t limit = std::min<int64_t>(lcp, (int64_t) tokens_new.size() - 1);

    if (prompt.n_tokens() <= limit) {
        return prompt.n_tokens();
    }

    int64_t best = -1;
    for (const auto & cp : prompt.checkpoints) {
        if (!cp.empty() && cp.n_tokens <= limit && cp.n_tokens > best) {
            best        = cp.n_tokens;
            *checkpoint = &cp;
        }
    }

    return best;
}

size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.n_tokens();
    }

    return res;
}

//
// on-disk layout of a cached prompt: header, checkpoint states, state
//

static constexpr uint32_t SERVER_PROMPT_FILE_MAGIC   = 0x4643504c; // "LPCF"
static constexpr uint32_t SERVER_PROMPT_FILE_VERSION = 1;

static bool server_prompt_fwrite(FILE * f, const void * src, size_t size) {
    return size == 0 || std::fwrite(src, 1, size, f) == size;
}

static bool server_prompt_fread(FILE * f, void * dst, size_t size) {
    return size == 0 || std::fread(dst, 1, size, f) == size;
}

// drop the file's pages from the page cache: a state is read once per restore, and its pages would otherwise take
// memory from the server's cgroup, where the expert caches need it
static void server_prompt_file_drop_cache(FILE * f) {
#if defined(__linux__)
    posix_fadvise(fileno(f), 0, 0, POSIX_FADV_DONTNEED);
#else
    (void) f;
#endif
}

static void server_prompt_file_touch(const std::string & path) {
    std::error_code ec;
    std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), ec);
}

// read the header of a cached prompt file into `prompt` (tokens, checkpoint and state locations); false if the
// file is not a prompt of this fingerprint
static bool server_prompt_read_header(FILE * f, const std::string & fingerprint, server_prompt & prompt) {
    uint32_t magic = 0, version = 0, n_fp = 0;
    if (!server_prompt_fread(f, &magic, sizeof(magic)) || magic != SERVER_PROMPT_FILE_MAGIC ||
        !server_prompt_fread(f, &version, sizeof(version)) || version != SERVER_PROMPT_FILE_VERSION ||
        !server_prompt_fread(f, &n_fp, sizeof(n_fp)) || n_fp > 4096) {
        return false;
    }
    std::string fp(n_fp, '\0');
    if (!server_prompt_fread(f, fp.data(), n_fp) || fp != fingerprint) {
        return false;
    }

    uint32_t n_tokens = 0;
    if (!server_prompt_fread(f, &n_tokens, sizeof(n_tokens)) || n_tokens > (1u << 26)) {
        return false;
    }
    llama_tokens tokens(n_tokens);
    if (!server_prompt_fread(f, tokens.data(), n_tokens*sizeof(llama_token))) {
        return false;
    }

    uint32_t n_checkpoints = 0;
    if (!server_prompt_fread(f, &n_checkpoints, sizeof(n_checkpoints)) || n_checkpoints > 4096) {
        return false;
    }
    std::list<server_prompt_checkpoint> checkpoints;
    for (uint32_t i = 0; i < n_checkpoints; ++i) {
        server_prompt_checkpoint cp {};
        uint64_t size = 0;
        if (!server_prompt_fread(f, &cp.pos_min, sizeof(cp.pos_min)) ||
            !server_prompt_fread(f, &cp.pos_max, sizeof(cp.pos_max)) ||
            !server_prompt_fread(f, &cp.n_tokens, sizeof(cp.n_tokens)) ||
            !server_prompt_fread(f, &size, sizeof(size))) {
            return false;
        }
        cp.file_size = size;
        checkpoints.push_back(std::move(cp));
    }

    uint64_t data_size = 0;
    if (!server_prompt_fread(f, &data_size, sizeof(data_size))) {
        return false;
    }

    size_t offset = std::ftell(f);
    for (auto & cp : checkpoints) {
        cp.file_offset = offset;
        offset += cp.file_size;
    }

    prompt.tokens           = server_tokens(tokens, false);
    prompt.checkpoints      = std::move(checkpoints);
    prompt.file_data_offset = offset;
    prompt.file_data_size   = data_size;

    return true;
}

static bool server_prompt_read_blob(const std::string & path, size_t offset, size_t size, std::vector<uint8_t> & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    out.resize(size);
    const bool ok = std::fseek(f, (long) offset, SEEK_SET) == 0 && server_prompt_fread(f, out.data(), size);
    server_prompt_file_drop_cache(f);
    std::fclose(f);
    return ok;
}

// the checkpoint's partial state, from memory or from the prompt's file
static bool server_prompt_checkpoint_data(const server_prompt & prompt, const server_prompt_checkpoint & cp, std::vector<uint8_t> & out) {
    if (!cp.data.empty()) {
        out = cp.data;
        return true;
    }
    return server_prompt_read_blob(prompt.file, cp.file_offset, cp.file_size, out);
}

// Restore a cached prompt into the slot's sequence: its whole state, then, to start at one of its checkpoints, the
// checkpoint's partial state over it. A partial state holds only what the memory cannot truncate (a sliding window,
// a recurrent or compressed state) and relies on the rest being this prompt's, not whatever the slot held before.
static bool server_prompt_restore(
        const server_prompt            & prompt,
        const server_prompt_checkpoint * checkpoint,
        llama_context                  * ctx,
        int32_t                          id_slot) {
    if (prompt.file.empty()) {
        if (llama_state_seq_set_data_ext(ctx, prompt.data.data(), prompt.data.size(), id_slot, 0) != prompt.data.size()) {
            return false;
        }
    } else {
        FILE * f = std::fopen(prompt.file.c_str(), "rb");
        if (!f) {
            return false;
        }
        bool ok = std::fseek(f, (long) prompt.file_data_offset, SEEK_SET) == 0 &&
            llama_state_seq_fread_ext(ctx, f, id_slot, 0) == prompt.file_data_size;
        server_prompt_file_drop_cache(f);
        std::fclose(f);
        if (!ok) {
            return false;
        }
    }

    if (checkpoint != nullptr) {
        // straight from the prompt's file when it is there, without a copy in memory
        if (!checkpoint->data.empty()) {
            return llama_state_seq_set_data_ext(ctx, checkpoint->data.data(), checkpoint->data.size(), id_slot,
                    LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == checkpoint->data.size();
        }
        FILE * f = std::fopen(prompt.file.c_str(), "rb");
        if (!f) {
            return false;
        }
        const bool ok = std::fseek(f, (long) checkpoint->file_offset, SEEK_SET) == 0 &&
            llama_state_seq_fread_ext(ctx, f, id_slot, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == checkpoint->file_size;
        server_prompt_file_drop_cache(f);
        std::fclose(f);
        return ok;
    }

    return true;
}

void server_prompt_cache::use_dir(const std::string & dir, const std::string & fingerprint) {
    this->dir         = dir;
    this->fingerprint = fingerprint;
    limit_tokens      = 0;

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        throw std::runtime_error("cannot create the prompt cache directory " + dir + ": " + ec.message());
    }

    std::vector<std::pair<std::filesystem::file_time_type, server_prompt>> found;
    for (const auto & entry : std::filesystem::directory_iterator(dir, ec)) {
        const std::string path = entry.path().string();
        if (entry.path().extension() == ".tmp") {
            std::filesystem::remove(entry.path(), ec);
            continue;
        }
        if (entry.path().extension() != ".prompt") {
            continue;
        }
        server_prompt prompt;
        bool ok = false;
        if (FILE * f = std::fopen(path.c_str(), "rb")) {
            ok = server_prompt_read_header(f, fingerprint, prompt);
            std::fclose(f);
        }
        if (!ok) {
            SRV_WRN(" - removing cached prompt %s: other model, build or format\n", path.c_str());
            std::filesystem::remove(entry.path(), ec);
            continue;
        }
        prompt.file = path;
        found.emplace_back(std::filesystem::last_write_time(entry.path(), ec), std::move(prompt));
    }
    std::sort(found.begin(), found.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    for (auto & item : found) {
        states.push_back(std::move(item.second));
    }

    SRV_WRN("prompt cache on disk in %s: %zu prompts found\n", dir.c_str(), states.size());
    update();
}

bool server_prompt_cache::contains(const server_prompt & prompt, common_context_seq_rm_type seq_rm_type) {
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp = it->tokens.get_common_prefix(prompt.tokens);
        if (lcp == (int) prompt.tokens.size() && server_prompt_can_restore_prefix(*it, prompt.n_tokens(), seq_rm_type)) {
            states.splice(states.end(), states, it);
            if (!states.back().file.empty()) {
                server_prompt_file_touch(states.back().file);
            }
            return true;
        }
    }
    return false;
}

void server_prompt_cache::remove(std::list<server_prompt>::iterator it) {
    if (!it->file.empty()) {
        std::error_code ec;
        std::filesystem::remove(it->file, ec);
    }
    states.erase(it);
}

void server_prompt_cache::remove_obsolete(const server_prompt & prompt, common_context_seq_rm_type seq_rm_type) {
        // remove any cached prompts that are fully contained in the current prompt
        for (auto it = states.begin(); it != states.end();) {
            const int len = it->tokens.get_common_prefix(prompt.tokens);

            if (len == (int) it->tokens.size() &&
                    server_prompt_can_restore_prefix(prompt, it->n_tokens(), seq_rm_type)) {
                SRV_WRN(" - removing obsolete cached prompt with length %d\n", len);

                it = states.erase(it);
            } else {
                ++it;
            }
        }

}

void server_prompt_cache::save(
        const server_prompt        & prompt,
        bool                         current,
        llama_context              * ctx,
        int32_t                      id_slot,
        common_context_seq_rm_type   seq_rm_type) {
    server_ssd_memory_change budget_changes(ctx);
    const int32_t n_state = current ? -1 : prompt.n_tokens();

    if (contains(prompt, seq_rm_type)) {
        SRV_WRN("%s", " - prompt is already in the cache, skipping\n");
        return;
    }

    const size_t state_size = llama_state_seq_get_size_at(ctx, id_slot, 0, n_state);
    if (state_size == 0) {
        SRV_WRN(" - cannot save the state after %d tokens\n", prompt.n_tokens());
        return;
    }

    SRV_WRN(" - saving prompt with length %d, total state size = %.3f MiB\n", prompt.n_tokens(), state_size / (1024.0 * 1024.0));

    if (!on_disk()) {
        remove_obsolete(prompt, seq_rm_type);

        server_ssd_memory_change changes(ctx);
        size_t allocation_bytes = state_size;
        for (const auto & checkpoint : prompt.checkpoints) {
            if (checkpoint.data.size() > SIZE_MAX - allocation_bytes) {
                SRV_WRN("%s", " - prompt cache allocation size overflow\n"); return;
            }
            allocation_bytes += checkpoint.data.size();
        }
        server_ssd_reservation reservation(ctx, allocation_bytes);
        if (!reservation) { SRV_WRN("%s", " - no host budget for prompt cache state\n"); return; }
        std::vector<uint8_t> state_data;

        // check if we can allocate enough memory for the new state
        try {
            state_data.resize(state_size);
        } catch (const std::bad_alloc & e) {
            SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

            limit_size = std::max<size_t>(1, 0.4*size());

            SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

            update();

            return;
        }

        if (llama_state_seq_get_data_at(ctx, state_data.data(), state_size, id_slot, 0, n_state) != state_size) {
            SRV_WRN("%s", " - failed to save the state\n");
            return;
        }

        server_prompt entry;
        entry.tokens      = prompt.tokens.clone();
        entry.data        = std::move(state_data);
        entry.checkpoints = prompt.checkpoints;
        reservation.commit();
        states.push_back(std::move(entry));

        return;
    }

    const llama_tokens & tokens = prompt.tokens.get_tokens();
    if (std::find(tokens.begin(), tokens.end(), LLAMA_TOKEN_NULL) != tokens.end()) {
        SRV_WRN("%s", " - prompts with media are not cached on disk\n");
        return;
    }

    uint64_t hash = 1469598103934665603ull; // FNV-1a over the tokens names the file
    for (llama_token t : tokens) {
        for (int b = 0; b < 4; ++b) {
            hash = (hash ^ ((uint32_t) t >> (8*b) & 0xff))*1099511628211ull;
        }
    }
    char name[32];
    std::snprintf(name, sizeof(name), "%016" PRIx64 ".prompt", hash);
    const std::string path = (std::filesystem::path(dir) / name).string();
    const std::string tmp  = path + ".tmp";

    server_prompt entry;
    entry.tokens = prompt.tokens.clone();
    entry.file   = path;

    bool ok = false;
    if (FILE * f = std::fopen(tmp.c_str(), "wb")) {
        const uint32_t magic = SERVER_PROMPT_FILE_MAGIC, version = SERVER_PROMPT_FILE_VERSION;
        const uint32_t n_fp = fingerprint.size(), n_tokens = tokens.size(), n_checkpoints = prompt.checkpoints.size();
        const uint64_t data_size = state_size;
        ok = server_prompt_fwrite(f, &magic, sizeof(magic)) && server_prompt_fwrite(f, &version, sizeof(version)) &&
             server_prompt_fwrite(f, &n_fp, sizeof(n_fp)) && server_prompt_fwrite(f, fingerprint.data(), n_fp) &&
             server_prompt_fwrite(f, &n_tokens, sizeof(n_tokens)) &&
             server_prompt_fwrite(f, tokens.data(), n_tokens*sizeof(llama_token)) &&
             server_prompt_fwrite(f, &n_checkpoints, sizeof(n_checkpoints));
        for (const auto & cp : prompt.checkpoints) {
            const uint64_t size = cp.size();
            ok = ok && server_prompt_fwrite(f, &cp.pos_min, sizeof(cp.pos_min)) &&
                 server_prompt_fwrite(f, &cp.pos_max, sizeof(cp.pos_max)) &&
                 server_prompt_fwrite(f, &cp.n_tokens, sizeof(cp.n_tokens)) &&
                 server_prompt_fwrite(f, &size, sizeof(size));
        }
        ok = ok && server_prompt_fwrite(f, &data_size, sizeof(data_size));

        size_t offset = ok ? std::ftell(f) : 0;
        for (const auto & cp : prompt.checkpoints) {
            // a checkpoint in memory is written as it is; one in another prompt's file is read back first
            server_ssd_memory_change changes(ctx);
            std::vector<uint8_t> copy;
            const bool in_memory = !cp.data.empty();
            server_ssd_reservation reservation;
            if (ok && !in_memory) {
                reservation = server_ssd_reservation(ctx, cp.size());
                ok = bool(reservation) && server_prompt_checkpoint_data(prompt, cp, copy);
                if (ok) { reservation.commit(); }
            }
            const std::vector<uint8_t> & data = in_memory ? cp.data : copy;
            ok = ok && server_prompt_fwrite(f, data.data(), data.size());
            // A copied-and-cleared vector retains its payload capacity. Disk
            // cache entries keep only metadata, not a second host checkpoint.
            server_prompt_checkpoint meta = {};
            meta.pos_min  = cp.pos_min;
            meta.pos_max  = cp.pos_max;
            meta.n_tokens = cp.n_tokens;
            meta.file_offset = offset;
            meta.file_size   = data.size();
            offset += data.size();
            entry.checkpoints.push_back(std::move(meta));
        }
        entry.file_data_offset = offset;
        entry.file_data_size   = state_size;

        ok = ok && llama_state_seq_fwrite_at(ctx, f, id_slot, 0, n_state) == state_size;
        ok = ok && std::fflush(f) == 0;
#if defined(__linux__)
        ok = ok && fdatasync(fileno(f)) == 0;
#endif
        server_prompt_file_drop_cache(f);
        ok = std::fclose(f) == 0 && ok;
    }
    std::error_code ec;
    if (ok) {
        std::filesystem::rename(tmp, path, ec);
        ok = !ec;
    }
    if (!ok) {
        SRV_WRN(" - failed to write cached prompt %s\n", path.c_str());
        std::filesystem::remove(tmp, ec);
        return;
    }

    states.push_back(std::move(entry));
}

bool server_prompt_cache::load(
        server_prompt              & prompt,
        const server_tokens        & tokens_new,
        int64_t                      n_reuse_slot,
        llama_context              * ctx,
        int32_t                      id_slot,
        common_context_seq_rm_type   seq_rm_type) {
    server_ssd_memory_change budget_changes(ctx);
    auto it_best = states.end();
    const server_prompt_checkpoint * checkpoint_best = nullptr;
    int64_t n_tokens_best = -1;

    if (seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
        // A state that cannot be rolled back is reused up to its end or one of its checkpoints. Take the cached
        // prompt that lets the slot skip the most of the new prompt, if that is more than its own state does.
        SRV_WRN(" - looking for a better prompt, the slot reuses %" PRId64 " of %zu tokens\n", n_reuse_slot, tokens_new.size());

        n_tokens_best = n_reuse_slot;
        for (auto it = states.begin(); it != states.end(); ++it) {
            const server_prompt_checkpoint * checkpoint_cur = nullptr;
            const int64_t n_tokens_cur = server_prompt_reuse_full(*it, tokens_new, &checkpoint_cur);
            if (n_tokens_cur > n_tokens_best) {
                it_best         = it;
                checkpoint_best = checkpoint_cur;
                n_tokens_best   = n_tokens_cur;
            }
        }
    } else {
        const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

        float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
        float sim_best    = float(lcp_best) / tokens_new.size();

        SRV_WRN(" - looking for better prompt, base f_keep = %.3f, sim = %.3f\n", f_keep_best, sim_best);

        // find the most similar cached prompt, that would also preserve the most context
        for (auto it = states.begin(); it != states.end(); ++it) {
            const int lcp_cur = it->tokens.get_common_prefix(tokens_new);

            const float f_keep_cur = float(lcp_cur) / it->tokens.size();
            const float sim_cur    = float(lcp_cur) / tokens_new.size();

            // don't trash large prompts
            if (f_keep_cur < 0.25f) {
                continue;
            }

            if (f_keep_best < f_keep_cur && sim_best < sim_cur) {
                f_keep_best = f_keep_cur;
                sim_best    = sim_cur;

                it_best       = it;
                n_tokens_best = lcp_cur;
            }
        }
    }

    if (it_best == states.end()) {
        return true;
    }

    SRV_WRN(" - found a better prompt%s: reuses %" PRId64 " tokens\n", checkpoint_best ? " checkpoint" : "", n_tokens_best);

    if (!server_prompt_restore(*it_best, checkpoint_best, ctx, id_slot)) {
        SRV_WRN("%s", "failed to restore the cached prompt\n");
        remove(it_best);
        return false;
    }

    // the slot takes over the cached prompt, cut back to the checkpoint it starts from; its checkpoints move to memory
    server_prompt loaded;
    loaded.tokens = it_best->tokens.clone();
    for (const auto & cp : it_best->checkpoints) {
        if (checkpoint_best != nullptr && cp.n_tokens > n_tokens_best) {
            continue;
        }
        // into memory as far as the memory limit allows (the RAM expert cache gives back room first)
        server_ssd_memory_change changes(ctx);
        server_ssd_reservation reservation(ctx, cp.size());
        if (!reservation) { continue; }
        server_prompt_checkpoint cur = cp;
        if (!server_prompt_checkpoint_data(*it_best, cp, cur.data)) { continue; }
        reservation.commit();
        cur.file_offset = 0;
        cur.file_size   = 0;
        loaded.checkpoints.push_back(std::move(cur));
    }
    if (checkpoint_best != nullptr) {
        loaded.tokens.keep_first(n_tokens_best);
    }
    prompt = std::move(loaded);

    if (on_disk()) {
        // the file stays: other conversations may start from it too
        states.splice(states.end(), states, it_best);
        server_prompt_file_touch(states.back().file);
    } else {
        states.erase(it_best);
    }

    return true;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        // always keep at least one state, regardless of the limits
        while (states.size() > 1 && size() > limit_size) {
            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            remove(states.begin());
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (states.size() > 1 && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            remove(states.begin());
        }
    }

    SRV_WRN(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)%s\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur,
            on_disk() ? " on disk" : "");

    for (const auto & state : states) {
        SRV_WRN("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.n_tokens(), state.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}
