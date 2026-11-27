// SPDX-License-Identifier: MIT
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
#include "gguf.h"

extern "C" {
#include "sha256/sha256.h"
}

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>

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
        res += state.read_only ? 0 : state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.read_only ? 0 : state.n_tokens();
    }

    return res;
}

//
// on-disk layout of a cached prompt: header, checkpoint states, state
//

static constexpr uint32_t SERVER_PROMPT_FILE_MAGIC   = 0x4643504c; // "LPCF"
static constexpr uint32_t SERVER_PROMPT_FILE_VERSION = 1;
static constexpr char SERVER_PROMPT_ADAPTER_TAG[] = "|lora=";
static constexpr char SERVER_PROMPT_BASE_PROVENANCE[] = "|adapter-cache=1";
static void server_fnv1a(uint64_t & hash, const char * data, size_t size);

static bool server_prompt_fwrite(FILE * f, const void * src, size_t size) {
    return size == 0 || std::fwrite(src, 1, size, f) == size;
}

static bool server_prompt_fread(FILE * f, void * dst, size_t size) {
    return size == 0 || std::fread(dst, 1, size, f) == size;
}

static int server_prompt_file_seek(FILE * f, int64_t offset, int origin) {
#if defined(_WIN32)
    return _fseeki64(f, offset, origin);
#else
    return fseeko(f, offset, origin);
#endif
}

static int64_t server_prompt_file_tell(FILE * f) {
#if defined(_WIN32)
    return _ftelli64(f);
#else
    return ftello(f);
#endif
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

static void server_prompt_file_close(FILE * f) {
    server_prompt_file_drop_cache(f);
    std::fclose(f);
}

using server_prompt_file = std::unique_ptr<FILE, decltype(&server_prompt_file_close)>;

static void server_prompt_file_touch(const std::string & path) {
    std::error_code ec;
    std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), ec);
}

// Files already published before the adapter provenance marker have immutable
// sizes and SHA-256 digests in the official prefix-cache.json listing.
static bool server_prompt_is_published_legacy(FILE * f, uint64_t size) {
    static constexpr struct {
        uint64_t size;
        const char * sha256;
    } legacy[] = {
#include "prompt-cache-legacy.inc"
    };
    if (std::none_of(std::begin(legacy), std::end(legacy), [=](const auto & entry) {
            return entry.size == size;
        })) {
        return false;
    }

    const int64_t offset = server_prompt_file_tell(f);
    if (offset < 0 || server_prompt_file_seek(f, 0, SEEK_SET) != 0) {
        return false;
    }
    sha256_t hash;
    sha256_init(&hash);
    std::vector<unsigned char> buffer(256 * 1024);
    for (uint64_t remaining = size; remaining > 0;) {
        const size_t count = std::min<uint64_t>(remaining, buffer.size());
        if (std::fread(buffer.data(), 1, count, f) != count) {
            return false;
        }
        sha256_update(&hash, buffer.data(), count);
        remaining -= count;
    }
    if (std::fgetc(f) != EOF || std::ferror(f) || server_prompt_file_seek(f, offset, SEEK_SET) != 0) {
        return false;
    }

    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&hash, digest);
    char hex[2 * SHA256_DIGEST_SIZE + 1];
    for (size_t i = 0; i < SHA256_DIGEST_SIZE; ++i) {
        std::snprintf(hex + 2 * i, 3, "%02x", static_cast<unsigned>(digest[i]));
    }
    return std::any_of(std::begin(legacy), std::end(legacy), [&](const auto & entry) {
        return entry.size == size && std::strcmp(entry.sha256, hex) == 0;
    });
}

// read the header of a cached prompt file into `prompt` (tokens, checkpoint and state locations); false if the
// file is not a prompt of this fingerprint, or with `prefix`, of a fingerprint that starts with it and a '|' (a
// prefix cache: the model's fingerprint, then the build that saved it)
static bool server_prompt_read_header(FILE * f, const std::string & fingerprint, server_prompt & prompt,
                                      bool prefix = false) {
    if (server_prompt_file_seek(f, 0, SEEK_END) != 0) {
        return false;
    }
    const int64_t file_size = server_prompt_file_tell(f);
    if (file_size < 0 || uint64_t(file_size) > SIZE_MAX || server_prompt_file_seek(f, 0, SEEK_SET) != 0) {
        return false;
    }

    uint32_t magic = 0, version = 0, n_fp = 0;
    if (!server_prompt_fread(f, &magic, sizeof(magic)) || magic != SERVER_PROMPT_FILE_MAGIC ||
        !server_prompt_fread(f, &version, sizeof(version)) || version != SERVER_PROMPT_FILE_VERSION ||
        !server_prompt_fread(f, &n_fp, sizeof(n_fp)) || n_fp > 4096) {
        return false;
    }
    std::string fp(n_fp, '\0');
    if (!server_prompt_fread(f, fp.data(), n_fp)) {
        return false;
    }
    const size_t adapter = fp.rfind(SERVER_PROMPT_ADAPTER_TAG);
    if (adapter != std::string::npos) {
        const std::string key = fp.substr(adapter + sizeof(SERVER_PROMPT_ADAPTER_TAG) - 1);
        // Downloaded prefix caches are portable base-model states.
        if (prefix || key.size() != 16 || key.find_first_not_of("0123456789abcdef") != std::string::npos) {
            return false;
        }
        prompt.adapter_key = key;
        fp.resize(adapter);
    }
    const bool same = prefix ? fp == fingerprint || fp.compare(0, fingerprint.size() + 1, fingerprint + "|") == 0
                             : fp == fingerprint;
    if (!same || (prefix && !string_ends_with(fp, SERVER_PROMPT_BASE_PROVENANCE) &&
                  !server_prompt_is_published_legacy(f, file_size))) {
        return false;
    }

    uint32_t n_tokens = 0;
    if (!server_prompt_fread(f, &n_tokens, sizeof(n_tokens)) || n_tokens > (1u << 26) ||
        n_tokens > uint64_t(file_size) / sizeof(llama_token)) {
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
            !server_prompt_fread(f, &size, sizeof(size)) || size > uint64_t(file_size)) {
            return false;
        }
        cp.file_size = size;
        checkpoints.push_back(std::move(cp));
    }

    uint64_t data_size = 0;
    if (!server_prompt_fread(f, &data_size, sizeof(data_size))) {
        return false;
    }

    const int64_t header_size = server_prompt_file_tell(f);
    if (header_size < 0 || header_size > file_size) {
        return false;
    }
    size_t offset = header_size;
    for (auto & cp : checkpoints) {
        if (cp.file_size > size_t(file_size) - offset) {
            return false;
        }
        cp.file_offset = offset;
        offset += cp.file_size;
    }
    if (data_size == 0 || data_size != size_t(file_size) - offset) {
        return false;
    }

    prompt.tokens           = server_tokens(tokens, false);
    prompt.checkpoints      = std::move(checkpoints);
    prompt.file_data_offset = offset;
    prompt.file_data_size   = data_size;

    return true;
}

// The index is a lookup hint. A replacement may have different tokens or
// checkpoints, so its state cannot be restored using the previous match.
static bool server_prompt_same_layout(const server_prompt & indexed, const server_prompt & opened) {
    return indexed.tokens.get_tokens() == opened.tokens.get_tokens() &&
           indexed.file_data_offset == opened.file_data_offset &&
           indexed.file_data_size == opened.file_data_size &&
           std::equal(indexed.checkpoints.begin(), indexed.checkpoints.end(),
                      opened.checkpoints.begin(), opened.checkpoints.end(),
                      [](const server_prompt_checkpoint & a, const server_prompt_checkpoint & b) {
                          return a.pos_min == b.pos_min && a.pos_max == b.pos_max && a.n_tokens == b.n_tokens &&
                                 a.file_offset == b.file_offset && a.file_size == b.file_size;
                      });
}

// The checkpoint's partial state, from memory or from the already open prompt file.
static bool server_prompt_checkpoint_data(FILE * f, const server_prompt_checkpoint & cp, std::vector<uint8_t> & out) {
    if (!cp.data.empty()) {
        out = cp.data;
        return true;
    }
    if (!f || server_prompt_file_seek(f, cp.file_offset, SEEK_SET) != 0) {
        return false;
    }
    out.resize(cp.file_size);
    return server_prompt_fread(f, out.data(), cp.file_size);
}

// Restore a cached prompt into the slot's sequence: its whole state, then, to start at one of its checkpoints, the
// checkpoint's partial state over it. A partial state holds only what the memory cannot truncate (a sliding window,
// a recurrent or compressed state) and relies on the rest being this prompt's, not whatever the slot held before.
static bool server_prompt_restore(
        const server_prompt            & prompt,
        const server_prompt_checkpoint * checkpoint,
        FILE                           * f,
        llama_context                  * ctx,
        int32_t                          id_slot) {
    if (prompt.file.empty()) {
        if (llama_state_seq_set_data_ext(ctx, prompt.data.data(), prompt.data.size(), id_slot, 0) != prompt.data.size()) {
            return false;
        }
    } else {
        if (!f || server_prompt_file_seek(f, prompt.file_data_offset, SEEK_SET) != 0 ||
            llama_state_seq_fread_ext(ctx, f, id_slot, 0) != prompt.file_data_size) {
            return false;
        }
    }

    if (checkpoint != nullptr) {
        // straight from the prompt's file when it is there, without a copy in memory
        if (!checkpoint->data.empty()) {
            return llama_state_seq_set_data_ext(ctx, checkpoint->data.data(), checkpoint->data.size(), id_slot,
                    LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == checkpoint->data.size();
        }
        return f && server_prompt_file_seek(f, checkpoint->file_offset, SEEK_SET) == 0 &&
            llama_state_seq_fread_ext(ctx, f, id_slot, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == checkpoint->file_size;
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

void server_prompt_cache::use_prefix_dir(const std::string & dir, const std::string & fingerprint) {
    states.remove_if([](const server_prompt & prompt) { return prompt.read_only; });
    prefix_unusable.clear();
    prefix_dir         = dir;
    prefix_fingerprint = fingerprint;
    prefix_dir_time    = -1;
    scan_prefix_dir();
}

void server_prompt_cache::scan_prefix_dir() {
    if (prefix_dir.empty()) {
        return;
    }
    std::error_code ec;
    const auto directory = std::filesystem::u8path(prefix_dir);
    const auto    time = std::filesystem::last_write_time(directory, ec);
    const int64_t t    = ec ? 0 : (int64_t) time.time_since_epoch().count();
    if (t == prefix_dir_time) {
        return;
    }
    // Keep validated metadata for unchanged files. Only the selected prefix is verified again at restore.
    std::unordered_map<std::string, std::list<server_prompt>::iterator> indexed;
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (it->read_only) {
            indexed.emplace(it->file, it);
        }
    }

    decltype(prefix_unusable) still_unusable;
    size_t n_found = 0;
    const std::filesystem::directory_iterator end;
    for (auto it = ec ? end : std::filesystem::directory_iterator(directory, ec);
         !ec && it != end; it.increment(ec)) {
        if (it->path().extension() != ".prompt") {
            continue;
        }
        const std::string path = it->path().u8string();
        std::unique_ptr<FILE, decltype(&std::fclose)> file(ssd::platform::open_stream(path.c_str()), std::fclose);
        ssd::platform::file_identity identity;
        const bool identified = file && ssd::platform::identity(ssd::platform::stream_fd(file.get()), identity);
        auto old = indexed.find(path);
        if (old != indexed.end() && identified && old->second->prefix_identity == identity) {
            indexed.erase(old);
            ++n_found;
            continue;
        }

        const auto failed = prefix_unusable.find(path);
        if (identified && failed != prefix_unusable.end() && failed->second == identity) {
            still_unusable.insert(prefix_unusable.extract(failed));
            continue;
        }
        server_prompt prompt;
        server_prompt_file validated(file.release(), server_prompt_file_close);
        const bool valid = validated && server_prompt_read_header(validated.get(), prefix_fingerprint, prompt, true);
        if (identified) {
            ssd::platform::file_identity after;
            if (!ssd::platform::identity(ssd::platform::stream_fd(validated.get()), after) || identity != after) {
                continue;
            }
            // Remember a rejected header or digest for this file version. Retry read errors on the next scan.
            if (!valid && !std::ferror(validated.get())) {
                still_unusable.emplace(path, identity);
            }
            prompt.prefix_identity = identity;
        }
        if (!valid) {
            SRV_WRN(" - skipping prefix cache %s: made for another model or format\n", path.c_str());
            continue;
        }
        prompt.file = path;
        prompt.read_only = true;
        if (old != indexed.end()) {
            *old->second = std::move(prompt);
            indexed.erase(old);
        } else {
            states.push_front(std::move(prompt));
        }
        ++n_found;
    }
    for (const auto & entry : indexed) {
        states.erase(entry.second);
    }
    prefix_unusable.swap(still_unusable);
    prefix_dir_time = ec && ec != std::errc::no_such_file_or_directory ? -1 : t;
    SRV_WRN("prefix caches in %s: %zu found\n", prefix_dir.c_str(), n_found);
}

bool server_prompt_cache::contains(const server_prompt & prompt, common_context_seq_rm_type seq_rm_type) {
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (!prompt.adapter_key || it->adapter_key != prompt.adapter_key) {
            continue;
        }
        const int lcp = it->tokens.get_common_prefix(prompt.tokens);
        if (lcp == (int) prompt.tokens.size() && server_prompt_can_restore_prefix(*it, prompt.n_tokens(), seq_rm_type)) {
            states.splice(states.end(), states, it);
            if (!states.back().file.empty() && !states.back().read_only) {
                server_prompt_file_touch(states.back().file);
            }
            return true;
        }
    }
    return false;
}

void server_prompt_cache::remove(std::list<server_prompt>::iterator it) {
    if (it->read_only && it->prefix_identity) {
        prefix_unusable.insert_or_assign(it->file, *it->prefix_identity);
    } else if (!it->read_only && !it->file.empty()) {
        std::error_code ec;
        std::filesystem::remove(it->file, ec);
    }
    states.erase(it);
}

void server_prompt_cache::remove_obsolete(const server_prompt & prompt, common_context_seq_rm_type seq_rm_type) {
        // remove any cached prompts that are fully contained in the current prompt
        for (auto it = states.begin(); it != states.end();) {
            const int len = it->tokens.get_common_prefix(prompt.tokens);

            if (!it->read_only && it->adapter_key == prompt.adapter_key && len == (int) it->tokens.size() &&
                    server_prompt_can_restore_prefix(prompt, it->n_tokens(), seq_rm_type)) {
                SRV_WRN(" - removing obsolete cached prompt with length %d\n", len);

                it = states.erase(it);
            } else {
                ++it;
            }
        }

}

void server_prompt_cache::remove_superseded(const server_prompt & prompt, const server_prompt * saved) {
    for (auto it = states.begin(); it != states.end();) {
        if (it->read_only || &*it == saved || (saved != nullptr && it->file == saved->file) ||
                !server_prompt_supersedes(prompt, *it)) {
            ++it;
            continue;
        }
        SRV_WRN(" - removing cached prompt with length %d (%.3f MiB): an earlier state of the conversation saved with "
                "length %d, same first %" PRId64 " tokens\n",
                it->n_tokens(), it->size() / (1024.0 * 1024.0), prompt.n_tokens(), prompt.n_conversation);

        auto cur = it++;
        remove(cur);
    }
}

void server_prompt_cache::save(
        const server_prompt        & prompt,
        bool                         current,
        llama_context              * ctx,
        int32_t                      id_slot,
        common_context_seq_rm_type   seq_rm_type) {
    if (!prompt.adapter_key) {
        return;
    }
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
        remove_superseded(prompt, nullptr);

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
        entry.n_conversation = prompt.n_conversation;
        entry.adapter_key = prompt.adapter_key;
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
    std::string file_fingerprint = fingerprint;
    if (!prompt.adapter_key->empty()) {
        server_fnv1a(hash, prompt.adapter_key->data(), prompt.adapter_key->size());
        file_fingerprint += SERVER_PROMPT_ADAPTER_TAG + *prompt.adapter_key;
    }
    char name[32];
    std::snprintf(name, sizeof(name), "%016" PRIx64 ".prompt", hash);
    const std::string path = (std::filesystem::path(dir) / name).string();
    const std::string tmp  = path + ".tmp";

    server_prompt entry;
    entry.tokens = prompt.tokens.clone();
    entry.file   = path;
    entry.n_conversation = prompt.n_conversation;
    entry.adapter_key = prompt.adapter_key;

    bool ok = false;
    if (FILE * f = std::fopen(tmp.c_str(), "wb")) {
        server_prompt_file source(nullptr, server_prompt_file_close);
        const uint32_t magic = SERVER_PROMPT_FILE_MAGIC, version = SERVER_PROMPT_FILE_VERSION;
        const uint32_t n_fp = file_fingerprint.size(), n_tokens = tokens.size(), n_checkpoints = prompt.checkpoints.size();
        const uint64_t data_size = state_size;
        ok = server_prompt_fwrite(f, &magic, sizeof(magic)) && server_prompt_fwrite(f, &version, sizeof(version)) &&
             server_prompt_fwrite(f, &n_fp, sizeof(n_fp)) && server_prompt_fwrite(f, file_fingerprint.data(), n_fp) &&
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

        const int64_t header_size = ok ? server_prompt_file_tell(f) : -1;
        ok = ok && header_size >= 0 && uint64_t(header_size) <= SIZE_MAX;
        size_t offset = ok ? size_t(header_size) : 0;
        for (const auto & cp : prompt.checkpoints) {
            // a checkpoint in memory is written as it is; one in another prompt's file is read back first
            server_ssd_memory_change changes(ctx);
            std::vector<uint8_t> copy;
            const bool in_memory = !cp.data.empty();
            server_ssd_reservation reservation;
            if (ok && !in_memory) {
                if (!source && !prompt.file.empty()) {
                    source.reset(std::fopen(prompt.file.c_str(), "rb"));
                }
                reservation = server_ssd_reservation(ctx, cp.size());
                ok = bool(reservation) && server_prompt_checkpoint_data(source.get(), cp, copy);
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

    // only now that the new state is on disk
    remove_superseded(prompt, &states.back());
}

std::list<server_prompt>::iterator server_prompt_cache::best_to_load(
        const server_prompt              & prompt,
        const server_tokens              & tokens_new,
        int64_t                            n_reuse_slot,
        common_context_seq_rm_type         seq_rm_type,
        const server_prompt_checkpoint  ** checkpoint,
        int64_t                          * n_tokens) {
    auto it_best = states.end();
    const server_prompt_checkpoint * checkpoint_best = nullptr;
    int64_t n_tokens_best = -1;

    if (seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
        // A state that cannot be rolled back is reused up to its end or one of its checkpoints. Take the cached
        // prompt that lets the slot skip the most of the new prompt, if that is more than its own state does.
        SRV_WRN(" - looking for a better prompt, the slot reuses %" PRId64 " of %zu tokens\n", n_reuse_slot, tokens_new.size());

        n_tokens_best = n_reuse_slot;
        for (auto it = states.begin(); it != states.end(); ++it) {
            if (it->adapter_key != prompt.adapter_key) {
                continue;
            }
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
            if (it->adapter_key != prompt.adapter_key) {
                continue;
            }
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

    *checkpoint = checkpoint_best;
    *n_tokens   = n_tokens_best;
    return it_best;
}

bool server_prompt_cache::load(
        server_prompt              & prompt,
        const server_tokens        & tokens_new,
        int64_t                      n_reuse_slot,
        llama_context              * ctx,
        int32_t                      id_slot,
        common_context_seq_rm_type   seq_rm_type) {
    if (!prompt.adapter_key) {
        return true;
    }
    server_ssd_memory_change budget_changes(ctx);
    scan_prefix_dir();

    // Discard stale candidates before touching the slot. If a native restore fails after starting to replace
    // the sequence, clear the slot before trying the next candidate.
    for (;;) {
        const server_prompt_checkpoint * checkpoint_best = nullptr;
        int64_t n_tokens_best = -1;
        const auto it_best = best_to_load(prompt, tokens_new, n_reuse_slot, seq_rm_type, &checkpoint_best, &n_tokens_best);
        if (it_best == states.end()) {
            return true;
        }
        const auto result = take(it_best, checkpoint_best, n_tokens_best, prompt, ctx, id_slot);
        if (result == take_result::restored) {
            return true;
        }
        if (result == take_result::stale_index) {
            continue;
        }
        llama_memory_seq_rm(llama_get_memory(ctx), id_slot, -1, -1);
        prompt.tokens.clear();
        prompt.checkpoints.clear();
        n_reuse_slot = 0;
    }
}

server_prompt_cache::take_result server_prompt_cache::take(
        std::list<server_prompt>::iterator   it_best,
        const server_prompt_checkpoint     * checkpoint_best,
        int64_t                              n_tokens_best,
        server_prompt                      & prompt,
        llama_context                      * ctx,
        int32_t                              id_slot) {
    server_prompt_file file(nullptr, server_prompt_file_close);
    if (!it_best->file.empty()) {
        file.reset(it_best->read_only ? ssd::platform::open_stream(it_best->file.c_str())
                                      : std::fopen(it_best->file.c_str(), "rb"));
    }
    server_prompt opened;
    if (it_best->read_only) {
        // Bind validation and every read to this handle. The directory index can
        // be stale even after a scan if the pathname is replaced concurrently.
        if (!file || !server_prompt_read_header(file.get(), prefix_fingerprint, opened, true) ||
            !server_prompt_same_layout(*it_best, opened)) {
            SRV_WRN(" - prefix cache changed since indexing: %s\n", it_best->file.c_str());
            states.erase(it_best);
            prefix_dir_time = -1;
            return take_result::stale_index;
        }
        opened.file = it_best->file;
    }
    const server_prompt & entry = it_best->read_only ? opened : *it_best;

    SRV_WRN(" - found a better prompt%s%s: reuses %" PRId64 " tokens\n", checkpoint_best ? " checkpoint" : "",
            it_best->read_only ? " (prefix cache)" : "", n_tokens_best);

    if (!server_prompt_restore(entry, checkpoint_best, file.get(), ctx, id_slot)) {
        SRV_WRN("%s", "failed to restore the cached prompt\n");
        if (it_best->read_only) {
            ssd::platform::file_identity identity;
            it_best->prefix_identity = ssd::platform::identity(ssd::platform::stream_fd(file.get()), identity)
                ? std::optional{identity} : std::nullopt;
        }
        file.reset();
        remove(it_best);
        return take_result::restore_failed;
    }

    // the slot takes over the cached prompt, cut back to the checkpoint it starts from; its checkpoints move to memory
    server_prompt loaded;
    loaded.tokens = entry.tokens.clone();
    loaded.adapter_key = entry.adapter_key;
    for (const auto & cp : entry.checkpoints) {
        if (checkpoint_best != nullptr && cp.n_tokens > n_tokens_best) {
            continue;
        }
        // into memory as far as the memory limit allows (the RAM expert cache gives back room first)
        server_ssd_memory_change changes(ctx);
        server_ssd_reservation reservation(ctx, cp.size());
        if (!reservation) { continue; }
        server_prompt_checkpoint cur = cp;
        if (!server_prompt_checkpoint_data(file.get(), cp, cur.data)) { continue; }
        reservation.commit();
        cur.file_offset = 0;
        cur.file_size   = 0;
        loaded.checkpoints.push_back(std::move(cur));
    }
    if (checkpoint_best != nullptr) {
        loaded.tokens.keep_first(n_tokens_best);
    }
    file.reset();
    prompt = std::move(loaded);

    if (it_best->read_only) {
        states.splice(states.end(), states, it_best);
    } else if (on_disk()) {
        // the file stays: other conversations may start from it too
        states.splice(states.end(), states, it_best);
        server_prompt_file_touch(states.back().file);
    } else {
        states.erase(it_best);
    }

    return take_result::restored;
}

void server_prompt_cache::update() {
    // the cache's own states, least recently used first; the prefix caches are not the cache's to drop
    const auto oldest_own = [this]() {
        return std::find_if(states.begin(), states.end(), [](const server_prompt & s) { return !s.read_only; });
    };
    const auto n_own = [this]() {
        return (size_t) std::count_if(states.begin(), states.end(), [](const server_prompt & s) { return !s.read_only; });
    };

    if (limit_size > 0) {
        // always keep at least one state, regardless of the limits
        while (n_own() > 1 && size() > limit_size) {
            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", oldest_own()->size() / (1024.0 * 1024.0));

            remove(oldest_own());
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (n_own() > 1 && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, oldest_own()->size() / (1024.0 * 1024.0));

            remove(oldest_own());
        }
    }

    SRV_WRN(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)%s\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur,
            on_disk() ? " on disk" : "");

    for (const auto & state : states) {
        SRV_WRN("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB%s\n",
                (const void *)&state, state.n_tokens(), state.checkpoints.size(), state.size() / (1024.0 * 1024.0),
                state.read_only ? " (prefix cache)" : "");
    }
}

//
// a model's identity for prefix caches
//

static void server_fnv1a(uint64_t & hash, const char * data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        hash = (hash ^ (uint8_t) data[i])*1099511628211ull;
    }
}

std::string server_adapter_file_identity(const std::string & path) {
    std::ifstream in(std::filesystem::u8path(path), std::ios::binary);
    if (!in) {
        return "";
    }
    uint64_t hash = 1469598103934665603ull;
    std::vector<char> buf(1 << 20);
    while (in.read(buf.data(), (std::streamsize) buf.size()) || in.gcount() > 0) {
        server_fnv1a(hash, buf.data(), in.gcount());
    }
    if (!in.eof() || in.bad()) {
        return "";
    }
    char out[17];
    std::snprintf(out, sizeof(out), "%016" PRIx64, hash);
    return out;
}

std::optional<std::string> server_prompt_adapter_key(const std::vector<std::string> & identities,
        const std::vector<common_adapter_lora_info> & adapters) {
    uint64_t hash = 1469598103934665603ull;
    bool active = false;
    for (size_t i = 0; i < adapters.size(); ++i) {
        const auto & adapter = adapters[i];
        if (adapter.scale == 0) {
            continue;
        }
        // aLoRA depends on activation position, including tokens after the
        // prefix. Its per-slot reuse still follows the invocation rules.
        if (i >= identities.size() || identities[i].empty() || !std::isfinite(adapter.scale) ||
                llama_adapter_get_alora_n_invocation_tokens(adapter.ptr) != 0) {
            return std::nullopt;
        }
        active = true;
        server_fnv1a(hash, identities[i].data(), identities[i].size());
        uint32_t bits;
        static_assert(sizeof(bits) == sizeof(adapter.scale), "expected a 32-bit scale");
        std::memcpy(&bits, &adapter.scale, sizeof(bits));
        char scale[9];
        std::snprintf(scale, sizeof(scale), "%08" PRIx32, bits);
        server_fnv1a(hash, scale, sizeof(scale) - 1);
    }
    if (!active) {
        return std::string();
    }
    char out[17];
    std::snprintf(out, sizeof(out), "%016" PRIx64, hash);
    return std::string(out);
}

// "<size>:<hash of the header>" of one GGUF file, empty if it cannot be read
static std::string server_gguf_file_identity(const std::string & path, int64_t * split_count) {
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * gguf = gguf_init_from_file(path.c_str(), params);
    if (gguf == nullptr) {
        return "";
    }
    const size_t header = gguf_get_data_offset(gguf);
    if (split_count != nullptr) {
        const int64_t key = gguf_find_key(gguf, "split.count");
        *split_count = key >= 0 ? (int64_t) gguf_get_val_u16(gguf, key) : 0;
    }
    gguf_free(gguf);

    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    std::ifstream in(path, std::ios::binary);
    if (ec || !in) {
        return "";
    }
    uint64_t hash = 1469598103934665603ull;
    std::vector<char> buf(1 << 20);
    for (size_t left = header; left > 0;) {
        const size_t n = std::min(left, buf.size());
        if (!in.read(buf.data(), (std::streamsize) n)) {
            return "";
        }
        server_fnv1a(hash, buf.data(), n);
        left -= n;
    }
    char out[64];
    std::snprintf(out, sizeof(out), "%" PRIu64 ":%016" PRIx64, (uint64_t) size, hash);
    return out;
}

std::string server_model_files_identity(const std::vector<std::string> & paths) {
    std::string id;
    for (size_t i = 0; i < paths.size(); ++i) {
        int64_t n_split = 0;
        const std::string first = server_gguf_file_identity(paths[i], &n_split);
        if (first.empty()) {
            return "";
        }
        id += (id.empty() ? "" : ",") + first;
        if (n_split > 1) {
            // the other files of a split model, named after the first
            char prefix[4096];
            if (llama_split_prefix(prefix, sizeof(prefix), paths[i].c_str(), 0, (int) n_split) <= 0) {
                return "";
            }
            for (int k = 1; k < n_split; ++k) {
                char split[4096];
                llama_split_path(split, sizeof(split), prefix, k, (int) n_split);
                const std::string other = server_gguf_file_identity(split, nullptr);
                if (other.empty()) {
                    return "";
                }
                id += "," + other;
            }
        }
    }
    return id;
}
