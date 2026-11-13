// SPDX-License-Identifier: Apache-2.0
#include "index.h"
#include "legacy-index.h"
#include "../platform/file.h"

namespace ssd { namespace store {
namespace {
struct close_probe { void operator()(FILE * file) const noexcept { std::fclose(file); } };
}
struct source_index::implementation {
    std::unique_ptr<c0::expert_index> index;
    std::vector<std::unique_ptr<source_index>> replicas;
    std::vector<int> replica_fds;
    size_t max_experts = 0;
};

source_index::source_index(std::unique_ptr<implementation> state) : state_(std::move(state)) {}
source_index::~source_index() = default;

result<std::unique_ptr<source_index>> source_index::open(const std::vector<std::string> & paths, const std::vector<std::string> & replicas, size_t max_experts) noexcept {
    try {
        if (paths.empty()) { return {nullptr, status::invalid, "expert.source requires an index path"}; }
        c0::description description;
        // The index opens direct handles; the CPU executor owns its read
        // ring. No transport choice or mutable environment is parsed here.
        description.io_backend = 1;
        description.require_direct = true;
        for (const auto & path : paths) {
            std::unique_ptr<FILE, close_probe> probe(platform::open_stream(path.c_str()));
            if (!probe) { return {nullptr, status::io, "cannot open expert index: " + path}; }
            char magic[16] = {};
            const auto count = std::fread(magic, 1, sizeof(magic), probe.get());
            if (count >= 13 && std::memcmp(magic, "GGMLMOEPACKv1", 13) == 0) {
                description.packs.push_back(path);
            } else if (count >= 4 && std::memcmp(magic, "GGUF", 4) == 0) {
                return {nullptr, status::invalid, "expert.source takes an alias TSV for direct GGUF reads: " + path};
            } else {
                description.aliases.push_back(path);
            }
        }
        auto index = c0::expert_index::open(description);
        if (!index) { return {nullptr, status::invalid, "expert source index validation failed"}; }
        // Buffered metadata and direct payload handles must identify the same
        // file even if a path was replaced between the two opens.
        std::vector<c0::source_identity> identities;
        for (size_t i = 0; i < index->source_count(); ++i) {
            const auto identity = index->source_identity_for_test(i);
            platform::file_identity direct;
            const int fd = index->borrowed_direct_fd(int(i));
            if (fd < 0 || !platform::identity(fd, direct) || identity.device != direct.device ||
                identity.inode != direct.id_lo || identity.inode_hi != direct.id_hi || identity.size != direct.size) {
                return {nullptr, status::io, "direct expert source identity differs from metadata handle"};
            }
            identities.push_back(identity);
        }
        for (const auto & entry : index->entries_for_test()) {
            if (entry.source_idx < 0 || size_t(entry.source_idx) >= identities.size() ||
                entry.offset > identities[size_t(entry.source_idx)].size ||
                entry.nbytes > identities[size_t(entry.source_idx)].size - entry.offset) {
                return {nullptr, status::invalid, "expert index range exceeds its source file"};
            }
        }
        auto state = std::make_unique<implementation>();
        state->index = std::move(index);
        if (!replicas.empty()) {
            if (replicas.size() < 2 || replicas.size() > 8 || !max_experts || max_experts > 8) {
                return {nullptr, status::invalid, "replica experiment requires 2..8 indexes and max_experts 1..8"};
            }
            std::vector<c0::source_identity> replica_ids;
            for (const auto & path : replicas) {
                auto replica = source_index::open({path});
                if (!replica || replica.value->state_->index->source_count() != 1) {
                    return {nullptr, status::invalid, "replica must resolve to one regular payload file"};
                }
                const auto & candidate = *replica.value->state_->index;
                const auto id = candidate.source_identity_for_test(0);
                for (const auto & previous : replica_ids) {
                    if (id == previous || id.size != previous.size) {
                        return {nullptr, status::invalid, "replicas must be distinct equally sized files"};
                    }
                }
                for (const auto & entry : state->index->entries_for_test()) {
                    c0::extent extent;
                    if (!candidate.locate(entry.tensor, entry.expert_idx, entry.nbytes, extent) || extent.offset != entry.offset) {
                        return {nullptr, status::invalid, "replica tensor layout differs from logical source"};
                    }
                }
                replica_ids.push_back(id);
                state->replica_fds.push_back(candidate.borrowed_direct_fd(0));
                state->replicas.push_back(std::move(replica.value));
            }
            for (const auto & id : identities) {
                if (std::find(replica_ids.begin(), replica_ids.end(), id) == replica_ids.end()) {
                    return {nullptr, status::invalid, "logical source is not a declared replica"};
                }
            }
            state->max_experts = max_experts;
        }
        return {std::unique_ptr<source_index>(new source_index(std::move(state))), status::ok, {}};
    } catch (const std::bad_alloc &) {
        return {nullptr, status::resource, "oom"};
    } catch (const std::exception & error) {
        try { return {nullptr, status::invalid, error.what()}; }
        catch (...) { return {nullptr, status::resource, "oom"}; }
    } catch (...) {
        return {nullptr, status::invalid, "source invalid"};
    }
}

source_view source_index::view() noexcept {
    source_view result{this, [](void * opaque, const char * name, int expert, size_t bytes, int64_t experts,
                    int * fd, uint64_t * offset, size_t * logical_bytes) noexcept {
        if (!opaque || !name || expert < 0 || experts <= 0 || expert >= experts || !bytes || !fd || !offset || !logical_bytes) {
            return false;
        }
        const auto & index = *static_cast<source_index *>(opaque)->state_->index;
        c0::extent extent;
        if (!index.locate(name, expert, bytes, extent, false)) { return false; }
        const int direct = index.borrowed_direct_fd(extent.source_id);
        if (!platform::is_direct(direct)) { return false; }
        *fd = direct; *offset = extent.offset; *logical_bytes = extent.logical_bytes;
        return true;
    }, [](void * opaque, const char * name, size_t bytes, size_t * block_bytes) noexcept -> size_t {
        if (!opaque || !name || !name[0] || !bytes) { return 0; }
        const auto & entries = static_cast<source_index *>(opaque)->state_->index->entries_for_test();
        const auto first = std::lower_bound(entries.begin(), entries.end(), name,
            [](const c0::expert_pack_entry & entry, const char * value) { return std::strcmp(entry.tensor, value) < 0; });
        if (first == entries.end() || std::strcmp(first->tensor, name) || first->expert_idx != 0 ||
                !first->nbytes || first->nbytes > SIZE_MAX || bytes % first->nbytes) { return 0; }
        const size_t block = size_t(first->nbytes), count = bytes / block;
        if (!count || count > INT32_MAX || count > size_t(entries.end() - first)) { return 0; }
        for (size_t i = 0; i < count; ++i) {
            const auto & entry = first[i];
            if (std::strcmp(entry.tensor, name) || entry.expert_idx != int32_t(i) || entry.nbytes != block) { return 0; }
        }
        if (first + count != entries.end() && !std::strcmp(first[count].tensor, name)) { return 0; }
        if (block_bytes) { *block_bytes = block; }
        return count;
    }};
    result.replica_fds = state_->replica_fds.data();
    result.replica_count = state_->replica_fds.size();
    result.range_max_experts = state_->max_experts;
    return result;
}
} }
