// SPDX-License-Identifier: Apache-2.0
#include "index.h"
#include "legacy-index.h"

namespace ssd { namespace store {
namespace {
struct close_probe { void operator()(FILE * file) const noexcept { std::fclose(file); } };
FILE * open_probe(const char * path) noexcept {
    const int fd = ::open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) { return nullptr; }
    struct stat info {};
    if (fstat(fd, &info) || !S_ISREG(info.st_mode)) { ::close(fd); return nullptr; }
    FILE * file = fdopen(fd, "rb");
    if (!file) { ::close(fd); }
    return file;
}
}
struct source_index::implementation {
    std::unique_ptr<c0::expert_index> index;
};

source_index::source_index(std::unique_ptr<implementation> state) : state_(std::move(state)) {}
source_index::~source_index() = default;

result<std::unique_ptr<source_index>> source_index::open(const std::vector<std::string> & paths) noexcept {
    try {
        if (paths.empty()) { return {nullptr, status::invalid, "expert.source requires an index path"}; }
        c0::description description;
        // The index opens direct handles; the CPU executor owns its io_uring
        // ring. No transport choice or mutable environment is parsed here.
        description.io_backend = 1;
        description.require_direct = true;
        for (const auto & path : paths) {
            std::unique_ptr<FILE, close_probe> probe(open_probe(path.c_str()));
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
            struct stat direct = {};
            const int fd = index->borrowed_direct_fd(int(i));
            if (fd < 0 || fstat(fd, &direct) || direct.st_size < 0 ||
                identity.device != uint64_t(direct.st_dev) || identity.inode != uint64_t(direct.st_ino) ||
                identity.size != uint64_t(direct.st_size)) {
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
    return {this, [](void * opaque, const char * name, int expert, size_t bytes, int64_t experts,
                    int * fd, uint64_t * offset, size_t * logical_bytes) noexcept {
        if (!opaque || !name || expert < 0 || experts <= 0 || expert >= experts || !bytes || !fd || !offset || !logical_bytes) {
            return false;
        }
        const auto & index = *static_cast<source_index *>(opaque)->state_->index;
        c0::extent extent;
        if (!index.locate(name, expert, bytes, extent, false)) { return false; }
        const int direct = index.borrowed_direct_fd(extent.source_id);
#if defined(__linux__)
        const int flags = direct < 0 ? -1 : fcntl(direct, F_GETFL);
        if (flags < 0 || !(flags & O_DIRECT)) { return false; }
#else
        if (direct < 0) { return false; }
#endif
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
}
} }
