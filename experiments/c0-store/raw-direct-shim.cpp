// One-process Linux feasibility adapter. Never used by production targets.
#include "prototype.h"
#include <mutex>

namespace {
std::unique_ptr<c0::expert_index> shared_store;
std::once_flag once;
std::vector<std::string> paths(const char * text) {
    std::vector<std::string> result;
    if (!text) { return result; }
    std::string value(text);
    size_t begin = 0;
    while (begin < value.size()) {
        const auto end = value.find(':', begin);
        const auto item = value.substr(begin, end == std::string::npos ? end : end - begin);
        if (!item.empty()) { result.push_back(item); }
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    return result;
}
}

extern "C" bool ggml_cuda_moe_expert_pack_resolve_raw_direct_shape(
        const char * name, int expert, size_t bytes, int64_t experts,
        int * fd, uint64_t * offset, size_t * logical_bytes) {
#if defined(__linux__)
    if (!name || expert < 0 || experts <= 0 || expert >= experts || !bytes || !fd || !offset || !logical_bytes) { return false; }
    // Retain the unsupported K3-local shapes in this narrow test bridge.
    if (experts == 896 && (bytes == 3612672 || bytes == 2150400)) { return false; }
    std::call_once(once, [] {
        c0::description desc;
        desc.io_backend = 1;
        desc.packs = paths(std::getenv("GGML_MOE_EXPERT_PACK"));
        desc.aliases = paths(std::getenv("GGML_MOE_EXPERT_GGUF_ALIAS_TSV"));
        shared_store = c0::expert_index::open(desc);
    });
    c0::extent range;
    if (!shared_store || !shared_store->locate(name, expert, bytes, range)) { return false; }
    const int source_fd = shared_store->borrowed_direct_fd(range.source_id);
    const int flags = source_fd < 0 ? -1 : ::fcntl(source_fd, F_GETFL);
    if (flags < 0 || !(flags & O_DIRECT)) { return false; }
    *fd = source_fd;
    *offset = range.offset;
    *logical_bytes = bytes;
    return true;
#else
    (void)name; (void)expert; (void)bytes; (void)experts; (void)fd; (void)offset; (void)logical_bytes;
    return false;
#endif
}
