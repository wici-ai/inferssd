#pragma once
// Compatibility catalog for the private shared-owner experiment. Entry storage
// is frozen after initialization. CUDA alone mutates its existing mmap fields.
#include <cstdint>
#include <cstdio>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

namespace c0 {
#include "types.inc"

struct parser_state {
    std::vector<expert_pack_source> sources;
    std::vector<expert_pack_entry> entries;
    FILE * file = nullptr; // borrowed first-source aliases used by legacy CUDA
    int fd_direct = -1;
    bool inited = false;
    bool enabled = false;
    int io_backend = 0;
    uint64_t page_cache_sources = 0;
    parser_state() = default;
    parser_state(const parser_state &) = delete;
    parser_state & operator=(const parser_state &) = delete;
    ~parser_state() { reset(); }
    void reset() {
        for (auto & source : sources) {
            if (source.mmap_base && source.mmap_size) { ::munmap(source.mmap_base, source.mmap_size); }
            if (source.file) { std::fclose(source.file); }
            if (source.fd_direct >= 0) { ::close(source.fd_direct); }
        }
        sources.clear(); entries.clear(); file = nullptr; fd_direct = -1;
        inited = false; enabled = false; io_backend = 0; page_cache_sources = 0;
    }
};

// The one instance is defined in a shared library, never as an inline singleton.
// Providers may populate it only inside their registered initializer. Once
// published, vectors and file descriptors must remain stable until shutdown.
parser_state & shared_catalog();
struct provider {
    void (*initialize)();
    const expert_pack_entry * (*lookup)(const char *, int, size_t);
};
bool register_provider(const provider * value);
bool unregister_provider(const provider * value);
bool initialize_shared_catalog();
// Does not initialize. Diagnostics must not inspect vectors until publication.
// A retired provider makes the catalog unavailable even while its fds live.
bool catalog_available();
} // namespace c0
