#include "store.h"
#include "legacy-config.h"

#include <atomic>
#include <exception>
#include <mutex>

namespace {
struct runtime_owner {
    // Constructed before CUDA's borrowing state; destroyed after it. Normal
    // shutdown must stop readers before either backend can be unloaded.
    c0::parser_state catalog;
    std::mutex registration_mutex;
    std::once_flag initialized;
    const c0::provider * external = nullptr;
    bool started = false;
    std::atomic<bool> retired{false};
    std::atomic<bool> published{false};
};
runtime_owner & owner() { static runtime_owner value; return value; }

void initialize_cpu(c0::parser_state & destination) {
    for (const char * name : {"GGML_MOE_Q4_COLIBRI_PACK", "GGML_MOE_Q4_COLIBRI_MANIFEST",
            "GGML_MOE_GLM52_COLIBRI_PACK", "GGML_MOE_GLM52_COLIBRI_MANIFEST"}) {
        const char * value = std::getenv(name);
        if (value && value[0]) {
            std::fprintf(stderr, "[ssd_store] unsupported CPU-only source family: %s\n", name);
            return;
        }
    }
    if (c0::lossless_requested()) {
        std::fprintf(stderr, "[ssd_store] unsupported CPU-only source family: active lossless transport\n");
        return;
    }
    c0::detail::builder builder;
    auto parsed = builder.build(c0::description_from_env());
    if (!parsed) { return; }
    destination.sources = std::move(parsed->sources);
    destination.entries = std::move(parsed->entries);
    destination.io_backend = parsed->io_backend;
    destination.page_cache_sources = parsed->page_cache_sources;
    destination.file = destination.sources.empty() ? nullptr : destination.sources[0].file;
    destination.fd_direct = destination.sources.empty() ? -1 : destination.sources[0].fd_direct;
    destination.enabled = true;
    std::fprintf(stderr, "[ssd_store] directly_linked=1 initialized=1 sources=%zu\n", destination.sources.size());
}

const c0::expert_pack_entry * lookup_cpu(const char * name, int expert, size_t bytes) {
    const auto & entries = owner().catalog.entries;
    const auto found = std::lower_bound(entries.begin(), entries.end(), 0,
            [=](const c0::expert_pack_entry & item, int) {
                const int compare = std::strcmp(item.tensor, name);
                return compare ? compare < 0 : item.expert_idx != expert ? item.expert_idx < expert : item.nbytes < bytes;
            });
    return found != entries.end() && !std::strcmp(found->tensor, name) &&
            found->expert_idx == expert && found->nbytes == bytes ? &*found : nullptr;
}
} // namespace

namespace c0 {
parser_state & shared_catalog() { return owner().catalog; }
bool catalog_available() {
    auto & state = owner();
    return state.published.load(std::memory_order_acquire) && !state.retired.load(std::memory_order_acquire);
}
bool register_provider(const provider * value) {
    if (!value || !value->initialize || !value->lookup) { return false; }
    auto & state = owner();
    std::lock_guard<std::mutex> lock(state.registration_mutex);
    if (state.started || state.external || state.retired) { return false; }
    state.external = value;
    return true;
}
bool unregister_provider(const provider * value) {
    auto & state = owner();
    std::lock_guard<std::mutex> lock(state.registration_mutex);
    if (!value || state.external != value) { return false; }
    // An unloaded provider can never be silently replaced by the CPU loader.
    // The caller must quiesce all consumers before unloading its module.
    state.external = nullptr;
    state.retired = true;
    return true;
}
bool initialize_shared_catalog() {
    auto & state = owner();
    std::call_once(state.initialized, [&state] {
        const provider * selected;
        bool retired;
        {
            std::lock_guard<std::mutex> lock(state.registration_mutex);
            state.started = true;
            selected = state.external;
            retired = state.retired;
        }
        try {
            if (retired) {
                std::fprintf(stderr, "[ssd_store] provider was unloaded before initialization\n");
            } else if (selected) {
                selected->initialize();
            } else {
#if defined(GGML_SSD_REQUIRE_CUDA_PROVIDER)
                std::fprintf(stderr, "[ssd_store] CUDA provider is required but not registered\n");
#else
                initialize_cpu(state.catalog);
#endif
            }
        } catch (const std::exception & error) {
            std::fprintf(stderr, "[ssd_store] initialization failed: %s\n", error.what());
            state.catalog.enabled = false;
        } catch (...) {
            std::fprintf(stderr, "[ssd_store] initialization failed\n");
            state.catalog.enabled = false;
        }
        // A failed initializer must not expose partially opened/published data.
        if (!state.catalog.enabled) { state.catalog.reset(); }
        state.catalog.inited = true;
        state.published.store(state.catalog.enabled, std::memory_order_release);
        if (state.catalog.enabled) {
            std::fprintf(stderr, "[ssd_store] provider=%s sources=%zu entries=%zu published=1\n",
                    selected ? "cuda" : "portable", state.catalog.sources.size(), state.catalog.entries.size());
        }
    });
    return catalog_available();
}
} // namespace c0

extern "C" bool ggml_ssd_c0_resolve_raw_direct_shape(
        const char * name, int expert, size_t bytes, int64_t experts,
        int * fd, uint64_t * offset, size_t * logical_bytes) {
    if (!name || expert < 0 || experts <= 0 || expert >= experts || !bytes) { return false; }
    if (!fd || !offset || !logical_bytes) { return false; }
    try {
        if (!c0::initialize_shared_catalog()) { return false; }
        const char * disabled = std::getenv("GGML_MOE_EXPERT_PACK_RUNTIME_DISABLE");
        if (disabled && disabled[0] && disabled[0] != '0') { return false; }
        auto & state = owner();
        const c0::provider * selected;
        {
            std::lock_guard<std::mutex> lock(state.registration_mutex);
            if (state.retired) { return false; }
            selected = state.external;
        }
        const auto * entry = selected ? selected->lookup(name, expert, bytes) : lookup_cpu(name, expert, bytes);
#if defined(__linux__)
        if (!entry || entry->nbytes != bytes || entry->source_idx < 0 || size_t(entry->source_idx) >= state.catalog.sources.size()) { return false; }
        const auto & source = state.catalog.sources[size_t(entry->source_idx)];
        if (source.fd_direct < 0 || entry->offset > source.file_size || bytes > source.file_size - entry->offset) { return false; }
        const int flags = ::fcntl(source.fd_direct, F_GETFL);
        if (flags < 0 || !(flags & O_DIRECT)) { return false; }
        *fd = source.fd_direct; *offset = entry->offset; *logical_bytes = bytes;
        return true;
#else
        (void)entry;
        return false;
#endif
    } catch (const std::exception & error) {
        std::fprintf(stderr, "[ssd_store] resolve failed: %s\n", error.what());
        return false;
    } catch (...) {
        std::fprintf(stderr, "[ssd_store] resolve failed\n");
        return false;
    }
}
