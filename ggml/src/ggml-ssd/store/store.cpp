#include "store.h"
#include "legacy-config.h"

#include <atomic>
#include <exception>
#include <mutex>

namespace {
// Match the existing process-wide first-shape choice without importing CUDA
// transport. CUDA builds cannot enable this prototype: they still own the
// original dispatch and index until the shared-owner cut is independently gated.
std::atomic<int> selected_shape{0};
std::once_flag initialized;
std::unique_ptr<c0::expert_index> index;

bool common_shape(int64_t experts, size_t bytes) {
    const int requested = experts == 896 && (bytes == 3612672 || bytes == 2150400) ? 2 : 1;
    int expected = 0;
    selected_shape.compare_exchange_strong(expected, requested, std::memory_order_acq_rel);
    return selected_shape.load(std::memory_order_acquire) == 1;
}

void initialize() {
    // This prototype deliberately supports only the legacy v1/alias catalog.
    // Refuse unrelated formats rather than silently treating them as v1 input.
    for (const char * name : {"GGML_MOE_Q4_COLIBRI_PACK", "GGML_MOE_Q4_COLIBRI_MANIFEST",
            "GGML_MOE_GLM52_COLIBRI_PACK", "GGML_MOE_GLM52_COLIBRI_MANIFEST"}) {
        const char * value = std::getenv(name);
        if (value && value[0]) {
            std::fprintf(stderr, "[c0_store_prototype] unsupported source family: %s\n", name);
            return;
        }
    }
    index = c0::expert_index::open(c0::description_from_env());
    std::fprintf(stderr, "[c0_store_prototype] directly_linked=1 initialized=%d sources=%zu\n",
            index ? 1 : 0, index ? index->source_count() : 0);
}
} // namespace

extern "C" bool ggml_ssd_c0_resolve_raw_direct_shape(
        const char * name, int expert, size_t bytes, int64_t experts,
        int * fd, uint64_t * offset, size_t * logical_bytes) {
    if (!name || expert < 0 || experts <= 0 || expert >= experts || !bytes || !fd || !offset || !logical_bytes) {
        return false;
    }
    if (!common_shape(experts, bytes)) { return false; }
    try {
        // Failed initialization is also final, like the legacy inited flag.
        std::call_once(initialized, [] {
            try { initialize(); }
            catch (const std::exception & error) {
                std::fprintf(stderr, "[c0_store_prototype] initialization failed: %s\n", error.what());
            }
            catch (...) { std::fprintf(stderr, "[c0_store_prototype] initialization failed\n"); }
        });
#if defined(__linux__)
        c0::extent range;
        if (!index || !index->locate(name, expert, bytes, range)) { return false; }
        const int source = index->borrowed_direct_fd(range.source_id);
        const int flags = source < 0 ? -1 : ::fcntl(source, F_GETFL);
        if (flags < 0 || !(flags & O_DIRECT)) { return false; }
        *fd = source;
        *offset = range.offset;
        *logical_bytes = bytes;
        return true;
#else
        return false;
#endif
    } catch (const std::exception & error) {
        std::fprintf(stderr, "[c0_store_prototype] resolve failed: %s\n", error.what());
        return false;
    } catch (...) {
        std::fprintf(stderr, "[c0_store_prototype] resolve failed\n");
        return false;
    }
}
