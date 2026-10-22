#include "catalog.h"
#include "store.h"

#include <atomic>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>

namespace {
std::string filename;
int failure_mode = 0;
std::atomic<unsigned> calls{0};
int observed_fd = -1;
bool registered = false;
std::atomic<bool> half_ready{false};
std::atomic<bool> finish_load{false};
int observed_direct = -1;
void * observed_mapping = nullptr;

void initialize() {
    ++calls;
    auto & catalog = c0::shared_catalog();
    c0::expert_pack_source source;
    source.file = std::fopen(filename.c_str(), "rb");
    if (!source.file) { throw std::runtime_error("mock fopen"); }
    // Immediately transfer ownership, including on later failure.
    catalog.sources.push_back(source);
    auto & stored = catalog.sources.back();
    observed_fd = ::fileno(stored.file);
    struct stat status{};
    if (::fstat(observed_fd, &status)) { throw std::runtime_error("mock fstat"); }
    stored.file_size = uint64_t(status.st_size);
    stored.mmap_base = ::mmap(nullptr, size_t(status.st_size), PROT_READ, MAP_PRIVATE, observed_fd, 0);
    if (stored.mmap_base == MAP_FAILED) {
        stored.mmap_base = nullptr;
        throw std::runtime_error("mock mmap");
    }
    stored.mmap_size = size_t(status.st_size);
    observed_mapping = stored.mmap_base;
    stored.mmap_attempted = stored.mmap_enabled = true;
    stored.page_cache = failure_mode == 3;
#if defined(__linux__)
    if (!stored.page_cache) {
        stored.fd_direct = ::open(filename.c_str(), O_RDONLY | O_DIRECT);
        observed_direct = stored.fd_direct;
        if (stored.fd_direct < 0) { throw std::runtime_error("mock O_DIRECT"); }
    }
#endif
    c0::expert_pack_entry entry;
    std::strcpy(entry.tensor, "blk.0.ffn_up_exps.weight");
    entry.expert_idx = 3;
    entry.offset = 4097;
    entry.nbytes = 5000;
    entry.alias_source = true;
    catalog.entries.push_back(entry);
    catalog.file = stored.file;
    catalog.fd_direct = stored.fd_direct;
    catalog.enabled = failure_mode != 2;
    if (failure_mode == 4 || failure_mode == 5) {
        half_ready.store(true, std::memory_order_release);
        while (!finish_load.load(std::memory_order_acquire)) { std::this_thread::yield(); }
    }
    if (failure_mode == 1 || failure_mode == 5) { throw std::runtime_error("expected half-catalog failure"); }
}

const c0::expert_pack_entry * lookup(const char * name, int expert, size_t bytes) {
    // Matches the legacy callback's re-entry after initialization completed.
    if (!c0::initialize_shared_catalog()) { return nullptr; }
    const auto & catalog = c0::shared_catalog();
    if (catalog.entries.empty()) { return nullptr; }
    const auto & entry = catalog.entries.front();
    return std::strcmp(entry.tensor, name) == 0 && expert == entry.expert_idx && bytes == entry.nbytes ? &entry : nullptr;
}
const c0::provider callbacks{initialize, lookup};
struct teardown {
    ~teardown() {
        if (registered && !c0::unregister_provider(&callbacks)) { std::abort(); }
    }
} retire_before_module_unload;
} // namespace

extern "C" bool mock_register(const char * file, int mode) {
    filename = file;
    failure_mode = mode;
    const bool accepted = c0::register_provider(&callbacks);
    if (accepted) { registered = true; }
    return accepted;
}
extern "C" const void * mock_catalog() { return &c0::shared_catalog(); }
extern "C" const void * mock_entries() { return c0::shared_catalog().entries.data(); }
extern "C" const void * mock_sources() { return c0::shared_catalog().sources.data(); }
extern "C" unsigned mock_calls() { return calls.load(); }
extern "C" int mock_fd() { return observed_fd; }
extern "C" int mock_direct() { return observed_direct; }
extern "C" const void * mock_mapping() { return observed_mapping; }
extern "C" bool mock_half_ready() { return half_ready.load(std::memory_order_acquire); }
extern "C" void mock_finish_load() { finish_load.store(true, std::memory_order_release); }
extern "C" void mock_shape(int64_t experts, size_t bytes) { ggml_ssd_c0_select_model_shape(experts, bytes, "mock-cuda"); }
extern "C" bool mock_common() { return ggml_ssd_c0_uses_common(); }
