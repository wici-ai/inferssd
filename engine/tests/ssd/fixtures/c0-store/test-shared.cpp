#include "legacy-runtime/catalog.h"
#include "mock-provider.h"
#include "legacy-runtime/store.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <chrono>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

namespace {
void check(bool value, const char * message) { if (!value) { throw std::runtime_error(message); } }
bool mapping_absent(const void * pointer) {
#if defined(__APPLE__)
    // Darwin mincore succeeds for unmapped pages too; query the next VM region.
    mach_vm_address_t address = reinterpret_cast<mach_vm_address_t>(pointer);
    const auto requested = address;
    mach_vm_size_t length = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const auto result = ::mach_vm_region(mach_task_self(), &address, &length, VM_REGION_BASIC_INFO_64,
            reinterpret_cast<vm_region_info_t>(&info), &count, &object);
    if (object != MACH_PORT_NULL) { ::mach_port_deallocate(mach_task_self(), object); }
    return result == KERN_INVALID_ADDRESS || (result == KERN_SUCCESS && address > requested);
#else
    unsigned char resident = 0;
    return ::mincore(const_cast<void *>(pointer), 1, &resident) == -1 && errno == ENOMEM;
#endif
}
// Constructed before the function-local catalog owner. Its destructor runs
// after that owner and verifies normal teardown actually closed both handles.
struct shutdown_check {
    int buffered = -1;
    int direct = -1;
    ~shutdown_check() {
        for (int descriptor : {buffered, direct}) {
            if (descriptor >= 0 && (::fcntl(descriptor, F_GETFD) != -1 || errno != EBADF)) {
                std::fputs("FAIL: shared owner did not close its descriptor at shutdown\n", stderr);
                std::abort();
            }
        }
    }
} after_catalog;
struct fixture {
    std::filesystem::path root;
    ~fixture() { std::error_code error; if (!root.empty()) { std::filesystem::remove_all(root, error); } }
};
struct module {
    void * handle = nullptr;
    explicit module(const char * filename) { handle = ::dlopen(filename, RTLD_NOW | RTLD_LOCAL); check(handle, ::dlerror()); }
    ~module() { if (handle) { ::dlclose(handle); } }
    void close() { check(::dlclose(handle) == 0, "dlclose"); handle = nullptr; }
    template <typename T> T symbol(const char * name) {
        auto result = reinterpret_cast<T>(::dlsym(handle, name)); check(result != nullptr, name); return result;
    }
};
}

int main(int argc, char ** argv) {
    try {
        check(argc == 3, "usage: test-shared MODE MODULE");
        const std::string mode = argv[1];
        fixture files;
        char pattern[] = "/tmp/ssdash-c0-shared-XXXXXX";
        const char * directory = ::mkdtemp(pattern); check(directory, "mkdtemp"); files.root = directory;
        const auto data = files.root / "weights.bin", alias = files.root / "alias.tsv";
        { std::ofstream out(data, std::ios::binary); const std::string bytes(16384, '1'); out.write(bytes.data(), bytes.size()); check(bool(out), "fixture write"); }
        { std::ofstream out(alias); out << "tensor\texpert_idx\tnbytes\tsource_path\toffset\nblk.0.ffn_up_exps.weight\t3\t5000\t" << data.string() << "\t4097\n"; }
        for (const char * name : {"GGML_MOE_EXPERT_PACK", "GGML_MOE_EXPERT_PACK_LIST", "GGML_MOE_PAGECACHE_PACK", "GGML_MOE_PAGECACHE_PACK_LIST",
                "GGML_MOE_Q4_COLIBRI_PACK", "GGML_MOE_Q4_COLIBRI_MANIFEST", "GGML_MOE_GLM52_COLIBRI_PACK", "GGML_MOE_GLM52_COLIBRI_MANIFEST",
                "GGML_MOE_LOSSLESS_PACK", "GGML_MOE_GLM52_LOSSLESS_PACK", "GGML_MOE_LOSSLESS_MODE", "GGML_MOE_GLM52_LOSSLESS_MODE",
                "GGML_MOE_EXPERT_PACK_RUNTIME_DISABLE", "GGML_MOE_IO_REQUIRE_DIRECT"}) { ::unsetenv(name); }
        ::setenv("GGML_MOE_EXPERT_GGUF_ALIAS_TSV", alias.c_str(), 1);
        ::setenv("GGML_MOE_IO_BACKEND", "mmap", 1);
        module plugin(argv[2]);
        auto register_provider = plugin.symbol<decltype(&mock_register)>("mock_register");
        auto catalog_address = plugin.symbol<decltype(&mock_catalog)>("mock_catalog");
        auto entries_address = plugin.symbol<decltype(&mock_entries)>("mock_entries");
        auto sources_address = plugin.symbol<decltype(&mock_sources)>("mock_sources");
        auto calls = plugin.symbol<decltype(&mock_calls)>("mock_calls");
        auto opened_fd = plugin.symbol<decltype(&mock_fd)>("mock_fd");
        auto opened_direct = plugin.symbol<decltype(&mock_direct)>("mock_direct");
        auto mapping = plugin.symbol<decltype(&mock_mapping)>("mock_mapping");
        auto half_ready = plugin.symbol<decltype(&mock_half_ready)>("mock_half_ready");
        auto finish_load = plugin.symbol<decltype(&mock_finish_load)>("mock_finish_load");
        check(catalog_address() == &c0::shared_catalog(), "DSOs have distinct owners");
        if (mode == "late" || mode == "missing") {
            const bool initialized = c0::initialize_shared_catalog();
#if defined(C0_EXPECT_PROVIDER)
            check(!initialized && c0::shared_catalog().sources.empty(), "required provider silently fell back to CPU owner");
#else
            check(initialized && c0::shared_catalog().sources.size() == 1, "CPU-only initializer failed");
#endif
            const auto * source_before = c0::shared_catalog().sources.data();
            check(!register_provider(data.c_str(), 0), "late provider accepted");
            check(calls() == 0 && source_before == c0::shared_catalog().sources.data(), "late registration changed owner");
            if (initialized) { after_catalog.buffered = ::fileno(c0::shared_catalog().sources[0].file); }
            std::puts("PASS: CPU-only or required-provider state is final; late registration rejected");
            return 0;
        }
        const int failure = mode == "throw" ? 1 : mode == "disabled" ? 2 : mode == "page-cache" ? 3 :
            mode == "half-success" ? 4 : mode == "half-failure" ? 5 : 0;
        bool released_mapping = false;
        check(register_provider(data.c_str(), failure), "initial registration rejected");
        check(!register_provider(data.c_str(), failure), "duplicate registration accepted");
        if (failure == 4 || failure == 5) {
            auto loader = std::async(std::launch::async, [] { return c0::initialize_shared_catalog(); });
            while (!half_ready()) { std::this_thread::yield(); }
            const bool prematurely_visible = c0::catalog_available();
            auto observer = std::async(std::launch::async, [] { return c0::initialize_shared_catalog(); });
            const bool waited = observer.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
            finish_load();
            check(loader.get() == (failure == 4) && observer.get() == (failure == 4), "partial initialization result mismatch");
            if (failure == 5) { released_mapping = mapping_absent(mapping()); }
            check(waited && !prematurely_visible, "half catalog published or consumer bypassed once");
        }
        int fd = -1; uint64_t offset = 0; size_t count = 0;
        auto resolve = [&] { return ggml_ssd_c0_resolve_raw_direct_shape("blk.0.ffn_up_exps.weight", 3, 5000, 16, &fd, &offset, &count); };
        if (mode != "concurrent") {
            ::setenv("GGML_MOE_EXPERT_PACK_RUNTIME_DISABLE", "1", 1);
            check(!resolve() && calls() == 1, "runtime disable bypassed initial load or admitted lookup");
            if (failure == 1 || failure == 2) { released_mapping = mapping_absent(mapping()); }
            ::unsetenv("GGML_MOE_EXPERT_PACK_RUNTIME_DISABLE");
        }
        std::promise<void> release_readers;
        const auto start = release_readers.get_future().share();
        std::vector<std::future<bool>> readers;
        for (int i = 0; i < 8; ++i) {
            readers.emplace_back(std::async(std::launch::async, [start] {
                start.wait();
                bool result = false;
                for (int j = 0; j < 100; ++j) { result = c0::initialize_shared_catalog(); }
                return result;
            }));
        }
        release_readers.set_value();
        for (auto & reader : readers) { check(reader.get() == (failure != 1 && failure != 2 && failure != 5), "concurrent initialization state mismatch"); }
        check(calls() == 1, "initializer ran more than once");
        auto & catalog = c0::shared_catalog();
        if (failure == 1 || failure == 2 || failure == 5) {
            check(!catalog.enabled && catalog.inited && catalog.entries.empty() && catalog.sources.empty(), "failed initializer exposed half catalog");
            check(!c0::catalog_available(), "failure left catalog published");
            check(::fcntl(opened_fd(), F_GETFD) == -1 && errno == EBADF, "failed initialization leaked fd");
            if (opened_direct() >= 0) { check(::fcntl(opened_direct(), F_GETFD) == -1 && errno == EBADF, "failure leaked direct fd"); }
            // Check immediately after cleanup, before new worker stacks or
            // sanitizer allocations can legitimately reuse this virtual range.
            check(released_mapping, "failure leaked mapping");
            check(!register_provider(data.c_str(), 0) && !resolve(), "failed initialization was replaced or retried");
            std::puts("PASS: half-catalog failure cleaned up and remained final");
            return 0;
        }
        check(catalog.sources.size() == 1 && catalog.entries.size() == 1, "wrong catalog size");
        check(c0::catalog_available(), "successful catalog was not published");
        check(entries_address() == catalog.entries.data() && sources_address() == catalog.sources.data(), "DSO array addresses differ");
#if defined(__linux__)
        check(resolve() == (failure != 3), "direct/page-cache capability mismatch");
        if (failure != 3) { check(fd == catalog.sources[0].fd_direct && offset == 4097 && count == 5000, "logical direct range changed"); }
#else
        check(!resolve(), "non-Linux platform advertised O_DIRECT");
#endif
        const int buffered = ::fileno(catalog.sources[0].file);
        std::filesystem::rename(data, files.root / "original.bin");
        { std::ofstream out(data); out << std::string(16384, 'r'); }
        char bytes[5000];
        check(::pread(buffered, bytes, sizeof(bytes), 4097) == sizeof(bytes), "borrowed fd unreadable");
        for (char byte : bytes) { check(byte == '1', "path replacement changed source identity"); }
        check(entries_address() == catalog.entries.data() && sources_address() == catalog.sources.data(), "published arrays moved");
        after_catalog.buffered = buffered; after_catalog.direct = catalog.sources[0].fd_direct;
        if (mode == "unload") {
            plugin.close();
            check(!c0::initialize_shared_catalog() && !resolve(), "unloaded provider allowed replacement/fallback");
            check(::fcntl(buffered, F_GETFD) >= 0, "module unload closed shared-owner fd");
        }
        std::puts("PASS: shared owner, once load, stable arrays, source identity and final descriptor cleanup");
        return 0;
    } catch (const std::exception & error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
}
