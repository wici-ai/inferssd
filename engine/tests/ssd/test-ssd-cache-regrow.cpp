// SPDX-License-Identifier: Apache-2.0
// Actual cache geometry and decode boundaries, with injected budget observations.
// No model execution or expert payload is needed to inspect this metadata work.
#include "ggml-ssd/host/expert-cache.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

using cache_type = ssd::host::expert_cache;

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

struct budget_files {
    uint64_t available = uint64_t(8) << 30;
    size_t reads = 0;
    bool readable = true;
    bool read(const std::string & path, std::string & text) {
        ++reads;
        if (!readable) { return false; }
        if (path == "/proc/self/cgroup") { text = "0::/fixture\n"; }
        else if (path == "/sys/fs/cgroup/fixture/memory.max") { text = "8589934592"; }
        else if (path == "/sys/fs/cgroup/fixture/memory.current") { text = "0"; }
        else if (path == "/sys/fs/cgroup/fixture/memory.stat") {
            text = "active_file 0\ninactive_file 0\n";
        } else if (path == "/proc/meminfo") {
            text = "MemAvailable: " + std::to_string(available / 1024) + " kB\n";
        } else { return false; }
        return true;
    }
};

struct fixture {
    budget_files files;
    std::shared_ptr<ssd::host::process_arbiter> owner = std::make_shared<ssd::host::process_arbiter>(
        [this](const std::string & path, std::string & text) { return files.read(path, text); });
    std::unique_ptr<ssd::host::context_client> client;
    std::unique_ptr<cache_type> cache;

    fixture(int layers, int experts) {
        auto opened = owner->make_client();
        require(bool(opened), "cannot create host client");
        client = std::move(opened.value);
        ssd::host::expert_cache_options options;
        options.host = client.get();
        options.cache_policy = "lru";
        options.cache_allocation = "uniform";
        options.cache_keep = true;
        options.cache_partial_reclaim = true;
        cache = std::make_unique<cache_type>(options);
        std::vector<int32_t> priority(size_t(layers), 0);
        std::iota(priority.begin(), priority.end(), 0);
        cache->configure(cache_type::auto_budget, priority.data(), layers, experts);
        ggml_tensor tensor{};
        tensor.ne[2] = experts;
        const size_t roles[] = {4096, 4096, 4096};
        require(cache->geometry(roles, &tensor, 2), "cannot initialize cache geometry");
        require(cache->entries.size() == size_t(layers) * size_t(experts), "wrong cache capacity");
        require(cache->enabled_slots() == cache->entries.size(), "initial cache is not full");
    }

    void consistent() const {
        const auto & c = *cache;
        require(c.stride && c.stats.enabled_bytes == c.enabled_slots() * c.stride,
                "enabled byte accounting differs from actual slots");
    }
};

static void check_transitions(int layers, int experts) {
    fixture f(layers, experts);
    auto & c = *f.cache;
    f.consistent();
    const size_t full = c.entries.size();
    const size_t reads = f.files.reads;
    c.begin(1, 0, true);
    for (uint64_t token = 1; token < 4; ++token) { c.begin(1, token, true); }
    require(c.enabled_slots() == full && f.files.reads == reads, "full cache changed or measured room");
    const size_t partial = full / 2;
    require(c.shrink_to(partial, false, "fixture", true) == partial, "shrink changed its target");
    f.consistent();
    f.files.available = 0;
    c.begin(1, 4, true);
    require(c.enabled_slots() == partial && f.files.reads > reads, "pressure grew the cache or skipped measurement");
    f.consistent();
    f.files.readable = false;
    c.begin(1, 5, true);
    require(c.enabled_slots() == partial, "unreadable budget grew the cache");
    f.files.readable = true;
    f.files.available = uint64_t(8) << 30;
    c.begin(1, 6, true);
    require(c.enabled_slots() == full, "new room did not grow the cache");
    f.consistent();
    c.release();
    f.consistent();
    c.begin(2, 0, true);
    require(c.enabled_slots() == full, "release changed reusable capacity");
    require(c.shrink_to(0, false, "fixture", true) == 0, "cannot reclaim all slots");
    f.consistent();
    c.begin(2, 1, true);
    require(c.enabled_slots() == full, "empty cache did not regrow");
    f.consistent();
    ggml_tensor tensor{};
    tensor.ne[2] = experts;
    const size_t larger_roles[] = {8192, 4096, 4096};
    require(c.geometry(larger_roles, &tensor, 2), "cannot resize cache geometry");
    f.consistent();
    require(c.stride == 16384 && c.enabled_slots() == full, "geometry resize changed capacity");
    c.begin(2, 2, true);
    f.consistent();
    std::printf("CACHE_TRANSITIONS_PASS layers=%d experts=%d\n", layers, experts);
}

static void check_full_cache_without_entry_reads() {
    fixture f(64, 256);
    auto & c = *f.cache;
    c.begin(1, 0, true);
    const pid_t child = fork();
    require(child >= 0, "cannot fork metadata access check");
    if (child == 0) {
        alarm(5);
        // Protect only pages entirely inside the live entry vector. A full-cache
        // token boundary needs its recorded count, not any per-expert metadata.
        const size_t page = size_t(sysconf(_SC_PAGESIZE));
        const uintptr_t first = reinterpret_cast<uintptr_t>(c.entries.data());
        const uintptr_t start = (first + page - 1) / page * page;
        const uintptr_t end = (first + c.entries.size() * sizeof(cache_type::entry)) / page * page;
        if (end <= start || mprotect(reinterpret_cast<void *>(start), end - start, PROT_NONE)) { _exit(2); }
        c.begin(1, 1, true);
        if (mprotect(reinterpret_cast<void *>(start), end - start, PROT_READ | PROT_WRITE)) { _exit(3); }
        _exit(c.token == 1 && c.enabled_slots() == c.entries.size() ? 0 : 4);
    }
    int status = 0;
    pid_t waited;
    do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
    require(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "full-cache decode boundary touched per-expert metadata");
}

int main(int argc, char ** argv) {
    try {
        if (argc == 4) {
            const int layers = std::stoi(argv[1]), experts = std::stoi(argv[2]), tokens = std::stoi(argv[3]);
            require(tokens > 0, "benchmark token count must be positive");
            fixture f(layers, experts);
            auto & c = *f.cache;
            c.begin(1, 0, true);
            const size_t reads = f.files.reads;
            const auto start = std::chrono::steady_clock::now();
            for (int i = 1; i <= tokens; ++i) { c.begin(1, uint64_t(i), true); }
            const auto end = std::chrono::steady_clock::now();
            f.consistent();
            require(c.token == uint64_t(tokens) && f.files.reads == reads, "benchmark changed decode state");
            std::printf("{\"layers\":%d,\"experts\":%d,\"tokens\":%d,\"slots\":%zu,\"enabled_bytes\":%llu,\"us\":%.3f}\n",
                        layers, experts, tokens, c.entries.size(), (unsigned long long)c.stats.enabled_bytes,
                        std::chrono::duration<double, std::micro>(end - start).count());
        } else {
            require(argc == 1, "expected optional LAYERS EXPERTS TOKENS arguments");
            for (const auto & dims : {std::pair<int, int>{2, 8}, {3, 7}, {64, 256}, {64, 896}}) {
                check_transitions(dims.first, dims.second);
            }
            check_full_cache_without_entry_reads();
            std::puts("CACHE_REGROW_PASS");
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "CACHE_REGROW_FAIL: %s\n", error.what());
        return 1;
    }
}
