// SPDX-License-Identifier: Apache-2.0
// Real reader threads must publish every completion even when their heap is exhausted.
#include "io/pool-ring.h"
#include "platform/file.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <new>
#include <random>
#include <thread>
#include <vector>

static std::thread::id owner;
static std::atomic<bool> observing{false};
static std::atomic<size_t> worker_allocations{0};
static bool refuse_worker_allocations = true;

void * operator new(size_t bytes) {
    if (observing.load(std::memory_order_acquire) && std::this_thread::get_id() != owner) {
        worker_allocations.fetch_add(1, std::memory_order_relaxed);
        if (refuse_worker_allocations) { throw std::bad_alloc(); }
    }
    if (void * value = std::malloc(bytes ? bytes : 1)) { return value; }
    throw std::bad_alloc();
}
void * operator new[](size_t bytes) { return ::operator new(bytes); }
// GCC can misdiagnose inlined replacement deletes as mismatched new/free calls.
#if defined(__GNUC__)
#define NOINLINE_DELETE __attribute__((noinline))
#else
#define NOINLINE_DELETE
#endif
NOINLINE_DELETE void operator delete(void * value) noexcept { std::free(value); }
NOINLINE_DELETE void operator delete[](void * value) noexcept { std::free(value); }
NOINLINE_DELETE void operator delete(void * value, size_t) noexcept { std::free(value); }
NOINLINE_DELETE void operator delete[](void * value, size_t) noexcept { std::free(value); }

namespace {
constexpr size_t block = 4096;
constexpr size_t file_bytes = 17 * block + 1234;

void check(bool value, const char * message) {
    if (!value) {
        observing.store(false, std::memory_order_release);
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

void exercise(int fd, unsigned depth) {
    ssd::io::pool_ring ring;
    check(bool(ring.init(depth, false)), "pool init failed");
    const size_t slots = 2 * size_t(depth);
    auto * data = static_cast<unsigned char *>(::operator new(slots * block, std::align_val_t(block)));
    std::vector<bool> seen(slots);
    std::vector<int32_t> expected(slots);
    observing.store(true, std::memory_order_release);
    // Two batches can be outstanding: the portable completion limit is twice the SQ depth.
    for (unsigned round = 0; round < 128; ++round) {
        std::fill(seen.begin(), seen.end(), false);
        for (unsigned batch = 0; batch < 2; ++batch) {
            for (unsigned i = 0; i < depth; ++i) {
                const size_t slot = size_t(batch) * depth + i;
                const size_t at = (round + slot) % 20;
                const bool bad = at == 19;
                expected[slot] = bad ? -EBADF : at < 17 ? int32_t(block) : at == 17 ? 1234 : 0;
                check(bool(ring.queue({bad ? -1 : fd, at * block, block}, data + slot * block, slot, false)),
                      "pool queue failed");
            }
            check(bool(ring.submit(depth)), "pool submit failed");
        }
        check(ring.pending() == slots, "two batches were not retained");
        check(!ring.queue({fd, 0, block}, data, 0, false), "completion bound admitted an extra read");
        check(ring.pending() == slots, "refused read changed the pending count");
        size_t received = 0;
        while (received < slots) {
            ssd::io::completion entry{};
            const auto popped = ring.pop(entry);
            check(bool(popped), "pool pop failed");
            if (!popped.value) { check(bool(ring.wait()), "pool wait failed"); continue; }
            check(entry.user_data < slots && !seen[entry.user_data], "missing or duplicate completion tag");
            seen[entry.user_data] = true;
            check(entry.res == expected[entry.user_data], "wrong completion result");
            for (int32_t i = 0; i < entry.res; ++i) {
                check(data[entry.user_data * block + size_t(i)] == 'x', "read contents differ");
            }
            ++received;
        }
        check(ring.pending() == 0, "completed reads remain pending");
    }
    check(ring.completed() == 128 * slots, "completion counter differs");
    observing.store(false, std::memory_order_release);
    ring.close_ring();
    ::operator delete(data, std::align_val_t(block));
}
}

int main(int argc, char ** argv) {
    owner = std::this_thread::get_id();
    refuse_worker_allocations = argc < 3 || std::strcmp(argv[2], "--count-allocations") != 0;
    namespace fs = std::filesystem;
    const fs::path base = argc > 1 ? fs::path(argv[1]) : fs::temp_directory_path();
    const fs::path directory = base / ("inferssd-pool-completions-" + std::to_string(std::random_device{}()));
    fs::create_directories(directory);
    const fs::path path = directory / "data.bin";
    {
        std::ofstream output(path, std::ios::binary);
        const std::vector<char> contents(file_bytes, 'x');
        output.write(contents.data(), std::streamsize(contents.size()));
        check(bool(output), "fixture write failed");
    }
    const auto name = path.u8string();
    const int fd = ssd::platform::open_direct(std::string(name.begin(), name.end()).c_str());
    check(fd >= 0, "fixture direct open failed");
    for (unsigned depth : {1u, 3u, 24u}) { exercise(fd, depth); }
    ssd::platform::close_fd(fd);
    fs::remove_all(directory);
    std::printf("worker allocations=%zu; all 7168 reads completed with checked contents and tags\n",
                worker_allocations.load());
    return 0;
}
