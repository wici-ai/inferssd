// SPDX-License-Identifier: Apache-2.0
// Platform memory layer: aligned blocks and the elastic arena round trip,
// checking that discard returns a slot's charge where each OS accounts it
// (Linux residency, macOS phys_footprint, Windows commit charge).
#include "ggml-ssd/platform/memory.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#else
#include <sys/mman.h>
#endif

using namespace ssd::platform;

static void require(bool value, const std::string & why) {
    if (!value) { throw std::runtime_error(why); }
}

// Bytes the OS charges for `slot` (Linux) or for the process (macOS, Windows).
static uint64_t charged(const unsigned char * slot, size_t bytes) {
#if defined(_WIN32)
    (void) slot;
    (void) bytes;
    PROCESS_MEMORY_COUNTERS_EX counters = {};
    require(GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters),
            sizeof(counters)) != 0, "GetProcessMemoryInfo");
    return counters.PrivateUsage;
#elif defined(__APPLE__)
    (void) slot;
    (void) bytes;
    task_vm_info_data_t info = {};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    require(task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS,
            "task_info");
    return info.phys_footprint;
#else
    const size_t page = page_size();
    std::vector<unsigned char> resident((bytes + page - 1) / page);
    require(mincore(const_cast<unsigned char *>(slot), bytes, resident.data()) == 0, "mincore");
    uint64_t total = 0;
    for (unsigned char r : resident) { total += (r & 1) ? page : 0; }
    return total;
#endif
}

static void fill(unsigned char * p, size_t bytes, unsigned seed) {
    for (size_t i = 0; i < bytes; ++i) { p[i] = static_cast<unsigned char>(i * 131 + seed); }
}

static bool holds(const unsigned char * p, size_t bytes, unsigned seed) {
    for (size_t i = 0; i < bytes; ++i) {
        if (p[i] != static_cast<unsigned char>(i * 131 + seed)) { return false; }
    }
    return true;
}

static void test_aligned() {
    const size_t page = page_size();
    require(page >= 4096 && (page & (page - 1)) == 0, "page size is a power of two of at least 4 KiB");
    for (size_t alignment : { sizeof(void *), size_t(64), size_t(4096), page }) {
        auto * p = static_cast<unsigned char *>(aligned_alloc_bytes(alignment, 12345));
        require(p && reinterpret_cast<uintptr_t>(p) % alignment == 0, "aligned to " + std::to_string(alignment));
        fill(p, 12345, 7);
        require(holds(p, 12345, 7), "aligned block holds data");
        aligned_free(p);
    }
}

static void test_arena() {
    const size_t page = page_size(), slot = size_t(32) << 20, slots = 4, bytes = slot * slots;
    const size_t margin = slot / 8; // other allocations of this process during a step
    std::string why;
    auto * arena = static_cast<unsigned char *>(arena_reserve(bytes, &why));
    require(arena != nullptr, "arena_reserve: " + why);
    require(reinterpret_cast<uintptr_t>(arena) % page == 0, "the arena is page aligned");
    require(arena_no_hugepage(arena, bytes), "arena_no_hugepage");
    unsigned char * a = arena + slot, * b = arena + 2 * slot;

    const uint64_t base = charged(a, slot);
    require(arena_commit(a, slot), "commit a slot");
    fill(a, slot, 1);
    require(holds(a, slot, 1), "a committed slot holds data");
    const uint64_t loaded = charged(a, slot);
    require(loaded >= base + slot - margin, "a written slot is charged");

    require(arena_discard(a, slot), "discard a slot");
    const uint64_t dropped = charged(a, slot);
    std::printf("slot charge: before=%llu written=%llu discarded=%llu (MiB, %s)\n",
                (unsigned long long) (base >> 20), (unsigned long long) (loaded >> 20), (unsigned long long) (dropped >> 20),
#if defined(_WIN32)
                "commit charge"
#elif defined(__APPLE__)
                "phys_footprint"
#else
                "resident"
#endif
                );
    require(dropped + slot - margin <= loaded, "discard returns the slot's charge");

    require(arena_commit(a, slot), "commit the slot again");
    fill(a, slot, 2);
    require(holds(a, slot, 2), "a recommitted slot holds new data");
    require(charged(a, slot) >= dropped + slot - margin, "a rewritten slot is charged again");

    require(arena_commit(b, slot), "commit a second slot");
    const uint64_t before_populate = charged(b, slot);
    const bool kernel = arena_populate(b, slot);
    const uint64_t populated = charged(b, slot);
    std::printf("populate: %s, %llu MiB\n", kernel ? "kernel" : "touched",
                (unsigned long long) ((populated - before_populate) >> 20));
#if defined(_WIN32)
    (void) populated; // commit charge was taken by arena_commit already
#else
    require(populated >= before_populate + slot - margin, "populate makes the slot resident");
#endif
    require(arena_populate(b, 0), "an empty populate is a no-op");
    fill(b, slot, 3);
    require(holds(b, slot, 3) && holds(a, slot, 2), "slots are independent");
    require(arena_commit(b, 0) && arena_discard(b, 0), "empty ranges are no-ops");

    arena_release(arena, bytes);
}

int main() {
    try {
        test_aligned();
        test_arena();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    std::printf("ssd platform memory: OK\n");
    return 0;
}
