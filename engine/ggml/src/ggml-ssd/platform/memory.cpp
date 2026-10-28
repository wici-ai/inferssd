// SPDX-License-Identifier: Apache-2.0
#include "memory.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace ssd { namespace platform {

size_t page_size() {
    static const size_t value = [] {
#if defined(_WIN32)
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        return size_t(info.dwPageSize);
#else
        return size_t(sysconf(_SC_PAGESIZE));
#endif
    }();
    return value;
}

void * aligned_alloc_bytes(size_t alignment, size_t bytes) {
#if defined(_WIN32)
    return _aligned_malloc(bytes, alignment);
#else
    void * p = nullptr;
    return posix_memalign(&p, alignment, bytes) == 0 ? p : nullptr;
#endif
}

void aligned_free(void * p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

void * arena_reserve(size_t bytes, std::string * why) {
#if defined(_WIN32)
    void * p = VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_NOACCESS);
    if (!p && why) {
        const DWORD error = GetLastError();
        *why = "VirtualAlloc reserve failed: error " + std::to_string(error);
    }
    return p;
#else
    void * p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        const int error = errno;
        if (why) { *why = std::string("mmap: ") + std::strerror(error); }
        return nullptr;
    }
    return p;
#endif
}

bool arena_commit(void * p, size_t bytes) {
    if (!bytes) { return true; }
#if defined(_WIN32)
    return VirtualAlloc(p, bytes, MEM_COMMIT, PAGE_READWRITE) != nullptr;
#elif defined(__APPLE__)
    return madvise(p, bytes, MADV_FREE_REUSE) == 0;
#else
    (void) p; // first touch charges
    return true;
#endif
}

bool arena_discard(void * p, size_t bytes) {
    if (!bytes) { return true; } // VirtualFree would read 0 as the whole region
#if defined(_WIN32)
    return VirtualFree(p, bytes, MEM_DECOMMIT) != 0;
#elif defined(__APPLE__)
    return madvise(p, bytes, MADV_FREE_REUSABLE) == 0;
#else
    return madvise(p, bytes, MADV_DONTNEED) == 0;
#endif
}

bool arena_no_hugepage(void * p, size_t bytes) {
#if defined(__linux__)
    return madvise(p, bytes, MADV_NOHUGEPAGE) == 0;
#else
    (void) p;
    (void) bytes;
    return true;
#endif
}

bool arena_populate(void * begin, size_t bytes) {
    if (!bytes) { return true; }
#if defined(__linux__) && defined(MADV_POPULATE_WRITE)
    if (madvise(begin, bytes, MADV_POPULATE_WRITE) == 0) { return true; }
#endif
    auto * data = static_cast<volatile char *>(begin);
    const size_t page = page_size();
    for (size_t at = 0; at < bytes; at += page) { data[at] = 0; }
    return false;
}

void arena_release(void * p, size_t bytes) {
    if (!p) { return; }
#if defined(_WIN32)
    (void) bytes;
    const bool released = VirtualFree(p, 0, MEM_RELEASE) != 0;
#else
    const bool released = munmap(p, bytes) == 0;
#endif
    if (!released) {
        std::fprintf(stderr, "[ssd_platform] cannot release arena; aborting\n");
        std::abort();
    }
}

} } // namespace ssd::platform
