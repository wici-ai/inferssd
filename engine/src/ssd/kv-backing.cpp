// SPDX-License-Identifier: Apache-2.0
#include "kv-backing.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>
#include <atomic>
#include <chrono>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#endif

namespace ssd {

struct kv_backing::impl {
    void * address = nullptr;
    size_t bytes = 0;
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
    ~impl() {
        if (address) { UnmapViewOfFile(address); }
        if (mapping) { CloseHandle(mapping); }
        if (file != INVALID_HANDLE_VALUE) { CloseHandle(file); }
    }
#else
    int file = -1;
    ~impl() {
        if (address) { munmap(address, bytes); }
        if (file >= 0) { close(file); }
    }
#endif
};

static std::runtime_error backing_error(const std::filesystem::path & directory,
        const char * operation, int error) {
#ifdef _WIN32
    const std::error_code code(error, std::system_category());
#else
    const std::error_code code(error, std::generic_category());
#endif
    return std::runtime_error("SSD-backed KV cache: " + std::string(operation) + " in '" +
        directory.u8string() + "': " + code.message() +
        ". Use a writable local SSD directory with enough free space for the entire KV cache.");
}

kv_backing::kv_backing(const std::filesystem::path & directory, size_t bytes) : pimpl(new impl) {
    if (!bytes) { throw std::invalid_argument("SSD-backed KV cache: mapping size must be positive"); }
    pimpl->bytes = bytes;

#ifdef _WIN32
    if (bytes > size_t(std::numeric_limits<LONGLONG>::max())) {
        throw std::invalid_argument("SSD-backed KV cache: file size exceeds the platform limit");
    }
    static std::atomic<unsigned long long> next{0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    for (int attempt = 0; attempt < 128; ++attempt) {
        const auto path = directory / ("inferssd-kv-" + std::to_string(GetCurrentProcessId()) + "-" +
            std::to_string(stamp) + "-" + std::to_string(next.fetch_add(1)) + ".tmp");
        pimpl->file = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
            CREATE_NEW, FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
        if (pimpl->file != INVALID_HANDLE_VALUE) { break; }
        const DWORD error = GetLastError();
        if (error != ERROR_FILE_EXISTS && error != ERROR_ALREADY_EXISTS) {
            throw backing_error(directory, "cannot create a private temporary backing file", int(error));
        }
    }
    if (pimpl->file == INVALID_HANDLE_VALUE) {
        throw backing_error(directory, "cannot create a unique backing file", ERROR_FILE_EXISTS);
    }
    FILE_ALLOCATION_INFO allocation{};
    allocation.AllocationSize.QuadPart = LONGLONG(bytes);
    if (!SetFileInformationByHandle(pimpl->file, FileAllocationInfo, &allocation, sizeof(allocation))) {
        throw backing_error(directory, "cannot reserve disk space", int(GetLastError()));
    }
    LARGE_INTEGER end{};
    end.QuadPart = LONGLONG(bytes);
    if (!SetFilePointerEx(pimpl->file, end, nullptr, FILE_BEGIN) || !SetEndOfFile(pimpl->file)) {
        throw backing_error(directory, "cannot set the backing file size", int(GetLastError()));
    }
    // Explicitly guarantee zero contents without touching the future mapped pages.
    // SetFileValidData must not be used: it can expose uninitialized disk contents.
    FILE_ZERO_DATA_INFORMATION zero{};
    zero.BeyondFinalZero.QuadPart = LONGLONG(bytes);
    DWORD returned = 0;
    if (!DeviceIoControl(pimpl->file, FSCTL_SET_ZERO_DATA, &zero, sizeof(zero),
            nullptr, 0, &returned, nullptr)) {
        throw backing_error(directory, "cannot zero the backing file", int(GetLastError()));
    }
    FILE_STANDARD_INFO info{};
    if (!GetFileInformationByHandleEx(pimpl->file, FileStandardInfo, &info, sizeof(info))) {
        throw backing_error(directory, "cannot verify reserved disk space", int(GetLastError()));
    }
    if (info.AllocationSize.QuadPart < LONGLONG(bytes)) {
        throw std::runtime_error("SSD-backed KV cache: backing file is not fully allocated; "
            "use an uncompressed, non-sparse local SSD directory");
    }
    pimpl->mapping = CreateFileMappingW(pimpl->file, nullptr, PAGE_READWRITE, 0, 0, nullptr);
    if (!pimpl->mapping) {
        throw backing_error(directory, "cannot create the shared file mapping", int(GetLastError()));
    }
    pimpl->address = MapViewOfFile(pimpl->mapping, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, bytes);
    if (!pimpl->address) {
        throw backing_error(directory, "cannot map the backing file", int(GetLastError()));
    }
#else
    if (bytes > size_t(std::numeric_limits<off_t>::max())) {
        throw std::invalid_argument("SSD-backed KV cache: file size exceeds the platform limit");
    }
    auto name = (directory / "inferssd-kv-XXXXXX").string();
    std::vector<char> path(name.begin(), name.end());
    path.push_back('\0');
    pimpl->file = mkstemp(path.data()); // private mode0600, exclusive creation
    if (pimpl->file < 0) { throw backing_error(directory, "cannot create a private backing file", errno); }
    if (unlink(path.data()) != 0) {
        const int error = errno;
        throw backing_error(directory, "cannot unlink the temporary backing file", error);
    }
    if (fcntl(pimpl->file, F_SETFD, FD_CLOEXEC) == -1) {
        throw backing_error(directory, "cannot protect the backing descriptor", errno);
    }
#ifdef __APPLE__
    fstore_t store{};
    store.fst_flags = F_ALLOCATECONTIG | F_ALLOCATEALL;
    store.fst_posmode = F_PEOFPOSMODE;
    store.fst_length = off_t(bytes);
    if (fcntl(pimpl->file, F_PREALLOCATE, &store) == -1) {
        store.fst_flags = F_ALLOCATEALL;
        if (fcntl(pimpl->file, F_PREALLOCATE, &store) == -1) {
            throw backing_error(directory, "cannot reserve disk space", errno);
        }
    }
    if (store.fst_bytesalloc < off_t(bytes)) {
        throw std::runtime_error("SSD-backed KV cache: filesystem did not reserve the entire backing file");
    }
#else
    const int allocated = posix_fallocate(pimpl->file, 0, off_t(bytes));
    if (allocated) { throw backing_error(directory, "cannot reserve disk space", allocated); }
#endif
    // Extending a fresh POSIX file supplies zero bytes, including allocated but unwritten extents.
    if (ftruncate(pimpl->file, off_t(bytes)) != 0) {
        throw backing_error(directory, "cannot set the backing file size", errno);
    }
    void * address = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, pimpl->file, 0);
    if (address == MAP_FAILED) { throw backing_error(directory, "cannot map the backing file", errno); }
    pimpl->address = address;
#endif
}

kv_backing::~kv_backing() = default;
void * kv_backing::data() const noexcept { return pimpl->address; }
size_t kv_backing::size() const noexcept { return pimpl->bytes; }

} // namespace ssd
