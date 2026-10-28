// SPDX-License-Identifier: Apache-2.0
#include "file.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <new>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winternl.h>
#include <fcntl.h>
#include <io.h>
#include <stdlib.h>
// ntdll export, declared by the driver kit; the SDK's winternl.h has its types.
extern "C" NTSYSAPI NTSTATUS NTAPI NtQueryInformationFile(HANDLE, PIO_STATUS_BLOCK, PVOID, ULONG,
                                                           FILE_INFORMATION_CLASS);
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ssd { namespace platform {
namespace {

void describe(std::string * error, const char * path, const char * what, int code) noexcept {
    if (!error) { return; }
    try {
        *error = std::string(path ? path : "(null)") + ": " + what;
        if (code) { *error += std::string(": ") + std::strerror(code); }
    } catch (...) {
        error->clear();
    }
}

#if defined(_WIN32)
void ignore_invalid_parameter(const wchar_t *, const wchar_t *, const wchar_t *, unsigned int, uintptr_t) {}

// The CRT treats an unknown descriptor as a fatal invalid parameter; here a bad
// fd is an ordinary error to report.
struct quiet_crt {
    _invalid_parameter_handler previous;
    quiet_crt() : previous(_set_thread_local_invalid_parameter_handler(ignore_invalid_parameter)) {}
    ~quiet_crt() { _set_thread_local_invalid_parameter_handler(previous); }
    quiet_crt(const quiet_crt &) = delete;
    quiet_crt & operator=(const quiet_crt &) = delete;
};

HANDLE handle_of(int fd) noexcept {
    if (fd < 0) { return INVALID_HANDLE_VALUE; }
    quiet_crt quiet;
    return reinterpret_cast<HANDLE>(_get_osfhandle(fd));
}

bool widen(const char * utf8, std::wstring & out) {
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, nullptr, 0);
    if (n <= 0) { return false; }
    out.resize(size_t(n));
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, &out[0], n) != n) { return false; }
    out.resize(size_t(n) - 1);
    return true;
}

int64_t unix_ns(LARGE_INTEGER filetime) {
    // FILETIME counts 100 ns ticks from 1601-01-01.
    return (filetime.QuadPart - 116444736000000000LL) * 100;
}

// An unbuffered handle refuses an unaligned offset even at the end of the
// file, which is where a short final read leaves the next one; POSIX reports
// the end of file there, so this does too.
int64_t read_result(HANDLE handle, DWORD error, uint64_t offset) {
    if (error == ERROR_HANDLE_EOF) { return 0; }
    const int code = errno_from_win32(error);
    LARGE_INTEGER size {};
    if (code == EINVAL && GetFileSizeEx(handle, &size) && offset >= uint64_t(size.QuadPart)) { return 0; }
    return -int64_t(code);
}

// Overlapped handles need an event per wait, so a concurrent read on the same
// handle cannot satisfy it; one per thread avoids creating one per read.
HANDLE thread_read_event() {
    struct holder {
        HANDLE event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        ~holder() { if (event) { CloseHandle(event); } }
    };
    thread_local holder slot;
    return slot.event;
}

int open_windows(const char * path, DWORD flags, std::string * error) noexcept {
    try {
        std::wstring wide;
        if (!path || !widen(path, wide)) {
            errno = EINVAL;
            describe(error, path, "path is not valid UTF-8", 0);
            return -1;
        }
        // Sharing delete lets a model file be replaced while it is being read;
        // no sharing write keeps it from changing under the reader.
        const HANDLE handle = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                                          OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | flags, nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            const int code = errno_from_win32(GetLastError());
            describe(error, path, "CreateFileW", code);
            errno = code;
            return -1;
        }
        if (GetFileType(handle) != FILE_TYPE_DISK) {
            CloseHandle(handle);
            describe(error, path, "not a regular file", 0);
            errno = EINVAL;
            return -1;
        }
        const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(handle), _O_RDONLY | _O_BINARY);
        if (fd < 0) {
            const int code = errno;
            CloseHandle(handle);
            describe(error, path, "_open_osfhandle", code);
            errno = code;
            return -1;
        }
        return fd;
    } catch (const std::bad_alloc &) {
        describe(error, path, "out of memory", 0);
        errno = ENOMEM;
        return -1;
    }
}
#else
int open_posix(const char * path, int flags, std::string * error) noexcept {
    // O_NONBLOCK keeps a FIFO path from blocking the open; the fd then has the
    // same flags the store has always used for its sources.
    const int fd = ::open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC | flags);
    if (fd < 0) {
        const int code = errno;
        describe(error, path, "open", code);
        errno = code;
        return -1;
    }
    // Check the opened descriptor, not the path, so a path swap cannot bypass it.
    struct stat info {};
    if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode)) {
        ::close(fd);
        describe(error, path, "not a regular file", 0);
        errno = EINVAL;
        return -1;
    }
    return fd;
}

int64_t stat_ns(const struct timespec & t) { return int64_t(t.tv_sec) * 1000000000 + int64_t(t.tv_nsec); }
#endif

} // namespace

#if defined(_WIN32)
int errno_from_win32(unsigned long error) {
    switch (error) {
        case ERROR_FILE_NOT_FOUND:
        case ERROR_PATH_NOT_FOUND:
        case ERROR_INVALID_DRIVE:
        case ERROR_BAD_NETPATH:
        case ERROR_BAD_PATHNAME:
        case ERROR_INVALID_NAME:          return ENOENT;
        case ERROR_ACCESS_DENIED:
        case ERROR_SHARING_VIOLATION:
        case ERROR_LOCK_VIOLATION:        return EACCES;
        case ERROR_INVALID_HANDLE:        return EBADF;
        case ERROR_NOT_ENOUGH_MEMORY:
        case ERROR_OUTOFMEMORY:
        case ERROR_NO_SYSTEM_RESOURCES:
        case ERROR_WORKING_SET_QUOTA:
        case ERROR_NOT_ENOUGH_QUOTA:      return ENOMEM;
#ifdef ERROR_OFFSET_ALIGNMENT_VIOLATION
        case ERROR_OFFSET_ALIGNMENT_VIOLATION:
#endif
        case ERROR_INVALID_PARAMETER:
        case ERROR_INVALID_USER_BUFFER:   return EINVAL;
        case ERROR_NOACCESS:              return EFAULT;
        case ERROR_OPERATION_ABORTED:     return ECANCELED;
        case ERROR_TOO_MANY_OPEN_FILES:   return EMFILE;
        case ERROR_FILENAME_EXCED_RANGE:  return ENAMETOOLONG;
        case ERROR_DIRECTORY:             return ENOTDIR;
        default:                          return EIO;
    }
}

void * native_handle(int fd) { return handle_of(fd); }

int open_direct(const char * utf8_path, std::string * error) {
    // Overlapped: the read rings keep many reads in flight on one handle, which
    // a synchronous handle would serialize.
    return open_windows(utf8_path, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, error);
}

int open_buffered(const char * utf8_path, std::string * error) { return open_windows(utf8_path, 0, error); }

void close_fd(int fd) {
    if (fd < 0) { return; }
    quiet_crt quiet;
    _close(fd);
}

bool file_size(int fd, uint64_t & bytes) {
    const HANDLE handle = handle_of(fd);
    LARGE_INTEGER size {};
    if (handle == INVALID_HANDLE_VALUE || !GetFileSizeEx(handle, &size) || size.QuadPart < 0) { return false; }
    bytes = uint64_t(size.QuadPart);
    return true;
}

bool is_direct(int fd) {
    const HANDLE handle = handle_of(fd);
    if (handle == INVALID_HANDLE_VALUE) { return false; }
    // FILE_FLAG_NO_BUFFERING shows as this mode bit (FileModeInformation = 16).
    IO_STATUS_BLOCK status {};
    ULONG mode = 0;
    if (NtQueryInformationFile(handle, &status, &mode, sizeof(mode), FILE_INFORMATION_CLASS(16)) < 0) { return false; }
    return (mode & FILE_NO_INTERMEDIATE_BUFFERING) != 0;
}

size_t direct_alignment(int fd) {
    const HANDLE handle = handle_of(fd);
    FILE_STORAGE_INFO info {};
    if (handle != INVALID_HANDLE_VALUE && GetFileInformationByHandleEx(handle, FileStorageInfo, &info, sizeof(info))) {
        const ULONG sector = std::max(info.LogicalBytesPerSector, info.PhysicalBytesPerSectorForAtomicity);
        if (sector && !(sector & (sector - 1)) && sector <= 65536) { return size_t(sector); }
    }
    return 4096;
}

bool identity(int fd, file_identity & out) {
    const HANDLE handle = handle_of(fd);
    if (handle == INVALID_HANDLE_VALUE) { return false; }
    FILE_BASIC_INFO basic {};
    LARGE_INTEGER size {};
    if (!GetFileInformationByHandleEx(handle, FileBasicInfo, &basic, sizeof(basic)) || !GetFileSizeEx(handle, &size)) {
        return false;
    }
    file_identity value;
    FILE_ID_INFO id {};
    if (GetFileInformationByHandleEx(handle, FileIdInfo, &id, sizeof(id))) {
        value.device = id.VolumeSerialNumber;
        static_assert(sizeof(id.FileId.Identifier) == 16, "FILE_ID_128 is 16 bytes");
        std::memcpy(&value.id_lo, id.FileId.Identifier, 8);
        std::memcpy(&value.id_hi, id.FileId.Identifier + 8, 8);
    } else {
        // Volumes without 128-bit ids (FAT family) still have a unique 64-bit index.
        BY_HANDLE_FILE_INFORMATION info {};
        if (!GetFileInformationByHandle(handle, &info)) { return false; }
        value.device = info.dwVolumeSerialNumber;
        value.id_lo = (uint64_t(info.nFileIndexHigh) << 32) | info.nFileIndexLow;
    }
    value.size = uint64_t(size.QuadPart);
    value.mtime_ns = unix_ns(basic.LastWriteTime);
    value.ctime_ns = unix_ns(basic.ChangeTime);
    out = value;
    return true;
}

int64_t pread_some(int fd, void * data, size_t bytes, uint64_t offset) {
    const HANDLE handle = handle_of(fd);
    if (handle == INVALID_HANDLE_VALUE) { return -EBADF; }
    if (!bytes) { return 0; }
    const HANDLE event = thread_read_event();
    if (!event) { return -ENOMEM; }
    OVERLAPPED overlapped {};
    overlapped.Offset = DWORD(offset);
    overlapped.OffsetHigh = DWORD(offset >> 32);
    overlapped.hEvent = event;
    // 1 GiB per call fits a DWORD and is a multiple of every sector size.
    const DWORD request = DWORD(std::min(bytes, size_t(1) << 30));
    if (!ReadFile(handle, data, request, nullptr, &overlapped)) {
        const DWORD error = GetLastError();
        if (error != ERROR_IO_PENDING) { return read_result(handle, error, offset); }
    }
    DWORD got = 0;
    if (!GetOverlappedResult(handle, &overlapped, &got, TRUE)) { return read_result(handle, GetLastError(), offset); }
    return int64_t(got);
}
#else
int open_direct(const char * utf8_path, std::string * error) {
#if defined(__APPLE__)
    const int fd = open_posix(utf8_path, 0, error);
    if (fd < 0) { return -1; }
    if (fcntl(fd, F_NOCACHE, 1) == -1) {
        const int code = errno;
        ::close(fd);
        describe(error, utf8_path, "F_NOCACHE", code);
        errno = code;
        return -1;
    }
    return fd;
#else
    return open_posix(utf8_path, O_DIRECT, error);
#endif
}

int open_buffered(const char * utf8_path, std::string * error) { return open_posix(utf8_path, 0, error); }

void close_fd(int fd) {
    if (fd >= 0) { ::close(fd); }
}

bool file_size(int fd, uint64_t & bytes) {
    struct stat info {};
    if (fstat(fd, &info) != 0 || info.st_size < 0) { return false; }
    bytes = uint64_t(info.st_size);
    return true;
}

bool is_direct(int fd) {
    if (fd < 0) { return false; }
    const int flags = fcntl(fd, F_GETFL);
#if defined(__APPLE__)
    // F_GETFL reports XNU's FNOCACHE file flag, which F_NOCACHE sets; the SDK
    // keeps its name kernel-only.
    constexpr int fnocache = 0x40000;
    return flags >= 0 && (flags & fnocache);
#else
    return flags >= 0 && (flags & O_DIRECT);
#endif
}

size_t direct_alignment(int) {
#if defined(__APPLE__)
    return 1;    // F_NOCACHE has no alignment contract
#else
    return 4096; // the granularity every O_DIRECT caller already pads to
#endif
}

bool identity(int fd, file_identity & out) {
    struct stat info {};
    if (fstat(fd, &info) != 0) { return false; }
    file_identity value;
    value.device = uint64_t(info.st_dev);
    value.id_lo = uint64_t(info.st_ino);
    value.size = uint64_t(info.st_size);
#if defined(__APPLE__)
    value.mtime_ns = stat_ns(info.st_mtimespec);
    value.ctime_ns = stat_ns(info.st_ctimespec);
#else
    value.mtime_ns = stat_ns(info.st_mtim);
    value.ctime_ns = stat_ns(info.st_ctim);
#endif
    out = value;
    return true;
}

int64_t pread_some(int fd, void * data, size_t bytes, uint64_t offset) {
    if (offset > uint64_t(std::numeric_limits<off_t>::max())) { return -EINVAL; }
    // macOS refuses counts above INT_MAX outright; 1 GiB per call reads short
    // everywhere instead, as on Windows, and callers loop.
    bytes = std::min(bytes, size_t(1) << 30);
    for (;;) {
        const ssize_t got = ::pread(fd, data, bytes, off_t(offset));
        if (got >= 0) { return int64_t(got); }
        if (errno != EINTR) { return -int64_t(errno); }
    }
}
#endif

std::FILE * open_stream(const char * utf8_path, std::string * error) {
    const int fd = open_buffered(utf8_path, error);
    if (fd < 0) { return nullptr; }
#if defined(_WIN32)
    std::FILE * stream = _fdopen(fd, "rb");
#else
    std::FILE * stream = fdopen(fd, "rb");
#endif
    if (!stream) {
        const int code = errno;
        close_fd(fd);
        describe(error, utf8_path, "fdopen", code);
        errno = code;
    }
    return stream;
}

int stream_fd(std::FILE * stream) {
    if (!stream) { return -1; }
#if defined(_WIN32)
    return _fileno(stream);
#else
    return fileno(stream);
#endif
}

bool unmap_view(void * base, size_t bytes) {
    if (!base) { return true; }
#if defined(_WIN32)
    (void) bytes;
    return UnmapViewOfFile(base) != 0;
#else
    return munmap(base, bytes) == 0;
#endif
}

bool pread_exact(int fd, void * data, size_t bytes, uint64_t offset, int * error) {
    auto * target = static_cast<unsigned char *>(data);
    while (bytes) {
        const int64_t got = pread_some(fd, target, bytes, offset);
        if (got <= 0) {
            if (error) { *error = got < 0 ? int(-got) : 0; }
            return false;
        }
        target += got;
        offset += uint64_t(got);
        bytes -= size_t(got);
    }
    if (error) { *error = 0; }
    return true;
}

} } // namespace ssd::platform
