// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ssd { namespace platform {

// Read-only regular files behind int descriptors, so the C ABIs that carry an
// fd keep their shape on every OS. On Windows an fd is a CRT descriptor
// wrapping a CreateFileW HANDLE; native_handle() recovers it. Paths are UTF-8.
//
// The opens return -1 on failure, set errno and, when error is given, describe
// the failure there. errno is EINVAL when the path is not a regular file (and,
// for open_direct on Linux, when the filesystem refuses O_DIRECT).

// Bypasses the OS page cache: O_DIRECT on Linux, F_NOCACHE on macOS, and
// FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED on Windows. Linux and Windows
// then need offsets, lengths and buffer addresses that are multiples of
// direct_alignment(); macOS has no such contract.
int open_direct(const char * utf8_path, std::string * error = nullptr);
// Ordinary cached reads, for metadata. The Windows handle is synchronous, so
// the descriptor also works with the CRT and stdio.
int open_buffered(const char * utf8_path, std::string * error = nullptr);
// Closes a descriptor from either open. Direct descriptors must be closed here:
// macOS and Windows track them for is_direct().
void close_fd(int fd);

bool file_size(int fd, uint64_t & bytes);
// True for a descriptor that bypasses the page cache.
bool is_direct(int fd);
// The offset, length and address granularity of direct reads on fd.
size_t direct_alignment(int fd);

// Names one file version: same identity means same file and same contents as
// far as the OS reports. POSIX keeps st_ino in id_lo; Windows uses the volume
// serial number and the 128-bit file id. Times are ns since the Unix epoch.
struct file_identity {
    uint64_t device = 0, id_lo = 0, id_hi = 0, size = 0;
    int64_t  mtime_ns = 0, ctime_ns = 0;
};
bool identity(int fd, file_identity & out);
inline bool operator==(const file_identity & a, const file_identity & b) {
    return a.device == b.device && a.id_lo == b.id_lo && a.id_hi == b.id_hi && a.size == b.size &&
           a.mtime_ns == b.mtime_ns && a.ctime_ns == b.ctime_ns;
}
inline bool operator!=(const file_identity & a, const file_identity & b) { return !(a == b); }

// One positional read; safe to call concurrently on one descriptor. Returns the
// bytes read (0 at end of file) or a negative errno code.
int64_t pread_some(int fd, void * data, size_t bytes, uint64_t offset);
// Reads exactly bytes. On failure *error is the errno code, or 0 when the
// file ended first.
bool pread_exact(int fd, void * data, size_t bytes, uint64_t offset, int * error = nullptr);

#if defined(_WIN32)
// The HANDLE behind fd, or INVALID_HANDLE_VALUE; never raises the CRT's
// invalid-parameter handler for a bad descriptor.
void * native_handle(int fd);
// The errno code for a Win32 error, so every OS reports reads the same way.
int errno_from_win32(unsigned long error);
#endif

} } // namespace ssd::platform
