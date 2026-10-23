#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ssd { namespace io {

// Owns one regular expert-source file. macOS disables the filesystem data cache;
// other POSIX systems retain their ordinary pread behavior.
struct source_file {
    std::string path;
    int fd;
    uint64_t bytes;
    explicit source_file(const char * source_path);
    ~source_file();
    source_file(const source_file &) = delete;
    source_file & operator=(const source_file &) = delete;
};

// Reads exactly the requested range, retrying interrupted calls. On error this
// preserves the reader's existing exception contract; the source C API owns the
// fatal-error boundary. The caller retains the destination until completion.
void pread_exact(int fd, void * data, size_t bytes, uint64_t offset);

} } // namespace ssd::io
