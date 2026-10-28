#include "pread-nocache.h"

#include "../platform/file.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace ssd { namespace io {

source_file::source_file(const char * source_path) : path(source_path), fd(-1), bytes(0) {
        std::string error;
        // F_NOCACHE has no alignment contract, so macOS reads uncached; the
        // direct opens elsewhere need aligned reads these callers do not make.
#if defined(__APPLE__)
        fd = platform::open_direct(source_path, &error);
#else
        fd = platform::open_buffered(source_path, &error);
#endif
        if (fd < 0) {
            if (errno == EINVAL) { throw std::runtime_error("expert source must be a regular file"); }
            throw std::runtime_error("cannot open expert source: " + error);
        }
        if (!platform::file_size(fd, bytes)) {
            const int cause = errno;
            platform::close_fd(fd);
            // Close before constructing an error string: this constructor's
            // destructor will not run if allocating the message fails.
            throw std::runtime_error("cannot prepare expert source: " + std::string(std::strerror(cause)));
        }
    }

source_file::~source_file() { platform::close_fd(fd); }

void pread_exact(int fd, void * data, size_t bytes, uint64_t offset) {
    int error = 0;
    if (!platform::pread_exact(fd, data, bytes, offset, &error)) {
        throw std::runtime_error("short/failed expert-pack read: " +
                                 std::string(error ? std::strerror(error) : "unexpected end of file"));
    }
}

} } // namespace ssd::io
