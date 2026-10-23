#include "pread-nocache.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace ssd { namespace io {

source_file::source_file(const char * source_path) : path(source_path), fd(open(source_path, O_RDONLY | O_NONBLOCK)) {
        if (fd < 0) { throw std::runtime_error("cannot open expert source: " + std::string(std::strerror(errno))); }
        struct stat info {};
        const bool stat_failed = fstat(fd, &info) != 0;
        if (!stat_failed && !S_ISREG(info.st_mode)) {
            close(fd);
            throw std::runtime_error("expert source must be a regular file");
        }
        if (stat_failed || info.st_size < 0
#if defined(__APPLE__)
                || fcntl(fd, F_NOCACHE, 1) == -1
#endif
                ) {
            const int cause = errno;
            close(fd);
            // Close before constructing an error string: this constructor's
            // destructor will not run if allocating the message fails.
            throw std::runtime_error("cannot prepare expert source: " + std::string(std::strerror(cause)));
        }
        bytes = uint64_t(info.st_size);
    }

source_file::~source_file() { close(fd); }

void pread_exact(int fd, void * data, size_t bytes, uint64_t offset) {
    auto * target = static_cast<char *>(data);
    while (bytes) {
        const size_t chunk = std::min(bytes, size_t(64) << 20);
        const ssize_t got = pread(fd, target, chunk, static_cast<off_t>(offset));
        if (got < 0 && errno == EINTR) { continue; }
        if (got <= 0) { throw std::runtime_error("short/failed expert-pack read: " + std::string(std::strerror(errno))); }
        target += got;
        offset += uint64_t(got);
        bytes -= size_t(got);
    }
}

} } // namespace ssd::io
