#include "llama-mmap.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct temporary_file {
    char path[64] = "/tmp/llama-mmap-dontneed-XXXXXX";
    int fd = -1;

    temporary_file() {
        fd = mkstemp(path);
        require(fd >= 0, "mkstemp failed");
    }

    ~temporary_file() {
        if (fd >= 0) {
            close(fd);
        }
        unlink(path);
    }
};

int main() {
    try {
        const long page_size_long = sysconf(_SC_PAGESIZE);
        require(page_size_long > 0, "sysconf(_SC_PAGESIZE) failed");
        const size_t page = size_t(page_size_long);
        std::vector<uint8_t> expected(5 * page);
        for (size_t i = 0; i < expected.size(); ++i) {
            expected[i] = uint8_t(i % 251);
        }
        temporary_file fixture;
        require(write(fixture.fd, expected.data(), expected.size()) == ssize_t(expected.size()), "fixture write failed");
        llama_file file(fixture.path, "rb");
        llama_mmap mapping(&file, 0, false);
        const auto * base = static_cast<const uint8_t *>(mapping.addr());
        require(std::memcmp(base, expected.data(), expected.size()) == 0, "initial mapped data differs");

        size_t advised = 0;
        require(mapping.dontneed_fragment(page + 17, 4 * page - 19, &advised), "MADV_DONTNEED failed");
        require(advised == page, "advice included a partial boundary page");
        require(mapping.addr() == base, "advice changed the mapping address");
        require(std::memcmp(base, expected.data(), expected.size()) == 0, "advice changed mapped data or boundary pages");
        advised = 1;
        require(mapping.dontneed_fragment(17, page - 19, &advised), "empty whole-page range failed");
        require(advised == 0, "partial-page range reported advice");

        // Darwin must reject an unmapped range. This distinguishes a real
        // syscall from the former Apple implementation that returned true.
        mapping.unmap_fragment(2 * page, 3 * page);
        errno = 0;
        require(!mapping.dontneed_fragment(2 * page, 3 * page), "unmapped advice falsely reported success");
        require(errno == EINVAL, "unmapped advice did not report EINVAL");
        require(std::memcmp(base, expected.data(), 2 * page) == 0, "prefix mapping changed");
        require(std::memcmp(base + 3 * page, expected.data() + 3 * page, 2 * page) == 0, "suffix mapping changed");
        std::puts("PASS mmap dontneed: mapped bytes, boundary pages, address and syscall errors");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-llama-mmap-dontneed: %s\n", error.what());
        return 1;
    }
    return 0;
}
