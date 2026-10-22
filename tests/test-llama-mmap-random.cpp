#include "llama-mmap.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include <fcntl.h>
#include <unistd.h>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-llama-mmap-random: %s\n", message);
        std::exit(1);
    }
}

static std::string vm_flags_for(const void * ptr) {
    const uintptr_t target = (uintptr_t) ptr;
    std::ifstream smaps("/proc/self/smaps");
    std::string line;
    bool in_target_vma = false;

    while (std::getline(smaps, line)) {
        uintptr_t first = 0;
        uintptr_t last = 0;
        if (std::sscanf(line.c_str(), "%" SCNxPTR "-%" SCNxPTR, &first, &last) == 2) {
            in_target_vma = first <= target && target < last;
            continue;
        }
        if (in_target_vma && line.rfind("VmFlags:", 0) == 0) {
            return line;
        }
    }
    return {};
}

static bool has_flag(const std::string & flags, const std::string & wanted) {
    std::istringstream words(flags);
    std::string word;
    while (words >> word) {
        if (word == wanted) {
            return true;
        }
    }
    return false;
}

int main() {
    const long page_size_long = sysconf(_SC_PAGESIZE);
    require(page_size_long > 0, "sysconf(_SC_PAGESIZE) failed");
    const size_t page_size = (size_t) page_size_long;

    char path[] = "/tmp/llama-mmap-random-XXXXXX";
    const int fd = mkstemp(path);
    require(fd >= 0, "mkstemp failed");
    require(ftruncate(fd, (off_t) (4 * page_size)) == 0, "ftruncate failed");
    close(fd);

    {
        llama_file file(path, "rb");
        llama_mmap mapping(&file, 0, false);
        unlink(path);

        const auto * base = (const uint8_t *) mapping.addr();
        require(!has_flag(vm_flags_for(base + page_size + 17), "rr"), "mapping unexpectedly starts with rr");

        size_t advised = 0;
        require(mapping.advise_random_fragment(page_size + 17, 2 * page_size + 33, &advised),
                "POSIX_MADV_RANDOM failed");
        require(advised == 2 * page_size, "advice did not expand to the two touched pages");
        require(has_flag(vm_flags_for(base + page_size + 17), "rr"), "first touched page is missing rr");
        require(has_flag(vm_flags_for(base + 2 * page_size + 17), "rr"), "second touched page is missing rr");
        require(!has_flag(vm_flags_for(base + 17), "rr"), "page before the requested range acquired rr");
        require(!has_flag(vm_flags_for(base + 3 * page_size + 17), "rr"),
                "page after the requested range acquired rr");

        advised = 1;
        require(!mapping.advise_random_fragment(0, mapping.size() + 1, &advised),
                "out-of-range advice unexpectedly succeeded");
        require(advised == 0, "failed advice reported nonzero bytes");
    }

    return 0;
}
