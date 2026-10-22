#include "mac-expert-pack-cache.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>
#include <cstring>
#include <unistd.h>

int main(int argc, char ** argv) {
    if (argc < 5 || !ggml_mac_expert_pack_enabled()) { return 2; }
    for (int i = 1; i < argc; i += 4) {
        if (std::strcmp(argv[i], "--truncate") == 0) {
            if (i + 1 >= argc || truncate(argv[i + 1], 0) != 0) { return 6; }
            i -= 2;
            continue;
        }
        if (i + 3 >= argc) { return 2; }
        const auto expert = int32_t(std::strtol(argv[i + 1], nullptr, 10));
        const auto bytes = size_t(std::strtoull(argv[i + 2], nullptr, 10));
        std::ifstream file(argv[i + 3], std::ios::binary);
        std::vector<char> expected((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (expected.size() != bytes) { return 3; }
        const auto * data = static_cast<const char *>(ggml_mac_expert_pack_acquire(argv[i], expert, bytes));
        const void * pinned = ggml_mac_expert_pack_acquire(argv[i], expert, bytes);
        if (data != pinned || uintptr_t(data) % 4096 != 0 ||
                !std::equal(expected.begin(), expected.end(), data)) { return 4; }
        ggml_mac_expert_pack_release(pinned);
        if (!std::equal(expected.begin(), expected.end(), data)) { return 5; }
        ggml_mac_expert_pack_release(data);
    }
    std::puts("expert source byte and pin checks passed");
}
