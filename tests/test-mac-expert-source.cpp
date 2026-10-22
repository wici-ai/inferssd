#include "mac-expert-pack-cache.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <unistd.h>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 5 || !ggml_mac_expert_pack_enabled()) { return 2; }
    bool cache_updates = true;
    for (int begin = 1; begin < argc;) {
        if (std::strcmp(argv[begin], "--prefill") == 0 || std::strcmp(argv[begin], "--decode") == 0) {
            cache_updates = std::strcmp(argv[begin++], "--decode") == 0;
            continue;
        }
        if (std::strcmp(argv[begin], "--truncate") == 0) {
            if (begin + 1 >= argc || truncate(argv[begin + 1], 0) != 0) { return 6; }
            begin += 2;
            continue;
        }
        if (begin + 3 >= argc) { return 2; }
        const size_t bytes = size_t(std::strtoull(argv[begin + 2], nullptr, 10));
        int end = begin;
        std::vector<int64_t> counts;
        while (end + 3 < argc && std::strcmp(argv[end], argv[begin]) == 0 &&
                std::strcmp(argv[end + 2], argv[begin + 2]) == 0) {
            const int expert = std::atoi(argv[end + 1]);
            if (expert < 0 || expert > 4096 || (size_t(expert) < counts.size() && counts[expert])) { return 7; }
            counts.resize(size_t(expert) + 1, 0);
            counts[expert] = 1;
            end += 4;
        }
        void * batch = ggml_mac_expert_pack_begin_batch(argv[begin], counts.data(), int(counts.size()), bytes, cache_updates);
        for (int i = begin; i < end; i += 4) {
            const auto expert = int32_t(std::strtol(argv[i + 1], nullptr, 10));
            std::ifstream file(argv[i + 3], std::ios::binary);
            std::vector<char> expected((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            if (expected.size() != bytes) { return 3; }
            const auto * data = static_cast<const char *>(ggml_mac_expert_pack_acquire_batch(batch, argv[i], expert, bytes));
            const void * pinned = cache_updates ? ggml_mac_expert_pack_acquire(argv[i], expert, bytes) : data;
            const char * alignment = std::getenv("SSD_LLAMA_MAC_EXPERT_READ_ALIGNMENT");
            const bool page_aligned_payload = !alignment || std::strcmp(alignment, "1") == 0 || std::getenv("SSD_LLAMA_MAC_EXPERT_PACK");
            if (data != pinned || (page_aligned_payload && uintptr_t(data) % 4096 != 0) ||
                    !std::equal(expected.begin(), expected.end(), data)) { return 4; }
            if (cache_updates) { ggml_mac_expert_pack_release(pinned); }
            if (!std::equal(expected.begin(), expected.end(), data)) { return 5; }
            ggml_mac_expert_pack_release(data);
        }
        ggml_mac_expert_pack_end_batch(batch);
        begin = end;
    }
    std::puts("expert source byte, pipeline and pin checks passed");
}
