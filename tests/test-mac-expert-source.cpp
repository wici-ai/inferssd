#include "ssd/mac-expert-source.h"
#include "ssd/mac-runtime-fixture.h"
#include <limits>
#include <memory>
#include <stdexcept>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <unistd.h>
#include <vector>

int main(int argc, char ** argv) {
    ssd::mac::testing::options options;
    std::vector<std::string> sources;
    int first = 1;
    try {
        const auto integer = [](const char * text) -> uint64_t {
            if (!*text || std::strspn(text, "0123456789") != std::strlen(text)) {
                throw std::runtime_error("fixture CACHE_MIB must be a nonnegative MiB value fitting size_t");
            }
            size_t used = 0; const auto value = std::stoull(text, &used);
            if (used != std::strlen(text)) { throw std::runtime_error("invalid fixture integer"); }
            return value;
        };
        while (first < argc && std::strcmp(argv[first], "--") != 0) {
            if (first + 1 >= argc) { return 2; }
            const std::string key = argv[first++]; const char * value = argv[first++];
            if (key == "--source") { sources.emplace_back(value); }
            else if (key == "--cache-mib") {
                const auto mib = integer(value);
                if (mib > (std::numeric_limits<size_t>::max() >> 20)) {
                    throw std::runtime_error("fixture CACHE_MIB must be a nonnegative MiB value fitting size_t");
                }
                options.budget_bytes = size_t(mib) << 20;
            } else if (key == "--depth") { options.read_ahead = size_t(integer(value)); }
              else if (key == "--alignment") { options.read_alignment = size_t(integer(value)); }
              else if (key == "--trace") { options.trace_path = value; }
              else { return 2; }
        }
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 2; }
    if (first == argc || argc - ++first < 4) { return 2; }
    char error[512] {};
    std::unique_ptr<ggml_ssd_mac_cache, decltype(&ssd::mac::destroy)> owner(
            ssd::mac::testing::create(sources, options, error, sizeof(error)), ssd::mac::destroy);
    if (!owner) { std::fprintf(stderr, "%s\n", error); return 2; }
    void * cache = owner.get();
    bool cache_updates = true;
    for (int begin = first; begin < argc;) {
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
        void * batch = ggml_mac_expert_pack_begin_batch(cache, argv[begin], counts.data(), int(counts.size()), bytes, cache_updates);
        for (int i = begin; i < end; i += 4) {
            const auto expert = int32_t(std::strtol(argv[i + 1], nullptr, 10));
            std::ifstream file(argv[i + 3], std::ios::binary);
            std::vector<char> expected(bytes);
            if (!file.read(expected.data(), static_cast<std::streamsize>(bytes)) || file.peek() != EOF) { return 3; }
            const auto * data = static_cast<const char *>(ggml_mac_expert_pack_acquire_batch(cache, batch, argv[i], expert, bytes));
            const void * pinned = cache_updates ? ggml_mac_expert_pack_acquire(cache, argv[i], expert, bytes) : data;
            const bool page_aligned_payload = options.read_alignment == 1 || sources.front().find(".pack") != std::string::npos;
            if (data != pinned || (page_aligned_payload && uintptr_t(data) % 4096 != 0) ||
                    !std::equal(expected.begin(), expected.end(), data)) { return 4; }
            if (cache_updates) { ggml_mac_expert_pack_release(cache, pinned); }
            if (!std::equal(expected.begin(), expected.end(), data)) { return 5; }
            ggml_mac_expert_pack_release(cache, data);
        }
        ggml_mac_expert_pack_end_batch(cache, batch);
        begin = end;
    }
    std::puts("expert source byte, pipeline and pin checks passed");
}
