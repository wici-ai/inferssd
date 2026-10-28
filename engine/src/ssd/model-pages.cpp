// SPDX-License-Identifier: Apache-2.0
#include "model-pages.h"
#include <algorithm>

namespace ssd {
void drop_expert_pages(const std::vector<std::unique_ptr<llama_mmap>> & mappings,
        const std::vector<std::vector<llama_file_range>> & expert_ranges) {
    const int64_t start = ggml_time_us();
    size_t bytes = 0, ranges = 0, failures = 0;
    const size_t count = std::min(mappings.size(), expert_ranges.size());
    for (size_t i = 0; i < count; ++i) {
        if (!mappings[i]) { continue; }
        for (const auto & range : expert_ranges[i]) {
            size_t advised = 0;
            const bool ok = mappings[i]->dontneed_fragment(range.first, range.last, &advised);
            bytes += advised;
            ++ranges;
            if (!ok) { ++failures; }
        }
    }
    LLAMA_LOG_INFO("SSD expert mmap advice: bytes=%.2f MiB ranges=%zu failures=%zu wall_ms=%.3f\n",
        bytes / 1024.0 / 1024.0, ranges, failures, double(ggml_time_us() - start) / 1000.0);
}
}
