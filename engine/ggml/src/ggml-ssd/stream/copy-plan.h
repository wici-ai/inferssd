#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace ggml_moe_pack_source {

struct copy_plan {
    std::vector<int32_t> range_first;
    std::vector<int32_t> range_last;
    uint64_t expert_count = 0;
    size_t bytes = 0;
};

template <typename IsSelected>
bool build_copy_plan(
        int64_t n_expert,
        size_t expert_bytes,
        size_t padding,
        IsSelected is_selected,
        copy_plan & result) {
    if (n_expert <= 0 || n_expert > std::numeric_limits<int32_t>::max() ||
            expert_bytes == 0 || padding > expert_bytes) {
        return false;
    }

    copy_plan plan;
    auto append = [&](int32_t first, int32_t last) {
        const size_t count = (size_t) (last - first) + 1;
        if (count > (std::numeric_limits<size_t>::max() - plan.bytes) / expert_bytes) {
            return false;
        }
        plan.bytes += count * expert_bytes;
        if (last < n_expert - 1) {
            if (padding > std::numeric_limits<size_t>::max() - plan.bytes) {
                return false;
            }
            plan.bytes += padding;
        }
        plan.expert_count += count;
        plan.range_first.push_back(first);
        plan.range_last.push_back(last);
        return true;
    };

    int32_t id = 0;
    while (id < n_expert && !is_selected(id)) {
        ++id;
    }
    if (id < n_expert) {
        int32_t first = id;
        int32_t last = id;
        for (++id; id < n_expert; ++id) {
            if (!is_selected(id)) {
                continue;
            }
            if (id == last + 1) {
                last = id;
                continue;
            }
            if (!append(first, last)) {
                return false;
            }
            first = id;
            last = id;
        }
        if (!append(first, last)) {
            return false;
        }
    }

    result = std::move(plan);
    return true;
}

} // namespace ggml_moe_pack_source
