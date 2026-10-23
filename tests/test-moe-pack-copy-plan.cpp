#include "../ggml/src/ggml-ssd/stream/copy-plan.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

#undef NDEBUG
#include <cassert>

using ggml_moe_pack_source::copy_plan;

static copy_plan make_plan(
        int64_t n_expert,
        size_t expert_bytes,
        size_t padding,
        std::initializer_list<int32_t> selected) {
    const std::vector<int32_t> ids(selected);
    copy_plan plan;
    assert(ggml_moe_pack_source::build_copy_plan(
        n_expert, expert_bytes, padding,
        [&](int32_t id) {
            for (int32_t selected_id : ids) {
                if (id == selected_id) {
                    return true;
                }
            }
            return false;
        },
        plan));
    return plan;
}

static void test_empty_and_full() {
    const copy_plan empty = make_plan(8, 1000, 64, {});
    assert(empty.range_first.empty());
    assert(empty.range_last.empty());
    assert(empty.expert_count == 0);
    assert(empty.bytes == 0);

    const copy_plan full = make_plan(8, 1000, 64, {0, 1, 2, 3, 4, 5, 6, 7});
    assert((full.range_first == std::vector<int32_t>{0}));
    assert((full.range_last == std::vector<int32_t>{7}));
    assert(full.expert_count == 8);
    assert(full.bytes == 8000);
}

static void test_grouping_and_padding() {
    const copy_plan grouped = make_plan(8, 1000, 64, {0, 1, 3, 5, 6, 7});
    assert((grouped.range_first == std::vector<int32_t>{0, 3, 5}));
    assert((grouped.range_last == std::vector<int32_t>{1, 3, 7}));
    assert(grouped.expert_count == 6);
    assert(grouped.bytes == 6128);

    const copy_plan middle = make_plan(8, 1000, 64, {2});
    assert((middle.range_first == std::vector<int32_t>{2}));
    assert((middle.range_last == std::vector<int32_t>{2}));
    assert(middle.expert_count == 1);
    assert(middle.bytes == 1064);

    const copy_plan tail = make_plan(8, 1000, 64, {7});
    assert((tail.range_first == std::vector<int32_t>{7}));
    assert((tail.range_last == std::vector<int32_t>{7}));
    assert(tail.expert_count == 1);
    assert(tail.bytes == 1000);
}

static void test_invalid_geometry_and_overflow() {
    copy_plan plan;
    assert(!ggml_moe_pack_source::build_copy_plan(0, 1, 0, [](int32_t) { return false; }, plan));
    assert(!ggml_moe_pack_source::build_copy_plan(-1, 1, 0, [](int32_t) { return false; }, plan));
    assert(!ggml_moe_pack_source::build_copy_plan(
        (int64_t) std::numeric_limits<int32_t>::max() + 1, 1, 0,
        [](int32_t) { return false; }, plan));
    assert(!ggml_moe_pack_source::build_copy_plan(2, 0, 0, [](int32_t) { return true; }, plan));
    assert(!ggml_moe_pack_source::build_copy_plan(2, 8, 9, [](int32_t) { return true; }, plan));

    const size_t maximum = std::numeric_limits<size_t>::max();
    assert(!ggml_moe_pack_source::build_copy_plan(
        2, maximum / 2 + 1, 0, [](int32_t) { return true; }, plan));
    assert(!ggml_moe_pack_source::build_copy_plan(
        2, maximum, maximum, [](int32_t id) { return id == 0; }, plan));
}

int main() {
    test_empty_and_full();
    test_grouping_and_padding();
    test_invalid_geometry_and_overflow();
    return 0;
}
