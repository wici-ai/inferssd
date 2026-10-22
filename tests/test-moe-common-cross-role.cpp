#include "../ggml/src/ggml-cuda/moe-common-cross-role.h"

#include <array>
#include <cstdio>
#include <limits>

#undef NDEBUG
#include <cassert>

using ggml_moe_common::cross_role_key;
using ggml_moe_common::cross_role_phase;
using ggml_moe_common::cross_role_window;

static cross_role_key make_key(int expert, bool down = false) {
    return {down ? "blk.7.ffn_down_exps.weight" : "blk.7.ffn_up_exps.weight", 4,
            static_cast<uint64_t>(expert) * 12288 + (down ? 4096 : 0), down ? 8192U : 4096U, expert};
}

static void finish_slot(cross_role_window & window, size_t slot) {
    assert(window.complete(slot, window.generation, window.keys[slot].bytes));
    assert(window.copied(slot));
    assert(window.release(slot));
}

static void test_full_key_identity() {
    const cross_role_key base = make_key(17);
    std::vector<cross_role_key> keys(6, base);
    keys[1].tensor += ".alias";
    ++keys[2].source;
    keys[3].offset += 4096;
    keys[4].bytes += 4096;
    ++keys[5].expert;
    cross_role_window window;
    assert(window.begin(keys));
    for (size_t i = 0; i < keys.size(); ++i) {
        assert(window.find(keys[i]) == i);
        assert((keys[i] == base) == (i == 0));
    }
    assert(window.find(make_key(17, true)) == SIZE_MAX);
    assert(window.reading() == keys.size());
}

static void test_completion_and_lifetime() {
    const cross_role_key key = make_key(5);
    cross_role_window window;
    assert(window.idle() && window.reading() == 0 && window.find(key) == SIZE_MAX);
    assert(!window.complete(0, 0, key.bytes) && !window.copied(0) && !window.release(0));
    assert(window.begin({key}));
    const uint64_t first_generation = window.generation;
    assert(first_generation == 1 && !window.idle() && window.reading() == 1);
    assert(!window.begin({key}));
    assert(!window.complete(SIZE_MAX, first_generation, key.bytes));
    assert(!window.complete(0, first_generation - 1, key.bytes));
    assert(!window.complete(0, first_generation + 1, key.bytes));
    assert(!window.complete(0, first_generation, key.bytes - 1));
    assert(!window.complete(0, first_generation, key.bytes + 1));
    assert(!window.complete(0, first_generation, 0));
    assert(window.reading() == 1 && !window.copied(0) && !window.release(0));
    assert(window.complete(0, first_generation, key.bytes));
    assert(window.reading() == 0 && window.phases[0] == cross_role_phase::ready);
    assert(!window.complete(0, first_generation, key.bytes));
    assert(!window.release(0) && !window.begin({key}));
    assert(window.copied(0));
    assert(window.phases[0] == cross_role_phase::h2d_pending);
    assert(!window.copied(0) && !window.complete(0, first_generation, key.bytes));
    // A delayed H2D reader still prevents the next generation from beginning.
    assert(!window.idle() && !window.begin({make_key(19)}));
    assert(window.release(0));
    assert(!window.release(0) && window.idle());
    assert(window.begin({key}));
    assert(window.generation == first_generation + 1);
    assert(!window.complete(0, first_generation, key.bytes) && window.reading() == 1);
    assert(!window.copied(SIZE_MAX) && !window.release(SIZE_MAX));
    finish_slot(window, 0);
    assert(window.idle());
}

static void test_invalid_begin() {
    const cross_role_key key = make_key(3);
    cross_role_window window;
    assert(!window.begin({}) && !window.begin({key, key}));
    assert(window.generation == 0 && window.keys.empty() && window.phases.empty() && window.idle());
    assert(window.begin({key, make_key(3, true)}));
    finish_slot(window, 0);
    finish_slot(window, 1);
    const auto keys = window.keys;
    const auto phases = window.phases;
    const uint64_t generation = window.generation;
    assert(!window.begin({}) && !window.begin({key, key}));
    assert(window.keys == keys && window.phases == phases && window.generation == generation);
    assert(window.begin({make_key(9)}) && window.generation == generation + 1);
    assert(window.find(key) == SIZE_MAX && window.keys.size() == 1);
}

static void test_generation_overflow() {
    cross_role_window window;
    window.generation = std::numeric_limits<uint64_t>::max() - 1;
    assert(window.begin({make_key(1)}));
    assert(window.generation == std::numeric_limits<uint64_t>::max());
    finish_slot(window, 0);
    const auto keys = window.keys;
    assert(!window.begin({make_key(2)}));
    assert(window.generation == std::numeric_limits<uint64_t>::max());
    assert(window.keys == keys && window.idle());
}

static void test_two_roles_out_of_order() {
    std::vector<cross_role_key> keys;
    for (int i = 0; i < 16; ++i) {
        const int expert = (i * 7) % 32;
        keys.push_back(make_key(expert));
        keys.push_back(make_key(expert, true));
    }
    cross_role_window window;
    assert(window.begin(keys) && window.reading() == 32);
    std::array<size_t, 2> read_bytes{};
    std::array<size_t, 2> copy_bytes{};
    for (size_t i = keys.size(); i > 0; --i) {
        const size_t slot = i - 1;
        assert(window.complete(slot, window.generation, keys[slot].bytes));
        assert(!window.complete(slot, window.generation, keys[slot].bytes));
        read_bytes[slot % 2] += keys[slot].bytes;
        assert(window.reading() == slot);
    }
    assert(!window.idle() && window.reading() == 0);
    for (size_t role = 0; role < 2; ++role) {
        for (size_t i = role; i < keys.size(); i += 2) {
            const size_t slot = window.find(keys[i]);
            assert(slot == i && window.copied(slot));
            copy_bytes[role] += keys[slot].bytes;
            // Up H2D finishes first; down still owns its staging buffers.
            if (role == 0) assert(window.release(slot));
        }
    }
    assert(read_bytes[0] == 16 * 4096 && read_bytes[1] == 16 * 8192);
    assert(copy_bytes == read_bytes);
    assert(!window.idle() && !window.begin(keys));
    for (size_t i = 1; i < keys.size(); i += 2) {
        assert(window.phases[i] == cross_role_phase::h2d_pending && window.release(i));
    }
    assert(window.idle() && window.begin(keys) && window.generation == 2);
    assert(window.reading() == keys.size());
    assert(!window.complete(31, 1, keys[31].bytes));
}

int main() {
    test_full_key_identity();
    test_completion_and_lifetime();
    test_invalid_begin();
    test_generation_overflow();
    test_two_roles_out_of_order();
    std::puts("common cross-role state: 5 CPU tests passed");
    return 0;
}
