#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace ggml_moe_common {

struct cross_role_key {
    std::string tensor;
    int32_t source = -1;
    uint64_t offset = 0;
    size_t bytes = 0;
    int expert = -1;

    bool operator==(const cross_role_key & other) const {
        return tensor == other.tensor && source == other.source &&
               offset == other.offset && bytes == other.bytes && expert == other.expert;
    }
};

enum class cross_role_phase {
    reading,
    ready,
    h2d_pending,
    released,
};

// The caller serializes access and retains the I/O and H2D buffers. A read
// completion alone never makes a slot available to another generation.
struct cross_role_window {
    uint64_t generation = 0;
    std::vector<cross_role_key> keys;
    std::vector<cross_role_phase> phases;

    bool begin(std::vector<cross_role_key> next_keys) {
        if (!idle() || next_keys.empty() || generation == std::numeric_limits<uint64_t>::max()) {
            return false;
        }
        for (size_t i = 0; i < next_keys.size(); ++i) {
            for (size_t j = 0; j < i; ++j) {
                if (next_keys[i] == next_keys[j]) return false;
            }
        }
        std::vector<cross_role_phase> next_phases(next_keys.size(), cross_role_phase::reading);
        keys = std::move(next_keys);
        phases = std::move(next_phases);
        ++generation;
        return true;
    }

    size_t find(const cross_role_key & key) const {
        for (size_t i = 0; i < keys.size(); ++i) {
            if (keys[i] == key) return i;
        }
        return SIZE_MAX;
    }

    bool complete(size_t slot, uint64_t gen, size_t actual_bytes) {
        if (gen != generation || slot >= keys.size() ||
                phases[slot] != cross_role_phase::reading || actual_bytes != keys[slot].bytes) {
            return false;
        }
        phases[slot] = cross_role_phase::ready;
        return true;
    }

    bool copied(size_t slot) {
        if (slot >= phases.size() || phases[slot] != cross_role_phase::ready) return false;
        phases[slot] = cross_role_phase::h2d_pending;
        return true;
    }

    bool release(size_t slot) {
        if (slot >= phases.size() || phases[slot] != cross_role_phase::h2d_pending) return false;
        phases[slot] = cross_role_phase::released;
        return true;
    }

    bool idle() const {
        return std::all_of(phases.begin(), phases.end(), [](cross_role_phase phase) {
            return phase == cross_role_phase::released;
        });
    }

    size_t reading() const {
        return static_cast<size_t>(std::count(phases.begin(), phases.end(), cross_role_phase::reading));
    }
};

} // namespace ggml_moe_common
