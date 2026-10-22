#pragma once

#include "moe-common-object-engine.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace ggml_moe_common {

enum class dependency_transfer_status : uint8_t {
    invalid,
    ready,
    planned,
};

struct dependency_role_copy {
    role kind = role::gate;
    uint64_t host_offset = 0;
    size_t bytes = 0;
};

struct dependency_transfer_plan {
    object_key key;
    uint8_t required_role_mask = 0;
    uint8_t resident_role_mask = 0;
    uint8_t missing_role_mask = 0;
    uint8_t critical_role_mask = 0;
    uint8_t deferred_role_mask = 0;
    dependency_role_copy critical[3] = {};
    dependency_role_copy deferred[3] = {};
    uint8_t critical_count = 0;
    uint8_t deferred_count = 0;
    size_t critical_bytes = 0;
    size_t deferred_bytes = 0;
    bool all_required_roles_missing = false;
    bool overlap_available = false;
};

inline bool dependency_transfer_extent_valid(
        const object_key &key,
        const role_extent &extent) {
    return static_cast<uint8_t>(extent.kind) < static_cast<uint8_t>(role::count) &&
        extent.bytes != 0 && extent.relative_offset <= key.bytes &&
        extent.bytes <= key.bytes - extent.relative_offset;
}

inline bool dependency_transfer_extents_overlap(
        const role_extent &left,
        const role_extent &right) {
    const uint64_t left_end = left.relative_offset + left.bytes;
    const uint64_t right_end = right.relative_offset + right.bytes;
    return left.relative_offset < right_end && right.relative_offset < left_end;
}

inline bool dependency_transfer_add_copy(
        dependency_role_copy *copies,
        uint8_t &count,
        size_t &total_bytes,
        const role_extent &extent) {
    if (!copies || count >= 3 ||
            total_bytes > std::numeric_limits<size_t>::max() - extent.bytes) {
        return false;
    }
    copies[count++] = {extent.kind, extent.relative_offset, extent.bytes};
    total_bytes += extent.bytes;
    return true;
}

inline dependency_transfer_status build_dependency_transfer_plan(
        const object_request &request,
        uint8_t resident_role_mask,
        uint8_t critical_role_mask,
        dependency_transfer_plan &plan) {
    plan = dependency_transfer_plan{};
    plan.key = request.key;
    plan.required_role_mask = request.required_role_mask;
    plan.resident_role_mask = resident_role_mask;

    constexpr uint8_t all_role_mask =
        role_bit(role::gate) | role_bit(role::up) | role_bit(role::down);
    if (request.key.bytes == 0 || request.role_count == 0 || request.role_count > 3 ||
            request.required_role_mask == 0 ||
            (request.required_role_mask & ~all_role_mask) != 0 ||
            (resident_role_mask & ~request.required_role_mask) != 0) {
        return dependency_transfer_status::invalid;
    }

    uint8_t seen_role_mask = 0;
    for (uint8_t i = 0; i < request.role_count; ++i) {
        const role_extent &extent = request.roles[i];
        if (!dependency_transfer_extent_valid(request.key, extent)) {
            return dependency_transfer_status::invalid;
        }
        const uint8_t bit = role_bit(extent.kind);
        if ((seen_role_mask & bit) != 0) {
            return dependency_transfer_status::invalid;
        }
        for (uint8_t previous = 0; previous < i; ++previous) {
            if (dependency_transfer_extents_overlap(request.roles[previous], extent)) {
                return dependency_transfer_status::invalid;
            }
        }
        seen_role_mask |= bit;
    }
    if (seen_role_mask != request.required_role_mask) {
        return dependency_transfer_status::invalid;
    }

    plan.missing_role_mask = request.required_role_mask & ~resident_role_mask;
    plan.critical_role_mask = plan.missing_role_mask & critical_role_mask;
    plan.deferred_role_mask = plan.missing_role_mask & ~critical_role_mask;
    plan.all_required_roles_missing =
        plan.missing_role_mask == request.required_role_mask;

    for (uint8_t i = 0; i < request.role_count; ++i) {
        const role_extent &extent = request.roles[i];
        const uint8_t bit = role_bit(extent.kind);
        if ((plan.critical_role_mask & bit) != 0) {
            if (!dependency_transfer_add_copy(
                        plan.critical, plan.critical_count,
                        plan.critical_bytes, extent)) {
                return dependency_transfer_status::invalid;
            }
        } else if ((plan.deferred_role_mask & bit) != 0) {
            if (!dependency_transfer_add_copy(
                        plan.deferred, plan.deferred_count,
                        plan.deferred_bytes, extent)) {
                return dependency_transfer_status::invalid;
            }
        }
    }

    plan.overlap_available = plan.critical_count != 0 && plan.deferred_count != 0;
    return plan.missing_role_mask == 0 ?
        dependency_transfer_status::ready :
        dependency_transfer_status::planned;
}

} // namespace ggml_moe_common
