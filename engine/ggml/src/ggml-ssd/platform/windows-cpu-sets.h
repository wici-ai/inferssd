// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "threads.h"

#include <algorithm>
#include <cstddef>
#include <optional>

namespace ssd { namespace platform { namespace detail {

// Indexed by group * 64 + group-local CPU. CPU Set IDs are opaque Win32 ULONGs.
template<typename Id>
struct windows_cpu_sets {
    std::vector<std::optional<Id>> ids;
    std::vector<bool> reserved_elsewhere;

    explicit windows_cpu_sets(std::size_t slots) : ids(slots), reserved_elsewhere(slots) {}

    void record(std::size_t cpu, Id id, bool allocated, bool allocated_to_process) {
        if (cpu < ids.size()) {
            ids[cpu] = id;
            reserved_elsewhere[cpu] = allocated && !allocated_to_process;
        }
    }

    bool process_cpus(const cpu_list & hard, const std::vector<Id> & defaults, cpu_list & cpus, std::string * why) const {
        cpu_list selected;
        for (Id id : defaults) {
            const auto at = std::find(ids.begin(), ids.end(), id);
            if (at == ids.end()) {
                if (why) { *why = "unknown process-default CPU Set " + std::to_string(id); }
                return false;
            }
            const int cpu = int(at - ids.begin());
            if (!reserved_elsewhere[cpu] && std::binary_search(hard.begin(), hard.end(), cpu)) {
                selected.push_back(cpu);
            }
        }
        if (selected.empty()) {
            // Windows ignores an unusable default assignment. It can still only
            // schedule on hard-allowed CPUs that are not reserved by another process.
            for (int cpu : hard) {
                if (!reserved_elsewhere[cpu]) { selected.push_back(cpu); }
            }
        } else {
            std::sort(selected.begin(), selected.end());
            selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
        }
        if (selected.empty()) {
            if (why) { *why = "Windows reported no available CPUs in the process groups"; }
            return false;
        }
        cpus.swap(selected);
        return true;
    }
};

}}} // namespace ssd::platform::detail
