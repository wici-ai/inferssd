// SPDX-License-Identifier: Apache-2.0
#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00 // CPU Sets need Windows 10; set before any header pulls in the SDK
#endif
#include "threads.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <pthread.h>
#include <sys/sysctl.h>
#include <time.h>
#elif defined(__linux__)
#include <fstream>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#else
#error "InferSSD platform layer: unsupported OS"
#endif

namespace ssd { namespace platform {

static void set_why(std::string * why, std::string text) {
    if (why) { *why = std::move(text); }
}

namespace detail {
bool split_fastest(const std::vector<std::pair<int, double>> & values, cpu_topology & out) {
    if (values.empty()) { return false; }
    std::vector<std::pair<int, double>> sorted(values);
    std::stable_sort(sorted.begin(), sorted.end(), [](const auto & a, const auto & b) { return a.second > b.second; });
    size_t cut = sorted.size();
    double step = 0.10;
    for (size_t i = 1; i < sorted.size(); ++i) {
        const double drop = 1.0 - sorted[i].second / sorted[i - 1].second;
        if (drop > step) { step = drop; cut = i; }
    }
    out.fast.clear();
    out.slow.clear();
    for (size_t i = 0; i < sorted.size(); ++i) { (i < cut ? out.fast : out.slow).push_back(sorted[i].first); }
    std::sort(out.fast.begin(), out.fast.end());
    std::sort(out.slow.begin(), out.slow.end());
    return true;
}
} // namespace detail

scoped_affinity::~scoped_affinity() {
    if (active_ && !restore()) {
        std::fprintf(stderr, "[ssd_platform] cannot restore thread affinity; aborting\n");
        std::abort();
    }
}

bool pin_current_thread(const cpu_list & cpus) {
#if defined(_WIN32)
    return pin_thread(GetCurrentThread(), cpus);
#else
    return pin_thread(pthread_self(), cpus);
#endif
}

#if defined(__linux__)

bool process_cpus(cpu_list & cpus, std::string * why) {
    cpus.clear();
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        const int error = errno;
        set_why(why, std::string("sched_getaffinity: ") + std::strerror(error));
        return false;
    }
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed)) { cpus.push_back(cpu); }
    }
    return true;
}

// Per-CPU capacity from the kernel (Arm big.LITTLE, Intel P/E cores), else the
// maximum frequency. One unreadable CPU makes the machine one cluster.
bool topology(cpu_topology & out, std::string * why) {
    cpu_list cpus;
    if (!process_cpus(cpus, why)) { return false; }
    if (cpus.empty()) {
        set_why(why, "no allowed CPUs");
        return false;
    }
    std::vector<std::pair<int, double>> values;
    for (int cpu : cpus) {
        double value = 0;
        for (const char * file : { "cpu_capacity", "cpufreq/cpuinfo_max_freq" }) {
            std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/" + file);
            if (in >> value && value > 0) { break; }
            value = 0;
        }
        if (value <= 0) {
            out.fast = std::move(cpus);
            out.slow.clear();
            return true;
        }
        values.emplace_back(cpu, value);
    }
    return detail::split_fastest(values, out);
}

bool affinity_supported() { return true; }

bool pin_thread(std::thread::native_handle_type thread, const cpu_list & cpus) {
    if (cpus.empty()) { return true; }
    cpu_set_t wanted, actual;
    CPU_ZERO(&wanted);
    for (int cpu : cpus) {
        if (cpu < 0 || cpu >= CPU_SETSIZE) { return false; }
        CPU_SET(cpu, &wanted);
    }
    return pthread_setaffinity_np(thread, sizeof(wanted), &wanted) == 0 &&
        pthread_getaffinity_np(thread, sizeof(actual), &actual) == 0 && CPU_EQUAL(&wanted, &actual);
}

static_assert(sizeof(cpu_set_t) <= sizeof(uint64_t[16]), "saved affinity storage too small");

scoped_affinity::scoped_affinity(const cpu_list & cpus) {
    if (cpus.empty()) { return; }
    cpu_set_t saved;
    if (pthread_getaffinity_np(pthread_self(), sizeof(saved), &saved) != 0) {
        ok_ = false;
        return;
    }
    std::memcpy(saved_, &saved, sizeof(saved));
    if (!pin_current_thread(cpus)) {
        restore();
        ok_ = false;
        return;
    }
    active_ = true;
}

bool scoped_affinity::restore() const {
    cpu_set_t saved;
    std::memcpy(&saved, saved_, sizeof(saved));
    return pthread_setaffinity_np(pthread_self(), sizeof(saved), &saved) == 0;
}

// The kernel scheduler needs no hints; Linux behaviour stays as it is.
bool set_thread_role(thread_role) { return true; }

int current_cpu() { return sched_getcpu(); }

#elif defined(__APPLE__)

static int sysctl_int(const char * name) {
    int value = 0;
    size_t size = sizeof(value);
    return sysctlbyname(name, &value, &size, nullptr, 0) == 0 && size == sizeof(value) ? value : 0;
}

bool process_cpus(cpu_list & cpus, std::string * why) {
    cpus.clear();
    const int count = sysctl_int("hw.logicalcpu");
    if (count <= 0) {
        set_why(why, "sysctl hw.logicalcpu failed");
        return false;
    }
    for (int cpu = 0; cpu < count; ++cpu) { cpus.push_back(cpu); }
    return true;
}

// perflevel0 is the fastest level; the other levels (perflevel1 on current
// Apple silicon) are the rest. Intel Macs report no levels: one cluster.
bool topology(cpu_topology & out, std::string * why) {
    cpu_list cpus;
    if (!process_cpus(cpus, why)) { return false; }
    const int fast = sysctl_int("hw.perflevel0.logicalcpu");
    const size_t cut = fast > 0 && size_t(fast) < cpus.size() ? size_t(fast) : cpus.size();
    out.fast.assign(cpus.begin(), cpus.begin() + std::ptrdiff_t(cut));
    out.slow.assign(cpus.begin() + std::ptrdiff_t(cut), cpus.end());
    return true;
}

// macOS has no thread affinity; QoS roles steer cluster placement instead.
bool affinity_supported() { return false; }

bool pin_thread(std::thread::native_handle_type, const cpu_list &) { return true; }

scoped_affinity::scoped_affinity(const cpu_list &) {}

bool scoped_affinity::restore() const { return true; }

bool set_thread_role(thread_role role) {
    const qos_class_t qos = role == thread_role::compute ? QOS_CLASS_USER_INTERACTIVE :
                            role == thread_role::io      ? QOS_CLASS_USER_INITIATED :
                                                           QOS_CLASS_UTILITY;
    return pthread_set_qos_class_self_np(qos, 0) == 0;
}

int current_cpu() { return -1; }

#elif defined(_WIN32)

// CPU Sets of processor group 0, by logical processor index. Ids and
// efficiency classes are fixed while the system runs.
struct cpu_set_table {
    bool available = false;       // every active group-0 processor has a CPU Set
    std::vector<ULONG> id;
    std::vector<int> efficiency;  // -1: no CPU Set
    ULONG total = 0;              // CPU Sets in all groups
};

static cpu_set_table read_cpu_sets() {
    cpu_set_table table;
    const DWORD active = GetActiveProcessorCount(0);
    if (active == 0 || active > sizeof(DWORD_PTR) * 8) { return table; }
    table.id.assign(active, 0);
    table.efficiency.assign(active, -1);
    ULONG bytes = 0;
    GetSystemCpuSetInformation(nullptr, 0, &bytes, GetCurrentProcess(), 0);
    if (bytes == 0) { return table; }
    std::vector<uint64_t> storage((bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t)); // entries hold DWORD64
    auto * base = reinterpret_cast<unsigned char *>(storage.data());
    if (!GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(base), bytes, &bytes,
            GetCurrentProcess(), 0)) {
        return table;
    }
    for (ULONG at = 0; at + sizeof(DWORD) * 2 <= bytes;) {
        const auto * entry = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION *>(base + at);
        if (entry->Size == 0 || at + entry->Size > bytes) { break; }
        if (entry->Type == CpuSetInformation) {
            ++table.total;
            const DWORD index = entry->CpuSet.LogicalProcessorIndex;
            if (entry->CpuSet.Group == 0 && index < active) {
                table.id[index] = entry->CpuSet.Id;
                table.efficiency[index] = entry->CpuSet.EfficiencyClass;
            }
        }
        at += entry->Size;
    }
    table.available = std::all_of(table.efficiency.begin(), table.efficiency.end(), [](int e) { return e >= 0; });
    return table;
}

static const cpu_set_table & cpu_sets() {
    static const cpu_set_table table = read_cpu_sets();
    return table;
}

// Group-0 affinity masks are one DWORD_PTR (64 bits on x64, LLP64).
constexpr int mask_bits = int(sizeof(DWORD_PTR) * 8);

static bool single_group() { return GetActiveProcessorGroupCount() == 1; }

static bool process_mask(DWORD_PTR & mask) {
    DWORD_PTR system = 0;
    mask = 0;
    return single_group() && GetProcessAffinityMask(GetCurrentProcess(), &mask, &system) && mask != 0;
}

bool process_cpus(cpu_list & cpus, std::string * why) {
    cpus.clear();
    if (!single_group()) {
        set_why(why, "more than one processor group (over 64 logical CPUs) is not supported");
        return false;
    }
    DWORD_PTR mask = 0;
    if (!process_mask(mask)) {
        set_why(why, "GetProcessAffinityMask failed: error " + std::to_string(GetLastError()));
        return false;
    }
    for (int cpu = 0; cpu < mask_bits; ++cpu) {
        if ((mask >> cpu) & 1) { cpus.push_back(cpu); }
    }
    return true;
}

// The highest EfficiencyClass is the fast cluster; without CPU Sets the
// machine is one cluster.
bool topology(cpu_topology & out, std::string * why) {
    cpu_list cpus;
    if (!process_cpus(cpus, why)) { return false; }
    const auto & sets = cpu_sets();
    int best = -1;
    if (sets.available) {
        for (int cpu : cpus) { best = std::max(best, sets.efficiency[size_t(cpu)]); }
    }
    out.fast.clear();
    out.slow.clear();
    for (int cpu : cpus) { (best < 0 || sets.efficiency[size_t(cpu)] == best ? out.fast : out.slow).push_back(cpu); }
    return true;
}

bool affinity_supported() { return true; }

static bool cpu_mask(const cpu_list & cpus, DWORD_PTR & mask) {
    mask = 0;
    for (int cpu : cpus) {
        if (cpu < 0 || cpu >= mask_bits) { return false; }
        mask |= DWORD_PTR(1) << cpu;
    }
    return true;
}

// A thread's selected CPU Sets as a group-0 mask (0: none selected).
static bool selected_mask(HANDLE thread, const cpu_set_table & sets, DWORD_PTR & mask) {
    std::vector<ULONG> ids(std::max<ULONG>(1, sets.total));
    ULONG count = 0;
    if (!GetThreadSelectedCpuSets(thread, ids.data(), ULONG(ids.size()), &count)) { return false; }
    mask = 0;
    for (ULONG i = 0; i < count; ++i) {
        const auto at = std::find(sets.id.begin(), sets.id.end(), ids[i]);
        if (at == sets.id.end()) { return false; }
        mask |= DWORD_PTR(1) << (at - sets.id.begin());
    }
    return true;
}

static bool select_mask(HANDLE thread, const cpu_set_table & sets, DWORD_PTR mask) {
    std::vector<ULONG> ids;
    for (size_t cpu = 0; cpu < sets.id.size(); ++cpu) {
        if ((mask >> cpu) & 1) { ids.push_back(sets.id[cpu]); }
    }
    return SetThreadSelectedCpuSets(thread, ids.empty() ? nullptr : ids.data(), ULONG(ids.size())) != 0;
}

bool pin_thread(std::thread::native_handle_type handle, const cpu_list & cpus) {
    if (cpus.empty()) { return true; }
    const HANDLE thread = static_cast<HANDLE>(handle);
    DWORD_PTR wanted = 0, allowed = 0;
    // Like a Linux mask outside the allowed set, CPUs the process may not use are refused.
    if (!cpu_mask(cpus, wanted) || !process_mask(allowed) || (wanted & ~allowed) != 0) { return false; }
    const auto & sets = cpu_sets();
    if (sets.available) {
        // A hard thread affinity outranks selected CPU Sets: the thread runs on
        // their intersection, or ignores the sets when it is empty. Exact only
        // if that affinity admits every wanted CPU.
        GROUP_AFFINITY hard = {};
        DWORD_PTR actual = 0;
        return GetThreadGroupAffinity(thread, &hard) && hard.Group == 0 && (DWORD_PTR(hard.Mask) & wanted) == wanted &&
            select_mask(thread, sets, wanted) && selected_mask(thread, sets, actual) && actual == wanted;
    }
    GROUP_AFFINITY affinity = {}, actual = {};
    affinity.Mask = KAFFINITY(wanted);
    affinity.Group = 0;
    return SetThreadGroupAffinity(thread, &affinity, nullptr) && GetThreadGroupAffinity(thread, &actual) &&
        actual.Group == 0 && actual.Mask == affinity.Mask;
}

// saved_: [0] 1 = selected CPU Sets, 2 = group affinity; [1] mask; [2] group.
scoped_affinity::scoped_affinity(const cpu_list & cpus) {
    if (cpus.empty()) { return; }
    const HANDLE self = GetCurrentThread();
    const auto & sets = cpu_sets();
    if (sets.available) {
        DWORD_PTR mask = 0;
        if (!selected_mask(self, sets, mask)) {
            ok_ = false;
            return;
        }
        saved_[0] = 1;
        saved_[1] = uint64_t(mask);
    } else {
        GROUP_AFFINITY affinity = {};
        if (!GetThreadGroupAffinity(self, &affinity)) {
            ok_ = false;
            return;
        }
        saved_[0] = 2;
        saved_[1] = uint64_t(affinity.Mask);
        saved_[2] = affinity.Group;
    }
    if (!pin_current_thread(cpus)) {
        restore();
        ok_ = false;
        return;
    }
    active_ = true;
}

bool scoped_affinity::restore() const {
    const HANDLE self = GetCurrentThread();
    if (saved_[0] == 1) { return select_mask(self, cpu_sets(), DWORD_PTR(saved_[1])); }
    GROUP_AFFINITY affinity = {};
    affinity.Mask = KAFFINITY(saved_[1]);
    affinity.Group = WORD(saved_[2]);
    return SetThreadGroupAffinity(self, &affinity, nullptr) != 0;
}

// Windows 11 parks cores and moves throttled (EcoQoS) threads onto efficiency
// cores; compute and io threads opt out, as ggml's own workers do.
bool set_thread_role(thread_role role) {
    THREAD_POWER_THROTTLING_STATE state = {};
    state.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    state.ControlMask = role == thread_role::background ? 0 : THREAD_POWER_THROTTLING_EXECUTION_SPEED;
    state.StateMask = 0;
    return SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &state, sizeof(state)) != 0;
}

uint64_t thread_cpu_time_ns() {
    FILETIME creation, exit, kernel, user;
    if (!GetThreadTimes(GetCurrentThread(), &creation, &exit, &kernel, &user)) { return 0; }
    const auto ticks = [](const FILETIME & t) { return (uint64_t(t.dwHighDateTime) << 32) | t.dwLowDateTime; };
    return (ticks(kernel) + ticks(user)) * 100; // 100 ns units
}

int current_cpu() {
    PROCESSOR_NUMBER number = {};
    GetCurrentProcessorNumberEx(&number);
    return number.Group == 0 ? int(number.Number) : -1;
}

#endif

#if !defined(_WIN32)
uint64_t thread_cpu_time_ns() {
    timespec value;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) { return 0; }
    return uint64_t(value.tv_sec) * 1000000000ull + uint64_t(value.tv_nsec);
}
#endif

} } // namespace ssd::platform
