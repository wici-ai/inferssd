// SPDX-License-Identifier: Apache-2.0
#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00 // CPU Sets need Windows 10; set before any header pulls in the SDK
#endif
#include "threads.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <optional>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include "windows-cpu-sets.h"
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <os/os_sync_wait_on_address.h>
#include <pthread.h>
#include <sched.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>
#elif defined(__linux__)
#include <climits>
#include <fstream>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#else
#error "InferSSD platform layer: unsupported OS"
#endif

namespace ssd { namespace platform {

static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t) && std::atomic<uint32_t>::is_always_lock_free,
              "a parking word is a plain 32-bit word");

static void set_why(std::string * why, std::string text) {
    if (why) { *why = std::move(text); }
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

// sched_getaffinity(0) reads the calling thread's mask.
bool current_thread_cpus(cpu_list & cpus, std::string * why) { return process_cpus(cpus, why); }

pin_policy thread_pinning() { return pin_policy::exact; }

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

// The kernel scheduler needs no hints; Linux behaviour stays as it is.
bool set_thread_role(thread_role) { return true; }

int current_cpu() { return sched_getcpu(); }

// /proc/stat per CPU: user nice system idle iowait irq softirq steal guest guest_nice, in USER_HZ ticks; guest time
// is already in user.
bool cpus_busy_ns(const cpu_list & cpus, uint64_t & busy_ns, uint64_t & tick_ns, std::string * why) {
    const long hz = sysconf(_SC_CLK_TCK);
    std::ifstream in("/proc/stat");
    if (hz <= 0 || !in) {
        set_why(why, "/proc/stat is not readable");
        return false;
    }
    tick_ns = uint64_t(1000000000 / hz);
    uint64_t ticks = 0;
    size_t found = 0;
    std::string line;
    while (std::getline(in, line)) {
        int cpu = -1;
        unsigned long long user, nice, system, idle, iowait, irq, softirq, steal = 0;
        if (std::sscanf(line.c_str(), "cpu%d %llu %llu %llu %llu %llu %llu %llu %llu", &cpu, &user, &nice, &system, &idle,
                        &iowait, &irq, &softirq, &steal) < 8) {
            continue;
        }
        if (!std::binary_search(cpus.begin(), cpus.end(), cpu)) { continue; }
        ticks += user + nice + system + irq + softirq + steal;
        ++found;
    }
    if (found != cpus.size()) {
        set_why(why, "/proc/stat lists " + std::to_string(found) + " of " + std::to_string(cpus.size()) + " CPUs");
        return false;
    }
    busy_ns = ticks * tick_ns;
    return true;
}

void wait_on(const std::atomic<uint32_t> & word, uint32_t expected) noexcept {
    syscall(SYS_futex, &word, FUTEX_WAIT_PRIVATE, expected, nullptr, nullptr, 0);
}

void wake_all(const std::atomic<uint32_t> & word) noexcept {
    syscall(SYS_futex, &word, FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
}

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

// No thread affinity: every thread may run on every CPU.
bool current_thread_cpus(cpu_list & cpus, std::string * why) { return process_cpus(cpus, why); }

// macOS has no thread affinity; QoS roles steer cluster placement instead.
pin_policy thread_pinning() { return pin_policy::no_op; }

bool pin_thread(std::thread::native_handle_type, const cpu_list &) { return true; }

bool set_thread_role(thread_role role) {
    const qos_class_t qos = role == thread_role::io ? QOS_CLASS_USER_INITIATED : QOS_CLASS_UTILITY;
    return pthread_set_qos_class_self_np(qos, 0) == 0;
}

int current_cpu() { return -1; }

// Each CPU's user, system and nice ticks (host_processor_info), counted at the clock tick rate.
bool cpus_busy_ns(const cpu_list & cpus, uint64_t & busy_ns, uint64_t & tick_ns, std::string * why) {
    natural_t count = 0;
    processor_info_array_t info = nullptr;
    mach_msg_type_number_t words = 0;
    if (host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO, &count, &info, &words) != KERN_SUCCESS) {
        set_why(why, "host_processor_info failed");
        return false;
    }
    const auto * load = reinterpret_cast<const processor_cpu_load_info_data_t *>(info);
    uint64_t ticks = 0;
    bool ok = true;
    for (int cpu : cpus) {
        if (cpu < 0 || natural_t(cpu) >= count) { ok = false; break; }
        ticks += uint64_t(load[cpu].cpu_ticks[CPU_STATE_USER]) + load[cpu].cpu_ticks[CPU_STATE_SYSTEM] +
                 load[cpu].cpu_ticks[CPU_STATE_NICE];
    }
    vm_deallocate(mach_task_self(), vm_address_t(info), vm_size_t(words) * sizeof(integer_t));
    const long hz = sysconf(_SC_CLK_TCK);
    if (!ok || hz <= 0) {
        set_why(why, ok ? "no clock tick rate" : "a CPU beyond the processors the host reports");
        return false;
    }
    tick_ns = uint64_t(1000000000 / hz);
    busy_ns = ticks * tick_ns;
    return true;
}

// Before macOS 14.4 there is no public wait on an address; the waiter yields and rechecks.
void wait_on(const std::atomic<uint32_t> & word, uint32_t expected) noexcept {
    if (__builtin_available(macOS 14.4, *)) {
        os_sync_wait_on_address(const_cast<std::atomic<uint32_t> *>(&word), expected, sizeof(uint32_t),
                                OS_SYNC_WAIT_ON_ADDRESS_NONE);
    } else {
        sched_yield();
    }
}

void wake_all(const std::atomic<uint32_t> & word) noexcept {
    if (__builtin_available(macOS 14.4, *)) {
        os_sync_wake_by_address_all(const_cast<std::atomic<uint32_t> *>(&word), sizeof(uint32_t),
                                    OS_SYNC_WAKE_BY_ADDRESS_NONE);
    }
}

#elif defined(_WIN32)

// A Windows processor number has a group and a local index in 0..63.
constexpr int group_width = 64;
using windows_cpu_sets = detail::windows_cpu_sets<ULONG>;

static bool read_cpu_sets(HANDLE process, windows_cpu_sets & table, std::string * why) {
    ULONG bytes = 0;
    if (!GetSystemCpuSetInformation(nullptr, 0, &bytes, process, 0) &&
            (GetLastError() != ERROR_INSUFFICIENT_BUFFER || !bytes)) {
        set_why(why, "GetSystemCpuSetInformation failed: error " + std::to_string(GetLastError()));
        return false;
    }
    if (!bytes) { return true; }
    std::vector<uint64_t> storage((bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t)); // entries hold DWORD64
    auto * base = reinterpret_cast<unsigned char *>(storage.data());
    if (!GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(base), bytes, &bytes, process, 0)) {
        set_why(why, "GetSystemCpuSetInformation failed: error " + std::to_string(GetLastError()));
        return false;
    }
    ULONG at = 0;
    while (at < bytes) {
        if (bytes - at < sizeof(DWORD) * 2) { break; }
        const auto * entry = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION *>(base + at);
        if (entry->Size < sizeof(DWORD) * 2 || entry->Size > bytes - at) { break; }
        if (entry->Type == CpuSetInformation) {
            if (entry->Size <= offsetof(SYSTEM_CPU_SET_INFORMATION, CpuSet.AllFlags)) { break; }
            const auto & set = entry->CpuSet;
            table.record(DWORD(set.Group) * group_width + set.LogicalProcessorIndex,
                         set.Id, set.Allocated, set.AllocatedToTargetProcess);
        }
        at += entry->Size;
    }
    if (at == bytes) { return true; }
    set_why(why, "invalid GetSystemCpuSetInformation record");
    return false;
}

// Only the stable group-0 IDs are cached for exact single-group pinning.
struct cpu_set_table {
    bool available;
    std::vector<std::optional<ULONG>> id;
};

static const cpu_set_table & cpu_sets() {
    static const cpu_set_table table = [] {
        windows_cpu_sets snapshot(GetActiveProcessorCount(0));
        const bool complete = read_cpu_sets(GetCurrentProcess(), snapshot, nullptr) && !snapshot.ids.empty() &&
            std::all_of(snapshot.ids.begin(), snapshot.ids.end(), [](auto id) { return id.has_value(); });
        return cpu_set_table{complete, std::move(snapshot.ids)};
    }();
    return table;
}

// Group-0 affinity masks are one DWORD_PTR (64 bits on x64, LLP64).
constexpr int mask_bits = int(sizeof(DWORD_PTR) * 8);

static bool single_group() { return GetActiveProcessorGroupCount() == 1; }

static bool process_mask(DWORD_PTR & mask) {
    DWORD_PTR system = 0;
    mask = 0;
    return GetProcessAffinityMask(GetCurrentProcess(), &mask, &system) && mask != 0;
}

// Windows 10 can start this process in just one group, and Windows 11 can schedule
// it across several. A process affinity mask describes CPUs only in the former case.
static bool grouped_process_cpus(USHORT group_count, cpu_list & cpus, std::string * why) {
    const HANDLE process = GetCurrentProcess();
    USHORT count = group_count;
    std::vector<USHORT> groups(count);
    if (!GetProcessGroupAffinity(process, &count, groups.data()) || !count) {
        set_why(why, "GetProcessGroupAffinity failed: error " + std::to_string(GetLastError()));
        return false;
    }
    groups.resize(count);
    std::sort(groups.begin(), groups.end());
    if (groups.back() >= group_count) {
        set_why(why, "GetProcessGroupAffinity returned a group outside the active topology: " +
                std::to_string(groups.back()));
        return false;
    }
    DWORD_PTR mask = ~DWORD_PTR(0);
    if (count == 1 && !process_mask(mask)) {
        set_why(why, "GetProcessAffinityMask failed: error " + std::to_string(GetLastError()));
        return false;
    }
    cpu_list hard;
    for (USHORT group : groups) {
        const DWORD active = GetActiveProcessorCount(group);
        if (!active || active > DWORD(mask_bits)) {
            set_why(why, "GetActiveProcessorCount failed for group " + std::to_string(group));
            return false;
        }
        for (DWORD i = 0; i < active; ++i) {
            if ((mask >> i) & 1) { hard.push_back(int(group) * group_width + int(i)); }
        }
    }
    ULONG n = 0;
    if (!GetProcessDefaultCpuSets(process, nullptr, 0, &n) && GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        set_why(why, "GetProcessDefaultCpuSets failed: error " + std::to_string(GetLastError()));
        return false;
    }
    std::vector<ULONG> defaults(n);
    if (n) {
        if (!GetProcessDefaultCpuSets(process, defaults.data(), n, &n)) {
            set_why(why, "GetProcessDefaultCpuSets failed: error " + std::to_string(GetLastError()));
            return false;
        }
        defaults.resize(n);
    }
    // Reservations can change while the process runs. Query against its handle
    // even without defaults so that foreign reservations are always excluded.
    windows_cpu_sets sets(size_t(group_count) * group_width);
    return read_cpu_sets(process, sets, why) && sets.process_cpus(hard, defaults, cpus, why);
}

bool process_cpus(cpu_list & cpus, std::string * why) {
    cpus.clear();
    const WORD groups = GetActiveProcessorGroupCount();
    if (groups > 1) { return grouped_process_cpus(groups, cpus, why); }
    DWORD_PTR mask = 0;
    if (!groups || !process_mask(mask)) {
        set_why(why, "cannot read process affinity: error " + std::to_string(GetLastError()));
        return false;
    }
    for (int cpu = 0; cpu < mask_bits; ++cpu) {
        if ((mask >> cpu) & 1) { cpus.push_back(cpu); }
    }
    return true;
}

pin_policy thread_pinning() { return single_group() ? pin_policy::exact : pin_policy::unsupported; }

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
    std::vector<ULONG> ids(sets.id.size());
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
        if ((mask >> cpu) & 1) { ids.push_back(*sets.id[cpu]); }
    }
    return SetThreadSelectedCpuSets(thread, ids.empty() ? nullptr : ids.data(), ULONG(ids.size())) != 0;
}

// The hard affinity, narrowed by the selected CPU Sets when they meet it; the
// scheduler ignores sets outside it.
bool current_thread_cpus(cpu_list & cpus, std::string * why) {
    cpus.clear();
    if (!single_group()) {
        set_why(why, "the thread's exact affinity across Windows processor groups is unknown");
        return false;
    }
    const HANDLE thread = GetCurrentThread();
    GROUP_AFFINITY hard = {};
    if (!GetThreadGroupAffinity(thread, &hard) || hard.Group != 0) {
        set_why(why, "GetThreadGroupAffinity failed: error " + std::to_string(GetLastError()));
        return false;
    }
    DWORD_PTR mask = DWORD_PTR(hard.Mask), selected = 0;
    const auto & sets = cpu_sets();
    if (sets.available) {
        if (!selected_mask(thread, sets, selected)) {
            set_why(why, "GetThreadSelectedCpuSets failed: error " + std::to_string(GetLastError()));
            return false;
        }
        if (mask & selected) { mask &= selected; }
    }
    for (int cpu = 0; cpu < mask_bits; ++cpu) {
        if ((mask >> cpu) & 1) { cpus.push_back(cpu); }
    }
    return true;
}

bool pin_thread(std::thread::native_handle_type handle, const cpu_list & cpus) {
    if (cpus.empty()) { return true; }
    if (!single_group()) { return false; }
    const HANDLE thread = static_cast<HANDLE>(handle);
    DWORD_PTR wanted = 0, allowed = 0;
    // Like a Linux mask outside the allowed set, CPUs the process may not use are refused.
    if (!cpu_mask(cpus, wanted) || !process_mask(allowed) || (wanted & ~allowed) != 0) { return false; }
    const auto & sets = cpu_sets();
    GROUP_AFFINITY saved = {}, affinity = {}, actual = {};
    DWORD_PTR saved_selected = 0, selected = 0;
    if (!GetThreadGroupAffinity(thread, &saved) || saved.Group != 0 ||
            (sets.available && !selected_mask(thread, sets, saved_selected))) { return false; }
    affinity.Mask = KAFFINITY(wanted);
    affinity.Group = 0;
    // CPU Sets are a scheduler preference: a parked core may be ignored even
    // when the selected IDs read back correctly. Hard affinity makes the pin exact.
    if (!SetThreadGroupAffinity(thread, &affinity, nullptr)) { return false; }
    if (GetThreadGroupAffinity(thread, &actual) && actual.Group == 0 && actual.Mask == affinity.Mask &&
            (!sets.available || (select_mask(thread, sets, wanted) &&
                selected_mask(thread, sets, selected) && selected == wanted))) { return true; }
    SetThreadGroupAffinity(thread, &saved, nullptr);
    if (sets.available) { select_mask(thread, sets, saved_selected); }
    return false;
}

// Windows 11 parks cores and moves throttled (EcoQoS) threads onto efficiency
// cores; io threads opt out, as ggml's own workers do.
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

// GetSystemTimes sums every processor of the group, its kernel time including idle time, as the clock interrupt
// advances it.
bool cpus_busy_ns(const cpu_list & cpus, uint64_t & busy_ns, uint64_t & tick_ns, std::string * why) {
    const DWORD count = GetActiveProcessorCount(0);
    if (!single_group() || cpus.size() != count || (count && cpus.back() != int(count) - 1)) {
        set_why(why, "Windows reads the busy time of every processor of the group only");
        return false;
    }
    FILETIME idle, kernel, user;
    DWORD adjustment = 0, increment = 0;
    BOOL disabled = FALSE;
    if (!GetSystemTimes(&idle, &kernel, &user) || !GetSystemTimeAdjustment(&adjustment, &increment, &disabled) ||
        !increment) {
        set_why(why, "GetSystemTimes failed");
        return false;
    }
    const auto ticks = [](const FILETIME & t) { return (uint64_t(t.dwHighDateTime) << 32) | t.dwLowDateTime; };
    tick_ns = uint64_t(increment) * 100;
    busy_ns = (ticks(kernel) + ticks(user) - ticks(idle)) * 100; // 100 ns units
    return true;
}

int current_cpu() {
    PROCESSOR_NUMBER number = {};
    GetCurrentProcessorNumberEx(&number);
    return int(number.Group) * group_width + number.Number;
}

void wait_on(const std::atomic<uint32_t> & word, uint32_t expected) noexcept {
    WaitOnAddress(const_cast<std::atomic<uint32_t> *>(&word), &expected, sizeof(uint32_t), INFINITE);
}

void wake_all(const std::atomic<uint32_t> & word) noexcept {
    WakeByAddressAll(const_cast<std::atomic<uint32_t> *>(&word));
}

#endif

#if !defined(_WIN32)
uint64_t thread_cpu_time_ns() {
    timespec value;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) { return 0; }
    return uint64_t(value.tv_sec) * 1000000000ull + uint64_t(value.tv_nsec);
}
#endif

bool affinity_supported() { return thread_pinning() == pin_policy::exact; }

} } // namespace ssd::platform
