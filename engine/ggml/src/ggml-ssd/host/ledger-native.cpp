// SPDX-License-Identifier: Apache-2.0
// The native budget sources of ledger-native.h, one implementation per OS.
#include "ledger-native.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/vm_page_size.h>
#else
#error "ledger-native is the macOS and Windows budget source; Linux uses ledger.h"
#endif

namespace ssd { namespace host { namespace native {

#if defined(_WIN32)

bool process_use(uint64_t & bytes) {
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters),
                              sizeof(counters))) {
        return false;
    }
    bytes = uint64_t(counters.PrivateUsage);
    return true;
}

bool available(uint64_t & bytes) {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) { return false; }
    // Physical memory alone overstates room when the commit limit binds first.
    bytes = std::min<uint64_t>(status.ullAvailPhys, status.ullAvailPageFile);
    return true;
}

bool job_limits(envelopes & out) {
    BOOL in_job = FALSE;
    if (!IsProcessInJob(GetCurrentProcess(), nullptr, &in_job)) { return false; }
    if (!in_job) { return true; }
    // A null handle names the process's immediate job.
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    if (!QueryInformationJobObject(nullptr, JobObjectExtendedLimitInformation, &limits, sizeof(limits), nullptr)) {
        return false;
    }
    const DWORD flags = limits.BasicLimitInformation.LimitFlags;
    if (flags & JOB_OBJECT_LIMIT_PROCESS_MEMORY) {
        uint64_t used = 0;
        if (!process_use(used)) { return false; }
        out.push({"job process", uint64_t(limits.ProcessMemoryLimit), used});
    }
    if (flags & JOB_OBJECT_LIMIT_JOB_MEMORY) {
        // JobMemory is the commit of every process in the job now; the extended limits carry only its peak.
        JOBOBJECT_LIMIT_VIOLATION_INFORMATION job{};
        if (!QueryInformationJobObject(nullptr, JobObjectLimitViolationInformation, &job, sizeof(job), nullptr)) {
            return false;
        }
        out.push({"job", uint64_t(limits.JobMemoryLimit), uint64_t(job.JobMemory)});
    }
    return true;
}

#else // __APPLE__

bool process_use(uint64_t & bytes) {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS ||
            count < TASK_VM_INFO_REV1_COUNT) {
        return false;
    }
    bytes = info.phys_footprint;
    return true;
}

bool available(uint64_t & bytes) {
    // Each mach_host_self() call adds a send-right reference; keep one for the process.
    static const mach_port_t host = mach_host_self();
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) != KERN_SUCCESS) {
        return false;
    }
    // free_count includes the speculative pages (vm_stat prints free - speculative), and the file-backed
    // external_page_count includes them again (internal + external = active + inactive + speculative). Purgeable
    // pages are anonymous, so neither count holds them. Inactive anonymous pages are not available: the kernel
    // must compress or swap them first.
    const uint64_t free_count = vm.free_count, speculative = vm.speculative_count;
    const uint64_t pages = free_count - std::min(free_count, speculative) + uint64_t(vm.external_page_count) +
                           uint64_t(vm.purgeable_count);
    bytes = pages * uint64_t(vm_kernel_page_size);
    return true;
}

bool job_limits(envelopes &) { return true; }

#endif

}}} // namespace ssd::host::native
