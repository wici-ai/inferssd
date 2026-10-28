#pragma once
// Test prerequisites only: never add a runtime fallback or hide numerical/I/O defects.
#if defined(__linux__)
#include "../../ggml/src/ggml-ssd/host/ledger.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/io_uring.h>
#include <sys/syscall.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <unistd.h>

namespace test_resources {
struct unavailable : std::runtime_error { using std::runtime_error::runtime_error; };

inline void need(bool available, const std::string & reason) {
    if (!available) { throw unavailable(reason); }
}

inline void cpu_threads(int compute, int io) {
    if (compute < 1 || io < 0 || compute > 255 || io > 255) {
        throw std::runtime_error("invalid test compute/IO thread count");
    }
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask)) {
        throw std::runtime_error(std::string("test sched_getaffinity: ") + std::strerror(errno));
    }
    const int available = CPU_COUNT(&mask);
    need(available >= compute + io, "CPU affinity has " + std::to_string(available) +
            " CPUs; fixture needs compute=" + std::to_string(compute) + " IO=" + std::to_string(io));
}

// Unpinned fixtures (no cpu.compute_cpus/io_cpus): participants may share CPUs. Placement
// changes timing only, never results, so a small machine still runs the correctness cases.
inline void cpu_participants(int compute, int io) {
    if (compute < 1 || io < 0 || compute > 255 || io > 255) {
        throw std::runtime_error("invalid test compute/IO thread count");
    }
}

inline void io_uring() {
    io_uring_params parameters = {};
    const int fd = int(syscall(__NR_io_uring_setup, 2, &parameters));
    if (fd < 0) {
        const int error = errno;
        const std::string message = std::string("io_uring_setup (no fallback): ") + std::strerror(error);
        if (error == ENOSYS || error == EPERM || error == EACCES) { throw unavailable(message); }
        throw std::runtime_error(message);
    }
    close(fd);
}

inline bool direct_unsupported(int error) { return error == EINVAL || error == EOPNOTSUPP || error == ENOSYS; }

// Probe the fixture's actual file, not another mount. ENOENT/EACCES/EIO/short reads are errors.
inline int open_direct(const std::string & path) {
    const int fd = open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd < 0) {
        const int error = errno;
        if (direct_unsupported(error)) { throw unavailable("O_DIRECT open unsupported on fixture filesystem: " + path); }
        throw std::runtime_error("fixture open: " + path + ": " + std::strerror(error));
    }
    void * buffer = nullptr;
    if (posix_memalign(&buffer, 4096, 4096)) { close(fd); throw std::bad_alloc(); }
    ssize_t count;
    do { count = pread(fd, buffer, 4096, 0); } while (count < 0 && errno == EINTR);
    const int error = errno;
    free(buffer);
    if (count != 4096) {
        close(fd);
        if (count < 0 && direct_unsupported(error)) {
            throw unavailable("aligned O_DIRECT read unsupported on fixture filesystem: " + path);
        }
        throw std::runtime_error("fixture direct read: " + path + ": " +
                (count < 0 ? std::strerror(error) : "short header page"));
    }
    return fd;
}

inline void finite_cgroup(kimi_host_ledger & ledger) {
    kimi_host_ledger::room room;
    ledger.cgroup_term(0, room);
    need(room.cgroup != kimi_host_ledger::room_unreadable,
            "cache fixture needs readable finite cgroup v2 memory.max/current ancestors");
    // Do not infer a payload budget here: individual fixtures change reserves
    // and intentionally exercise pressure. Their size/ownership checks remain authoritative.
}
inline void finite_cgroup() {
    kimi_host_ledger ledger;
    ledger.configure(false);
    finite_cgroup(ledger);
}

// These messages originate from the named production syscalls. Check their exact suffix;
// EINVAL/EFAULT from registration, unrelated ENOMEM, and other exceptions remain failures.
inline bool operation_error(const std::string & message, const char * operation, int error) {
    const std::string suffix = std::string(operation) + ": " + std::strerror(error);
    return message == suffix || (message.size() > suffix.size() &&
            message.compare(message.size() - suffix.size(), suffix.size(), suffix) == 0 &&
            message[message.size() - suffix.size() - 1] == ' ');
}
inline bool syscall_unavailable(const std::string & message) {
    for (int error : { ENOSYS, EPERM, EACCES }) {
        if (operation_error(message, "io_uring_setup (no fallback)", error)) { return true; }
    }
    // The actual registration is authoritative: it includes the fixture's real
    // geometry and other live rings. A tiny probe cannot establish its pin budget.
    for (int error : { EPERM, ENOMEM }) {
        if (operation_error(message, "IORING_REGISTER_BUFFERS (check RLIMIT_MEMLOCK; no fallback)", error)) { return true; }
    }
    return false;
}
inline int report(const char * prefix, const std::exception & error) {
    const bool skip = dynamic_cast<const unavailable *>(&error) || syscall_unavailable(error.what());
    std::fprintf(stderr, "%s_%s: %s\n", prefix, skip ? "SKIP" : "FAILED", error.what());
    if (skip && std::strstr(error.what(), "IORING_REGISTER_BUFFERS")) {
        rlimit limit = {};
        if (!getrlimit(RLIMIT_MEMLOCK, &limit)) {
            std::fprintf(stderr, "actual fixture buffer registration refused; RLIMIT_MEMLOCK=%llu bytes\n",
                    static_cast<unsigned long long>(limit.rlim_cur));
        }
    }
    return skip ? 77 : 1;
}
} // namespace test_resources
#endif
