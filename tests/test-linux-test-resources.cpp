#include "linux-test-resources.h"
#include <functional>

static void require(bool ok, const char * why) { if (!ok) { throw std::runtime_error(why); } }
static void skipped(const std::function<void()> & run) {
    try { run(); } catch (const test_resources::unavailable &) { return; }
    throw std::runtime_error("missing resource was not classified as unavailable");
}
int main() {
    try {
        using namespace test_resources;
        for (int code : { ENOSYS, EPERM, EACCES }) {
            require(syscall_unavailable(std::string("create: io_uring_setup (no fallback): ") + std::strerror(code)), "setup capability failure");
        }
        for (int code : { ENOMEM, EPERM }) {
            require(syscall_unavailable(std::string("run: IORING_REGISTER_BUFFERS (check RLIMIT_MEMLOCK; no fallback): ") + std::strerror(code)), "actual registration budget failure");
        }
        for (int code : { EINVAL, EFAULT, EIO }) {
            require(!syscall_unavailable(std::string("IORING_REGISTER_BUFFERS (check RLIMIT_MEMLOCK; no fallback): ") + std::strerror(code)), "registration bug hidden as skip");
        }
        for (const char * message : { "fixture write failed", "numeric mismatch", "allocation: Cannot allocate memory",
                "io_uring_enter wait: Operation not permitted", "io_uring_setup (no fallback): Operation not permitted then failed" }) {
            require(!syscall_unavailable(message), "unclassified failure hidden as skip");
        }
        cpu_set_t mask; CPU_ZERO(&mask);
        require(!sched_getaffinity(0, sizeof(mask), &mask), "read actual affinity");
        cpu_threads(1, 0);
        if (CPU_COUNT(&mask) <= 128) { skipped([&] { cpu_threads(CPU_COUNT(&mask), 1); }); }
        bool invalid_failed = false;
        try { cpu_threads(0, 1); } catch (const unavailable &) { throw; }
        catch (const std::runtime_error &) { invalid_failed = true; }
        require(invalid_failed, "invalid threads must fail");

        kimi_host_ledger ledger;
        ledger.cgroup = "/sys/fs/cgroup/fixture";
        ledger.reserve = 512ull << 20;
        std::string limit = "max", current = "0";
        ledger.read = [&](const std::string & path, std::string & value) {
            if (path == "/sys/fs/cgroup/fixture/memory.max") { value = limit; return true; }
            if (path == "/sys/fs/cgroup/fixture/memory.current") { value = current; return true; }
            return false;
        };
        skipped([&] { finite_cgroup(ledger); });
        limit = std::to_string(1ull << 30); finite_cgroup(ledger);
        current = std::to_string(768ull << 20); finite_cgroup(ledger); // pressure is tested by the fixture
        // A finite ancestor is sufficient; the immediate child may be unlimited.
        ledger.cgroup += "/child"; current = "0";
        auto ancestor = ledger.read;
        ledger.read = [&](const std::string & path, std::string & value) {
            if (path == "/sys/fs/cgroup/fixture/child/memory.max") { value = "max"; return true; }
            return ancestor(path, value);
        };
        finite_cgroup(ledger);
        // Missing real paths are I/O failures, never unsupported-filesystem skips.
        bool missing_failed = false;
        try { open_direct("/proc/self/ssdash-nonexistent-resource-fixture"); }
        catch (const unavailable &) { throw; }
        catch (const std::runtime_error &) { missing_failed = true; }
        require(missing_failed, "missing file must fail");
        std::puts("LINUX_TEST_RESOURCES_OK");
        return 0;
    } catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
