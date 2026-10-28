// SPDX-License-Identifier: Apache-2.0
// The test prerequisites classify a missing resource as a skip and everything else as a failure.
#include "test-resources.h"
#include <functional>

static void require(bool ok, const char * why) { if (!ok) { throw std::runtime_error(why); } }
static void skipped(const std::function<void()> & run) {
    try { run(); } catch (const test_resources::unavailable &) { return; }
    throw std::runtime_error("missing resource was not classified as unavailable");
}
static void failed(const std::function<void()> & run, const char * why) {
    bool failure = false;
    try { run(); } catch (const test_resources::unavailable &) { throw; }
    catch (const std::runtime_error &) { failure = true; }
    require(failure, why);
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
        const size_t cpus = process_cpus().size();
        require(cpus >= 1, "read actual process CPUs");
        cpu_threads(1, 0);
        if (cpus <= 128) { skipped([&] { cpu_threads(int(cpus), 1); }); }
        failed([&] { cpu_threads(0, 1); }, "invalid threads must fail");
        failed([&] { cpu_participants(1, 256); }, "invalid participants must fail");
        // A missing ring is the fixtures' skip; the checks here still run.
        try { io_ring(); } catch (const unavailable & e) { std::printf("read ring unavailable: %s\n", e.what()); }

#if defined(__linux__)
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
#else
        finite_envelope(); // always finite and readable on the native ledger
#endif

        // A fixture file: created empty and unique, read directly once it has an aligned page.
        const std::string path = temp_file("inferssd-test-resources"), other = temp_file("inferssd-test-resources");
        require(path != other && std::filesystem::is_regular_file(path) && std::filesystem::is_regular_file(other),
                "fixture files are not unique regular files");
        remove_file(other);
        {
            std::ofstream file(path, std::ios::binary | std::ios::trunc);
            const std::string page(4096, 'x');
            file.write(page.data(), std::streamsize(page.size()));
            require(bool(file), "fixture write failed");
        }
        bool direct = true;
        try {
            const int fd = open_direct(path);
            require(ssd::platform::is_direct(fd), "fixture open is not direct");
            ssd::platform::close_fd(fd);
        } catch (const unavailable & e) {
            direct = false;
            std::printf("fixture filesystem refuses direct I/O: %s\n", e.what());
        }
        if (direct) {
            std::filesystem::resize_file(path, 0);
            failed([&] { ssd::platform::close_fd(open_direct(path)); }, "an empty fixture must fail its header read");
        }
        remove_file(path);
        require(!std::filesystem::exists(path), "fixture file was not removed");
        // Missing paths and directories are I/O failures, never unsupported-filesystem skips.
        failed([&] { open_direct(path); }, "missing file must fail");
        failed([&] { open_direct(std::filesystem::temp_directory_path().string()); }, "a directory must fail");

        const uint64_t anonymous = anonymous_bytes(), resident = resident_bytes();
        require(anonymous > 0 && resident > 0, "process memory counters");
        const auto times = usage();
        require(times.user_us >= 0 && times.system_us >= 0, "process times");
        std::puts("TEST_RESOURCES_OK");
        return 0;
    } catch (const std::exception & error) { return test_resources::report("TEST_RESOURCES", error); }
}
