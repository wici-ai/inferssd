// SPDX-License-Identifier: Apache-2.0
// Every read ring this OS builds, against one fixture: many concurrent reads
// checked byte for byte, the end of file, a bad descriptor, drain, the
// counters, and fixed buffers. Windows runs both of its backends.
#include "io/pool-ring.h"
#include "platform/file.h"
#if defined(__linux__)
#include "io/linux-ring.h"
#endif
#if defined(_WIN32)
#include "io/win-ring.h"
#endif

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <new>
#include <random>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;
using ssd::io::buffer_span;
using ssd::io::completion;
using ssd::io::read_range;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

static unsigned char pattern(uint64_t i) {
    uint64_t x = (i + 1) * 0x9E3779B97F4A7C15ull;
    x ^= x >> 31;
    x *= 0xBF58476D1CE4E5B9ull;
    return static_cast<unsigned char>(x >> 56);
}
static bool matches(const unsigned char * data, size_t bytes, uint64_t offset) {
    for (size_t i = 0; i < bytes; ++i) {
        if (data[i] != pattern(offset + i)) { return false; }
    }
    return true;
}

struct aligned_buffer {
    explicit aligned_buffer(size_t bytes) : data(static_cast<unsigned char *>(::operator new(bytes, std::align_val_t(4096)))) {}
    ~aligned_buffer() { ::operator delete(data, std::align_val_t(4096)); }
    aligned_buffer(const aligned_buffer &) = delete;
    aligned_buffer & operator=(const aligned_buffer &) = delete;
    unsigned char * data;
};

struct fixture { int fd; uint64_t size; };
constexpr size_t block = 4096;
constexpr size_t slot_bytes = 64 * 1024;

// One expected read per tag, so any completion can be checked.
struct expectation { uint64_t offset = 0; size_t nbytes = 0; unsigned char * target = nullptr; bool busy = false; };

static int32_t expected_result(const fixture & f, const expectation & e) {
    return e.offset >= f.size ? 0 : int32_t(std::min<uint64_t>(e.nbytes, f.size - e.offset));
}

template<class Ring> bool check_completion(const fixture & f, std::vector<expectation> & expected, const completion & c) {
    if (c.user_data >= expected.size() || !expected[c.user_data].busy) { return false; }
    expectation & e = expected[c.user_data];
    e.busy = false;
    const int32_t want = expected_result(f, e);
    return c.res == want && matches(e.target, size_t(want), e.offset);
}

// Pops until count completions arrived, waiting when none is ready.
template<class Ring> bool collect(Ring & ring, const fixture & f, std::vector<expectation> & expected, unsigned count) {
    bool good = true;
    while (count) {
        completion c {};
        const auto popped = ring.pop(c);
        if (!popped) { return false; }
        if (popped.value) {
            good &= check_completion<Ring>(f, expected, c);
            --count;
        } else if (!ring.wait()) {
            return false;
        }
    }
    return good;
}

template<class Ring> void exercise(const char * label, const fixture & f, bool portable,
                                   const char * (*backend)(const Ring &)) {
    const unsigned depth = 16;
    aligned_buffer arena(depth * slot_bytes);
    auto slot = [&](size_t i) { return arena.data + i * slot_bytes; };
    const uint64_t blocks = f.size / block;
    const uint64_t last = blocks * block;
    std::mt19937_64 rng(7);
    {
        Ring ring;
        const auto started = ring.init(depth, false);
        if (!started) {
#if defined(__linux__)
            if (!portable) {
                std::printf("%s: skipped, ring unavailable here: %s\n", label, started.error.c_str());
                return;
            }
#endif
            std::fprintf(stderr, "%s: init failed: %s\n", label, started.error.c_str());
            ++failures;
            return;
        }
        std::printf("%s: backend=%s\n", label, backend(ring));
        CHECK(!ring.read_fixed());

        // Many reads in flight, each checked byte for byte; some cross the end.
        std::vector<expectation> expected(depth);
        const int total = 400;
        int issued = 0, done = 0;
        uint64_t inflight = 0;
        bool good = true;
        while (done < total) {
            unsigned queued = 0;
            for (unsigned i = 0; i < depth && issued < total; ++i) {
                if (expected[i].busy) { continue; }
                const uint64_t offset = (rng() % blocks) * block;
                const size_t nbytes = size_t(rng() % (slot_bytes / block) + 1) * block;
                expected[i] = {offset, nbytes, slot(i), true};
                good &= bool(ring.queue({f.fd, offset, nbytes}, slot(i), i, false));
                ++queued;
                ++issued;
            }
            good &= ring.pending() == inflight + queued;
            if (queued) {
                good &= bool(ring.submit(queued));
                inflight += queued;
            }
            bool any = false;
            for (;;) {
                completion c {};
                const auto popped = ring.pop(c);
                if (!popped) { good = false; break; }
                if (!popped.value) { break; }
                any = true;
                good &= check_completion<Ring>(f, expected, c);
                --inflight;
                ++done;
            }
            if (!good) { break; }
            if (!any && inflight) { good &= bool(ring.wait()); }
        }
        CHECK(good);
        CHECK(ring.pending() == 0);
        CHECK(ring.completed() == uint64_t(total));

        // The tail returns the bytes that exist; past the end returns 0.
        expected[0] = {last, 2 * block, slot(0), true};
        expected[1] = {last + 2 * block, block, slot(1), true};
        CHECK(ring.queue({f.fd, last, 2 * block}, slot(0), 0, false));
        CHECK(ring.queue({f.fd, last + 2 * block, block}, slot(1), 1, false));
        CHECK(ring.submit(2));
        CHECK(collect(ring, f, expected, 2));

        // A bad descriptor fails its own read, not the ring.
        CHECK(ring.queue({-1, 0, block}, slot(2), 2, false));
        CHECK(ring.submit(1));
        completion bad {};
        for (;;) {
            const auto popped = ring.pop(bad);
            if (!popped || popped.value) { CHECK(popped); break; }
            if (!ring.wait()) { CHECK(false); break; }
        }
        CHECK(bad.user_data == 2 && bad.res == -EBADF);

        // Drain submits what was only queued and hands every completion over.
        struct drained { const fixture * f; std::vector<expectation> * expected; unsigned count; bool good; };
        drained state {&f, &expected, 0, true};
        for (unsigned i = 0; i < 8; ++i) {
            const uint64_t offset = (rng() % blocks) * block;
            expected[i] = {offset, slot_bytes, slot(i), true};
            CHECK(ring.queue({f.fd, offset, slot_bytes}, slot(i), i, false));
        }
        CHECK(ring.submit(3));
        CHECK(ring.pending() == 8);
        CHECK(ring.drain(&state, [](void * userdata, const completion & c) {
            auto & s = *static_cast<drained *>(userdata);
            s.good &= check_completion<Ring>(*s.f, *s.expected, c);
            ++s.count;
        }));
        CHECK(state.count == 8 && state.good);
        CHECK(ring.pending() == 0);
        CHECK(ring.completed() == uint64_t(total) + 2 + 1 + 8);

        // Caller errors are refused, and leave the ring usable.
        CHECK(!ring.queue({f.fd, 0, block}, arena.data + 512, 0, false));
        CHECK(!ring.queue({f.fd, 0, block}, slot(0), 0, true));
        CHECK(!ring.register_buffers({{slot(0), slot_bytes}}));
        if (portable) { CHECK(!ring.wait()); }
        CHECK(ring.pending() == 0);
        ring.close_ring();
    }
    {
        // Depth one: a single reader, a single read in flight.
        Ring ring;
        CHECK(ring.init(1, false));
        std::vector<expectation> expected(1);
        expected[0] = {block, slot_bytes, slot(0), true};
        CHECK(ring.queue({f.fd, block, slot_bytes}, slot(0), 0, false));
        CHECK(ring.submit(1));
        CHECK(collect(ring, f, expected, 1));
        ring.close_ring();
    }
    {
        // Fixed buffers: register, read by index, then unregister and register
        // a resized set, as the sink does when its slots grow.
        Ring ring;
        CHECK(ring.init(8, true));
        CHECK(ring.read_fixed() && !ring.buffers_registered());
        std::vector<buffer_span> spans;
        for (unsigned i = 0; i < 8; ++i) { spans.push_back({slot(i), slot_bytes}); }
        CHECK(ring.register_buffers(spans));
        CHECK(ring.buffers_registered());
        CHECK(!ring.register_buffers(spans));
        std::vector<expectation> expected(8);
        for (unsigned round = 0; round < 3; ++round) {
            for (unsigned i = 0; i < 8; ++i) {
                const uint64_t offset = (rng() % blocks) * block;
                const size_t nbytes = size_t(rng() % (slot_bytes / block) + 1) * block;
                expected[i] = {offset, nbytes, slot(i), true};
                CHECK(ring.queue({f.fd, offset, nbytes}, slot(i), i, true));
            }
            CHECK(ring.submit(8));
            CHECK(collect(ring, f, expected, 8));
        }
        CHECK(!ring.queue({f.fd, 0, block}, slot(1), 0, true));            // tag 0 is slot 0
        CHECK(!ring.queue({f.fd, 0, slot_bytes + block}, slot(0), 0, true)); // beyond the buffer
        CHECK(ring.pending() == 0);
        CHECK(ring.unregister_buffers());
        CHECK(!ring.buffers_registered());
        spans.clear();
        for (unsigned i = 0; i < 4; ++i) { spans.push_back({slot(8 + 2 * i), 2 * slot_bytes}); }
        CHECK(ring.register_buffers(spans));
        std::vector<expectation> resized(4);
        for (unsigned i = 0; i < 4; ++i) {
            const uint64_t offset = (rng() % blocks) * block;
            resized[i] = {offset, 2 * slot_bytes, slot(8 + 2 * i), true};
            CHECK(ring.queue({f.fd, offset, 2 * slot_bytes}, slot(8 + 2 * i), i, true));
        }
        CHECK(ring.submit(4));
        CHECK(collect(ring, f, resized, 4));
        // Plain reads still work on a fixed ring.
        std::vector<expectation> plain(1);
        plain[0] = {0, block, slot(0), true};
        CHECK(ring.queue({f.fd, 0, block}, slot(0), 0, false));
        CHECK(ring.submit(1));
        CHECK(collect(ring, f, plain, 1));
        ring.close_ring();
    }
    std::printf("%s: done\n", label);
}

int main(int argc, char ** argv) {
    // An optional directory argument: direct I/O needs a disk filesystem.
    const fs::path base = argc > 1 ? fs::path(argv[1]) : fs::temp_directory_path();
    const fs::path dir = base / ("inferssd-read-ring-" + std::to_string(std::random_device{}()));
    fs::create_directories(dir);
    const fs::path path = dir / "fixture.bin";
    const uint64_t size = 4 * 1048576 + 1234;
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        std::vector<char> chunk(1 << 16);
        for (uint64_t at = 0; at < size; at += chunk.size()) {
            const size_t n = size_t(std::min<uint64_t>(chunk.size(), size - at));
            for (size_t i = 0; i < n; ++i) { chunk[i] = static_cast<char>(pattern(at + i)); }
            out.write(chunk.data(), std::streamsize(n));
        }
    }
    const auto text = path.u8string();
    std::string error;
    const int fd = ssd::platform::open_direct(std::string(text.begin(), text.end()).c_str(), &error);
    if (fd < 0) {
        std::fprintf(stderr, "open_direct failed: %s (pass a directory on a disk filesystem)\n", error.c_str());
        fs::remove_all(dir);
        return 1;
    }
    const fixture f {fd, size};

    exercise<ssd::io::pool_ring>("pool_ring", f, true, [](const ssd::io::pool_ring & r) { return r.backend_name(); });
#if defined(__linux__)
    exercise<ssd::io::linux_ring>("linux_ring", f, false, [](const ssd::io::linux_ring &) { return "io_uring"; });
#endif
#if defined(_WIN32)
    exercise<ssd::io::win_ring>("win_ring", f, true, [](const ssd::io::win_ring & r) { return r.backend_name(); });
    ssd::io::win_ring::force_fallback_for_testing(true);
    {
        ssd::io::win_ring probe;
        CHECK(probe.init(4, false));
        CHECK(std::strcmp(probe.backend_name(), "overlapped") == 0);
        probe.close_ring();
    }
    exercise<ssd::io::win_ring>("win_ring[fallback]", f, true, [](const ssd::io::win_ring & r) { return r.backend_name(); });
    ssd::io::win_ring::force_fallback_for_testing(false);
#endif

    ssd::platform::close_fd(fd);
    std::error_code ignored;
    fs::remove_all(dir, ignored);
    if (failures) {
        std::fprintf(stderr, "test-ssd-read-ring: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-ssd-read-ring: ok\n");
    return 0;
}
