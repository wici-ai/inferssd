// SPDX-License-Identifier: Apache-2.0
// The platform file layer on this OS: direct and buffered opens, positional
// reads against known bytes, identity, and error reporting.
#include "platform/file.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <new>
#include <random>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/stat.h>
#endif

namespace fs = std::filesystem;
using namespace ssd::platform;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

static unsigned char pattern(uint64_t i, uint32_t seed) {
    uint64_t x = (i + 1) * 0x9E3779B97F4A7C15ull ^ (uint64_t(seed) << 32);
    x ^= x >> 31;
    x *= 0xBF58476D1CE4E5B9ull;
    return static_cast<unsigned char>(x >> 56);
}

static bool matches(const unsigned char * data, size_t bytes, uint64_t offset, uint32_t seed) {
    for (size_t i = 0; i < bytes; ++i) {
        if (data[i] != pattern(offset + i, seed)) { return false; }
    }
    return true;
}

static void write_fixture(const fs::path & path, uint64_t size, uint32_t seed) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    std::vector<char> chunk(1 << 16);
    for (uint64_t at = 0; at < size; at += chunk.size()) {
        const size_t n = size_t(std::min<uint64_t>(chunk.size(), size - at));
        for (size_t i = 0; i < n; ++i) { chunk[i] = static_cast<char>(pattern(at + i, seed)); }
        out.write(chunk.data(), std::streamsize(n));
    }
}

static std::string utf8(const fs::path & path) {
    const auto text = path.u8string();
    return std::string(text.begin(), text.end());
}

struct aligned_buffer {
    explicit aligned_buffer(size_t bytes) : data(static_cast<unsigned char *>(::operator new(bytes, std::align_val_t(4096)))) {}
    ~aligned_buffer() { ::operator delete(data, std::align_val_t(4096)); }
    aligned_buffer(const aligned_buffer &) = delete;
    aligned_buffer & operator=(const aligned_buffer &) = delete;
    unsigned char * data;
};

int main(int argc, char ** argv) {
    // An optional directory argument: direct I/O needs a disk filesystem.
    const fs::path base = argc > 1 ? fs::path(argv[1]) : fs::temp_directory_path();
    const fs::path dir = base / ("inferssd-platform-file-" + std::to_string(std::random_device{}()));
    fs::create_directories(dir);

    const uint64_t size = 3 * 1048576 + 1234; // an unaligned tail
    const fs::path path_a = dir / "a.bin", path_b = dir / "b.bin";
    const fs::path path_u = dir / fs::u8path("\xE6\xA8\xA1\xE5\x9E\x8B-\xC3\xA9.bin");
    write_fixture(path_a, size, 1);
    write_fixture(path_b, 65536, 2);
    write_fixture(path_u, 12345, 3);

    std::string error;
    const int fd = open_direct(utf8(path_a).c_str(), &error);
    if (fd < 0) {
        std::fprintf(stderr, "open_direct failed: %s (pass a directory on a disk filesystem)\n", error.c_str());
        fs::remove_all(dir);
        return 1;
    }
    CHECK(is_direct(fd));
    const size_t alignment = direct_alignment(fd);
    CHECK(alignment >= 1 && !(alignment & (alignment - 1)) && alignment <= 65536);
#if defined(__linux__)
    CHECK(alignment == 4096);
#elif defined(__APPLE__)
    CHECK(alignment == 1);
#endif
    uint64_t bytes = 0;
    CHECK(file_size(fd, bytes) && bytes == size);

    // Random sector-aligned reads through the cache-bypassing descriptor.
    const size_t unit = std::max<size_t>(alignment, 4096);
    const uint64_t blocks = size / unit;
    aligned_buffer buffer(1 << 20);
    std::mt19937_64 rng(42);
    for (int i = 0; i < 300; ++i) {
        const uint64_t offset = (rng() % blocks) * unit;
        const size_t length = size_t(std::min<uint64_t>((rng() % 256 + 1) * unit, blocks * unit - offset));
        int code = -1;
        const bool ok = pread_exact(fd, buffer.data, length, offset, &code);
        CHECK(ok && code == 0);
        CHECK(ok && matches(buffer.data, length, offset, 1));
    }
    // Across and past the end of the file.
    const uint64_t last = blocks * unit;
    CHECK(pread_some(fd, buffer.data, 2 * unit, last) == int64_t(size - last));
    CHECK(matches(buffer.data, size_t(size - last), last, 1));
    CHECK(pread_some(fd, buffer.data, unit, last + unit) == 0);
    int code = -1;
    CHECK(!pread_exact(fd, buffer.data, 2 * unit, last, &code) && code == 0);

    // Concurrent positional reads on one descriptor.
    std::atomic<int> bad_reads{0};
    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t) {
        readers.emplace_back([&, t] {
            aligned_buffer local(256 * 1024);
            std::mt19937_64 local_rng(uint64_t(t) + 7);
            for (int i = 0; i < 100; ++i) {
                const uint64_t offset = (local_rng() % (blocks - 16)) * unit;
                const size_t length = size_t((local_rng() % 16 + 1) * unit);
                if (pread_some(fd, local.data, length, offset) != int64_t(length) || !matches(local.data, length, offset, 1)) {
                    ++bad_reads;
                }
            }
        });
    }
    for (auto & reader : readers) { reader.join(); }
    CHECK(bad_reads.load() == 0);

    // Buffered descriptors take any offset and length.
    const int fd_a = open_buffered(utf8(path_a).c_str(), &error);
    const int fd_b = open_buffered(utf8(path_b).c_str(), &error);
    CHECK(fd_a >= 0 && fd_b >= 0);
    CHECK(!is_direct(fd_a));
    std::vector<unsigned char> raw(70001);
    for (int i = 0; i < 300; ++i) {
        const uint64_t offset = rng() % (size - 1);
        const size_t length = size_t(std::min<uint64_t>(rng() % raw.size() + 1, size - offset));
        CHECK(pread_exact(fd_a, raw.data(), length, offset) && matches(raw.data(), length, offset, 1));
    }

    // One file through two descriptors has one identity; another file differs.
    file_identity direct_id, buffered_id, other_id;
    CHECK(identity(fd, direct_id) && identity(fd_a, buffered_id) && identity(fd_b, other_id));
    CHECK(direct_id == buffered_id);
    CHECK(direct_id.size == size && other_id.size == 65536);
    CHECK(direct_id.id_lo != other_id.id_lo || direct_id.id_hi != other_id.id_hi);
    CHECK(direct_id != other_id);
    CHECK(direct_id.mtime_ns > 0 && direct_id.ctime_ns > 0);
    file_identity again;
    CHECK(identity(fd, again) && again == direct_id);

    // A UTF-8 name with non-ASCII characters.
    const int fd_u = open_direct(utf8(path_u).c_str(), &error);
    CHECK(fd_u >= 0);
    if (fd_u >= 0) {
        CHECK(file_size(fd_u, bytes) && bytes == 12345);
        CHECK(pread_some(fd_u, buffer.data, 16384, 0) == 12345 && matches(buffer.data, 12345, 0, 3));
        close_fd(fd_u);
    }

    // Errors are reported, never raised.
    error.clear();
    errno = 0;
    CHECK(open_direct(utf8(dir / "missing.bin").c_str(), &error) < 0 && errno == ENOENT && !error.empty());
    CHECK(open_buffered(utf8(dir).c_str(), &error) < 0);
#if !defined(_WIN32)
    // A FIFO must neither block the open nor pass as a regular file.
    const fs::path fifo = dir / "fifo";
    if (mkfifo(fifo.c_str(), 0600) == 0) {
        errno = 0;
        CHECK(open_buffered(fifo.c_str(), &error) < 0 && errno == EINVAL);
        CHECK(open_direct(fifo.c_str(), &error) < 0);
    }
#endif
    file_identity unused;
    CHECK(pread_some(-1, buffer.data, 4096, 0) == -EBADF);
    CHECK(!identity(-1, unused) && !file_size(-1, bytes) && !is_direct(-1));

    close_fd(fd);
    CHECK(!is_direct(fd));
    CHECK(pread_some(fd, buffer.data, 4096, 0) == -EBADF);
    close_fd(fd_a);
    close_fd(fd_b);

    std::error_code ignored;
    fs::remove_all(dir, ignored);
    if (failures) {
        std::fprintf(stderr, "test-ssd-platform-file: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-ssd-platform-file: ok (alignment=%zu)\n", alignment);
    return 0;
}
