// SPDX-License-Identifier: Apache-2.0
// inferssd-ssdprobe: random-read bandwidth and latency of the drives that hold
// the given files, read the way the engine reads experts: uncached opens
// (platform::open_direct) and the engine's own read ring (io_uring on Linux,
// IoRing or overlapped reads on Windows, the reader pool on macOS). It only
// reads existing files; it never writes one.
//
// Direct reads on Linux and Windows always go to the drive. macOS F_NOCACHE
// reads still return pages that are already cached, so there the probe maps
// each file once, asks mincore which pages are resident and reads only extents
// that hold none of them.
//
// For each repeat, sizes ascend; within a size the QD ascends on odd repeats
// and descends on even ones. Each (size, QD) point keeps QD reads in flight at
// uniformly random aligned offsets of uniformly random files until its byte or
// time budget is spent. Latency is a read's submission to its completion.
// Prints one JSON object (schema inferssd-ssdprobe/1) on stdout.
#include "io/read-ring.h"
#include "platform/file.h"
#include "platform/memory.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#include <cwchar>
#endif
#if defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

constexpr const char * tool = "inferssd-ssdprobe";
constexpr int exit_usage = 1;  // bad arguments
constexpr int exit_file = 2;   // a file cannot be read directly or is too small
constexpr int exit_io = 3;     // reader setup or a read failed

constexpr uint64_t max_request_bytes = uint64_t(1) << 30;
constexpr uint64_t max_qd = 1024;

#if defined(__linux__)
constexpr const char * reader_name_fixed = "io_uring";
#endif
#if defined(__linux__) || defined(_WIN32)
constexpr unsigned reader_cap = 0;  // no cap
#else
constexpr unsigned reader_cap = 8;  // pool_ring runs at most 8 reader threads (pool-ring.cpp)
#endif

struct options {
    std::vector<std::string> files;
    std::vector<uint64_t> sizes{1048576, 4194304, 16777216};
    std::vector<uint64_t> qds{1, 2, 4, 8, 16, 32};
    uint64_t bytes_per_point = 2147483648;
    double max_seconds = 5.0;
    uint64_t repeats = 2;
    uint64_t seed = 1;
};

// Bytes [begin, end) of a file that reads may cover.
struct extent { uint64_t begin = 0, end = 0; };

struct source { std::string path; int fd = -1; uint64_t size = 0; std::vector<extent> readable; };

struct latency_us { double mean = 0, p50 = 0, p90 = 0, p99 = 0, max = 0; };
struct run { uint64_t bytes_read = 0; double seconds = 0, bps = 0; latency_us lat; };
struct point { uint64_t bytes = 0; unsigned qd = 0; std::vector<run> runs; };

[[noreturn]] void fail(int code, const std::string & why) {
    std::fprintf(stderr, "%s: %s\n", tool, why.c_str());
    std::exit(code);
}

void print_usage(std::FILE * out) {
    std::fprintf(out,
        "usage: %s --file PATH [--file PATH ...] [options]\n"
        "\n"
        "Measures random-read bandwidth and latency of the drives holding the files\n"
        "with the engine's own reader. Reads random aligned extents of the files and\n"
        "never writes. Linux and Windows reads bypass the page cache; macOS reads\n"
        "skip the extents that hold cached pages, since uncached reads there still\n"
        "return those pages. Prints one JSON object on stdout.\n"
        "\n"
        "  --file PATH                  a file to read; repeat for more files\n"
        "  --sizes N,N,...              request bytes, each a multiple of the direct-I/O\n"
        "                               alignment (default 1048576,4194304,16777216)\n"
        "  --qd N,N,...                 reads in flight (default 1,2,4,8,16,32); values\n"
        "                               above the reader's cap are skipped\n"
        "  --bytes-per-point N          end a point after N bytes (default 2147483648)\n"
        "  --max-seconds-per-point S    ... or after S seconds, whichever comes first\n"
        "                               (default 5)\n"
        "  --repeats N                  sweeps over all points (default 2)\n"
        "  --seed N                     seed of the random offsets (default 1)\n"
        "  --help                       show this help\n"
        "\n"
        "Exit status: 0 success, 1 bad arguments, 2 a file cannot be opened for\n"
        "direct reads or is smaller than the largest size (on macOS: no file has an\n"
        "uncached extent of the largest size), 3 a read failed.\n",
        tool);
}

bool parse_u64(const char * text, uint64_t & value) {
    if (!text || *text < '0' || *text > '9') { return false; }
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno || !end || *end) { return false; }
    value = uint64_t(parsed);
    return true;
}

// JSON strings are UTF-8, while POSIX paths may hold any bytes.
bool valid_utf8(const std::string & text) {
    for (size_t i = 0; i < text.size();) {
        const unsigned char lead = static_cast<unsigned char>(text[i]);
        size_t extra = 0;
        uint32_t code = 0;
        if (lead < 0x80) {
            ++i;
            continue;
        }
        if (lead >= 0xC2 && lead <= 0xDF) {
            extra = 1;
            code = lead & 0x1F;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            extra = 2;
            code = lead & 0x0F;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            extra = 3;
            code = lead & 0x07;
        } else {
            return false;
        }
        if (text.size() - i <= extra) { return false; }
        for (size_t k = 1; k <= extra; ++k) {
            const unsigned char next = static_cast<unsigned char>(text[i + k]);
            if ((next & 0xC0) != 0x80) { return false; }
            code = (code << 6) | (next & 0x3F);
        }
        if ((extra == 2 && code < 0x800) || (extra == 3 && (code < 0x10000 || code > 0x10FFFF)) ||
            (code >= 0xD800 && code <= 0xDFFF)) {
            return false;
        }
        i += extra + 1;
    }
    return true;
}

// Positive integers, sorted and deduplicated.
bool parse_list(const char * text, std::vector<uint64_t> & values) {
    values.clear();
    std::string all(text ? text : "");
    size_t begin = 0;
    for (;;) {
        const size_t comma = all.find(',', begin);
        const std::string item = all.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
        uint64_t value = 0;
        if (!parse_u64(item.c_str(), value) || !value) { return false; }
        values.push_back(value);
        if (comma == std::string::npos) { break; }
        begin = comma + 1;
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return !values.empty();
}

options parse(const std::vector<std::string> & args) {
    options opt;
    for (size_t i = 1; i < args.size(); ++i) {
        const std::string & name = args[i];
        if (name == "--help" || name == "-h") {
            print_usage(stdout);
            std::exit(0);
        }
        if (i + 1 >= args.size()) { fail(exit_usage, "unknown option or missing value: " + name + " (see --help)"); }
        const char * value = args[++i].c_str();
        uint64_t number = 0;
        if (name == "--file") {
            if (!*value) { fail(exit_usage, "--file needs a path"); }
            if (!valid_utf8(value)) { fail(exit_usage, "--file path is not valid UTF-8"); }
            opt.files.emplace_back(value);
        } else if (name == "--sizes") {
            if (!parse_list(value, opt.sizes)) { fail(exit_usage, "--sizes takes positive integers separated by commas"); }
        } else if (name == "--qd") {
            if (!parse_list(value, opt.qds)) { fail(exit_usage, "--qd takes positive integers separated by commas"); }
        } else if (name == "--bytes-per-point") {
            if (!parse_u64(value, number) || !number) { fail(exit_usage, "--bytes-per-point takes a positive integer"); }
            opt.bytes_per_point = number;
        } else if (name == "--max-seconds-per-point") {
            char * end = nullptr;
            const double seconds = std::strtod(value, &end);
            if (!*value || !end || *end || !std::isfinite(seconds) || seconds <= 0 || seconds > 86400) {
                fail(exit_usage, "--max-seconds-per-point takes a positive number of seconds up to 86400");
            }
            opt.max_seconds = seconds;
        } else if (name == "--repeats") {
            if (!parse_u64(value, number) || !number) { fail(exit_usage, "--repeats takes a positive integer"); }
            opt.repeats = number;
        } else if (name == "--seed") {
            if (!parse_u64(value, number)) { fail(exit_usage, "--seed takes a non-negative integer"); }
            opt.seed = number;
        } else {
            fail(exit_usage, "unknown option: " + name + " (see --help)");
        }
    }
    if (opt.files.empty()) { fail(exit_usage, "at least one --file is required (see --help)"); }
    for (uint64_t size : opt.sizes) {
        if (size % 4096 || size > max_request_bytes) {
            fail(exit_usage, "--sizes: " + std::to_string(size) + " must be a multiple of 4096 and at most " +
                             std::to_string(max_request_bytes));
        }
    }
    if (opt.qds.back() > max_qd) { fail(exit_usage, "--qd: at most " + std::to_string(max_qd)); }
    return opt;
}

// splitmix64: the same offsets for the same seed on every OS and standard library.
struct random_source {
    uint64_t state;
    explicit random_source(uint64_t seed) : state(seed) {}
    uint64_t next() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    // Uniform in [0, n), without modulo bias.
    uint64_t below(uint64_t n) {
        const uint64_t limit = std::numeric_limits<uint64_t>::max() - std::numeric_limits<uint64_t>::max() % n;
        uint64_t x;
        do { x = next(); } while (x >= limit);
        return x % n;
    }
};

// Each point gets its own stream, so its offsets do not depend on where the
// budgets of earlier points cut their streams.
uint64_t point_seed(uint64_t seed, uint64_t repeat, uint64_t bytes, uint64_t qd) {
    random_source mix(seed);
    uint64_t s = mix.next() ^ repeat;
    s = random_source(s).next() ^ bytes;
    s = random_source(s).next() ^ qd;
    return random_source(s).next();
}

// Linear interpolation between the closest ranks.
double percentile(const std::vector<double> & sorted, double p) {
    const double rank = p / 100.0 * double(sorted.size() - 1);
    const size_t lo = size_t(rank);
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - double(lo));
}

// Random reads of one size: a uniformly random file among those with room for
// one, then a uniformly random aligned offset whose read stays inside one
// readable extent of that file. With one extent per file (every OS but macOS,
// or no cached pages) this is a uniform offset over the whole file.
struct offset_sampler {
    struct candidates {
        size_t file = 0;
        std::vector<uint64_t> starts;  // first aligned offset in each extent with room
        std::vector<uint64_t> ends;    // running count of offsets up to that extent
    };
    std::vector<candidates> files;
    uint64_t alignment = 1;

    offset_sampler(const std::vector<source> & sources, uint64_t bytes, uint64_t align) : alignment(align) {
        for (size_t f = 0; f < sources.size(); ++f) {
            candidates c;
            c.file = f;
            uint64_t count = 0;
            for (const extent & e : sources[f].readable) {
                const uint64_t first = (e.begin + align - 1) / align * align;
                if (first > e.end || e.end - first < bytes) { continue; }
                count += (e.end - first - bytes) / align + 1;
                c.starts.push_back(first);
                c.ends.push_back(count);
            }
            if (count) { files.push_back(std::move(c)); }
        }
    }
    bool has(size_t file) const {
        for (const candidates & c : files) {
            if (c.file == file) { return true; }
        }
        return false;
    }
    void draw(random_source & random, size_t & file, uint64_t & offset) const {
        const candidates & c = files[size_t(random.below(files.size()))];
        const uint64_t k = random.below(c.ends.back());
        const size_t i = size_t(std::upper_bound(c.ends.begin(), c.ends.end(), k) - c.ends.begin());
        file = c.file;
        offset = c.starts[i] + (k - (i ? c.ends[i - 1] : 0)) * alignment;
    }
};

#if defined(__APPLE__)
// The runs of pages of a file that are not in the page cache, and the cached
// bytes. Mapping and mincore fault nothing in.
bool uncached_extents(const source & file, std::vector<extent> & out, uint64_t & cached, std::string & error) {
    const long page_size = sysconf(_SC_PAGESIZE);
    const uint64_t page = page_size > 0 ? uint64_t(page_size) : 4096;
    constexpr uint64_t window = uint64_t(1) << 30;  // a multiple of every page size
    std::vector<char> pages;
    out.clear();
    cached = 0;
    uint64_t run_begin = 0;
    bool in_run = false;
    for (uint64_t base = 0; base < file.size; base += window) {
        const size_t length = size_t(std::min(window, file.size - base));
        void * view = mmap(nullptr, length, PROT_READ, MAP_SHARED, file.fd, off_t(base));
        if (view == MAP_FAILED) {
            error = "cannot map " + file.path + " to find its cached pages: " + std::strerror(errno);
            return false;
        }
        pages.assign(size_t((length + page - 1) / page), 0);
        const int status = mincore(static_cast<caddr_t>(view), length, pages.data());
        const int code = errno;
        munmap(view, length);
        if (status != 0) {
            error = "cannot find the cached pages of " + file.path + ": mincore: " + std::strerror(code);
            return false;
        }
        for (size_t i = 0; i < pages.size(); ++i) {
            const uint64_t at = base + uint64_t(i) * page;
            if (pages[i] & MINCORE_INCORE) {
                cached += std::min(page, file.size - at);
                if (in_run) { out.push_back({run_begin, at}); }
                in_run = false;
            } else if (!in_run) {
                run_begin = at;
                in_run = true;
            }
        }
    }
    if (in_run) { out.push_back({run_begin, file.size}); }
    return true;
}
#endif

void ignore_completion(void *, const ssd::io::completion &) {}

struct measurement {
    ssd::io::read_ring & ring;
    const std::vector<source> & files;
    unsigned char * buffers;  // qd slots of `stride` bytes
    uint64_t stride;
    const options & opt;
};

// Keeps qd reads of `bytes` in flight until the byte or time budget is spent,
// then waits for the reads still in flight. On failure the ring is drained.
bool measure(const measurement & m, const offset_sampler & offsets, uint64_t bytes, unsigned qd, uint64_t seed,
             run & out, std::string & error) {
    using clock = std::chrono::steady_clock;
    struct slot { clock::time_point submitted; size_t file = 0; uint64_t offset = 0; };
    std::vector<slot> slots(qd);
    std::vector<unsigned> idle, batch;
    idle.reserve(qd);
    batch.reserve(qd);
    for (unsigned i = qd; i-- > 0;) { idle.push_back(i); }
    std::vector<double> latencies;
    latencies.reserve(size_t(std::min<uint64_t>(m.opt.bytes_per_point / bytes + qd, 1u << 20)));
    random_source random(seed);

    auto abandon = [&](std::string why) {
        auto drained = m.ring.drain(nullptr, ignore_completion);
        if (!drained) { why += "; draining the read ring failed: " + drained.error; }
        error = std::move(why);
        return false;
    };

    const clock::time_point start = clock::now();
    const clock::time_point deadline =
        start + std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(m.opt.max_seconds));
    clock::time_point last = start;
    uint64_t issued = 0, done = 0;
    unsigned inflight = 0;
    bool issuing = true;
    for (;;) {
        // At least one read, so every point has a result.
        if (issuing && issued && (issued >= m.opt.bytes_per_point || clock::now() >= deadline)) { issuing = false; }
        batch.clear();
        while (issuing && !idle.empty() && issued < m.opt.bytes_per_point) {
            const unsigned s = idle.back();
            size_t f = 0;
            uint64_t offset = 0;
            offsets.draw(random, f, offset);
            auto queued = m.ring.queue({m.files[f].fd, offset, size_t(bytes)}, m.buffers + s * m.stride, s, false);
            if (!queued) { return abandon("queueing a read failed: " + queued.error); }
            idle.pop_back();
            slots[s].file = f;
            slots[s].offset = offset;
            batch.push_back(s);
            issued += bytes;
        }
        if (!batch.empty()) {
            const clock::time_point submitted = clock::now();
            for (unsigned s : batch) { slots[s].submitted = submitted; }
            auto result = m.ring.submit(unsigned(batch.size()));
            if (!result) { return abandon("submitting reads failed: " + result.error); }
            inflight += unsigned(batch.size());
        }
        if (!inflight) { break; }
        bool completed = false;
        for (;;) {
            ssd::io::completion entry{};
            auto popped = m.ring.pop(entry);
            if (!popped) { return abandon("collecting a completion failed: " + popped.error); }
            if (!popped.value) { break; }
            const clock::time_point now = clock::now();
            completed = true;
            --inflight;
            if (entry.user_data >= qd) { return abandon("the read ring returned an unknown read"); }
            const slot & read = slots[size_t(entry.user_data)];
            const std::string & path = m.files[read.file].path;
            if (entry.res < 0) {
                return abandon("read failed: " + path + " at offset " + std::to_string(read.offset) + ": " +
                               std::strerror(-entry.res));
            }
            if (uint64_t(entry.res) != bytes) {
                return abandon("short read: " + path + " at offset " + std::to_string(read.offset) + ": " +
                               std::to_string(entry.res) + " of " + std::to_string(bytes) + " bytes");
            }
            latencies.push_back(std::chrono::duration<double, std::micro>(now - read.submitted).count());
            done += bytes;
            last = now;
            idle.push_back(unsigned(entry.user_data));
        }
        if (!completed) {
            auto waited = m.ring.wait();
            if (!waited) { return abandon("waiting for a completion failed: " + waited.error); }
        }
    }

    std::sort(latencies.begin(), latencies.end());
    double sum = 0;
    for (double value : latencies) { sum += value; }
    out.bytes_read = done;
    out.seconds = std::max(std::chrono::duration<double>(last - start).count(), 1e-9);
    out.bps = double(done) / out.seconds;
    out.lat.mean = sum / double(latencies.size());
    out.lat.p50 = percentile(latencies, 50);
    out.lat.p90 = percentile(latencies, 90);
    out.lat.p99 = percentile(latencies, 99);
    out.lat.max = latencies.back();
    return true;
}

void print_json_string(const std::string & text) {
    std::fputc('"', stdout);
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') {
            std::fputc('\\', stdout);
            std::fputc(c, stdout);
        } else if (c < 0x20) {
            std::fprintf(stdout, "\\u%04x", unsigned(c));
        } else {
            std::fputc(c, stdout);
        }
    }
    std::fputc('"', stdout);
}

void print_report(const char * reader, uint64_t alignment, const std::vector<source> & files,
                  const std::vector<point> & points) {
    std::printf("{\"schema\": \"inferssd-ssdprobe/1\",\n \"reader\": ");
    print_json_string(reader);
    std::printf(", \"alignment_bytes\": %llu, \"qd_cap\": ", static_cast<unsigned long long>(alignment));
    if (reader_cap) { std::printf("%u", reader_cap); } else { std::printf("null"); }
    std::printf(",\n \"files\": [");
    for (size_t i = 0; i < files.size(); ++i) {
        std::printf("%s{\"path\": ", i ? ",\n           " : "");
        print_json_string(files[i].path);
        std::printf(", \"size_bytes\": %llu}", static_cast<unsigned long long>(files[i].size));
    }
    std::printf("],\n \"points\": [");
    for (size_t i = 0; i < points.size(); ++i) {
        const point & p = points[i];
        std::printf("%s{\"bytes\": %llu, \"qd\": %u, \"runs\": [", i ? ",\n            " : "",
                    static_cast<unsigned long long>(p.bytes), p.qd);
        for (size_t r = 0; r < p.runs.size(); ++r) {
            const run & v = p.runs[r];
            std::printf("%s{\"bytes_read\": %llu, \"seconds\": %.9g, \"Bps\": %.6g, "
                        "\"lat_us\": {\"mean\": %.6g, \"p50\": %.6g, \"p90\": %.6g, \"p99\": %.6g, \"max\": %.6g}}",
                        r ? ", " : "", static_cast<unsigned long long>(v.bytes_read), v.seconds, v.bps,
                        v.lat.mean, v.lat.p50, v.lat.p90, v.lat.p99, v.lat.max);
        }
        std::printf("]}");
    }
    std::printf("]}\n");
}

std::vector<std::string> arguments(int argc, char ** argv) {
#if defined(_WIN32)
    // The CRT's argv is in the ANSI code page; open_direct takes UTF-8 paths.
    int count = 0;
    if (LPWSTR * wide = CommandLineToArgvW(GetCommandLineW(), &count)) {
        std::vector<std::string> out;
        for (int i = 0; i < count; ++i) {
            const int length = int(std::wcslen(wide[i]));
            const int bytes = length ? WideCharToMultiByte(CP_UTF8, 0, wide[i], length, nullptr, 0, nullptr, nullptr) : 0;
            std::string value(size_t(std::max(bytes, 0)), '\0');
            if (bytes > 0) { WideCharToMultiByte(CP_UTF8, 0, wide[i], length, &value[0], bytes, nullptr, nullptr); }
            out.push_back(std::move(value));
        }
        LocalFree(wide);
        return out;
    }
#endif
    return std::vector<std::string>(argv, argv + argc);
}

} // namespace

int main(int argc, char ** argv) {
    const options opt = parse(arguments(argc, argv));
    const uint64_t largest = opt.sizes.back();

    std::vector<source> files;
    uint64_t alignment = 4096;
    for (const std::string & path : opt.files) {
        std::string why;
        source file;
        file.path = path;
        file.fd = ssd::platform::open_direct(path.c_str(), &why);
        if (file.fd < 0) { fail(exit_file, "cannot open for direct reads: " + (why.empty() ? path : why)); }
        if (!ssd::platform::file_size(file.fd, file.size)) { fail(exit_file, "cannot read the size of " + path); }
        if (file.size < largest) {
            fail(exit_file, path + " is " + std::to_string(file.size) + " bytes, smaller than the largest size " +
                            std::to_string(largest));
        }
        alignment = std::max<uint64_t>(alignment, ssd::platform::direct_alignment(file.fd));
        files.push_back(std::move(file));
    }
    for (uint64_t size : opt.sizes) {
        if (size % alignment) {
            fail(exit_usage, "--sizes: " + std::to_string(size) + " is not a multiple of the direct-I/O alignment " +
                             std::to_string(alignment));
        }
    }

    for (source & file : files) {
#if defined(__APPLE__)
        uint64_t cached = 0;
        std::string why;
        if (!uncached_extents(file, file.readable, cached, why)) { fail(exit_file, why); }
        const double share = double(cached) / double(file.size);
        std::fprintf(stderr, "%s: %s: %.2f%% of the file is in the page cache%s\n", tool, file.path.c_str(),
                     100.0 * share, share > 0.01 ? "; reads skip those pages, so fewer extents are measured" : "");
#else
        file.readable.push_back({0, file.size});
#endif
    }
    std::vector<offset_sampler> samplers;
    for (uint64_t size : opt.sizes) { samplers.emplace_back(files, size, alignment); }
    // Sizes ascend, so a file with room for the largest has room for all.
    if (samplers.back().files.empty()) {
        fail(exit_file, "no file has an extent of " + std::to_string(largest) +
                        " bytes outside the page cache, so reads would measure memory instead of the drive");
    }
    for (size_t s = 0; s < samplers.size(); ++s) {
        for (size_t f = 0; f < files.size(); ++f) {
            if (!samplers[s].has(f)) {
                std::fprintf(stderr, "%s: note: %s has no uncached extent of %llu bytes; not read at that size\n",
                             tool, files[f].path.c_str(), static_cast<unsigned long long>(opt.sizes[s]));
            }
        }
    }

    std::vector<unsigned> qds;
    for (uint64_t qd : opt.qds) {
        if (reader_cap && qd > reader_cap) {
            std::fprintf(stderr, "%s: skipping qd %llu: the reader runs at most %u reads at once\n", tool,
                         static_cast<unsigned long long>(qd), reader_cap);
            continue;
        }
        qds.push_back(unsigned(qd));
    }
    if (qds.empty()) { fail(exit_usage, "no --qd value within the reader's cap of " + std::to_string(reader_cap)); }
    for (uint64_t size : opt.sizes) {
        for (unsigned qd : qds) {
            const uint64_t reads = (opt.bytes_per_point + size - 1) / size;
            if (reads < 2 * uint64_t(qd)) {
                std::fprintf(stderr, "%s: note: bytes %llu qd %u: --bytes-per-point allows only %llu reads, %s\n",
                             tool, static_cast<unsigned long long>(size), qd, static_cast<unsigned long long>(reads),
                             reads < qd ? "so the point never has that many reads in flight"
                                        : "under two rounds, so the point is mostly ramp-up and drain");
            }
        }
    }

    // One target slot per read in flight, reused by every point, touched once
    // so that page faults stay out of the measurement.
    const unsigned depth = qds.back();
    const uint64_t buffer_bytes = uint64_t(depth) * largest;
    auto * buffers = static_cast<unsigned char *>(ssd::platform::aligned_alloc_bytes(size_t(alignment), size_t(buffer_bytes)));
    if (!buffers) { fail(exit_io, "cannot allocate " + std::to_string(buffer_bytes) + " bytes of read buffers"); }
    std::memset(buffers, 0, size_t(buffer_bytes));

    std::vector<point> points;
    std::string reader;
    {
        ssd::io::read_ring ring;
        auto ready = ring.init(depth, false);
        if (!ready) { fail(exit_io, "cannot start the read ring: " + ready.error); }
#if defined(__linux__)
        reader = reader_name_fixed;
#else
        reader = ring.backend_name();
#endif
        for (uint64_t size : opt.sizes) {
            for (unsigned qd : qds) { points.push_back({size, qd, {}}); }
        }
        const measurement m{ring, files, buffers, largest, opt};
        for (uint64_t repeat = 1; repeat <= opt.repeats; ++repeat) {
            for (size_t s = 0; s < opt.sizes.size(); ++s) {
                for (size_t k = 0; k < qds.size(); ++k) {
                    const size_t q = repeat % 2 ? k : qds.size() - 1 - k;
                    point & p = points[s * qds.size() + q];
                    run result;
                    std::string error;
                    if (!measure(m, samplers[s], p.bytes, p.qd, point_seed(opt.seed, repeat, p.bytes, p.qd), result,
                                 error)) {
                        fail(exit_io, error);
                    }
                    p.runs.push_back(result);
                    std::fprintf(stderr, "%s: repeat %llu/%llu bytes %llu qd %u: %.4g B/s, p50 %.1f us, p99 %.1f us\n",
                                 tool, static_cast<unsigned long long>(repeat),
                                 static_cast<unsigned long long>(opt.repeats), static_cast<unsigned long long>(p.bytes),
                                 p.qd, result.bps, result.lat.p50, result.lat.p99);
                }
            }
        }
    }
    ssd::platform::aligned_free(buffers);
    for (const source & file : files) { ssd::platform::close_fd(file.fd); }

    print_report(reader.c_str(), alignment, files, points);
    return std::fflush(stdout) == 0 && !std::ferror(stdout) ? 0 : exit_io;
}
