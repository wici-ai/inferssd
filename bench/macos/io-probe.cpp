#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unistd.h>
#include <vector>

struct request { int fd; uint64_t offset; size_t bytes; };
int main(int argc, char ** argv) {
    try {
        if (argc != 5) { throw std::runtime_error("usage: io-probe sequential|trace FILE DEPTH MAX_BYTES"); }
        const int depth = std::stoi(argv[3]);
        const uint64_t maximum = std::stoull(argv[4]);
        if (depth < 1 || depth > 4 || maximum == 0) { throw std::runtime_error("invalid limits"); }
        std::unordered_map<std::string, int> files;
        auto open_source = [&](const std::string & path) {
            auto found = files.find(path);
            if (found != files.end()) { return found->second; }
            const int fd = open(path.c_str(), O_RDONLY);
            if (fd < 0 || fcntl(fd, F_NOCACHE, 1) < 0) { throw std::runtime_error("cannot open uncached source"); }
            files.emplace(path, fd);
            return fd;
        };
        std::vector<request> requests;
        uint64_t total = 0;
        if (std::strcmp(argv[1], "sequential") == 0) {
            const int fd = open_source(argv[2]);
            for (uint64_t offset = uint64_t(1) << 30; total < maximum;) {
                const size_t bytes = size_t(std::min(maximum - total, uint64_t(8) << 20));
                requests.push_back({fd, offset, bytes});
                offset += bytes;
                total += bytes;
            }
        } else if (std::strcmp(argv[1], "trace") == 0) {
            std::ifstream input(argv[2]);
            std::string line;
            if (!std::getline(input, line) || line != "source_path\toffset\tread_bytes\tlogical_bytes\tbegin_ns\tduration_ns") {
                throw std::runtime_error("invalid read trace header");
            }
            while (total < maximum && std::getline(input, line)) {
                std::istringstream row(line);
                std::string path, offset, bytes;
                if (!std::getline(row, path, '\t') || !std::getline(row, offset, '\t') || !std::getline(row, bytes, '\t')) {
                    throw std::runtime_error("invalid read trace row");
                }
                const auto n = std::stoull(bytes);
                if (n == 0 || n > (uint64_t(64) << 20)) { throw std::runtime_error("invalid read extent"); }
                if (total + n > maximum) { break; }
                requests.push_back({open_source(path), std::stoull(offset), size_t(n)});
                total += n;
            }
        } else { throw std::runtime_error("unknown probe mode"); }
        if (requests.empty()) { throw std::runtime_error("no read requests"); }
        size_t largest = 0;
        for (const auto & item : requests) { largest = std::max(largest, item.bytes); }
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::vector<double> elapsed(requests.size());
        const auto start = std::chrono::steady_clock::now();
        std::vector<std::thread> workers;
        for (int i = 0; i < depth; ++i) {
            workers.emplace_back([&] {
                void * memory = nullptr;
                if (posix_memalign(&memory, 4096, largest) != 0) { failed = true; return; }
                while (!failed.load()) {
                    const size_t index = next.fetch_add(1);
                    if (index >= requests.size()) { break; }
                    const auto & item = requests[index];
                    size_t done = 0;
                    const auto begin = std::chrono::steady_clock::now();
                    while (done < item.bytes) {
                        const ssize_t count = pread(item.fd, static_cast<char *>(memory) + done, item.bytes - done, off_t(item.offset + done));
                        if (count < 0 && errno == EINTR) { continue; }
                        if (count <= 0) { failed = true; break; }
                        done += size_t(count);
                    }
                    elapsed[index] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
                }
                free(memory);
            });
        }
        for (auto & worker : workers) { worker.join(); }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        for (const auto & file : files) { close(file.second); }
        if (failed) { throw std::runtime_error("read failed or allocation unavailable"); }
        std::sort(elapsed.begin(), elapsed.end());
        double sum = 0;
        for (double value : elapsed) { sum += value; }
        auto percentile = [&](double p) { return elapsed[size_t(p * (elapsed.size() - 1))]; };
        std::printf("{\"mode\":\"%s\",\"depth\":%d,\"requests\":%zu,\"bytes\":%llu,\"seconds\":%.9f,\"logical_gbps\":%.6f,\"read_call_ms\":%.3f,\"p50_us\":%.3f,\"p95_us\":%.3f,\"p99_us\":%.3f}\n",
                argv[1], depth, requests.size(), (unsigned long long)total, seconds, total / seconds / 1e9,
                sum / 1000, percentile(.5), percentile(.95), percentile(.99));
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
