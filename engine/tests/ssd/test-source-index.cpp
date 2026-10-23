// SPDX-License-Identifier: MIT
#include "ggml-ssd/store/index.h"
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

[[maybe_unused]] static void require(bool ok, const char * message) { if (!ok) { throw std::runtime_error(message); } }

int main() {
#if !defined(__linux__)
    return 77;
#else
    char pattern[] = "/tmp/ssdash-source-index-XXXXXX";
    require(mkdtemp(pattern), "temporary directory failed");
    const std::filesystem::path directory(pattern);
    std::vector<std::string> aliases;
    for (int seed = 1; seed <= 2; ++seed) {
        const auto data = directory / (std::to_string(seed) + ".bin");
        const auto alias = directory / (std::to_string(seed) + ".tsv");
        std::ofstream stream(data, std::ios::binary);
        for (size_t i = 0; i < 16384; ++i) { stream.put(char((i + seed) % 251)); }
        stream.close();
        std::ofstream(alias) << "tensor\texpert_idx\tnbytes\tsource_path\toffset\n"
            << "blk.0.ffn_up_exps.weight\t3\t5000\t" << data.string() << "\t4097\n";
        aliases.push_back(alias.string());
    }
    auto first = ssd::store::source_index::open({aliases[0]});
    auto second = ssd::store::source_index::open({aliases[1]});
    require(bool(first) && bool(second), "direct source creation failed");
    const auto a = first.value->view(), b = second.value->view();
    int fd_a = -1, fd_b = -1;
    uint64_t offset = 0;
    size_t bytes = 0;
    const char * name = "blk.0.ffn_up_exps.weight";
    require(a.resolve(a.userdata, name, 3, 5000, 4, &fd_a, &offset, &bytes), "first source lookup failed");
    require(b.resolve(b.userdata, name, 3, 5000, 4, &fd_b, &offset, &bytes), "second source lookup failed");
    require(fd_a != fd_b && offset == 4097 && bytes == 5000, "source contexts shared a descriptor or changed logical range");
    require(!b.resolve(b.userdata, name, 3, 5000, 3, &fd_b, &offset, &bytes), "expert-count bound ignored");
    require(!b.resolve(b.userdata, name, 3, 5001, 4, &fd_b, &offset, &bytes), "tensor byte bound ignored");
    first.value.reset();
    errno = 0;
    require(fcntl(fd_a, F_GETFD) == -1 && errno == EBADF, "source descriptor survived its owner");
    require(fcntl(fd_b, F_GETFL) & O_DIRECT, "second context lost direct I/O");
    void * buffer = nullptr;
    require(posix_memalign(&buffer, 4096, 8192) == 0, "aligned buffer failed");
    std::unique_ptr<void, decltype(&std::free)> owned(buffer, std::free);
    require(pread(fd_b, buffer, 8192, 4096) == 8192, "actual direct read failed");
    const auto * payload = static_cast<unsigned char *>(buffer);
    for (size_t i = 0; i < 5000; ++i) { require(payload[i + 1] == (4097 + i + 2) % 251, "direct payload changed"); }
    second.value.reset();
    const auto bad = directory / "bad.tsv";
    std::ofstream(bad) << "tensor\texpert_idx\tnbytes\tsource_path\toffset\n"
        << name << "\t3\t5000\t" << (directory / "1.bin").string() << "\t16000\n";
    require(!ssd::store::source_index::open({bad.string()}), "out-of-file source published");
    require(!ssd::store::source_index::open({aliases[0], (directory / "missing.tsv").string()}), "partial source set published");
    const auto fifo = directory / "fifo";
    require(mkfifo(fifo.c_str(), 0600) == 0, "FIFO creation failed");
    const pid_t child = fork();
    require(child >= 0, "FIFO guard fork failed");
    if (!child) {
        alarm(3);
        if (ssd::store::source_index::open({fifo.string()})) { _exit(1); }
        std::ofstream(bad) << "tensor\texpert_idx\tnbytes\tsource_path\toffset\n"
            << name << "\t3\t5000\t" << fifo.string() << "\t0\n";
        _exit(ssd::store::source_index::open({bad.string()}) ? 2 : 0);
    }
    int child_status = 0;
    require(waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) && !WEXITSTATUS(child_status),
        "non-regular index/payload blocked startup or was published");
    std::filesystem::remove_all(directory);
    std::puts("SSD_CONTEXT_SOURCE_DIRECT_IO_PASS");
#endif
}
