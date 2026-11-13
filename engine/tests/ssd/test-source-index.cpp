// SPDX-License-Identifier: Apache-2.0
#include "ggml-ssd/store/index.h"
#include "ggml-ssd/store/legacy-index.h"
#include "ggml-ssd/platform/file.h"
#include "ggml-ssd/platform/memory.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

static void require(bool ok, const char * message) { if (!ok) { throw std::runtime_error(message); } }

int main() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / ("inferssd-source-index-" + std::to_string(stamp));
    require(std::filesystem::create_directory(directory), "temporary directory failed");
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
    uint64_t size = 0;
    require(!ssd::platform::file_size(fd_a, size), "source descriptor survived its owner");
    require(ssd::platform::is_direct(fd_b), "second context lost direct I/O");
    void * buffer = ssd::platform::aligned_alloc_bytes(4096, 8192);
    require(buffer != nullptr, "aligned buffer failed");
    std::unique_ptr<void, decltype(&ssd::platform::aligned_free)> owned(buffer, ssd::platform::aligned_free);
    require(ssd::platform::pread_exact(fd_b, buffer, 8192, 4096), "actual direct read failed");
    const auto * payload = static_cast<unsigned char *>(buffer);
    for (size_t i = 0; i < 5000; ++i) { require(payload[i + 1] == (4097 + i + 2) % 251, "direct payload changed"); }
    second.value.reset();
    const auto bad = directory / "bad.tsv";
    std::ofstream(bad) << "tensor\texpert_idx\tnbytes\tsource_path\toffset\n"
        << name << "\t3\t5000\t" << (directory / "1.bin").string() << "\t16000\n";
    require(!ssd::store::source_index::open({bad.string()}), "out-of-file source published");
    require(!ssd::store::source_index::open({aliases[0], (directory / "missing.tsv").string()}), "partial source set published");
    // A valid numeric prefix must not silently select a different expert or byte range.
    struct numeric_row { const char * expert; const char * size; const char * offset; };
    for (const auto & row : std::vector<numeric_row>{
            {"3tail", "5000", "4097"}, {"junk", "5000", "4097"},
            {"4294967296", "5000", "4097"}, {"18446744073709551616", "5000", "4097"},
            {"3", "5000tail", "4097"}, {"3", "5000", "4097tail"},
            {"3", "5000", "junk"}, {"3", "5000", "-0"},
            {"3", "18446744073709551616", "0"}, {"3", "5000", "18446744073709551616"}}) {
        std::ofstream(bad) << "tensor\texpert_idx\tnbytes\tsource_path\toffset\n"
            << name << '\t' << row.expert << '\t' << row.size << '\t'
            << (directory / "1.bin").string() << '\t' << row.offset << '\n';
        require(!ssd::store::source_index::open({bad.string()}), "invalid numeric source index published");
    }
    std::ofstream(bad) << "tensor\texpert_idx\tnbytes\tsource_path\toffset\n"
        << name << "\t0003\t05000\t" << (directory / "1.bin").string() << "\t04097\n";
    require(bool(ssd::store::source_index::open({bad.string()})), "valid decimal fields rejected");
    // The default legacy parser keeps the frozen oracle's numeric-prefix behavior.
    std::ofstream(bad) << "tensor\texpert_idx\tnbytes\tsource_path\toffset\n"
        << name << "\t3tail\t5000tail\t" << (directory / "1.bin").string() << "\t4097tail\n";
    c0::description legacy;
    legacy.aliases.push_back(bad.string());
    require(bool(c0::expert_index::open(legacy)), "legacy numeric-prefix behavior changed");
#if !defined(_WIN32)
    // A FIFO index or payload must neither block startup nor be published.
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
#endif
    std::filesystem::remove_all(directory);
    std::puts("SSD_CONTEXT_SOURCE_DIRECT_IO_PASS");
}
