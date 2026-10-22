#include "prototype.h"
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
#if !defined(__linux__)
    std::cerr << "Linux O_DIRECT test requires Linux; not exercised\n";
    return 77;
#else
    char pattern[] = "/tmp/ssdash-c0-direct-XXXXXX";
    char * made = ::mkdtemp(pattern);
    if (!made) { return 1; }
    const std::filesystem::path dir(made);
    const auto data = dir / "source.bin";
    const auto alias = dir / "alias.tsv";
    {
        std::ofstream out(data, std::ios::binary);
        std::vector<char> payload(16384);
        for (size_t i = 0; i < payload.size(); ++i) { payload[i] = char(i % 251); }
        out.write(payload.data(), payload.size());
        std::ofstream(alias) << "tensor\texpert_idx\tnbytes\tsource_path\toffset\n"
            << "blk.0.ffn_up_exps.weight\t3\t5000\t" << data.string() << "\t4097\n";
    }
    c0::description desc;
    desc.aliases = {alias.string()};
    desc.io_backend = 1;
    auto index = c0::expert_index::open(desc);
    c0::extent range;
    bool ok = index && index->locate("blk.0.ffn_up_exps.weight", 3, 5000, range) && range.offset == 4097;
    const int fd = ok ? index->borrowed_direct_fd(range.source_id) : -1;
    const int flags = fd < 0 ? -1 : ::fcntl(fd, F_GETFL);
    ok = ok && flags >= 0 && (flags & O_DIRECT);
    void * buffer = nullptr;
    ok = ok && ::posix_memalign(&buffer, 4096, 8192) == 0;
    if (ok) {
        const auto count = ::pread(fd, buffer, 8192, 4096);
        ok = count == 8192;
        const auto * bytes = static_cast<const unsigned char *>(buffer);
        for (size_t i = 0; ok && i < 5000; ++i) { ok = bytes[i + 1] == (4097 + i) % 251; }
    }
    std::free(buffer);
    index.reset();
    errno = 0;
    ok = ok && ::fcntl(fd, F_GETFD) == -1 && errno == EBADF;
    std::filesystem::remove_all(dir);
    std::cout << (ok ? "PASS" : "FAIL") << ": O_DIRECT borrowed descriptor, unaligned logical range, bounded aligned read, lifetime\n";
    return ok ? 0 : 1;
#endif
}
