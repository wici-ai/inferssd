#include "legacy-runtime/store.h"
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

int main() {
    struct cleanup { std::filesystem::path path; ~cleanup() { if (!path.empty()) { std::filesystem::remove_all(path); } } } guard;
    try {
        auto check = [](bool ok, const char * text) { if (!ok) { throw std::runtime_error(text); } };
        char pattern[] = "/tmp/ssdash-c0-linked-XXXXXX";
        const char * dir = ::mkdtemp(pattern); check(dir, "mkdtemp"); guard.path = dir;
        const auto data = guard.path / "weights.bin", alias = guard.path / "alias.tsv";
        { std::ofstream f(data, std::ios::binary); std::vector<char> bytes(16384, 0x31); f.write(bytes.data(), bytes.size()); }
        std::ofstream(alias) << "tensor\texpert_idx\tnbytes\tsource_path\toffset\nblk.0.ffn_up_exps.weight\t3\t5000\t" << data.string() << "\t4097\n";
        for (const char * name : {"GGML_MOE_EXPERT_PACK", "GGML_MOE_EXPERT_PACK_LIST", "GGML_MOE_PAGECACHE_PACK", "GGML_MOE_PAGECACHE_PACK_LIST",
                "GGML_MOE_Q4_COLIBRI_PACK", "GGML_MOE_Q4_COLIBRI_MANIFEST", "GGML_MOE_GLM52_COLIBRI_PACK", "GGML_MOE_GLM52_COLIBRI_MANIFEST",
                "GGML_MOE_LOSSLESS_PACK", "GGML_MOE_GLM52_LOSSLESS_PACK", "GGML_MOE_LOSSLESS_MODE", "GGML_MOE_GLM52_LOSSLESS_MODE"}) { ::unsetenv(name); }
        ::setenv("GGML_MOE_EXPERT_GGUF_ALIAS_TSV", alias.c_str(), 1);
        ::setenv("GGML_MOE_IO_BACKEND", "direct", 1);
        int fd = -1; uint64_t offset = 0; size_t bytes = 0;
        auto resolve = [&] { return ggml_ssd_c0_resolve_raw_direct_shape("blk.0.ffn_up_exps.weight", 3, 5000, 16, &fd, &offset, &bytes); };
        ::setenv("GGML_MOE_EXPERT_PACK_RUNTIME_DISABLE", "1", 1);
        check(!resolve(), "runtime disable must reject initial lookup");
        // Initialization must already have happened even though lookup was disabled.
        ::setenv("GGML_MOE_EXPERT_GGUF_ALIAS_TSV", "/nonexistent/c0-changed-after-init", 1);
        ::setenv("GGML_MOE_EXPERT_PACK_RUNTIME_DISABLE", "0tail", 1);
        check(resolve() && offset == 4097 && bytes == 5000, "initialized source must survive configuration change");
        check((::fcntl(fd, F_GETFL) & O_DIRECT) != 0, "returned descriptor is not direct");
        const int original_fd = fd;
        ::setenv("GGML_MOE_EXPERT_PACK_RUNTIME_DISABLE", "false", 1);
        check(!resolve(), "legacy disable uses first character, not boolean words");
        ::unsetenv("GGML_MOE_EXPERT_PACK_RUNTIME_DISABLE");
        check(resolve() && fd == original_fd, "same process must retain its descriptor");
        std::filesystem::rename(data, guard.path / "original.bin");
        { std::ofstream f(data, std::ios::binary); std::vector<char> replacement(16384, 0x72); f.write(replacement.data(), replacement.size()); }
        void * aligned = nullptr; check(::posix_memalign(&aligned, 4096, 8192) == 0, "aligned buffer");
        const auto got = ::pread(fd, aligned, 8192, 4096);
        bool same = got == 8192;
        for (size_t i = 0; same && i < 5000; ++i) { same = static_cast<unsigned char *>(aligned)[i+1] == 0x31; }
        std::free(aligned);
        check(same, "source replacement retargeted a borrowed descriptor");
        check(!ggml_ssd_c0_resolve_raw_direct_shape("blk.0.ffn_up_exps.weight", 16, 5000, 16, &fd, &offset, &bytes), "invalid expert accepted");
        std::puts("PASS: direct linked source, runtime disable/enable, initialize-before-disable, fixed source identity and borrowed lifetime");
        return 0;
    } catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
