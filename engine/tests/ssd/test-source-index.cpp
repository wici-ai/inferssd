// SPDX-License-Identifier: Apache-2.0
#include "ggml-ssd/store/index.h"
#include "ggml-ssd/store/expert-layout.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-ssd/platform/file.h"
#include "ggml-ssd/platform/memory.h"
#include <chrono>
#include <cstring>
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

// A GGUF file holding TENSORS ({name, type, ne0, ne1, ne2}), each byte of expert e of tensor k equal to 16 * k + e + 1, and
// a 16 KiB tensor after them, so no expert's whole-page read passes the end of the file.
struct gguf_tensor { const char * name; ggml_type type; int64_t ne[3]; };
static std::string write_gguf(const std::filesystem::path & path, const std::vector<gguf_tensor> & tensors) {
    ggml_init_params params = { 1u << 20, nullptr, false };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(gguf_init_empty(), gguf_free);
    for (size_t k = 0; k < tensors.size(); ++k) {
        const auto & in = tensors[k];
        ggml_tensor * t = in.ne[2] > 1 ? ggml_new_tensor_3d(ctx.get(), in.type, in.ne[0], in.ne[1], in.ne[2])
                                       : ggml_new_tensor_2d(ctx.get(), in.type, in.ne[0], in.ne[1]);
        ggml_set_name(t, in.name);
        for (int64_t e = 0; e < (in.ne[2] > 1 ? in.ne[2] : 1); ++e) {
            std::memset(static_cast<char *>(t->data) + e * t->nb[2], int(16 * k + e + 1), t->nb[2]);
        }
        gguf_add_tensor(gguf.get(), t);
    }
    ggml_tensor * tail = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4096);
    ggml_set_name(tail, "output.weight");
    std::memset(tail->data, 0, ggml_nbytes(tail));
    gguf_add_tensor(gguf.get(), tail);
    require(gguf_write_to_file(gguf.get(), path.string().c_str(), false), "GGUF write failed");
    return path.string();
}

// A model's GGUF files as expert sources: every expert of every routed expert tensor at its bytes in its file.
static void gguf_sources(const std::filesystem::path & directory) {
    const std::string first = write_gguf(directory / "model-00001-of-00002.gguf", {
        { "blk.0.ffn_up_exps.weight", GGML_TYPE_Q8_0, { 64, 3, 4 } },
        { "blk.0.ffn_gate_exps.weight", GGML_TYPE_Q4_K, { 256, 3, 4 } },
        { "blk.0.attn_q.weight", GGML_TYPE_F32, { 64, 64, 1 } },
    });
    const std::string second = write_gguf(directory / "model-00002-of-00002.gguf", {
        { "blk.0.ffn_down_exps.weight", GGML_TYPE_Q8_0, { 32, 64, 4 } },
        { "blk.0.ffn_gate_up_exps.weight", GGML_TYPE_Q8_0, { 64, 6, 4 } },
    });
    auto opened = ssd::store::source_index::open({first, second});
    require(bool(opened), "GGUF source creation failed");
    const auto view = opened.value->view();
    void * buffer = ssd::platform::aligned_alloc_bytes(4096, 16384);
    require(buffer != nullptr, "aligned buffer failed");
    std::unique_ptr<void, decltype(&ssd::platform::aligned_free)> owned(buffer, ssd::platform::aligned_free);
    const struct { const char * name; size_t bytes; int fill; } roles[] = {
        { "blk.0.ffn_up_exps.weight", 3 * 68, 1 }, { "blk.0.ffn_gate_exps.weight", 3 * 144, 17 },
        { "blk.0.ffn_down_exps.weight", 64 * 34, 1 },
    };
    for (const auto & role : roles) {
        for (int expert = 0; expert < 4; ++expert) {
            int fd = -1; uint64_t offset = 0; size_t bytes = 0;
            require(view.resolve(view.userdata, role.name, expert, role.bytes, 4, &fd, &offset, &bytes) &&
                    bytes == role.bytes && ssd::platform::is_direct(fd), "GGUF expert lookup failed");
            const ssd_expert_read read = ssd_expert_read_of(offset, bytes);
            require(read.bytes <= 16384 && ssd::platform::pread_exact(fd, buffer, size_t(read.bytes), read.offset),
                    "GGUF expert direct read failed");
            const auto * payload = static_cast<unsigned char *>(buffer) + read.data_offset;
            for (size_t i = 0; i < bytes; ++i) { require(payload[i] == role.fill + expert, "GGUF expert bytes changed"); }
        }
    }
    int fd = -1, up = -1; uint64_t offset = 0; size_t bytes = 0;
    require(view.resolve(view.userdata, "blk.0.ffn_up_exps.weight", 0, 3 * 68, 4, &up, &offset, &bytes) &&
            view.resolve(view.userdata, "blk.0.ffn_down_exps.weight", 0, 64 * 34, 4, &fd, &offset, &bytes) && fd != up,
            "the shards shared a descriptor");
    require(!view.resolve(view.userdata, "blk.0.attn_q.weight", 0, 64 * 64 * 4, 1, &fd, &offset, &bytes) &&
            !view.resolve(view.userdata, "blk.0.ffn_gate_up_exps.weight", 0, 6 * 68, 4, &fd, &offset, &bytes),
            "a tensor that is not a routed expert role was indexed");
    // a routed expert name on a tensor that is not experts after experts refuses the file
    const std::string flat = write_gguf(directory / "flat.gguf", { { "blk.1.ffn_up_exps.weight", GGML_TYPE_Q8_0, { 64, 3, 1 } } });
    require(!ssd::store::source_index::open({flat}), "a two-dimensional expert tensor was indexed");
}

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
        for (size_t i = 0; i < 16384; ++i) {
            // Two 640-column Q2_0 experts, ten 18-byte blocks each, with scale 0.5.
            const size_t byte = (i - 12288) % 18;
            stream.put(i >= 12288 && i < 12648 ? char(byte == 0 ? 0 : byte == 1 ? 0x38 : 0xe4)
                                              : char((i + seed) % 251));
        }
        stream.close();
        std::ofstream(alias) << "tensor\texpert_idx\tnbytes\tsource_path\toffset\n"
            << "blk.0.ffn_up_exps.weight\t3\t5000\t" << data.string() << "\t4097\n"
            << "blk.8.ffn_down_exps.weight\t0\t180\t" << data.string() << "\t12288\n"
            << "blk.8.ffn_down_exps.weight\t1\t180\t" << data.string() << "\t12468\n";
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
    for (int expert = 0; expert < 2; ++expert) {
        require(b.resolve(b.userdata, "blk.8.ffn_down_exps.weight", expert, 180, 2, &fd_b, &offset, &bytes),
                "Q2_0 expert alias lookup failed");
        require(offset == uint64_t(12288 + expert * 180) && bytes == 180, "Q2_0 expert range changed");
    }
    require(!b.resolve(b.userdata, "blk.8.ffn_down_exps.weight", 1, 1290, 2, &fd_b, &offset, &bytes),
            "legacy F8 byte count accepted for Q2_0");
    require(ssd::platform::pread_exact(fd_b, buffer, 4096, 12288), "Q2_0 direct read failed");
    for (size_t i = 0; i < 360; ++i) {
        require(payload[i] == (i % 18 == 0 ? 0 : i % 18 == 1 ? 0x38 : 0xe4), "Q2_0 payload changed");
    }
    second.value.reset();
    const auto bad = directory / "bad.tsv";
    std::ofstream(bad) << "tensor\texpert_idx\tnbytes\tsource_path\toffset\n"
        << name << "\t3\t5000\t" << (directory / "1.bin").string() << "\t16000\n";
    require(!ssd::store::source_index::open({bad.string()}), "out-of-file source published");
    require(!ssd::store::source_index::open({aliases[0], (directory / "missing.tsv").string()}), "partial source set published");
    gguf_sources(directory);
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
