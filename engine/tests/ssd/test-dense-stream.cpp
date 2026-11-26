// SPDX-License-Identifier: Apache-2.0
// Streamed weights through the context's real SSD owner, its GPU transport and the scheduler: matrices in host memory
// (a plain CPU buffer and the GPU's pinned host buffer, one through a reshape) whose products the GPU runs, their
// copies in a device ring smaller than a graph's weights. Each product equals the one of the same weights in device
// memory, at 1 and 64 tokens, graph after graph, and after a graph with other weights changes the plan; every copy
// after the first graph's first is issued ahead of its use. Matrices in a mapped file the transport reads around the
// page cache (their offsets 32-byte aligned, as in a GGUF) give the same products; once the file is cut short, the
// graph fails rather than read the mapping.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ssd/options.h"
#include "ggml-ssd/gpu/provider.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include "ggml-cuda.h"
#include "ggml-ssd/platform/file.h"

#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

void require(bool ok, const std::string & message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

constexpr int64_t K = 512;   // columns: 16 Q8_0 blocks of 34 bytes, 544 bytes a row
constexpr int64_t M = 256;   // rows
constexpr int n_host = 6;    // w0..w5 in host memory, w5 through a reshape; the ring holds four of them

struct copy_delta {
    uint64_t tensors, bytes, ahead;
};

copy_delta copies_since(ggml_backend_sched_t sched, const ggml_backend_sched_ssd_copy_stats & a) {
    ggml_backend_sched_ssd_copy_stats b = {};
    ggml_backend_sched_ssd_copy_stats_get(sched, &b);
    return { b.dense_tensors - a.dense_tensors, b.dense_bytes - a.dense_bytes, b.dense_readahead - a.dense_readahead };
}

// out = the sum of w[i] x (the last w seen as [K, M/2, 2] against x repeated twice) and r x, r in device memory
// a temporary file, removed with the object
struct temp_file {
    std::string path;
    int fd = -1;
    temp_file() {
        const char * tmp = std::getenv("TMPDIR");
        std::string pattern = std::string(tmp && *tmp ? tmp : "/tmp") + "/inferssd-dense-stream-XXXXXX";
        std::vector<char> buf(pattern.begin(), pattern.end());
        buf.push_back('\0');
        fd = mkstemp(buf.data());
        require(fd >= 0, "mkstemp failed");
        path = buf.data();
    }
    ~temp_file() {
        if (fd >= 0) {
            close(fd);
        }
        unlink(path.c_str());
    }
    void write(const void * data, size_t bytes, uint64_t offset) const {
        const uint8_t * p = static_cast<const uint8_t *>(data);
        for (size_t done = 0; done < bytes;) {
            const ssize_t n = pwrite(fd, p + done, bytes - done, off_t(offset + done));
            require(n > 0 || errno == EINTR, "pwrite failed");
            done += n > 0 ? size_t(n) : 0;
        }
    }
};

std::vector<float> run_graph(ggml_backend_sched_t sched, const std::vector<ggml_tensor *> & w, ggml_tensor * r,
                             int n_tokens, const std::vector<float> & x_data, ggml_status * status = nullptr) {
    ggml_init_params params = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n_tokens);
    ggml_set_input(x);
    ggml_tensor * out = ggml_mul_mat(ctx, r, x);
    for (size_t i = 0; i + 1 < w.size(); ++i) {
        out = ggml_add(ctx, out, ggml_mul_mat(ctx, w[i], x));
    }
    // [K, M] seen as [K, M/2, 2] against x repeated as [K, n, 2]: [M/2, n, 2], laid out as [M, n] again
    ggml_tensor * x3 = ggml_cont(ctx, ggml_repeat(ctx, ggml_reshape_3d(ctx, x, K, n_tokens, 1),
                                                  ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, n_tokens, 2)));
    ggml_tensor * y3 = ggml_mul_mat(ctx, ggml_reshape_3d(ctx, w.back(), K, M / 2, 2), x3);
    out = ggml_add(ctx, out, ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, y3, 0, 2, 1, 3)), M, n_tokens));
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_sched_reset(sched);
    require(ggml_backend_sched_alloc_graph(sched, gf), "graph allocation failed");
    ggml_backend_tensor_set(x, x_data.data(), 0, sizeof(float) * K * n_tokens);
    const ggml_status computed = ggml_backend_sched_graph_compute(sched, gf);
    if (status) {
        *status = computed;
        ggml_backend_sched_synchronize(sched);
        ggml_backend_sched_reset(sched);
        ggml_free(ctx);
        return {};
    }
    require(computed == GGML_STATUS_SUCCESS, "graph compute failed");
    std::vector<float> result(size_t(M) * n_tokens);
    ggml_backend_tensor_get(out, result.data(), 0, sizeof(float) * result.size());
    ggml_free(ctx);
    return result;
}

void run() {
    const size_t nbytes = ggml_row_size(GGML_TYPE_Q8_0, K) * M;
    std::mt19937 rng(20261120);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<std::vector<uint8_t>> q(n_host + 1, std::vector<uint8_t>(nbytes));
    for (auto & m : q) {
        std::vector<float> f(size_t(K) * M);
        for (float & v : f) {
            v = normal(rng);
        }
        ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), m.data(), 0, M, K, nullptr);
    }

    const auto settings = ssd::settings::parse("", "", {});
    require(bool(settings), settings.error);
    const llama_ssd_options options{settings.value, {}, {}};
    // L3's actual owner: no routed experts, the host budget and the GPU transport every SSD context has
    auto runtime = std::make_unique<llama_ssd_context>(options);
    using backend_owner = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    backend_owner gpu_owned(ggml_backend_cuda_init(0), ggml_backend_free);
    backend_owner cpu_owned(ggml_backend_cpu_init(), ggml_backend_free);
    ggml_backend_t gpu = gpu_owned.get(), cpu = cpu_owned.get();
    require(gpu && cpu, "backend init failed");
    runtime->attach_backend(gpu, LLAMA_SPLIT_MODE_NONE);
    require(runtime->services.gpu && runtime->services.gpu_transport, "the context has no GPU transport");
    ssd::host::client_config budget;
    budget.unified = runtime->services.gpu->device_is_unified(runtime->services.gpu_device);
    require(bool(runtime->host->configure(budget)), "host budget configuration failed");

    // w0, w2, w4 in a plain CPU buffer, w1, w3, w5 in the GPU's pinned host buffer, r and the reference in the device
    ggml_init_params wparams = { ggml_tensor_overhead() * 16, nullptr, true };
    ggml_context * plain_ctx = ggml_init(wparams);
    ggml_context * pinned_ctx = ggml_init(wparams);
    ggml_context * dev_ctx = ggml_init(wparams);
    std::vector<ggml_tensor *> host(n_host), dev(n_host + 1);
    for (int i = 0; i < n_host; ++i) {
        host[i] = ggml_new_tensor_2d(i % 2 ? pinned_ctx : plain_ctx, GGML_TYPE_Q8_0, K, M);
        ggml_format_name(host[i], "blk.%d.attn_q.weight", i);
    }
    for (int i = 0; i <= n_host; ++i) {
        dev[i] = ggml_new_tensor_2d(dev_ctx, GGML_TYPE_Q8_0, K, M);
        ggml_format_name(dev[i], "dev.%d.weight", i);
    }
    ggml_backend_buffer_type_t pinned_type = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(gpu));
    require(pinned_type != nullptr, "the GPU has no pinned host buffer type");
    ggml_backend_buffer_t plain_buf = ggml_backend_alloc_ctx_tensors(plain_ctx, cpu);
    ggml_backend_buffer_t pinned_buf = ggml_backend_alloc_ctx_tensors_from_buft(pinned_ctx, pinned_type);
    ggml_backend_buffer_t dev_buf = ggml_backend_alloc_ctx_tensors(dev_ctx, gpu);
    require(plain_buf && pinned_buf && dev_buf, "weight buffers");
    for (ggml_backend_buffer_t b : { plain_buf, pinned_buf, dev_buf }) {
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    for (int i = 0; i < n_host; ++i) {
        ggml_backend_tensor_set(host[i], q[i].data(), 0, nbytes);
        ggml_backend_tensor_set(dev[i], q[i].data(), 0, nbytes);
    }
    ggml_backend_tensor_set(dev[n_host], q[n_host].data(), 0, nbytes);
    ggml_tensor * r = dev[n_host];

    ggml_backend_t backends[2] = { gpu, cpu };
    using sched_owner = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;
    sched_owner scheduler(ggml_backend_sched_new(backends, nullptr, 2, 2048, false, true), ggml_backend_sched_free);
    ggml_backend_sched_t sched = scheduler.get();
    require(sched, "scheduler creation failed");
    runtime->stream.attach(sched);

    const std::vector<ggml_tensor *> ref_w(dev.begin(), dev.begin() + n_host);
    bool first = true;
    for (int n_tokens : { 1, 64 }) {
        std::vector<float> x(size_t(K) * n_tokens);
        for (float & v : x) {
            v = normal(rng);
        }
        const std::vector<float> ref = run_graph(sched, ref_w, r, n_tokens, x);
        for (int pass = 0; pass < 3; ++pass) {
            ggml_backend_sched_ssd_copy_stats before = {};
            ggml_backend_sched_ssd_copy_stats_get(sched, &before);
            const std::vector<float> got = run_graph(sched, host, r, n_tokens, x);
            const copy_delta d = copies_since(sched, before);
            require(std::memcmp(got.data(), ref.data(), sizeof(float) * got.size()) == 0,
                    "the streamed weights' product differs from the device weights' at " + std::to_string(n_tokens) +
                    " tokens, graph " + std::to_string(pass));
            require(d.tensors == n_host && d.bytes == n_host * nbytes,
                    "copies " + std::to_string(d.tensors) + " of " + std::to_string(d.bytes) + " bytes");
            // the first graph's first copy waits for its use; every later one was issued ahead, the next graph's
            // first ones while the graph before it still ran
            require(d.ahead == (first ? n_host - 1 : n_host), "ahead " + std::to_string(d.ahead));
            std::printf("tokens=%d graph=%d copies=%llu ahead=%llu splits=%d\n", n_tokens, pass,
                        (unsigned long long) d.tensors, (unsigned long long) d.ahead,
                        ggml_backend_sched_get_n_splits(sched));
            first = false;
        }
    }
    // other weights: a new plan, its first copy issued at its use; the old one's again after it
    {
        std::vector<float> x(size_t(K) * 64);
        for (float & v : x) {
            v = normal(rng);
        }
        const std::vector<ggml_tensor *> few_host(host.begin() + 2, host.end());
        const std::vector<ggml_tensor *> few_dev(dev.begin() + 2, dev.begin() + n_host);
        const std::vector<float> ref = run_graph(sched, few_dev, r, 64, x);
        const std::vector<float> got = run_graph(sched, few_host, r, 64, x);
        require(std::memcmp(got.data(), ref.data(), sizeof(float) * got.size()) == 0,
                "the streamed weights' product differs after the plan changed");
        const std::vector<float> all_ref = run_graph(sched, ref_w, r, 64, x);
        const std::vector<float> all_got = run_graph(sched, host, r, 64, x);
        require(std::memcmp(all_got.data(), all_ref.data(), sizeof(float) * all_got.size()) == 0,
                "the streamed weights' product differs after the plan changed back");
    }
    // the same matrices in a mapped file, as a GGUF holds them (32-byte aligned, none on a page boundary), read from
    // the drive: the file registered with the transport, then cut short
    temp_file data;
    std::vector<uint64_t> offset(n_host);
    uint64_t cursor = 4096 + 96;
    for (int i = 0; i < n_host; ++i) {
        offset[i] = cursor;
        data.write(q[i].data(), nbytes, cursor);
        cursor += (nbytes + 32 + 31) / 32 * 32;
    }
    require(fsync(data.fd) == 0, "fsync failed");
    const size_t map_bytes = size_t(cursor);
    void * map = mmap(nullptr, map_bytes, PROT_READ, MAP_SHARED, data.fd, 0);
    require(map != MAP_FAILED, "mmap failed");
    ggml_backend_buffer_t mapped = ggml_backend_cpu_buffer_from_ptr(map, map_bytes);
    ggml_backend_buffer_set_usage(mapped, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_context * file_ctx = ggml_init(wparams);
    std::vector<ggml_tensor *> filed(n_host);
    for (int i = 0; i < n_host; ++i) {
        filed[i] = ggml_new_tensor_2d(file_ctx, GGML_TYPE_Q8_0, K, M);
        ggml_format_name(filed[i], "blk.%d.attn_k.weight", i);
        require(ggml_backend_tensor_alloc(mapped, filed[i], (char *) map + offset[i]) == GGML_STATUS_SUCCESS, "alloc");
    }
    std::string error;
    const int direct = ssd::platform::open_direct(data.path.c_str(), &error);
    require(direct >= 0, "the test file has no direct reads: " + error);
    const ssd::gpu::stream_file file = { map, map_bytes, direct };
    require(runtime->stream.stream_files(&file, 1), "the transport refused the file");
    for (int n_tokens : { 1, 64 }) {
        std::vector<float> x(size_t(K) * n_tokens);
        for (float & v : x) {
            v = normal(rng);
        }
        const std::vector<float> ref = run_graph(sched, ref_w, r, n_tokens, x);
        for (int pass = 0; pass < 2; ++pass) {
            ggml_backend_sched_ssd_copy_stats before = {};
            ggml_backend_sched_ssd_copy_stats_get(sched, &before);
            const std::vector<float> got = run_graph(sched, filed, r, n_tokens, x);
            const copy_delta d = copies_since(sched, before);
            require(std::memcmp(got.data(), ref.data(), sizeof(float) * got.size()) == 0,
                    "the product of the weights read from the drive differs at " + std::to_string(n_tokens) +
                    " tokens, graph " + std::to_string(pass));
            require(d.tensors == n_host && d.bytes == n_host * nbytes, "drive copies " + std::to_string(d.tensors));
            std::printf("drive tokens=%d graph=%d copies=%llu ahead=%llu\n", n_tokens, pass,
                        (unsigned long long) d.tensors, (unsigned long long) d.ahead);
        }
    }
    // the file cut inside the last matrix: its read comes back short, and the graph fails
    require(ftruncate(data.fd, off_t(offset[n_host - 1] + nbytes / 2)) == 0, "truncate failed");
    {
        std::vector<float> x(size_t(K) * 1, 1.0f);
        ggml_status status = GGML_STATUS_SUCCESS;
        (void) run_graph(sched, filed, r, 1, x, &status);
        require(status != GGML_STATUS_SUCCESS, "a short drive read was accepted or read from the mapping");
    }
    scheduler.reset();
    ggml_backend_buffer_free(mapped);
    ggml_free(file_ctx);
    munmap(map, map_bytes);
    ggml_backend_buffer_free(plain_buf);
    ggml_backend_buffer_free(pinned_buf);
    ggml_backend_buffer_free(dev_buf);
    ggml_free(plain_ctx);
    ggml_free(pinned_ctx);
    ggml_free(dev_ctx);
    cpu_owned.reset();
    gpu_owned.reset(); // the backend quiesces the transport before its CUDA streams go
    runtime.reset();
    std::printf("SSD_DENSE_STREAM_OK\n");
}

} // namespace

int main() {
    try {
        run();
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "SSD_DENSE_STREAM_FAILED: %s\n", e.what());
        return 1;
    }
}

#else
int main() {
    return 77;
}
#endif
