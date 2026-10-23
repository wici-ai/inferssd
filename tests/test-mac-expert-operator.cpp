// Exercise the production Apple CPU hook with actual MUL_MAT_ID graphs.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ssd/mac-runtime-fixture.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

struct temporary_source {
    char directory[64] = "/tmp/ssdash-mac-operator-XXXXXX";
    std::string weights, alias;
    temporary_source() {
        require(mkdtemp(directory) != nullptr, "create fixture directory");
        weights = std::string(directory) + "/weights.bin";
        alias = std::string(directory) + "/index.tsv";
    }
    ~temporary_source() {
        unlink(alias.c_str());
        unlink(weights.c_str());
        rmdir(directory);
    }
};
}

int main(int argc, char ** argv) {
    try {
        require(argc >= 5 && argc <= 7, "usage: test-mac-expert-operator EXPERTS ROWS DEPTH f32|q8_0 [THREADS] [vary]");
        const int experts = std::stoi(argv[1]), rows = std::stoi(argv[2]), depth = std::stoi(argv[3]);
        const int threads = argc >= 6 ? std::stoi(argv[5]) : 4;
        const bool vary_threads = argc == 7;
        require(!vary_threads || std::strcmp(argv[6], "vary") == 0, "invalid lifecycle mode");
        require(threads == 1 || threads == 2 || threads == 4 || threads == 12, "invalid fixture thread count");
        require((experts == 384 || experts == 256) && (rows == 1 || rows == 32 || rows == 512) &&
                (depth == 0 || depth == 2 || depth == 4), "invalid fixture dimensions");
        const ggml_type type = std::strcmp(argv[4], "f32") == 0 ? GGML_TYPE_F32 : GGML_TYPE_Q8_0;
        require(type == GGML_TYPE_F32 || std::strcmp(argv[4], "q8_0") == 0, "invalid fixture type");
        constexpr int width = 128, height = 64;
        const int selected = experts == 384 ? 12 : 24;
        const char * name = "blk.0.ffn_up_exps.weight";
        std::vector<float> weights(size_t(width) * height * experts);
        for (size_t i = 0; i < weights.size(); ++i) { weights[i] = float(int((i * 17 + i / width) % 127) - 63) / 64.0f; }
        const size_t slice_bytes = ggml_row_size(type, width) * height;
        std::vector<char> packed(slice_bytes * experts);
        require(ggml_quantize_chunk(type, weights.data(), packed.data(), 0, int64_t(height) * experts, width, nullptr) == packed.size(),
                "quantize fixture weights");
        temporary_source source;
        {
            std::ofstream output(source.weights, std::ios::binary);
            const char prefix[32] {};
            output.write(prefix, sizeof(prefix));
            output.write(packed.data(), packed.size());
            require(bool(output), "write expert source");
            std::ofstream index(source.alias);
            index << "tensor\texpert_idx\tnbytes\tsource_path\toffset\n";
            for (int expert = 0; expert < experts; ++expert) {
                index << name << '\t' << expert << '\t' << slice_bytes << '\t' << source.weights << '\t'
                      << 32 + size_t(expert) * slice_bytes << '\n';
            }
            require(bool(index), "write expert alias");
        }
        auto registry = ggml_backend_cpu_reg();
        auto set_cache = reinterpret_cast<ssd::mac::set_backend_cache>(
                ggml_backend_reg_get_proc_address(registry, "ggml_backend_cpu_set_ssd_mac_cache"));
        require(set_cache != nullptr, "CPU registry has no explicit Mac cache setter");
        for (int lanes : {1, selected}) {
            constexpr size_t nodes = 32;
            std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({
                    ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false), nullptr, true}), ggml_free);
            require(bool(ctx), "create graph context");
            auto pool_params = ggml_threadpool_params_default(vary_threads ? 12 : threads);
            pool_params.paused = true;
            pool_params.poll = 0;
            std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool(
                    ggml_threadpool_new(&pool_params), ggml_threadpool_free);
            require(bool(pool), "create paused threadpool");
            ssd::mac::testing::options cache_options;
            cache_options.budget_bytes = size_t(1) << 20;
            cache_options.read_ahead = size_t(depth);
            char cache_error[512] {};
            std::unique_ptr<ggml_ssd_mac_cache, decltype(&ssd::mac::destroy)> cache(
                    ssd::mac::testing::create({source.alias}, cache_options, cache_error, sizeof(cache_error)),
                    ssd::mac::destroy);
            require(bool(cache), cache_error);
            std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
            require(bool(cpu), "create CPU backend");
            ggml_backend_cpu_set_n_threads(cpu.get(), threads);
            ggml_backend_cpu_set_threadpool(cpu.get(), pool.get());
            auto * a = ggml_new_tensor_3d(ctx.get(), type, width, height, experts);
            ggml_set_name(a, name);
            auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, lanes, rows);
            auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, selected, rows);
            auto * result = ggml_mul_mat_id(ctx.get(), a, x, ids);
            auto * graph = ggml_new_graph_custom(ctx.get(), nodes, false);
            ggml_build_forward_expand(graph, result);
            std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
                    ggml_backend_alloc_ctx_tensors(ctx.get(), cpu.get()), ggml_backend_buffer_free);
            require(bool(buffer), "allocate graph tensors");
            std::vector<float> input(size_t(width) * lanes * rows);
            for (size_t i = 0; i < input.size(); ++i) { input[i] = float(int((i * 11 + i / width) % 31) - 15) / 16.0f; }
            std::vector<int32_t> routing(size_t(selected) * rows);
            for (int row = 0; row < rows; ++row) {
                for (int slot = 0; slot < selected; ++slot) { routing[size_t(row) * selected + slot] = (row * 13 + slot * 7) % experts; }
            }
            ggml_backend_tensor_set(a, packed.data(), 0, packed.size());
            ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
            ggml_backend_tensor_set(ids, routing.data(), 0, routing.size() * sizeof(int32_t));
            require(set_cache(cpu.get(), nullptr), "detach cache for resident reference");
            require(ggml_backend_graph_compute(cpu.get(), graph) == GGML_STATUS_SUCCESS, "compute resident reference");
            std::vector<float> expected(ggml_nelements(result)), actual(expected.size());
            ggml_backend_tensor_get(result, expected.data(), 0, ggml_nbytes(result));
            require(std::any_of(expected.begin(), expected.end(), [](float value) { return value != 0; }), "empty reference output");
            // Zero the resident weights: equality can only pass if the source
            // hook actually reads the original bytes from the alias file.
            std::vector<char> zero(packed.size(), 0);
            ggml_backend_tensor_set(a, zero.data(), 0, zero.size());
            require(set_cache(cpu.get(), cache.get()), "attach context-owned alias cache");
            require(set_cache(cpu.get(), cache.get()), "idempotent cache attach");
            require(!set_cache(nullptr, cache.get()), "null backend accepted a cache");
            char peer_error[512] {};
            std::unique_ptr<ggml_ssd_mac_cache, decltype(&ssd::mac::destroy)> peer(
                    ssd::mac::testing::create({source.alias}, cache_options, peer_error, sizeof(peer_error)),
                    ssd::mac::destroy);
            require(bool(peer), peer_error);
            require(!set_cache(cpu.get(), peer.get()), "nonempty backend replaced a borrowed cache");
            peer.reset();
            for (int repeat = 0; repeat < 4; ++repeat) {
                ggml_threadpool_pause(pool.get());
                require(ggml_backend_graph_compute(cpu.get(), graph) == GGML_STATUS_SUCCESS, "compute alias graph after pause");
                ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));
                require(std::memcmp(expected.data(), actual.data(), ggml_nbytes(result)) == 0, "MUL_MAT_ID alias output differs bitwise");
            }
            // Abort is observed between graph nodes, after all participants
            // leave the operator. The next graph must reuse the pool safely.
            ggml_backend_cpu_set_abort_callback(cpu.get(), [](void *) { return true; }, nullptr);
            require(ggml_backend_graph_compute(cpu.get(), graph) == GGML_STATUS_ABORTED, "abort alias graph");
            ggml_backend_cpu_set_abort_callback(cpu.get(), nullptr, nullptr);
            ggml_threadpool_pause(pool.get());
            require(ggml_backend_graph_compute(cpu.get(), graph) == GGML_STATUS_SUCCESS, "compute after abort and pause");
            ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));
            require(std::memcmp(expected.data(), actual.data(), ggml_nbytes(result)) == 0, "output differs after abort");
            if (vary_threads) {
                for (int active_threads : {1, 4, 12, 4, 1}) {
                    ggml_threadpool_pause(pool.get());
                    ggml_backend_cpu_set_n_threads(cpu.get(), active_threads);
                    require(ggml_backend_graph_compute(cpu.get(), graph) == GGML_STATUS_SUCCESS,
                            "compute with changed active thread count");
                    ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));
                    require(std::memcmp(expected.data(), actual.data(), ggml_nbytes(result)) == 0,
                            "output differs with changed active thread count");
                }
                std::printf("MAC_EXPERT_POOL_REUSE_OK capacity=12 active=1,4,12,4,1 lanes=%d\n", lanes);
            }
            require(set_cache(cpu.get(), nullptr), "detach alias cache before owner destruction");
        }
        std::printf("MAC_EXPERT_OPERATOR_OK experts=%d rows=%d depth=%d type=%s threads=%d broadcast=PASS routed=PASS lifecycle=PASS\n",
                    experts, rows, depth, argv[4], threads);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "MAC_EXPERT_OPERATOR_FAILED: %s\n", error.what());
        return 1;
    }
}
