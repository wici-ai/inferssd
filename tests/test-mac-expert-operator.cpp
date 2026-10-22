// Exercise the production Apple CPU hook with actual MUL_MAT_ID graphs.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

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
        require(argc == 5, "usage: test-mac-expert-operator EXPERTS ROWS DEPTH f32|q8_0");
        const int experts = std::stoi(argv[1]), rows = std::stoi(argv[2]), depth = std::stoi(argv[3]);
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
        unsetenv("SSD_LLAMA_MAC_EXPERT_PACK");
        unsetenv("SSD_LLAMA_MAC_EXPERT_TRACE");
        setenv("SSD_LLAMA_MAC_EXPERT_CACHE_MIB", "1", 1);
        setenv("SSD_LLAMA_MAC_EXPERT_READ_AHEAD", argv[3], 1);
        setenv("SSD_LLAMA_MAC_EXPERT_READ_ALIGNMENT", "4096", 1);
        setenv("SSD_LLAMA_MAC_EXPERT_PREFILL", "1", 1);
        for (int lanes : {1, selected}) {
            constexpr size_t nodes = 32;
            std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({
                    ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false), nullptr, true}), ggml_free);
            require(bool(ctx), "create graph context");
            std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
            require(bool(cpu), "create CPU backend");
            ggml_backend_cpu_set_n_threads(cpu.get(), 4);
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
            unsetenv("GGML_MOE_EXPERT_GGUF_ALIAS_TSV");
            require(ggml_backend_graph_compute(cpu.get(), graph) == GGML_STATUS_SUCCESS, "compute resident reference");
            std::vector<float> expected(ggml_nelements(result)), actual(expected.size());
            ggml_backend_tensor_get(result, expected.data(), 0, ggml_nbytes(result));
            require(std::any_of(expected.begin(), expected.end(), [](float value) { return value != 0; }), "empty reference output");
            // Zero the resident weights: equality can only pass if the source
            // hook actually reads the original bytes from the alias file.
            std::vector<char> zero(packed.size(), 0);
            ggml_backend_tensor_set(a, zero.data(), 0, zero.size());
            setenv("GGML_MOE_EXPERT_GGUF_ALIAS_TSV", source.alias.c_str(), 1);
            require(ggml_backend_graph_compute(cpu.get(), graph) == GGML_STATUS_SUCCESS, "compute alias graph");
            ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));
            require(std::memcmp(expected.data(), actual.data(), ggml_nbytes(result)) == 0, "MUL_MAT_ID alias output differs bitwise");
        }
        std::printf("MAC_EXPERT_OPERATOR_OK experts=%d rows=%d depth=%d type=%s broadcast=PASS routed=PASS\n",
                    experts, rows, depth, argv[4]);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "MAC_EXPERT_OPERATOR_FAILED: %s\n", error.what());
        return 1;
    }
}
