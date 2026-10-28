// SPDX-License-Identifier: Apache-2.0
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>
#include <unistd.h>

extern "C" bool inferssd_test_kimi_matmul(ggml_tensor *, void *, size_t);
static void require(bool value, const char * message) { if (!value) { throw std::runtime_error(message); } }

int main() {
    try {
        ggml_cpu_init();
        const std::string trace = "/tmp/inferssd-retired-route-" + std::to_string(getpid());
        int cases = 0;
        for (auto type : { GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_MXFP4 }) {
            for (int tokens : { 1, 4 }) {
                const int columns = type == GGML_TYPE_MXFP4 ? 320 : 256, rows = 16, experts = 8, used = 3;
                auto * context = ggml_init({ 4 << 20, nullptr, false }); require(context, "context");
                auto * weights = ggml_new_tensor_3d(context, type, columns, rows, experts);
                auto * input = ggml_new_tensor_3d(context, GGML_TYPE_F32, columns, 1, tokens);
                auto * ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, used, tokens);
                std::vector<float> source(size_t(columns) * rows * experts);
                for (size_t i = 0; i < source.size(); ++i) { source[i] = float(std::sin(double(i) * .013) * .08); }
                require(ggml_quantize_chunk(type, source.data(), weights->data, 0, rows * experts, columns, nullptr) == ggml_nbytes(weights), "quantized weights");
                for (int i = 0; i < columns * tokens; ++i) { static_cast<float *>(input->data)[i] = float(std::cos(double(i) * .017)); }
                for (int i = 0; i < used * tokens; ++i) { static_cast<int32_t *>(ids->data)[i] = i == used * tokens - 1 ? -1 : (i * 5 + 1) % experts; }
                ggml_set_name(weights, "blk.0.ffn_up_exps.weight");
                auto * output = ggml_mul_mat_id(context, weights, input, ids); ggml_set_name(output, "ffn_moe_out");
                auto * graph = ggml_new_graph(context); ggml_build_forward_expand(graph, output);
                auto plan = ggml_graph_plan(graph, 1, nullptr);
                void * raw = nullptr; require(posix_memalign(&raw, 64, std::max<size_t>(64, plan.work_size)) == 0, "work allocation");
                plan.work_data = static_cast<uint8_t *>(raw);
                require(inferssd_test_kimi_matmul(output, raw, plan.work_size), "explicit Kimi oracle compute");
                std::vector<unsigned char> expected(ggml_nbytes(output)); std::memcpy(expected.data(), output->data, expected.size());
                // Ambient legacy selection/pruning/trace must no longer affect
                // the production CPU op; the oracle receives an explicit call.
                setenv("MODEL_TYPE", "kimi", 1); setenv("GGML_MOE_KEEP_TOPK_UPDOWN", "1", 1);
                setenv("GGML_MOE_CPU_CHUNK_TRACE_OUT", trace.c_str(), 1);
                setenv("GGML_KIMI_PIPELINE_V2_ACTIVE", "1", 1); setenv("GGML_KIMI_PIPELINE_V2_MODE", "depth1", 1);
                setenv("GGML_KIMI_PIPELINE_M1_CHECKPOINT_OUT", trace.c_str(), 1);
                require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS &&
                    std::memcmp(expected.data(), output->data, expected.size()) == 0, "ambient controls changed production output");
                std::memset(output->data, 0, expected.size());
                require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS &&
                    std::memcmp(expected.data(), output->data, expected.size()) == 0, "repeated generic math differs");
                require(!std::filesystem::exists(trace), "retired observer wrote output");
                free(raw); ggml_free(context); ++cases;
            }
        }
        std::printf("CPU_FIXED_ROUTE_ORACLE_PASS cases=%d ambient_ignored=1 bit_exact=1\n", cases);
        return 0;
    } catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
