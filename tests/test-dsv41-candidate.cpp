// Checks the DeepSeek-V4.1 two-level candidate filter graph against a direct implementation of the
// reference rule (vLLM deepseek_v41 candidate_blocks.py): per query, blocks of block_size visible rows
// scored by their highest score, the newest visible block always kept, the topk_blocks best kept.

#include "models/deepseek41-candidate.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-dsv41-candidate: %s\n", message);
        std::exit(1);
    }
}

static std::set<int> reference_blocks(const std::vector<float> & score, int64_t n_visible, int64_t block_size,
                                      int64_t topk_blocks, std::vector<bool> & finite) {
    const int64_t n_rows   = int64_t(score.size());
    const int64_t n_blocks = (n_rows + block_size - 1) / block_size;
    std::vector<float> block(n_blocks, -INFINITY);
    for (int64_t r = 0; r < n_visible; ++r) {
        block[r / block_size] = std::max(block[r / block_size], score[r]);
    }
    finite.assign(n_blocks, false);
    for (int64_t b = 0; b < n_blocks; ++b) {
        finite[b] = std::isfinite(block[b]);
    }
    if (n_visible > 0) {
        block[(n_visible - 1) / block_size] = INFINITY;
        finite[(n_visible - 1) / block_size] = true;
    }
    std::vector<int> order(n_blocks);
    for (int64_t b = 0; b < n_blocks; ++b) { order[b] = int(b); }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return block[a] > block[b]; });
    std::set<int> out;
    for (int64_t i = 0; i < std::min(topk_blocks, n_blocks); ++i) {
        if (std::isfinite(block[order[i]]) || block[order[i]] > 0) { out.insert(order[i]); }
    }
    return out;
}

static void run_case(ggml_backend_t backend, int64_t n_rows, std::vector<int64_t> visible, int64_t block_size,
                     int64_t topk_blocks, int seed) {
    const int64_t n_query = int64_t(visible.size());
    ggml_init_params params = { size_t(64) << 20, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "ggml_init failed");

    ggml_tensor * score = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rows, n_query);
    ggml_tensor * mask  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rows, n_query);
    ggml_tensor * zeros = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rows, n_query);
    ggml_set_input(score); ggml_set_input(mask); ggml_set_input(zeros);

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> score_data(n_rows*n_query), mask_data(n_rows*n_query), zero_data(n_rows*n_query, 0.0f);
    std::vector<std::vector<float>> masked(n_query, std::vector<float>(n_rows));
    for (int64_t q = 0; q < n_query; ++q) {
        for (int64_t r = 0; r < n_rows; ++r) {
            const bool vis = r < visible[q];
            const float s = std::max(0.0f, dist(rng)) * 3.0f + dist(rng) * 0.01f;
            // the unmasked scores are finite on hidden rows too, and must not leak into the choice
            score_data[q*n_rows + r] = s;
            mask_data[q*n_rows + r]  = vis ? 0.0f : -INFINITY;
            masked[q][r] = vis ? s : -INFINITY;
        }
    }

    ggml_tensor * blocks = dsv41_candidate_blocks(ctx, score, mask, block_size, topk_blocks);
    ggml_tensor * rows   = dsv41_apply_candidates(ctx, zeros, blocks, block_size);
    ggml_set_output(blocks); ggml_set_output(rows);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, blocks);
    ggml_build_forward_expand(gf, rows);

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        require(ggml_backend_supports_op(backend, ggml_graph_node(gf, i)), "backend does not support an op of the filter");
    }
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(ggml_gallocr_alloc_graph(galloc, gf), "graph allocation failed");
    ggml_backend_tensor_set(score, score_data.data(), 0, ggml_nbytes(score));
    ggml_backend_tensor_set(mask,  mask_data.data(),  0, ggml_nbytes(mask));
    ggml_backend_tensor_set(zeros, zero_data.data(),  0, ggml_nbytes(zeros));
    require(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS, "graph compute failed");

    std::vector<int32_t> block_ids(ggml_nelements(blocks));
    std::vector<float>   row_mask(ggml_nelements(rows));
    require(ggml_is_contiguous(blocks) && ggml_is_contiguous(rows), "outputs are not contiguous");
    ggml_backend_tensor_get(blocks, block_ids.data(), 0, ggml_nbytes(blocks));
    ggml_backend_tensor_get(rows,   row_mask.data(),  0, ggml_nbytes(rows));

    const int64_t n_blocks = (n_rows + block_size - 1) / block_size;
    for (int64_t q = 0; q < n_query; ++q) {
        std::vector<bool> finite;
        const std::set<int> want = reference_blocks(masked[q], visible[q], block_size, topk_blocks, finite);
        std::set<int> got;
        for (int64_t i = 0; i < topk_blocks; ++i) {
            const int b = block_ids[q*topk_blocks + i];
            require(b >= 0 && b < n_blocks, "block id out of range");
            got.insert(b);
        }
        // blocks without a visible row are masked anyway; compare the choice among the others
        for (int64_t b = 0; b < n_blocks; ++b) {
            if (finite[b]) {
                require((got.count(int(b)) != 0) == (want.count(int(b)) != 0), "candidate blocks differ from the reference");
            }
        }
        require(got.count(int((visible[q] - 1) / block_size)) != 0, "newest visible block not kept");
        for (int64_t r = 0; r < n_rows; ++r) {
            const float v = row_mask[q*n_rows + r];
            const bool keep = got.count(int(r / block_size)) != 0;
            require(keep ? v == 0.0f : (std::isinf(v) && v < 0), "candidate row mask differs from the chosen blocks");
        }
    }
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
}

static void run_all(ggml_backend_t backend) {
    // more visible blocks than candidates, with and without a partial last block
    run_case(backend, 16640, { 16640, 16500, 16389, 16391 }, 8, 2048, 1);
    run_case(backend, 16389, { 16389, 16385, 16300 }, 8, 2048, 2);
    // queries that see fewer rows than the candidate budget keep every visible block
    run_case(backend, 20000, { 20000, 16384, 12000, 9 }, 8, 2048, 3);
    // a small configuration with short queries
    run_case(backend, 200, { 200, 199, 97, 1 }, 4, 8, 4);
    std::printf("test-dsv41-candidate: %s OK\n", ggml_backend_name(backend));
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    require(cpu != nullptr, "no CPU backend");
    run_all(cpu);
    ggml_backend_free(cpu);
    // the GPU backend when there is one (the model builds this filter on it)
    if (ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr)) {
        run_all(gpu);
        ggml_backend_free(gpu);
    }
    return 0;
}
