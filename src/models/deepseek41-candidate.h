#pragma once

// Two-level candidate filter of DeepSeek-V4.1's sparse attention, as in vLLM deepseek_v41
// candidate_blocks.py. The candidate source splits its visible compressed rows into blocks of
// block_size rows, scores each block by its highest indexer score, always keeps the block that holds
// the newest visible row, and publishes the topk_blocks best blocks per query. The index sources after
// it score only the rows inside those blocks before taking their own top-k.

#include "ggml.h"

#include <cmath>
#include <cstdint>

// Pads the rows of x [n_rows, n_query] F32 with value up to a multiple of block_size.
static inline ggml_tensor * dsv41_pad_blocks(ggml_context * ctx, ggml_tensor * x, int64_t block_size, float value) {
    const int64_t n_rows = x->ne[0];
    const int64_t padded = (n_rows + block_size - 1) / block_size * block_size;
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx, x);
    }
    if (padded != n_rows) {
        ggml_tensor * pad = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, padded - n_rows, x->ne[1]);
        x = ggml_concat(ctx, x, ggml_fill(ctx, pad, value), 0);
    }
    return x;
}

// raw: [n_rows, n_query] F32 indexer scores before the causal mask, finite everywhere. mask: [n_rows,
// n_query] F32 causal mask, 0 on the visible prefix of each query's rows and -inf after it. Returns the
// score of every block [n_blocks, n_query] F32: its highest visible score, far below every score for a
// hidden block, and +inf for the block of the newest visible row. Nothing here copies a tensor of the
// scores' size unless n_rows is not a multiple of block_size.
static inline ggml_tensor * dsv41_candidate_block_scores(
        ggml_context * ctx, ggml_tensor * raw, ggml_tensor * mask, int64_t block_size) {
    GGML_ASSERT(raw->type == GGML_TYPE_F32 && mask->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(raw, mask) && raw->ne[2] == 1 && raw->ne[3] == 1);

    const int64_t n_query  = raw->ne[1];
    const int64_t n_blocks = (raw->ne[0] + block_size - 1) / block_size;

    // the maximum of each block: argmax over its rows (CUDA's argmax returns -1 for a row that is all -inf,
    // which is why the unmasked scores are reduced), then gather that element
    ggml_tensor * rows   = ggml_reshape_2d(ctx, dsv41_pad_blocks(ctx, raw, block_size, 0.0f), block_size, n_blocks*n_query);
    ggml_tensor * arg    = ggml_argmax(ctx, rows);
    ggml_tensor * blocks = ggml_get_rows(ctx, ggml_reshape_3d(ctx, rows, 1, block_size, n_blocks*n_query),
            ggml_reshape_2d(ctx, arg, 1, n_blocks*n_query));
    blocks = ggml_reshape_2d(ctx, blocks, n_blocks, n_query);

    // the visible rows form a prefix, so a block is visible when its first row is; only the newest visible
    // block can be partly hidden, and it is kept regardless of its score
    ggml_tensor * m     = ggml_reshape_3d(ctx, dsv41_pad_blocks(ctx, mask, block_size, -INFINITY), block_size, n_blocks, n_query);
    ggml_tensor * first = ggml_cont(ctx, ggml_view_3d(ctx, m, 1, n_blocks, n_query, m->nb[1], m->nb[2], 0));
    ggml_tensor * visible = ggml_step(ctx, ggml_scale_bias(ctx, ggml_reshape_2d(ctx, first, n_blocks, n_query), 1.0f, 0.5f));

    // hidden blocks fall far below every score, and the newest visible block is pinned above them all
    blocks = ggml_add(ctx, blocks, ggml_scale_bias(ctx, visible, 1e30f, -1e30f));
    ggml_tensor * newest = ggml_argmax(ctx, ggml_mul(ctx, visible, ggml_arange(ctx, 1.0f, float(n_blocks + 1), 1.0f)));
    ggml_tensor * pin    = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, n_query), INFINITY);
    blocks = ggml_set_rows(ctx, ggml_reshape_3d(ctx, blocks, 1, n_blocks, n_query), pin, ggml_reshape_2d(ctx, newest, 1, n_query));

    return ggml_reshape_2d(ctx, blocks, n_blocks, n_query);
}

// The candidate blocks [topk_blocks, n_query] I32 of the block scores above; the caller applies the filter
// only when there are more blocks than that.
static inline ggml_tensor * dsv41_candidate_blocks(
        ggml_context * ctx, ggml_tensor * raw, ggml_tensor * mask, int64_t block_size, int64_t topk_blocks) {
    ggml_tensor * scores = dsv41_candidate_block_scores(ctx, raw, mask, block_size);
    GGML_ASSERT(scores->ne[0] > topk_blocks);
    return ggml_top_k(ctx, scores, int(topk_blocks));
}

// score: [n_rows, n_query] F32; candidates: [k, n_query] I32 block ids. Returns score with -inf on the
// rows outside the candidate blocks, for an index source to take its top-k from; score is overwritten.
static inline ggml_tensor * dsv41_apply_candidates(
        ggml_context * ctx, ggml_tensor * score, ggml_tensor * candidates, int64_t block_size) {
    GGML_ASSERT(score->type == GGML_TYPE_F32 && score->ne[2] == 1 && score->ne[3] == 1);
    GGML_ASSERT(candidates->type == GGML_TYPE_I32 && candidates->ne[1] == score->ne[1]);

    const int64_t n_rows   = score->ne[0];
    const int64_t n_query  = score->ne[1];
    const int64_t n_blocks = (n_rows + block_size - 1) / block_size;
    const int64_t k        = candidates->ne[0];

    ggml_tensor * flags = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_blocks, n_query), -INFINITY);
    ggml_tensor * zero  = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, k, n_query), 0.0f);
    flags = ggml_set_rows(ctx, flags, zero, ggml_is_contiguous(candidates) ? candidates : ggml_cont(ctx, candidates));

    // broadcast each block's flag over its rows, in place: the scores are not needed afterwards
    ggml_tensor * rows = ggml_reshape_3d(ctx, dsv41_pad_blocks(ctx, score, block_size, -INFINITY), block_size, n_blocks, n_query);
    rows = ggml_reshape_2d(ctx, ggml_add_inplace(ctx, rows, flags), block_size*n_blocks, n_query);
    if (block_size*n_blocks != n_rows) {
        rows = ggml_cont(ctx, ggml_view_2d(ctx, rows, n_rows, n_query, rows->nb[1], 0));
    }
    return rows;
}
