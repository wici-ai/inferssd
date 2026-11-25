#include "ggml-metal-ops.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-impl.h"
#include "ggml-metal-common.h"
#include "ggml-metal-device.h"

#include <cassert>
#include <algorithm>
#include <limits>
#include <cmath>

static ggml_metal_buffer_id ggml_metal_get_buffer_id(const ggml_tensor * t) {
    if (!t) {
        return { nullptr, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t) buffer->context;

    return ggml_metal_buffer_get_id(ctx, t);
}

struct ggml_metal_op {
    ggml_metal_op(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_cgraph * gf,
        int  idx_start,
        int  idx_end,
        bool use_fusion,
        bool use_concurrency,
        bool use_capture,
        int  debug_graph,
        int  debug_fusion) {
        this->dev             = dev;
        this->lib             = ggml_metal_device_get_library(dev);
        this->enc             = ggml_metal_encoder_init(cmd_buf, use_concurrency);
        this->mem_ranges      = ggml_mem_ranges_init(debug_graph);
        this->idx_start       = idx_start;
        this->idx_end         = idx_end;
        this->use_fusion      = use_fusion;
        this->use_concurrency = use_concurrency;
        this->use_capture     = use_capture;
        this->debug_graph     = debug_graph;
        this->debug_fusion    = debug_fusion;
        this->gf              = gf;

        idxs.reserve(gf->n_nodes);

        // filter empty nodes
        // TODO: this can be removed when the allocator starts filtering them earlier
        //       https://github.com/ggml-org/llama.cpp/pull/16130#issuecomment-3327905830
        for (int i = idx_start; i < idx_end; i++) {
            if (!ggml_op_is_empty(gf->nodes[i]->op) && !ggml_is_empty(gf->nodes[i])) {
                idxs.push_back(i);
            }
        }
    }

    ~ggml_metal_op() {
        if (this->enc) {
            ggml_metal_encoder_end_encoding(this->enc);
            ggml_metal_encoder_free(this->enc);
        }
        ggml_mem_ranges_free(this->mem_ranges);
    }

    int n_nodes() const {
        return idxs.size();
    }

    ggml_tensor * node(int i) const {
        assert(i >= 0 && i < (int) idxs.size());
        return ggml_graph_node(gf, idxs[i]);
    }

    bool can_fuse(int i0, const ggml_op * ops, int n_ops) const {
        assert(use_fusion);
        assert(i0 >= 0 && i0 < n_nodes());

        if (i0 + n_ops > n_nodes()) {
            return false;
        }

        return ggml_can_fuse_ext(gf, idxs.data() + i0, ops, n_ops);
    }

    bool node_has_n_uses(int i0, int n_uses) const {
        assert(i0 >= 0 && i0 < n_nodes());
        return ggml_node_has_n_uses(gf, idxs[i0], n_uses);
    }

    int32_t tensor_use_count(const ggml_tensor * t) const {
        size_t hash_pos = ggml_hash_find(&gf->visited_hash_set, t);
        if (!ggml_bitset_get(gf->visited_hash_set.used, hash_pos)) {
            return 0;
        }
        return gf->use_counts[hash_pos];
    }

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;
    ggml_metal_encoder_t enc;
    ggml_mem_ranges_t    mem_ranges;

    bool use_fusion;
    bool use_concurrency;
    bool use_capture;

    int debug_graph;
    int debug_fusion;

    // Per-command-buffer private scratch cache (1152 KB: two 28 KB FA_EXT blk slots + 8 KB MoE sort/group slot + two 544 KB Split-K partial slots).
    // Keyed strictly by graph tensor pointers within a single CB so gallocr reuse cannot alias cached data.
    void * aux_buf      = nullptr;
    size_t aux_buf_size = 0;

    struct fa_blk_cache_entry {
        const ggml_tensor * mask = nullptr;
        const ggml_tensor * bias = nullptr;
        int32_t ne01 = 0, ne30 = 0, ne31 = 0, ne32 = 0, ne33 = 0, ncpsg = 0, bias_n_bins = 0, bias_n_keys = 0;
        bool valid = false;
    };
    fa_blk_cache_entry aux_blk[2];
    int aux_blk_next = 0;

    const ggml_tensor * aux_sort_ids = nullptr;
    int32_t aux_sort_ne20 = 0;
    int32_t aux_sort_ne21 = 0;
    int32_t aux_sort_ne02 = 0;
    int32_t aux_sort_mode = 0; // 1 = sorted_nvfp4, 2 = group_compact

    bool aux_sk_dirty[2] = { false, false };
    int  aux_sk_next     = 0;

private:
    ggml_cgraph * gf;

    int idx_start;
    int idx_end;

    // non-empty node indices
    std::vector<int> idxs;
};

ggml_metal_op_t ggml_metal_op_init(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_cgraph * gf,
        int idx_start,
        int idx_end,
        bool use_fusion,
        bool use_concurrency,
        bool use_capture,
        int debug_graph,
        int debug_fusion) {
    ggml_metal_op_t res = new ggml_metal_op(
        dev,
        cmd_buf,
        gf,
        idx_start,
        idx_end,
        use_fusion,
        use_concurrency,
        use_capture,
        debug_graph,
        debug_fusion);

    return res;
}

void ggml_metal_op_free(ggml_metal_op_t ctx) {
    delete ctx;
}

int ggml_metal_op_n_nodes(ggml_metal_op_t ctx) {
    return ctx->n_nodes();
}

struct ggml_tensor * ggml_metal_op_node(ggml_metal_op_t ctx, int idx) {
    return ctx->node(idx);
}

void ggml_metal_op_set_aux_buf(ggml_metal_op_t ctx, void * metal_buf, size_t size) {
    ctx->aux_buf      = metal_buf;
    ctx->aux_buf_size = size;
}

static bool ggml_metal_op_concurrency_reset(ggml_metal_op_t ctx) {
    ctx->aux_sk_dirty[0] = false;
    ctx->aux_sk_dirty[1] = false;

    if (!ctx->mem_ranges) {
        return true;
    }

    ggml_metal_encoder_memory_barrier(ctx->enc);

    ggml_mem_ranges_reset(ctx->mem_ranges);

    return true;
}

static bool ggml_metal_op_alloc_sk_buf(ggml_metal_op_t ctx, size_t n_bytes, ggml_metal_buffer_id * bid_out, int * slot_out) {
    constexpr size_t SK_BASE_OFFS = 65536;
    constexpr size_t SK_SLOT_SIZE = 557056; // 544 KB per slot (2 slots -> 1152 KB total aux_buf)
    if (ctx->aux_buf == nullptr || ctx->aux_buf_size < SK_BASE_OFFS + 2 * SK_SLOT_SIZE || n_bytes > SK_SLOT_SIZE) {
        return false;
    }
    const int s = ctx->aux_sk_next;
    ctx->aux_sk_next ^= 1;
    if (ctx->aux_sk_dirty[s]) {
        ggml_metal_op_concurrency_reset(ctx);
    }
    *bid_out  = ggml_metal_buffer_id{ ctx->aux_buf, SK_BASE_OFFS + (size_t) s * SK_SLOT_SIZE };
    *slot_out = s;
    return true;
}

static void ggml_metal_op_dispatch_sk_reduce(
        ggml_metal_op_t ctx, ggml_metal_buffer_id bid_partial, ggml_metal_buffer_id bid_dst, int32_t n, int32_t sk, int slot) {
    ggml_metal_op_concurrency_reset(ctx);
    ctx->aux_sk_dirty[slot] = true;

    auto pl_red = ggml_metal_library_get_pipeline_mul_mm_splitk_reduce(ctx->lib);
    ggml_metal_kargs_mul_mm_splitk_reduce rargs = { /*.n =*/ n, /*.sk =*/ sk };
    ggml_metal_encoder_set_pipeline(ctx->enc, pl_red);
    ggml_metal_encoder_set_bytes   (ctx->enc, &rargs, sizeof(rargs), 0);
    ggml_metal_encoder_set_buffer  (ctx->enc, bid_partial, 1);
    ggml_metal_encoder_set_buffer  (ctx->enc, bid_dst,     2);
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc, (n + 255) / 256, 1, 1, 256, 1, 1);
}

static bool ggml_metal_op_concurrency_check(ggml_metal_op_t ctx, const ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return false;
    }

    return ggml_mem_ranges_check(ctx->mem_ranges, node);
}

static bool ggml_metal_op_concurrency_add(ggml_metal_op_t ctx, const ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return true;
    }

    return ggml_mem_ranges_add(ctx->mem_ranges, node);
}

static void ggml_metal_op_concurrency_check_fused(ggml_metal_op_t ctx, int idx, int n_consumed) {
    for (int k = 1; k < n_consumed; ++k) {
        if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + k))) {
            ggml_metal_op_concurrency_reset(ctx);
            return;
        }
    }
}

// Guard multi-op fusions against gallocr buffer reuse: when A -> n0 -> dst_t is fused,
// gallocr may free A after n0 and allocate dst_t overlapping A at a shifted offset, causing
// a cross-threadgroup read/write race inside a single fused dispatch.
static bool ggml_metal_tensor_overlaps_unsafe(
        const ggml_tensor * src, const ggml_tensor * dst, bool allow_exact_inplace) {
    if (!src || !dst) {
        return false;
    }
    const ggml_metal_buffer_id bs = ggml_metal_get_buffer_id(src);
    const ggml_metal_buffer_id bd = ggml_metal_get_buffer_id(dst);
    if (bs.metal != bd.metal) {
        return false;
    }
    const size_t s0 = bs.offs;
    const size_t s1 = s0 + ggml_nbytes(src);
    const size_t d0 = bd.offs;
    const size_t d1 = d0 + ggml_nbytes(dst);
    if (s1 <= d0 || d1 <= s0) {
        return false;
    }
    if (allow_exact_inplace && s0 == d0 && ggml_are_same_layout(src, dst)) {
        return false;
    }
    return true;
}

// Check whether `t` is either `target` itself or a single-use, zero-offset, full-size contiguous
// RESHAPE/VIEW chain rooted at `target` (none marked OUTPUT).
static bool ggml_metal_unwrap_zero_view(ggml_metal_op_t ctx, const ggml_tensor * t, const ggml_tensor * target) {
    if (!t || !target) {
        return false;
    }
    const size_t nbytes = ggml_nbytes(target);
    const ggml_tensor * cur = t;
    for (int depth = 0; depth < 4 && cur != target; ++depth) {
        if ((cur->op != GGML_OP_RESHAPE && cur->op != GGML_OP_VIEW) ||
            cur->view_src == nullptr || cur->view_offs != 0 ||
            !ggml_is_contiguous(cur) || ggml_nbytes(cur) != nbytes ||
            (cur->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
            ctx->tensor_use_count(cur) != 1) {
            return false;
        }
        cur = cur->view_src;
    }
    return cur == target;
}

static int ggml_metal_op_inkling_mix_impl(ggml_metal_op_t ctx, int idx_mix, ggml_tensor * pre_rms_op);

static int ggml_metal_op_encode_impl(ggml_metal_op_t ctx, int idx) {
    struct ggml_tensor * node = ctx->node(idx);

    //GGML_LOG_INFO("%s: encoding node %3d, op = %8s\n", __func__, idx, ggml_op_name(node->op));

    if (ggml_is_empty(node)) {
        return 1;
    }

    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
            {
                // noop -> next node
                if (ctx->debug_graph > 0) {
                    GGML_LOG_DEBUG("%s: node[%5d] - %-12s %s\n", __func__, idx, ggml_op_name(node->op), "(noop)");
                }
            } return 1;
        default:
            {
            } break;
    }

    if (!ggml_metal_device_supports_op(ctx->dev, node)) {
        GGML_LOG_ERROR("%s: error: unsupported op '%s'\n", __func__, ggml_op_desc(node));
        GGML_ABORT("unsupported op");
    }

    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return 1;
    }

    int n_fuse = 1;

    // check if the current node can run concurrently with other nodes before it
    // the condition is that:
    //  - the current node cannot write to any previous src or dst ranges
    //  - the current node cannot read from any previous dst ranges
    //
    // if the condition is not satisfied, we put a memory barrier and clear all ranges
    // otherwise, we add the new ranges to the encoding context and process the node concurrently
    //
    {
        const bool is_concurrent = ggml_metal_op_concurrency_check(ctx, node);

        if (!is_concurrent) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        if (ctx->debug_graph > 0) {
            GGML_LOG_DEBUG("%s: node[%5d] - %-12s %-12s %s\n", __func__, idx, ggml_op_name(node->op), ggml_get_name(node), is_concurrent ? "(concurrent)" : "");
        }
        if (ctx->debug_graph > 1) {
            GGML_TENSOR_LOCALS( int64_t, ne0, node->src[0], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb0, node->src[0], nb);
            GGML_TENSOR_LOCALS( int64_t, ne1, node->src[1], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb1, node->src[1], nb);
            GGML_TENSOR_LOCALS( int64_t, ne2, node->src[2], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb2, node->src[2], nb);
            GGML_TENSOR_LOCALS( int64_t, ne3, node->src[3], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb3, node->src[3], nb);
            GGML_TENSOR_LOCALS( int64_t, ne,  node,         ne);
            GGML_TENSOR_LOCALS(uint64_t, nb,  node,         nb);

            if (node->src[0]) {
                GGML_LOG_DEBUG("%s: src0 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[0]->type), ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03,
                        ggml_is_contiguous(node->src[0]), node->src[0]->name);
            }
            if (node->src[1]) {
                GGML_LOG_DEBUG("%s: src1 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[1]->type), ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13,
                        ggml_is_contiguous(node->src[1]), node->src[1]->name);
            }
            if (node->src[2]) {
                GGML_LOG_DEBUG("%s: src2 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[2]->type), ne20, ne21, ne22, ne23, nb20, nb21, nb22, nb23,
                        ggml_is_contiguous(node->src[2]), node->src[2]->name);
            }
            if (node->src[3]) {
                GGML_LOG_DEBUG("%s: src3 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[3]->type), ne30, ne31, ne32, ne33, nb30, nb31, nb32, nb33,
                        ggml_is_contiguous(node->src[3]), node->src[3]->name);
            }
            if (node) {
                GGML_LOG_DEBUG("%s: node  - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], 1, %s\n", __func__, ggml_type_name(node->type), ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                        node->name);
            }
        }
    }

    switch (node->op) {
        case GGML_OP_CONCAT:
            {
                n_fuse = ggml_metal_op_concat(ctx, idx);
            } break;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            {
                n_fuse = ggml_metal_op_bin(ctx, idx);
            } break;
        case GGML_OP_ADD_ID:
            {
                n_fuse = ggml_metal_op_add_id(ctx, idx);
            } break;
        case GGML_OP_REPEAT:
            {
                n_fuse = ggml_metal_op_repeat(ctx, idx);
            } break;
        case GGML_OP_ACC:
            {
                n_fuse = ggml_metal_op_acc(ctx, idx);
            } break;
        case GGML_OP_SCALE:
        case GGML_OP_FILL:
        case GGML_OP_CLAMP:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
        case GGML_OP_UNARY:
            {
                n_fuse = ggml_metal_op_unary(ctx, idx);
            } break;
        case GGML_OP_GLU:
            {
                n_fuse = ggml_metal_op_glu(ctx, idx);
            } break;
        case GGML_OP_SUM:
            {
                n_fuse = ggml_metal_op_sum(ctx, idx);
            } break;
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            {
                n_fuse = ggml_metal_op_sum_rows(ctx, idx);
            } break;
        case GGML_OP_CUMSUM:
            {
                n_fuse = ggml_metal_op_cumsum(ctx, idx);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                n_fuse = ggml_metal_op_soft_max(ctx, idx);
            } break;
        case GGML_OP_SSM_CONV:
            {
                n_fuse = ggml_metal_op_ssm_conv(ctx, idx);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                n_fuse = ggml_metal_op_ssm_scan(ctx, idx);
            } break;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            {
                n_fuse = ggml_metal_op_rwkv(ctx, idx);
            } break;
        case GGML_OP_GATED_DELTA_NET:
            {
                n_fuse = ggml_metal_op_gated_delta_net(ctx, idx);
            } break;
        case GGML_OP_SOLVE_TRI:
            {
                n_fuse = ggml_metal_op_solve_tri(ctx, idx);
            } break;
        case GGML_OP_MUL_MAT:
            {
                n_fuse = ggml_metal_op_mul_mat(ctx, idx);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                n_fuse = ggml_metal_op_mul_mat_id(ctx, idx);
            } break;
        case GGML_OP_GET_ROWS:
            {
                n_fuse = ggml_metal_op_get_rows(ctx, idx);
            } break;
        case GGML_OP_SET_ROWS:
            {
                n_fuse = ggml_metal_op_set_rows(ctx, idx);
            } break;
        case GGML_OP_DIAG:
            {
                n_fuse = ggml_metal_op_diag(ctx, idx);
            } break;
        case GGML_OP_L2_NORM:
            {
                n_fuse = ggml_metal_op_l2_norm(ctx, idx);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                n_fuse = ggml_metal_op_group_norm(ctx, idx);
            } break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            {
                n_fuse = ggml_metal_op_norm(ctx, idx);
            } break;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            {
                n_fuse = ggml_metal_op_rope(ctx, idx);
            } break;
        case GGML_OP_IM2COL:
            {
                n_fuse = ggml_metal_op_im2col(ctx, idx);
            } break;
        case GGML_OP_CONV_2D:
            {
                n_fuse = ggml_metal_op_conv_2d(ctx, idx);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                n_fuse = ggml_metal_op_conv_transpose_1d(ctx, idx);
            } break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_fuse = ggml_metal_op_conv_transpose_2d(ctx, idx);
            } break;
        case GGML_OP_CONV_3D:
            {
                n_fuse = ggml_metal_op_conv_3d(ctx, idx);
            } break;
        case GGML_OP_UPSCALE:
            {
                n_fuse = ggml_metal_op_upscale(ctx, idx);
            } break;
        case GGML_OP_PAD:
            {
                n_fuse = ggml_metal_op_pad(ctx, idx);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                n_fuse = ggml_metal_op_pad_reflect_1d(ctx, idx);
            } break;
        case GGML_OP_ROLL:
            {
                n_fuse = ggml_metal_op_roll(ctx, idx);
            } break;
        case GGML_OP_ARANGE:
            {
                n_fuse = ggml_metal_op_arange(ctx, idx);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                n_fuse = ggml_metal_op_timestep_embedding(ctx, idx);
            } break;
        case GGML_OP_ARGSORT:
            {
                n_fuse = ggml_metal_op_argsort(ctx, idx);
            } break;
        case GGML_OP_TOP_K:
            {
                n_fuse = ggml_metal_op_top_k(ctx, idx);
            } break;
        case GGML_OP_TRI:
            {
                n_fuse = ggml_metal_op_tri(ctx, idx);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                n_fuse = ggml_metal_op_flash_attn_ext(ctx, idx);
            } break;
        case GGML_OP_SET:
            {
                n_fuse = ggml_metal_op_set(ctx, idx);
            } break;
        case GGML_OP_DUP:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            {
                n_fuse = ggml_metal_op_cpy(ctx, idx);
            } break;
        case GGML_OP_POOL_1D:
            {
                n_fuse = ggml_metal_op_pool_1d(ctx, idx);
            } break;
        case GGML_OP_POOL_2D:
            {
                n_fuse = ggml_metal_op_pool_2d(ctx, idx);
            } break;
        case GGML_OP_ARGMAX:
            {
                n_fuse = ggml_metal_op_argmax(ctx, idx);
            } break;
        case GGML_OP_DSV4_HC_COMB:
            {
                n_fuse = ggml_metal_op_dsv4_hc_comb(ctx, idx);
            } break;
        case GGML_OP_DSV4_HC_PRE:
            {
                n_fuse = ggml_metal_op_dsv4_hc_pre(ctx, idx);
            } break;
        case GGML_OP_DSV4_HC_POST:
            {
                n_fuse = ggml_metal_op_dsv4_hc_post(ctx, idx);
            } break;
        case GGML_OP_INKLING_SCONV:
            {
                n_fuse = ggml_metal_op_inkling_sconv(ctx, idx);
            } break;
        case GGML_OP_INKLING_SCONV_STATE:
            {
                n_fuse = ggml_metal_op_inkling_sconv_state(ctx, idx);
            } break;
        case GGML_OP_INKLING_MIX:
            {
                n_fuse = ggml_metal_op_inkling_mix(ctx, idx);
            } break;
        case GGML_OP_INKLING_CTRL_BLEND:
            {
                n_fuse = ggml_metal_op_inkling_ctrl_blend(ctx, idx);
            } break;
        case GGML_OP_INKLING_MOE_IDS:
            {
                n_fuse = ggml_metal_op_inkling_moe_ids(ctx, idx);
            } break;
        case GGML_OP_INKLING_MOE_WEIGHTS:
            {
                n_fuse = ggml_metal_op_inkling_moe_weights(ctx, idx);
            } break;
        case GGML_OP_INKLING_MOE_REDUCE:
            {
                n_fuse = ggml_metal_op_inkling_moe_reduce(ctx, idx);
            } break;
        case GGML_OP_OPT_STEP_ADAMW:
            {
                n_fuse = ggml_metal_op_opt_step_adamw(ctx, idx);
            } break;
        case GGML_OP_OPT_STEP_SGD:
            {
                n_fuse = ggml_metal_op_opt_step_sgd(ctx, idx);
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                n_fuse = ggml_metal_op_count_equal(ctx, idx);
            } break;
        default:
            {
                GGML_LOG_ERROR("%s: error: node %3d, op = %8s not implemented\n", __func__, idx, ggml_op_name(node->op));
                GGML_ABORT("fatal error");
            }
    }

    if (ctx->debug_graph > 0) {
        if (n_fuse > 1) {
            GGML_LOG_DEBUG("%s:               fuse %d ops\n", __func__, n_fuse);
        }
    }

    // update the mem ranges in the encoding context
    for (int i = 0; i < n_fuse; ++i) {
        if (!ggml_metal_op_concurrency_add(ctx, ctx->node(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);
        }
    }

    return n_fuse;
}

int ggml_metal_op_encode(ggml_metal_op_t ctx, int idx) {
    if (ctx->use_capture) {
        ggml_metal_encoder_debug_group_push(ctx->enc, ggml_op_desc(ctx->node(idx)));
    }

    int res = ggml_metal_op_encode_impl(ctx, idx);
    if (idx + res > ctx->n_nodes()) {
        GGML_ABORT("fusion error: nodes spanning multiple encoders have been fused. this indicates a bug in the fusion logic %s",
                "https://github.com/ggml-org/llama.cpp/pull/14849");
    }

    if (ctx->use_capture) {
        ggml_metal_encoder_debug_group_pop(ctx->enc);
    }

    return res;
}

// Fused (optional REPEAT) + CONCAT(a, b, dim=0) + MUL_MAT(BF16 [2*n_embd, n_out], concat) for Inkling loop_ctrl_down.
static int try_fuse_inkling_ctrl_down(ggml_metal_op_t ctx, int idx, bool is_repeat) {
    if (!ctx->use_fusion) {
        return 0;
    }
    const auto * props_dev = ggml_metal_device_get_props(ctx->dev);
    if (!props_dev->has_bfloat) {
        return 0;
    }

    ggml_tensor * a    = nullptr;
    ggml_tensor * b    = nullptr;
    ggml_tensor * n_mm = nullptr;
    int bcast_a        = 0;
    int n_consumed     = 0;

    if (is_repeat) {
        if (idx + 2 >= ctx->n_nodes() || !ctx->node_has_n_uses(idx, 1) || !ctx->node_has_n_uses(idx + 1, 1)) {
            return 0;
        }
        ggml_tensor * n_rep = ctx->node(idx);
        ggml_tensor * n_cat = ctx->node(idx + 1);
        n_mm                = ctx->node(idx + 2);
        if (n_cat->op != GGML_OP_CONCAT || ((const int32_t *) n_cat->op_params)[0] != 0 || n_cat->src[0] != n_rep) {
            return 0;
        }
        if (n_mm->op != GGML_OP_MUL_MAT || n_mm->src[1] != n_cat) {
            return 0;
        }
        a = n_rep->src[0];
        b = n_cat->src[1];
        if (a->ne[0] != b->ne[0] || a->ne[1] != b->ne[1] || a->ne[2] != 1 || a->ne[3] != 1 || b->ne[3] != 1) {
            return 0;
        }
        bcast_a    = 1;
        n_consumed = 3;
    } else {
        if (idx + 1 >= ctx->n_nodes() || !ctx->node_has_n_uses(idx, 1)) {
            return 0;
        }
        ggml_tensor * n_cat = ctx->node(idx);
        n_mm                = ctx->node(idx + 1);
        if (((const int32_t *) n_cat->op_params)[0] != 0 || n_mm->op != GGML_OP_MUL_MAT || n_mm->src[1] != n_cat) {
            return 0;
        }
        a = n_cat->src[0];
        b = n_cat->src[1];
        if (a->ne[0] != b->ne[0] || a->ne[1] != b->ne[1] || a->ne[2] != b->ne[2] || a->ne[3] != 1 || b->ne[3] != 1) {
            return 0;
        }
        bcast_a    = 0;
        n_consumed = 2;
    }

    ggml_tensor * w = n_mm->src[0];
    if (w->type != GGML_TYPE_BF16 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || n_mm->type != GGML_TYPE_F32) {
        return 0;
    }
    if ((a->ne[0] % 4) != 0 || w->ne[0] != 2 * a->ne[0] || w->ne[1] > 128 || w->ne[2] != b->ne[2] || w->ne[3] != 1) {
        return 0;
    }
    if (b->ne[1] < 1 || b->ne[1] > 16) {
        return 0;
    }
    if (w->nb[0] != sizeof(uint16_t) || a->nb[0] != sizeof(float) || b->nb[0] != sizeof(float) || n_mm->nb[0] != sizeof(float)) {
        return 0;
    }
    if (ggml_metal_tensor_overlaps_unsafe(a, n_mm, false) || ggml_metal_tensor_overlaps_unsafe(b, n_mm, false)) {
        return 0;
    }

    ggml_metal_op_concurrency_check_fused(ctx, idx, n_consumed);

    ggml_metal_kargs_inkling_ctrl_down args = {
        /*.n_embd   =*/ (int32_t) a->ne[0],
        /*.n_out    =*/ (int32_t) w->ne[1],
        /*.n_tokens =*/ (int32_t) b->ne[1],
        /*.n_ctrl   =*/ (int32_t) b->ne[2],
        /*.bcast_a  =*/ bcast_a,
        /*.nbw1     =*/ w->nb[1],
        /*.nbw2     =*/ w->nb[2],
        /*.nba1     =*/ a->nb[1],
        /*.nba2     =*/ a->nb[2],
        /*.nbb1     =*/ b->nb[1],
        /*.nbb2     =*/ b->nb[2],
        /*.nbd1     =*/ n_mm->nb[1],
        /*.nbd2     =*/ n_mm->nb[2],
    };

    auto pl = ggml_metal_library_get_pipeline_inkling_ctrl_down(ctx->lib, (int) b->ne[1]);
    ggml_metal_encoder_set_pipeline(ctx->enc, pl);
    ggml_metal_encoder_set_bytes   (ctx->enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(w),    1);
    ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(a),    2);
    ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(b),    3);
    ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(n_mm), 4);
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc, (int) w->ne[1], 1, (int) b->ne[2], 32, 2, 1);

    return n_consumed;
}

int ggml_metal_op_concat(ggml_metal_op_t ctx, int idx) {
    if (const int fused = try_fuse_inkling_ctrl_down(ctx, idx, false)) {
        return fused;
    }

    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t dim = ((const int32_t *) op->op_params)[0];

    ggml_metal_kargs_concat args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.dim  =*/ dim,
    };

    auto pipeline = ggml_metal_library_get_pipeline_base(lib, GGML_OP_CONCAT);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_repeat(ggml_metal_op_t ctx, int idx) {
    if (const int fused = try_fuse_inkling_ctrl_down(ctx, idx, true)) {
        return fused;
    }

    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_repeat(lib, op->type);

    ggml_metal_kargs_repeat args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_acc(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[1]));

    const size_t pnb1 = ((const int32_t *) op->op_params)[0];
    const size_t pnb2 = ((const int32_t *) op->op_params)[1];
    const size_t pnb3 = ((const int32_t *) op->op_params)[2];
    const size_t offs = ((const int32_t *) op->op_params)[3];

    const bool inplace = (bool) ((const int32_t *) op->op_params)[4];

    if (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

        ggml_metal_kargs_cpy args = {
            /*.nk0  =*/ ne00,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne10,
        /*.ne01 =*/ ne11,
        /*.ne02 =*/ ne12,
        /*.ne03 =*/ ne13,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ pnb1,
        /*.nb02 =*/ pnb2,
        /*.nb03 =*/ pnb3,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne10,
        /*.ne1  =*/ ne11,
        /*.ne2  =*/ ne12,
        /*.ne3  =*/ ne13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
        /*.offs =*/ offs,
        /*.o1   =*/ { 0 },
    };

    auto pipeline = ggml_metal_library_get_pipeline_bin_one(lib, GGML_OP_ADD);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    int nth = 1;

    while (2*nth < args.ne0 && nth < nth_max) {
        nth *= 2;
    }

    ggml_metal_encoder_dispatch_threadgroups(enc, ne11, ne12, ne13, nth, 1, 1);

    return 1;
}

int ggml_metal_op_unary(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_unary args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.slope =*/ 0.0,
        /*.scale =*/ 0.0,
        /*.bias  =*/ 0.0,
        /*.val   =*/ 0.0,
        /*.min   =*/ 0.0,
        /*.max   =*/ 0.0,
    };

    if (op->op == GGML_OP_LEAKY_RELU) {
        args.slope = ggml_get_op_params_f32(op, 0);
    }

    if (op->op == GGML_OP_SCALE) {
        args.scale = ggml_get_op_params_f32(op, 0);
        args.bias  = ggml_get_op_params_f32(op, 1);
    }

    if (op->op == GGML_OP_FILL) {
        args.val = ggml_get_op_params_f32(op, 0);
    }

    if (op->op == GGML_OP_CLAMP) {
        args.min = ggml_get_op_params_f32(op, 0);
        args.max = ggml_get_op_params_f32(op, 1);
    }

    if (op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
        args.slope = ggml_get_op_params_f32(op, 1); // alpha_n
        args.scale = ggml_get_op_params_f32(op, 2); // alpha_p
        args.bias  = ggml_get_op_params_f32(op, 3); // beta
        args.val   = ggml_get_op_params_f32(op, 4); // eps
    }

    if (op->op == GGML_OP_SCALE && op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
        ne02 == 1 && ne03 == 1 && args.bias == 0.0f && ctx->use_fusion) {
        const ggml_op ops_sc[3] = { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE };
        if (ctx->can_fuse(idx, ops_sc, 3)) {
            ggml_tensor * n1 = ctx->node(idx + 1);
            ggml_tensor * n2 = ctx->node(idx + 2);
            if (ggml_get_unary_op(n1) == GGML_UNARY_OP_TANH &&
                n1->src[0] == op && n2->src[0] == n1 &&
                n1->type == GGML_TYPE_F32 && n2->type == GGML_TYPE_F32 &&
                ggml_get_op_params_f32(n2, 1) == 0.0f &&
                ggml_are_same_shape(n1, op) && ggml_are_same_shape(n2, op) &&
                ggml_is_contiguous_rows(n2) &&
                !ggml_metal_tensor_overlaps_unsafe(op->src[0], n2, true)) {
                ggml_metal_op_concurrency_check_fused(ctx, idx, 3);
                auto pl = ggml_metal_library_get_pipeline_softcap(lib);
                args.slope = ggml_get_op_params_f32(n2, 0);
                args.nb1   = n2->nb[1];
                const int n_work = (ne00 + 3) / 4;
                const int nth = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pl));
                ggml_metal_encoder_set_pipeline(enc, pl);
                ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(n2), 2);
                ggml_metal_encoder_dispatch_threadgroups(enc, (n_work + nth - 1) / nth, ne01, 1, nth, 1, 1);
                return 3;
            }
        }
    }

    if (op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == GGML_UNARY_OP_SIGMOID &&
        op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
        ne02 == 1 && ne03 == 1 && ctx->use_fusion) {
        const ggml_op ops_ms[2] = { GGML_OP_UNARY, GGML_OP_MUL };
        if (ctx->can_fuse(idx, ops_ms, 2)) {
            ggml_tensor * next  = ctx->node(idx + 1);
            ggml_tensor * other = (next->src[0] == op) ? next->src[1] : ((next->src[1] == op) ? next->src[0] : nullptr);
            if (other && other->type == GGML_TYPE_F32 && next->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(other, op) && ggml_are_same_shape(next, op) &&
                ggml_is_contiguous_rows(other) && ggml_is_contiguous_rows(next)) {
                // F4C: UNARY(sigmoid) -> MUL -> MUL_MAT(BF16, ne01 <= 2560, B in 3..16)
                const auto * props_dev = ggml_metal_device_get_props(ctx->dev);
                if (props_dev->has_bfloat && props_dev->has_simdgroup_mm &&
                    idx + 2 < ctx->n_nodes() && ctx->node_has_n_uses(idx + 1, 1) &&
                    ne1 >= 3 && ne1 <= 16) {
                    ggml_tensor * mm = ctx->node(idx + 2);
                    if (mm->op == GGML_OP_MUL_MAT && mm->src[1] == next &&
                        mm->src[0]->type == GGML_TYPE_BF16 && mm->type == GGML_TYPE_F32 &&
                        !ggml_is_transposed(mm->src[0]) &&
                        (mm->src[0]->ne[0] % 64 == 0) &&
                        mm->src[0]->ne[1] > 384 && mm->src[0]->ne[1] <= 2560 &&
                        mm->src[0]->ne[2] == 1 && mm->src[0]->ne[3] == 1 &&
                        mm->ne[2] == 1 && mm->ne[3] == 1 &&
                        ggml_is_contiguous(mm) &&
                        !ggml_metal_tensor_overlaps_unsafe(mm->src[0], mm, false) &&
                        !ggml_metal_tensor_overlaps_unsafe(other,      mm, false) &&
                        !ggml_metal_tensor_overlaps_unsafe(op->src[0], mm, false)) {
                        ggml_metal_op_concurrency_check_fused(ctx, idx, 3);
                        auto pl_mm = ggml_metal_library_get_pipeline_mul_mm_bf16_f32act_siggate(lib, ne1);
                        ggml_metal_kargs_mul_mm_bf16_siggate sargs = {
                            /*.ne00 =*/ (int32_t) mm->src[0]->ne[0],
                            /*.ne01 =*/ (int32_t) mm->src[0]->ne[1],
                            /*.ne11 =*/ (int32_t) ne1,
                            /*.ne0  =*/ (int32_t) mm->ne[0],
                            /*.nb01 =*/ mm->src[0]->nb[1],
                            /*.nb11 =*/ other->nb[1],
                            /*.nbg1 =*/ nb01,
                        };
                        const ggml_metal_buffer_id bid_mm = ggml_metal_get_buffer_id(mm);
                        ggml_metal_buffer_id bid_out = bid_mm;
                        int sk = 1;
                        int sk_slot = -1;
                        if (sargs.ne00 >= 1536 && sargs.ne01 <= 2560) {
                            const int try_sk = 3;
                            const size_t sk_bytes = (size_t) try_sk * (size_t) sargs.ne0 * (size_t) sargs.ne11 * sizeof(float);
                            if (ggml_metal_op_alloc_sk_buf(ctx, sk_bytes, &bid_out, &sk_slot)) {
                                sk = try_sk;
                            }
                        }
                        ggml_metal_encoder_set_pipeline(enc, pl_mm);
                        ggml_metal_encoder_set_bytes   (enc, &sargs, sizeof(sargs), 0);
                        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mm->src[0]), 1);
                        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(other),      2);
                        ggml_metal_encoder_set_buffer  (enc, bid_src0,                             3);
                        ggml_metal_encoder_set_buffer  (enc, bid_out,                              4);
                        ggml_metal_encoder_set_threadgroup_memory_size(enc, pl_mm.smem, 0);
                        ggml_metal_encoder_dispatch_threadgroups(enc, (sargs.ne01 + pl_mm.nr0 - 1) / pl_mm.nr0, sk, 1, 32, pl_mm.nsg, 1);
                        if (sk > 1) {
                            ggml_metal_op_dispatch_sk_reduce(ctx, bid_out, bid_mm, sargs.ne0 * sargs.ne11, sk, sk_slot);
                        }
                        return 3;
                    }
                }
                if (!ggml_metal_tensor_overlaps_unsafe(op->src[0], next, true) &&
                    !ggml_metal_tensor_overlaps_unsafe(other,      next, true)) {
                    ggml_metal_op_concurrency_check_fused(ctx, idx, 2);
                    const bool is_c4 = (ne0 % 4 == 0);
                    auto pl = ggml_metal_library_get_pipeline_mul_sigmoid(lib, is_c4);
                    ggml_metal_kargs_mul_sigmoid ms_args = {
                        /*.ne0   =*/ ne0,
                        /*.ne1   =*/ ne1,
                        /*.nb_x1 =*/ other->nb[1],
                        /*.nb_g1 =*/ nb01,
                        /*.nb_d1 =*/ next->nb[1],
                    };
                    const int n_work = is_c4 ? (ne0 / 4) : ne0;
                    const int nth = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pl));
                    ggml_metal_encoder_set_pipeline(enc, pl);
                    ggml_metal_encoder_set_bytes   (enc, &ms_args, sizeof(ms_args), 0);
                    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(other), 1);
                    ggml_metal_encoder_set_buffer  (enc, bid_src0,                        2);
                    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(next),  3);
                    ggml_metal_encoder_dispatch_threadgroups(enc, (n_work + nth - 1) / nth, ne1, 1, nth, 1, 1);
                    return 2;
                }
            }
        }
    }

    auto pipeline = ggml_metal_library_get_pipeline_unary(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    if (pipeline.cnt) {
        const int n = pipeline.c4 ? ggml_nelements(op)/4 : ggml_nelements(op);

        const int nth = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_dispatch_threadgroups(enc, (n + nth - 1)/nth, 1, 1, nth, 1, 1);
    } else {
        const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        const int nth = MIN(args.ne00, nth_max);

        const int nk0 = (args.ne00 + nth - 1)/nth;

        ggml_metal_encoder_dispatch_threadgroups(enc, nk0*ne01, ne02, ne03, nth, 1, 1);
    }

    return 1;
}

int ggml_metal_op_glu(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    if (op->src[1]) {
        GGML_ASSERT(ggml_are_same_shape(op->src[0], op->src[1]));
    }

    auto pipeline = ggml_metal_library_get_pipeline_glu(lib, op);

    const int32_t swp = ggml_get_op_params_i32(op, 1);
    const float alpha = ggml_get_op_params_f32(op, 2);
    const float limit = ggml_get_op_params_f32(op, 3);

    const int32_t i00 = swp ? ne0 : 0;
    const int32_t i10 = swp ? 0 : ne0;

    const bool has_scales = op->src[2] != nullptr && op->src[3] != nullptr;

    ggml_metal_kargs_glu args = {
        /*.ne00       =*/ ne00,
        /*.nb01       =*/ nb01,
        /*.ne10       =*/ op->src[1] ? ne10 : ne00,
        /*.nb11       =*/ op->src[1] ? nb11 : nb01,
        /*.ne0        =*/ ne0,
        /*.ne1        =*/ ne1,
        /*.nb1        =*/ nb1,
        /*.i00        =*/ op->src[1] ? 0 : i00,
        /*.i10        =*/ op->src[1] ? 0 : i10,
        /*.alpha      =*/ alpha,
        /*.limit      =*/ limit,
        /*.nb30       =*/ has_scales ? op->src[3]->nb[0] : 0,
        /*.nb31       =*/ has_scales ? op->src[3]->nb[1] : 0,
        /*.has_scales =*/ has_scales ? 1 : 0,
    };

    const int64_t nrows = ggml_nrows(op->src[0]);

    const int32_t max_nth = ggml_get_glu_op(op) == GGML_GLU_OP_SITU
        ? std::min(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline))
        : ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    const int32_t nth = std::max(1, std::min(max_nth, ne0));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    if (op->src[1]) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 2);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);
    if (ggml_get_glu_op(op) == GGML_GLU_OP_SITU) {
        ggml_metal_encoder_set_buffer(enc, has_scales ? ggml_metal_get_buffer_id(op->src[2]) : bid_src0, 4);
        ggml_metal_encoder_set_buffer(enc, has_scales ? ggml_metal_get_buffer_id(op->src[3]) : bid_src0, 5);
    }

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_sum(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op  = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const uint64_t n = (uint64_t) ggml_nelements(op->src[0]);

    ggml_metal_kargs_sum args = {
        /*.np =*/ n,
    };

    auto pipeline = ggml_metal_library_get_pipeline_sum(lib, op);

    int nth = 32; // SIMD width

    while (nth < (int) n && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (int) n);

    const int nsg = (nth + 31) / 32;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, nsg * sizeof(float), 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_sum_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_sum_rows args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_sum_rows(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    int nth = 32; // SIMD width

    while (nth < args.ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (int) args.ne00);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_cumsum(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline_blk = ggml_metal_library_get_pipeline_cumsum_blk(lib, op);

    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_blk)) {
        nth *= 2;
    }

    GGML_ASSERT(ne00 <= nth*nth);

    const int64_t net0 = (ne00 + nth - 1) / nth;
    const int64_t net1 = ne01;
    const int64_t net2 = ne02;
    const int64_t net3 = ne03;

    const uint64_t nbt0 = sizeof(float);
    const uint64_t nbt1 = net0*nbt0;
    const uint64_t nbt2 = net1*nbt1;
    const uint64_t nbt3 = net2*nbt2;

    const size_t smem = GGML_PAD(32*sizeof(float), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += ggml_nbytes(op);

    {
        ggml_metal_kargs_cumsum_blk args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.net0 =*/ net0,
            /*.net1 =*/ net1,
            /*.net2 =*/ net2,
            /*.net3 =*/ net3,
            /*.nbt0 =*/ nbt0,
            /*.nbt1 =*/ nbt1,
            /*.nbt2 =*/ nbt2,
            /*.nbt3 =*/ nbt3,
            /*.outb =*/ ne00 > nth,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
    }

    if (ne00 > nth) {
        ggml_metal_op_concurrency_reset(ctx);

        {
            ggml_metal_kargs_cumsum_blk args = {
                /*.ne00 =*/ net0,
                /*.ne01 =*/ net1,
                /*.ne02 =*/ net2,
                /*.ne03 =*/ net3,
                /*.nb00 =*/ nbt0,
                /*.nb01 =*/ nbt1,
                /*.nb02 =*/ nbt2,
                /*.nb03 =*/ nbt3,
                /*.net0 =*/ net0,
                /*.net1 =*/ net1,
                /*.net2 =*/ net2,
                /*.net3 =*/ net3,
                /*.nbt0 =*/ nbt0,
                /*.nbt1 =*/ nbt1,
                /*.nbt2 =*/ nbt2,
                /*.nbt3 =*/ nbt3,
                /*.outb =*/ false,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 3);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, net1, net2, net3, nth, 1, 1);
        }

        ggml_metal_op_concurrency_reset(ctx);

        {
            auto pipeline_add = ggml_metal_library_get_pipeline_cumsum_add(lib, op);

            ggml_metal_kargs_cumsum_add args = {
                /*.ne00 =*/ ne00,
                /*.ne01 =*/ ne01,
                /*.ne02 =*/ ne02,
                /*.ne03 =*/ ne03,
                /*.nb00 =*/ nb00,
                /*.nb01 =*/ nb01,
                /*.nb02 =*/ nb02,
                /*.nb03 =*/ nb03,
                /*.net0 =*/ net0,
                /*.net1 =*/ net1,
                /*.net2 =*/ net2,
                /*.net3 =*/ net3,
                /*.nbt0 =*/ nbt0,
                /*.nbt1 =*/ nbt1,
                /*.nbt2 =*/ nbt2,
                /*.nbt3 =*/ nbt3,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline_add);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

            ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
        }
    }

    return 1;
}

int ggml_metal_op_get_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const auto * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // Fused DSpark Markov GRU step (rank = 256):
    //   Non-empty nodes: GET_ROWS(w1, prev) -> CONCAT(state, w1) -> MUL_MAT(wstep[512,768]) -> ADD(ctx_s)
    //   -> TANH(view(raw, 512)) [-> TANH(view(raw, 256)) -> SIGMOID(view(raw, 0)) -> SUB -> MUL -> ADD(state_out)]
    if (ctx->use_fusion && props_dev->has_bfloat &&
        idx + 4 < ctx->n_nodes() &&
        op->src[0]->type == GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_I32 &&
        ne00 == 256 && nb00 == sizeof(uint16_t) && ggml_nelements(op->src[1]) == 1 &&
        ctx->node_has_n_uses(idx, 1)) {
        ggml_tensor * n1 = ctx->node(idx + 1);
        ggml_tensor * n2 = ctx->node(idx + 2);
        ggml_tensor * n3 = ctx->node(idx + 3);
        ggml_tensor * n4 = ctx->node(idx + 4);
        if (n1->op == GGML_OP_CONCAT && n1->src[1] == op &&
            n1->src[0]->type == GGML_TYPE_F32 && n1->src[0]->ne[0] == 256 && ggml_is_contiguous(n1->src[0]) &&
            n1->ne[0] == 512 && n1->ne[1] == 1 && ctx->node_has_n_uses(idx + 1, 1) &&
            n2->op == GGML_OP_MUL_MAT && n2->src[1] == n1 &&
            n2->src[0]->type == GGML_TYPE_BF16 && n2->src[0]->ne[0] == 512 && n2->src[0]->ne[1] == 768 &&
            n2->src[0]->nb[0] == sizeof(uint16_t) && n2->ne[1] == 1 && ctx->node_has_n_uses(idx + 2, 1) &&
            n3->op == GGML_OP_ADD && (n3->src[0] == n2 || n3->src[1] == n2) &&
            n4->op == GGML_OP_UNARY && ggml_get_unary_op(n4) == GGML_UNARY_OP_TANH &&
            n4->src[0]->view_src == n3 && n4->src[0]->view_offs == 512 * sizeof(float) && n4->src[0]->ne[0] == 256 &&
            n4->type == GGML_TYPE_F32 && ggml_is_contiguous(n4)) {
            ggml_tensor * state_in = n1->src[0];
            ggml_tensor * ctx_s    = n3->src[0] == n2 ? n3->src[1] : n3->src[0];
            if (ctx_s->type == GGML_TYPE_F32 && ggml_nelements(ctx_s) == 768 && ggml_is_contiguous(ctx_s)) {
                ggml_tensor * state_out = nullptr;
                int n_consumed = 0;
                if (idx + 9 < ctx->n_nodes()) {
                    ggml_tensor * n5 = ctx->node(idx + 5);
                    ggml_tensor * n6 = ctx->node(idx + 6);
                    ggml_tensor * n7 = ctx->node(idx + 7);
                    ggml_tensor * n8 = ctx->node(idx + 8);
                    ggml_tensor * n9 = ctx->node(idx + 9);
                    if (ctx->node_has_n_uses(idx + 3, 3) &&
                        n5->op == GGML_OP_UNARY && ggml_get_unary_op(n5) == GGML_UNARY_OP_TANH &&
                        n5->src[0]->view_src == n3 && n5->src[0]->view_offs == 256 * sizeof(float) && n5->src[0]->ne[0] == 256 &&
                        ctx->node_has_n_uses(idx + 5, 2) &&
                        n6->op == GGML_OP_UNARY && ggml_get_unary_op(n6) == GGML_UNARY_OP_SIGMOID &&
                        n6->src[0]->view_src == n3 && n6->src[0]->view_offs == 0 && n6->src[0]->ne[0] == 256 &&
                        ctx->node_has_n_uses(idx + 6, 1) &&
                        n7->op == GGML_OP_SUB && n7->src[0] == state_in && n7->src[1] == n5 && ctx->node_has_n_uses(idx + 7, 1) &&
                        n8->op == GGML_OP_MUL && n8->src[0] == n6 && n8->src[1] == n7 && ctx->node_has_n_uses(idx + 8, 1) &&
                        n9->op == GGML_OP_ADD && n9->src[0] == n5 && n9->src[1] == n8 &&
                        n9->type == GGML_TYPE_F32 && ggml_is_contiguous(n9)) {
                        state_out  = n9;
                        n_consumed = 10;
                    }
                }
                if (n_consumed == 0 && ctx->node_has_n_uses(idx + 3, 1)) {
                    n_consumed = 5;
                }
                if (n_consumed > 0) {
                    ggml_metal_op_concurrency_check_fused(ctx, idx, n_consumed);
                    auto pipeline = ggml_metal_library_get_pipeline_dspark_markov_step(lib);

                    // Gate speculative Markov GRU steps on the same ctrl[gate_idx] threshold as dspark_vocab_argmax
                    // so pruned alt branches and low-confidence trunk tails skip reading the 768 KB BF16 wstep matrix.
                    int32_t gate_idx = -1;
                    int32_t has_ctrl = 0;
                    ggml_metal_buffer_id bid_ctrl = { nullptr, 0 };
                    {
                        ggml_tensor * prev_base = op->src[1];
                        size_t prev_offs = 0;
                        while (prev_base->view_src) {
                            prev_offs += prev_base->view_offs;
                            prev_base = prev_base->view_src;
                        }
                        if (prev_base->type == GGML_TYPE_I32 && ggml_nelements(prev_base) == 152) {
                            const int32_t prev_slot = (int32_t) (prev_offs / sizeof(int32_t));
                            switch (prev_slot) {
                                case 0: case 1: case 2: case 3: case 4:
                                case 5: case 6: case 7: case 8: case 9:
                                case 10: case 11: case 12: case 13:      gate_idx =  2; break;
                                case 15: case 16: case 17: case 18: case 19: gate_idx = 3; break;
                                case 21: case 22: case 23:               gate_idx =  4; break;
                                case 25: case 26: case 27: case 28:      gate_idx =  5; break;
                                case 30: case 31: case 32:               gate_idx =  6; break;
                                case 34: case 35:                        gate_idx =  7; break;
                                case 37:                                 gate_idx =  8; break;
                                case 39: case 40:                        gate_idx =  9; break;
                                case 42:                                 gate_idx = 10; break;
                                default:                                 gate_idx = -1; break;
                            }
                            if (gate_idx >= 2) {
                                has_ctrl = 1;
                                bid_ctrl = ggml_metal_get_buffer_id(prev_base);
                                bid_ctrl.offs += (size_t) 88 * sizeof(int32_t);
                            }
                        }
                    }

                    ggml_metal_kargs_dspark_markov_step args = {
                        /*.rank        =*/ 256,
                        /*.write_state =*/ state_out ? 1 : 0,
                        /*.gate_idx    =*/ gate_idx,
                        /*.has_ctrl    =*/ has_ctrl,
                        /*.nb_w1       =*/ nb01,
                        /*.nb_wstep    =*/ n2->src[0]->nb[1],
                    };
                    ggml_metal_buffer_id bid_lat = ggml_metal_get_buffer_id(n4);
                    ggml_metal_encoder_set_pipeline(enc, pipeline);
                    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
                    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
                    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
                    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(state_in),   3);
                    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(n2->src[0]), 4);
                    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(ctx_s),      5);
                    ggml_metal_encoder_set_buffer  (enc, bid_lat,                              6);
                    ggml_metal_encoder_set_buffer  (enc, state_out ? ggml_metal_get_buffer_id(state_out) : bid_lat, 7);
                    ggml_metal_encoder_set_buffer  (enc, has_ctrl != 0 ? bid_ctrl : bid_lat,   8);
                    ggml_metal_encoder_dispatch_threadgroups(enc, 8, 1, 1, 32, 4, 1);
                    return n_consumed;
                }
            }
        }
    }

    auto pipeline = ggml_metal_library_get_pipeline_get_rows(lib, op->src[0]->type);

    ggml_metal_kargs_get_rows args = {
        /*.ne00t =*/ ggml_is_quantized(op->src[0]->type) ? ne00/16 : ne00,
        /*.ne00  =*/ ne00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne10  =*/ ne10,
        /*.nb10  =*/ nb10,
        /*.nb11  =*/ nb11,
        /*.nb12  =*/ nb12,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
    };

    const int nth = std::min(args.ne00t, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const int nw0 = (args.ne00t + nth - 1)/nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*ne10, ne11, ne12, nth, 1, 1);

    return 1;
}

int ggml_metal_op_set_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_set_rows(lib, op->src[1]->type, op->type);

    const int32_t nk0 = ne0/ggml_blck_size(op->type);

    int nth = 32; // SIMD width

    while (nth < nk0 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    int nrptg = 1;
    if (nth > nk0) {
        nrptg = (nth + nk0 - 1)/nk0;
        nth   = nk0;

        if (nrptg*nth > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nrptg--;
        }
    }

    nth = std::min(nth, nk0);

    ggml_metal_kargs_set_rows args = {
        /*.nk0  =*/ nk0,
        /*.ne01 =*/ ne01,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_diag(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t,  ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(int32_t,  ne, op, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb, op, nb);

    ggml_metal_kargs_diag args = {
        /*.ne00 =*/ne00,
        /*.ne01 =*/ne01,
        /*.ne02 =*/ne02,
        /*.ne03 =*/ne03,
        /*.nb00 =*/nb00,
        /*.nb01 =*/nb01,
        /*.nb02 =*/nb02,
        /*.nb03 =*/nb03,
        /*.ne0  =*/ne0,
        /*.ne1  =*/ne1,
        /*.ne2  =*/ne2,
        /*.ne3  =*/ne3,
        /*.nb0  =*/nb0,
        /*.nb1  =*/nb1,
        /*.nb2  =*/nb2,
        /*.nb3  =*/nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_diag(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, 32, 1, 1);

    return 1;
}

int ggml_metal_op_soft_max(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float scale;
    float max_bias;

    memcpy(&scale,    ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias, ((const int32_t *) op->op_params) + 1, sizeof(max_bias));

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // softmax

    ggml_metal_kargs_soft_max args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne11        =*/ ne11,
        /*.ne12        =*/ ne12,
        /*.ne13        =*/ ne13,
        /*.nb11        =*/ nb11,
        /*.nb12        =*/ nb12,
        /*.nb13        =*/ nb13,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.scale       =*/ scale,
        /*.max_bias    =*/ max_bias,
        /*.m0          =*/ m0,
        /*.m1          =*/ m1,
        /*.n_head_log2 =*/ n_head_log2,
    };

    auto pipeline = ggml_metal_library_get_pipeline_soft_max(lib, op);

    int nth = 32; // SIMD width

    if (ne00%4 == 0) {
        while (nth < ne00/4 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    } else {
        while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    }

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->src[1]) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    if (op->src[2]) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_ssm_conv(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_ssm_conv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
    };

    // Use batched kernel for prefill (ne1 > 1) to reduce threadgroup dispatch overhead
    const bool use_batched = (ne1 > 1);

    if (use_batched) {
        // Determine the smallest power of 2 that's >= ne1, but <= 256
        int BATCH_SIZE;
        if      (ne1 > 128) BATCH_SIZE = 256;
        else if (ne1 > 64 ) BATCH_SIZE = 128;
        else if (ne1 > 32 ) BATCH_SIZE = 64;
        else if (ne1 > 16 ) BATCH_SIZE = 32;
        else if (ne1 > 8  ) BATCH_SIZE = 16;
        else if (ne1 > 4  ) BATCH_SIZE = 8;
        else                BATCH_SIZE = 2;

        auto pipeline = ggml_metal_library_get_pipeline_ssm_conv_batched(lib, op, BATCH_SIZE);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),         3);

        // Dispatch: ne01 rows, ceil(ne1/BATCH_SIZE) token batches, ne02 sequences
        // Each threadgroup has BATCH_SIZE threads, each handling one token
        const int n_token_batches = (ne1 + BATCH_SIZE - 1) / BATCH_SIZE;
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, n_token_batches, ne02, BATCH_SIZE, 1, 1);
    } else {
        auto pipeline = ggml_metal_library_get_pipeline_ssm_conv(lib, op);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne1, ne02, 1, 1, 1);
    }

    return 1;
}

int ggml_metal_op_ssm_scan(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne4, op->src[4], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb4, op->src[4], nb);
    GGML_TENSOR_LOCALS( int32_t, ne5, op->src[5], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb5, op->src[5], nb);
    GGML_TENSOR_LOCALS( int32_t, ne6, op->src[6], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb6, op->src[6], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const ggml_tensor * src3 = op->src[3];
    const ggml_tensor * src4 = op->src[4];
    const ggml_tensor * src5 = op->src[5];
    const ggml_tensor * src6 = op->src[6];

    GGML_ASSERT(src3);
    GGML_ASSERT(src4);
    GGML_ASSERT(src5);
    GGML_ASSERT(src6);

    const int64_t d_state      = ne00;
    const int64_t d_inner      = ne01;
    const int64_t n_head       = ne02;
    const int64_t n_group      = ne41;
    const int64_t n_seq_tokens = ne12;
    const int64_t n_seqs       = ne13;

    ggml_metal_kargs_ssm_scan args = {
        /*.d_state      =*/ d_state,
        /*.d_inner      =*/ d_inner,
        /*.n_head       =*/ n_head,
        /*.n_group      =*/ n_group,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seqs       =*/ n_seqs,
        /*.s_off        =*/ ggml_nelements(op->src[1]) * sizeof(float),
        /*.nb00         =*/ nb00,
        /*.nb01         =*/ nb01,
        /*.nb02         =*/ nb02,
        /*.nb03         =*/ nb03,
        /*.nb10         =*/ nb10,
        /*.nb11         =*/ nb11,
        /*.nb12         =*/ nb12,
        /*.ns12         =*/ nb12/nb10,
        /*.nb13         =*/ nb13,
        /*.nb20         =*/ nb20,
        /*.nb21         =*/ nb21,
        /*.ns21         =*/ nb21/nb20,
        /*.nb22         =*/ nb22,
        /*.ne30         =*/ ne30,
        /*.nb31         =*/ nb31,
        /*.nb41         =*/ nb41,
        /*.nb42         =*/ nb42,
        /*.ns42         =*/ nb42/nb40,
        /*.nb43         =*/ nb43,
        /*.nb51         =*/ nb51,
        /*.nb52         =*/ nb52,
        /*.ns52         =*/ nb52/nb50,
        /*.nb53         =*/ nb53,
        /*.nb0          =*/ nb0,
    };

    auto pipeline = ggml_metal_library_get_pipeline_ssm_scan(lib, op);

    GGML_ASSERT(d_state <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), 5);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), 6);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), 7);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         8);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, d_inner, n_head, n_seqs, d_state, 1, 1);

    return 1;
}

int ggml_metal_op_rwkv(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int64_t B = op->op == GGML_OP_RWKV_WKV6 ? op->src[5]->ne[1] : op->src[6]->ne[1];
    const int64_t T = op->src[0]->ne[2];
    const int64_t C = op->ne[0];
    const int64_t H = op->src[0]->ne[1];

    auto pipeline = ggml_metal_library_get_pipeline_rwkv(lib, op);

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), ida++);
    if (op->op == GGML_OP_RWKV_WKV7) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), ida++);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &B, sizeof(B), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &T, sizeof(T), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &C, sizeof(C), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &H, sizeof(H), ida++);

    ggml_metal_encoder_dispatch_threadgroups(enc, B * H, 1, 1, C/H, 1, 1);

    return 1;
}

int ggml_metal_op_gated_delta_net(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;


    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_gated_delta_net(lib, op);

    int ida = 0;

    ggml_metal_kargs_gated_delta_net args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne20 =*/ ne20,
        /*.ne21 =*/ ne21,
        /*.ne22 =*/ ne22,
        /*.ne23 =*/ ne23,
        /*.nb20 =*/ nb20,
        /*.nb21 =*/ nb21,
        /*.nb22 =*/ nb22,
        /*.nb23 =*/ nb23,
        /*.ns02 =*/ (int32_t) (nb02/sizeof(float)),
        /*.ns12 =*/ (int32_t) (nb12/sizeof(float)),
        /*.ns22 =*/ (int32_t) (nb22/sizeof(float)),
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args),                  ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++); // q
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++); // k
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++); // v
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++); // gate
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++); // beta
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), ida++); // state
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++); // dst

    const int nsg = pipeline.nsg;

    ggml_metal_encoder_dispatch_threadgroups(enc, op->src[2]->ne[0]/nsg, op->src[2]->ne[1], op->src[2]->ne[3], 32, nsg, 1);

    return 1;
}

int ggml_metal_op_solve_tri(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_solve_tri args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_solve_tri(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nsg = pipeline.nsg;

    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne10 + nsg - 1)/nsg, ne02, ne03, 32, nsg, 1);

    return 1;
}

int ggml_metal_op_set(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    const size_t pnb1 = ((const int32_t *) op->op_params)[0];
    const size_t pnb2 = ((const int32_t *) op->op_params)[1];
    const size_t pnb3 = ((const int32_t *) op->op_params)[2];
    const size_t offs = ((const int32_t *) op->op_params)[3];

    const bool inplace = (bool) ((const int32_t *) op->op_params)[4];

    if (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

        ggml_metal_kargs_cpy args = {
            /*.nk0  =*/ ne00,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[1]->type, op->type);

    GGML_ASSERT(ne10 % ggml_blck_size(op->src[1]->type) == 0);

    int64_t nk0 = ne10;
    if (ggml_is_quantized(op->src[1]->type)) {
        nk0 = ne10/16;
    } else if (ggml_is_quantized(op->type)) {
        nk0 = ne10/ggml_blck_size(op->type);
    }

    int nth = std::min<int>(nk0, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;

    // TODO: relax this constraint in the future
    if (ggml_blck_size(op->src[1]->type) == 1 && ggml_blck_size(op->type) == 1) {
        if (nth > nk0) {
            nrptg = (nth + nk0 - 1)/nk0;
            nth   = nk0;

            if (nrptg*nth > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
                nrptg--;
            }
        }
    }

    nth = std::min<int>(nth, nk0);

    ggml_metal_kargs_cpy args = {
        /*.nk0  =*/ nk0,
        /*.ne00 =*/ ne10,
        /*.ne01 =*/ ne11,
        /*.ne02 =*/ ne12,
        /*.ne03 =*/ ne13,
        /*.nb00 =*/ nb10,
        /*.nb01 =*/ nb11,
        /*.nb02 =*/ nb12,
        /*.nb03 =*/ nb13,
        /*.ne0  =*/ ne10,
        /*.ne1  =*/ ne11,
        /*.ne2  =*/ ne12,
        /*.ne3  =*/ ne13,
        /*.nb0  =*/ ggml_element_size(op),
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
    };

    const int nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    bid_dst.offs += offs;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*(ne11 + nrptg - 1)/nrptg, ne12, ne13, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_cpy(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

    GGML_ASSERT(ne00 % ggml_blck_size(op->src[0]->type) == 0);

    int64_t nk0 = ne00;
    if (ggml_is_quantized(op->src[0]->type)) {
        nk0 = ne00/16;
    } else if (ggml_is_quantized(op->type)) {
        nk0 = ne00/ggml_blck_size(op->type);
    }

    int nth = std::min<int>(nk0, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;

    // TODO: relax this constraint in the future
    if (ggml_blck_size(op->src[0]->type) == 1 && ggml_blck_size(op->type) == 1) {
        if (nth > nk0) {
            nrptg = (nth + nk0 - 1)/nk0;
            nth   = nk0;

            if (nrptg*nth > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
                nrptg--;
            }
        }
    }

    nth = std::min<int>(nth, nk0);

    ggml_metal_kargs_cpy args = {
        /*.nk0  =*/ nk0,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    const int nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*(ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_pool_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    const int32_t k0 = opts[1];
    const int32_t s0 = opts[2];
    const int32_t p0 = opts[3];

    const int64_t IW = op->src[0]->ne[0];
    const int64_t OW = op->ne[0];

    const int64_t np = ggml_nelements(op);

    ggml_metal_kargs_pool_1d args_pool_1d = {
        /* .k0 = */  k0,
        /* .s0 = */  s0,
        /* .p0 = */  p0,
        /* .IW = */  IW,
        /* .OW = */  OW,
        /* .np = */  np
    };

    auto pipeline = ggml_metal_library_get_pipeline_pool_1d(lib, op, op_pool);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) np);
    const int ntg = (np + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args_pool_1d, sizeof(args_pool_1d),  0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}


int ggml_metal_op_pool_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    const int32_t k0 = opts[1];
    const int32_t k1 = opts[2];
    const int32_t s0 = opts[3];
    const int32_t s1 = opts[4];
    const int32_t p0 = opts[5];
    const int32_t p1 = opts[6];

    const int64_t IH = op->src[0]->ne[1];
    const int64_t IW = op->src[0]->ne[0];

    const int64_t N  = op->ne[3];
    const int64_t OC = op->ne[2];
    const int64_t OH = op->ne[1];
    const int64_t OW = op->ne[0];

    const int64_t np = N * OC * OH * OW;

    ggml_metal_kargs_pool_2d args_pool_2d = {
        /* .k0 = */ k0,
        /* .k1 = */ k1,
        /* .s0 = */ s0,
        /* .s1 = */ s1,
        /* .p0 = */ p0,
        /* .p1 = */ p1,
        /* .IH = */ IH,
        /* .IW = */ IW,
        /* .OH = */ OH,
        /* .OW = */ OW,
        /* .np = */ np
    };

    auto pipeline = ggml_metal_library_get_pipeline_pool_2d(lib, op, op_pool);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) np);
    const int ntg = (np + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args_pool_2d, sizeof(args_pool_2d), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_mul_mat(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ne00 == ne10);

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    // Fused Inkling2 relative-bias projection: MUL_MAT(F32 [16, ne01], F32 [16, ne11, ne12]) - optional sinks [1, ne11, ne12]
    if (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
        !ggml_is_transposed(op->src[0]) && !ggml_is_transposed(op->src[1]) &&
        ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]) &&
        ne00 == 16 && ne02 == 1 && ne03 == 1 && ne13 == 1 &&
        ne11 >= 1 && ne11 <= 64 && ne01 >= 64) {
        ggml_tensor * dst_t  = op;
        ggml_tensor * sink_t = nullptr;
        int n_consumed = 1;
        if (ctx->use_fusion) {
            const ggml_op ops_sub[2] = { GGML_OP_MUL_MAT, GGML_OP_SUB };
            if (ctx->can_fuse(idx, ops_sub, 2)) {
                ggml_tensor * next = ctx->node(idx + 1);
                if (next->src[0] == op && next->src[1] &&
                    next->src[1]->type == GGML_TYPE_F32 && next->type == GGML_TYPE_F32 &&
                    next->src[1]->ne[0] == 1 && next->src[1]->ne[1] == ne11 &&
                    next->src[1]->ne[2] == ne12 && next->src[1]->ne[3] == 1 &&
                    ggml_are_same_shape(next, op) &&
                    !ggml_metal_tensor_overlaps_unsafe(op->src[1],  next, false) &&
                    !ggml_metal_tensor_overlaps_unsafe(next->src[1], next, false)) {
                    dst_t      = next;
                    sink_t     = next->src[1];
                    n_consumed = 2;
                }
            }
        }

        if (n_consumed > 1) {
            ggml_metal_op_concurrency_check_fused(ctx, idx, n_consumed);
        }

        auto pl_rb = ggml_metal_library_get_pipeline_inkling_rel_bias(lib);
        ggml_metal_kargs_inkling_rel_bias rb_args = {
            /*.ne01       =*/ ne01,
            /*.ne11       =*/ ne11,
            /*.ne12       =*/ ne12,
            /*.has_sink   =*/ sink_t ? 1 : 0,
            /*.has_clip   =*/ 0,
            /*.has_tau    =*/ 0,
            /*.clip_eps   =*/ 0.0f,
            /*.post_scale =*/ 1.0f,
            /*.nb01       =*/ nb01,
            /*.nb11       =*/ nb11,
            /*.nb12       =*/ nb12,
            /*.nb21       =*/ sink_t ? sink_t->nb[1] : 0,
            /*.nb22       =*/ sink_t ? sink_t->nb[2] : 0,
            /*.nb1        =*/ dst_t->nb[1],
            /*.nb2        =*/ dst_t->nb[2],
            /*.nb_cw1     =*/ 0,
            /*.nb_tau2    =*/ 0,
        };
        ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
        ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
        ggml_metal_encoder_set_pipeline(enc, pl_rb);
        ggml_metal_encoder_set_bytes   (enc, &rb_args, sizeof(rb_args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer  (enc, sink_t ? ggml_metal_get_buffer_id(sink_t) : bid_src1, 3);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst_t), 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 5);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 6);
        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + 31) / 32, ne12, 1, 32, 4, 1);
        return n_consumed;
    }

    // Fused DSpark Markov vocab projection + biases + argmax:
    //   MUL_MAT(BF16 [256, ne01], F32 [256, 1]) -> ADD(F32 [ne01]) -> ADD(F32 [ne01, 1]) -> ARGMAX
    if (ctx->use_fusion && props_dev->has_bfloat &&
        op->src[0]->type == GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_F32 &&
        ne00 == 256 && ne01 >= 4096 && ne02 == 1 && ne03 == 1 &&
        ne11 == 1 && ne12 == 1 && ne13 == 1 &&
        nb00 == sizeof(uint16_t) && nb10 == sizeof(float)) {
        const ggml_op fops[3] = { GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_ADD };
        if (ctx->can_fuse(idx, fops, 3) && idx + 3 < ctx->n_nodes() && ctx->node_has_n_uses(idx + 2, 1)) {
            ggml_tensor * n_add0 = ctx->node(idx + 1);
            ggml_tensor * n_add1 = ctx->node(idx + 2);
            ggml_tensor * n_amax = ctx->node(idx + 3);
            if (n_amax->op == GGML_OP_ARGMAX && n_amax->src[0] == n_add1 && ggml_nrows(n_add1) == 1) {
                ggml_tensor * add_a = n_add0->src[0] == op     ? n_add0->src[1] : n_add0->src[0];
                ggml_tensor * add_b = n_add1->src[0] == n_add0 ? n_add1->src[1] : n_add1->src[0];
                if (add_a->type == GGML_TYPE_F32 && add_b->type == GGML_TYPE_F32 &&
                    ggml_nelements(add_a) == ne01 && ggml_nelements(add_b) == ne01 &&
                    add_a->nb[0] == sizeof(float) && add_b->nb[0] == sizeof(float) &&
                    ggml_nbytes(op) >= 256 * (4 * sizeof(float) + 3 * sizeof(int32_t))) {
                    auto pl_part = ggml_metal_library_get_pipeline_dspark_vocab_argmax(lib, false);
                    auto pl_red  = ggml_metal_library_get_pipeline_dspark_vocab_argmax(lib, true);

                    ggml_metal_buffer_id bid_src0  = ggml_metal_get_buffer_id(op->src[0]);
                    ggml_metal_buffer_id bid_src1  = ggml_metal_get_buffer_id(op->src[1]);
                    ggml_metal_buffer_id bid_add_a = ggml_metal_get_buffer_id(add_a);
                    ggml_metal_buffer_id bid_add_b = ggml_metal_get_buffer_id(add_b);
                    ggml_metal_buffer_id bid_tmp   = ggml_metal_get_buffer_id(op);
                    ggml_metal_buffer_id bid_dst   = ggml_metal_get_buffer_id(n_amax);
                    ggml_metal_buffer_id bid_base  = { nullptr, 0 };
                    ggml_metal_buffer_id bid_ctrl  = { nullptr, 0 };
                    int n_consumed = 4;

                    bool    has_cpy_buf = false;
                    int32_t has_ctrl    = 0;
                    int32_t cpy_slots   = 0;
                    int32_t dst_slot    = 0;
                    int32_t gmax_idx    = 0;
                    int32_t gate_idx    = -1;
                    int32_t top2_slot   = -1;
                    int32_t top3_slot   = -1;
                    int32_t alt2_ctrl   = -1;
                    int32_t alt3_ctrl   = -1;

                    for (int j = idx + 4; j < ctx->n_nodes(); ++j) {
                        ggml_tensor * n_cpy = ctx->node(j);
                        if (n_cpy->op != GGML_OP_CPY || n_cpy->src[0] != n_amax) {
                            continue;
                        }
                        ggml_tensor * base_t = n_cpy;
                        size_t total_offs = 0;
                        while (base_t->view_src) {
                            total_offs += base_t->view_offs;
                            base_t = base_t->view_src;
                        }
                        const int64_t base_ne = ggml_nelements(base_t);
                        if (n_cpy->type == GGML_TYPE_I32 && ggml_nelements(n_cpy) == 1 && base_ne == 152) {
                            bid_base    = ggml_metal_get_buffer_id(base_t);
                            bid_ctrl    = bid_base;
                            bid_ctrl.offs += (size_t) 88 * sizeof(int32_t);
                            has_cpy_buf = true;
                            cpy_slots   = 44;
                            dst_slot    = (int32_t) (total_offs / sizeof(int32_t));
                            has_ctrl    = 2;

                            switch (dst_slot) {
                                case  0: gmax_idx =  0; gate_idx = -1; top2_slot = 15; top3_slot = 21; alt2_ctrl =  3; alt3_ctrl =  4; break;
                                case  1: gmax_idx =  1; gate_idx =  2; top2_slot = 25; top3_slot = 42; alt2_ctrl =  5; alt3_ctrl = 10; break;
                                case  2: gmax_idx =  2; gate_idx =  2; top2_slot = 30;                 alt2_ctrl =  6;                 break;
                                case  3: gmax_idx =  3; gate_idx =  2; top2_slot = 34;                 alt2_ctrl =  7;                 break;
                                case  4: gmax_idx =  4; gate_idx =  2; top2_slot = 37;                 alt2_ctrl =  8;                 break;
                                case  5: gmax_idx =  5; gate_idx =  2; break;
                                case  6: gmax_idx =  6; gate_idx =  2; break;
                                case  7: gmax_idx =  7; gate_idx =  2; break;
                                case  8: gmax_idx =  8; gate_idx =  2; break;
                                case  9: gmax_idx =  9; gate_idx =  2; break;
                                case 10: gmax_idx = 10; gate_idx =  2; break;
                                case 11: gmax_idx = 11; gate_idx =  2; break;
                                case 12: gmax_idx = 12; gate_idx =  2; break;
                                case 13: gmax_idx = 13; gate_idx =  2; break;
                                case 14: gmax_idx = 14; gate_idx =  2; break;
                                case 16: gmax_idx = 15; gate_idx =  3; top2_slot = 39;                 alt2_ctrl =  9;                 break;
                                case 17: gmax_idx = 16; gate_idx =  3; break;
                                case 18: gmax_idx = 17; gate_idx =  3; break;
                                case 19: gmax_idx = 18; gate_idx =  3; break;
                                case 20: gmax_idx = 19; gate_idx =  3; break;
                                case 22: gmax_idx = 20; gate_idx =  4; break;
                                case 23: gmax_idx = 21; gate_idx =  4; break;
                                case 24: gmax_idx = 22; gate_idx =  4; break;
                                case 26: gmax_idx = 23; gate_idx =  5; break;
                                case 27: gmax_idx = 24; gate_idx =  5; break;
                                case 28: gmax_idx = 25; gate_idx =  5; break;
                                case 29: gmax_idx = 26; gate_idx =  5; break;
                                case 31: gmax_idx = 27; gate_idx =  6; break;
                                case 32: gmax_idx = 28; gate_idx =  6; break;
                                case 33: gmax_idx = 29; gate_idx =  6; break;
                                case 35: gmax_idx = 30; gate_idx =  7; break;
                                case 36: gmax_idx = 31; gate_idx =  7; break;
                                case 38: gmax_idx = 32; gate_idx =  8; break;
                                case 40: gmax_idx = 33; gate_idx =  9; break;
                                case 41: gmax_idx = 34; gate_idx =  9; break;
                                case 43: gmax_idx = 35; gate_idx = 10; break;
                                default: gmax_idx = 36; gate_idx =  2; break;
                            }
                            if (j == idx + 4) {
                                n_consumed = 5;
                            }
                        }
                        break;
                    }

                    // Optional Cauchy-Schwarz upper-bound norm table packed in row 1 of fold_b [n_slots, 2]
                    int32_t norm_stride = 0;
                    if (add_a->view_src && add_a->view_src->ne[1] == 2 && add_a->view_offs == 0 &&
                        add_a->view_src->type == GGML_TYPE_F32 && add_a->view_src->ne[0] >= ne01 &&
                        add_a->view_src->nb[1] == (size_t) add_a->view_src->ne[0] * sizeof(float)) {
                        norm_stride = (int32_t) add_a->view_src->ne[0];
                    }

                    ggml_metal_op_concurrency_check_fused(ctx, idx, n_consumed);

                    ggml_metal_kargs_dspark_vocab_argmax args = {
                        /*.ne01        =*/ ne01,
                        /*.gmax_idx    =*/ gmax_idx,
                        /*.gate_idx    =*/ gate_idx,
                        /*.has_ctrl    =*/ has_ctrl,
                        /*.norm_stride =*/ norm_stride,
                        /*.nb01        =*/ nb01,
                    };

                    ggml_metal_encoder_set_pipeline(enc, pl_part);
                    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
                    ggml_metal_encoder_set_buffer  (enc, bid_src0,  1);
                    ggml_metal_encoder_set_buffer  (enc, bid_src1,  2);
                    ggml_metal_encoder_set_buffer  (enc, bid_add_a, 3);
                    ggml_metal_encoder_set_buffer  (enc, bid_add_b, 4);
                    ggml_metal_encoder_set_buffer  (enc, bid_tmp,   5);
                    ggml_metal_encoder_set_buffer  (enc, has_ctrl != 0 ? bid_ctrl : bid_tmp, 6);
                    ggml_metal_encoder_set_threadgroup_memory_size(enc, pl_part.smem, 0);
                    ggml_metal_encoder_dispatch_threadgroups(enc, 256, 1, 1, 32, 4, 1);

                    ggml_metal_op_concurrency_reset(ctx);

                    ggml_metal_kargs_dspark_vocab_reduce rargs = {
                        /*.slots     =*/ has_cpy_buf ? cpy_slots : 0,
                        /*.dst_slot  =*/ dst_slot,
                        /*.top2_slot =*/ top2_slot,
                        /*.top3_slot =*/ top3_slot,
                        /*.gate_idx  =*/ gate_idx,
                        /*.alt2_ctrl =*/ alt2_ctrl,
                        /*.alt3_ctrl =*/ alt3_ctrl,
                        /*.has_ctrl  =*/ has_ctrl != 0 ? 1 : 0,
                    };
                    ggml_metal_encoder_set_pipeline(enc, pl_red);
                    ggml_metal_encoder_set_bytes   (enc, &rargs, sizeof(rargs), 0);
                    ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
                    ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);
                    ggml_metal_encoder_set_buffer  (enc, has_cpy_buf ? bid_base : bid_dst, 3);
                    ggml_metal_encoder_set_buffer  (enc, has_ctrl != 0 ? bid_ctrl : bid_dst, 4);
                    ggml_metal_encoder_set_threadgroup_memory_size(enc, pl_red.smem, 0);
                    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 256, 1, 1);

                    // Ensure side-writes to top2/top3 slots & ctrl are globally visible to downstream Markov steps
                    ggml_metal_op_concurrency_reset(ctx);

                    return n_consumed;
                }
            }
        }
    }

    // F4A: Fused Inkling2 loop_ctrl_up BF16 [64, 2*n_embd, c] -> INKLING_CTRL_BLEND -> RMS_NORM -> MUL(norm_w)
    if (ctx->use_fusion && props_dev->has_bfloat &&
        idx + 3 < ctx->n_nodes() &&
        op->src[0]->type == GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_F32 &&
        !ggml_is_transposed(op->src[0]) && !ggml_is_transposed(op->src[1]) &&
        ne00 == 64 && ne03 == 1 && ne13 == 1 && ne02 == ne12 &&
        ctx->node_has_n_uses(idx, 1) && ctx->node_has_n_uses(idx + 1, 1) && ctx->node_has_n_uses(idx + 2, 1)) {
        ggml_tensor * blend = ctx->node(idx + 1);
        ggml_tensor * rn    = ctx->node(idx + 2);
        ggml_tensor * mul   = ctx->node(idx + 3);
        if (blend->op == GGML_OP_INKLING_CTRL_BLEND && blend->src[2] == op &&
            rn->op == GGML_OP_RMS_NORM && rn->src[0] == blend &&
            mul->op == GGML_OP_MUL && (mul->src[0] == rn || mul->src[1] == rn)) {
            const ggml_tensor * cur    = blend->src[0];
            const ggml_tensor * msg    = blend->src[1];
            const ggml_tensor * norm_w = (mul->src[0] == rn) ? mul->src[1] : mul->src[0];
            const int32_t n_embd   = (int32_t) msg->ne[0];
            const int32_t n_tokens = (int32_t) msg->ne[1];
            const int32_t n_ctrl   = (int32_t) msg->ne[2];
            if (ne01 == 2 * n_embd && ne11 == n_tokens && ne12 == n_ctrl &&
                n_embd % 256 == 0 && n_embd <= 4096 && n_tokens * n_ctrl >= 4 &&
                cur->type == GGML_TYPE_F32 && msg->type == GGML_TYPE_F32 &&
                norm_w && norm_w->type == GGML_TYPE_F32 &&
                norm_w->ne[0] == n_embd && norm_w->ne[1] == 1 &&
                (norm_w->ne[2] == 1 || norm_w->ne[2] == n_ctrl) && norm_w->ne[3] == 1 &&
                ggml_is_contiguous(norm_w) && ggml_is_contiguous(mul) &&
                !ggml_metal_tensor_overlaps_unsafe(op->src[0], mul, false) &&
                !ggml_metal_tensor_overlaps_unsafe(op->src[1], mul, false) &&
                !ggml_metal_tensor_overlaps_unsafe(cur,        mul, cur->ne[2] == n_ctrl) &&
                !ggml_metal_tensor_overlaps_unsafe(msg,        mul, true) &&
                !ggml_metal_tensor_overlaps_unsafe(norm_w,     mul, false)) {
                ggml_metal_op_concurrency_check_fused(ctx, idx, 4);
                float eps = 0.0f;
                memcpy(&eps, rn->op_params, sizeof(float));
                ggml_metal_kargs_inkling_ctrl_up_blend cargs = {
                    /*.n_embd    =*/ n_embd,
                    /*.n_tokens  =*/ n_tokens,
                    /*.n_ctrl    =*/ n_ctrl,
                    /*.bcast_cur =*/ cur->ne[2] == 1 ? 1 : 0,
                    /*.eps       =*/ eps,
                    /*.nbw1      =*/ nb01,
                    /*.nbw2      =*/ nb02,
                    /*.nbg1      =*/ nb11,
                    /*.nbg2      =*/ nb12,
                    /*.nbc1      =*/ cur->nb[1],
                    /*.nbc2      =*/ cur->nb[2],
                    /*.nbm1      =*/ msg->nb[1],
                    /*.nbm2      =*/ msg->nb[2],
                    /*.nbnw2     =*/ (norm_w->ne[2] == n_ctrl && n_ctrl > 1) ? norm_w->nb[2] : 0,
                };
                auto pl = ggml_metal_library_get_pipeline_inkling_ctrl_up_blend(lib);
                ggml_metal_encoder_set_pipeline(enc, pl);
                ggml_metal_encoder_set_bytes   (enc, &cargs, sizeof(cargs), 0);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(cur),        3);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(msg),        4);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(norm_w),     5);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mul),        6);
                ggml_metal_encoder_dispatch_threadgroups(enc, 1, n_tokens, n_ctrl, 256, 1, 1);
                return 4;
            }
        }
    }

    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    const int ne11_mm_min = 8;

    // High-throughput BF16 simdgroup_mm kernels for B = 3..16 (ne00 % 64 == 0):
    // - Small output dims (ne01 <= 384: MoE router, loop ctrl down) stay on exact-F32 mul_mv_rows (NR0=1, NSG=1) up to B=16.
    // - Target projections (ne01 >= 1024, B = 3..16) and rank-64 loop ctrl up (ne00 < 128, B = 5..16) use exact-F32 simdgroup_float8x8 (BK=64).
    // - DSpark-only shapes (ne01 == 5120 || ne01 == 18432 || ne00 == 9216 || 30000 < ne01 < 100000) at B = 4..16 use simdgroup_half8x8 -> float AMX (drafter only).
    if (props_dev->has_bfloat && props_dev->has_simdgroup_mm &&
        op->src[0]->type == GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_F32 &&
        !ggml_is_transposed(op->src[0]) && !ggml_is_transposed(op->src[1]) &&
        (ne00 % 64 == 0) &&
        ((ne11 >= 4 && ne11 <= 16 && ne00 >= 512 && (ne01 == 5120 || ne01 == 18432 || ne00 == 9216 || (ne01 > 10000 && ne01 < 100000))) ||
         (ne11 >= 3 && ne11 <= 16 && ne01 > 384 && ne00 >= 512) ||
         (ne11 >= 3 && ne11 <= 16 && ne01 > 384 && ne00 < 128) ||
         (ne11 >= 9 && ne11 <= 16 && ne01 > 384))) {
        const bool is_dspark_shape = (ne01 == 5120 || ne01 == 18432 || ne00 == 9216 || (ne01 > 10000 && ne01 < 100000));
        const bool use_h16         = (is_dspark_shape && ne11 >= 4 && ne11 <= 16 && ne00 >= 512);
        ggml_metal_pipeline_with_params pipeline;
        bool use_bk32 = false;
        if (use_h16) {
            const int nr0 = (ne01 < 10000) ? 32 : 64;
            pipeline = ggml_metal_library_get_pipeline_mul_mm_bf16_h16act(lib, nr0, ne11);
        } else if (ne11 > 8 && ne01 >= 4000) {
            const int nr0 = (ne01 >= 8000) ? 64 : 32;
            use_bk32 = true;
            pipeline = ggml_metal_library_get_pipeline_mul_mm_bf16_f32act(lib, nr0, ne11, true);
        } else {
            const int nr0 = (ne01 < 1024) ? 16 : 32;
            pipeline = ggml_metal_library_get_pipeline_mul_mm_bf16_f32act(lib, nr0, ne11, false);
        }

        ggml_metal_kargs_mul_mv_ext args = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne10  =*/ ne10,
            /*.ne11  =*/ ne11,
            /*.ne12  =*/ ne12,
            /*.nb10  =*/ nb10,
            /*.nb11  =*/ nb11,
            /*.nb12  =*/ nb12,
            /*.nb13  =*/ nb13,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.r2    =*/ r2,
            /*.r3    =*/ r3,
        };

        const ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);
        ggml_metal_buffer_id bid_out = bid_dst;
        int sk = 1;
        int sk_slot = -1;
        const bool want_sk = !use_bk32 &&
            ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 && ggml_is_contiguous(op) &&
            ne01 > 384 && ne01 <= 2560 &&
            (use_h16 ? (ne00 >= 3072) : (ne00 >= 1536));
        if (want_sk) {
            const int try_sk = 3;
            const size_t sk_bytes = (size_t) try_sk * (size_t) ne0 * (size_t) ne1 * sizeof(float);
            if (ggml_metal_op_alloc_sk_buf(ctx, sk_bytes, &bid_out, &sk_slot)) {
                sk = try_sk;
            }
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, bid_out,                              3);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + pipeline.nr0 - 1) / pipeline.nr0, sk, ne12 * ne13, 32, pipeline.nsg, 1);
        if (sk > 1) {
            ggml_metal_op_dispatch_sk_reduce(ctx, bid_out, bid_dst, (int32_t) (ne0 * ne1), sk, sk_slot);
        }

        return 1;
    }

    // Batched BF16 mat-vec paths for small batches:
    //  1) ne00 % 64 == 0 && ne00 % 128 != 0 (e.g. rank-64 loop controller up-proj): k64 kernel for ne11 = 2..8
    //  2) ne00 % 128 == 0 && ne01 <= 384 (small output dim): NR0=1, NSG=1 for full GPU core occupancy at ne11 = 2..16
    //  3) ne00 % 128 == 0 && 384 < ne01 <= 1536 && ne11 = 5..8: NR0=2, NSG=2 exact-F32 zero-barrier streaming
    //  4) ne00 % 128 == 0 && ne01 > 1536 && ne11 = 5..8: fallback if simdgroup_mm unavailable
    if (props_dev->has_bfloat && op->src[0]->type == GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_F32 &&
        !ggml_is_transposed(op->src[0]) && !ggml_is_transposed(op->src[1]) &&
        ne11 >= 2 && ne11 <= 16) {
        const bool use_k64   = (ne00 % 64 == 0 && ne00 % 128 != 0 && ne11 <= N_R1_MV_ROWS);
        const bool use_small = (ne00 % 128 == 0 && ne01 <= 384);
        const bool use_med   = (ne00 % 128 == 0 && ne01 > 384 && ne01 <= 1536 && ne11 >= 5 && ne11 <= N_R1_MV_ROWS);
        const bool use_large = (ne00 % 128 == 0 && ne01 > 1536 && ne11 >= 5 && ne11 <= N_R1_MV_ROWS);
        if (use_k64 || use_small || use_med || use_large) {
            const int nr0 = use_small ? 1 : (use_med ? 2 : 4);
            const int nsg = use_small ? 1 : (use_med ? 2 : N_SG_MV_ROWS);
            auto pipeline = ggml_metal_library_get_pipeline_mul_mv_rows(lib, nr0, ne11, nsg, use_k64);

            ggml_metal_kargs_mul_mv_ext args = {
                /*.ne00  =*/ ne00,
                /*.ne01  =*/ ne01,
                /*.ne02  =*/ ne02,
                /*.nb00  =*/ nb00,
                /*.nb01  =*/ nb01,
                /*.nb02  =*/ nb02,
                /*.nb03  =*/ nb03,
                /*.ne10  =*/ ne10,
                /*.ne11  =*/ ne11,
                /*.ne12  =*/ ne12,
                /*.nb10  =*/ nb10,
                /*.nb11  =*/ nb11,
                /*.nb12  =*/ nb12,
                /*.nb13  =*/ nb13,
                /*.ne0   =*/ ne0,
                /*.ne1   =*/ ne1,
                /*.r2    =*/ r2,
                /*.r3    =*/ r3,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0*nsg - 1)/(nr0*nsg), 1, ne12*ne13, 32, nsg, 1);

            return 1;
        }
    }

    // first try to use small-batch mat-mv kernels
    // these should be efficient for BS [2, ~8] when ne01 is small/medium; for small ne01 <= 384 (e.g. 4608x64, 2304x128)
    // keep using mul_mv_ext up to BS=16 because mul_mm (NR0=64) launches only 1-2 threadgroups across all GPU cores
    const bool use_mv_ext_small_rows = props_dev->has_bfloat &&
        op->src[0]->type == GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_F32 &&
        (ne00 % 128 == 0) && ne01 <= 384 && ne11 > ne11_mm_min && ne11 <= 16;

    const bool use_mm_k64 = props_dev->has_bfloat &&
        op->src[0]->type == GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_F32 &&
        ne00 >= 64 && ne00 < 128 && ne11 >= 5 && ne11 <= 8;

    if (op->src[1]->type == GGML_TYPE_F32 && (ne00%128 == 0) && (ne11 <= ne11_mm_min || use_mv_ext_small_rows) &&
        (
         use_mv_ext_small_rows ||
         (
          (
           op->src[0]->type == GGML_TYPE_F32  || // TODO: helper function
           op->src[0]->type == GGML_TYPE_F16  ||
           op->src[0]->type == GGML_TYPE_BF16 ||
           op->src[0]->type == GGML_TYPE_Q1_0 ||
           op->src[0]->type == GGML_TYPE_Q4_0 ||
           op->src[0]->type == GGML_TYPE_Q4_1 ||
           op->src[0]->type == GGML_TYPE_Q5_0 ||
           op->src[0]->type == GGML_TYPE_Q5_1 ||
           op->src[0]->type == GGML_TYPE_Q8_0 ||
           op->src[0]->type == GGML_TYPE_MXFP4 ||
           op->src[0]->type == GGML_TYPE_NVFP4 ||
           op->src[0]->type == GGML_TYPE_MXFP8 ||
           op->src[0]->type == GGML_TYPE_IQ4_NL ||
           false) && (ne11 >= 2 && ne11 <= 8)
         ) ||
         (
          (
           op->src[0]->type == GGML_TYPE_Q4_K ||
           op->src[0]->type == GGML_TYPE_Q5_K ||
           op->src[0]->type == GGML_TYPE_Q6_K ||
           op->src[0]->type == GGML_TYPE_Q2_K ||
           op->src[0]->type == GGML_TYPE_Q3_K ||
           false) && (ne11 >= 4 && ne11 <= 8)
         )
        )
       ) {
        const int nsg    = 2;                 // num simdgroups per threadgroup

        // num threads along row per simdgroup
        int16_t nxpsg = 0;
        if (ne00 % 256 == 0 && ne11 < 3) {
            nxpsg = 16;
        } else if (ne00 % 128 == 0) {
            nxpsg = 8;
        } else {
            nxpsg = 4;
        }

        const int16_t nypsg  = 32/nxpsg;          // num threads along col per simdgroup (i.e. a simdgroup processes that many src0 rows at a time)
        const int16_t r0ptg  = nypsg*nsg;         // num src0 rows per threadgroup
              int16_t r1ptg  = 4;                 // num src1 rows per threadgroup

        switch (ne11) {
            case 2:
                r1ptg = 2; break;
            case 3:
            case 6:
                r1ptg = 3; break;
            case 4:
            case 7:
            case 8:
                r1ptg = 4; break;
            case 5:
                r1ptg = 5; break;
            default:
                if (use_mv_ext_small_rows) {
                    r1ptg = 4;
                } else {
                    GGML_ABORT("unsupported ne11");
                }
                break;
        };

        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_ext(lib, op->src[0]->type, op->src[1]->type, nsg, nxpsg, r1ptg);

        ggml_metal_kargs_mul_mv_ext args = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne10  =*/ ne10,
            /*.ne11  =*/ ne11,
            /*.ne12  =*/ ne12,
            /*.nb10  =*/ nb10,
            /*.nb11  =*/ nb11,
            /*.nb12  =*/ nb12,
            /*.nb13  =*/ nb13,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.r2    =*/ r2,
            /*.r3    =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + r0ptg - 1)/r0ptg), ((ne11 + r1ptg - 1)/r1ptg), ne12*ne13, 32, nsg, 1);
    } else if (
        !ggml_is_transposed(op->src[0]) &&
        !ggml_is_transposed(op->src[1]) &&
        // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
        // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
        props_dev->has_simdgroup_mm && ne00 >= 64 && (ne11 > ne11_mm_min || use_mm_k64)) {
        //GGML_LOG_INFO("matrix: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        auto pipeline = ggml_metal_library_get_pipeline_mul_mm(lib, op);

        ggml_metal_kargs_mul_mm args = {
            /*.ne00 =*/ ne00,
            /*.ne02 =*/ ne02,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        const size_t smem = pipeline.smem;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne11 + nr1 - 1) / nr1), ((ne01 + nr0 - 1) / nr0), ne12 * ne13, 32, nsg, 1);
    } else {
        auto pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, op);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        const size_t smem = pipeline.smem;

        ggml_metal_kargs_mul_mv args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nr0  =*/ nr0,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        if (op->src[0]->type == GGML_TYPE_F32 ||
            op->src[0]->type == GGML_TYPE_F16 ||
            op->src[0]->type == GGML_TYPE_BF16 ||
            op->src[0]->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0 - 1)/(nr0)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0*nsg - 1)/(nr0*nsg)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
        }
    }

    return 1;
}

size_t ggml_metal_op_mul_mat_id_extra_tpe(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    const int64_t ne02 = op->src[0]->ne[2]; // n_expert

    return ggml_type_size(GGML_TYPE_I32)*ne02;
}

size_t ggml_metal_op_mul_mat_id_extra_ids(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    const int64_t ne02 = op->src[0]->ne[2]; // n_expert
    const int64_t ne21 = op->src[2]->ne[1]; // n_token

    return ggml_type_size(GGML_TYPE_I32)*ne02*ne21;
}

int ggml_metal_op_mul_mat_id(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // src2 = ids
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);

    GGML_ASSERT(!ggml_is_transposed(op->src[0]));
    GGML_ASSERT(!ggml_is_transposed(op->src[1]));

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    const uint32_t r2 = 1;
    const uint32_t r3 = 1;

    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    // ne20 = n_used_experts
    // ne21 = n_rows (batch size)
    const bool has_map0 =
        (ne02 <= 1024) &&
        ((size_t) ne02 * ne20 * sizeof(uint16_t) <= props_dev->max_theadgroup_memory_size) &&
        (ne20 == 1 || ne20 == 2 || ne20 == 4 || ne20 == 5 || ne20 == 6 || ne20 == 8 || ne20 == 10 || ne20 == 16 || ne20 == 22);

    const bool has_mm_id_nr1_8 =
        has_map0 &&
        op->src[1]->type == GGML_TYPE_F32 &&
        (op->src[0]->type == GGML_TYPE_NVFP4 ||
         op->src[0]->type == GGML_TYPE_MXFP8 ||
         op->src[0]->type == GGML_TYPE_MXFP4 ||
         op->src[0]->type == GGML_TYPE_BF16  ||
         op->src[0]->type == GGML_TYPE_F16   ||
         op->src[0]->type == GGML_TYPE_Q8_0  ||
         op->src[0]->type == GGML_TYPE_Q4_0);

    const bool is_nvfp4_or_mxfp8 = (op->src[0]->type == GGML_TYPE_NVFP4 || op->src[0]->type == GGML_TYPE_MXFP8);
    const int ne21_mm_id_min = has_mm_id_nr1_8 ? (is_nvfp4_or_mxfp8 ? 17 : 16) : 32;

    if (props_dev->has_simdgroup_mm && ne00 >= 64 && (ne21 >= ne21_mm_id_min)) {
        // extra buffers for intermediate id mapping
        ggml_metal_buffer_id bid_tpe = bid_dst;
        bid_tpe.offs += ggml_nbytes(op);

        ggml_metal_buffer_id bid_ids = bid_tpe;
        bid_ids.offs += ggml_metal_op_mul_mat_id_extra_tpe(op);

        {
            ggml_metal_kargs_mul_mm_id_map0 args = {
                ne02,
                ne10,
                ne11, // n_expert_used (bcast)
                nb11,
                nb12,
                ne21, // n_tokens
                ne20, // n_expert_used
                nb21,
            };

            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id_map0(lib, ne02, ne20);

            const size_t smem = pipeline.smem;

            GGML_ASSERT(ne02 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

            GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src2, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  2);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  3);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, ne02, 1, 1);
        }

        // this barrier is always needed because the next kernel has to wait for the id maps to be computed
        ggml_metal_op_concurrency_reset(ctx);

        {
            const int nr1 = (ne21 < 32 && has_mm_id_nr1_8) ? 8 : 32;
            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id(lib, op, nr1);

            ggml_metal_kargs_mul_mm_id args = {
                /*.ne00  =*/ ne00,
                /*.ne02  =*/ ne02,
                /*.nb01  =*/ nb01,
                /*.nb02  =*/ nb02,
                /*.nb03  =*/ nb03,
                /*.ne11  =*/ ne11, // n_expert_used (bcast)
                /*.nb10  =*/ nb10,
                /*.nb11  =*/ nb11,
                /*.nb12  =*/ nb12,
                /*.nb13  =*/ nb13,
                /*.ne20  =*/ ne20, // n_expert_used
                /*.ne21  =*/ ne21, // n_tokens
                /*.ne0   =*/ ne0,
                /*.ne1   =*/ ne1,
                /*.r2    =*/ r2,
                /*.r3    =*/ r3,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  3);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  4);
            ggml_metal_encoder_set_buffer  (enc, bid_dst,  5);

            const size_t smem = pipeline.smem;

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, (ne21 + nr1 - 1)/nr1, (ne01 + 63)/64, ne02, 128, 1, 1);
        }
    } else {
        // Optional fusion: MoE gate_up (NVFP4/MXFP8 MUL_MAT_ID) + GLU(SITU_EX) into a single kernel dispatch.
        ggml_tensor * situ = nullptr;
        if (ctx->use_fusion && idx + 1 < ctx->n_nodes() && ctx->node_has_n_uses(idx, 1) &&
            op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
            ((op->src[0]->type == GGML_TYPE_NVFP4 && (ne00 % 256 == 0)) ||
             (op->src[0]->type == GGML_TYPE_MXFP8 && (ne00 % 128 == 0) && (ne01 % 8 == 0)))) {
            ggml_tensor * cand = ctx->node(idx + 1);
            if (cand->op == GGML_OP_GLU && ggml_get_glu_op(cand) == GGML_GLU_OP_SITU &&
                cand->src[0] == op && cand->src[1] == nullptr &&
                ggml_get_op_params_i32(cand, 1) == 0 &&
                cand->type == GGML_TYPE_F32 && ggml_is_contiguous(cand) &&
                op->ne[0] == 2 * cand->ne[0] && op->ne[1] == cand->ne[1] && op->ne[2] == cand->ne[2] &&
                (cand->src[2] == nullptr ||
                 (cand->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(cand->src[2]) && cand->src[3] == op->src[2])) &&
                !ggml_metal_tensor_overlaps_unsafe(op->src[0], cand, false) &&
                !ggml_metal_tensor_overlaps_unsafe(op->src[1], cand, false) &&
                !ggml_metal_tensor_overlaps_unsafe(op->src[2], cand, false) &&
                !ggml_metal_tensor_overlaps_unsafe(cand->src[2], cand, false)) {
                situ = cand;
            }
        }

        const bool can_sort =
            ne21 <= 16 && ne20 <= 16 && ne02 <= 256 &&
            op->src[1]->type == GGML_TYPE_F32 &&
            ((op->src[0]->type == GGML_TYPE_NVFP4 && ne21 >= 3) ||
             (op->src[0]->type == GGML_TYPE_MXFP8 && ne21 >= 2 && (ne00 % 128 == 0) && (ne01 % 8 == 0))) &&
            (size_t) (ne20 * ne21) * sizeof(uint32_t) <= ggml_metal_op_mul_mat_id_extra_tpe(op) + ggml_metal_op_mul_mat_id_extra_ids(op);

        if (situ != nullptr) {
            ggml_metal_op_concurrency_check_fused(ctx, idx, 2);

            ggml_metal_buffer_id bid_ids_in = bid_src2;
            if (can_sort) {
                const bool use_aux_srt = (ctx->aux_buf != nullptr && ctx->aux_buf_size >= 65536 && (size_t) (ne20 * ne21) * sizeof(uint32_t) <= 8192);
                ggml_metal_buffer_id bid_srt = use_aux_srt
                    ? ggml_metal_buffer_id{ ctx->aux_buf, 57344 }
                    : ggml_metal_buffer_id{ bid_dst.metal, bid_dst.offs + ggml_nbytes(op) };
                const bool srt_hit = use_aux_srt &&
                    ctx->aux_sort_ids == op->src[2] && ctx->aux_sort_ne20 == ne20 &&
                    ctx->aux_sort_ne21 == ne21 && ctx->aux_sort_ne02 == ne02 && ctx->aux_sort_mode == 1;

                if (!srt_hit) {
                    if (use_aux_srt && ctx->aux_sort_ids != nullptr) {
                        ggml_metal_op_concurrency_reset(ctx);
                    }
                    ggml_metal_kargs_mul_mv_id_group_compact cargs = {
                        /*.nei0 =*/ ne20,
                        /*.nei1 =*/ ne21,
                        /*.ne02 =*/ ne02,
                        /*.nbi1 =*/ nb21,
                    };

                    auto pl_srt = ggml_metal_library_get_pipeline_mul_mv_id_sort(lib);
                    const int nth = ((ne02 + 31) / 32) * 32;

                    ggml_metal_encoder_set_pipeline(enc, pl_srt);
                    ggml_metal_encoder_set_bytes   (enc, &cargs, sizeof(cargs), 0);
                    ggml_metal_encoder_set_buffer  (enc, bid_src2, 1);
                    ggml_metal_encoder_set_buffer  (enc, bid_srt,  2);

                    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

                    ggml_metal_op_concurrency_reset(ctx);

                    if (use_aux_srt) {
                        ctx->aux_sort_ids  = op->src[2];
                        ctx->aux_sort_ne20 = ne20;
                        ctx->aux_sort_ne21 = ne21;
                        ctx->aux_sort_ne02 = ne02;
                        ctx->aux_sort_mode = 1;
                    }
                }
                bid_ids_in = bid_srt;
            }

            auto pipeline = (op->src[0]->type == GGML_TYPE_MXFP8)
                ? ggml_metal_library_get_pipeline_mul_mv_id_mxfp8_situ(lib)
                : ggml_metal_library_get_pipeline_mul_mv_id_nvfp4_situ(lib);
            const int nr0 = pipeline.nr0;
            const int nsg = pipeline.nsg;

            const float beta = ggml_get_op_params_f32(situ, 2);
            const float lb   = ggml_get_op_params_f32(situ, 3);
            const bool has_scales = (situ->src[2] != nullptr);

            ggml_metal_buffer_id bid_situ_dst = ggml_metal_get_buffer_id(situ);
            ggml_metal_buffer_id bid_scales   = has_scales ? ggml_metal_get_buffer_id(situ->src[2]) : bid_situ_dst;

            ggml_metal_kargs_mul_mv_id_situ args = {
                /*.nei0       =*/ ne20,
                /*.nei1       =*/ ne21,
                /*.nbi1       =*/ nb21,
                /*.ne00       =*/ ne00,
                /*.ne01       =*/ ne01,
                /*.ne02       =*/ ne02,
                /*.nb00       =*/ nb00,
                /*.nb01       =*/ nb01,
                /*.nb02       =*/ nb02,
                /*.ne10       =*/ ne10,
                /*.ne11       =*/ ne11,
                /*.ne12       =*/ ne12,
                /*.ne13       =*/ ne13,
                /*.nb10       =*/ nb10,
                /*.nb11       =*/ nb11,
                /*.nb12       =*/ nb12,
                /*.ne0        =*/ (int32_t) situ->ne[0],
                /*.ne1        =*/ (int32_t) situ->ne[1],
                /*.nb1        =*/ situ->nb[1],
                /*.nr0        =*/ nr0,
                /*.beta       =*/ beta,
                /*.inv_beta   =*/ 1.0f / beta,
                /*.lb         =*/ lb,
                /*.inv_lb     =*/ lb > 0.0f ? (1.0f / lb) : 0.0f,
                /*.has_scales =*/ has_scales ? 1 : 0,
                /*.use_sorted =*/ can_sort ? 1 : 0,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src0,     1);
            ggml_metal_encoder_set_buffer  (enc, bid_src1,     2);
            ggml_metal_encoder_set_buffer  (enc, bid_situ_dst, 3);
            ggml_metal_encoder_set_buffer  (enc, bid_ids_in,   4);
            ggml_metal_encoder_set_buffer  (enc, bid_scales,   5);

            ggml_metal_encoder_dispatch_threadgroups(enc, ((int32_t) situ->ne[0] + nr0 * nsg - 1) / (nr0 * nsg), 1, ne20 * ne21, 32, nsg, 1);

            return 2;
        } else if (can_sort) {
        // Expert-sorted co-scheduled NVFP4/MXFP8 mat-vec: exact same zero-spill register kernel as ungrouped mul_mv_id,
        // but consecutive Z-slices share the same expert so repeat reads hit SLC/L2 at 1+ TB/s.
        const bool use_aux_srt = (ctx->aux_buf != nullptr && ctx->aux_buf_size >= 65536 && (size_t) (ne20 * ne21) * sizeof(uint32_t) <= 8192);
        ggml_metal_buffer_id bid_srt = use_aux_srt
            ? ggml_metal_buffer_id{ ctx->aux_buf, 57344 }
            : ggml_metal_buffer_id{ bid_dst.metal, bid_dst.offs + ggml_nbytes(op) };
        const bool srt_hit = use_aux_srt &&
            ctx->aux_sort_ids == op->src[2] && ctx->aux_sort_ne20 == ne20 &&
            ctx->aux_sort_ne21 == ne21 && ctx->aux_sort_ne02 == ne02 && ctx->aux_sort_mode == 1;

        if (!srt_hit) {
            if (use_aux_srt && ctx->aux_sort_ids != nullptr) {
                ggml_metal_op_concurrency_reset(ctx);
            }
            ggml_metal_kargs_mul_mv_id_group_compact cargs = {
                /*.nei0 =*/ ne20,
                /*.nei1 =*/ ne21,
                /*.ne02 =*/ ne02,
                /*.nbi1 =*/ nb21,
            };

            auto pl_srt = ggml_metal_library_get_pipeline_mul_mv_id_sort(lib);
            const int nth = ((ne02 + 31) / 32) * 32;

            ggml_metal_encoder_set_pipeline(enc, pl_srt);
            ggml_metal_encoder_set_bytes   (enc, &cargs, sizeof(cargs), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src2, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_srt,  2);

            ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

            ggml_metal_op_concurrency_reset(ctx);

            if (use_aux_srt) {
                ctx->aux_sort_ids  = op->src[2];
                ctx->aux_sort_ne20 = ne20;
                ctx->aux_sort_ne21 = ne21;
                ctx->aux_sort_ne02 = ne02;
                ctx->aux_sort_mode = 1;
            }
        }

        {
            auto pipeline = (op->src[0]->type == GGML_TYPE_MXFP8)
                ? ggml_metal_library_get_pipeline_mul_mv_id_sorted_mxfp8(lib)
                : ggml_metal_library_get_pipeline_mul_mv_id_sorted_nvfp4(lib, ne00);
            const int nr0 = pipeline.nr0;
            const int nsg = pipeline.nsg;

            ggml_metal_kargs_mul_mv_id args = {
                /*.nei0 =*/ ne20,
                /*.nei1 =*/ ne21,
                /*.nbi1 =*/ nb21,
                /*.ne00 =*/ ne00,
                /*.ne01 =*/ ne01,
                /*.ne02 =*/ ne02,
                /*.nb00 =*/ nb00,
                /*.nb01 =*/ nb01,
                /*.nb02 =*/ nb02,
                /*.ne10 =*/ ne10,
                /*.ne11 =*/ ne11,
                /*.ne12 =*/ ne12,
                /*.ne13 =*/ ne13,
                /*.nb10 =*/ nb10,
                /*.nb11 =*/ nb11,
                /*.nb12 =*/ nb12,
                /*.ne0  =*/ ne0,
                /*.ne1  =*/ ne1,
                /*.nb1  =*/ nb1,
                /*.nr0  =*/ nr0,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);
            ggml_metal_encoder_set_buffer  (enc, bid_srt,  4);

            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0*nsg - 1)/(nr0*nsg), 1, ne20*ne21, 32, nsg, 1);
        }
    } else {
        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_id(lib, op);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        const size_t smem = pipeline.smem;

        ggml_metal_kargs_mul_mv_id args = {
            /*.nei0 =*/ ne20,
            /*.nei1 =*/ ne21,
            /*.nbi1 =*/ nb21,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.ne13 =*/ ne13,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nb1  =*/ nb1,
            /*.nr0  =*/ nr0,
        };

        if (ggml_is_quantized(op->src[0]->type)) {
            GGML_ASSERT(ne00 >= nsg*nr0);
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer(enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst,  3);
        ggml_metal_encoder_set_buffer(enc, bid_src2, 4);

        const int64_t _ne1 = 1;
        const int64_t ne123 = ne20*ne21;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        if (op->src[0]->type == GGML_TYPE_F32 ||
            op->src[0]->type == GGML_TYPE_F16 ||
            op->src[0]->type == GGML_TYPE_BF16 ||
            op->src[0]->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0 - 1)/(nr0), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0*nsg - 1)/(nr0*nsg), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        }
        }
    }

    return 1;
}

int ggml_metal_op_add_id(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_kargs_add_id args = {
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb11 =*/ nb11,
        /*.nb21 =*/ nb21,
    };

    auto pipeline = ggml_metal_library_get_pipeline_base(lib, GGML_OP_ADD_ID);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, 1, nth, 1, 1);

    return 1;
}

bool ggml_metal_op_flash_attn_ext_use_vec(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    const int64_t ne00 = op->src[0]->ne[0]; // head size
    const int64_t ne01 = op->src[0]->ne[1]; // batch size

    // use vec kernel if the batch size is small and if the head size is supported
    return (ne01 < 20) && (ne00 % 32 == 0);
}

// Query rows per threadgroup of kernel_flash_attn_ext_mq for a vec-kernel op, or 0 to keep the vec kernel. The vec
// kernel reads the K and V rows of a KV head once per query and head; the GPU caches serve the repeats of a short cache
// but not of a long one, where reading each row once for all the queries of the KV head pays. On an M4 Pro, with
// 3 heads per KV head, that was from 2048 cells for 2 to 8 queries and from 16384 for one; the 1536 cells of the sliding
// window caches of Inkling 2 Mini stay with the vec kernel, which skips their masked chunks.
// GGML_METAL_FA_MQ_MIN_KV sets the cell count for every batch, to compare the kernels.
static int ggml_metal_op_flash_attn_ext_mq_rows(const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];

    static const int64_t min_kv_env = [] {
        const char * env = getenv("GGML_METAL_FA_MQ_MIN_KV");
        return env ? atoll(env) : -1;
    }();

    const int64_t rows   = q->ne[1]*(q->ne[2]/k->ne[2]);
    const int64_t min_kv = min_kv_env >= 0 ? min_kv_env
        : (q->ne[1] > 1 ? (k->ne[0] == 128 && v->ne[0] == 128 ? 1024 : 2048) : 16384);

    // the kernel reads whole rows of K and V
    const bool rows_f16 = k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 &&
                          k->nb[0] == sizeof(ggml_fp16_t) && v->nb[0] == sizeof(ggml_fp16_t);

    const int64_t max_rows = (k->ne[0] == 128 && v->ne[0] == 128) ? OP_FLASH_ATTN_EXT_MQ_ROWS : 24;
    if (!rows_f16 || k->ne[0] != v->ne[0] || (k->ne[0] != 64 && k->ne[0] != 128) ||
        rows < 2 || rows > max_rows || k->ne[1] < min_kv) {
        return 0;
    }

    return (int) GGML_PAD(rows, 8);
}

size_t ggml_metal_op_flash_attn_ext_extra_pad(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    const bool has_mask = op->src[3] != nullptr;

    // note: the non-vec kernel requires more extra memory, so always reserve for it
    GGML_ASSERT(OP_FLASH_ATTN_EXT_NCPSG >= OP_FLASH_ATTN_EXT_VEC_NCPSG);
    GGML_ASSERT(OP_FLASH_ATTN_EXT_NCPSG >= OP_FLASH_ATTN_EXT_MQ_NCPSG);

    //if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
    if (false) {
        // note: always reserve the padding space to avoid graph reallocations
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_VEC_NCPSG != 0;
        const bool has_kvpad = true;

        if (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_VEC_NCPSG*(
                nb11*ne12*ne13 +
                nb21*ne22*ne23 +
                (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    } else {
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_NCPSG != 0;
        const bool has_kvpad = true;

        if (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_NCPSG*(
                nb11*ne12*ne13 +
                nb21*ne22*ne23 +
                (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    }

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_blk(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    const bool has_mask = op->src[3] != nullptr;

    if (!has_mask) {
        return res;
    }

    const bool is_vec = ggml_metal_op_flash_attn_ext_use_vec(op);

    // this optimization is not useful for the vector kernels
    // note: always reserve the blk buffer to avoid graph reallocations
    //if (is_vec) {
    //    return res;
    //}

    const int nqptg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NQPSG : OP_FLASH_ATTN_EXT_NQPSG;
    const int ncpsg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NCPSG : OP_FLASH_ATTN_EXT_NCPSG;

    const int64_t ne1 = (ne01 + nqptg - 1)/nqptg;
    const int64_t ne0 = (ne30 + ncpsg - 1)/ncpsg;

    res += GGML_PAD(ggml_type_size(GGML_TYPE_I8)*ne0*ne1*ne32*ne33, 32);

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_tmp(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    // note: always reserve the temp buffer to avoid graph reallocations
    //if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
    if (true) {
        const int64_t nwg = 32;
        const int64_t ne01_max = std::min(ne01, 32);

        // temp buffer for writing the results from each workgroup
        // - ne20: the size of the Value head
        // -  + 2: the S and M values for each intermediate result
        res += ggml_type_size(GGML_TYPE_F32)*(ne01_max*ne02*ne03*nwg*(ne20 + 2));
    }

    return res;
}

// One encoding of a flash attention over the query rows that op holds. The extra buffers (padding, mask blocks,
// workgroup results) are the ones of the whole attention (ggml_metal_op_flash_attn_ext).
static int ggml_metal_op_flash_attn_ext_rows(ggml_metal_op_t ctx, ggml_tensor * op,
        ggml_metal_buffer_id bid_pad, ggml_metal_buffer_id bid_blk, ggml_metal_buffer_id bid_tmp) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS( int32_t, nb,  op,         nb);

    GGML_ASSERT(ne00 % 4 == 0);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == op->src[2]->type);

    //GGML_ASSERT(ggml_are_same_shape (src1, src2));
    GGML_ASSERT(ne11 == ne21);
    GGML_ASSERT(ne12 == ne22);

    GGML_ASSERT(!op->src[3] || op->src[3]->type == GGML_TYPE_F16);
    GGML_ASSERT(!op->src[3] || op->src[3]->ne[1] >= op->src[0]->ne[1] &&
            "the Flash-Attention Metal kernel requires the mask to be at least n_queries big");

    float scale;
    float max_bias;
    float logit_softcap;

    memcpy(&scale,         ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias,      ((const int32_t *) op->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) op->op_params) + 2, sizeof(logit_softcap));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const bool has_mask  = op->src[3] != NULL;
    const bool has_sinks = op->src[4] != NULL;
    const bool has_indexed_bias = ggml_flash_attn_ext_has_indexed_bias(op);

    const ggml_tensor * bias_table   = has_indexed_bias ? op->src[6] : nullptr;
    const ggml_tensor * bias_indices = has_indexed_bias ? op->src[7] : nullptr;

    GGML_ASSERT(!has_indexed_bias || (has_mask && bias_table && bias_indices));
    const int32_t bias_n_bins       = has_indexed_bias ? (int32_t) bias_table->ne[0] : 0;
    const int32_t bias_n_keys       = has_indexed_bias ? (int32_t) bias_indices->ne[0] : 0;
    const int32_t bias_query_stride = has_indexed_bias ? (int32_t) (bias_table->ne[0]*bias_table->ne[1]) : 0;

    const bool has_bias  = max_bias != 0.0f;
    const bool has_scap  = logit_softcap != 0.0f;

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    GGML_ASSERT(ne01 < 65536);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_src3 = has_mask         ? ggml_metal_get_buffer_id(op->src[3])   : bid_src0;
    ggml_metal_buffer_id bid_src4 = has_sinks        ? ggml_metal_get_buffer_id(op->src[4])   : bid_src0;
    ggml_metal_buffer_id bid_src6 = has_indexed_bias ? ggml_metal_get_buffer_id(bias_table)   : bid_src0;
    ggml_metal_buffer_id bid_src7 = has_indexed_bias ? ggml_metal_get_buffer_id(bias_indices) : bid_src0;

    ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);

    if (!ggml_metal_op_flash_attn_ext_use_vec(op)) {
        // half8x8 kernel
        const int nqptg = OP_FLASH_ATTN_EXT_NQPSG; // queries per threadgroup
        const int ncpsg = OP_FLASH_ATTN_EXT_NCPSG; // cache values per simdgroup

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg  % 8  == 0);
        GGML_ASSERT(ncpsg  % 32 == 0);

        bool need_sync = false;

        const bool has_kvpad = ne11 % ncpsg != 0;

        if (has_kvpad) {
            assert(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

            ggml_metal_kargs_flash_attn_ext_pad args0 = {
                /*.ne11    =*/ne11,
                /*.ne_12_2 =*/ne12,
                /*.ne_12_3 =*/ne13,
                /*.nb11    =*/nb11,
                /*.nb12    =*/nb12,
                /*.nb13    =*/nb13,
                /*.nb21    =*/nb21,
                /*.nb22    =*/nb22,
                /*.nb23    =*/nb23,
                /*.ne31    =*/ne31,
                /*.ne32    =*/ne32,
                /*.ne33    =*/ne33,
                /*.nb31    =*/nb31,
                /*.nb32    =*/nb32,
                /*.nb33    =*/nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_src2, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
            ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);

            assert(ne12 == ne22);
            assert(ne13 == ne23);

            ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        if (has_mask) {
            assert(ggml_metal_op_flash_attn_ext_extra_blk(op) != 0);

            ggml_metal_kargs_flash_attn_ext_blk args0 = {
                /*.ne01        =*/ ne01,
                /*.ne30        =*/ ne30,
                /*.ne31        =*/ ne31,
                /*.ne32        =*/ ne32,
                /*.ne33        =*/ ne33,
                /*.nb31        =*/ nb31,
                /*.nb32        =*/ nb32,
                /*.nb33        =*/ nb33,
                /*.bias_n_bins =*/ 0,
                /*.bias_n_keys =*/ 0,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_blk,  2);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);

            const int32_t nblk1 = ((ne01 + nqptg - 1)/nqptg);
            const int32_t nblk0 = ((ne30 + ncpsg - 1)/ncpsg);

            ggml_metal_encoder_dispatch_threadgroups(enc, nblk0, nblk1, ne32*ne33, 32, 1, 1);

            need_sync = true;
        }

        if (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        const int is_q = ggml_is_quantized(op->src[1]->type) ? 1 : 0;

        // 2*(2*ncpsg)
        // ncpsg soft_max values + ncpsg mask values
        //
        // 16*32*(nsg)
        // the shared memory needed for the simdgroups to load the KV cache
        // each thread loads (dequantizes) 16 head elements, there are 32 threads in th SG
        //
#define FATTN_SMEM(nsg) (GGML_PAD((nqptg*(ne00 + 2*GGML_PAD(ne20, 64) + 2*(2*ncpsg)) + is_q*(16*32*(nsg)))*(sizeof(float)/2), 16))

        //int64_t nsgmax = 4;
        //
        //if (is_q) {
        //    nsgmax = 2;
        //    while (true) {
        //        const size_t smem = FATTN_SMEM(nsgmax);
        //        if (smem > props_dev->max_theadgroup_memory_size) {
        //            break;
        //        }
        //        nsgmax *= 2;
        //    }
        //    nsgmax /= 2;
        //}

        // simdgroups per threadgroup (a.k.a. warps)
        //nsg = ne01 <= nqptg ? MAX(4, MIN(nsgmax, MIN(ne11/ncpsg, (int64_t) pipeline.maxTotalThreadsPerThreadgroup/32))) : 4;
        int32_t nsg = ne00 >= 512 ? 8 : 4;

        const size_t smem = FATTN_SMEM(nsg);

        ggml_metal_kargs_flash_attn_ext args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ int32_t(nb11/nb10),
            /*.nb11          =*/ nb11,
            /*.nb12          =*/ nb12,
            /*.nb13          =*/ nb13,
            /*.ns20          =*/ int32_t(nb21/nb20),
            /*.nb21          =*/ nb21,
            /*.nb22          =*/ nb22,
            /*.nb23          =*/ nb23,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
            /*.bias_n_bins       =*/ bias_n_bins,
            /*.bias_n_keys       =*/ bias_n_keys,
            /*.bias_query_stride =*/ bias_query_stride,
        };

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, has_indexed_bias, nsg);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, bid_pad,  6);
        ggml_metal_encoder_set_buffer  (enc, bid_blk,  7);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  8);
        ggml_metal_encoder_set_buffer  (enc, bid_src6, 9);
        ggml_metal_encoder_set_buffer  (enc, bid_src7, 10);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03, 32, nsg, 1);
#undef FATTN_SMEM
    } else {
        // half4x4 kernel, or its variant that takes all the query rows of a KV head
        const int mq_rows = ggml_metal_op_flash_attn_ext_mq_rows(op);

        const int nqptg = OP_FLASH_ATTN_EXT_VEC_NQPSG; // queries per threadgroup
        const int ncpsg = mq_rows ? OP_FLASH_ATTN_EXT_MQ_NCPSG : OP_FLASH_ATTN_EXT_VEC_NCPSG; // cache values per simdgroup !! sync with kernel template arguments !!
        const int nhptg = 1;                           // heads per threadgroup

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg  % 1  == 0);
        GGML_ASSERT(ncpsg  % 32 == 0);

        bool need_sync = false;

        const bool has_kvpad = ne11 % ncpsg != 0;

        if (has_kvpad) {
            assert(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

            ggml_metal_kargs_flash_attn_ext_pad args0 = {
                /*.ne11    =*/ne11,
                /*.ne_12_2 =*/ne12,
                /*.ne_12_3 =*/ne13,
                /*.nb11    =*/nb11,
                /*.nb12    =*/nb12,
                /*.nb13    =*/nb13,
                /*.nb21    =*/nb21,
                /*.nb22    =*/nb22,
                /*.nb23    =*/nb23,
                /*.ne31    =*/ne31,
                /*.ne32    =*/ne32,
                /*.ne33    =*/ne33,
                /*.nb31    =*/nb31,
                /*.nb32    =*/nb32,
                /*.nb33    =*/nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_src2, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
            ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);

            assert(ne12 == ne22);
            assert(ne13 == ne23);

            ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        // the multi-query kernel skips the chunks that the mask hides from all its queries, by blocks of 1 query and
        // ncpsg cache items; cache the blk flags per (mask, bias_indices) tensor pair in the CB's private aux_buf
        // so all 40 SWA layers and all 8 Full-Attn layers reuse the exact same mask scan without extra barriers.
        if (mq_rows && has_mask) {
            static_assert(OP_FLASH_ATTN_EXT_VEC_NQPSG == 1 && OP_FLASH_ATTN_EXT_MQ_NCPSG >= OP_FLASH_ATTN_EXT_VEC_NCPSG, "blk buffer too small");

            const bool check_far_bins = has_indexed_bias && (ne03 == ne33) && (bias_n_bins > 0);
            const int32_t eff_bins    = check_far_bins ? bias_n_bins : 0;
            const int32_t eff_keys    = check_far_bins ? bias_n_keys : 0;
            const size_t  blk_bytes   = (size_t) ((ne30 + ncpsg - 1) / ncpsg) * (size_t) ne01 * (size_t) ne32 * (size_t) ne33;
            const bool    can_cache   = (ctx->aux_buf != nullptr && ctx->aux_buf_size >= 65536 && blk_bytes <= 28672);

            int cache_slot = -1;
            bool blk_hit   = false;
            if (can_cache) {
                const ggml_tensor * key_bias = check_far_bins ? bias_indices : nullptr;
                for (int s = 0; s < 2; ++s) {
                    const auto & e = ctx->aux_blk[s];
                    if (e.valid && e.mask == op->src[3] && e.bias == key_bias &&
                        e.ne01 == ne01 && e.ne30 == ne30 && e.ne31 == ne31 && e.ne32 == ne32 && e.ne33 == ne33 &&
                        e.ncpsg == ncpsg && e.bias_n_bins == eff_bins && e.bias_n_keys == eff_keys) {
                        cache_slot = s;
                        blk_hit    = true;
                        break;
                    }
                }
                if (!blk_hit) {
                    cache_slot = ctx->aux_blk_next;
                    ctx->aux_blk_next ^= 1;
                    if (ctx->aux_blk[cache_slot].valid) {
                        ggml_metal_op_concurrency_reset(ctx);
                    }
                }
                bid_blk = ggml_metal_buffer_id{ ctx->aux_buf, (size_t) cache_slot * 28672 };
            }

            if (!blk_hit) {
                ggml_metal_kargs_flash_attn_ext_blk args0 = {
                    /*.ne01        =*/ ne01,
                    /*.ne30        =*/ ne30,
                    /*.ne31        =*/ ne31,
                    /*.ne32        =*/ ne32,
                    /*.ne33        =*/ ne33,
                    /*.nb31        =*/ nb31,
                    /*.nb32        =*/ nb32,
                    /*.nb33        =*/ nb33,
                    /*.bias_n_bins =*/ eff_bins,
                    /*.bias_n_keys =*/ eff_keys,
                };

                auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, 1, ncpsg);

                ggml_metal_encoder_set_pipeline(enc, pipeline0);
                ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_blk,  2);
                ggml_metal_encoder_set_buffer  (enc, check_far_bins ? bid_src7 : bid_src3, 3);

                ggml_metal_encoder_dispatch_threadgroups(enc, (ne30 + ncpsg - 1)/ncpsg, ne01, ne32*ne33, 32, 1, 1);

                need_sync = true;

                if (can_cache) {
                    auto & e = ctx->aux_blk[cache_slot];
                    e.mask        = op->src[3];
                    e.bias        = check_far_bins ? bias_indices : nullptr;
                    e.ne01        = ne01;
                    e.ne30        = ne30;
                    e.ne31        = ne31;
                    e.ne32        = ne32;
                    e.ne33        = ne33;
                    e.ncpsg       = ncpsg;
                    e.bias_n_bins = eff_bins;
                    e.bias_n_keys = eff_keys;
                    e.valid       = true;
                }
            }
        }

        if (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        // note: for simplicity assume the K is larger or equal than V
        GGML_ASSERT(ne10 >= ne20);

        // ne00 + 2*ncpsg*(nsg)
        // for each query, we load it as f16 in shared memory (ne00)
        // and store the soft_max values and the mask
        //
        // ne20*(nsg)
        // each simdgroup has a full f32 head vector in shared mem to accumulate results
        //
#define FATTN_SMEM(nsg) (GGML_PAD(((GGML_PAD(ne00, 128) + 4*ncpsg + 2*GGML_PAD(ne20, 128))*(nsg))*(sizeof(float)/2), 16))

        int64_t nsg = 1;

        // workgroups
        // each workgroup handles nsg*nkpsg cache values
        int32_t nwg = 1;
        if (mq_rows) {
            nwg = (ne11 <= 512 || (ne11 <= 2048 && mq_rows >= 24)) ? 4
                : (ne11 <= 2048) ? 8
                : ((ne11 <= 8192 && mq_rows >= 24) || (ne11 > 8192 && mq_rows >= 40)) ? 20
                : OP_FLASH_ATTN_EXT_MQ_NWG;
            const bool use_mq_nsg4 = ((mq_rows == 24 || mq_rows == 32) && ne10 == 128 && ne20 == 128 && ne11 > 8192);
            nsg = (mq_rows >= 24 && ne10 == 128 && ne20 == 128 && !use_mq_nsg4) ? 8 : OP_FLASH_ATTN_EXT_MQ_NSG;
        } else if (false) {
            // for small KV caches, we could launch a single workgroup and write the results directly to dst/
            // however, this does not lead to significant improvement, so disabled
            nwg = 1;
            nsg = 4;
        } else {
            nwg = (ne11 <= 512) ? 4 : ((ne11 <= 2048) ? 8 : 32);
            nsg = 1;
            while (2*nwg*nsg*ncpsg < ne11 && nsg < 4) {
                nsg *= 2;
            }
        }

        ggml_metal_kargs_flash_attn_ext_vec args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ int32_t(nb11/nb10),
            /*.nb11          =*/ nb11,
            /*.nb12          =*/ nb12,
            /*.nb13          =*/ nb13,
            /*.ns20          =*/ int32_t(nb21/nb20),
            /*.nb21          =*/ nb21,
            /*.nb22          =*/ nb22,
            /*.nb23          =*/ nb23,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
            /*.bias_n_bins       =*/ bias_n_bins,
            /*.bias_n_keys       =*/ bias_n_keys,
            /*.bias_query_stride =*/ bias_query_stride,
            /*.ne30              =*/ ne30,
        };

        // Keep sq in half (13 KB smem -> 2 TGs/core on M4 Pro instead of 19.2 KB -> 1 TG/core);
        // test-flash-attn-indexed-bias verifies max error ~1e-8 for both SWA and full attention.
        auto pipeline = mq_rows
            ? ggml_metal_library_get_pipeline_flash_attn_ext_mq (lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, has_indexed_bias, mq_rows, nsg, nwg)
            : ggml_metal_library_get_pipeline_flash_attn_ext_vec(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, has_indexed_bias, nsg, nwg);

        GGML_ASSERT(nsg*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, bid_src6, 8);
        ggml_metal_encoder_set_buffer  (enc, bid_src7, 9);
        if (mq_rows) {
            ggml_metal_encoder_set_buffer(enc, bid_blk, 10);
        }

        // the multi-query kernel holds queries (half) and scores (float); with half Q (HALF_PV=true),
        // sd/s_ms/s_rescale pack inside the unused upper half of ss so r32 fits in exact 16,384 B (2 TGs/core).
        const size_t smem = mq_rows
            ? GGML_PAD(std::max<size_t>(
                    (size_t) 2 * mq_rows * ne00 +
                    (size_t) 4 * mq_rows * ncpsg,
                    (size_t) 4 * 8 * ne20), 16)
            : FATTN_SMEM(nsg);

        //printf("smem: %zu, max: %zu, nsg = %d, nsgmax = %d\n", smem, props_dev->max_theadgroup_memory_size, (int) nsg, (int) nsgmax);
        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        if (nwg == 1) {
            assert(ggml_metal_op_flash_attn_ext_extra_tmp(op) == 0);

            // using 1 workgroup -> write the result directly into dst
            ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            ggml_metal_encoder_set_buffer(enc, bid_dst, 7);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, (ne02 + nhptg - 1)/nhptg, ne03*nwg, 32, nsg, 1);
        } else {
            // sanity checks
            assert(ggml_metal_op_flash_attn_ext_extra_tmp(op) != 0);

            GGML_ASSERT(ne01*ne02*ne03 == ne1*ne2*ne3);
            GGML_ASSERT((uint64_t)ne1*ne2*ne3 <= (1u << 31));

            // write the results from each workgroup into a temp buffer
            ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            ggml_metal_encoder_set_buffer(enc, bid_tmp, 7);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
            if (mq_rows) {
                ggml_metal_encoder_dispatch_threadgroups(enc, 1, ne12, ne03*nwg, 32, nsg, 1);
            } else {
                ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, (ne02 + nhptg - 1)/nhptg, ne03*nwg, 32, nsg, 1);
            }

            // sync the 2 kernels
            ggml_metal_op_concurrency_reset(ctx);

            // reduce the results from the workgroups
            {
                const int32_t nrows = ne1*ne2*ne3;

                ggml_metal_kargs_flash_attn_ext_vec_reduce args0 = {
                    nrows,
                };

                auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, ne20, nwg);

                ggml_metal_encoder_set_pipeline(enc, pipeline0);
                ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

                ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, 32*nwg, 1, 1);
            }
        }
#undef FATTN_SMEM
    }

    return 1;
}

// rows [r0, r0 + rows) of t along dim, as a tensor at the same place in the same buffer
static ggml_tensor ggml_metal_op_row_range(const ggml_tensor * t, int dim, int64_t r0, int64_t rows) {
    ggml_tensor range = *t;
    const size_t offset = (size_t) r0 * t->nb[dim];
    range.ne[dim] = rows;
    range.data = (char *) t->data + offset;
    if (t->view_src) {
        range.view_offs += offset;
    }
    return range;
}

// The flash attention kernel holds a query row's index in 16 bits (ushort iq1), so one encoding takes fewer than 65536
// query rows. A prompt of more tokens in one ubatch runs as encodings over ranges of its rows, each from its rows of q,
// the mask and dst. A range starts at a multiple of 32 rows, which the row tiles of the kernel divide, so each range
// computes what its rows get in one encoding; the ranges share the extra buffers, so each waits for the one before.
static constexpr int64_t METAL_FA_MAX_ROWS = 65536 - 32;

int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_buffer_id bid_pad = ggml_metal_get_buffer_id(op);
    bid_pad.offs += ggml_nbytes(op);

    ggml_metal_buffer_id bid_blk = bid_pad;
    bid_blk.offs += ggml_metal_op_flash_attn_ext_extra_pad(op);

    ggml_metal_buffer_id bid_tmp = bid_blk;
    bid_tmp.offs += ggml_metal_op_flash_attn_ext_extra_blk(op);

    const ggml_tensor * q    = op->src[0];
    const ggml_tensor * mask = op->src[3];
    const int64_t N = q->ne[1];
    if (N <= METAL_FA_MAX_ROWS) {
        return ggml_metal_op_flash_attn_ext_rows(ctx, op, bid_pad, bid_blk, bid_tmp);
    }

    // the kernel addresses q, the mask and dst as one block of rows per sequence
    GGML_ASSERT(q->ne[3] == 1 && op->ne[3] == 1 && (!mask || (mask->ne[2] == 1 && mask->ne[3] == 1)));

    const int64_t ranges = (N + METAL_FA_MAX_ROWS - 1) / METAL_FA_MAX_ROWS;
    const int64_t rows   = ((N + ranges - 1) / ranges + 31) / 32 * 32;
    for (int64_t r0 = 0; r0 < N; r0 += rows) {
        const int64_t r1 = std::min(N, r0 + rows);
        if (r0 > 0) {
            ggml_metal_op_concurrency_reset(ctx);
        }
        ggml_tensor q_range = ggml_metal_op_row_range(q, 1, r0, r1 - r0);
        ggml_tensor mask_range;
        if (mask) {
            mask_range = ggml_metal_op_row_range(mask, 1, r0, (r1 == N ? mask->ne[1] : r1) - r0);
        }
        ggml_tensor table_range, indices_range;
        const bool has_ib = ggml_flash_attn_ext_has_indexed_bias(op);
        if (has_ib) {
            table_range   = ggml_metal_op_row_range(op->src[6], 2, r0, r1 - r0);
            indices_range = ggml_metal_op_row_range(op->src[7], 1, r0, r1 - r0);
        }
        ggml_tensor op_range = ggml_metal_op_row_range(op, 2, r0, r1 - r0);
        op_range.src[0] = &q_range;
        op_range.src[3] = mask ? &mask_range : nullptr;
        if (has_ib) {
            op_range.src[6] = &table_range;
            op_range.src[7] = &indices_range;
        }
        ggml_metal_op_flash_attn_ext_rows(ctx, &op_range, bid_pad, bid_blk, bid_tmp);
    }

    return 1;
}

int ggml_metal_op_bin(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const bool use_fusion = ctx->use_fusion;

    const int debug_fusion = ctx->debug_fusion;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[1]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    // Fused Inkling2 clip: MUL(x, w) -> L2_NORM(eps) [-> SCALE(post_scale) | MUL(tau [1, 1, ne02])]
    if (use_fusion && op->op == GGML_OP_MUL &&
        ne00 % 4 == 0 && ne00 <= 128 && ne03 == 1 && ne13 == 1 &&
        ne00 == ne10 && (ne11 == 1 || ne11 == ne01) && (ne12 == 1 || ne12 == ne02)) {
        const ggml_op ops_2[2] = { GGML_OP_MUL, GGML_OP_L2_NORM };
        if (ctx->can_fuse(idx, ops_2, 2)) {
            ggml_tensor * n1 = ctx->node(idx + 1);
            if (n1->src[0] == op && n1->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(n1, op) && ggml_is_contiguous_rows(n1)) {
                float eps = 0.0f;
                memcpy(&eps, n1->op_params, sizeof(float));

                ggml_tensor * dst_t = n1;
                ggml_tensor * tau_t = nullptr;
                float post_scale = 1.0f;
                int n_consumed = 2;

                const ggml_op ops_scale[3] = { GGML_OP_MUL, GGML_OP_L2_NORM, GGML_OP_SCALE };
                const ggml_op ops_mul[3]   = { GGML_OP_MUL, GGML_OP_L2_NORM, GGML_OP_MUL };
                if (ctx->can_fuse(idx, ops_scale, 3)) {
                    ggml_tensor * n2 = ctx->node(idx + 2);
                    float s = 1.0f, b = 0.0f;
                    memcpy(&s, n2->op_params,     sizeof(float));
                    memcpy(&b, n2->op_params + 1, sizeof(float));
                    if (n2->src[0] == n1 && n2->type == GGML_TYPE_F32 && b == 0.0f &&
                        ggml_are_same_shape(n2, n1) && ggml_is_contiguous_rows(n2) &&
                        !ggml_metal_tensor_overlaps_unsafe(op->src[0], n2, true) &&
                        !ggml_metal_tensor_overlaps_unsafe(op->src[1], n2, false)) {
                        post_scale = s;
                        dst_t      = n2;
                        n_consumed = 3;
                    }
                } else if (ctx->can_fuse(idx, ops_mul, 3)) {
                    ggml_tensor * n2 = ctx->node(idx + 2);
                    ggml_tensor * tt = (n2->src[0] == n1) ? n2->src[1] : ((n2->src[1] == n1) ? n2->src[0] : nullptr);
                    if (tt && tt->type == GGML_TYPE_F32 && n2->type == GGML_TYPE_F32 &&
                        tt->ne[0] == 1 && tt->ne[1] == 1 && tt->ne[2] == ne02 && tt->ne[3] == 1 &&
                        ggml_is_contiguous(tt) && ggml_are_same_shape(n2, n1) && ggml_is_contiguous_rows(n2) &&
                        !ggml_metal_tensor_overlaps_unsafe(op->src[0], n2, true) &&
                        !ggml_metal_tensor_overlaps_unsafe(op->src[1], n2, false) &&
                        !ggml_metal_tensor_overlaps_unsafe(tt,         n2, false)) {
                        tau_t      = tt;
                        dst_t      = n2;
                        n_consumed = 3;
                    }
                }

                if (n_consumed == 2 &&
                    (ggml_metal_tensor_overlaps_unsafe(op->src[0], dst_t, true) ||
                     ggml_metal_tensor_overlaps_unsafe(op->src[1], dst_t, false))) {
                    n_consumed = 0;
                }

                if (n_consumed >= 2) {
                    // Also fuse trailing Inkling2 rel_bias MUL_MAT(rel_table [16, ne01], Rcur) [+ SUB(sinks)]
                    if (ne00 == 16 && ne01 <= 64 && ne03 == 1 && (ne11 == 1 || ne11 == ne01) && ne12 == 1 &&
                        idx + n_consumed < ctx->n_nodes() && ctx->node_has_n_uses(idx + n_consumed - 1, 1)) {
                        ggml_tensor * n_mm = ctx->node(idx + n_consumed);
                        if (n_mm->op == GGML_OP_MUL_MAT && n_mm->src[1] == dst_t &&
                            n_mm->src[0]->type == GGML_TYPE_F32 && n_mm->type == GGML_TYPE_F32 &&
                            n_mm->src[0]->ne[0] == 16 && n_mm->src[0]->ne[2] == 1 && n_mm->src[0]->ne[3] == 1 &&
                            !ggml_is_transposed(n_mm->src[0]) && ggml_is_contiguous_rows(n_mm->src[0]) &&
                            ggml_is_contiguous_rows(n_mm)) {
                            ggml_tensor * out_t  = n_mm;
                            ggml_tensor * sink_t = nullptr;
                            int total_consumed   = n_consumed + 1;
                            if (idx + total_consumed < ctx->n_nodes() && ctx->node_has_n_uses(idx + n_consumed, 1)) {
                                ggml_tensor * n_sub = ctx->node(idx + total_consumed);
                                if (n_sub->op == GGML_OP_SUB && n_sub->src[0] == n_mm &&
                                    n_sub->type == GGML_TYPE_F32 && n_sub->src[1]->type == GGML_TYPE_F32 &&
                                    n_sub->src[1]->ne[0] == 1 && n_sub->src[1]->ne[1] == ne01 &&
                                    n_sub->src[1]->ne[2] == ne02 && n_sub->src[1]->ne[3] == 1 &&
                                    ggml_are_same_shape(n_sub, n_mm) && ggml_is_contiguous_rows(n_sub)) {
                                    out_t  = n_sub;
                                    sink_t = n_sub->src[1];
                                    total_consumed += 1;
                                }
                            }
                            if (!ggml_metal_tensor_overlaps_unsafe(n_mm->src[0], out_t, false) &&
                                !ggml_metal_tensor_overlaps_unsafe(op->src[0],   out_t, false) &&
                                !ggml_metal_tensor_overlaps_unsafe(op->src[1],   out_t, false) &&
                                !ggml_metal_tensor_overlaps_unsafe(tau_t,        out_t, false) &&
                                !ggml_metal_tensor_overlaps_unsafe(sink_t,       out_t, false)) {
                                ggml_metal_op_concurrency_check_fused(ctx, idx, total_consumed);
                                auto pl_rb = ggml_metal_library_get_pipeline_inkling_rel_bias(lib);
                                ggml_metal_kargs_inkling_rel_bias rb_args = {
                                    /*.ne01       =*/ (int32_t) n_mm->src[0]->ne[1],
                                    /*.ne11       =*/ ne01,
                                    /*.ne12       =*/ ne02,
                                    /*.has_sink   =*/ sink_t ? 1 : 0,
                                    /*.has_clip   =*/ 1,
                                    /*.has_tau    =*/ tau_t ? 1 : 0,
                                    /*.clip_eps   =*/ eps,
                                    /*.post_scale =*/ post_scale,
                                    /*.nb01       =*/ n_mm->src[0]->nb[1],
                                    /*.nb11       =*/ nb01,
                                    /*.nb12       =*/ nb02,
                                    /*.nb21       =*/ sink_t ? sink_t->nb[1] : 0,
                                    /*.nb22       =*/ sink_t ? sink_t->nb[2] : 0,
                                    /*.nb1        =*/ out_t->nb[1],
                                    /*.nb2        =*/ out_t->nb[2],
                                    /*.nb_cw1     =*/ (ne11 == 1) ? 0 : nb11,
                                    /*.nb_tau2    =*/ tau_t ? tau_t->nb[2] : 0,
                                };
                                ggml_metal_buffer_id bid_rt = ggml_metal_get_buffer_id(n_mm->src[0]);
                                ggml_metal_encoder_set_pipeline(enc, pl_rb);
                                ggml_metal_encoder_set_bytes   (enc, &rb_args, sizeof(rb_args), 0);
                                ggml_metal_encoder_set_buffer  (enc, bid_rt, 1);
                                ggml_metal_encoder_set_buffer  (enc, bid_src0, 2);
                                ggml_metal_encoder_set_buffer  (enc, sink_t ? ggml_metal_get_buffer_id(sink_t) : bid_src0, 3);
                                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(out_t), 4);
                                ggml_metal_encoder_set_buffer  (enc, bid_src1, 5);
                                ggml_metal_encoder_set_buffer  (enc, tau_t ? ggml_metal_get_buffer_id(tau_t) : bid_src1, 6);
                                ggml_metal_encoder_dispatch_threadgroups(enc, ((int32_t) n_mm->src[0]->ne[1] + 31) / 32, ne02, 1, 32, 4, 1);
                                return total_consumed;
                            }
                        }
                    }

                ggml_metal_op_concurrency_check_fused(ctx, idx, n_consumed);

                auto pl_cl2 = ggml_metal_library_get_pipeline_clip_l2_norm(lib);
                ggml_metal_kargs_clip_l2_norm cargs = {
                    /*.ne04       =*/ ne00 / 4,
                    /*.ne01       =*/ ne01,
                    /*.ne02       =*/ ne02,
                    /*.ne_w1      =*/ ne11,
                    /*.ne_w2      =*/ ne12,
                    /*.has_tau    =*/ tau_t ? 1 : 0,
                    /*.eps        =*/ eps,
                    /*.post_scale =*/ post_scale,
                    /*.nb01       =*/ nb01,
                    /*.nb02       =*/ nb02,
                    /*.nbw1       =*/ nb11,
                    /*.nbw2       =*/ nb12,
                    /*.nbd1       =*/ dst_t->nb[1],
                    /*.nbd2       =*/ dst_t->nb[2],
                };
                ggml_metal_encoder_set_pipeline(enc, pl_cl2);
                ggml_metal_encoder_set_bytes   (enc, &cargs, sizeof(cargs), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
                ggml_metal_encoder_set_buffer  (enc, tau_t ? ggml_metal_get_buffer_id(tau_t) : bid_src1, 3);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst_t), 4);
                ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, 1, 32, 1, 1);
                return n_consumed;
                }
            }
        }
    }

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.offs =*/ 0,
        /*.o1   =*/ { bid_src1.offs },
    };

    ggml_op fops[8];

    int n_fuse = 1;

    // c[0] = add(a,    b[0])
    // c[1] = add(c[0], b[1])
    // c[2] = add(c[1], b[2])
    // ...
    if (use_fusion) {
        fops[0] = GGML_OP_ADD;
        fops[1] = GGML_OP_ADD;
        fops[2] = GGML_OP_ADD;
        fops[3] = GGML_OP_ADD;
        fops[4] = GGML_OP_ADD;
        fops[5] = GGML_OP_ADD;
        fops[6] = GGML_OP_ADD;
        fops[7] = GGML_OP_ADD;

        // note: in metal, we sometimes encode the graph in parallel so we have to avoid fusing ops
        //       across splits. idx_end indicates the last node in the current split
        for (n_fuse = 0; n_fuse <= 6; ++n_fuse) {
            if (!ctx->can_fuse(idx + n_fuse, fops + n_fuse, 2)) {
                break;
            }

            ggml_tensor * f0 = ctx->node(idx + n_fuse);
            ggml_tensor * f1 = ctx->node(idx + n_fuse + 1);

            if (f0 != f1->src[0]) {
                break;
            }

            // b[0] === b[1] === ...
            if (!ggml_are_same_layout(f0->src[1], f1->src[1])) {
                break;
            }

            // only fuse ops if src1 is in the same Metal buffer
            ggml_metal_buffer_id bid_fuse = ggml_metal_get_buffer_id(f1->src[1]);
            if (bid_fuse.metal != bid_src1.metal) {
                break;
            }

            //ctx->fuse_cnt[ops[n_fuse + 1]->op]++;

            args.o1[n_fuse + 1] = bid_fuse.offs;
        }

        ++n_fuse;

        if (debug_fusion > 1 && n_fuse > 1) {
            GGML_LOG_DEBUG("%s: fuse: ADD x %d\n", __func__, n_fuse);
        }
    }

    // the offsets of src1 and all fused buffers are relative to the start of the src1 buffer
    bid_src1.offs = 0;

    struct ggml_metal_pipeline_with_params pipeline;

    pipeline = ggml_metal_library_get_pipeline_bin(lib, op, n_fuse);

    if (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1));

        for (int i = 1; i < n_fuse; ++i) {
            if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
                ggml_metal_op_concurrency_reset(ctx);

                break;
            }
        }
    }

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne10 = ne10/4;
        args.ne0  = ne0/4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);

    if (pipeline.cnt) {
        const int64_t n = (int64_t) args.ne0*ggml_nrows(op);

        const int nth = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_dispatch_threadgroups(enc, (n + nth - 1)/nth, 1, 1, nth, 1, 1);
    } else {
        const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        int nth = 1;

        while (2*nth < args.ne0 && nth < nth_max) {
            nth *= 2;
        }

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    return n_fuse;
}

int ggml_metal_op_l2_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_kargs_l2_norm args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.eps   =*/ eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_l2_norm(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    int nth = 32; // SIMD width

    while (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_group_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t ngrp = ((const int32_t *) op->op_params)[0];

    float eps;
    memcpy(&eps, op->op_params + 1, sizeof(float));

    ggml_metal_kargs_group_norm args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ngrp =*/ ngrp,
        /*.eps  =*/ eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_group_norm(lib, op);

    int nth = 32; // SIMD width
    //while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
    //    nth *= 2;
    //}

    //nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    //nth = std::min(nth, ne00/4);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ngrp, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const bool use_fusion = ctx->use_fusion;

    const int debug_fusion = ctx->debug_fusion;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_norm args = {
        /*.ne00   =*/ ne00,
        /*.ne00_t =*/ ne00 % 4 == 0 ? ne00/4 : ne00,
        /*.nb1    =*/ nb1,
        /*.nb2    =*/ nb2,
        /*.nb3    =*/ nb3,
        /*.eps    =*/ eps,
        /*.nef1   =*/ { ne01 },
        /*.nef2   =*/ { ne02 },
        /*.nef3   =*/ { ne03 },
        /*.nbf1   =*/ { nb01 },
        /*.nbf2   =*/ { nb02 },
        /*.nbf3   =*/ { nb03 },
    };

    ggml_op fops[8];

    int n_fuse = 1;

    ggml_metal_buffer_id bid_fuse[2] = { bid_src0, bid_src0 };

    // d[0] = norm(a)
    // d[1] = mul(d[0], b)
    // d[2] = add(d[1], c)
    if (use_fusion) {
        if (op->op == GGML_OP_RMS_NORM && idx + 1 < ctx->n_nodes() && ctx->node_has_n_uses(idx, 1)) {
            ggml_tensor * n_mix = ctx->node(idx + 1);
            if (n_mix->op == GGML_OP_INKLING_MIX && n_mix->src[0] == op &&
                op->src[0]->type == GGML_TYPE_F32 && op->src[0]->nb[0] == sizeof(float) &&
                op->src[0]->ne[2] == 1 && op->src[0]->ne[3] == 1) {
                if (const int fused = ggml_metal_op_inkling_mix_impl(ctx, idx + 1, op)) {
                    return fused;
                }
            }
        }
        fops[0] = op->op;
        fops[1] = GGML_OP_MUL;
        fops[2] = GGML_OP_ADD;

        for (n_fuse = 0; n_fuse <= 1; ++n_fuse) {
            if (!ctx->can_fuse(idx + n_fuse, fops + n_fuse, 2)) {
                break;
            }

            ggml_tensor * f0 = ctx->node(idx + n_fuse);
            ggml_tensor * f1 = ctx->node(idx + n_fuse + 1);

            if (f0 != f1->src[0]) {
                break;
            }

            if (f1->src[1]->ne[0] != op->ne[0]) {
                break;
            }

            if (!ggml_is_contiguous_rows(f1->src[1])) {
                break;
            }

            if (f1->type != GGML_TYPE_F32) {
                break;
            }

            //ctx->fuse_cnt[f1->op]++;

            bid_fuse[n_fuse] = ggml_metal_get_buffer_id(f1->src[1]);

            args.nef1[n_fuse + 1] = f1->src[1]->ne[1];
            args.nef2[n_fuse + 1] = f1->src[1]->ne[2];
            args.nef3[n_fuse + 1] = f1->src[1]->ne[3];

            args.nbf1[n_fuse + 1] = f1->src[1]->nb[1];
            args.nbf2[n_fuse + 1] = f1->src[1]->nb[2];
            args.nbf3[n_fuse + 1] = f1->src[1]->nb[3];
        }

        ++n_fuse;

        if (debug_fusion > 1 && n_fuse > 1) {
            if (n_fuse == 2) {
                GGML_LOG_DEBUG("%s: fuse: %s + MUL\n", __func__, ggml_op_name(op->op));
            }
            if (n_fuse == 3) {
                GGML_LOG_DEBUG("%s: fuse: %s + MUL + ADD\n", __func__, ggml_op_name(op->op));
            }
        }
    }

    if (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1));

        for (int i = 1; i < n_fuse; ++i) {
            if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
                ggml_metal_op_concurrency_reset(ctx);

                break;
            }
        }
    }

    auto pipeline = ggml_metal_library_get_pipeline_norm(lib, op, n_fuse);

    int nth = 32; // SIMD width

    while (nth < args.ne00_t && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, args.ne00_t);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0,    1);
    ggml_metal_encoder_set_buffer  (enc, bid_fuse[0], 2);
    ggml_metal_encoder_set_buffer  (enc, bid_fuse[1], 3);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,     4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return n_fuse;
}

int ggml_metal_op_rope(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // make sure we have one or more position id(ne10) per token(ne02)
    GGML_ASSERT(ne10 % ne02 == 0);
    GGML_ASSERT(ne10 >= ne02);

    const int nth = std::min(1024, ne00);

    const int n_past     = ((const int32_t *) op->op_params)[0];
    const int n_dims     = ((const int32_t *) op->op_params)[1];
  //const int mode       = ((const int32_t *) op->op_params)[2];
    // skip 3, n_ctx, used in GLM RoPE, unimplemented in metal
    const int n_ctx_orig = ((const int32_t *) op->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) op->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) op->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) op->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) op->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) op->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) op->op_params + 10, sizeof(float));

    // mrope
    const int sect_0 = ((const int32_t *) op->op_params)[11];
    const int sect_1 = ((const int32_t *) op->op_params)[12];
    const int sect_2 = ((const int32_t *) op->op_params)[13];
    const int sect_3 = ((const int32_t *) op->op_params)[14];

    const int n_offs = ((const int32_t *) op->op_params)[15];

    // when dst aliases src0, the channels outside the rotated window already hold the correct data
    const bool inplace = op->data == op->src[0]->data;

    ggml_metal_kargs_rope args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.ne03        =*/ ne03,
        /*.nb00        =*/ nb00,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne0         =*/ ne0,
        /*.ne1         =*/ ne1,
        /*.ne2         =*/ ne2,
        /*.ne3         =*/ ne3,
        /*.nb0         =*/ nb0,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.n_past      =*/ n_past,
        /*.n_dims      =*/ n_dims,
        /*.n_offs      =*/ n_offs,
        /*.n_ctx_orig  =*/ n_ctx_orig,
        /*.freq_base   =*/ freq_base,
        /*.freq_scale  =*/ freq_scale,
        /*.ext_factor  =*/ ext_factor,
        /*.attn_factor =*/ attn_factor,
        /*.beta_fast   =*/ beta_fast,
        /*.beta_slow   =*/ beta_slow,
        /* sect_0      =*/ sect_0,
        /* sect_1      =*/ sect_1,
        /* sect_2      =*/ sect_2,
        /* sect_3      =*/ sect_3,
        /* src2        =*/ op->src[2] != nullptr,
        /* inplace     =*/ inplace,
    };

    auto pipeline = ggml_metal_library_get_pipeline_rope(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    if (op->src[2]) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_im2col(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t s1 = ((const int32_t *)(op->op_params))[1];
    const int32_t p0 = ((const int32_t *)(op->op_params))[2];
    const int32_t p1 = ((const int32_t *)(op->op_params))[3];
    const int32_t d0 = ((const int32_t *)(op->op_params))[4];
    const int32_t d1 = ((const int32_t *)(op->op_params))[5];

    const bool is_2D = ((const int32_t *)(op->op_params))[6] == 1;

    const int32_t N  = op->src[1]->ne[is_2D ? 3 : 2];
    const int32_t IC = op->src[1]->ne[is_2D ? 2 : 1];
    const int32_t IH = is_2D ? op->src[1]->ne[1] : 1;
    const int32_t IW =         op->src[1]->ne[0];

    const int32_t KH = is_2D ? op->src[0]->ne[1] : 1;
    const int32_t KW =         op->src[0]->ne[0];

    const int32_t OH = is_2D ? op->ne[2] : 1;
    const int32_t OW =         op->ne[1];

    const int32_t CHW = IC * KH * KW;

    const uint64_t ofs0 = op->src[1]->nb[is_2D ? 3 : 2] / 4;
    const uint64_t ofs1 = op->src[1]->nb[is_2D ? 2 : 1] / 4;

    ggml_metal_kargs_im2col args = {
        /*.ofs0 =*/ ofs0,
        /*.ofs1 =*/ ofs1,
        /*.IW   =*/ IW,
        /*.IH   =*/ IH,
        /*.CHW  =*/ CHW,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
        /*.N    =*/ N,
        /*.KH   =*/ KH,
        /*.KW   =*/ KW,
        /*.KHW  =*/ KH * KW,
    };

    auto pipeline = ggml_metal_library_get_pipeline_im2col(lib, op);

    GGML_ASSERT(KH*KW <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const uint64_t ntptg0 = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)/(KH*KW), N);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, IC, OH, OW, ntptg0, KH, KW);

    return 1;
}

int ggml_metal_op_conv_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t *) op->op_params)[0];
    const int32_t s1 = ((const int32_t *) op->op_params)[1];
    const int32_t p0 = ((const int32_t *) op->op_params)[2];
    const int32_t p1 = ((const int32_t *) op->op_params)[3];
    const int32_t d0 = ((const int32_t *) op->op_params)[4];
    const int32_t d1 = ((const int32_t *) op->op_params)[5];

    ggml_metal_kargs_conv_2d args = {
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.IW   =*/ ne10,
        /*.IH   =*/ ne11,
        /*.KW   =*/ ne00,
        /*.KH   =*/ ne01,
        /*.IC   =*/ ne02,
        /*.OC   =*/ ne03,
        /*.OW   =*/ ne0,
        /*.OH   =*/ ne1,
        /*.N    =*/ ne3,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_2d(lib, op);

    int nth = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    nth = std::min(nth, 256);
    nth = std::max(nth, 1);

    const uint64_t n_out = ggml_nelements(op);

    uint64_t tg = (n_out + nth - 1)/nth;
    tg = std::max<uint64_t>(tg, 1);
    tg = std::min<uint64_t>(tg, (uint64_t) std::numeric_limits<int>::max());

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, tg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_conv_3d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    // 1. Extract standard dimensions and byte strides
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // 2. Extract hyperparams from op_params
    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t s1 = ((const int32_t *)(op->op_params))[1];
    const int32_t s2 = ((const int32_t *)(op->op_params))[2];
    const int32_t p0 = ((const int32_t *)(op->op_params))[3];
    const int32_t p1 = ((const int32_t *)(op->op_params))[4];
    const int32_t p2 = ((const int32_t *)(op->op_params))[5];
    const int32_t d0 = ((const int32_t *)(op->op_params))[6];
    const int32_t d1 = ((const int32_t *)(op->op_params))[7];
    const int32_t d2 = ((const int32_t *)(op->op_params))[8];
    const int32_t IC = ((const int32_t *)(op->op_params))[9];
    const int32_t N  = ((const int32_t *)(op->op_params))[10];
    const int32_t OC = ((const int32_t *)(op->op_params))[11];

    // 3. Build the parameter struct using the macro-generated variables
    ggml_metal_kargs_conv_3d args = {
        /*.IW =*/ (int32_t)op->src[1]->ne[0],
        /*.IH =*/ (int32_t)op->src[1]->ne[1],
        /*.ID =*/ (int32_t)op->src[1]->ne[2],
        /*.OW =*/ (int32_t)op->ne[0],
        /*.OH =*/ (int32_t)op->ne[1],
        /*.OD =*/ (int32_t)op->ne[2],
        /*.KW =*/ (int32_t)op->src[0]->ne[0],
        /*.KH =*/ (int32_t)op->src[0]->ne[1],
        /*.KD =*/ (int32_t)op->src[0]->ne[2],
        s0, s1, s2,
        p0, p1, p2,
        d0, d1, d2,
        IC, N, OC,
        nb00, nb01, nb02, nb03, // Weight strides
        nb10, nb11, nb12, nb13, // Input strides
        nb0,  nb1,  nb2,  nb3   // Output strides
    };

    // 4. Fetch the JIT pipeline
    auto pipeline = ggml_metal_library_get_pipeline_conv_3d(lib, op);

    // 5. Grid mapping
    int nth0 = 32; // Standard SIMD width for Apple Silicon
    int nth1 = 1;
    int nth2 = 1;

    int64_t spatial_volume = args.OW * args.OH * args.OD;

    int ntg0 = (spatial_volume + nth0 - 1) / nth0;
    int ntg1 = args.OC;
    int ntg2 = args.N;

    // 6. Bind and Dispatch via the ggml C wrapper
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg0, ntg1, ntg2, nth0, nth1, nth2);

    return 1;
}

int ggml_metal_op_conv_transpose_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];

    const int32_t IC = op->src[1]->ne[1];
    const int32_t IL = op->src[1]->ne[0];

    const int32_t K  = op->src[0]->ne[0];

    const int32_t OL = op->ne[0];
    const int32_t OC = op->ne[1];

    ggml_metal_kargs_conv_transpose_1d args = {
        /*.IC  =*/ IC,
        /*.IL  =*/ IL,
        /*.K   =*/ K,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_transpose_1d(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, OL, OC, 1, 1, 1, 1);

    return 1;
}

int ggml_metal_op_conv_transpose_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];

    const int32_t IC = op->src[1]->ne[2];
    const int32_t IH = op->src[1]->ne[1];
    const int32_t IW = op->src[1]->ne[0];

    const int32_t KH = op->src[0]->ne[1];
    const int32_t KW = op->src[0]->ne[0];

    const int32_t OW = op->ne[0];
    const int32_t OH = op->ne[1];
    const int32_t OC = op->ne[2];

    ggml_metal_kargs_conv_transpose_2d args = {
        /*.IC  =*/ IC,
        /*.IH  =*/ IH,
        /*.IW  =*/ IW,
        /*.KH  =*/ KH,
        /*.KW  =*/ KW,
        /*.OC  =*/ OC,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
        /*.nb2 =*/ nb2,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_transpose_2d(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    // Metal requires buffer size to be multiple of 16 bytes
    const size_t smem = GGML_PAD(KW * KH * sizeof(float), 16);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, OW, OH, OC, KW, KH, 1);

    return 1;
}

int ggml_metal_op_upscale(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float sf0 = (float)ne0/op->src[0]->ne[0];
    float sf1 = (float)ne1/op->src[0]->ne[1];
    float sf2 = (float)ne2/op->src[0]->ne[2];
    float sf3 = (float)ne3/op->src[0]->ne[3];

    const int32_t mode_flags = ggml_get_op_params_i32(op, 0);

    float poffs = 0.5f;

    if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        poffs = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (float)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (float)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    ggml_metal_kargs_upscale args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.sf0   =*/ sf0,
        /*.sf1   =*/ sf1,
        /*.sf2   =*/ sf2,
        /*.sf3   =*/ sf3,
        /*.poffs =*/ poffs,
    };

    auto pipeline = ggml_metal_library_get_pipeline_upscale(lib, op);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_roll(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ggml_get_op_params_i32(op, 0);
    const int32_t s1 = ggml_get_op_params_i32(op, 1);
    const int32_t s2 = ggml_get_op_params_i32(op, 2);
    const int32_t s3 = ggml_get_op_params_i32(op, 3);

    ggml_metal_kargs_roll args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.s2   =*/ s2,
        /*.s3   =*/ s3
    };

    auto pipeline = ggml_metal_library_get_pipeline_roll(lib, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_pad(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_pad args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3
    };

    auto pipeline = ggml_metal_library_get_pipeline_pad(lib, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_pad_reflect_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_pad_reflect_1d args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.p0 =*/ ((const int32_t *)(op->op_params))[0],
        /*.p1 =*/ ((const int32_t *)(op->op_params))[1]
    };

    auto pipeline = ggml_metal_library_get_pipeline_pad_reflect_1d(lib, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_arange(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float start;
    float step;

    memcpy(&start, ((const int32_t *) op->op_params) + 0, sizeof(float));
    memcpy(&step,  ((const int32_t *) op->op_params) + 2, sizeof(float));

    ggml_metal_kargs_arange args = {
        /*.ne0   =*/ ne0,
        /*.start =*/ start,
        /*.step  =*/ step
    };

    const int nth = std::min(1024, ne0);

    auto pipeline = ggml_metal_library_get_pipeline_arange(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 1);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_timestep_embedding(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int dim        = op->op_params[0];
    const int max_period = op->op_params[1];

    ggml_metal_kargs_timestep_embedding args = {
        /*.nb1 =*/ nb1,
        /*.dim =*/ dim,
        /*.max_period =*/ max_period,
    };

    auto pipeline = ggml_metal_library_get_pipeline_timestep_embedding(lib, op);

    const int nth = std::max(1, std::min(1024, dim/2));

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne00, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_argmax(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_argmax args = {
        /*.ne00 = */ ne00,
        /*.nb01 = */ nb01,
    };

    auto pipeline = ggml_metal_library_get_pipeline_argmax(lib, op);

    const int64_t nrows = ggml_nrows(op->src[0]);

    int nth = 32; // SIMD width
    while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
        nth *= 2;
    }

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_dsv4_hc_comb(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * mixes = op->src[0];
    const ggml_tensor * scale = op->src[1];
    const ggml_tensor * base  = op->src[2];

    ggml_metal_kargs_dsv4_hc_comb args = {
        /*.n_tokens =*/ (int32_t) op->ne[2],
        /*.n_iter   =*/ ggml_get_op_params_i32(op, 1),
        /*.eps      =*/ ggml_get_op_params_f32(op, 0),
        /*.nbm0     =*/ mixes->nb[0],
        /*.nbm1     =*/ mixes->nb[1],
        /*.nbs0     =*/ scale->nb[0],
        /*.nbb0     =*/ base->nb[0],
        /*.nbd0     =*/ op->nb[0],
        /*.nbd1     =*/ op->nb[1],
        /*.nbd2     =*/ op->nb[2],
    };

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_hc(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mixes), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(scale), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(base),  3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),    4);

    ggml_metal_encoder_dispatch_threadgroups(enc, op->ne[2], 1, 1, 1, 1, 1);

    return 1;
}

int ggml_metal_op_dsv4_hc_pre(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * x       = op->src[0];
    const ggml_tensor * weights = op->src[1];

    ggml_metal_kargs_dsv4_hc_pre args = {
        /*.n_embd =*/ (int32_t) op->ne[0],
        /*.hc     =*/ (int32_t) x->ne[1],
        /*.gated  =*/ ggml_get_op_params_i32(op, 1),
        /*.scale  =*/ ggml_get_op_params_f32(op, 0),
        /*.nbx0   =*/ x->nb[0],
        /*.nbx1   =*/ x->nb[1],
        /*.nbx2   =*/ x->nb[2],
        /*.nbw0   =*/ weights->nb[0],
        /*.nbw1   =*/ weights->nb[1],
        /*.nbw2   =*/ weights->nb[2],
        /*.nbd0   =*/ op->nb[0],
        /*.nbd1   =*/ op->nb[1],
    };

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_hc(lib, op);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) op->ne[0]);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(x),       1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(weights), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),      3);

    ggml_metal_encoder_dispatch_threadgroups(enc, op->ne[1], 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_dsv4_hc_post(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * x        = op->src[0];
    const ggml_tensor * residual = op->src[1];
    const ggml_tensor * post     = op->src[2];
    const ggml_tensor * comb     = op->src[3];

    ggml_metal_kargs_dsv4_hc_post args = {
        /*.n_embd   =*/ (int32_t) op->ne[0],
        /*.hc       =*/ (int32_t) residual->ne[1],
        /*.has_comb =*/ comb != nullptr,
        /*.nbx0     =*/ x->nb[0],
        /*.nbx1     =*/ x->nb[1],
        /*.nbr0     =*/ residual->nb[0],
        /*.nbr1     =*/ residual->nb[1],
        /*.nbr2     =*/ residual->nb[2],
        /*.nbp0     =*/ post->nb[0],
        /*.nbp1     =*/ post->nb[1],
        /*.nbc0     =*/ comb ? comb->nb[0] : 0,
        /*.nbc1     =*/ comb ? comb->nb[1] : 0,
        /*.nbc2     =*/ comb ? comb->nb[2] : 0,
        /*.nbd0     =*/ op->nb[0],
        /*.nbd1     =*/ op->nb[1],
        /*.nbd2     =*/ op->nb[2],
    };

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_hc(lib, op);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) op->ne[0]);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(x),        1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(residual), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(post),     3);
    // an identity comb binds a source the kernel does not read
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(comb ? comb : post), 4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),       5);

    ggml_metal_encoder_dispatch_threadgroups(enc, op->ne[1], op->ne[2], 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_inkling_sconv(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * cur     = op->src[0];
    const ggml_tensor * past    = op->src[1];
    const ggml_tensor * weight  = op->src[2];
    const ggml_tensor * dynamic = op->src[3];

    const int32_t n_channels   = (int32_t) cur->ne[0];
    const int32_t n_tokens     = (int32_t) cur->ne[1];
    const int32_t n_conv       = (int32_t) weight->ne[0];
    const int32_t n_seqs       = (int32_t) past->ne[2];
    const int32_t n_seq_tokens = n_tokens / n_seqs;
    const int32_t n_heads      = dynamic ? (int32_t) (dynamic->ne[0] / n_conv) : 1;
    const int32_t head_dim     = dynamic ? (n_channels / n_heads) : n_channels;

    ggml_tensor * dst_t = op;
    ggml_tensor * res_t = nullptr;
    int n_consumed      = 1;
    if (ctx->use_fusion) {
        // Fused Inkling2 K/V prep: INKLING_SCONV (n_conv=4, n_seqs=1) -> (L2_NORM | RMS_NORM) -> SET_ROWS (F16)
        if (n_conv == 4 && n_seqs == 1 && (n_channels % 128 == 0) && (!dynamic || head_dim == 128) &&
            idx + 2 < ctx->n_nodes() && ctx->node_has_n_uses(idx, 1) && ctx->node_has_n_uses(idx + 1, 1)) {
            ggml_tensor * n_norm = ctx->node(idx + 1);
            ggml_tensor * n_sr   = ctx->node(idx + 2);
            if ((n_norm->op == GGML_OP_L2_NORM || n_norm->op == GGML_OP_RMS_NORM) &&
                n_norm->type == GGML_TYPE_F32 && ggml_is_contiguous(n_norm) &&
                n_norm->ne[0] == 128 && n_norm->ne[1] == n_channels / 128 &&
                n_norm->ne[2] == n_tokens && n_norm->ne[3] == 1 &&
                ggml_metal_unwrap_zero_view(ctx, n_norm->src[0], op) &&
                n_sr->op == GGML_OP_SET_ROWS && n_sr->type == GGML_TYPE_F16 &&
                n_sr->src[1]->type == GGML_TYPE_I64 && n_sr->src[1]->nb[0] == sizeof(int64_t) &&
                n_sr->ne[0] == n_channels && n_sr->ne[2] == 1 && n_sr->ne[3] == 1 &&
                n_sr->src[0]->ne[0] == n_channels && n_sr->src[0]->ne[1] == n_tokens && n_sr->src[0]->ne[2] == 1 &&
                n_sr->src[1]->ne[0] == n_tokens && n_sr->src[1]->ne[1] == 1 && n_sr->src[1]->ne[2] == 1 &&
                ggml_metal_unwrap_zero_view(ctx, n_sr->src[0], n_norm) &&
                !ggml_metal_tensor_overlaps_unsafe(cur,          n_sr, false) &&
                !ggml_metal_tensor_overlaps_unsafe(past,         n_sr, false) &&
                !ggml_metal_tensor_overlaps_unsafe(weight,       n_sr, false) &&
                !ggml_metal_tensor_overlaps_unsafe(dynamic,      n_sr, false) &&
                !ggml_metal_tensor_overlaps_unsafe(n_sr->src[1], n_sr, false)) {
                ggml_metal_op_concurrency_check_fused(ctx, idx, 3);
                float norm_eps = 0.0f;
                memcpy(&norm_eps, n_norm->op_params, sizeof(float));
                const uint64_t tp_lo = (uint32_t) ggml_get_op_params_i32(op, 1);
                const uint64_t tp_hi = (uint32_t) ggml_get_op_params_i32(op, 2);
                ggml_metal_kargs_inkling_sconv_norm_sr sargs = {
                    /*.n_channels   =*/ n_channels,
                    /*.n_seq_tokens =*/ n_seq_tokens,
                    /*.n_conv       =*/ n_conv,
                    /*.head_dim     =*/ 128,
                    /*.has_dynamic  =*/ dynamic != nullptr ? 1 : 0,
                    /*.norm_mode    =*/ (n_norm->op == GGML_OP_RMS_NORM) ? 1 : 0,
                    /*.scale        =*/ ggml_get_op_params_f32(op, 0),
                    /*.norm_eps     =*/ norm_eps,
                    /*.nbc1         =*/ cur->nb[1],
                    /*.nbp1         =*/ past->nb[1],
                    /*.nbw1         =*/ weight->nb[1],
                    /*.nbd1         =*/ dynamic ? dynamic->nb[1] : 0,
                    /*.nb_dst1      =*/ n_sr->nb[1],
                    /*.tree_parents =*/ tp_lo | (tp_hi << 32),
                };
                auto pl = ggml_metal_library_get_pipeline_inkling_sconv_norm_sr(lib);
                ggml_metal_buffer_id bid_cur = ggml_metal_get_buffer_id(cur);
                ggml_metal_encoder_set_pipeline(enc, pl);
                ggml_metal_encoder_set_bytes   (enc, &sargs, sizeof(sargs), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_cur, 1);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(past), 2);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(weight), 3);
                ggml_metal_encoder_set_buffer  (enc, dynamic ? ggml_metal_get_buffer_id(dynamic) : bid_cur, 4);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(n_sr->src[1]), 5);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(n_sr), 6);
                ggml_metal_encoder_dispatch_threadgroups(enc, n_channels / 128, n_seq_tokens, 1, 128, 1, 1);
                return 3;
            }
        }

        const ggml_op ops_add[2] = { GGML_OP_INKLING_SCONV, GGML_OP_ADD };
        if (ctx->can_fuse(idx, ops_add, 2)) {
            ggml_tensor * next  = ctx->node(idx + 1);
            ggml_tensor * other = (next->src[0] == op) ? next->src[1] : ((next->src[1] == op) ? next->src[0] : nullptr);
            if (other && other->type == GGML_TYPE_F32 && next->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(other, op) && ggml_are_same_shape(next, op) &&
                ggml_is_contiguous(other) && ggml_is_contiguous(next) &&
                !ggml_metal_tensor_overlaps_unsafe(cur,     next, false) &&
                !ggml_metal_tensor_overlaps_unsafe(past,    next, false) &&
                !ggml_metal_tensor_overlaps_unsafe(dynamic, next, false) &&
                !ggml_metal_tensor_overlaps_unsafe(other,   next, true)) {
                dst_t      = next;
                res_t      = other;
                n_consumed = 2;
            }
        }
    }

    if (n_consumed > 1) {
        ggml_metal_op_concurrency_check_fused(ctx, idx, n_consumed);
    }

    const uint64_t tp_lo = (uint32_t) ggml_get_op_params_i32(op, 1);
    const uint64_t tp_hi = (uint32_t) ggml_get_op_params_i32(op, 2);

    ggml_metal_kargs_inkling_sconv args = {
        /*.n_channels   =*/ n_channels,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seqs       =*/ n_seqs,
        /*.n_conv       =*/ n_conv,
        /*.head_dim     =*/ head_dim,
        /*.has_dynamic  =*/ dynamic != nullptr ? 1 : 0,
        /*.has_res      =*/ res_t != nullptr ? 1 : 0,
        /*.scale        =*/ ggml_get_op_params_f32(op, 0),
        /*.nbc1         =*/ cur->nb[1],
        /*.nbp1         =*/ past->nb[1],
        /*.nbp2         =*/ past->nb[2],
        /*.nbw1         =*/ weight->nb[1],
        /*.nbd1         =*/ dynamic ? dynamic->nb[1] : 0,
        /*.tree_parents =*/ tp_lo | (tp_hi << 32),
    };

    ggml_metal_buffer_id bid_cur = ggml_metal_get_buffer_id(cur);
    ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst_t);

    auto pipeline = ggml_metal_library_get_pipeline_inkling_sconv(lib, op);

    const int nth = std::min(256, std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) ((n_channels + 31) / 32 * 32)));
    const int n_blocks = (n_channels + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_cur,                                     1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(past),              2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(weight),            3);
    ggml_metal_encoder_set_buffer  (enc, dynamic ? ggml_metal_get_buffer_id(dynamic) : bid_cur, 4);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,                                     5);
    ggml_metal_encoder_set_buffer  (enc, res_t   ? ggml_metal_get_buffer_id(res_t)   : bid_dst, 6);

    ggml_metal_encoder_dispatch_threadgroups(enc, n_blocks, n_seq_tokens, n_seqs, nth, 1, 1);

    return n_consumed;
}

int ggml_metal_op_inkling_sconv_state(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * past = op->src[0];
    const ggml_tensor * cur0 = op->src[1];
    const ggml_tensor * cur1 = op->src[3];
    const ggml_tensor * cur2 = op->src[4];
    const ggml_tensor * cur3 = op->src[5];
    const ggml_tensor * hist = op->src[6];
    const ggml_tensor * pos  = op->src[7];

    const bool has_hist = hist != nullptr && pos != nullptr;

    const int32_t l_cache      = (int32_t) past->ne[0];
    const int32_t n_channels   = (int32_t) past->ne[1];
    const int32_t n_seqs       = (int32_t) past->ne[2];
    const int32_t n_tokens     = (int32_t) cur0->ne[1];
    const int32_t n_seq_tokens = n_tokens / n_seqs;

    ggml_metal_kargs_inkling_sconv_state args = {
        /*.l_cache      =*/ l_cache,
        /*.n_channels   =*/ n_channels,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seqs       =*/ n_seqs,
        /*.c0           =*/ (int32_t) cur0->ne[0],
        /*.c1           =*/ cur1 ? (int32_t) cur1->ne[0] : 0,
        /*.c2           =*/ cur2 ? (int32_t) cur2->ne[0] : 0,
        /*.c3           =*/ cur3 ? (int32_t) cur3->ne[0] : 0,
        /*.hist_w       =*/ has_hist ? (int32_t) hist->ne[1] : 0,
        /*.has_hist     =*/ has_hist ? 1 : 0,
        /*.is_tree      =*/ ggml_get_op_params_i32(op, 0) != 0 ? 1 : 0,
        /*.nbc01        =*/ cur0->nb[1],
        /*.nbc11        =*/ cur1 ? cur1->nb[1] : 0,
        /*.nbc21        =*/ cur2 ? cur2->nb[1] : 0,
        /*.nbc31        =*/ cur3 ? cur3->nb[1] : 0,
        /*.nbp1         =*/ past->nb[1],
        /*.nbp2         =*/ past->nb[2],
        /*.nbd1         =*/ op->nb[1],
        /*.nbh1         =*/ has_hist ? hist->nb[1] : 0,
        /*.nbh2         =*/ has_hist ? hist->nb[2] : 0,
    };

    auto pipeline = ggml_metal_library_get_pipeline_inkling_sconv(lib, op);

    const int total = n_channels;
    const int nth = std::min(256, std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (total + 31) / 32 * 32));
    const int n_blocks = (total + nth - 1) / nth;

    ggml_metal_buffer_id bid_cur0 = ggml_metal_get_buffer_id(cur0);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(past), 1);
    ggml_metal_encoder_set_buffer  (enc, bid_cur0,                       2);
    ggml_metal_encoder_set_buffer  (enc, cur1 ? ggml_metal_get_buffer_id(cur1) : bid_cur0, 3);
    ggml_metal_encoder_set_buffer  (enc, cur2 ? ggml_metal_get_buffer_id(cur2) : bid_cur0, 4);
    ggml_metal_encoder_set_buffer  (enc, cur3 ? ggml_metal_get_buffer_id(cur3) : bid_cur0, 5);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,                        6);
    ggml_metal_encoder_set_buffer  (enc, has_hist ? ggml_metal_get_buffer_id(hist) : bid_dst,  7);
    ggml_metal_encoder_set_buffer  (enc, has_hist ? ggml_metal_get_buffer_id(pos)  : bid_cur0, 8);

    ggml_metal_encoder_dispatch_threadgroups(enc, n_blocks, n_seqs, 1, nth, 1, 1);

    return 1;
}

static int ggml_metal_op_inkling_mix_impl(ggml_metal_op_t ctx, int idx_mix, ggml_tensor * pre_rms_op) {
    ggml_tensor * op = ctx->node(idx_mix);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * cur_in   = pre_rms_op ? pre_rms_op->src[0] : op->src[0];
    const ggml_tensor * query    = op->src[1];
    const ggml_tensor * bank     = op->src[2];

    const int32_t n_embd   = (int32_t) cur_in->ne[0];
    const int32_t n_tokens = (int32_t) cur_in->ne[1];
    const int32_t n_query  = (int32_t) query->ne[1];
    const int32_t k_bank   = bank ? (int32_t) bank->ne[1] : 0;

    const int idx_start    = pre_rms_op ? (idx_mix - 1) : idx_mix;
    const int base_consumed = pre_rms_op ? 2 : 1;

    ggml_tensor * dst_t  = op;
    ggml_tensor * norm_w = nullptr;
    int32_t has_rms  = 0;
    int32_t has_mul  = 0;
    float   eps      = 0.0f;
    float   pre_eps  = 0.0f;
    uint64_t nbw2    = 0;
    int n_consumed   = base_consumed;

    if (pre_rms_op) {
        memcpy(&pre_eps, pre_rms_op->op_params, sizeof(float));
    }

    if (ctx->use_fusion) {
        const ggml_op ops_rms[2]     = { GGML_OP_INKLING_MIX, GGML_OP_RMS_NORM };
        const ggml_op ops_rms_mul[3] = { GGML_OP_INKLING_MIX, GGML_OP_RMS_NORM, GGML_OP_MUL };
        if (ctx->can_fuse(idx_mix, ops_rms_mul, 3)) {
            ggml_tensor * n1 = ctx->node(idx_mix + 1);
            ggml_tensor * n2 = ctx->node(idx_mix + 2);
            ggml_tensor * w  = (n2->src[0] == n1) ? n2->src[1] : n2->src[0];
            if (n1->src[0] == op &&
                w && w->type == GGML_TYPE_F32 &&
                w->ne[0] == n_embd && w->ne[1] == 1 && (w->ne[2] == 1 || w->ne[2] == n_query) && w->ne[3] == 1 &&
                ggml_is_contiguous(w) && ggml_is_contiguous(n2) &&
                !ggml_metal_tensor_overlaps_unsafe(cur_in, n2, n_query == 1) &&
                !ggml_metal_tensor_overlaps_unsafe(query,  n2, false) &&
                !ggml_metal_tensor_overlaps_unsafe(bank,   n2, false) &&
                !ggml_metal_tensor_overlaps_unsafe(w,      n2, false)) {
                memcpy(&eps, n1->op_params, sizeof(float));
                has_rms    = 1;
                has_mul    = 1;
                norm_w     = w;
                nbw2       = (w->ne[2] == n_query && n_query > 1) ? w->nb[2] : 0;
                dst_t      = n2;
                n_consumed = base_consumed + 2;
            }
        }
        if (has_rms == 0 && ctx->can_fuse(idx_mix, ops_rms, 2)) {
            ggml_tensor * n1 = ctx->node(idx_mix + 1);
            if (n1->src[0] == op && ggml_is_contiguous(n1) &&
                !ggml_metal_tensor_overlaps_unsafe(cur_in, n1, n_query == 1) &&
                !ggml_metal_tensor_overlaps_unsafe(query,  n1, false) &&
                !ggml_metal_tensor_overlaps_unsafe(bank,   n1, false)) {
                memcpy(&eps, n1->op_params, sizeof(float));
                has_rms    = 1;
                dst_t      = n1;
                n_consumed = base_consumed + 1;
            }
        }
    }

    if (pre_rms_op && ggml_metal_tensor_overlaps_unsafe(cur_in, dst_t, n_query == 1)) {
        return 0;
    }

    if (n_consumed > 1) {
        ggml_metal_op_concurrency_check_fused(ctx, idx_start, n_consumed);
    }

    ggml_metal_kargs_inkling_mix args = {
        /*.n_embd   =*/ n_embd,
        /*.n_tokens =*/ n_tokens,
        /*.n_query  =*/ n_query,
        /*.k_bank   =*/ k_bank,
        /*.scale    =*/ 1.0f / sqrtf((float) n_embd),
        /*.has_rms  =*/ has_rms,
        /*.has_mul  =*/ has_mul,
        /*.eps      =*/ eps,
        /*.pre_rms  =*/ pre_rms_op ? 1 : 0,
        /*.pre_eps  =*/ pre_eps,
        /*.nbc1     =*/ cur_in->nb[1],
        /*.nbq1     =*/ query->nb[1],
        /*.nbb1     =*/ bank ? bank->nb[1] : 0,
        /*.nbb2     =*/ bank ? bank->nb[2] : 0,
        /*.nbw2     =*/ nbw2,
    };

    auto pipeline = ggml_metal_library_get_pipeline_inkling_sconv(lib, op);
    ggml_metal_buffer_id bid_cur = ggml_metal_get_buffer_id(cur_in);
    ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst_t);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_cur,                                  1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(query),          2);
    ggml_metal_encoder_set_buffer  (enc, bank ? ggml_metal_get_buffer_id(bank) : bid_cur, 3);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,                                  4);
    ggml_metal_encoder_set_buffer  (enc, norm_w ? ggml_metal_get_buffer_id(norm_w) : bid_cur, 5);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, n_tokens, n_query, 1, 256, 1, 1);

    return n_consumed;
}

int ggml_metal_op_inkling_mix(ggml_metal_op_t ctx, int idx) {
    return ggml_metal_op_inkling_mix_impl(ctx, idx, nullptr);
}

int ggml_metal_op_inkling_ctrl_blend(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * cur    = op->src[0];
    const ggml_tensor * msg    = op->src[1];
    const ggml_tensor * gf_raw = op->src[2];

    const int32_t n_embd   = (int32_t) msg->ne[0];
    const int32_t n_tokens = (int32_t) msg->ne[1];
    const int32_t n_ctrl   = (int32_t) msg->ne[2];

    ggml_tensor * dst_t  = op;
    ggml_tensor * norm_w = nullptr;
    int32_t  has_rms = 0;
    int32_t  has_mul = 0;
    float    eps     = 0.0f;
    uint64_t nbw2    = 0;
    int n_consumed   = 1;

    if (ctx->use_fusion && n_embd % 4 == 0 && n_embd <= 4096) {
        const ggml_op ops_rms[2]     = { GGML_OP_INKLING_CTRL_BLEND, GGML_OP_RMS_NORM };
        const ggml_op ops_rms_mul[3] = { GGML_OP_INKLING_CTRL_BLEND, GGML_OP_RMS_NORM, GGML_OP_MUL };
        if (ctx->can_fuse(idx, ops_rms_mul, 3)) {
            ggml_tensor * n1 = ctx->node(idx + 1);
            ggml_tensor * n2 = ctx->node(idx + 2);
            ggml_tensor * w  = (n2->src[0] == n1) ? n2->src[1] : n2->src[0];
            if (n1->src[0] == op &&
                w && w->type == GGML_TYPE_F32 &&
                w->ne[0] == n_embd && w->ne[1] == 1 && (w->ne[2] == 1 || w->ne[2] == n_ctrl) && w->ne[3] == 1 &&
                ggml_is_contiguous(w) && ggml_is_contiguous(n2) &&
                !ggml_metal_tensor_overlaps_unsafe(cur,    n2, cur->ne[2] == n_ctrl) &&
                !ggml_metal_tensor_overlaps_unsafe(msg,    n2, true) &&
                !ggml_metal_tensor_overlaps_unsafe(gf_raw, n2, false) &&
                !ggml_metal_tensor_overlaps_unsafe(w,      n2, false)) {
                memcpy(&eps, n1->op_params, sizeof(float));
                has_rms    = 1;
                has_mul    = 1;
                norm_w     = w;
                nbw2       = (w->ne[2] == n_ctrl && n_ctrl > 1) ? w->nb[2] : 0;
                dst_t      = n2;
                n_consumed = 3;
            }
        }
        if (n_consumed == 1 && ctx->can_fuse(idx, ops_rms, 2)) {
            ggml_tensor * n1 = ctx->node(idx + 1);
            if (n1->src[0] == op && ggml_is_contiguous(n1) &&
                !ggml_metal_tensor_overlaps_unsafe(cur,    n1, cur->ne[2] == n_ctrl) &&
                !ggml_metal_tensor_overlaps_unsafe(msg,    n1, true) &&
                !ggml_metal_tensor_overlaps_unsafe(gf_raw, n1, false)) {
                memcpy(&eps, n1->op_params, sizeof(float));
                has_rms    = 1;
                dst_t      = n1;
                n_consumed = 2;
            }
        }
    }

    if (n_consumed > 1) {
        ggml_metal_op_concurrency_check_fused(ctx, idx, n_consumed);
    }

    ggml_metal_kargs_inkling_ctrl_blend args = {
        /*.n_embd    =*/ n_embd,
        /*.n_tokens  =*/ n_tokens,
        /*.n_ctrl    =*/ n_ctrl,
        /*.bcast_cur =*/ cur->ne[2] == 1 ? 1 : 0,
        /*.has_rms   =*/ has_rms,
        /*.has_mul   =*/ has_mul,
        /*.eps       =*/ eps,
        /*.nbc1      =*/ cur->nb[1],
        /*.nbc2      =*/ cur->nb[2],
        /*.nbm1      =*/ msg->nb[1],
        /*.nbm2      =*/ msg->nb[2],
        /*.nbg1      =*/ gf_raw->nb[1],
        /*.nbg2      =*/ gf_raw->nb[2],
        /*.nbw2      =*/ nbw2,
    };

    auto pipeline = ggml_metal_library_get_pipeline_inkling_sconv(lib, op);
    ggml_metal_buffer_id bid_cur = ggml_metal_get_buffer_id(cur);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_cur,                                              1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(msg),                        2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(gf_raw),                     3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst_t),                      4);
    ggml_metal_encoder_set_buffer  (enc, norm_w ? ggml_metal_get_buffer_id(norm_w) : bid_cur,  5);

    if (has_rms) {
        ggml_metal_encoder_dispatch_threadgroups(enc, 1, n_tokens, n_ctrl, 256, 1, 1);
    } else {
        const int work_x   = (n_embd % 4 == 0) ? (n_embd / 4) : n_embd;
        const int nth      = std::min(256, std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (work_x + 31) / 32 * 32));
        const int n_blocks = (work_x + nth - 1) / nth;
        ggml_metal_encoder_dispatch_threadgroups(enc, n_blocks, n_tokens, n_ctrl, nth, 1, 1);
    }

    return n_consumed;
}

int ggml_metal_op_inkling_moe_ids(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * logits = op->src[0];
    const ggml_tensor * bias   = op->src[1];
    const ggml_tensor * down_s = nullptr;
    ggml_tensor       * w_node = nullptr;

    int   n_fuse  = 1;
    float w_scale = 1.0f;

    if (ctx->use_fusion && idx + 1 < ctx->n_nodes()) {
        ggml_tensor * next = ctx->node(idx + 1);
        if (next->op == GGML_OP_INKLING_MOE_WEIGHTS && next->src[0] == logits && next->src[1] == op) {
            w_node  = next;
            down_s  = next->src[2];
            w_scale = ggml_get_op_params_f32(next, 0);
            n_fuse  = 2;
        }
    }

    const int32_t n_expert = (int32_t) logits->ne[0];
    const int32_t n_tokens = (int32_t) logits->ne[1];
    const int32_t k_used   = (int32_t) op->ne[0];

    ggml_metal_kargs_inkling_moe_route args = {
        /*.n_expert      =*/ n_expert,
        /*.n_tokens      =*/ n_tokens,
        /*.k_used        =*/ k_used,
        /*.has_bias      =*/ bias   != nullptr ? 1 : 0,
        /*.has_down_s    =*/ down_s != nullptr ? 1 : 0,
        /*.write_ids     =*/ 1,
        /*.write_weights =*/ w_node != nullptr ? 1 : 0,
        /*.w_scale       =*/ w_scale,
        /*.nbl1          =*/ logits->nb[1],
    };

    auto pipeline = ggml_metal_library_get_pipeline_inkling_sconv(lib, op);
    const bool use_1sg = (n_expert <= 256 && k_used <= 32);
    const int nth = use_1sg ? 32 : std::max(32, (n_expert + 31) / 32 * 32);

    ggml_metal_buffer_id bid_logits = ggml_metal_get_buffer_id(logits);
    ggml_metal_buffer_id bid_ids    = ggml_metal_get_buffer_id(op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_logits,                                         1);
    ggml_metal_encoder_set_buffer  (enc, bias   ? ggml_metal_get_buffer_id(bias)   : bid_logits, 2);
    ggml_metal_encoder_set_buffer  (enc, down_s ? ggml_metal_get_buffer_id(down_s) : bid_logits, 3);
    ggml_metal_encoder_set_buffer  (enc, bid_ids,                                            4);
    ggml_metal_encoder_set_buffer  (enc, w_node ? ggml_metal_get_buffer_id(w_node) : bid_ids,    5);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, use_1sg ? 0 : pipeline.smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, n_tokens, 1, 1, nth, 1, 1);

    return n_fuse;
}

int ggml_metal_op_inkling_moe_weights(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * logits = op->src[0];
    const ggml_tensor * ids    = op->src[1];
    const ggml_tensor * down_s = op->src[2];

    const int32_t n_expert = (int32_t) logits->ne[0];
    const int32_t n_tokens = (int32_t) logits->ne[1];
    const int32_t k_used   = (int32_t) ids->ne[0];

    ggml_metal_kargs_inkling_moe_route args = {
        /*.n_expert      =*/ n_expert,
        /*.n_tokens      =*/ n_tokens,
        /*.k_used        =*/ k_used,
        /*.has_bias      =*/ 0,
        /*.has_down_s    =*/ down_s != nullptr ? 1 : 0,
        /*.write_ids     =*/ 0,
        /*.write_weights =*/ 1,
        /*.w_scale       =*/ ggml_get_op_params_f32(op, 0),
        /*.nbl1          =*/ logits->nb[1],
    };

    auto pipeline = ggml_metal_library_get_pipeline_inkling_sconv(lib, op);
    ggml_metal_buffer_id bid_logits = ggml_metal_get_buffer_id(logits);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_logits,                                         1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(ids),                      2);
    ggml_metal_encoder_set_buffer  (enc, down_s ? ggml_metal_get_buffer_id(down_s) : bid_logits, 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),                       4);

    ggml_metal_encoder_dispatch_threadgroups(enc, n_tokens, 1, 1, 32, 1, 1);

    return 1;
}

int ggml_metal_op_inkling_moe_reduce(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * experts = op->src[0];
    const ggml_tensor * weights = op->src[1];

    const int32_t n_embd   = (int32_t) experts->ne[0];
    const int32_t k_used   = (int32_t) experts->ne[1];
    const int32_t n_tokens = (int32_t) experts->ne[2];

    // Fused Inkling2 MoE reduce + post-FFN shortconv (n_conv=4, n_tokens<=16) + residual add
    if (ctx->use_fusion && n_tokens >= 1 && n_tokens <= 16 && idx + 2 < ctx->n_nodes() && ctx->node_has_n_uses(idx + 1, 1)) {
        ggml_tensor * n_sc  = ctx->node(idx + 1);
        ggml_tensor * n_add = ctx->node(idx + 2);
        if (n_sc->op == GGML_OP_INKLING_SCONV && n_sc->src[0] == op &&
            n_sc->src[2]->ne[0] == 4 && n_sc->src[3] == nullptr && n_sc->src[1]->ne[2] == 1 &&
            n_add->op == GGML_OP_ADD) {
            ggml_tensor * other = (n_add->src[0] == n_sc) ? n_add->src[1] : ((n_add->src[1] == n_sc) ? n_add->src[0] : nullptr);
            const ggml_tensor * past   = n_sc->src[1];
            const ggml_tensor * conv_w = n_sc->src[2];
            if (other && other->type == GGML_TYPE_F32 && n_add->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(other, op) && ggml_are_same_shape(n_add, op) &&
                ggml_is_contiguous(op) && ggml_is_contiguous(other) && ggml_is_contiguous(n_add) &&
                !ggml_metal_tensor_overlaps_unsafe(experts, op,    false) &&
                !ggml_metal_tensor_overlaps_unsafe(weights, op,    false) &&
                !ggml_metal_tensor_overlaps_unsafe(past,    op,    false) &&
                !ggml_metal_tensor_overlaps_unsafe(conv_w,  op,    false) &&
                !ggml_metal_tensor_overlaps_unsafe(other,   op,    false) &&
                !ggml_metal_tensor_overlaps_unsafe(experts, n_add, false) &&
                !ggml_metal_tensor_overlaps_unsafe(weights, n_add, false) &&
                !ggml_metal_tensor_overlaps_unsafe(past,    n_add, false) &&
                !ggml_metal_tensor_overlaps_unsafe(conv_w,  n_add, false) &&
                !ggml_metal_tensor_overlaps_unsafe(other,   n_add, true) &&
                !ggml_metal_tensor_overlaps_unsafe(op,      n_add, false)) {
                ggml_metal_op_concurrency_check_fused(ctx, idx, 3);
                const uint64_t tp_lo = (uint32_t) ggml_get_op_params_i32(n_sc, 1);
                const uint64_t tp_hi = (uint32_t) ggml_get_op_params_i32(n_sc, 2);
                ggml_metal_kargs_inkling_moe_reduce_sconv rargs = {
                    /*.n_embd       =*/ n_embd,
                    /*.k_used       =*/ k_used,
                    /*.n_tokens     =*/ n_tokens,
                    /*.has_res      =*/ 1,
                    /*.scale        =*/ ggml_get_op_params_f32(n_sc, 0),
                    /*._pad         =*/ 0,
                    /*.nbe1         =*/ experts->nb[1],
                    /*.nbe2         =*/ experts->nb[2],
                    /*.nbp1         =*/ past->nb[1],
                    /*.nbw1         =*/ conv_w->nb[1],
                    /*.tree_parents =*/ tp_lo | (tp_hi << 32),
                };
                auto pl = ggml_metal_library_get_pipeline_inkling_moe_reduce_sconv(lib);
                ggml_metal_encoder_set_pipeline(enc, pl);
                ggml_metal_encoder_set_bytes   (enc, &rargs, sizeof(rargs), 0);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(experts), 1);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(weights), 2);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(past),    3);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(conv_w),  4);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(other),   5);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),      6);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(n_add),   7);
                ggml_metal_encoder_dispatch_threadgroups(enc, (n_embd + 31) / 32, 1, 1, 32, n_tokens, 1);
                return 3;
            }
        }
    }

    ggml_metal_kargs_inkling_moe_reduce args = {
        /*.n_embd   =*/ n_embd,
        /*.k_used   =*/ k_used,
        /*.n_tokens =*/ n_tokens,
        /*.nbe1     =*/ experts->nb[1],
        /*.nbe2     =*/ experts->nb[2],
    };

    auto pipeline = ggml_metal_library_get_pipeline_inkling_sconv(lib, op);
    const int work_x   = (n_embd % 4 == 0) ? (n_embd / 4) : n_embd;
    const int nth      = std::min(256, std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (work_x + 31) / 32 * 32));
    const int n_blocks = (work_x + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(experts), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(weights), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),      3);

    ggml_metal_encoder_dispatch_threadgroups(enc, n_blocks, n_tokens, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_argsort(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_argsort(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    const int npr = (ne00 + nth - 1)/nth;

    // Metal kernels require the buffer size to be multiple of 16 bytes
    // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength
    const size_t smem = GGML_PAD(nth*sizeof(int32_t), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += ggml_nbytes(op);

    if ((int) ceil(std::log(npr) / std::log(2)) % 2 == 1) {
        std::swap(bid_dst, bid_tmp);
    }

    ggml_metal_kargs_argsort args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.top_k =*/ nth,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, npr*ne01, ne02, ne03, nth, 1, 1);

    auto pipeline_merge = ggml_metal_library_get_pipeline_argsort_merge(lib, op);

    int len = nth;

    while (len < ne00) {
        ggml_metal_op_concurrency_reset(ctx);

        ggml_metal_kargs_argsort_merge args_merge = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.ne03  =*/ ne03,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.ne2   =*/ ne2,
            /*.ne3   =*/ ne3,
            /*.top_k =*/ ne00,
            /*.len   =*/ len,
        };

        // merges per row
        const int nm = (ne00 + 2*len - 1) / (2*len);

        const int nth = std::min(512, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_merge));

        ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        ggml_metal_encoder_set_bytes   (enc, &args_merge, sizeof(args_merge), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  3);

        ggml_metal_encoder_dispatch_threadgroups(enc, nm*ne01, ne02, ne03, nth, 1, 1);

        std::swap(bid_dst, bid_tmp);

        len <<= 1;
    }

    return 1;
}

int ggml_metal_op_top_k(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_top_k(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    // blocks per row
    const int npr = (ne00 + nth - 1)/nth;

    const size_t smem = GGML_PAD(nth*sizeof(int32_t), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += sizeof(int32_t)*ggml_nelements(op->src[0]);

    if ((int) ceil(std::log(npr) / std::log(2)) % 2 == 1) {
        std::swap(bid_dst, bid_tmp);
    }

    const int top_k = ne0;

    ggml_metal_kargs_argsort args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.top_k =*/ std::min(nth, top_k), // for each block, keep just the top_k indices
    };

    if (npr > 1) {
        args.ne0 = (npr - 1)*args.top_k + std::min(ne00 - (npr - 1)*nth, args.top_k);
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, npr*ne01, ne02, ne03, nth, 1, 1);

    auto pipeline_merge = ggml_metal_library_get_pipeline_top_k_merge(lib, op);

    int len = args.top_k;

    while (len < args.ne0) {
        ggml_metal_op_concurrency_reset(ctx);

        // merges per row
        const int nm = (args.ne0 + 2*len - 1) / (2*len);

        const int nth = std::min(512, std::min(len, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_merge)));

        ggml_metal_kargs_argsort_merge args_merge = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.ne03  =*/ ne03,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne0   =*/ args.ne0,
            /*.ne1   =*/ ne1,
            /*.ne2   =*/ ne2,
            /*.ne3   =*/ ne3,
            /*.top_k =*/ nm == 1 ? top_k : args.ne0, // the final merge outputs top_k elements
            /*.len   =*/ len,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        ggml_metal_encoder_set_bytes   (enc, &args_merge, sizeof(args_merge), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  3);

        ggml_metal_encoder_dispatch_threadgroups(enc, nm*ne01, ne02, ne03, nth, 1, 1);

        std::swap(bid_dst, bid_tmp);

        len <<= 1;
    }

    return 1;
}

int ggml_metal_op_tri(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_tri args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_tri(lib, op);

    int nth = 32; // SIMD width

    while (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, ne00);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_opt_step_adamw(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_opt_step_adamw(lib, op);

    const int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_step_adamw args = {
        /*.np =*/ np,
    };

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    const int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_opt_step_sgd(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_opt_step_sgd(lib, op);

    const int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_step_sgd args = {
        /*.np =*/ np,
    };

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    const int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_count_equal(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t,  ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);

    {
        ggml_metal_kargs_memset args = { /*.val =*/ 0 };

        auto pipeline = ggml_metal_library_get_pipeline_memset(lib, op);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 1);

        ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 1, 1, 1);
    }

    ggml_metal_op_concurrency_reset(ctx);

    {
        ggml_metal_kargs_count_equal args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
        };

        auto pipeline = ggml_metal_library_get_pipeline_count_equal(lib, op);

        const size_t smem = pipeline.smem;

        const int nth = 32*pipeline.nsg;

        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    return 1;
}
