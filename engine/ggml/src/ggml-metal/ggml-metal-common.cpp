#include "ggml-metal-common.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <mutex>
#include <unordered_map>
#include <vector>

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
struct ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

struct ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    int debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(int debug) {
    auto * res = new ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    return res;
}

void ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    delete mrs;
}

void ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

static bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    return true;
}

static ggml_mem_range ggml_mem_range_from_tensor(const ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    if (tensor->buffer) {
        // when the tensor is allocated, use the actual memory address range in the buffer
        //
        // take the actual allocated size with ggml_backend_buft_get_alloc_size()
        // this can be larger than the tensor size if the buffer type allocates extra memory
        // ref: https://github.com/ggml-org/llama.cpp/pull/15966
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } else {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    return mr;
}

static ggml_mem_range ggml_mem_range_from_tensor_src(const ggml_tensor * tensor) {
    return ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

static ggml_mem_range ggml_mem_range_from_tensor_dst(const ggml_tensor * tensor) {
    return ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

static bool ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    return ggml_mem_ranges_add(mrs, mr);
}

static bool ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    return ggml_mem_ranges_add(mrs, mr);
}

bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }
    if (tensor->op == GGML_OP_INKLING_SCONV_STATE && tensor->src[6]) {
        ggml_mem_ranges_add_dst(mrs, tensor->src[6]);
    }

    return ggml_mem_ranges_add_dst(mrs, tensor);
}

static bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    for (size_t i = 0; i < mrs->ranges.size(); i++) {
        const auto & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        if (mr.pb != cmp.pb) {
            continue;
        }

        // intersecting source ranges are allowed
        if (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            continue;
        }

        if (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            if (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            return false;
        }
    }

    return true;
}

static bool ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    const bool res = ggml_mem_ranges_check(mrs, mr);

    return res;
}

static bool ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    const bool res = ggml_mem_ranges_check(mrs, mr);

    return res;
}

bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            if (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                return false;
            }
        }
    }
    if (tensor->op == GGML_OP_INKLING_SCONV_STATE && tensor->src[6]) {
        if (!ggml_mem_ranges_check_dst(mrs, tensor->src[6])) {
            return false;
        }
    }

    return ggml_mem_ranges_check_dst(mrs, tensor);
}

struct node_info {
    ggml_tensor * node;

    std::vector<ggml_tensor *> fused;

    ggml_op op() const {
        return node->op;
    }

    const ggml_tensor * dst() const {
        return fused.empty() ? node : fused.back();
    }

    bool is_empty() const {
        return ggml_op_is_empty(node->op);
    }

    void add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }
};

static std::vector<int> ggml_metal_graph_optimize_reorder(const std::vector<node_info> & nodes) {
    // helper to add node src and dst ranges
    const auto & h_add = [](ggml_mem_ranges_t mrs, const node_info & node) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (node.node->src[i]) {
                if (!ggml_mem_ranges_add_src(mrs, node.node->src[i])) {
                    return false;
                }
            }
        }

        // keep track of the sources and destinations of node and all fused nodes
        if (!node.fused.empty() && !ggml_op_is_empty(node.node->op)) {
            if (!ggml_mem_ranges_add_dst(mrs, node.node)) {
                return false;
            }
        }
        if (node.node->op == GGML_OP_INKLING_SCONV_STATE && node.node->src[6]) {
            if (!ggml_mem_ranges_add_dst(mrs, node.node->src[6])) {
                return false;
            }
        }
        for (const auto * fused : node.fused) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (fused->src[i]) {
                    if (!ggml_mem_ranges_add_src(mrs, fused->src[i])) {
                        return false;
                    }
                }
            }
            if (fused != node.dst() && !ggml_op_is_empty(fused->op)) {
                if (!ggml_mem_ranges_add_dst(mrs, fused)) {
                    return false;
                }
            }
        }

        return ggml_mem_ranges_add_dst(mrs, node.dst());
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    const auto & h_check = [](ggml_mem_ranges_t mrs, const node_info & node) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (node.node->src[i]) {
                if (!ggml_mem_ranges_check_src(mrs, node.node->src[i])) {
                    return false;
                }
            }
        }

        if (!node.fused.empty() && !ggml_op_is_empty(node.node->op)) {
            if (!ggml_mem_ranges_check_dst(mrs, node.node)) {
                return false;
            }
        }
        if (node.node->op == GGML_OP_INKLING_SCONV_STATE && node.node->src[6]) {
            if (!ggml_mem_ranges_check_dst(mrs, node.node->src[6])) {
                return false;
            }
        }
        for (const auto * fused : node.fused) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (fused->src[i]) {
                    if (!ggml_mem_ranges_check_src(mrs, fused->src[i])) {
                        return false;
                    }
                }
            }
            if (fused != node.dst() && !ggml_op_is_empty(fused->op)) {
                if (!ggml_mem_ranges_check_dst(mrs, fused)) {
                    return false;
                }
            }
        }

        return ggml_mem_ranges_check_dst(mrs, node.dst());
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    const auto & h_safe = [](ggml_op op) {
        switch (op) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID:
            case GGML_OP_ROPE:
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM:
            case GGML_OP_SUM_ROWS:
            case GGML_OP_SSM_CONV:
            case GGML_OP_SSM_SCAN:
            case GGML_OP_INKLING_SCONV:
            case GGML_OP_INKLING_SCONV_STATE:
            case GGML_OP_INKLING_MIX:
            case GGML_OP_INKLING_CTRL_BLEND:
            case GGML_OP_INKLING_MOE_IDS:
            case GGML_OP_INKLING_MOE_WEIGHTS:
            case GGML_OP_INKLING_MOE_REDUCE:
            case GGML_OP_CLAMP:
            case GGML_OP_TRI:
            case GGML_OP_DIAG:
            case GGML_OP_MUL:
            case GGML_OP_ADD:
            case GGML_OP_SUB:
            case GGML_OP_DIV:
            case GGML_OP_GLU:
            case GGML_OP_SCALE:
            case GGML_OP_UNARY:
            case GGML_OP_GET_ROWS:
            case GGML_OP_SET_ROWS:
            case GGML_OP_SET:
            case GGML_OP_CPY:
            case GGML_OP_CONT:
            case GGML_OP_REPEAT:
                return true;
            default:
                return ggml_op_is_empty(op);
        }
    };

    const int n = nodes.size();

    std::vector<int> res;
    res.reserve(n);

    std::vector<bool> used(n, false);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    for (int i0 = 0; i0 < n; i0++) {
        if (used[i0]) {
            continue;
        }

        const auto & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        if (!node0.is_empty() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr int N_FORWARD = 64;

            for (int i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                if (used[i1]) {
                    continue;
                }

                const auto & node1 = nodes[i1];

                // disallow reordering of certain ops
                if (!h_safe(node1.op())) {
                    break;
                }

                const bool is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                if ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = true;
                } else {
                    // expand the set of nodes that haven't been processed yet
                    h_add(mrs1, node1);
                }
            }

            // finalize the concurrent set and begin a new one
            ggml_mem_ranges_reset(mrs0);
        }

        // expand the concurrent set with the current node
        {
            h_add(mrs0, node0);
            res.push_back(i0);
        }
    }

    ggml_mem_ranges_free(mrs0);
    ggml_mem_ranges_free(mrs1);

    return res;
}

static void ggml_graph_optimize_impl(ggml_cgraph * gf);

// the order depends only on which tensors the nodes read and write: on the tensors themselves, and on the addresses of
// the tensors that already have memory (weights, caches). A graph rebuilt with the same tensors gets the same order, so
// the order is kept for each such graph and reused. A decode step whose batch size differs from the previous one
// rebuilds its graph, and the reorder pass would otherwise cost about as much CPU time as the rest of the rebuild.
static void ggml_graph_optimize_key(const ggml_cgraph * gf, std::vector<uint64_t> & key) {
    key.clear();
    key.reserve((size_t) gf->n_nodes * (2 + 3*(GGML_MAX_SRC + 1)));

    const auto add_tensor = [&](const ggml_tensor * t) {
        const ggml_tensor * base = t->view_src ? t->view_src : t;
        key.push_back((uint64_t) t);
        key.push_back((uint64_t) base);
        key.push_back(base->buffer ? (uint64_t) base->data ^ ((uint64_t) ggml_nbytes(base) << 1) : 0);
    };

    for (int i = 0; i < gf->n_nodes; i++) {
        const ggml_tensor * node = gf->nodes[i];
        key.push_back((uint64_t) node->op | ((uint64_t) node->flags << 32));
        add_tensor(node);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j]) {
                add_tensor(node->src[j]);
            } else {
                key.push_back(0);
            }
        }
    }
}

void ggml_graph_optimize(ggml_cgraph * gf) {
    struct entry {
        std::vector<uint64_t> key;
        std::vector<int>      order;
    };
    static std::mutex                                mutex;
    static std::unordered_map<uint64_t, entry>       cache;
    static thread_local std::vector<uint64_t>        key;

    ggml_graph_optimize_key(gf, key);

    uint64_t hash = 1469598103934665603ull;
    for (uint64_t v : key) {
        hash = (hash ^ v) * 1099511628211ull;
    }

    const int n = gf->n_nodes;

    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = cache.find(hash);
        if (it != cache.end() && it->second.key == key && (int) it->second.order.size() == n) {
            std::vector<ggml_tensor *> orig(gf->nodes, gf->nodes + n);
            for (int j = 0; j < n; j++) {
                gf->nodes[j] = orig[it->second.order[j]];
            }
            return;
        }
    }

    std::vector<ggml_tensor *> orig(gf->nodes, gf->nodes + n);

    ggml_graph_optimize_impl(gf);

    std::unordered_map<const ggml_tensor *, int> index;
    index.reserve(n);
    for (int i = 0; i < n; i++) {
        index[orig[i]] = i;
    }

    entry e;
    e.key = key;
    e.order.resize(n);
    for (int j = 0; j < n; j++) {
        e.order[j] = index.at(gf->nodes[j]);
    }

    std::lock_guard<std::mutex> lock(mutex);
    if (cache.size() >= 256) {
        cache.clear();
    }
    cache[hash] = std::move(e);
}

static void ggml_graph_optimize_impl(ggml_cgraph * gf) {
    constexpr int MAX_FUSE = 16;

    const int n = gf->n_nodes;

    enum ggml_op ops[MAX_FUSE];

    std::vector<node_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    for (int i = 0; i < n; i++) {
        node_info node = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        // fuse only ops that start with these operations
        // can be expanded when needed
        if (node.op() == GGML_OP_ADD ||
            node.op() == GGML_OP_NORM ||
            node.op() == GGML_OP_RMS_NORM) {
            ops[0] = node.op();

            int f = i + 1;
            while (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                if (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    break;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            for (; f > 1; f--) {
                if (ggml_can_fuse(gf, i, ops, f)) {
                    break;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            for (int k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                node.add_fused(gf->nodes[i]);
            }
        } else if (node.op() == GGML_OP_MUL_MAT && i + 3 < n &&
                   gf->nodes[i]->src[0]->type == GGML_TYPE_BF16 &&
                   gf->nodes[i]->src[0]->ne[0] == 64 &&
                   gf->nodes[i + 1]->op == GGML_OP_INKLING_CTRL_BLEND &&
                   gf->nodes[i + 1]->src[2] == gf->nodes[i] &&
                   gf->nodes[i + 2]->op == GGML_OP_RMS_NORM &&
                   gf->nodes[i + 2]->src[0] == gf->nodes[i + 1] &&
                   gf->nodes[i + 3]->op == GGML_OP_MUL &&
                   (gf->nodes[i + 3]->src[0] == gf->nodes[i + 2] || gf->nodes[i + 3]->src[1] == gf->nodes[i + 2]) &&
                   ggml_node_has_n_uses(gf, i, 1) &&
                   ggml_node_has_n_uses(gf, i + 1, 1) &&
                   ggml_node_has_n_uses(gf, i + 2, 1)) {
            for (int k = 1; k <= 3; ++k) {
                ++i;
                node.add_fused(gf->nodes[i]);
            }
        } else if (node.op() == GGML_OP_MUL_MAT && i + 1 < n &&
                   gf->nodes[i]->src[0]->ne[0] == 16 &&
                   gf->nodes[i]->src[0]->type == GGML_TYPE_F32 &&
                   gf->nodes[i + 1]->op == GGML_OP_SUB) {
            ops[0] = GGML_OP_MUL_MAT;
            ops[1] = GGML_OP_SUB;
            if (ggml_can_fuse(gf, i, ops, 2)) {
                ++i;
                node.add_fused(gf->nodes[i]);
            }
        } else if (node.op() == GGML_OP_UNARY && i + 1 < n &&
                   ggml_get_unary_op(gf->nodes[i]) == GGML_UNARY_OP_SIGMOID &&
                   gf->nodes[i]->src[0]->ne[2] == 1 && gf->nodes[i]->src[0]->ne[3] == 1 &&
                   gf->nodes[i + 1]->op == GGML_OP_MUL) {
            ops[0] = GGML_OP_UNARY;
            ops[1] = GGML_OP_MUL;
            if (ggml_can_fuse(gf, i, ops, 2)) {
                ++i;
                node.add_fused(gf->nodes[i]);
                if (i + 1 < n &&
                    gf->nodes[i + 1]->op == GGML_OP_MUL_MAT &&
                    gf->nodes[i + 1]->src[1] == gf->nodes[i] &&
                    gf->nodes[i + 1]->src[0]->type == GGML_TYPE_BF16 &&
                    gf->nodes[i + 1]->src[0]->ne[0] % 64 == 0 &&
                    ggml_node_has_n_uses(gf, i, 1)) {
                    ++i;
                    node.add_fused(gf->nodes[i]);
                }
            }
        } else if (node.op() == GGML_OP_INKLING_SCONV && i + 4 < n &&
                   gf->nodes[i + 1]->op == GGML_OP_RESHAPE &&
                   (gf->nodes[i + 2]->op == GGML_OP_L2_NORM || gf->nodes[i + 2]->op == GGML_OP_RMS_NORM) &&
                   gf->nodes[i + 3]->op == GGML_OP_VIEW &&
                   gf->nodes[i + 4]->op == GGML_OP_SET_ROWS &&
                   gf->nodes[i + 1]->view_src == gf->nodes[i] &&
                   gf->nodes[i + 2]->src[0]   == gf->nodes[i + 1] &&
                   gf->nodes[i + 3]->view_src == gf->nodes[i + 2] &&
                   gf->nodes[i + 4]->src[0]   == gf->nodes[i + 3] &&
                   ggml_node_has_n_uses(gf, i, 1) &&
                   ggml_node_has_n_uses(gf, i + 2, 1)) {
            for (int k = 1; k <= 4; ++k) {
                ++i;
                node.add_fused(gf->nodes[i]);
            }
        } else if (node.op() == GGML_OP_INKLING_SCONV && i + 1 < n &&
                   gf->nodes[i + 1]->op == GGML_OP_ADD) {
            ops[0] = GGML_OP_INKLING_SCONV;
            ops[1] = GGML_OP_ADD;
            if (ggml_can_fuse(gf, i, ops, 2)) {
                ++i;
                node.add_fused(gf->nodes[i]);
            }
        } else if (node.op() == GGML_OP_INKLING_MIX && i + 1 < n &&
                   gf->nodes[i + 1]->op == GGML_OP_RMS_NORM) {
            ops[0] = GGML_OP_INKLING_MIX;
            ops[1] = GGML_OP_RMS_NORM;
            ops[2] = GGML_OP_MUL;
            if (i + 2 < n && gf->nodes[i + 2]->op == GGML_OP_MUL && ggml_can_fuse(gf, i, ops, 3)) {
                ++i; node.add_fused(gf->nodes[i]);
                ++i; node.add_fused(gf->nodes[i]);
            } else if (ggml_can_fuse(gf, i, ops, 2)) {
                ++i; node.add_fused(gf->nodes[i]);
            }
        } else if (node.op() == GGML_OP_INKLING_CTRL_BLEND && i + 1 < n &&
                   gf->nodes[i + 1]->op == GGML_OP_RMS_NORM) {
            ops[0] = GGML_OP_INKLING_CTRL_BLEND;
            ops[1] = GGML_OP_RMS_NORM;
            ops[2] = GGML_OP_MUL;
            if (i + 2 < n && gf->nodes[i + 2]->op == GGML_OP_MUL && ggml_can_fuse(gf, i, ops, 3)) {
                ++i; node.add_fused(gf->nodes[i]);
                ++i; node.add_fused(gf->nodes[i]);
            } else if (ggml_can_fuse(gf, i, ops, 2)) {
                ++i; node.add_fused(gf->nodes[i]);
            }
        } else if (node.op() == GGML_OP_MUL && i + 1 < n &&
                   gf->nodes[i + 1]->op == GGML_OP_L2_NORM &&
                   gf->nodes[i]->src[0]->ne[0] % 4 == 0 && gf->nodes[i]->src[0]->ne[0] <= 128) {
            ops[0] = GGML_OP_MUL;
            ops[1] = GGML_OP_L2_NORM;
            ops[2] = (i + 2 < n) ? gf->nodes[i + 2]->op : GGML_OP_NONE;
            if (i + 2 < n && (ops[2] == GGML_OP_SCALE || ops[2] == GGML_OP_MUL) && ggml_can_fuse(gf, i, ops, 3)) {
                ++i; node.add_fused(gf->nodes[i]);
                ++i; node.add_fused(gf->nodes[i]);
                // Keep trailing Inkling2 rel_bias MUL_MAT(rel_table [16, ne01], Rcur) [+ SUB(sinks)] adjacent
                if (gf->nodes[i]->ne[0] == 16 && i + 1 < n &&
                    gf->nodes[i + 1]->op == GGML_OP_MUL_MAT &&
                    gf->nodes[i + 1]->src[1] == gf->nodes[i] &&
                    gf->nodes[i + 1]->src[0]->ne[0] == 16 &&
                    gf->nodes[i + 1]->src[0]->type == GGML_TYPE_F32 &&
                    ggml_node_has_n_uses(gf, i, 1)) {
                    ++i; node.add_fused(gf->nodes[i]);
                    if (i + 1 < n &&
                        gf->nodes[i + 1]->op == GGML_OP_SUB &&
                        gf->nodes[i + 1]->src[0] == gf->nodes[i] &&
                        ggml_node_has_n_uses(gf, i, 1)) {
                        ++i; node.add_fused(gf->nodes[i]);
                    }
                }
            } else if (ggml_can_fuse(gf, i, ops, 2)) {
                ++i; node.add_fused(gf->nodes[i]);
            }
        } else if (node.op() == GGML_OP_GET_ROWS && i + 5 < n &&
                   gf->nodes[i]->src[0]->type == GGML_TYPE_BF16 &&
                   gf->nodes[i]->src[0]->ne[0] == 256 &&
                   gf->nodes[i + 1]->op == GGML_OP_CONCAT &&
                   gf->nodes[i + 2]->op == GGML_OP_MUL_MAT &&
                   gf->nodes[i + 2]->src[0]->ne[0] == 512 &&
                   gf->nodes[i + 2]->src[0]->ne[1] == 768 &&
                   gf->nodes[i + 3]->op == GGML_OP_ADD &&
                   gf->nodes[i + 4]->op == GGML_OP_VIEW &&
                   gf->nodes[i + 5]->op == GGML_OP_UNARY &&
                   ggml_get_unary_op(gf->nodes[i + 5]) == GGML_UNARY_OP_TANH) {
            int f = 6;
            if (i + 12 < n &&
                gf->nodes[i + 6]->op == GGML_OP_VIEW &&
                gf->nodes[i + 7]->op == GGML_OP_UNARY && ggml_get_unary_op(gf->nodes[i + 7]) == GGML_UNARY_OP_TANH &&
                gf->nodes[i + 8]->op == GGML_OP_VIEW &&
                gf->nodes[i + 9]->op == GGML_OP_UNARY && ggml_get_unary_op(gf->nodes[i + 9]) == GGML_UNARY_OP_SIGMOID &&
                gf->nodes[i + 10]->op == GGML_OP_SUB &&
                gf->nodes[i + 11]->op == GGML_OP_MUL &&
                gf->nodes[i + 12]->op == GGML_OP_ADD) {
                f = 13;
            }
            for (int k = 1; k < f; k++) {
                ++i;
                node.add_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(node));
    }

#if 1
    // reorder to improve concurrency
    const auto order = ggml_metal_graph_optimize_reorder(nodes);
#else
    std::vector<int> order(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        int j = 0;
        for (const auto i : order) {
            const auto & node = nodes[i];

            gf->nodes[j++] = node.node;

            for (auto * fused : node.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
