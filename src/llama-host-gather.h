#pragma once
// The page cache a graph's index gathers bring in: rows of weights the model's file mapping backs, read by an I32
// index operand while the graph runs. GET_ROWS gathers token embeddings; a custom op whose first operand is such a
// weight and whose second is an I32 index gathers rows and decodes them (DeepSeek-V4.1's Engram tables, whose rows
// are read ahead for the whole ubatch before the graph runs). A row of b bytes at an arbitrary offset spans
// 1 + (b - 1) / page pages on average, so each row gathered costs page + b - 1 bytes, and a table at most its own
// pages (and one page for the boundary). The host-memory ledger holds this room for a prefill until the first decode
// token (kimi_cpu_sink_cache_prefill_room).
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>

#if !defined(_WIN32)
#include <unistd.h>
#endif

static inline uint64_t llama_host_page_size() {
#if defined(_WIN32)
    return 4096;
#else
    const long page = sysconf(_SC_PAGESIZE);
    return page > 0 ? uint64_t(page) : 4096;
#endif
}

static inline uint64_t llama_host_gather_bytes(ggml_cgraph * gf, uint64_t page,
        const std::function<bool(const ggml_tensor *)> & file_mapped) {
    std::map<const ggml_tensor *, uint64_t> tables;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->op != GGML_OP_GET_ROWS && node->op != GGML_OP_CUSTOM) { continue; }
        const ggml_tensor * src = node->src[0], * index = node->src[1];
        if (!src || !index || index->type != GGML_TYPE_I32) { continue; }
        const ggml_tensor * table = src->view_src ? src->view_src : src;
        if (!file_mapped(table)) { continue; }
        const uint64_t row = ggml_row_size(src->type, src->ne[0]);
        tables[table] += uint64_t(ggml_nelements(index)) * (page + row - 1);
    }
    uint64_t bytes = 0;
    for (const auto & t : tables) {
        bytes += std::min<uint64_t>(t.second, (uint64_t(ggml_nbytes(t.first)) + page - 1) / page * page + page);
    }
    return bytes;
}
