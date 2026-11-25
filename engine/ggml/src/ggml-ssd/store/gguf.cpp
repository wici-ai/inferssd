// SPDX-License-Identifier: Apache-2.0
#include "parser.h"
#include "ggml.h"
#include "gguf.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

namespace c0 {
namespace {
struct free_gguf { void operator()(gguf_context * gguf) const noexcept { gguf_free(gguf); } };
struct free_ggml { void operator()(ggml_context * ggml) const noexcept { ggml_free(ggml); } };

// Scales and router tensors stay resident. Only complete 3-D weight banks are indexed.
bool routed_expert_tensor(const char * name) noexcept {
    unsigned layer;
    int prefix = 0;
    if (std::sscanf(name, "blk.%u.%n", &layer, &prefix) != 1 || !prefix) { return false; }
    const char * tensor = name + prefix;
    return !std::strcmp(tensor, "ffn_up_exps.weight") || !std::strcmp(tensor, "ffn_gate_exps.weight") ||
           !std::strcmp(tensor, "ffn_gate_up_exps.weight") || !std::strcmp(tensor, "ffn_down_exps.weight");
}
}

bool append_gguf_source(parser_state & state, const char * path, std::vector<expert_pack_entry> & entries,
        bool direct_required, source_open_hooks hooks) {
    const int32_t source_idx = int32_t(state.sources.size());
    if (!open_data_source(state, path, "GGUF", false, direct_required, hooks)) { return false; }
    // the header, from the handle the source keeps: the extents name the bytes of the file that is open
    FILE * file = state.sources.back().file;
    ggml_context * meta = nullptr;
    std::unique_ptr<gguf_context, free_gguf> gguf(
        std::fseek(file, 0, SEEK_SET) == 0 ? gguf_init_from_file_ptr(file, { /*no_alloc =*/ true, &meta }) : nullptr);
    std::unique_ptr<ggml_context, free_ggml> tensors(meta);
    if (!gguf || !tensors) {
        std::fprintf(stderr, "[moe_stream_batch] GGUF: invalid header: %s\n", path);
        return false;
    }
    const uint64_t data = gguf_get_data_offset(gguf.get());
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf.get()); ++i) {
        const char * name = gguf_get_tensor_name(gguf.get(), i);
        if (!routed_expert_tensor(name)) { continue; }
        const ggml_tensor * t = ggml_get_tensor(tensors.get(), name);
        // experts after experts, each its rows: ne[2] experts of nb[2] bytes
        if (!t || std::strlen(name) >= sizeof(expert_pack_entry::tensor) || ggml_n_dims(t) != 3 || t->ne[2] <= 0 ||
                t->ne[2] > INT32_MAX || !ggml_is_contiguous(t) || t->nb[2] == 0) {
            std::fprintf(stderr, "[moe_stream_batch] GGUF: %s is not one tensor of routed experts: %s\n", name, path);
            return false;
        }
        const uint64_t first = data + gguf_get_tensor_offset(gguf.get(), i);
        for (int64_t e = 0; e < t->ne[2]; ++e) {
            expert_pack_entry entry = {};
            std::snprintf(entry.tensor, sizeof(entry.tensor), "%s", name);
            entry.expert_idx = int32_t(e);
            entry.source_idx = source_idx;
            entry.offset = first + uint64_t(e) * uint64_t(t->nb[2]);
            entry.nbytes = uint64_t(t->nb[2]);
            entry.alias_source = true; // a range of a model file, as an alias row names one
            entries.push_back(entry);
        }
    }
    return true;
}
} // namespace c0
