// SPDX-License-Identifier: Apache-2.0
#include "inkling-experts.h"
#include "ggml-ssd/platform/file.h"
#include "ggml-ssd/store/source.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <array>
#include <cstring>

inkling_expert_bank index_inkling_bank(llama_model_loader & loader, const ssd::store::source_view & source,
        const std::string & name, int64_t columns, int64_t rows, int64_t experts) {
    const auto * tensor = loader.check_tensor_dims(name, {columns, rows, experts}, true);
    const auto & weight = loader.require_weight(name.c_str());
    int fd;
    uint64_t offset;
    size_t bytes;
    ssd::platform::file_identity indexed, model_file;
    if (!source.resolve(source.userdata, name.c_str(), 0, tensor->nb[2], experts, &fd, &offset, &bytes) ||
            bytes != tensor->nb[2] || offset != weight.offs || !ssd::platform::identity(fd, indexed) ||
            !ssd::platform::identity(loader.files.at(weight.idx)->file_id(), model_file) || indexed != model_file) {
        throw std::runtime_error("invalid direct-read source for " + name);
    }
    loader.account_external_tensor(*tensor);
    return {tensor->type, columns, rows, experts, bytes, fd, offset};
}

namespace {
// Both banks and the Metal matmuls use the same first-occurrence ordering.
// A routed expert appears once in a staging buffer even when many tokens use it.
template<class Visit>
bool visit_routes(const ggml_tensor * ids, int64_t experts, Visit visit) {
    std::array<int32_t, 256> slots;
    slots.fill(-1);
    int32_t count = 0;
    for (int64_t t = 0; t < ids->ne[1]; ++t) {
        for (int64_t k = 0; k < ids->ne[0]; ++k) {
            int32_t id;
            std::memcpy(&id, (const char *) ids->data + t * ids->nb[1] + k * ids->nb[0], sizeof(id));
            if (id < 0 || id >= experts) { return false; }
            const bool first = slots[id] < 0;
            if (first) { slots[id] = count++; }
            if (!visit(id, slots[id], first, t * ids->ne[0] + k)) { return false; }
        }
    }
    return true;
}

void remap(ggml_tensor * dst, int, int, void * userdata) {
    auto * out = static_cast<int32_t *>(dst->data);
    // Invalid routes are reported by the reads. Keep the matmul indices safe
    // until the context can return the I/O error to the caller.
    auto * ids = dst->src[0];
    if (!visit_routes(ids, static_cast<const inkling_expert_bank *>(userdata)->experts, [&](int, int slot, bool, int64_t i) {
        out[i] = slot;
        return true;
    })) { std::fill_n(out, ggml_nelements(ids), 0); }
}

void read_experts(ggml_tensor * dst, int, int, void * userdata) {
    auto & read = *static_cast<inkling_expert_read *>(userdata);
    const auto & bank = *read.bank;
    size_t loaded = 0;
    read.bytes = 0;
    read.failed = !visit_routes(dst->src[0], bank.experts, [&](int id, int slot, bool first, int64_t) {
        if (!first) { return true; }
        auto * target = static_cast<char *>(dst->data) + size_t(slot) * bank.expert_bytes;
        if (!ssd::platform::pread_exact(bank.fd, target, bank.expert_bytes,
                    bank.offset + uint64_t(id) * bank.expert_bytes)) { return false; }
        read.bytes += bank.expert_bytes;
        ++loaded;
        return true;
    });
    const size_t valid = loaded * bank.expert_bytes;
    std::memset(static_cast<char *>(dst->data) + valid, 0, ggml_nbytes(dst) - valid);
}
}

ggml_tensor * inkling_compact_ids(ggml_context * ctx, ggml_tensor * ids, const inkling_expert_bank & bank) {
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->ne[2] == 1 && ids->ne[3] == 1 &&
                bank.experts >= 1 && bank.experts <= 256);
    return ggml_custom_4d(ctx, GGML_TYPE_I32, ids->ne[0], ids->ne[1], 1, 1, &ids, 1, remap, 1, const_cast<inkling_expert_bank *>(&bank));
}

ggml_tensor * inkling_read_bank(ggml_context * ctx, ggml_tensor * ids, inkling_expert_read * read) {
    const auto & b = *read->bank;
    return ggml_custom_4d(ctx, b.type, b.columns, b.rows, std::min(ggml_nelements(ids), b.experts), 1,
            &ids, 1, read_experts, 1, read);
}
