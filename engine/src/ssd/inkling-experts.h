// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ggml-backend.h"

#include <cstdint>
#include <string>

struct llama_model_loader;
namespace ssd { namespace store { struct source_view; } }

// A physical Inkling layer has two packed expert banks. Its small per-expert
// scales stay in memory and continue to use the original router IDs.
struct inkling_expert_bank {
    ggml_type type;
    int64_t columns, rows, experts;
    size_t expert_bytes;
    int fd;
    uint64_t offset;
};

struct inkling_expert_layer {
    ggml_backend_dev_t device;
    inkling_expert_bank gate_up, down;
};

struct inkling_expert_read {
    const inkling_expert_bank * bank;
    uint64_t bytes = 0;
    bool failed = false;
};

inkling_expert_bank index_inkling_bank(llama_model_loader & loader, const ssd::store::source_view & source,
        const std::string & name, int64_t columns, int64_t rows, int64_t experts);

ggml_tensor * inkling_compact_ids(ggml_context * ctx, ggml_tensor * ids, const inkling_expert_bank & bank);
ggml_tensor * inkling_read_bank(ggml_context * ctx, ggml_tensor * ids, inkling_expert_read * read);
