// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "llama-model-loader.h"
#include "ggml-ssd/settings/settings.h"
#include "ggml-ssd/store/source.h"
#include <string>

void llama_coalesce_ranges(std::vector<llama_file_range> & ranges);
bool llama_is_expert_tensor_name(const std::string & name, uint32_t & layer);

namespace llama_ssd {
// Model-owned admission receipt. No process-wide cache or identity setter.
struct content_identity {
    struct expert { std::string name; size_t shard, offset, bytes; int64_t count; };
    std::string digest;
    std::vector<std::string> manifest;
    llama_files files;
    std::vector<expert> experts;
    std::vector<std::string> stamps;
    void verify(const ssd::settings::snapshot &, const llama_files &);
    std::string for_context(const ssd::settings::snapshot &, const ssd::store::source_view &) const;
};
}
