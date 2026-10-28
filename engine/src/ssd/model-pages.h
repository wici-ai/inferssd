// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "llama-model-loader.h"

namespace ssd {
// Fixed deferred-expert policy: release every complete mapped expert page at
// the end of each prompt batch. This reports advice, never physical reclamation.
void drop_expert_pages(const std::vector<std::unique_ptr<llama_mmap>> & mappings,
    const std::vector<std::vector<llama_file_range>> & ranges);
}
