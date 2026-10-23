// SPDX-License-Identifier: MIT
#pragma once
#include "llama-model-loader.h"
void llama_coalesce_ranges(std::vector<llama_file_range> & ranges);
bool llama_is_expert_tensor_name(const std::string & name, uint32_t & layer);
