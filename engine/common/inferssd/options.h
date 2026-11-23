// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <vector>
struct common_params;

std::string common_ssd_option_name(const char * key);
void common_ssd_prepare(common_params & params);
void common_ssd_print_schema();
// Prints the facts of the model PATH with its extra SHARDS (--shard/--ple) as JSON (llama_ssd_model_facts); false after
// printing why it failed.
bool common_ssd_print_model_facts(const std::string & path, const std::vector<std::string> & shards = {});
