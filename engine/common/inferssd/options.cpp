// SPDX-License-Identifier: Apache-2.0
#include "options.h"
#include "../common.h"
#include "llama-ssd.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

std::string common_ssd_option_name(const char * key) {
    std::string option = std::string("--ssd-") + key;
    std::replace(option.begin(), option.end(), '.', '-');
    std::replace(option.begin(), option.end(), '_', '-');
    return option;
}

void common_ssd_prepare(common_params & params) {
    char error[1024] = {};
    if (!llama_ssd_check_environment(error, sizeof(error))) { throw std::invalid_argument(error); }
    if (!params.ssd_enabled) { return; }
    std::vector<llama_ssd_override> overrides;
    for (const auto & entry : params.ssd_overrides) {
        overrides.push_back({entry.first.c_str(), entry.second.c_str()});
    }
    auto * options = llama_ssd_options_create(params.ssd_settings_file.c_str(),
        overrides.data(), overrides.size(), params.ssd_telemetry_file.c_str(), error, sizeof(error));
    if (!options) { throw std::invalid_argument(error); }
    params.ssd_options = std::shared_ptr<llama_ssd_options>(options, llama_ssd_options_free);
}

void common_ssd_print_schema() {
    const auto size = llama_ssd_schema_json(nullptr, 0);
    if (!size) { throw std::runtime_error("cannot serialize SSD settings schema"); }
    std::vector<char> output(size);
    if (llama_ssd_schema_json(output.data(), output.size()) != size) {
        throw std::runtime_error("SSD settings schema changed while serializing");
    }
    std::puts(output.data());
}
