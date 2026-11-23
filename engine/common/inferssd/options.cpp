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
    llama_ssd_options_set_layer_events(options, params.ssd_layer_events);
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

bool common_ssd_print_model_facts(const std::string & path, const std::vector<std::string> & shards) {
    // the files the model loads from, as common_init_from_params takes them
    common_params_model model;
    model.path   = path;
    model.shards = shards;
    const std::vector<std::string> paths = model.get_paths();
    std::vector<const char *> c_paths;
    for (const auto & item : paths) { c_paths.push_back(item.c_str()); }
    std::string facts;
    char error[1024];
    const auto append = [](const char * text, size_t size, void * user) { static_cast<std::string *>(user)->append(text, size); };
    if (!llama_ssd_model_facts(c_paths.data(), c_paths.size(), append, &facts, error, sizeof error)) {
        std::fprintf(stderr, "error: %s\n", error);
        return false;
    }
    std::fwrite(facts.data(), 1, facts.size(), stdout);
    std::fputc('\n', stdout);
    return std::fflush(stdout) == 0;
}
