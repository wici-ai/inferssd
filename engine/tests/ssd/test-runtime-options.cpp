// SPDX-License-Identifier: Apache-2.0
#include "ssd/options.h"
#include "ggml-ssd/settings/model-options.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

static void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

int main() {
    char directory[] = "/tmp/ssdash-options-XXXXXX";
    require(mkdtemp(directory), "temporary directory failed");
    const std::string first = std::string(directory) + "/first.jsonl";
    const std::string second = std::string(directory) + "/second.jsonl";
    char error[1024] = {};
    const llama_ssd_override initial[] = {
        {"cpu.threads", "4"},
        {"model.optimizations", "[\"fixture|on|uuid-a|deepseek41|36|2|Example GPU|1|0\"]"}
    };
    auto * options = llama_ssd_options_create(nullptr, initial, 2, first.c_str(), error, sizeof(error));
    require(options, error);
    auto context = std::make_unique<llama_ssd_context>(*options);
    llama_ssd_options_free(options);
    require(*context->settings.integer("cpu.threads") == 4, "context borrowed freed options");
    require(context->services.events == context->events.get() && context->services.debug == &context->debug,
        "context service pointers differ from owned objects");

    const llama_ssd_override other[] = {{"cpu.threads", "3"}};
    options = llama_ssd_options_create(nullptr, other, 1, second.c_str(), error, sizeof(error));
    require(options, error);
    auto independent = std::make_unique<llama_ssd_context>(*options);
    llama_ssd_options_free(options);
    const ssd::settings::model_identity identity {"uuid-a", "deepseek41", "36", "2", "Example GPU", ""};
    auto first_model = ssd::settings::resolve_model(context->settings, identity);
    auto other_model = ssd::settings::resolve_model(independent->settings, identity);
    auto unknown_model = ssd::settings::resolve_model(context->settings, {});
    require(first_model && first_model.value.engram_prefetch && !first_model.value.hc_fused,
            "context did not resolve matching model options");
    require(other_model && !other_model.value.engram_prefetch && other_model.value.hc_fused &&
            unknown_model && !unknown_model.value.engram_prefetch && unknown_model.value.hc_fused,
            "model options leaked between contexts or identities");
    require(*context->settings.integer("cpu.threads") == 4 && *independent->settings.integer("cpu.threads") == 3,
        "one context changed another context's settings");
    require(bool(context->events->event("ssd.io.read", "{\"bytes\":4096}")), "event publication failed");
    context.reset();
    require(bool(independent->events->event("ssd.io.read", "{\"bytes\":8192}")), "other context closed this writer");
    independent.reset();

    // Creation validates settings but does not truncate an existing event file.
    options = llama_ssd_options_create(nullptr, initial, 2, first.c_str(), error, sizeof(error));
    require(options, error);
    bool refused = false;
    try { llama_ssd_context collision(*options); } catch (const std::runtime_error &) { refused = true; }
    llama_ssd_options_free(options);
    require(refused, "existing telemetry file was overwritten");
    for (const auto & path : {first, second}) {
        std::ifstream stream(path);
        std::string settings, contract, event, extra;
        require(bool(std::getline(stream, settings)) && bool(std::getline(stream, contract)) &&
                bool(std::getline(stream, event)) && !std::getline(stream, extra),
            "event count differs");
        require(settings.find("\"ev\":\"ssd.settings\",\"seq\":0") != std::string::npos,
            "effective settings were not the first event");
        require(contract.find("\"ev\":\"ssd.controller.begin\",\"seq\":1") != std::string::npos,
            "controller observation contract is missing");
        require(event.find("\"seq\":2") != std::string::npos, "per-context sequence differs");
    }
    const llama_ssd_override invalid[] = {{"cpu.threads", "0"}};
    require(!llama_ssd_options_create(nullptr, invalid, 1, nullptr, error, sizeof(error)) && error[0],
        "invalid setting escaped C API status boundary");
    require(!llama_ssd_options_create(nullptr, nullptr, 1, nullptr, error, sizeof(error)), "missing override array accepted");
    const auto child = fork();
    require(child >= 0, "environment guard child failed");
    if (!child) {
        setenv("GGML_KIMI_CPU_SINK_THREADS", "private-value-must-not-be-reported", 1);
        auto * rejected = llama_ssd_options_create(nullptr, nullptr, 0, nullptr, error, sizeof(error));
        const std::string message(error);
        const bool safe = !rejected && !llama_ssd_check_environment(error, sizeof(error)) &&
            message.find("GGML_KIMI_CPU_SINK_THREADS") != std::string::npos &&
            message.find("private-value") == std::string::npos;
        llama_ssd_options_free(rejected);
        _exit(safe ? 0 : 1);
    }
    int status = 0;
    require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
        "retired environment did not fail closed by name");
    const auto schema_size = llama_ssd_schema_json(nullptr, 0);
    require(schema_size > 1, "schema unavailable");
    char small[] = "xxxx";
    require(llama_ssd_schema_json(small, sizeof(small)) == schema_size && small[0] == '\0', "short schema buffer differs");
    unlink(first.c_str());
    unlink(second.c_str());
    rmdir(directory);
    std::puts("SSD_RUNTIME_OPTIONS_OWNERSHIP_PASS");
}
