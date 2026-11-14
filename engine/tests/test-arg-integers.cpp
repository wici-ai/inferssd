// SPDX-License-Identifier: Apache-2.0
#include "arg.h"
#include "common.h"
#include "preset.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static int checks = 0;
static int failures = 0;

static void check(bool condition, const std::string & label) {
    ++checks;
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
    }
}

static bool parse(common_params & params, std::vector<std::string> arguments) {
    arguments.insert(arguments.begin(), "integer-options");
    std::vector<char *> argv;
    for (auto & argument : arguments) {
        argv.push_back(argument.data());
    }
    return common_params_parse(static_cast<int>(argv.size()), argv.data(), params, LLAMA_EXAMPLE_SERVER);
}

static void set_environment(const char * key, const char * value) {
#ifdef _WIN32
    const int status = _putenv_s(key, value ? value : "");
#else
    const int status = value ? setenv(key, value, 1) : unsetenv(key);
#endif
    if (status != 0) {
        std::fprintf(stderr, "cannot set test environment variable %s\n", key);
        std::exit(2);
    }
}

struct saved_environment {
    const char * key;
    bool present;
    std::string value;

    explicit saved_environment(const char * name) : key(name), present(std::getenv(name) != nullptr) {
        if (present) {
            value = std::getenv(name);
        }
        set_environment(key, nullptr);
    }
    ~saved_environment() {
        set_environment(key, present ? value.c_str() : nullptr);
    }
};

static const std::vector<std::string> malformed = {"12junk", "12.5", "1e3", "0x10", "12 34", "7\tjunk"};

static void test_cli() {
    for (const auto * option : {"--threads", "--ctx-size", "--batch-size", "--port", "--n-predict",
                                "--gpu-layers", "--spec-draft-ngl"}) {
        for (const auto & value : malformed) {
            common_params params;
            check(!parse(params, {option, value}), std::string(option) + " rejects " + value);
        }
    }
    for (const auto & range : {std::vector<std::string>{"1junk", "2"}, {"1", "2junk"}}) {
        common_params params;
        check(!parse(params, {"--control-vector-layer-range", range[0], range[1]}), "control-vector layer range rejects suffix");
    }
}

static void test_environment() {
    for (const auto * key : {"LLAMA_ARG_THREADS", "LLAMA_ARG_CTX_SIZE", "LLAMA_ARG_N_GPU_LAYERS"}) {
        saved_environment saved(key);
        for (const auto & value : malformed) {
            set_environment(key, value.c_str());
            common_params params;
            check(!parse(params, {}), std::string(key) + " rejects " + value);
        }
    }
}

static void test_presets() {
    common_preset_context context(LLAMA_EXAMPLE_SERVER);
    for (const auto * key : {"LLAMA_ARG_THREADS", "LLAMA_ARG_CTX_SIZE", "LLAMA_ARG_N_GPU_LAYERS"}) {
        for (const auto & value : malformed) {
            common_preset preset;
            preset.set_option(context, key, value);
            common_params params;
            bool rejected = false;
            try {
                preset.apply_to_params(params);
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            check(rejected, std::string("preset ") + key + " rejects " + value);
        }
    }
}

static void test_valid() {
    saved_environment saved("LLAMA_ARG_N_PREDICT");
    common_preset_context context(LLAMA_EXAMPLE_SERVER);
    const std::vector<std::pair<std::string, int>> values = {
        {"0", 0}, {"+12", 12}, {"-1", -1}, {" 12", 12}, {"12 \t\r\n", 12},
        {std::to_string(INT_MIN), INT_MIN}, {std::to_string(INT_MAX), INT_MAX},
    };
    for (const auto & item : values) {
        set_environment(saved.key, nullptr);
        common_params cli;
        check(parse(cli, {"--n-predict", item.first}) && cli.n_predict == item.second, "CLI preserves " + item.first);

        set_environment(saved.key, item.first.c_str());
        common_params environment;
        check(parse(environment, {}) && environment.n_predict == item.second, "environment preserves " + item.first);

        common_preset preset;
        preset.set_option(context, saved.key, item.first);
        common_params applied;
        preset.apply_to_params(applied);
        check(applied.n_predict == item.second, "preset preserves " + item.first);
    }
    set_environment(saved.key, nullptr);
    for (const auto * value : {"", "junk", "2147483648", "-2147483649"}) {
        common_params params;
        check(!parse(params, {"--n-predict", value}), std::string("existing invalid integer remains rejected: ") + value);
    }
    for (const auto & item : {std::make_pair("auto", -1), std::make_pair("all", -2)}) {
        common_params params;
        check(parse(params, {"--gpu-layers", item.first, "--spec-draft-ngl", item.first}) &&
                  params.n_gpu_layers == item.second && params.speculative.draft.n_gpu_layers == item.second,
              "symbolic GPU layer limits remain accepted");
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        return 2;
    }
    common_init();
    const std::string mode = argv[1];
    if (mode == "cli") { test_cli(); }
    else if (mode == "environment") { test_environment(); }
    else if (mode == "presets") { test_presets(); }
    else if (mode == "valid") { test_valid(); }
    else { return 2; }
    std::printf("%s: %d checks, %d failures\n", mode.c_str(), checks, failures);
    return failures ? 1 : 0;
}
