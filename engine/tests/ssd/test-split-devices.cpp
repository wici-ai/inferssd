// SPDX-License-Identifier: Apache-2.0
#include "common.h"
#include "inferssd/options.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

// Only device metadata is needed before opening the GGUF.
struct test_gpu {
    ggml_backend_device device{};
    const char * name;
    const char * id;
    enum ggml_backend_dev_type type;

    test_gpu(ggml_backend_reg_t reg, const char * name, const char * id, enum ggml_backend_dev_type type)
        : name(name), id(id), type(type) {
        device.reg = reg;
        device.context = this;
        device.iface.get_name = [](ggml_backend_dev_t d) { return static_cast<test_gpu *>(d->context)->name; };
        device.iface.get_description = device.iface.get_name;
        device.iface.get_type = [](ggml_backend_dev_t d) { return static_cast<test_gpu *>(d->context)->type; };
        device.iface.get_props = [](ggml_backend_dev_t d, ggml_backend_dev_props * props) {
            const auto & gpu = *static_cast<test_gpu *>(d->context);
            props->name = props->description = gpu.name;
            props->device_id = gpu.id;
            props->type = gpu.type;
            props->memory_free = props->memory_total = size_t{1} << 30;
        };
        ggml_backend_device_register(&device);
    }
};

static void check(const char * label, const std::vector<float> & split, bool refused,
                  std::vector<ggml_backend_dev_t> devices = {}, bool experts = true) {
    const char * path = "absent-split-topology-model.gguf";
    common_params params;
    params.split_mode = LLAMA_SPLIT_MODE_LAYER;
    params.split_mode_explicit = true;
    params.devices = std::move(devices);
    if (!params.devices.empty()) { params.devices.push_back(nullptr); }
    std::copy(split.begin(), split.end(), params.tensor_split);
    if (experts) {
        params.ssd_enabled = true;
        params.ssd_overrides.emplace_back("expert.source", "[\"absent-split-topology-model.gguf\"]");
    }
    common_ssd_prepare(params);
    std::string log;
    llama_log_set([](ggml_log_level, const char * line, void * data) {
        *static_cast<std::string *>(data) += line;
    }, &log);
    auto * model = llama_model_load_from_file(path, common_model_params_to_llama(params));
    llama_log_set(nullptr, nullptr);
    const std::string expected = refused ? "tensor split assigns layers to an unavailable GPU"
                                         : std::string("failed to load model from ") + path;
    if (model || log.find(expected) == std::string::npos) {
        llama_model_free(model);
        throw std::runtime_error(std::string(label) + ": expected " + expected + "\n" + log);
    }
    std::printf("%s: %s\n", label, expected.c_str());
}

int main() {
    // Keep the topology independent of the runner's own GPUs.
    const auto cpu = ggml_backend_dev_backend_reg(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    for (size_t i = ggml_backend_reg_count(); i > 0; --i) {
        auto reg = ggml_backend_reg_get(i - 1);
        if (reg != cpu) { ggml_backend_unload(reg); }
    }
    test_gpu integrated(cpu, "integrated", "igpu", GGML_BACKEND_DEVICE_TYPE_IGPU);
    check("integrated only", {1}, false);
    test_gpu discrete(cpu, "discrete", "slot-a", GGML_BACKEND_DEVICE_TYPE_GPU);
    check("discrete with integrated", {1, 0}, false);
    check("integrated excluded", {0, 1}, true);
    check("explicit integrated", {0, 1}, false, {&discrete.device, &integrated.device});
    check("explicit subset", {0, 1}, true, {&integrated.device});
    test_gpu alias(cpu, "discrete-alias", "slot-a", GGML_BACKEND_DEVICE_TYPE_GPU);
    check("duplicate excluded", {0, 1}, true);
    test_gpu other(cpu, "second-discrete", "slot-b", GGML_BACKEND_DEVICE_TYPE_GPU);
    check("two discrete", {0, 1}, false);
    check("third unavailable", {0, 0, 1}, true);
    check("ordinary model", {0, 0, 1}, false, {}, false);
    return 0;
}
