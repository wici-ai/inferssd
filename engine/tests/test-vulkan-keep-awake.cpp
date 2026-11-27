// SPDX-License-Identifier: Apache-2.0
// The Vulkan backend keeps a discrete GPU under Linux runtime power management awake while it uses it
// (vk_keep_awake in ggml-vulkan.cpp): with a buffer in its VRAM and no work for twice the autosuspend delay, the GPU
// must stay active, so the kernel never moves the buffer to host memory. Without the keeper an idle GPU suspends after
// the delay (amdgpu: 5 s) and this test fails. Needs an owned discrete GPU whose PCI device has power/control "auto"
// and nothing else using it; skips (77) elsewhere. Not in ctest.
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>

static std::string read_word(const std::string & path) {
    std::string word;
    std::ifstream(path) >> word;
    return word;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU && !strncmp(ggml_backend_dev_name(d), "Vulkan", 6)) {
            dev = d;
            break;
        }
    }
    if (!dev) {
        std::printf("SKIP: no discrete Vulkan GPU\n");
        return 77;
    }
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    if (!props.device_id) {
        std::printf("SKIP: %s has no PCI address\n", ggml_backend_dev_name(dev));
        return 77;
    }
    const std::string pci = std::string("/sys/bus/pci/devices/") + props.device_id;
    const std::string control = read_word(pci + "/power/control");
    long delay_ms = -1;
    std::ifstream(pci + "/power/autosuspend_delay_ms") >> delay_ms;
    if (control != "auto" || delay_ms < 0) {
        std::printf("SKIP: %s (%s): runtime power management does not suspend it (control=%s autosuspend_delay_ms=%ld)\n",
                ggml_backend_dev_name(dev), props.device_id, control.c_str(), delay_ms);
        return 77;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        std::printf("FAIL: cannot initialize %s\n", ggml_backend_dev_name(dev));
        return 1;
    }
    const size_t bytes = size_t(256) << 20;
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), bytes);
    if (!buffer) {
        std::printf("FAIL: cannot allocate %zu bytes on %s\n", bytes, ggml_backend_dev_name(dev));
        ggml_backend_free(backend);
        return 1;
    }
    ggml_backend_buffer_clear(buffer, 1);
    ggml_backend_synchronize(backend);

    const std::string suspended_before = read_word(pci + "/power/runtime_suspended_time");
    const long idle_ms = 2 * delay_ms + 1000;
    std::printf("%s (%s): %zu MiB in VRAM, idle for %ld ms (autosuspend delay %ld ms)\n", ggml_backend_dev_name(dev),
            props.device_id, bytes >> 20, idle_ms, delay_ms);
    std::this_thread::sleep_for(std::chrono::milliseconds(idle_ms));
    const std::string status = read_word(pci + "/power/runtime_status");
    const std::string suspended_after = read_word(pci + "/power/runtime_suspended_time");
    std::printf("runtime_status=%s runtime_suspended_time %s -> %s ms\n", status.c_str(), suspended_before.c_str(),
            suspended_after.c_str());

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    const bool ok = status == "active" && suspended_after == suspended_before;
    std::printf("%s: the GPU %s while the backend held it\n", ok ? "PASS" : "FAIL",
            ok ? "stayed awake" : "suspended");
    return ok ? 0 : 1;
}
