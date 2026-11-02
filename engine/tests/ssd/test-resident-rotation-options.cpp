// SPDX-License-Identifier: Apache-2.0
#include "llama-rotation-options.h"
#include "ggml-ssd/settings/json.h"
#include <cassert>
#include <iostream>
int main() {
    using namespace ssd::settings;
    const auto defaults = parse("", "");
    const auto frozen = parse("inferssd-settings 1\nmodel.hc_fused=1\n", "unchanged-key-fixture");
    const auto on = parse("inferssd-settings 1\nmodel.kv_rotation_resident=1\n", "fixture");
    const auto off = parse("inferssd-settings 1\nmodel.kv_rotation_resident=0\n", "fixture");
    assert(defaults && frozen && on && off);
    // Exercise every known architecture, plus unknown and future enum values.
    for (int i = 0; i <= int(LLM_ARCH_UNKNOWN) + 1; ++i) {
        const auto arch = static_cast<llm_arch>(i);
        for (const snapshot * settings : {&defaults.value, &frozen.value, &on.value, static_cast<const snapshot *>(nullptr)}) {
            assert(resident_rotation_requested(settings, arch) == (arch == LLM_ARCH_DEEPSEEK41));
        }
        assert(!resident_rotation_requested(&off.value, arch));
        const auto receipt = resident_rotation_receipt(arch, false);
        if (arch == LLM_ARCH_DEEPSEEK41 || arch == LLM_ARCH_MUSE_SPARK) {
            assert(ssd::json::parse(receipt));
        } else {
            assert(receipt.empty());
        }
    }
    assert(!resident_rotation_requested(nullptr, static_cast<llm_arch>(-1)));
    assert(resident_rotation_receipt(static_cast<llm_arch>(-1), false).empty());
    assert(!parse("inferssd-settings 1\nmodel.kv_rotation_resident=2\n", "invalid"));
    assert(!parse("inferssd-settings 1\nmodel.optimizations=[]\n", "unsupported"));
    assert(resident_rotation_receipt(LLM_ARCH_MUSE_SPARK, true) ==
        "{\"consumer\":\"backend-resident-kv-hadamard-rotation-constants\",\"model\":\"muse\",\"enabled\":false,\"once_per_context\":true}");
    assert(resident_rotation_receipt(LLM_ARCH_DEEPSEEK41, true) ==
        "{\"consumer\":\"backend-resident-kv-hadamard-rotation-constants\",\"model\":\"deepseek\",\"enabled\":true,\"once_per_context\":true}");
    std::cout << "PASS DeepSeek41 default activation, explicit off, Muse/all other architectures default off, unknown receipts absent, exact JSON, no PR76 rules\n";
}
