// SPDX-License-Identifier: Apache-2.0
#include "ssd/selection-bias.h"
#include "ggml-ssd/host/expert-cache.h"
#include "ggml-cuda/ssd/vram-slot-table.h"
#include <stdexcept>
#include <string>
#include <vector>

static int checks = 0;
static void require(bool ok) {
    ++checks;
    if (!ok) { throw std::runtime_error("selection bias assertion " + std::to_string(checks)); }
}
int main() { try {
    // Exhaust every declared architecture, including the unknown sentinel, with and without the execution path.
    for (int arch = 0; arch <= int(LLM_ARCH_UNKNOWN); ++arch) {
        for (bool cuda_sink : {false, true}) {
            FILE * receipt = std::tmpfile(); require(receipt != nullptr);
            auto option = llama_selection_bias_options::activate(llm_arch(arch), cuda_sink, receipt);
            require(option.enabled == (arch == int(LLM_ARCH_DEEPSEEK41) && cuda_sink));
            std::rewind(receipt); char line[512] = {}; require(std::fgets(line, sizeof(line), receipt));
            const char * model = arch == int(LLM_ARCH_DEEPSEEK41) ? "\"deepseek\"" :
                arch == int(LLM_ARCH_MUSE_SPARK) ? "\"muse\"" : "null";
            const std::string expected = std::string("rsi.design.active {\"consumer\":\"deepseek-cache-residency-selection-bias\",\"model\":") +
                model + ",\"enabled\":" + (option.enabled ? "true" : "false") + ",\"once_per_context\":true}\n";
            require(line == expected); require(std::fgetc(receipt) == EOF); std::fclose(receipt);
        }
    }
    llama_selection_bias_schedule schedule;
    require(schedule.refresh(false)); require(schedule.refresh(false));
    for (int step = 0; step < 65; ++step) { require(schedule.refresh(true) == (step % 16 == 0)); }
    require(schedule.refresh(false)); require(schedule.refresh(true));
    require(llama_selection_bias_options::bonus(false, false) == 0.0f);
    require(llama_selection_bias_options::bonus(true, false) == 0.2f);
    require(llama_selection_bias_options::bonus(false, true) == 0.3f);
    require(llama_selection_bias_options::bonus(true, true) == 0.3f);

    using cache_t = ssd::host::expert_cache;
    ssd::host::expert_cache_options config; config.require_shared = false;
    cache_t cache(config); cache_t::layer layer; layer.id = 1; layer.count = 1;
    cache.layers.push_back(layer); cache.entries.resize(1);
    ssd::host::expert_request request = {}; request.id = 0;
    auto & entry = cache.entries[0]; entry.id = 0; entry.valid = entry.resident = entry.enabled = true;
    for (int role = 0; role < 3; ++role) {
        auto & source = request.role[role]; source.offset = uint64_t(role) * 4096; source.nbytes = 4096;
        source.file = {1, 2, 3, 65536, 4, 5}; entry.source[role] = cache_t::key(source);
    }
    // The metadata query never dereferences the arena. Remove the sentinel before owner destruction.
    char sentinel = 0; cache.arena = &sentinel;
    struct arena_reset { cache_t & c; ~arena_reset() { c.arena = nullptr; } } reset{cache};
    require(!cache.selection_holds(1, request)); cache.active = true;
    require(cache.selection_holds(1, request)); require(!cache.selection_holds(2, request));
    entry.inflight = true; require(!cache.selection_holds(1, request)); entry.inflight = false;
    entry.valid = false; entry.reserved = true; require(!cache.selection_holds(1, request));
    entry.valid = true; entry.reserved = false;
    entry.resident = false; require(!cache.selection_holds(1, request)); entry.resident = true;
    entry.enabled = false; require(!cache.selection_holds(1, request)); entry.enabled = true;
    cache.active = false; cache.prefill_read = true; require(cache.selection_holds(1, request));
    for (int role = 0; role < 3; ++role) {
        auto stale = request; stale.role[role].file.ctime_ns++; require(!cache.selection_holds(1, stale));
        stale = request; stale.role[role].file.id_hi++; require(!cache.selection_holds(1, stale));
        stale = request; stale.role[role].offset++; require(!cache.selection_holds(1, stale));
    }

    vram_slot_table gpu(1, {1}, false, 0, false, 2, 1, 1, 2);
    kimi_vram_request key = {}; key.layer = 1; key.expert = 0;
    for (int role = 0; role < 3; ++role) {
        const auto & s = request.role[role];
        key.role[role] = kimi_vram_source_identity(s.file, s.offset, s.nbytes);
    }
    require(!gpu.selection_holds(key));
    gpu.change(gpu.slots[0], [&](vram_slot_table::entry & e) { e.key = key; e.keyed = e.valid = true; });
    require(gpu.selection_holds(key));
    auto weighted = key; weighted.weight = 100; require(gpu.selection_holds(weighted));
    for (int role = 0; role < 3; ++role) {
        auto stale = key; stale.role[role].mtime_ns++; require(!gpu.selection_holds(stale));
        stale = key; stale.role[role].bytes++; require(!gpu.selection_holds(stale));
        stale = key; stale.role[role].inode_hi++; require(!gpu.selection_holds(stale));
    }
    gpu.change(gpu.slots[0], [](vram_slot_table::entry & e) { e.valid = false; e.uploading = true; });
    require(gpu.holds(1, 0)); require(!gpu.selection_holds(key)); // pending bytes are not snapshot residents
    for (int value : {-1, 10000}) {
        auto invalid = key; invalid.layer = value; require(!gpu.selection_holds(invalid));
        invalid = key; invalid.expert = value; require(!gpu.selection_holds(invalid));
    }
    // Concrete receipts from four separate synthetic context activations.
    llama_selection_bias_options::activate(LLM_ARCH_DEEPSEEK41, true, stdout);
    llama_selection_bias_options::activate(LLM_ARCH_DEEPSEEK41, false, stdout);
    llama_selection_bias_options::activate(LLM_ARCH_MUSE_SPARK, true, stdout);
    llama_selection_bias_options::activate(LLM_ARCH_UNKNOWN, true, stdout);
    std::printf("SELECTION_BIAS_PASS assertions=%d models_loaded=0 gpu_contexts=0\n", checks);
    return 0;
} catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return 1; } }
