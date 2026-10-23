#include "ssd/explicit-fixture-inputs.h"
// The real CPU sink with the VRAM tier on MXFP4 experts and the DeepSeek-V4.1
// clamped SwiGLU: every output must equal a CPU-only sink bit for bit while GPU
// hits replace SSD reads. The engine's runtime parity gate must refuse a CPU
// reference that takes another accumulation path or differs in one bit. With the
// opt-in deferred uploads the same holds while borrowed uploads outlive their layer. With the
// opt-in GPU hidden chain, SWIGLU_CLAMP outputs stay within 1e-3 relative L2 of the
// CPU sink (not bitwise) and SiTU keeps the bitwise bridge.
// Synthetic packed weights avoid a model dependency.
// Run with libkimi-mxfp4-parity.so and libkimi-vram-tier.so preloaded.
#define main original_sink_test_main
#include "test-kimi-cpu-sink.cpp"
#undef main
#include <dlfcn.h>
#include "kimi-host-jobs-test.inc"

namespace {
struct tier_options {
    bool exclusive, async, global, ram, keep, deferred, early_up_gate, role_major, nonblocking_down_sched;
};

using selfcheck_hook_fn = void (*)(ggml_vec_dot_t);

float e8m0_half(uint8_t e) {
    const uint32_t bits = e < 2 ? uint32_t(0x00200000) << e : uint32_t(e - 1) << 23;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// ggml_vec_dot_mxfp4_q8_0_generic: one float sum over whole blocks. It agrees
// with the AVX2 lanes only up to rounding, as any other CPU kernel path would.
void scalar_order_dot(int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) {
    static const int8_t values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const auto * x = static_cast<const uint8_t *>(vx); // block_mxfp4: e, qs[16]
    const auto * y = static_cast<const uint8_t *>(vy); // block_q8_0: d (fp16), qs[32]
    float sum = 0.0f;
    for (int ib = 0; ib < n / 32; ++ib, x += 17, y += 34) {
        ggml_fp16_t half;
        std::memcpy(&half, y, sizeof(half));
        const float d = ggml_fp16_to_fp32(half) * e8m0_half(x[0]);
        int sumi = 0;
        for (int j = 0; j < 16; ++j) {
            sumi += int8_t(y[2 + j]) * values[x[1 + j] & 15] + int8_t(y[18 + j]) * values[x[1 + j] >> 4];
        }
        sum += d * float(sumi);
    }
    *s = sum;
}

// The linked CPU kernel with the lowest mantissa bit flipped on one column count.
int flipped_columns = 0;
void flipped_dot(int n, float * s, size_t bs, const void * x, size_t bx, const void * y, size_t by, int nrc) {
    ggml_get_type_traits_cpu(GGML_TYPE_MXFP4)->vec_dot(n, s, bs, x, bx, y, by, nrc);
    if (n != flipped_columns) { return; }
    uint32_t bits;
    std::memcpy(&bits, s, sizeof(bits));
    bits ^= 1u;
    std::memcpy(s, &bits, sizeof(bits));
}

// A sink whose MXFP4 engine self-check sees a mismatching CPU reference must
// fail the decode call with the failing check named, again on a retry (no
// fallback, no silent CPU-only continuation).
void selfcheck_mismatch(fixture_inputs & inputs, const char * label, ggml_vec_dot_t reference, const char * expected, size_t & checks) {
    const auto hook = reinterpret_cast<selfcheck_hook_fn>(dlsym(RTLD_DEFAULT, "kimi_vram_mxfp4_selfcheck_reference_for_testing"));
    require(hook != nullptr, "MXFP4 self-check test hook is not exported by the tier DSO");
    fixture f;
    f.type = f.down_type = GGML_TYPE_MXFP4;
    f.latent = 512; f.hidden = 256; f.experts = 4;
    make_synthetic(f, 20260916);
    inputs.set("vram_budget", std::to_string(uint64_t(2 * f.experts) * triplet_bytes(f)).c_str());
    for (const char * name : {"vram_exclusive", "vram_async",
                              "tier.global", "tier.keep_across_requests",
                              "vram_deferred"}) {
        inputs.set(name, "0");
    }
    auto sink = make_sink(inputs, f, 16, 2);
    configure_cache(sink.get(), 0, {1, 2}, f.experts);
    const kimi_cpu_sink_activation activation = {KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 10.0f};
    std::vector<int32_t> ids = {0, 1, 2};
    std::vector<float> weights(3, 0.3f), input(f.latent), out(f.latent);
    for (int i = 0; i < f.latent; ++i) { input[i] = std::sin(float(i)); }
    tensors meta(f, 1);
    std::string first_error;
    hook(reference);
    for (int attempt = 0; attempt < 2; ++attempt) {
        begin_cache(sink.get(), 51, uint64_t(attempt + 1));
        kimi_cpu_sink_stats stats = {};
        char error[512] = {};
        const bool ok = kimi_cpu_sink_run_activation(sink.get(), meta.role[0], meta.role[1], meta.role[2], input.data(),
            ids.data(), weights.data(), int(ids.size()), &activation, out.data(), &stats, error, sizeof(error));
        require(!ok, "MXFP4 parity self-check accepted a mismatching CPU reference");
        require(std::strstr(error, "MXFP4 VRAM parity self-check failed") && std::strstr(error, expected), error);
        require(stats.pending_reads == 0, "self-check failure returned with live reads");
        if (!attempt) { first_error = error; }
        checks += 2;
    }
    hook(nullptr);
    std::printf("MXFP4_SELFCHECK_MISMATCH_PASS case=%s error=\"%s\"\n", label, first_error.c_str());
}

// With the GPU hidden chain on, the engine skips the parity self-check: a mismatching CPU
// reference must not refuse the tier, and the chain keeps producing finite outputs.
void selfcheck_skipped_with_chain(fixture_inputs & inputs, size_t & checks) {
    const auto hook = reinterpret_cast<selfcheck_hook_fn>(dlsym(RTLD_DEFAULT, "kimi_vram_mxfp4_selfcheck_reference_for_testing"));
    require(hook != nullptr, "MXFP4 self-check test hook is not exported by the tier DSO");
    fixture f;
    f.type = f.down_type = GGML_TYPE_MXFP4;
    f.latent = 512; f.hidden = 256; f.experts = 4;
    make_synthetic(f, 20260918);
    inputs.set("vram_budget", std::to_string(uint64_t(2 * f.experts) * triplet_bytes(f)).c_str());
    for (const char * name : {"vram_exclusive", "vram_async",
                              "tier.global", "tier.keep_across_requests",
                              "vram_deferred"}) {
        inputs.set(name, "0");
    }
    inputs.set("tier.mxfp4_gpu_hidden", "1"); // read when the engine is created
    auto sink = make_sink(inputs, f, 16, 2);
    configure_cache(sink.get(), 0, {1, 2}, f.experts);
    const kimi_cpu_sink_activation activation = {KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 10.0f};
    std::vector<int32_t> ids = {0, 1, 2};
    std::vector<float> weights(3, 0.3f), input(f.latent), out(f.latent);
    for (int i = 0; i < f.latent; ++i) { input[i] = std::sin(float(i)); }
    tensors meta(f, 1);
    uint64_t first_reads = 0, last_reads = 0;
    hook(scalar_order_dot);
    for (int token = 1; token <= 3; ++token) {
        begin_cache(sink.get(), 71, uint64_t(token));
        kimi_cpu_sink_stats stats = {};
        char error[512] = {};
        const bool ok = kimi_cpu_sink_run_activation(sink.get(), meta.role[0], meta.role[1], meta.role[2], input.data(),
            ids.data(), weights.data(), int(ids.size()), &activation, out.data(), &stats, error, sizeof(error));
        require(ok, error);
        for (float v : out) { require(std::isfinite(v), "non-finite output with the self-check skipped"); }
        if (token == 1) { first_reads = stats.read_requests; }
        last_reads = stats.read_requests;
        ++checks;
    }
    hook(nullptr);
    inputs.set("tier.mxfp4_gpu_hidden", "0");
    require(last_reads < first_reads, "the chain served no VRAM hits after the skipped self-check");
    ++checks;
    std::printf("MXFP4_SELFCHECK_SKIPPED_WITH_CHAIN_PASS first_reads=%llu last_reads=%llu\n",
        (unsigned long long) first_reads, (unsigned long long) last_reads);
}

void run_shape(fixture_inputs & inputs, int latent, int hidden, int experts, int tokens, size_t & compared, size_t & checks) {
    fixture f;
    f.type = f.down_type = GGML_TYPE_MXFP4;
    f.latent = latent; f.hidden = hidden; f.experts = experts;
    make_synthetic(f, uint32_t(20260915 + latent));
    const uint64_t unit = triplet_bytes(f);
    const std::string gpu_bytes = std::to_string(uint64_t(2 * experts) * unit); // all experts of both layers
    uint64_t cpu_reads = 0, mixed_reads = 0;
    for (const auto cfg : {tier_options{false, false, false, false, false, false, false, false, false},
                           tier_options{false, true, true, false, true, false, false, false, false},
                           tier_options{true, true, true, true, true, false, true, false, false},
                           tier_options{true, true, true, true, true, false, true, true, false},
                           tier_options{true, true, true, true, true, false, true, true, true},
                           tier_options{true, false, false, true, false, false, false, false, false},
                           tier_options{true, true, true, true, true, true, true, true, false},
                           tier_options{true, true, true, true, true, true, true, true, true},
                           tier_options{false, true, false, true, false, true, true, true, false},
                           tier_options{false, true, false, true, false, true, true, true, true}}) {
        inputs.set("vram_budget", "0");
        inputs.set("early_up_gate", "0");
        inputs.set("role_major", "0");
        inputs.set("nonblocking_down_sched", "0");
        inputs.set("vram_exclusive", "0");
        inputs.set("vram_async", "0");
        inputs.set("tier.global", "0");
        inputs.set("tier.keep_across_requests", "0");
        inputs.set("vram_deferred", "0");
        auto cpu = make_sink(inputs, f, 16, 2);
        inputs.set("vram_budget", gpu_bytes.c_str());
        inputs.set("vram_exclusive", cfg.exclusive ? "1" : "0");
        inputs.set("vram_async", cfg.async ? "1" : "0");
        inputs.set("tier.global", cfg.global ? "1" : "0");
        inputs.set("tier.keep_across_requests", cfg.keep ? "1" : "0");
        inputs.set("vram_deferred", cfg.deferred ? "1" : "0");
        inputs.set("early_up_gate", cfg.early_up_gate ? "1" : "0");
        inputs.set("role_major", cfg.role_major ? "1" : "0");
        inputs.set("nonblocking_down_sched", cfg.nonblocking_down_sched ? "1" : "0");
        auto mixed = make_sink(inputs, f, 16, 2);
        const uint64_t ram = cfg.ram ? 2 * unit : 0;
        configure_cache(cpu.get(), ram, {1, 2}, experts);
        configure_cache(mixed.get(), ram, {1, 2}, experts);
        for (int token = 1; token <= tokens; ++token) {
            const bool decode = token != 5;
            const uint64_t epoch = token < 7 ? 41 : 42;
            begin_cache(cpu.get(), epoch, token, decode);
            begin_cache(mixed.get(), epoch, token, decode);
            uint64_t token_cpu_reads = 0, token_mixed_reads = 0;
            for (int layer : {1, 2}) {
                std::vector<int32_t> ids;
                for (int s = 0; s < std::min(6, experts); ++s) { ids.push_back((token * (s + 1) + layer * s) % experts); }
                std::vector<float> weights(ids.size(), 0.1f);
                weights[0] = 0.5f;
                const kimi_cpu_sink_activation activation = {KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f,
                    layer == 1 ? 0.75f : 10.0f};
                std::vector<float> input(latent);
                for (int i = 0; i < latent; ++i) { input[i] = 1.3f * std::sin(float(i * 7 + token * 31 + layer)); }
                std::vector<float> a(latent), b(latent);
                tensors meta(f, layer);
                for (int arm = 0; arm < 2; ++arm) {
                    kimi_cpu_sink_stats stats = {};
                    char error[512] = {};
                    require(kimi_cpu_sink_run_activation(arm ? mixed.get() : cpu.get(), meta.role[0], meta.role[1], meta.role[2],
                        input.data(), ids.data(), weights.data(), int(ids.size()), &activation, (arm ? b : a).data(),
                        &stats, error, sizeof(error)), error);
                    require(stats.read_errors == 0 && stats.pending_reads == 0 && stats.fallback_count == 0, "hybrid work did not drain");
                    if (decode) { (arm ? mixed_reads : cpu_reads) += stats.read_requests; }
                    (arm ? token_mixed_reads : token_cpu_reads) += stats.read_requests;
                    ++checks;
                }
                require(!std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), "VRAM tier changed an MXFP4 output bit");
                compared += a.size();
            }
            // Token 6 follows the prefill boundary at token 5: a retained arena keeps
            // serving hits, otherwise both arms start cold. Token 7 changes the
            // epoch, which releases the arena in every mode.
            if (token == 6) {
                require(cfg.keep ? token_mixed_reads < token_cpu_reads : token_mixed_reads == token_cpu_reads,
                    cfg.keep ? "retained VRAM served no hit after prefill" : "released VRAM served a hit after prefill");
                ++checks;
            }
            if (token == 7) {
                require(token_mixed_reads == token_cpu_reads, "VRAM hit survived an epoch change");
                ++checks;
            }
        }
        // A failing read beyond EOF after other work was queued must drain
        // uploads and return cleanly; the next decode token works again.
        begin_cache(mixed.get(), 42, uint64_t(tokens + 1), false);
        begin_cache(mixed.get(), 42, uint64_t(tokens + 2));
        f.fault = fixture::past_eof; f.fault_expert = experts - 1; f.fault_role = 2;
        tensors meta(f, 1);
        std::vector<int32_t> ids = {0, 1, experts - 1};
        std::vector<float> weights(3, 0.3f), input(latent, 0.1f), out(latent);
        const kimi_cpu_sink_activation activation = {KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 10.0f};
        char error[512] = {};
        kimi_cpu_sink_stats stats = {};
        require(!kimi_cpu_sink_run_activation(mixed.get(), meta.role[0], meta.role[1], meta.role[2], input.data(),
            ids.data(), weights.data(), 3, &activation, out.data(), &stats, error, sizeof(error)), "I/O failure accepted");
        require(stats.pending_reads == 0 && stats.fallback_count == 0, "I/O failure returned with live work");
        f.fault = fixture::none; f.fault_expert = f.fault_role = -1;
        begin_cache(mixed.get(), 42, uint64_t(tokens + 3), false);
        begin_cache(mixed.get(), 42, uint64_t(tokens + 4));
        require(kimi_cpu_sink_run_activation(mixed.get(), meta.role[0], meta.role[1], meta.role[2], input.data(),
            ids.data(), weights.data(), 3, &activation, out.data(), &stats, error, sizeof(error)), error);
        checks += 2;
        std::printf("MXFP4_TIER_MODE_PASS latent=%d hidden=%d exclusive=%d async=%d global=%d ram=%d keep=%d deferred=%d early_up_gate=%d role_major=%d nonblocking_down_sched=%d\n",
            latent, hidden, int(cfg.exclusive), int(cfg.async), int(cfg.global), int(cfg.ram), int(cfg.keep),
            int(cfg.deferred), int(cfg.early_up_gate), int(cfg.role_major), int(cfg.nonblocking_down_sched));
    }
    std::printf("MXFP4_TIER_SHAPE latent=%d hidden=%d cpu_reads=%llu mixed_reads=%llu\n", latent, hidden,
        (unsigned long long) cpu_reads, (unsigned long long) mixed_reads);
    require(mixed_reads < cpu_reads, "GPU tier served no hits; nothing was validated");
}

double rel_l2(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = double(a[i]) - double(b[i]);
        num += d * d;
        den += double(b[i]) * double(b[i]);
    }
    return den > 0 ? std::sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
}

// GGML_KIMI_CPU_SINK_VRAM_MXFP4_GPU_HIDDEN=1: layer 1 (SWIGLU_CLAMP, limits 0.75 and 10)
// runs its VRAM hits through the GPU chain, layer 2 (SiTU) keeps the bitwise bridge.
void run_chain_shape(fixture_inputs & inputs, int latent, int hidden, int experts, int tokens, size_t & checks, double & worst) {
    fixture f;
    f.type = f.down_type = GGML_TYPE_MXFP4;
    f.latent = latent; f.hidden = hidden; f.experts = experts;
    make_synthetic(f, uint32_t(20260917 + latent));
    const std::string gpu_bytes = std::to_string(uint64_t(2 * experts) * triplet_bytes(f));
    for (const char * name : {"vram_exclusive", "vram_async",
                              "tier.global", "tier.keep_across_requests",
                              "vram_deferred"}) {
        inputs.set(name, "0");
    }
    inputs.set("vram_budget", "0");
    inputs.set("early_up_gate", "0");
    inputs.set("role_major", "0");
    inputs.set("nonblocking_down_sched", "0");
    auto cpu = make_sink(inputs, f, 16, 2);
    inputs.set("vram_budget", gpu_bytes.c_str());
    inputs.set("vram_async", "1");
    inputs.set("tier.mxfp4_gpu_hidden", "1"); // read when the engine is created
    inputs.set("early_up_gate", "1");
    inputs.set("role_major", "1");
    inputs.set("nonblocking_down_sched", "1");
    auto mixed = make_sink(inputs, f, 16, 2);
    configure_cache(cpu.get(), 0, {1, 2}, experts);
    configure_cache(mixed.get(), 0, {1, 2}, experts);
    uint64_t cpu_reads = 0, mixed_reads = 0;
    for (int token = 1; token <= tokens; ++token) {
        begin_cache(cpu.get(), 61, uint64_t(token));
        begin_cache(mixed.get(), 61, uint64_t(token));
        for (int layer : {1, 2}) {
            std::vector<int32_t> ids;
            for (int s = 0; s < std::min(6, experts); ++s) { ids.push_back((token * (s + 1) + layer * s) % experts); }
            std::vector<float> weights(ids.size(), 0.1f);
            weights[0] = 0.5f;
            const bool situ = layer == 2;
            const kimi_cpu_sink_activation activation = situ ?
                kimi_cpu_sink_activation{KIMI_CPU_SINK_ACTIVATION_SITU, 4.0f, 25.0f, 0.0f} :
                kimi_cpu_sink_activation{KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, token % 2 ? 0.75f : 10.0f};
            std::vector<float> input(latent);
            for (int i = 0; i < latent; ++i) { input[i] = 1.3f * std::sin(float(i * 5 + token * 17 + layer)); }
            std::vector<float> a(latent), b(latent);
            tensors meta(f, layer);
            for (int arm = 0; arm < 2; ++arm) {
                kimi_cpu_sink_stats stats = {};
                char error[512] = {};
                require(kimi_cpu_sink_run_activation(arm ? mixed.get() : cpu.get(), meta.role[0], meta.role[1], meta.role[2],
                    input.data(), ids.data(), weights.data(), int(ids.size()), &activation, (arm ? b : a).data(),
                    &stats, error, sizeof(error)), error);
                require(stats.read_errors == 0 && stats.pending_reads == 0 && stats.fallback_count == 0, "chain work did not drain");
                (arm ? mixed_reads : cpu_reads) += stats.read_requests;
            }
            if (situ) {
                require(!std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), "SiTU left the bitwise bridge under the GPU hidden chain");
            } else {
                for (float v : b) { require(std::isfinite(v), "non-finite GPU chain output"); }
                const double rel = rel_l2(b, a);
                worst = std::max(worst, rel);
                require(rel <= 1e-3, "GPU chain output differs from the CPU sink by more than 1e-3 relative L2");
            }
            ++checks;
        }
    }
    inputs.set("tier.mxfp4_gpu_hidden", "0");
    std::printf("MXFP4_CHAIN_SHAPE_PASS latent=%d hidden=%d worst_rel_l2=%.3e cpu_reads=%llu mixed_reads=%llu\n", latent, hidden,
        worst, (unsigned long long) cpu_reads, (unsigned long long) mixed_reads);
    require(mixed_reads < cpu_reads, "GPU chain served no hits; nothing was validated");
}

// batch: the default batched GPU expert prefill, which rounds differently from the CPU; otherwise the
// per-expert path of GGML_KIMI_MXFP4_PREFILL_BATCH=0, which reproduces the CPU bitwise.
void run_grouped_prefill_hybrid(fixture_inputs & inputs, bool batch, size_t & compared, size_t & checks) {
    inputs.set("tier.mxfp4_batch", batch ? "1" : "0");
    fixture f;
    f.type = f.down_type = GGML_TYPE_MXFP4;
    f.latent = 512; f.hidden = 256; f.experts = 8;
    make_synthetic(f, 20260920);
    const std::string gpu_bytes = std::to_string(uint64_t(2 * f.experts) * triplet_bytes(f));
    inputs.set("vram_budget", "0");
    inputs.set("vram_prefill", "0");
    auto cpu = make_sink(inputs, f, 16, 2);
    inputs.set("vram_budget", gpu_bytes.c_str());
    inputs.set("tier.keep_across_requests", "1");
    inputs.set("vram_prefill", "1");
    inputs.set("tier.global", "1");
    inputs.set("tier.mxfp4_gpu_hidden", "0");
    auto mixed = make_sink(inputs, f, 16, 2);
    configure_cache(cpu.get(), 0, {1, 2}, f.experts);
    configure_cache(mixed.get(), 0, {1, 2}, f.experts);
    tensors meta(f, 1);
    const kimi_cpu_sink_activation activation = {
        KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 0.75f,
    };
    std::vector<int32_t> all_ids(size_t(f.experts));
    for (int i = 0; i < f.experts; ++i) { all_ids[size_t(i)] = i; }
    std::vector<float> all_weights(size_t(f.experts), 1.0f / float(f.experts));
    std::vector<float> warm_input(size_t(f.latent), 0.125f), warm_output(size_t(f.latent));
    begin_cache(mixed.get(), 81, 1);
    kimi_cpu_sink_stats warm_stats = {}; char error[512] = {};
    require(kimi_cpu_sink_run_activation(mixed.get(), meta.role[0], meta.role[1], meta.role[2],
        warm_input.data(), all_ids.data(), all_weights.data(), f.experts, &activation,
        warm_output.data(), &warm_stats, error, sizeof(error)), error);
    require(warm_stats.read_requests == uint64_t(3 * f.experts), "VRAM warm pass did not read every expert");

    const int ntokens = 3, nslots = 6;
    std::vector<float> input(size_t(ntokens) * size_t(f.latent));
    for (size_t i = 0; i < input.size(); ++i) { input[i] = 1.1f * std::sin(float(i * 3 + 17)); }
    std::vector<int32_t> ids(size_t(ntokens) * nslots);
    std::vector<float> weights(ids.size());
    for (int token = 0; token < ntokens; ++token) {
        for (int slot = 0; slot < nslots; ++slot) {
            ids[size_t(token) * nslots + slot] = (token * 3 + slot) % f.experts;
            weights[size_t(token) * nslots + slot] = float(nslots - slot) / 21.0f;
        }
    }
    std::vector<float> expected;
    for (int token = 0; token < ntokens; ++token) {
        const std::vector<float> token_input(input.begin() + size_t(token) * f.latent,
                input.begin() + size_t(token + 1) * f.latent);
        const std::vector<int32_t> token_ids(ids.begin() + size_t(token) * nslots,
                ids.begin() + size_t(token + 1) * nslots);
        const std::vector<float> token_weights(weights.begin() + size_t(token) * nslots,
                weights.begin() + size_t(token + 1) * nslots);
        const auto row = reference_moe_graph(f, 1, token_input, token_ids, token_weights, activation, 8);
        expected.insert(expected.end(), row.begin(), row.end());
    }
    std::vector<float> baseline(expected.size()), hybrid(expected.size());
    char create_error[512] = {};
    std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)> cpu_prefill(
        kimi_cpu_prefill_create(cpu.get(), create_error, sizeof(create_error)), kimi_cpu_prefill_destroy);
    require(cpu_prefill != nullptr, create_error);
    std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)> mixed_prefill(
        kimi_cpu_prefill_create(mixed.get(), create_error, sizeof(create_error)), kimi_cpu_prefill_destroy);
    require(mixed_prefill != nullptr, create_error);
    kimi_cpu_prefill_stats cpu_stats = {}, hybrid_stats = {};
    require(kimi_cpu_prefill_run_activation(cpu_prefill.get(), meta.role[0], meta.role[1], meta.role[2],
        input.data(), ids.data(), weights.data(), nslots, ntokens, &activation, baseline.data(),
        &cpu_stats, error, sizeof(error)), error);
    require(kimi_cpu_prefill_run_activation(mixed_prefill.get(), meta.role[0], meta.role[1], meta.role[2],
        input.data(), ids.data(), weights.data(), nslots, ntokens, &activation, hybrid.data(),
        &hybrid_stats, error, sizeof(error)), error);
    require(!std::memcmp(expected.data(), baseline.data(), expected.size() * sizeof(float)),
            "CPU grouped prefill differs from its graph reference");
    double rel = 0;
    if (batch) {
        for (float v : hybrid) { require(std::isfinite(v), "non-finite batched grouped prefill output"); }
        rel = rel_l2(hybrid, expected);
        require(rel <= 1e-5, "batched retained-VRAM grouped prefill differs from its graph reference by more than 1e-5 relative L2");
    } else {
        require(!std::memcmp(expected.data(), hybrid.data(), expected.size() * sizeof(float)),
                "retained-VRAM grouped prefill differs from its graph reference");
    }
    require(hybrid_stats.gpu_experts == uint64_t(f.experts) && hybrid_stats.gpu_routes == ids.size() &&
            hybrid_stats.ram_routes == 0 && hybrid_stats.ssd_routes == 0 &&
            cpu_stats.ssd_routes == ids.size() &&
            hybrid_stats.stream.read_requests == 0 && cpu_stats.stream.read_requests == uint64_t(3 * f.experts),
            "grouped prefill did not replace resident SSD reads with GPU work");
    compared += expected.size(); checks += 4;
    std::printf("MXFP4_GROUPED_PREFILL_HYBRID_PASS batch=%d rel_l2=%.3e tokens=%d routes=%zu gpu_experts=%llu gpu_routes=%llu cpu_reads=%llu hybrid_reads=%llu\n",
        int(batch), rel, ntokens, ids.size(), (unsigned long long) hybrid_stats.gpu_experts,
        (unsigned long long) hybrid_stats.gpu_routes, (unsigned long long) cpu_stats.stream.read_requests,
        (unsigned long long) hybrid_stats.stream.read_requests);
    inputs.set("vram_prefill", "0");
    inputs.unset("tier.mxfp4_batch");
}

// A cold tier: the makespan rule streams SSD transients to the GPU while the CPU computes the rest, both lanes reading
// through a two-expert ring in the merged lane order. A GPU fixed cost of 2 ms in the start values keeps both lanes
// busy; the second pass runs on the terms calibrated by the first. Lanes far shorter than the drive keep every cold
// expert on the CPU: the drive's line comes from the decode step's reads (the cost table fits it over decode's and
// prefill's layers, so a start value no longer outlives them), and a CPU of 1 ns an expert (start values) is far
// below it. The output stays within the batched path's rounding of the graph.
void run_grouped_prefill_cold_transients(fixture_inputs & inputs, size_t & compared, size_t & checks) {
    for (const bool drive_bound : {false, true}) {
        inputs.set("tier.mxfp4_batch", "1");
        inputs.set("place_costs", drive_bound ? "cpu_fixed_us=0.001,cpu_route_us=0.000001" : "gpu_fixed_us=2000");
        fixture f;
        f.type = f.down_type = GGML_TYPE_MXFP4;
        f.latent = 512; f.hidden = 256; f.experts = 16;
        make_synthetic(f, 20261020);
        inputs.set("vram_budget", std::to_string(uint64_t(2 * f.experts) * triplet_bytes(f)).c_str());
        inputs.set("tier.keep_across_requests", "1");
        inputs.set("vram_prefill", "1");
        inputs.set("tier.global", "1");
        inputs.set("tier.mxfp4_gpu_hidden", "0");
        auto mixed = make_sink(inputs, f, 4, 2);
        configure_cache(mixed.get(), 0, {1, 2}, f.experts);
        tensors meta(f, 1);
        const kimi_cpu_sink_activation activation = {
            KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 0.75f,
        };
        // One decode step gives the tier its format and two residents; the other fourteen experts stay cold.
        std::vector<int32_t> warm_ids = {0, 1};
        std::vector<float> warm_weights(2, 0.5f), warm_input(size_t(f.latent), 0.125f), warm_output(size_t(f.latent));
        begin_cache(mixed.get(), 91, 1);
        kimi_cpu_sink_stats warm_stats = {}; char error[512] = {};
        require(kimi_cpu_sink_run_activation(mixed.get(), meta.role[0], meta.role[1], meta.role[2],
            warm_input.data(), warm_ids.data(), warm_weights.data(), 2, &activation,
            warm_output.data(), &warm_stats, error, sizeof(error)), error);

        // Uneven routes: slot 0 cycles over three experts (heavy), the others spread over all sixteen.
        const int ntokens = 48, nslots = 6;
        std::vector<float> input(size_t(ntokens) * size_t(f.latent));
        for (size_t i = 0; i < input.size(); ++i) { input[i] = 0.9f * std::sin(float(i * 5 + 3)); }
        std::vector<int32_t> ids(size_t(ntokens) * nslots);
        std::vector<float> weights(ids.size());
        for (int token = 0; token < ntokens; ++token) {
            for (int slot = 0; slot < nslots; ++slot) {
                int id = slot == 0 ? 2 + token % 3 : (token * 7 + slot * 5) % f.experts;
                for (int previous = 0; previous < slot; ++previous) {
                    if (ids[size_t(token) * nslots + previous] == id) { id = (id + 1) % f.experts; previous = -1; }
                }
                ids[size_t(token) * nslots + slot] = id;
                weights[size_t(token) * nslots + slot] = float(nslots - slot) / 21.0f;
            }
        }
        std::vector<float> expected;
        for (int token = 0; token < ntokens; ++token) {
            const std::vector<float> token_input(input.begin() + size_t(token) * f.latent,
                    input.begin() + size_t(token + 1) * f.latent);
            const std::vector<int32_t> token_ids(ids.begin() + size_t(token) * nslots,
                    ids.begin() + size_t(token + 1) * nslots);
            const std::vector<float> token_weights(weights.begin() + size_t(token) * nslots,
                    weights.begin() + size_t(token + 1) * nslots);
            const auto row = reference_moe_graph(f, 1, token_input, token_ids, token_weights, activation, 8);
            expected.insert(expected.end(), row.begin(), row.end());
        }
        char create_error[512] = {};
        std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)> prefill(
            kimi_cpu_prefill_create(mixed.get(), create_error, sizeof(create_error)), kimi_cpu_prefill_destroy);
        require(prefill != nullptr, create_error);
        double worst = 0;
        for (int pass = 0; pass < (drive_bound ? 1 : 2); ++pass) {
            std::vector<float> hybrid(expected.size());
            kimi_cpu_prefill_stats stats = {};
            require(kimi_cpu_prefill_run_activation(prefill.get(), meta.role[0], meta.role[1], meta.role[2],
                input.data(), ids.data(), weights.data(), nslots, ntokens, &activation, hybrid.data(),
                &stats, error, sizeof(error)), error);
            for (float v : hybrid) { require(std::isfinite(v), "non-finite cold-transient grouped prefill output"); }
            const double rel = rel_l2(hybrid, expected);
            require(rel <= 1e-5, "cold-transient grouped prefill differs from its graph reference by more than 1e-5 relative L2");
            if (drive_bound) {
                require(stats.gpu_ssd_transient_experts == 0 && stats.ssd_experts > 0,
                        "a drive-bound grouped prefill moved cold experts to the GPU");
            } else {
                // The start values split the cold experts; the calibrated pass may legitimately move the split.
                require(pass || (stats.gpu_ssd_transient_experts > 0 && stats.gpu_ssd_transient_experts < stats.ssd_experts),
                        "cold-transient grouped prefill did not split the SSD experts between the GPU and the CPU");
            }
            require(stats.stream.pending_reads == 0, "cold-transient grouped prefill returned with live reads");
            worst = std::max(worst, rel);
            compared += expected.size(); checks += 4;
            std::printf("MXFP4_GROUPED_PREFILL_COLD_TRANSIENTS_PASS case=%s pass=%d rel_l2=%.3e gpu_experts=%llu "
                "ssd_gpu_experts=%llu ssd_experts=%llu\n", drive_bound ? "drive_bound" : "split", pass, rel,
                (unsigned long long) stats.gpu_experts, (unsigned long long) stats.gpu_ssd_transient_experts,
                (unsigned long long) stats.ssd_experts);
        }
        inputs.set("vram_prefill", "0");
        inputs.unset("place_costs");
        inputs.unset("tier.mxfp4_batch");
    }
}
}

int main() {
    fixture_inputs inputs;
    try {
        ggml_cpu_init();
        inputs.set("ready_first", "1");
        inputs.set("split_cv", "1");
        inputs.set("cache_policy", "lru");
        inputs.set("cache_allocation", "uniform");
        inputs.set("cache_keep", "0");
        inputs.set("tier.policy", "lru");
        inputs.set("tier.min_count", "0");
        inputs.set("tier.mxfp4_gpu_hidden", "0");
        inputs.set("early_up_gate", "0");
        inputs.set("role_major", "0");
        inputs.set("nonblocking_down_sched", "0");
        size_t compared = 0, checks = 0;
        // Mismatch path first; the shape runs below then create fresh engines
        // whose self-check passes against the linked CPU kernel.
        selfcheck_mismatch(inputs, "scalar_order", scalar_order_dot, "check=up_gate_c512/", checks);
        flipped_columns = 256;
        selfcheck_mismatch(inputs, "one_bit_down_stage", flipped_dot, "check=down_c256/scales input=0 row=0 ", checks);
        selfcheck_skipped_with_chain(inputs, checks);
        run_shape(inputs, 512, 256, 8, 12, compared, checks);
        run_shape(inputs, 5120, 2304, 4, 8, compared, checks);
        double chain_worst = 0;
        run_chain_shape(inputs, 512, 256, 8, 10, checks, chain_worst);
        run_chain_shape(inputs, 5120, 2304, 4, 8, checks, chain_worst);
        run_grouped_prefill_hybrid(inputs, false, compared, checks);
        run_grouped_prefill_hybrid(inputs, true, compared, checks);
        run_grouped_prefill_cold_transients(inputs, compared, checks);
        // last: it sets the tier's environment for itself
        run_host_jobs(inputs, GGML_TYPE_MXFP4, checks);
        std::printf("REAL_GPU_MXFP4_TIER_PASS compared=%zu assertions=%zu chain_worst_rel_l2=%.3e\n", compared, checks, chain_worst);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "REAL_GPU_MXFP4_TIER_FAIL: %s\n", e.what());
        return 1;
    }
}
