// End-to-end IQ2_XXS/Q2_K CPU-sink test with the real VRAM tier (run_shape also covers the generic engine's other
// formats: Q4_K/Q5_K as in Muse Spark UD-Q4_K_XL, and Q8_0). Synthetic
// packed weights cover non-uniform padded role sizes and mixed CPU/GPU expert
// reduction without requiring the full model. The CUDA path intentionally uses
// Q8_1 activations and is checked numerically, not bitwise.
#define main original_sink_test_main
#include "test-kimi-cpu-sink.cpp"
#undef main
#include "kimi-host-jobs-test.inc"

namespace {
double relative_l2(const std::vector<float> & got, const std::vector<float> & expected) {
    double error = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double delta = double(got[i]) - double(expected[i]);
        error += delta * delta;
        norm += double(expected[i]) * double(expected[i]);
    }
    return norm > 0.0 ? std::sqrt(error / norm) : std::sqrt(error);
}

double max_abs_error(const std::vector<float> & got, const std::vector<float> & expected) {
    double result = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        result = std::max(result, std::abs(double(got[i]) - double(expected[i])));
    }
    return result;
}

struct mode {
    bool production;
    const char * name;
};

void set_switch(const char * name, bool enabled) {
    setenv(name, enabled ? "1" : "0", 1);
}

void run_shape(int latent, int hidden, int experts, mode selected, size_t & checks,
        double & worst_rel, double & worst_abs,
        ggml_type up_type = GGML_TYPE_IQ2_XXS, ggml_type down_type = GGML_TYPE_Q2_K) {
    fixture f;
    f.type = up_type;
    f.down_type = down_type;
    f.latent = latent;
    f.hidden = hidden;
    f.experts = experts;
    make_synthetic(f, uint32_t(20260920 + latent + int(selected.production)));
    const uint64_t unit = triplet_bytes(f);

    setenv("GGML_KIMI_CPU_SINK_VRAM_BYTES", "0", 1);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_GLOBAL_POOL", false);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_EXCLUSIVE", false);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_ASYNC_UPLOAD", false);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_DEFERRED_UPLOAD", false);
    set_switch("GGML_KIMI_CPU_SINK_EARLY_UP_GATE", false);
    set_switch("GGML_KIMI_CPU_SINK_ROLE_MAJOR", false);
    set_switch("GGML_KIMI_CPU_SINK_NONBLOCKING_DOWN_SCHED", false);
    setenv("GGML_KIMI_CPU_PREFILL_Q2_SSD_GPU_MIN_ROUTES", "0", 1);
    setenv("GGML_KIMI_CPU_PREFILL_Q2_SSD_GPU_MAX_EXPERTS_PER_LAYER", "0", 1);
    auto cpu = make_sink(f, 9, 8);

    const uint64_t vram_bytes = selected.production ? uint64_t(2 * experts) * unit : 2 * unit;
    setenv("GGML_KIMI_CPU_SINK_VRAM_BYTES", std::to_string(vram_bytes).c_str(), 1);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_GLOBAL_POOL", selected.production);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_EXCLUSIVE", selected.production);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_ASYNC_UPLOAD", selected.production);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_DEFERRED_UPLOAD", selected.production);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_KEEP_ACROSS_REQUESTS", true);
    set_switch("GGML_KIMI_CPU_SINK_EARLY_UP_GATE", selected.production);
    set_switch("GGML_KIMI_CPU_SINK_ROLE_MAJOR", selected.production);
    set_switch("GGML_KIMI_CPU_SINK_NONBLOCKING_DOWN_SCHED", selected.production);
    auto mixed = make_sink(f, 9, 8);

    const uint64_t ram_bytes = selected.production ? 2 * unit : 0;
    configure_cache(cpu.get(), ram_bytes, {1, 2}, experts);
    configure_cache(mixed.get(), ram_bytes, {1, 2}, experts);
    uint64_t cpu_reads = 0;
    uint64_t mixed_reads = 0;
    const kimi_cpu_sink_activation activation = {
        KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 0.75f
    };

    for (int token = 1; token <= 4; ++token) {
        begin_cache(cpu.get(), 91, uint64_t(token));
        begin_cache(mixed.get(), 91, uint64_t(token));
        for (int layer : {1, 2}) {
            const int used = std::min(4, experts);
            std::vector<int32_t> ids;
            std::vector<float> weights;
            for (int i = 0; i < used; ++i) {
                ids.push_back((i + layer - 1) % experts);
                weights.push_back(i == 0 ? 0.4f : 0.6f / float(used - 1));
            }
            std::vector<float> input(static_cast<size_t>(latent));
            for (int i = 0; i < latent; ++i) {
                input[size_t(i)] = 1.7f * std::sin(float(i * 11 + token * 23 + layer * 5));
            }
            std::vector<float> expected(static_cast<size_t>(latent));
            std::vector<float> got(static_cast<size_t>(latent));
            tensors meta(f, layer);
            for (int arm = 0; arm < 2; ++arm) {
                kimi_cpu_sink_stats stats = {};
                char error[512] = {};
                const bool ok = kimi_cpu_sink_run_activation(arm ? mixed.get() : cpu.get(),
                    meta.role[0], meta.role[1], meta.role[2], input.data(), ids.data(), weights.data(),
                    int(ids.size()), &activation, (arm ? got : expected).data(), &stats, error, sizeof(error));
                require(ok, std::string("Q2 sink run failed mode=") + selected.name +
                    " token=" + std::to_string(token) + " layer=" + std::to_string(layer) +
                    " arm=" + std::to_string(arm) + ": " + error);
                require(stats.read_errors == 0 && stats.pending_reads == 0 && stats.fallback_count == 0,
                    "Q2 VRAM work did not drain");
                (arm ? mixed_reads : cpu_reads) += stats.read_requests;
                ++checks;
            }
            for (float value : got) {
                require(std::isfinite(value), "Q2 VRAM output is non-finite");
            }
            const double rel = relative_l2(got, expected);
            const double abs = max_abs_error(got, expected);
            worst_rel = std::max(worst_rel, rel);
            worst_abs = std::max(worst_abs, abs);
            require(rel <= 2e-2, "Q2 VRAM output exceeds 2e-2 relative L2");
            ++checks;
        }
    }
    require(mixed_reads < cpu_reads, "Q2 VRAM tier served no resident expert hits");
    ++checks;
    std::printf("Q2_VRAM_SHAPE_PASS mode=%s types=%s/%s latent=%d hidden=%d cpu_reads=%llu mixed_reads=%llu worst_rel_l2=%.6g worst_abs=%.6g\n",
        selected.name, ggml_type_name(up_type), ggml_type_name(down_type), latent, hidden, (unsigned long long) cpu_reads,
        (unsigned long long) mixed_reads, worst_rel, worst_abs);
}

void run_ram_prefill_scheduler(size_t & checks, double & worst_rel) {
    fixture f;
    f.type = GGML_TYPE_IQ2_XXS;
    f.down_type = GGML_TYPE_Q2_K;
    f.latent = 512;
    f.hidden = 256;
    f.experts = 4;
    make_synthetic(f, 20260921);
    const uint64_t unit = triplet_bytes(f);
    setenv("GGML_KIMI_CPU_SINK_CACHE_PREFILL_READ", "1", 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_PREFILL", "0", 1);
    setenv("GGML_KIMI_CPU_PREFILL_Q2_RAM_GPU_MIN_ROUTES", "0", 1);
    setenv("GGML_KIMI_CPU_PREFILL_Q2_SSD_GPU_MIN_ROUTES", "0", 1);
    setenv("GGML_KIMI_CPU_PREFILL_Q2_SSD_GPU_MAX_EXPERTS_PER_LAYER", "0", 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_BYTES", "0", 1);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_EXCLUSIVE", false);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_CACHE_LOCATIONS", false);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_ASYNC_UPLOAD", false);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_DEFERRED_UPLOAD", false);
    auto baseline = make_sink(f, 9, 8);
    configure_cache(baseline.get(), 0, {1}, 1);

    setenv("GGML_KIMI_CPU_SINK_VRAM_PREFILL", "1", 1);
    setenv("GGML_KIMI_CPU_PREFILL_Q2_RAM_GPU_MIN_ROUTES", "16", 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_BYTES", std::to_string(unit).c_str(), 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_GLOBAL_POOL", "0", 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_ASYNC_UPLOAD", "0", 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_DEFERRED_UPLOAD", "0", 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_KEEP_ACROSS_REQUESTS", "1", 1);
    auto scheduled = make_sink(f, 9, 8);
    configure_cache(scheduled.get(), 2 * unit, {1}, 2);
    tensors meta(f, 1);
    const kimi_cpu_sink_activation activation = {
        KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 10.0f,
    };
    char error[512] = {};
    std::vector<float> one_input(size_t(f.latent), 0.125f), one_output(size_t(f.latent));
    for (int expert : {0, 1}) {
        begin_cache(scheduled.get(), 121, uint64_t(expert + 1));
        const float weight = 1.0f;
        kimi_cpu_sink_stats seed = {};
        require(kimi_cpu_sink_run_activation(scheduled.get(), meta.role[0], meta.role[1], meta.role[2],
            one_input.data(), &expert, &weight, 1, &activation, one_output.data(), &seed,
            error, sizeof(error)), error);
        require(seed.cache_admitted == 1, "Q2 transient fixture did not seed RAM");
    }
    const auto policy_before = cache_snapshot(scheduled.get());

    constexpr int tokens = 16;
    std::vector<float> input(size_t(tokens) * size_t(f.latent));
    for (size_t i = 0; i < input.size(); ++i) input[i] = 0.4f * std::sin(float(i) * 0.013f);
    std::vector<int32_t> ids(tokens, 0);
    std::vector<float> weights(tokens, 1.0f);
    std::vector<float> expected(input.size()), actual(input.size());
    std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)> cpu_prefill(
        kimi_cpu_prefill_create(baseline.get(), error, sizeof(error)), kimi_cpu_prefill_destroy);
    std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)> gpu_prefill(
        kimi_cpu_prefill_create(scheduled.get(), error, sizeof(error)), kimi_cpu_prefill_destroy);
    require(cpu_prefill && gpu_prefill, error);
    kimi_cpu_prefill_stats cpu_stats = {}, gpu_stats = {};
    require(kimi_cpu_prefill_run_activation(cpu_prefill.get(), meta.role[0], meta.role[1], meta.role[2],
        input.data(), ids.data(), weights.data(), 1, tokens, &activation, expected.data(),
        &cpu_stats, error, sizeof(error)), error);
    require(kimi_cpu_prefill_run_activation(gpu_prefill.get(), meta.role[0], meta.role[1], meta.role[2],
        input.data(), ids.data(), weights.data(), 1, tokens, &activation, actual.data(),
        &gpu_stats, error, sizeof(error)), error);
    const double rel = relative_l2(actual, expected);
    worst_rel = std::max(worst_rel, rel);
    require(rel <= 2e-2 && gpu_stats.gpu_transient_experts == 1 &&
        gpu_stats.gpu_transient_routes == tokens && gpu_stats.gpu_transient_upload_bytes == unit &&
        gpu_stats.ram_experts == 1 && gpu_stats.ram_routes == tokens &&
        gpu_stats.ssd_experts == 0 && gpu_stats.ssd_routes == 0 && gpu_stats.ssd_bytes == 0,
        "Q2 RAM-to-GPU grouped-prefill selection or result differs");
    const auto policy_after = cache_snapshot(scheduled.get());
    require(policy_after.hits == policy_before.hits && policy_after.misses == policy_before.misses &&
        policy_after.admitted_bytes == policy_before.admitted_bytes && policy_after.evictions == policy_before.evictions,
        "Q2 transient prefill trained the decode RAM policy");

    begin_cache(scheduled.get(), 121, 3);
    const int expert = 0;
    const float weight = 1.0f;
    kimi_cpu_sink_stats decode = {};
    require(kimi_cpu_sink_run_activation(scheduled.get(), meta.role[0], meta.role[1], meta.role[2],
        one_input.data(), &expert, &weight, 1, &activation, one_output.data(), &decode,
        error, sizeof(error)), error);
    require(decode.cache_hits == 1, "Q2 transient prefill did not preserve the RAM entry for decode");
    checks += 5;
    std::printf("Q2_RAM_GPU_PREFILL_PASS tokens=%d transient_experts=%llu transient_routes=%llu upload_bytes=%llu rel_l2=%.6g decode_ram_hits=%llu\n",
        tokens, (unsigned long long)gpu_stats.gpu_transient_experts,
        (unsigned long long)gpu_stats.gpu_transient_routes,
        (unsigned long long)gpu_stats.gpu_transient_upload_bytes, rel,
        (unsigned long long)decode.cache_hits);
}

void run_ssd_prefill_scheduler(size_t & checks, double & worst_rel) {
    fixture f;
    f.type = GGML_TYPE_IQ2_XXS;
    f.down_type = GGML_TYPE_Q2_K;
    f.latent = 512;
    f.hidden = 256;
    f.experts = 4;
    make_synthetic(f, 20260923);
    const uint64_t unit = triplet_bytes(f);
    setenv("GGML_KIMI_CPU_SINK_CACHE_PREFILL_READ", "1", 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_PREFILL", "0", 1);
    setenv("GGML_KIMI_CPU_PREFILL_Q2_RAM_GPU_MIN_ROUTES", "0", 1);
    setenv("GGML_KIMI_CPU_PREFILL_Q2_SSD_GPU_MIN_ROUTES", "0", 1);
    setenv("GGML_KIMI_CPU_PREFILL_Q2_SSD_GPU_MAX_EXPERTS_PER_LAYER", "0", 1);
    setenv("GGML_KIMI_CPU_PREFILL_Q2_COLD_ADMIT_PER_LAYER", "0", 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_BYTES", "0", 1);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_EXCLUSIVE", false);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_CACHE_LOCATIONS", false);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_ASYNC_UPLOAD", false);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_DEFERRED_UPLOAD", false);
    auto baseline = make_sink(f, 9, 8);
    configure_cache(baseline.get(), 0, {1}, 1);

    setenv("GGML_KIMI_CPU_SINK_VRAM_PREFILL", "1", 1);
    setenv("GGML_KIMI_CPU_PREFILL_Q2_SSD_GPU_MIN_ROUTES", "16", 1);
    setenv("GGML_KIMI_CPU_PREFILL_Q2_SSD_GPU_MAX_EXPERTS_PER_LAYER", "1", 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_BYTES", std::to_string(4 * unit).c_str(), 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_GLOBAL_POOL", "0", 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_KEEP_ACROSS_REQUESTS", "1", 1);
    auto scheduled = make_sink(f, 9, 8);
    configure_cache(scheduled.get(), 0, {1}, 1);

    tensors meta(f, 1);
    const kimi_cpu_sink_activation activation = {
        KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 10.0f,
    };
    constexpr int tokens = 16;
    std::vector<float> input(size_t(tokens) * size_t(f.latent));
    for (size_t i = 0; i < input.size(); ++i) input[i] = 0.35f * std::sin(float(i) * 0.019f);
    std::vector<int32_t> ids(tokens, 0);
    std::vector<float> weights(tokens, 1.0f);
    std::vector<float> expected(input.size()), actual(input.size());
    char error[512] = {};
    std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)> cpu_prefill(
        kimi_cpu_prefill_create(baseline.get(), error, sizeof(error)), kimi_cpu_prefill_destroy);
    std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)> gpu_prefill(
        kimi_cpu_prefill_create(scheduled.get(), error, sizeof(error)), kimi_cpu_prefill_destroy);
    require(cpu_prefill && gpu_prefill, error);
    kimi_cpu_prefill_stats cpu_stats = {}, gpu_stats = {};
    require(kimi_cpu_prefill_run_activation(cpu_prefill.get(), meta.role[0], meta.role[1], meta.role[2],
        input.data(), ids.data(), weights.data(), 1, tokens, &activation, expected.data(),
        &cpu_stats, error, sizeof(error)), error);
    require(kimi_cpu_prefill_run_activation(gpu_prefill.get(), meta.role[0], meta.role[1], meta.role[2],
        input.data(), ids.data(), weights.data(), 1, tokens, &activation, actual.data(),
        &gpu_stats, error, sizeof(error)), error);
    const double rel = relative_l2(actual, expected);
    worst_rel = std::max(worst_rel, rel);
    require(rel <= 2e-2 && gpu_stats.gpu_ssd_transient_candidates == 1 &&
        gpu_stats.gpu_ssd_transient_experts == 1 && gpu_stats.gpu_ssd_transient_routes == tokens &&
        gpu_stats.gpu_ssd_transient_upload_bytes == unit &&
        gpu_stats.gpu_transient_experts == 0 && gpu_stats.ram_experts == 0 &&
        gpu_stats.ssd_experts == 1 && gpu_stats.ssd_routes == tokens && gpu_stats.ssd_bytes == unit,
        "Q2 SSD-to-GPU grouped-prefill selection, accounting, or result differs");
    require(gpu_stats.stream.read_requests == 3 && gpu_stats.stream.read_errors == 0 &&
        gpu_stats.stream.pending_reads == 0 && gpu_stats.stream.fallback_count == 0,
        "Q2 SSD-to-GPU staged O_DIRECT work did not drain");
    checks += 3;
    std::printf("Q2_SSD_GPU_PREFILL_PASS tokens=%d candidates=%llu transient_experts=%llu transient_routes=%llu upload_bytes=%llu ssd_bytes=%llu rel_l2=%.6g\n",
        tokens, (unsigned long long)gpu_stats.gpu_ssd_transient_candidates,
        (unsigned long long)gpu_stats.gpu_ssd_transient_experts,
        (unsigned long long)gpu_stats.gpu_ssd_transient_routes,
        (unsigned long long)gpu_stats.gpu_ssd_transient_upload_bytes,
        (unsigned long long)gpu_stats.ssd_bytes, rel);
}

void run_cold_prefill_admission(size_t & checks, double & worst_rel) {
    fixture f;
    f.type = GGML_TYPE_IQ2_XXS;
    f.down_type = GGML_TYPE_Q2_K;
    f.latent = 512;
    f.hidden = 256;
    f.experts = 4;
    make_synthetic(f, 20260922);
    const uint64_t unit = triplet_bytes(f);
    setenv("GGML_KIMI_CPU_SINK_CACHE_PREFILL_READ", "1", 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_PREFILL", "0", 1);
    setenv("GGML_KIMI_CPU_PREFILL_Q2_RAM_GPU_MIN_ROUTES", "0", 1);
    setenv("GGML_KIMI_CPU_PREFILL_Q2_SSD_GPU_MIN_ROUTES", "0", 1);
    setenv("GGML_KIMI_CPU_PREFILL_Q2_SSD_GPU_MAX_EXPERTS_PER_LAYER", "0", 1);
    setenv("GGML_KIMI_CPU_PREFILL_Q2_COLD_ADMIT_PER_LAYER", "0", 1);
    setenv("GGML_KIMI_CPU_PREFILL_Q2_COLD_ADMIT_FAIL_AT", "0", 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_BYTES", "0", 1);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_EXCLUSIVE", false);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_ASYNC_UPLOAD", false);
    set_switch("GGML_KIMI_CPU_SINK_VRAM_DEFERRED_UPLOAD", false);
    auto baseline = make_sink(f, 9, 8);
    configure_cache(baseline.get(), 0, {1}, 1);

    setenv("GGML_KIMI_CPU_SINK_VRAM_PREFILL", "1", 1);
    setenv("GGML_KIMI_CPU_PREFILL_Q2_COLD_ADMIT_PER_LAYER", "1", 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_BYTES", std::to_string(4 * unit).c_str(), 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_POLICY", "count_decay2", 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_MIN_COUNT", "2.5", 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_GLOBAL_POOL", "1", 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_KEEP_ACROSS_REQUESTS", "1", 1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_ELASTIC", "0", 1);
    auto admitted = make_sink(f, 9, 8);
    configure_cache(admitted.get(), 0, {1}, 1);

    tensors meta(f, 1);
    const kimi_cpu_sink_activation activation = {
        KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 10.0f,
    };
    constexpr int tokens = 16;
    std::vector<float> input(size_t(tokens) * size_t(f.latent));
    for (size_t i = 0; i < input.size(); ++i) input[i] = 0.3f * std::cos(float(i) * 0.017f);
    std::vector<int32_t> ids(tokens, 0);
    std::vector<float> weights(tokens, 1.0f);
    std::vector<float> expected(input.size()), actual(input.size());
    char error[512] = {};
    std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)> cpu_prefill(
        kimi_cpu_prefill_create(baseline.get(), error, sizeof(error)), kimi_cpu_prefill_destroy);
    std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)> admitted_prefill(
        kimi_cpu_prefill_create(admitted.get(), error, sizeof(error)), kimi_cpu_prefill_destroy);
    require(cpu_prefill && admitted_prefill, error);
    kimi_cpu_prefill_stats cpu_stats = {}, admitted_stats = {};
    require(kimi_cpu_prefill_run_activation(cpu_prefill.get(), meta.role[0], meta.role[1], meta.role[2],
        input.data(), ids.data(), weights.data(), 1, tokens, &activation, expected.data(),
        &cpu_stats, error, sizeof(error)), error);
    require(kimi_cpu_prefill_run_activation(admitted_prefill.get(), meta.role[0], meta.role[1], meta.role[2],
        input.data(), ids.data(), weights.data(), 1, tokens, &activation, actual.data(),
        &admitted_stats, error, sizeof(error)), error);
    require(!std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)) &&
        admitted_stats.gpu_cold_admit_candidates == 1 && admitted_stats.gpu_cold_admit_experts == 1 &&
        admitted_stats.gpu_cold_admit_routes == tokens && admitted_stats.gpu_cold_admit_upload_bytes == unit,
        "Q2 cold-prefill admission changed prefill output or counters");

    // The server restores the one-token decode graph before begin_cache().
    // This callback must grow the cold arena while it is quiescent; the later
    // request-end callback must then be a no-op rather than trying to grow
    // around live/deferred decode work.
    require(kimi_cpu_sink_vram_phase_complete(admitted.get(), true, error, sizeof(error)), error);
    begin_cache(admitted.get(), 20260922, 1);
    std::vector<float> decode_input(size_t(f.latent), 0.125f), decode_expected(size_t(f.latent)), decode_actual(size_t(f.latent));
    const int expert = 0;
    const float weight = 1.0f;
    kimi_cpu_sink_stats reference_stats = {}, decode_stats = {};
    require(kimi_cpu_sink_run_activation(baseline.get(), meta.role[0], meta.role[1], meta.role[2],
        decode_input.data(), &expert, &weight, 1, &activation, decode_expected.data(),
        &reference_stats, error, sizeof(error)), error);
    require(kimi_cpu_sink_run_activation(admitted.get(), meta.role[0], meta.role[1], meta.role[2],
        decode_input.data(), &expert, &weight, 1, &activation, decode_actual.data(),
        &decode_stats, error, sizeof(error)), error);
    const double rel = relative_l2(decode_actual, decode_expected);
    worst_rel = std::max(worst_rel, rel);
    require(decode_stats.read_requests == 0 && rel <= 2e-2,
        "Q2 cold-prefill seed was not retained for immediate decode");
    require(kimi_cpu_sink_vram_phase_complete(admitted.get(), true, error, sizeof(error)), error);

    // Inject the first cold H2D failure. The public prefill call must fail
    // closed after draining all O_DIRECT work and discarding the partial tier.
    setenv("GGML_KIMI_CPU_PREFILL_Q2_COLD_ADMIT_FAIL_AT", "1", 1);
    auto failed = make_sink(f, 9, 8);
    configure_cache(failed.get(), 0, {1}, 1);
    std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)> failed_prefill(
        kimi_cpu_prefill_create(failed.get(), error, sizeof(error)), kimi_cpu_prefill_destroy);
    kimi_cpu_prefill_stats failed_stats = {};
    std::fill(actual.begin(), actual.end(), -77.0f);
    const bool ok = kimi_cpu_prefill_run_activation(failed_prefill.get(), meta.role[0], meta.role[1], meta.role[2],
        input.data(), ids.data(), weights.data(), 1, tokens, &activation, actual.data(),
        &failed_stats, error, sizeof(error));
    require(!ok && std::strstr(error, "injected cold-prefill VRAM upload error") &&
        failed_stats.stream.pending_reads == 0 && failed_stats.stream.fallback_count == 0,
        "Q2 cold-prefill H2D failure did not fail closed and drain");
    setenv("GGML_KIMI_CPU_PREFILL_Q2_COLD_ADMIT_FAIL_AT", "0", 1);
    checks += 7;
    std::printf("Q2_COLD_PREFILL_ADMISSION_PASS tokens=%d admitted=1 upload_bytes=%llu decode_reads=%llu rel_l2=%.6g decode_phase_regrow=PASS request_end_quiescence=PASS injected_h2d_failure=PASS\n",
        tokens, (unsigned long long)admitted_stats.gpu_cold_admit_upload_bytes,
        (unsigned long long)decode_stats.read_requests, rel);
}
} // namespace

int main() {
    try {
        ggml_cpu_init();
        setenv("GGML_KIMI_CPU_SINK_READY_FIRST", "1", 1);
        setenv("GGML_KIMI_CPU_SINK_SPLIT_CV", "1", 1);
        setenv("GGML_KIMI_CPU_SINK_CACHE_POLICY", "lru", 1);
        setenv("GGML_KIMI_CPU_SINK_CACHE_ALLOCATION", "uniform", 1);
        setenv("GGML_KIMI_CPU_SINK_CACHE_KEEP_ACROSS_REQUESTS", "1", 1);
        setenv("GGML_KIMI_CPU_SINK_CACHE_RESERVE_BYTES", "536870912", 1);
        setenv("GGML_KIMI_CPU_SINK_VRAM_POLICY", "lru", 1);
        setenv("GGML_KIMI_CPU_SINK_VRAM_MIN_COUNT", "0", 1);
        setenv("GGML_KIMI_CPU_SINK_VRAM_SAFETY_MIB", "32", 1);
        setenv("GGML_KIMI_CPU_SINK_VRAM_ELASTIC", "0", 1);
        setenv("GGML_KIMI_CPU_PREFILL_Q2_SSD_GPU_MIN_ROUTES", "0", 1);
        setenv("GGML_KIMI_CPU_PREFILL_Q2_SSD_GPU_MAX_EXPERTS_PER_LAYER", "0", 1);
        size_t checks = 0;
        double worst_rel = 0.0;
        double worst_abs = 0.0;
        run_shape(512, 256, 8, {false, "basic"}, checks, worst_rel, worst_abs);
        run_shape(512, 256, 8, {true, "production"}, checks, worst_rel, worst_abs);
        run_shape(5120, 2304, 4, {true, "production-real-shape"}, checks, worst_rel, worst_abs);
        run_shape(512, 256, 8, {true, "production-generic"}, checks, worst_rel, worst_abs, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K);
        run_shape(512, 256, 8, {true, "production-generic"}, checks, worst_rel, worst_abs, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
        run_ram_prefill_scheduler(checks, worst_rel);
        run_ssd_prefill_scheduler(checks, worst_rel);
        run_cold_prefill_admission(checks, worst_rel);
        run_host_jobs(GGML_TYPE_Q8_0, checks);
        std::printf("REAL_GPU_Q2_TIER_PASS checks=%zu worst_rel_l2=%.6g worst_abs=%.6g\n",
            checks, worst_rel, worst_abs);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "REAL_GPU_Q2_TIER_FAIL: %s\n", e.what());
        return 1;
    }
}
